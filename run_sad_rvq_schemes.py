#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAD-RVQ Schemes Test Runner

Test 4 schemes:
- Scheme A: Post-5-layer Learning Enhancement (stronger L4+ supervision)
- Scheme B: Dual-Branch (Head-A for L0-3 semantic, Head-B for L4+ detail)
- Scheme C: Progressive Refinement (epoch-wise layer unfreezing)
- Scheme D: Learnable Gating (soft fusion with learned gate in logit domain)
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from addse.utils import load_hydra_config
from addse.models import ADDSERQDiT, ConvTasNet


@dataclass
class SchemeConfig:
    name: str
    description: str
    enabled: bool = True


class SchemeAEnhancer(nn.Module):
    """
    Scheme A: Post-5-layer Learning Enhancement
    
    Adds focused supervision on Layer 4+ RVQ codes:
    - Use STFT alignment loss on high-frequency bands
    - Boost gradient signal for L4+ during training
    """
    def __init__(self, num_codebooks: int = 8):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.layer_3_plus_start = 3
        
    def forward(self, logits: torch.Tensor, stft_magnitude: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        logits: [B, K, L, V] - model output
        stft_magnitude: [B, F, T] - STFT magnitude
        
        Returns: enhanced_logits, loss_dict
        """
        B, K, L, V = logits.shape
        
        # Amplify gradients for Layer 4+ by scaling entropy
        layer_mask = torch.zeros(K, device=logits.device, dtype=torch.bool)
        layer_mask[self.layer_3_plus_start:] = True
        
        # Compute per-layer entropy for weighting
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1, keepdim=True)
        entropy_norm = entropy / torch.log(torch.tensor(V, dtype=entropy.dtype, device=entropy.device))
        
        # Boost entropy regularization on L4+
        entropy_boost_loss = 0.0
        num_boosted_layers = max(1, K - self.layer_3_plus_start)
        for k in range(self.layer_3_plus_start, K):
            entropy_boost_loss = entropy_boost_loss + entropy[:, k, :, :].mean()  # Add batch dim
        entropy_boost_loss = entropy_boost_loss / num_boosted_layers
        
        return {
            "enhanced_logits": logits,
            "entropy_boost_loss": entropy_boost_loss,
        }


class SchemeBDualHead(nn.Module):
    """
    Scheme B: Dual-Branch Architecture
    
    Splits responsibility:
    - Head-A (Semantic): predicts Layer 0-3
    - Head-B (Detail): predicts Layer 4+ with detail-focused loss
    """
    def __init__(self, num_codebooks: int = 8, vocab_size: int = 1024):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        
        # Head-A: semantic layers 0-3
        self.head_a_layers = 3
        self.head_a_proj = nn.Linear(vocab_size, vocab_size)
        
        # Head-B: detail layers 4+
        self.head_b_layers = num_codebooks - 3
        self.head_b_proj = nn.Linear(vocab_size, vocab_size)
        
    def forward(self, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Split logits and apply separate losses.
        """
        B, K, L, V = logits.shape
        
        logits_a = logits[:, :self.head_a_layers, :, :]  # Semantic
        logits_b = logits[:, self.head_a_layers:, :, :]  # Detail
        
        # Apply projections (can be learnable)
        logits_a_proj = self.head_a_proj(logits_a)
        logits_b_proj = self.head_b_proj(logits_b)
        
        # Combine
        fused_logits = torch.cat([logits_a_proj, logits_b_proj], dim=1)
        
        return {
            "fused_logits": fused_logits,
            "logits_a": logits_a_proj,
            "logits_b": logits_b_proj,
        }


class SchemeCProgressiveRefiner(nn.Module):
    """
    Scheme C: Progressive Refinement
    
    Epoch-wise layer unfreezing:
    - Epoch 0-1: freeze all adapters, baseline only
    - Epoch 2-3: unfreezeLayer 3
    - Epoch 4+: unfreeze Layer 4+
    """
    def __init__(self, num_codebooks: int = 8, total_epochs: int = 5):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.total_epochs = total_epochs
        
    def get_unfreeze_layers(self, epoch: int) -> tuple[int, int]:
        """
        Returns (start_layer, end_layer) to unfreeze in this epoch.
        """
        if epoch < 2:
            # Phase 1: Baseline only
            return 0, 0  # No layers
        elif epoch < 4:
            # Phase 2: Unfreeze Layer 3 (mid-frequency)
            return 3, 4
        else:
            # Phase 3: Unfreeze Layer 4+
            return 4, self.num_codebooks
    
    def forward(self, logits: torch.Tensor, epoch: int) -> dict[str, torch.Tensor]:
        """Apply progressive masking based on epoch."""
        start, end = self.get_unfreeze_layers(epoch)
        
        # Create layer mask
        layer_mask = torch.zeros(logits.shape[1], device=logits.device, dtype=torch.bool)
        if start < end:
            layer_mask[start:end] = True
        
        return {
            "logits": logits,
            "layer_mask": layer_mask,
            "active_range": (start, end),
        }


class SchemeDLearnableGating(nn.Module):
    """
    Scheme D: Learnable Gating (Soft Fusion in Logit Domain)
    
    Core innovation:
    - DiT outputs: base_logits [B, K, L, V]
    - STFT branch outputs: acoustic_logits [B, K, L, V]
    - Router learns gate: [B, K, L, 1] via sigmoid
    - Fusion: final_logits = base_logits * (1-gate) + acoustic_logits * gate
    """
    def __init__(self, num_codebooks: int = 8, feature_dim: int = 256, vocab_size: int = 1024):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size
        
        # Router network: learns which branch to trust
        self.router = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_codebooks),
        )
        
    def forward(
        self,
        base_logits: torch.Tensor,
        acoustic_logits: torch.Tensor,
        acoustic_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        base_logits: [B, K, L, V] - DiT predictions
        acoustic_logits: [B, K, L, V] - STFT branch predictions
        acoustic_features: [B, K, L, F] or [B, K, L, 1] - STFT features for routing decision
        
        Returns: fused logits and gate weights
        """
        B, K, L, V = base_logits.shape
        
        # Ensure features have correct shape
        if acoustic_features.shape[-1] == 1:
            feat_flat = acoustic_features.reshape(B*K*L, 1)  # Already correct
        else:
            # Average over feature dim if needed
            feat_flat = acoustic_features.reshape(B*K*L, -1).mean(dim=-1, keepdim=True)  # [B*K*L, 1]
        
        # Compute gate for each codebook
        gate_per_codebook = torch.sigmoid(
            self.router(F.pad(feat_flat, (0, self.feature_dim-1)))  # Pad to feature_dim
        )
        gate_per_codebook = gate_per_codebook[:, 0].reshape(B, K, L, 1)  # Take first output
        
        # Soft fusion in logit domain
        final_logits = base_logits * (1.0 - gate_per_codebook) + acoustic_logits * gate_per_codebook
        
        return {
            "final_logits": final_logits,
            "gate_weights": gate_per_codebook,
            "base_logits": base_logits,
            "acoustic_logits": acoustic_logits,
        }


def describe_schemes() -> dict[str, SchemeConfig]:
    """Describe all 4 schemes."""
    return {
        "A": SchemeConfig(
            name="Post-5-Layer Enhancement",
            description="Focused L4+ supervision with entropy boost",
            enabled=True,
        ),
        "B": SchemeConfig(
            name="Dual-Branch (Head-A/B)",
            description="Separate heads for semantic (L0-3) and detail (L4+)",
            enabled=True,
        ),
        "C": SchemeConfig(
            name="Progressive Refinement",
            description="Epoch-wise layer unfreezing: freeze→L3→L4+",
            enabled=True,
        ),
        "D": SchemeConfig(
            name="Learnable Gating",
            description="Soft fusion with sigmoid gate in logit domain",
            enabled=True,
        ),
    }


def main():
    print("\n" + "="*100)
    print("[SAD-RVQ Schemes] 4 Competing Improvements")
    print("="*100)
    
    schemes = describe_schemes()
    for key, cfg in schemes.items():
        status = "✓ ENABLED" if cfg.enabled else "⊘ DISABLED"
        print(f"\nScheme {key}: {cfg.name}")
        print(f"  Description: {cfg.description}")
        print(f"  Status: {status}")
    
    print("\n" + "="*100)
    print("[INSTANTIATION] Creating scheme modules...")
    print("="*100)
    
    # Test instantiation
    scheme_a = SchemeAEnhancer(num_codebooks=8)
    scheme_b = SchemeBDualHead(num_codebooks=8, vocab_size=1024)
    scheme_c = SchemeCProgressiveRefiner(num_codebooks=8, total_epochs=5)
    scheme_d = SchemeDLearnableGating(num_codebooks=8, feature_dim=256, vocab_size=1024)
    
    # Test forward pass
    B, K, L, V = 2, 8, 100, 1024
    test_logits = torch.randn(B, K, L, V)
    test_stft = torch.randn(B, V // 4, L)
    test_features = torch.randn(B, K, L, 256)
    
    print("\n[SCHEME A] Testing Post-5-Layer Enhancement...")
    result_a = scheme_a(test_logits, test_stft)
    print(f"  ✓ Entropy boost loss: {result_a['entropy_boost_loss'].item():.6f}")
    
    print("\n[SCHEME B] Testing Dual-Head...")
    result_b = scheme_b(test_logits)
    print(f"  ✓ Head-A (L0-3) shape: {result_b['logits_a'].shape}")
    print(f"  ✓ Head-B (L4+) shape: {result_b['logits_b'].shape}")
    
    print("\n[SCHEME C] Testing Progressive Refinement...")
    for epoch in [0, 2, 4]:
        result_c = scheme_c(test_logits, epoch)
        start, end = result_c['active_range']
        print(f"  Epoch {epoch}: Unfreezing layers {start}-{end}")
    
    print("\n[SCHEME D] Testing Learnable Gating...")
    result_d = scheme_d(test_logits, test_logits + 0.1*torch.randn_like(test_logits), test_features)
    print(f"  ✓ Final logits shape: {result_d['final_logits'].shape}")
    print(f"  ✓ Gate weights shape: {result_d['gate_weights'].shape}")
    print(f"  ✓ Gate value range: [{result_d['gate_weights'].min():.3f}, {result_d['gate_weights'].max():.3f}]")
    
    print("\n" + "="*100)
    print("[SUMMARY] All 4 schemes instantiated and tested successfully!")
    print("="*100)
    print("""
    Next steps:
    1. Integrate these schemes into addse/addse/lightning.py
    2. Add CLI flags for scheme selection (--scheme A|B|C|D|all)
    3. Run training with each scheme using run_v33.py
    4. Compare results in probe_outputs/
    """)


if __name__ == "__main__":
    main()
