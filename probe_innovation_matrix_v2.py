#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P-SSA v2 (Parallel Side-Stream Adaptation with Probabilistic Navigation)
Innovation Matrix Probe v2

Four orthogonal improvement directions tested on the same clean/noisy pair:
- PATH A: Logit-Domain Navigation (LDN) - solve "wrong blueprint"
- PATH B: TF-Dual Track - extract phase gradient from frequency domain
- PATH C: Manifold Gating (P-MG) - adaptive gating based on signal energy
- PATH D: Direct Excitation (DEX) - direct residual loss to wake up side-stream

This probe manually simulates the OPTIMAL state for each path to establish upper bounds.
"""

from __future__ import annotations

import io
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
from hydra.utils import instantiate

from addse.metrics import PESQMetric, SDRMetric, STOIMetric
from addse.utils import load_hydra_config

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)


@dataclass
class ProbeContext:
    lm: nn.Module
    nac: nn.Module
    clean: torch.Tensor
    noisy: torch.Tensor
    sr: int
    device: torch.device
    clean_tokens: torch.Tensor
    clean_latent: torch.Tensor
    clean_token_sum: torch.Tensor
    oracle_residual: torch.Tensor


def _resolve_existing(paths: list[str]) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")


@torch.no_grad()
def _load_lm(config_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    cfg, _ = load_hydra_config(config_path)
    lm = instantiate(cfg.lm).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    lm.load_state_dict(ckpt["state_dict"], strict=False)
    lm.eval()
    return lm


@torch.no_grad()
def _pad_to_codec(nac: nn.Module, wav: torch.Tensor) -> torch.Tensor:
    n_pad = (nac.downsampling_factor - wav.shape[-1] % nac.downsampling_factor) % nac.downsampling_factor
    return F.pad(wav, (0, n_pad))


@torch.no_grad()
def _decode_from_tokens(lm: nn.Module, z_q_sum: torch.Tensor, target_len: int) -> torch.Tensor:
    if hasattr(lm, "_decode_latent_to_wave"):
        return lm._decode_latent_to_wave(z_q_sum, target_length=target_len)
    if hasattr(lm.nac, "decode"):
        return lm.nac.decode(z_q_sum, domain="q")[..., :target_len]
    raise AttributeError("Cannot decode from tokens")


@torch.no_grad()
def _compute_metrics(pred: torch.Tensor, ref: torch.Tensor, sr: int) -> dict[str, float]:
    pesq_metric = PESQMetric(fs=sr)
    estoi_metric = STOIMetric(fs=sr, extended=True)
    si_sdr_metric = SDRMetric(scale_invariant=True, zero_mean=True)
    sdr_metric = SDRMetric(scale_invariant=False, zero_mean=False)
    return {
        "pesq": float(pesq_metric(pred[0], ref[0])),
        "estoi": float(estoi_metric(pred[0], ref[0])),
        "si_sdr": float(si_sdr_metric(pred[0], ref[0])),
        "sdr": float(sdr_metric(pred[0], ref[0])),
    }


@torch.no_grad()
def _guided_token_solve(lm: nn.Module, noisy_wav: torch.Tensor, guide_tokens: torch.Tensor | None, 
                       alpha: float, steps: int):
    """
    Token prediction with optional guidance:
    - guide_tokens: one-hot guidance (LDN)
    - alpha: blend coefficient
    """
    nac = lm.nac
    x_pad = _pad_to_codec(nac, noisy_wav)
    x_lat = nac.encoder(x_pad)
    x_tok, x_q = nac.encode(x_pad, no_sum=True, domain="q")
    B, K, L = x_tok.shape

    y_tok = torch.full_like(x_tok, lm.mask_token)
    final_residual = None

    for step in range(steps):
        mask = y_tok == lm.mask_token
        if not mask.any():
            break

        y_q_step = nac.quantizer.decode(y_tok.masked_fill(mask, 0), output_no_sum=True, domain="code")
        if y_q_step.ndim == 4 and y_q_step.shape[2] == x_q.shape[1]:
            y_q_step = y_q_step.transpose(1, 2)
        y_q_step = y_q_step.masked_fill(mask.unsqueeze(1), 0)

        log_p, residual_pred, _ = lm.log_score(y_q_step, x_q, x_cont=x_lat)
        if residual_pred is not None:
            if final_residual is None:
                final_residual = residual_pred
            else:
                decay = min(max(getattr(lm, "residual_ema_decay", 0.8), 0.0), 0.99)
                final_residual = decay * final_residual + (1.0 - decay) * residual_pred

        logits = log_p
        
        # PATH A: Logit-Domain Navigation
        if guide_tokens is not None:
            clean_onehot = F.one_hot(guide_tokens, num_classes=logits.shape[-1]).to(logits.dtype)
            logits = logits + alpha * (clean_onehot - logits)

        probs = logits.softmax(dim=-1)
        sampled = probs.argmax(dim=-1)[mask]
        y_tok_new = y_tok.clone()
        y_tok_new[mask] = sampled

        if step == steps - 1:
            y_tok = y_tok_new
            break

        confidence = probs[mask].gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        conf_full = torch.full_like(y_tok, 1e9, dtype=torch.float32)
        conf_full[mask] = confidence
        ratio = math.cos(math.pi / 2 * (step + 1) / steps)
        num_mask = int(ratio * K * L)
        if num_mask > 0:
            conf_flat = conf_full.view(B, -1)
            cutoff_vals, _ = torch.topk(conf_flat, num_mask, dim=-1, largest=False)
            cutoff = cutoff_vals[:, -1:]
            new_mask = (conf_flat <= cutoff).view(B, K, L)
            y_tok_new[new_mask] = lm.mask_token
        y_tok = y_tok_new

    y_q_discrete = nac.quantizer.decode(y_tok, output_no_sum=False, domain="code")
    if y_q_discrete.ndim == 4:
        y_q_discrete = y_q_discrete.sum(dim=1) if y_q_discrete.shape[1] == K else y_q_discrete.sum(dim=2)
    return y_q_discrete, final_residual


class ManifoldGatingProbe:
    """
    PATH C: Manifold Gating (P-MG)
    Replace H + ΔH with H_final = H * σ(W_gate * C) + (1-σ) * C
    Simulate optimal state: directly inject voiced/unvoiced discriminator signal
    """
    def __init__(self, decoder: nn.Module, clean_audio: torch.Tensor, sr: int, device: torch.device):
        self.decoder = decoder
        self.clean_audio = clean_audio
        self.sr = sr
        self.device = device
        self.hook = None
        self.energy_gate = None

    def _compute_voicing(self):
        """Compute simple energy-based voicing mask: high energy = voiced (gate=1.0), low=unvoiced (gate=0.0)"""
        frame_len = self.sr // 100  # 10ms frames
        frames = [
            self.clean_audio[:, i:i+frame_len].pow(2).mean()
            for i in range(0, self.clean_audio.shape[-1], frame_len)
        ]
        energy = torch.stack(frames)
        threshold = energy.median()
        voicing = (energy > threshold).float()
        # Upsample to sample rate
        voicing_up = torch.nn.functional.interpolate(
            voicing.unsqueeze(0).unsqueeze(0),
            size=self.clean_audio.shape[-1],
            mode='nearest'
        ).squeeze(0).squeeze(0)
        return voicing_up

    def __enter__(self):
        mod = self.decoder
        for part in ["blocks", "2"]:
            if part.isdigit():
                mod = mod[int(part)]
            else:
                mod = getattr(mod, part)

        self.energy_gate = self._compute_voicing()

        def hook_fn(_module, _inputs, output):
            # output shape: (B, C, T)
            gate_signal = self.energy_gate[:output.shape[-1]].to(output.device).to(output.dtype)
            if gate_signal.shape[0] != output.shape[-1]:
                gate_signal = F.interpolate(gate_signal.unsqueeze(0).unsqueeze(0),
                                           size=output.shape[-1], mode='linear', align_corners=False).squeeze(0).squeeze(0)
            
            # H_final = H * σ(gate) + (1-σ) * dummy_side_stream
            sigma_gate = torch.sigmoid(5.0 * gate_signal).unsqueeze(0).unsqueeze(0)
            # Simulate side-stream contribution with small perturbation
            side_stream_dummy = torch.randn_like(output) * 0.01
            return output * sigma_gate + (1.0 - sigma_gate) * side_stream_dummy

        self.hook = mod.register_forward_hook(hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook is not None:
            self.hook.remove()


class DirectExcitationProbe:
    """
    PATH D: Direct Excitation (DEX)
    Directly supervise side-stream output with Y_clean - Y_base_wav
    We simulate this by blending oracle residual at different strengths to find where loss minimizes
    """
    @staticmethod
    def simulate_direct_residual_loss(base_latent: torch.Tensor, oracle_residual: torch.Tensor,
                                     clean_ref: torch.Tensor, target_len: int, nac: nn.Module, 
                                     sr: int, beta_values: list[float] = None):
        """
        Simulate DEX by injecting oracle_residual at strength beta and measuring reconstruction loss.
        Returns dict of (beta, PESQ, SI-SDR) showing the relationship.
        """
        if beta_values is None:
            beta_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        
        results = []
        for beta in beta_values:
            fused_latent = base_latent + beta * oracle_residual
            reconstructed = nac.decode(fused_latent, domain="q")[..., :target_len]
            metrics = _compute_metrics(reconstructed, clean_ref, sr)
            results.append({
                "beta": beta,
                "pesq": metrics["pesq"],
                "si_sdr": metrics["si_sdr"],
                "estoi": metrics["estoi"],
            })
        return results


@torch.no_grad()
def run_innovation_matrix() -> None:
    """
    Main probe routine: test all 4 paths on the same clean/noisy audio pair.
    """
    root = PROJECT_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load model & audio ===
    print("=" * 120)
    print("P-SSA v2 INNOVATION MATRIX PROBE")
    print("=" * 120)
    
    lm_cfg = _resolve_existing([str(root / "configs" / "addse-s-edbase-parallel60-a008-p02-spec.yaml")])
    lm_ckpt = _resolve_existing([str(root / "logs" / "addse-s-edbase-parallel60-a008-p02-spec" / "checkpoints" / "last.ckpt")])
    clean_path = _resolve_existing([str(root / "saved_audio_v33" / "edbase-local_000000_clean.wav")])

    print(f"[SETUP] Device: {device}")
    print(f"[SETUP] LM Config: {lm_cfg}")
    print(f"[SETUP] LM Checkpoint: {lm_ckpt}")
    print(f"[SETUP] Clean Audio: {clean_path}")

    clean, sr = torchaudio.load(clean_path)
    clean = clean[:1].unsqueeze(0).to(device)

    # Synthetic noisy version: 5dB SNR
    noise = torch.randn_like(clean)
    clean_rms = clean.pow(2).mean().sqrt().clamp_min(1e-8)
    noise_rms = noise.pow(2).mean().sqrt().clamp_min(1e-8)
    noisy = (clean + (clean_rms / (10 ** (5.0 / 20.0) * noise_rms)) * noise).clamp(-1.0, 1.0)

    lm = _load_lm(lm_cfg, lm_ckpt, device)
    nac = lm.nac

    # === Prepare oracle materials ===
    clean_pad = _pad_to_codec(nac, clean)
    noisy_pad = _pad_to_codec(nac, noisy)

    clean_lat = nac.encoder(clean_pad)
    clean_tokens, clean_q = nac.encode(clean_pad, no_sum=True, domain="q")
    clean_q_sum = clean_q.sum(dim=2) if clean_q.ndim == 4 else clean_q
    oracle_residual = clean_lat - clean_q_sum

    print(f"[INFO] Sample rate: {sr} Hz")
    print(f"[INFO] Clean shape: {clean.shape}, Clean tokens shape: {clean_tokens.shape}")
    print(f"[INFO] Oracle residual norm: {oracle_residual.norm().item():.4f}")
    print()

    results_all = {}

    # ========== BASELINE: Pure backbone on noisy ==========
    print("-" * 120)
    print("BASELINE: Backbone Token Prediction (no side-stream)")
    print("-" * 120)
    
    baseline_tokens, baseline_residual = _guided_token_solve(lm, noisy, guide_tokens=None, 
                                                             alpha=0.0, steps=400)
    baseline_wav = _decode_from_tokens(lm, baseline_tokens, clean.shape[-1])
    baseline_metrics = _compute_metrics(baseline_wav, clean, sr)
    results_all["baseline"] = baseline_metrics
    print(f"[BASE] PESQ={baseline_metrics['pesq']:.4f} | SI-SDR={baseline_metrics['si_sdr']:.4f} | "
          f"ESTOI={baseline_metrics['estoi']:.4f} | SDR={baseline_metrics['sdr']:.4f}")
    print()

    # ========== PATH A: LOGIT-DOMAIN NAVIGATION (LDN) ==========
    print("-" * 120)
    print("PATH A: LOGIT-DOMAIN NAVIGATION (LDN)")
    print("Innovation: Side-stream outputs logit offset, not residual")
    print("Simulation: Blend noisy token logits toward clean one-hot")
    print("-" * 120)
    
    ldn_results = {}
    best_ldn_pesq = baseline_metrics['pesq']
    best_ldn_alpha = 0.0
    
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        ldn_tokens, _ = _guided_token_solve(lm, noisy, guide_tokens=clean_tokens, 
                                           alpha=alpha, steps=400)
        ldn_wav = _decode_from_tokens(lm, ldn_tokens, clean.shape[-1])
        ldn_metrics = _compute_metrics(ldn_wav, clean, sr)
        ldn_results[f"alpha_{alpha}"] = ldn_metrics
        results_all[f"ldn_alpha_{alpha}"] = ldn_metrics
        
        print(f"  α={alpha:.2f} | PESQ={ldn_metrics['pesq']:.4f} | SI-SDR={ldn_metrics['si_sdr']:.4f} | "
              f"ESTOI={ldn_metrics['estoi']:.4f}")
        
        if ldn_metrics['pesq'] > best_ldn_pesq:
            best_ldn_pesq = ldn_metrics['pesq']
            best_ldn_alpha = alpha
    
    print(f"\n[LDN BEST] α={best_ldn_alpha:.2f} → PESQ={best_ldn_pesq:.4f} "
          f"(+{best_ldn_pesq - baseline_metrics['pesq']:.4f} vs baseline)")
    print()

    # ========== PATH B: TF-DUAL TRACK ==========
    print("-" * 120)
    print("PATH B: TF-DUAL TRACK - Frequency Domain Feature Extraction")
    print("Innovation: Side-stream sees time-domain wave + frequency-domain phase gradient")
    print("Simulation: Extract STFT phase derivatives, use as guidance for token prediction")
    print("-" * 120)
    
    # Compute STFT phase gradient and collapse to scalar for logit modulation
    # We'll use a simpler approach: compute global phase coherence as a scalar modulation
    stft = torch.stft(
        clean.squeeze(1),
        n_fft=512,
        hop_length=128,
        win_length=512,
        return_complex=True,
        center=True,
        window=torch.hann_window(512, device=device),
    )
    phase = torch.angle(stft)
    phase_dx = torch.diff(phase, dim=-1, prepend=phase[..., :1]).abs()
    phase_coherence = phase_dx.mean()  # Scalar: global phase coherence
    
    tf_results = {}
    best_tf_pesq = baseline_metrics['pesq']
    best_tf_alpha = 0.0
    
    for alpha in [0.1, 0.2, 0.5, 1.0]:
        # For TF-Dual, we simulate by injecting oracle residual scaled by phase coherence
        # Interpretation: where phase is less coherent, trust oracle more
        tf_residual_scaling = (1.0 - torch.tanh(phase_coherence * alpha))
        tf_latent = clean_q_sum + tf_residual_scaling * oracle_residual
        tf_wav = _decode_from_tokens(lm, tf_latent, clean.shape[-1])
        tf_metrics = _compute_metrics(tf_wav, clean, sr)
        tf_results[f"alpha_{alpha}"] = tf_metrics
        results_all[f"tf_alpha_{alpha}"] = tf_metrics
        
        print(f"  α={alpha:.2f} (coherence scaled by {tf_residual_scaling:.4f}) | PESQ={tf_metrics['pesq']:.4f} | "
              f"SI-SDR={tf_metrics['si_sdr']:.4f} | ESTOI={tf_metrics['estoi']:.4f}")
        
        if tf_metrics['pesq'] > best_tf_pesq:
            best_tf_pesq = tf_metrics['pesq']
            best_tf_alpha = alpha
    
    print(f"\n[TF BEST] α={best_tf_alpha:.2f} → PESQ={best_tf_pesq:.4f} "
          f"(+{best_tf_pesq - baseline_metrics['pesq']:.4f} vs baseline)")
    print()

    # ========== PATH C: MANIFOLD GATING ==========
    print("-" * 120)
    print("PATH C: MANIFOLD GATING (P-MG) - Adaptive Feature Gating")
    print("Innovation: Replace H + ΔH with adaptive gating: H * σ(gate) + (1-σ) * C")
    print("Simulation: Inject optimal voicing-based gating signal at decoder block.2")
    print("-" * 120)
    
    decoder = nac.decoder if hasattr(nac, 'decoder') else nac.generator
    
    mg_probe = ManifoldGatingProbe(decoder, clean, sr, device)
    with mg_probe:
        mg_wav = nac.decode(clean_q_sum, domain="q")[..., :clean.shape[-1]]
    mg_metrics = _compute_metrics(mg_wav, clean, sr)
    results_all["manifold_gating"] = mg_metrics
    
    print(f"[MG] PESQ={mg_metrics['pesq']:.4f} | SI-SDR={mg_metrics['si_sdr']:.4f} | "
          f"ESTOI={mg_metrics['estoi']:.4f}")
    print(f"[MG] vs baseline: {mg_metrics['pesq'] - baseline_metrics['pesq']:+.4f} PESQ")
    print()

    # ========== PATH D: DIRECT EXCITATION ==========
    print("-" * 120)
    print("PATH D: DIRECT EXCITATION (DEX) - Raw Residual Supervision")
    print("Innovation: Bypass alpha scaling, directly supervise residual with Y_clean - Y_base")
    print("Simulation: Sweep oracle residual contribution to find optimal blend strength")
    print("-" * 120)
    
    dex_probe = DirectExcitationProbe()
    dex_results = dex_probe.simulate_direct_residual_loss(
        base_latent=clean_q_sum,
        oracle_residual=oracle_residual,
        clean_ref=clean,
        target_len=clean.shape[-1],
        nac=nac,
        sr=sr,
        beta_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    )
    
    best_dex_pesq = 0.0
    best_dex_beta = 0.0
    
    for res in dex_results:
        beta, pesq, si_sdr = res['beta'], res['pesq'], res['si_sdr']
        results_all[f"dex_beta_{beta}"] = {"pesq": pesq, "si_sdr": si_sdr, "estoi": res['estoi']}
        
        print(f"  β={beta:.2f} | PESQ={pesq:.4f} | SI-SDR={si_sdr:.4f} | "
              f"ESTOI={res['estoi']:.4f}")
        
        if pesq > best_dex_pesq:
            best_dex_pesq = pesq
            best_dex_beta = beta
    
    print(f"\n[DEX BEST] β={best_dex_beta:.2f} → PESQ={best_dex_pesq:.4f} "
          f"(+{best_dex_pesq - baseline_metrics['pesq']:.4f} vs baseline)")
    print()

    # ========== COMPARATIVE SUMMARY ==========
    print("=" * 120)
    print("FINAL COMPARISON & RECOMMENDATIONS")
    print("=" * 120)
    
    improvements = {
        "PATH A (LDN)": {
            "pesq": best_ldn_pesq,
            "delta": best_ldn_pesq - baseline_metrics['pesq'],
            "verdict": "✓ BREAKTHROUGH" if best_ldn_pesq > 2.5 else "⚠ MODEST GAIN" if best_ldn_pesq > baseline_metrics['pesq'] else "✗ NO GAIN",
            "param": f"α={best_ldn_alpha:.2f}",
        },
        "PATH B (TF-Dual)": {
            "pesq": best_tf_pesq,
            "delta": best_tf_pesq - baseline_metrics['pesq'],
            "verdict": "✓ BREAKTHROUGH" if best_tf_pesq > 2.5 else "⚠ MODEST GAIN" if best_tf_pesq > baseline_metrics['pesq'] else "✗ NO GAIN",
            "param": f"α={best_tf_alpha:.2f}",
        },
        "PATH C (Manifold Gating)": {
            "pesq": mg_metrics['pesq'],
            "delta": mg_metrics['pesq'] - baseline_metrics['pesq'],
            "verdict": "✓ BREAKTHROUGH" if mg_metrics['pesq'] > 2.5 else "⚠ MODEST GAIN" if mg_metrics['pesq'] > baseline_metrics['pesq'] else "✗ NO GAIN",
            "param": "voicing-gate",
        },
        "PATH D (DEX)": {
            "pesq": best_dex_pesq,
            "delta": best_dex_pesq - baseline_metrics['pesq'],
            "verdict": "✓ BREAKTHROUGH" if best_dex_pesq > 2.5 else "⚠ MODEST GAIN" if best_dex_pesq > baseline_metrics['pesq'] else "✗ NO GAIN",
            "param": f"β={best_dex_beta:.2f}",
        },
    }
    
    print(f"{'Path':<25} {'PESQ':>10} {'Δ PESQ':>10} {'Param':>15} {'Verdict':<25}")
    print("-" * 85)
    print(f"{'BASELINE':25} {baseline_metrics['pesq']:>10.4f} {0.0:>10.4f} {'-':>15} {'Reference':25}")
    
    for path_name, info in improvements.items():
        print(f"{path_name:<25} {info['pesq']:>10.4f} {info['delta']:>+10.4f} {info['param']:>15} "
              f"{info['verdict']:<25}")
    
    print()
    print("DECISION LOGIC:")
    print("-" * 85)
    
    # Rank by PESQ improvement
    ranked = sorted(improvements.items(), key=lambda x: x[1]['pesq'], reverse=True)
    print(f"1st Choice: {ranked[0][0]} (PESQ={ranked[0][1]['pesq']:.4f})")
    print(f"2nd Choice: {ranked[1][0]} (PESQ={ranked[1][1]['pesq']:.4f})")
    
    best_path = ranked[0][0]
    best_pesq = ranked[0][1]['pesq']
    
    print()
    if "LDN" in best_path:
        print(f"🎯 RECOMMENDATION: Implement LDN + DEX combination")
        print(f"   - LDN ensures outputs stay on discrete manifold (prevents DisCo-TSE electronic artifacts)")
        print(f"   - DEX directly supervises residual learning (bypasses alpha gradient starvation)")
        print(f"   - Expected paper story: 'Discrete-System Robust Probabilistic Navigation'")
        print(f"   - Estimated training time: 500+ batches with validation every 50 batches")
    elif "Gating" in best_path:
        print(f"🎯 RECOMMENDATION: Implement Manifold Gating as primary mixer")
        print(f"   - Replace hard addition with learned gate σ(W*C)")
        print(f"   - Supports per-frame adaptive mix ratio")
        print(f"   - Expected paper story: 'Semantic-Credibility-Aware Adaptive Mixing'")
    else:
        print(f"🎯 RECOMMENDATION: Continue analysis with {best_path}")
    
    print()
    print("=" * 120)
    
    # Save results to JSON
    import json
    out_dir = root / "probe_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "baseline_pesq": float(baseline_metrics['pesq']),
        "ldn_best": float(best_ldn_pesq),
        "ldn_alpha": float(best_ldn_alpha),
        "tf_best": float(best_tf_pesq),
        "tf_alpha": float(best_tf_alpha),
        "manifold_gating_pesq": float(mg_metrics['pesq']),
        "dex_best": float(best_dex_pesq),
        "dex_beta": float(best_dex_beta),
        "recommendation": "LDN + DEX" if best_path == "PATH A (LDN)" else best_path,
    }
    
    with open(out_dir / "innovation_matrix_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"[SAVE] Results saved to {out_dir / 'innovation_matrix_v2_summary.json'}")


if __name__ == "__main__":
    run_innovation_matrix()
