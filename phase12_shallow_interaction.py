#!/usr/bin/env python3
"""
Phase12: Baseline Parallel ADDSE Test (Neural Front-end, No STFT Backbone)
===========================================================================

This file implements a controlled baseline for Experiment-2:
1. Neural complex front-end encoder (TCNAC-style ComplexConv1d)
2. Parallel dual-path processing:
   - Magnitude path: discrete via Vector Quantization (VQ)
   - Phase path: continuous embedding (sin/cos circular representation)
3. Shallow interaction at bottleneck:
   H_phase' = H_phase * (1 + 0.1 * sigmoid(W_mag(H_mag_discrete)))
4. Complex decoder reconstructs waveform directly.

Design objective:
- Keep architecture minimal and stable for first validation run.
- Avoid overloading novelty in the first experiment.
- Stay compatible with existing ADDSE code style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase9_nhfae_e1 import mrstft_loss
from phase9_tcnac_codec import ComplexConv1d


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap angle difference to [-pi, pi)."""
    return torch.atan2(torch.sin(x), torch.cos(x))


class SimpleVectorQuantizer(nn.Module):
    """
    Lightweight VQ for magnitude branch.

    Input shape: [B, D, T]
    Output:
      - quantized: [B, D, T]
      - indices: [B, T]
      - loss_vq: scalar
      - loss_commit: scalar
    """

    def __init__(self, codebook_size: int = 256, dim: int = 96, beta: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta = beta

        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, dim, tlen = x.shape
        assert dim == self.dim, f"VQ dim mismatch: got {dim}, expected {self.dim}"

        # [B, D, T] -> [B*T, D]
        x_flat = x.transpose(1, 2).contiguous().view(-1, dim)

        # Squared L2 distance to codebook
        code = self.codebook.weight  # [K, D]
        dist = (
            torch.sum(x_flat ** 2, dim=1, keepdim=True)
            - 2.0 * torch.matmul(x_flat, code.t())
            + torch.sum(code ** 2, dim=1).unsqueeze(0)
        )

        indices = torch.argmin(dist, dim=1)  # [B*T]
        quant_flat = self.codebook(indices)  # [B*T, D]

        # VQ losses
        loss_vq = F.mse_loss(quant_flat, x_flat.detach())
        loss_commit = self.beta * F.mse_loss(x_flat, quant_flat.detach())

        # Straight-through estimator
        quant_st = x_flat + (quant_flat - x_flat).detach()

        # [B*T, D] -> [B, D, T]
        quant = quant_st.view(bsz, tlen, dim).transpose(1, 2).contiguous()
        idx = indices.view(bsz, tlen)

        return quant, idx, loss_vq, loss_commit


class NeuralComplexFrontEnd(nn.Module):
    """
    Complex neural encoder from waveform to latent complex representation.

    Input:  y [B, 1, T] (real waveform)
    Output: z [B, C, T] (complex latent)
    """

    def __init__(self, latent_channels: int = 64):
        super().__init__()
        c = latent_channels
        self.enc1 = ComplexConv1d(1, c // 2, kernel_size=7, padding=3, euler_factor=0.5)
        self.enc2 = ComplexConv1d(c // 2, c, kernel_size=5, padding=2, euler_factor=0.5)
        self.enc3 = ComplexConv1d(c, c, kernel_size=3, padding=1, euler_factor=0.3)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() != 3 or wav.size(1) != 1:
            raise ValueError(f"Expected wav [B,1,T], got {tuple(wav.shape)}")
        z = torch.complex(wav.squeeze(1), torch.zeros_like(wav.squeeze(1))).unsqueeze(1)
        z = self.enc1(z)
        z = self.enc2(z)
        z = self.enc3(z)
        return z


class NeuralComplexBackEnd(nn.Module):
    """Complex decoder from latent complex back to waveform."""

    def __init__(self, latent_channels: int = 64):
        super().__init__()
        c = latent_channels
        self.dec1 = ComplexConv1d(c, c, kernel_size=3, padding=1, euler_factor=0.3)
        self.dec2 = ComplexConv1d(c, c // 2, kernel_size=5, padding=2, euler_factor=0.5)
        self.dec3 = ComplexConv1d(c // 2, 1, kernel_size=7, padding=3, euler_factor=0.5)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec1(z)
        x = self.dec2(x)
        x = self.dec3(x)
        return x.real.unsqueeze(1)


class MagnitudeDiscretePath(nn.Module):
    """Magnitude path: continuous magnitude features -> VQ discrete tokens."""

    def __init__(self, in_channels: int = 64, d_model: int = 96, codebook_size: int = 256):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.vq = SimpleVectorQuantizer(codebook_size=codebook_size, dim=d_model)

    def forward(self, mag_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.pre(mag_latent)
        quant, idx, loss_vq, loss_commit = self.vq(feat)
        return {
            "mag_feat": feat,
            "mag_discrete": quant,
            "mag_indices": idx,
            "loss_vq": loss_vq,
            "loss_commit": loss_commit,
        }


class PhaseContinuousPath(nn.Module):
    """
    Phase path: circular continuous embedding.

    Input: sin/cos from latent phase, shape [B, 2C, T]
    Output: phase hidden [B, D, T]
    """

    def __init__(self, in_channels: int = 128, d_model: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, phase_trig: torch.Tensor) -> torch.Tensor:
        return self.net(phase_trig)


class ShallowGatedInteraction(nn.Module):
    """
    Multiplicative gating from discrete magnitude to continuous phase.

    H_phase' = H_phase * (1 + alpha * sigmoid(W(H_mag_discrete)))
    """

    def __init__(self, d_model: int = 96, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.mag_to_phase = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

    def forward(self, mag_discrete: torch.Tensor, phase_feat: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.mag_to_phase(mag_discrete))
        return phase_feat * (1.0 + self.alpha * gate)


class ParallelLatentFusionDecoder(nn.Module):
    """
    Fuse discrete magnitude path and continuous phase path back to latent complex Z_hat.

    Output:
      - z_hat [B, C, T] complex
      - phase_angle_hat [B, C, T]
      - mag_hat [B, C, T]
    """

    def __init__(self, latent_channels: int = 64, d_model: int = 96):
        super().__init__()
        self.mag_out = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(d_model, latent_channels, kernel_size=1),
            nn.Softplus(),
        )
        self.phase_out = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(d_model, 2 * latent_channels, kernel_size=1),
        )

    def forward(self, mag_discrete: torch.Tensor, phase_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        mag_hat = self.mag_out(mag_discrete) + 1e-6

        phase_2c = self.phase_out(phase_feat)
        c2 = phase_2c.shape[1] // 2
        cos_hat = phase_2c[:, :c2]
        sin_hat = phase_2c[:, c2:]

        norm = torch.sqrt(cos_hat ** 2 + sin_hat ** 2 + 1e-8)
        cos_hat = cos_hat / norm
        sin_hat = sin_hat / norm

        phase_angle_hat = torch.atan2(sin_hat, cos_hat)
        z_hat = mag_hat * torch.exp(1j * phase_angle_hat)

        return {
            "z_hat": z_hat,
            "mag_hat": mag_hat,
            "phase_hat": phase_angle_hat,
        }


class Phase12ShallowInteractionModel(nn.Module):
    """
    Complete baseline model for Experiment-2.

    Pipeline:
    wav -> complex encoder -> split mag/phase -> parallel paths
        -> shallow interaction -> latent complex reconstruction -> complex decoder -> wav
    """

    def __init__(
        self,
        latent_channels: int = 64,
        d_model: int = 96,
        codebook_size: int = 256,
        interaction_alpha: float = 0.1,
    ):
        super().__init__()
        self.front = NeuralComplexFrontEnd(latent_channels=latent_channels)
        self.mag_path = MagnitudeDiscretePath(
            in_channels=latent_channels,
            d_model=d_model,
            codebook_size=codebook_size,
        )
        self.phase_path = PhaseContinuousPath(
            in_channels=2 * latent_channels,
            d_model=d_model,
        )
        self.interact = ShallowGatedInteraction(d_model=d_model, alpha=interaction_alpha)
        self.fuse = ParallelLatentFusionDecoder(latent_channels=latent_channels, d_model=d_model)
        self.back = NeuralComplexBackEnd(latent_channels=latent_channels)

    def encode_latent(self, wav: torch.Tensor) -> torch.Tensor:
        return self.front(wav)

    def forward(self, noisy_wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_noisy = self.front(noisy_wav)  # [B, C, T] complex

        mag_latent = torch.abs(z_noisy)
        phase_latent = torch.angle(z_noisy)

        phase_trig = torch.cat([
            torch.cos(phase_latent),
            torch.sin(phase_latent),
        ], dim=1)  # [B, 2C, T]

        mag_out = self.mag_path(mag_latent)
        phase_feat = self.phase_path(phase_trig)

        phase_interacted = self.interact(mag_out["mag_discrete"], phase_feat)

        fused = self.fuse(mag_out["mag_discrete"], phase_interacted)
        enhanced_wav = self.back(fused["z_hat"])  # [B,1,T]

        outputs = {
            "enhanced_wav": enhanced_wav,
            "z_noisy": z_noisy,
            "z_hat": fused["z_hat"],
            "mag_hat": fused["mag_hat"],
            "phase_hat": fused["phase_hat"],
            "mag_indices": mag_out["mag_indices"],
            "loss_vq": mag_out["loss_vq"],
            "loss_commit": mag_out["loss_commit"],
        }
        return outputs


@dataclass
class LossConfig:
    lambda_rec: float = 1.0
    lambda_mrstft: float = 0.2
    lambda_phase: float = 0.1
    lambda_vq: float = 1.0
    lambda_commit: float = 1.0
    lambda_gomp: float = 0.0  # keep 0.0 in first baseline run


def phase_distance(pred_phase: torch.Tensor, tgt_phase: torch.Tensor) -> torch.Tensor:
    """Circular phase distance in latent domain."""
    return torch.mean(torch.abs(wrap_to_pi(pred_phase - tgt_phase)))


def gomp_like_score(
    mag_pred: torch.Tensor,
    mag_tgt: torch.Tensor,
    phase_pred: torch.Tensor,
    phase_tgt: torch.Tensor,
    gamma: float = 0.2,
) -> torch.Tensor:
    """
    Lightweight GOMP-like objective proxy.
    Higher is better, loss uses negative score.
    """
    num = torch.mean(mag_tgt ** 2)
    den = torch.mean((mag_pred - mag_tgt) ** 2) + 1e-8
    snr_part = 10.0 * torch.log10(num / den)
    pd = phase_distance(phase_pred, phase_tgt)
    return snr_part - gamma * pd


def compute_losses(
    model: Phase12ShallowInteractionModel,
    noisy_wav: torch.Tensor,
    clean_wav: torch.Tensor,
    cfg: LossConfig,
) -> Dict[str, torch.Tensor]:
    """
    Compute mixed-domain losses for the baseline parallel architecture.
    """
    out = model(noisy_wav)
    enhanced_wav = out["enhanced_wav"]

    # Waveform reconstruction
    loss_rec = F.l1_loss(enhanced_wav, clean_wav)

    # Multi-resolution STFT consistency (as loss, not as model backbone)
    loss_mrstft = mrstft_loss(enhanced_wav.squeeze(1), clean_wav.squeeze(1))

    # Latent phase supervision: compare with clean latent phase
    with torch.no_grad():
        z_clean = model.encode_latent(clean_wav)
    phase_tgt = torch.angle(z_clean)
    phase_pred = out["phase_hat"]
    loss_phase = phase_distance(phase_pred, phase_tgt)

    # Optional GOMP-like objective
    mag_pred = out["mag_hat"]
    mag_tgt = torch.abs(z_clean)
    gomp_score = gomp_like_score(mag_pred, mag_tgt, phase_pred, phase_tgt)
    loss_gomp = -gomp_score

    total = (
        cfg.lambda_rec * loss_rec
        + cfg.lambda_mrstft * loss_mrstft
        + cfg.lambda_phase * loss_phase
        + cfg.lambda_vq * out["loss_vq"]
        + cfg.lambda_commit * out["loss_commit"]
        + cfg.lambda_gomp * loss_gomp
    )

    return {
        "loss_total": total,
        "loss_rec": loss_rec,
        "loss_mrstft": loss_mrstft,
        "loss_phase": loss_phase,
        "loss_vq": out["loss_vq"],
        "loss_commit": out["loss_commit"],
        "loss_gomp": loss_gomp,
        "gomp_score": gomp_score,
    }


def demo_forward_pass(device: str = "cuda") -> None:
    """Quick shape check for sanity before full training."""
    use_cuda = torch.cuda.is_available() and device.startswith("cuda")
    dev = torch.device("cuda" if use_cuda else "cpu")

    model = Phase12ShallowInteractionModel().to(dev)
    x = torch.randn(2, 1, 16000, device=dev)
    y = torch.randn(2, 1, 16000, device=dev)

    out = model(x)
    losses = compute_losses(model, x, y, LossConfig())

    print("[Phase12 Demo]")
    print("enhanced_wav:", tuple(out["enhanced_wav"].shape))
    print("mag_indices:", tuple(out["mag_indices"].shape))
    print("loss_total:", float(losses["loss_total"].detach().cpu()))


if __name__ == "__main__":
    demo_forward_pass()
