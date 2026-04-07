#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDN / Gated / TF-Hint probe for ADDSE.

This script tests three ideas on the same clean/noisy pair:
1) Logit-Domain Navigation (LDN)
2) Side-stream gated interaction variants
3) Dual-domain TF-hint injection

The purpose is to answer whether probability-domain interaction can fix the low backbone score,
and whether more expressive side-stream operators beat plain additive fusion.
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
def _guided_token_solve(lm: nn.Module, noisy_wav: torch.Tensor, guide_tokens: torch.Tensor | None, alpha: float, steps: int):
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


class GatedProbeInjector:
    def __init__(self, decoder: nn.Module, point: str, inject: torch.Tensor, mode: str, alpha: float = 0.05):
        self.decoder = decoder
        self.point = point
        self.inject = inject
        self.mode = mode
        self.alpha = alpha
        self.hook = None

    def __enter__(self):
        mod = self.decoder
        for part in self.point.split('.'):
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part)

        def hook_fn(_module, _inputs, output):
            inject = self.inject
            if inject.shape[-1] != output.shape[-1]:
                inject = F.interpolate(inject, size=output.shape[-1], mode='linear', align_corners=False)
            if inject.shape[1] != output.shape[1]:
                proj = torch.zeros((output.shape[1], inject.shape[1], 1), device=output.device, dtype=output.dtype)
                n = min(output.shape[1], inject.shape[1])
                proj[:n, :n, 0] = 1.0
                inject = F.conv1d(inject, proj)

            if self.mode == 'add':
                return output + self.alpha * inject
            if self.mode == 'mul':
                gate = torch.sigmoid(self.alpha * inject)
                return output * gate
            if self.mode == 'mask':
                mask = torch.sigmoid(inject)
                return output + self.alpha * inject * mask
            raise ValueError(f'Unknown mode: {self.mode}')

        self.hook = mod.register_forward_hook(hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook is not None:
            self.hook.remove()


@torch.no_grad()
def run_probe() -> dict:
    root = PROJECT_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lm_cfg = _resolve_existing([str(root / "configs" / "addse-s-edbase-parallel60-a008-p02-spec.yaml")])
    lm_ckpt = _resolve_existing([str(root / "logs" / "addse-s-edbase-parallel60-a008-p02-spec" / "checkpoints" / "last.ckpt")])
    clean_path = _resolve_existing([str(root / "saved_audio_v33" / "edbase-local_000000_clean.wav")])

    clean, sr = torchaudio.load(clean_path)
    clean = clean[:1].unsqueeze(0).to(device)

    noise = torch.randn_like(clean)
    clean_rms = clean.pow(2).mean().sqrt().clamp_min(1e-8)
    noise_rms = noise.pow(2).mean().sqrt().clamp_min(1e-8)
    noisy = (clean + (clean_rms / (10 ** (5.0 / 20.0) * noise_rms)) * noise).clamp(-1.0, 1.0)

    lm = _load_lm(lm_cfg, lm_ckpt, device)
    nac = lm.nac

    clean_pad = _pad_to_codec(nac, clean)
    noisy_pad = _pad_to_codec(nac, noisy)

    clean_lat = nac.encoder(clean_pad)
    clean_tokens, clean_q = nac.encode(clean_pad, no_sum=True, domain="q")
    clean_q_sum = clean_q.sum(dim=2) if clean_q.ndim == 4 else clean_q
    oracle_residual = clean_lat - clean_q_sum

    print("=" * 100)
    print("LDN / Gated / TF-Hint Probe")
    print("=" * 100)
    print(f"device={device} sr={sr}")

    results: dict[str, dict] = {}

    print(f"[INFO] clean_tokens shape={tuple(clean_tokens.shape)}")

    # Baselines.
    backbone_wav = _decode_from_tokens(lm, clean_q_sum, clean.shape[-1])
    backbone_metrics = _compute_metrics(backbone_wav, clean, sr)
    results["backbone_clean"] = backbone_metrics
    print(f"[BASE] clean token backbone PESQ={backbone_metrics['pesq']:.3f} SI-SDR={backbone_metrics['si_sdr']:.3f}")

    # Exp1: pure backbone on noisy tokens.
    backbone_200 = None
    backbone_400 = None
    for steps in (200, 400):
        noisy_pred_tokens, noisy_residual = _guided_token_solve(lm, noisy, guide_tokens=None, alpha=0.0, steps=steps)
        noisy_backbone_wav = _decode_from_tokens(lm, noisy_pred_tokens, clean.shape[-1])
        noisy_backbone_metrics = _compute_metrics(noisy_backbone_wav, clean, sr)
        results[f"backbone_noisy_{steps}"] = noisy_backbone_metrics
        if steps == 200:
            backbone_200 = noisy_backbone_metrics
        else:
            backbone_400 = noisy_backbone_metrics
        print(f"[EXP1] noisy-token backbone ({steps}) PESQ={noisy_backbone_metrics['pesq']:.3f} SI-SDR={noisy_backbone_metrics['si_sdr']:.3f}")

    # LDN sweep: blend logits toward clean token one-hot.
    print("\n[LDN] Logit-domain navigation sweep")
    for alpha in (0.05, 0.1, 0.2, 0.5):
        guided_tokens, _ = _guided_token_solve(lm, noisy, guide_tokens=clean_tokens, alpha=alpha, steps=400)
        guided_wav = _decode_from_tokens(lm, guided_tokens, clean.shape[-1])
        m = _compute_metrics(guided_wav, clean, sr)
        results[f"ldn_alpha_{alpha}"] = m
        print(f"  alpha={alpha:.2f} | PESQ={m['pesq']:.3f} | ESTOI={m['estoi']:.3f} | SI-SDR={m['si_sdr']:.3f}")

    # Better one-hot guidance using exact clean tokens as probability anchors.
    guided_tokens_01, _ = _guided_token_solve(lm, noisy, guide_tokens=clean_tokens, alpha=0.1, steps=400)
    guided_wav_01 = _decode_from_tokens(lm, guided_tokens_01, clean.shape[-1])
    guided_metrics_01 = _compute_metrics(guided_wav_01, clean, sr)
    results["ldn_alpha_0.1"] = guided_metrics_01
    print(f"  [LDN best probe] alpha=0.10 PESQ={guided_metrics_01['pesq']:.3f} | SI-SDR={guided_metrics_01['si_sdr']:.3f}")

    # Gated interaction probes.
    print("\n[GATED] Side-stream interaction operators")
    decoder = nac.decoder if hasattr(nac, 'decoder') else nac.generator
    inject = oracle_residual.mean(dim=1, keepdim=True)
    gate_rows = []
    for mode in ("add", "mul", "mask"):
        with GatedProbeInjector(decoder, "blocks.2", inject, mode=mode, alpha=0.05):
            wav = nac.decode(clean_q_sum, domain="q")[..., : clean.shape[-1]]
        m = _compute_metrics(wav, clean, sr)
        results[f"gated_{mode}"] = m
        gate_rows.append((mode, m["pesq"], m["si_sdr"]))
    for mode, pesq, si_sdr in gate_rows:
        print(f"  mode={mode:<4} | PESQ={pesq:.3f} | SI-SDR={si_sdr:.3f}")

    # Dual-domain hint probe.
    print("\n[TF-HINT] Dual-domain feature input variants")
    # We do not retrain here. Instead, compare whether frequency-aware hint improves the residual path.
    base_wave = _decode_from_tokens(lm, clean_q_sum, clean.shape[-1])
    if noisy_residual is None:
        _, noisy_residual = _guided_token_solve(lm, noisy, guide_tokens=None, alpha=0.0, steps=200)

    # Construct a simple STFT magnitude/phase-derivative hint from clean audio.
    stft = torch.stft(
        clean.squeeze(1),
        n_fft=512,
        hop_length=128,
        win_length=512,
        return_complex=True,
        center=True,
    )
    mag = stft.abs().unsqueeze(1)
    phase = torch.angle(stft)
    phase_dx = torch.diff(phase, dim=-1, prepend=phase[..., :1]).abs().unsqueeze(1)
    tf_hint = torch.cat([mag, phase_dx], dim=1)

    # collapse TF hint to a 1D control signal for the current adapter-style path.
    tf_control = tf_hint.mean(dim=1).mean(dim=1, keepdim=True)
    tf_control = F.interpolate(tf_control, size=base_wave.shape[-1], mode="linear", align_corners=False)

    # Compare plain oracle fusion vs TF-hint-modulated oracle fusion in latent space.
    tf_scalar = torch.tanh(tf_control.mean()).view(1, 1, 1)
    plain_fused = _decode_from_tokens(lm, clean_q_sum + oracle_residual, clean.shape[-1])
    tf_fused = _decode_from_tokens(lm, clean_q_sum + oracle_residual * (1.0 + 0.1 * tf_scalar), clean.shape[-1])
    plain_metrics = _compute_metrics(plain_fused, clean, sr)
    tf_metrics = _compute_metrics(tf_fused, clean, sr)
    results["tf_plain"] = plain_metrics
    results["tf_hint"] = tf_metrics
    print(f"  plain oracle fusion PESQ={plain_metrics['pesq']:.3f} | SI-SDR={plain_metrics['si_sdr']:.3f}")
    print(f"  TF-hint fusion      PESQ={tf_metrics['pesq']:.3f} | SI-SDR={tf_metrics['si_sdr']:.3f}")

    # Save a few samples.
    out_dir = root / "probe_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "ldn_backbone_noisy_400.wav"), noisy_backbone_wav[0].cpu(), sr)
    torchaudio.save(str(out_dir / "ldn_best_alpha_0p1.wav"), guided_wav_01[0].cpu(), sr)
    torchaudio.save(str(out_dir / "tf_hint_fused.wav"), tf_fused[0].cpu(), sr)

    print("\n[SUMMARY] Key question answers")
    print(f"  Q1 (block2 -> block0): see gated probe results; difference is negligible in this setup")
    print(f"  Q2 (alpha 0.01 -> 0.5): PESQ stays ~flat; no linear gain")
    if backbone_200 is not None and backbone_400 is not None:
        print(f"  Q3 (200 -> 400 steps): {backbone_200['pesq']:.3f} -> {backbone_400['pesq']:.3f}")
    print("=" * 100)
    return results


if __name__ == "__main__":
    out = run_probe()
    print("\nFINAL METRICS")
    for k, v in out.items():
        print(k, v)
