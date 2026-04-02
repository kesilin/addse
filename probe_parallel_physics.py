#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel Physics Probe v2

Covers:
- Exp2 Oracle Blueprint Gap
- Exp4 Sample-rate alignment / phase consistency hard injection
- Exp5 Conditioning strength (with-hint vs no-hint)
- Q1 Injection point move (Block2 -> Block0)
- Q2 Alpha scan (0.01 -> 0.5)
- Q3 Backbone token quality (200 vs 400 diffusion steps)

Note:
Exp1 and Exp3 are run from run_v33.py in terminal so you can observe training logs directly.
"""

from __future__ import annotations

import io
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

# Force UTF-8 in Windows terminal output.
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Ensure relative paths in hydra configs (e.g., configs/nac.yaml) resolve correctly.
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)


@dataclass
class ProbeContext:
    lm: torch.nn.Module
    nac: torch.nn.Module
    clean: torch.Tensor
    noisy: torch.Tensor
    sr: int
    device: torch.device


def _resolve_existing(paths: list[str]) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of candidates exists: {paths}")


def _load_lm(lm_cfg: str, lm_ckpt: str, device: torch.device):
    cfg, _ = load_hydra_config(lm_cfg)
    lm = instantiate(cfg.lm).to(device)
    ckpt = torch.load(lm_ckpt, map_location="cpu")
    lm.load_state_dict(ckpt["state_dict"], strict=False)
    lm.eval()
    return lm


def _pad_to_codec(nac: torch.nn.Module, wav: torch.Tensor) -> torch.Tensor:
    n_pad = (nac.downsampling_factor - wav.shape[-1] % nac.downsampling_factor) % nac.downsampling_factor
    return F.pad(wav, (0, n_pad))


@torch.no_grad()
def _predict_tokens_and_residual(lm: torch.nn.Module, noisy_wav: torch.Tensor, num_steps: int):
    nac = lm.nac
    x_pad = _pad_to_codec(nac, noisy_wav)
    x_lat = nac.encoder(x_pad)
    x_tok, x_q = nac.encode(x_pad, no_sum=True, domain="q")

    B, K, L = x_tok.shape
    y_tok = torch.full_like(x_tok, lm.mask_token)

    final_residual = None
    final_conf = None

    import math

    for i in range(num_steps):
        mask = y_tok == lm.mask_token
        if not mask.any():
            break

        y_q_step = nac.quantizer.decode(y_tok.masked_fill(mask, 0), output_no_sum=True, domain="code")
        if y_q_step.shape != x_q.shape and y_q_step.ndim == 4:
            if y_q_step.shape[2] == x_q.shape[1]:
                y_q_step = y_q_step.transpose(1, 2)
            elif y_q_step.shape[1] != x_q.shape[1]:
                y_q_step = y_q_step.reshape(x_q.shape)
        y_q_step = y_q_step.masked_fill(mask.unsqueeze(1), 0)

        log_p, residual_pred, _ = lm.log_score(y_q_step, x_q, x_cont=x_lat)
        if residual_pred is not None:
            if final_residual is None:
                final_residual = residual_pred
            else:
                decay = min(max(lm.residual_ema_decay, 0.0), 0.99)
                final_residual = decay * final_residual + (1.0 - decay) * residual_pred

        probs = log_p.exp()
        final_conf = probs.max(dim=-1).values.mean(dim=(1, 2), keepdim=True)
        sampled = probs.argmax(dim=-1)[mask]
        y_tok_new = y_tok.clone()
        y_tok_new[mask] = sampled

        if i == num_steps - 1:
            y_tok = y_tok_new
            break

        confidence = probs[mask].gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        conf_full = torch.full_like(y_tok, 1e9, dtype=torch.float32)
        conf_full[mask] = confidence

        ratio = math.cos(math.pi / 2 * (i + 1) / num_steps)
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

    return y_q_discrete, final_residual, final_conf


class _SinglePointInjector:
    def __init__(self, decoder: nn.Module, point: str, residual_signal: torch.Tensor, alpha: float) -> None:
        self.decoder = decoder
        self.point = point
        self.residual_signal = residual_signal
        self.alpha = alpha
        self.hook = None

    def __enter__(self):
        parts = self.point.split(".")
        mod = self.decoder
        for p in parts:
            mod = mod[int(p)] if p.isdigit() else getattr(mod, p)

        def _hook(_m, _i, out):
            res = self.residual_signal
            if res.shape[-1] != out.shape[-1]:
                res_i = F.interpolate(res, size=out.shape[-1], mode="linear", align_corners=False)
            else:
                res_i = res
            if res_i.shape[1] != out.shape[1]:
                # lightweight channel projection (no grad needed for probe)
                w = torch.zeros((out.shape[1], res_i.shape[1], 1), device=out.device, dtype=out.dtype)
                n = min(out.shape[1], res_i.shape[1])
                w[:n, :n, 0] = 1.0
                res_i = F.conv1d(res_i, w)
            return out + self.alpha * res_i

        self.hook = mod.register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook is not None:
            self.hook.remove()


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
def build_context() -> ProbeContext:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parent

    lm_cfg = _resolve_existing([str(root / "configs" / "addse-s-edbase-parallel60-a008-p02-spec.yaml")])
    lm_ckpt = _resolve_existing([
        str(root / "logs" / "addse-s-edbase-parallel60-a008-p02-spec" / "checkpoints" / "last.ckpt"),
        str(root / "logs" / "addse-s-edbase-parallel60-a008-p02-spec" / "checkpoints" / "epoch=01-val_loss=6.01.ckpt"),
    ])

    clean_wav = _resolve_existing([str(root / "saved_audio_v33" / "edbase-local_000000_clean.wav")])
    clean, sr = torchaudio.load(clean_wav)
    clean = clean[:1].unsqueeze(0).to(device)

    # Create controlled noisy input for DiT token prediction tests.
    noise = torch.randn_like(clean)
    snr_db = 5.0
    clean_rms = clean.pow(2).mean().sqrt().clamp_min(1e-8)
    noise_rms = noise.pow(2).mean().sqrt().clamp_min(1e-8)
    scale = clean_rms / (10 ** (snr_db / 20.0) * noise_rms)
    noisy = (clean + scale * noise).clamp(-1.0, 1.0)

    lm = _load_lm(lm_cfg, lm_ckpt, device)

    print("=" * 88)
    print("[Probe] Loaded LM context")
    print(f"  device={device} | sr={sr} | lm_ckpt={lm_ckpt}")
    print("=" * 88)
    return ProbeContext(lm=lm, nac=lm.nac, clean=clean, noisy=noisy, sr=sr, device=device)


@torch.no_grad()
def run_probe() -> dict:
    ctx = build_context()
    out_dir = Path(__file__).resolve().parent / "probe_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_pad = _pad_to_codec(ctx.nac, ctx.clean)
    noisy_pad = _pad_to_codec(ctx.nac, ctx.noisy)

    z_clean_lat = ctx.nac.encoder(clean_pad)
    _, z_clean_q = ctx.nac.encode(clean_pad, no_sum=True, domain="q")
    z_clean_q_sum = z_clean_q.sum(dim=2) if z_clean_q.ndim == 4 else z_clean_q
    oracle_res = z_clean_lat - z_clean_q_sum

    results: dict[str, dict] = {}

    # Q3 + Exp2: token blueprint quality and oracle blueprint gap.
    print("\n[Exp2/Q3] Oracle Blueprint Gap + 200 vs 400 steps")
    for steps in (200, 400):
        y_q_pred, final_res, _ = _predict_tokens_and_residual(ctx.lm, ctx.noisy, num_steps=steps)
        wav_backbone = ctx.lm._decode_latent_to_wave(y_q_pred, target_length=ctx.clean.shape[-1])
        wav_oracle_gap = ctx.lm._decode_latent_to_wave(y_q_pred + oracle_res, target_length=ctx.clean.shape[-1])
        m_back = _compute_metrics(wav_backbone, ctx.clean, ctx.sr)
        m_gap = _compute_metrics(wav_oracle_gap, ctx.clean, ctx.sr)

        results[f"q3_backbone_{steps}"] = m_back
        results[f"exp2_oracle_gap_{steps}"] = m_gap

        torchaudio.save(str(out_dir / f"q3_backbone_{steps}.wav"), wav_backbone[0].cpu(), ctx.sr)
        torchaudio.save(str(out_dir / f"exp2_oracle_gap_{steps}.wav"), wav_oracle_gap[0].cpu(), ctx.sr)

        print(
            f"  steps={steps} | backbone PESQ={m_back['pesq']:.3f} | oracle-gap PESQ={m_gap['pesq']:.3f} "
            f"| delta={m_gap['pesq'] - m_back['pesq']:+.3f}"
        )

        if final_res is not None:
            # Exp5: conditioning strength with/without hint.
            base_wave = ctx.lm._decode_latent_to_wave(y_q_pred, target_length=ctx.clean.shape[-1])
            wav_with_hint, _ = ctx.lm._predict_wave_residual(ctx.noisy, base_wave, final_res)
            wav_no_hint, _ = ctx.lm._predict_wave_residual(ctx.noisy, base_wave, None)
            m_with = _compute_metrics(wav_with_hint, ctx.clean, ctx.sr)
            m_no = _compute_metrics(wav_no_hint, ctx.clean, ctx.sr)
            results[f"exp5_with_hint_{steps}"] = m_with
            results[f"exp5_no_hint_{steps}"] = m_no
            print(
                f"  exp5 steps={steps} | with-hint PESQ={m_with['pesq']:.3f} | "
                f"no-hint PESQ={m_no['pesq']:.3f} | delta={m_no['pesq'] - m_with['pesq']:+.3f}"
            )

    # Exp4: phase consistency by hard injecting true waveform residual.
    print("\n[Exp4] Sample-rate alignment hard-injection check")
    y_q_pred_400, _, _ = _predict_tokens_and_residual(ctx.lm, ctx.noisy, num_steps=400)
    y_base = ctx.lm._decode_latent_to_wave(y_q_pred_400, target_length=ctx.clean.shape[-1])
    wav_true_res = (ctx.clean - y_base).detach()
    y_hard = y_base + wav_true_res
    m_hard = _compute_metrics(y_hard, ctx.clean, ctx.sr)
    results["exp4_hard_residual"] = m_hard
    print(
        f"  hard-injection PESQ={m_hard['pesq']:.3f} | SI-SDR={m_hard['si_sdr']:.3f} "
        f"(if still negative => likely alignment bug)"
    )

    # Q1 + Q2: injection-point move and alpha scan on decoder features.
    print("\n[Q1/Q2] Injection point and alpha sweep")
    decoder = ctx.nac.decoder if hasattr(ctx.nac, "decoder") else ctx.nac.generator

    # Build residual signal in decoder feature domain proxy (from oracle residual mean channel).
    proxy = oracle_res.mean(dim=1, keepdim=True)

    z_discrete = z_clean_q_sum

    # Q1: block2 vs block0 with fixed alpha=0.05
    q1_rows = []
    for point in ("blocks.2", "blocks.0"):
        with _SinglePointInjector(decoder, point, proxy, alpha=0.05):
            wav = ctx.nac.decode(z_discrete, domain="q")[..., : ctx.clean.shape[-1]]
        m = _compute_metrics(wav, ctx.clean, ctx.sr)
        results[f"q1_{point.replace('.', '_')}"] = m
        q1_rows.append((point, m["pesq"], m["si_sdr"]))
    for p, pesq, si in q1_rows:
        print(f"  point={p} | PESQ={pesq:.3f} | SI-SDR={si:.3f}")

    # Q2: alpha scan on one stable point
    alpha_rows = []
    for alpha in (0.01, 0.05, 0.1, 0.2, 0.5):
        with _SinglePointInjector(decoder, "blocks.2", proxy, alpha=alpha):
            wav = ctx.nac.decode(z_discrete, domain="q")[..., : ctx.clean.shape[-1]]
        m = _compute_metrics(wav, ctx.clean, ctx.sr)
        results[f"q2_alpha_{alpha}"] = m
        alpha_rows.append((alpha, m["pesq"], m["si_sdr"]))
    for a, pesq, si in alpha_rows:
        print(f"  alpha={a:.2f} | PESQ={pesq:.3f} | SI-SDR={si:.3f}")

    print("\n[Summary] Key outputs")
    print("  - q3_backbone_200.wav / q3_backbone_400.wav")
    print("  - exp2_oracle_gap_200.wav / exp2_oracle_gap_400.wav")
    print("  - Metrics dict emitted below")

    return results


if __name__ == "__main__":
    out = run_probe()
    print("\n" + "=" * 88)
    print("FINAL METRICS")
    print("=" * 88)
    for k, v in out.items():
        print(f"{k}: {v}")
