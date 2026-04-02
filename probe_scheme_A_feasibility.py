#!/usr/bin/env python3
"""Scheme A feasibility probe: decoder-local unfreeze with oracle residual.

This script validates whether lightly unfreezing NAC decoder tail layers can
adapt latent-manifold mismatch (Z_q + residual) on a single clean sample.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torchaudio
import yaml
from hydra.utils import instantiate

from addse.losses import MSMelSpecLoss, SDRLoss
from addse.metrics import PESQMetric, SDRMetric, STOIMetric
from addse.models.msstftd import MSSTFTDiscriminator


def _resolve_existing(paths: list[str]) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")


def _decoder_module(nac: torch.nn.Module) -> torch.nn.Module:
    # NAC in this repo exposes decoder directly.
    if hasattr(nac, "decoder"):
        return nac.decoder
    # Defensive fallback for other wrappers.
    if hasattr(nac, "generator"):
        return nac.generator
    raise AttributeError("Cannot find decoder/generator module in NAC")


def _set_trainable_tail(decoder: torch.nn.Module) -> list[str]:
    # Freeze all by default.
    for _, p in decoder.named_parameters():
        p.requires_grad = False

    trainable_names: list[str] = []

    # Expected structure: decoder.blocks is nn.Sequential of NACDecoderBlock.
    # Expand surgery area to include:
    # 1) the last upsampler conv (decoder.blocks[-1].conv),
    # 2) the last block residual units,
    # 3) output projection (decoder.out_conv, analogous to conv_post).
    if hasattr(decoder, "blocks") and isinstance(decoder.blocks, torch.nn.Sequential):
        n_blocks = len(decoder.blocks)
        last_idx = n_blocks - 1
        block = decoder.blocks[last_idx]
        for name, p in block.named_parameters():
            if name.startswith("conv.") or name.startswith("residual_blocks."):
                p.requires_grad = True
                trainable_names.append(f"blocks.{last_idx}.{name}")

        # Also keep one more residual block trainable for extra capacity.
        if n_blocks >= 2:
            prev_idx = n_blocks - 2
            prev_block = decoder.blocks[prev_idx]
            for name, p in prev_block.named_parameters():
                if name.startswith("residual_blocks."):
                    p.requires_grad = True
                    trainable_names.append(f"blocks.{prev_idx}.{name}")

    if hasattr(decoder, "out_conv"):
        for name, p in decoder.out_conv.named_parameters():
            p.requires_grad = True
            trainable_names.append(f"out_conv.{name}")

    # If structure mismatch, fallback to last ~20% params.
    if not trainable_names:
        named = list(decoder.named_parameters())
        cut = int(len(named) * 0.8)
        for name, p in named[cut:]:
            p.requires_grad = True
            trainable_names.append(name)

    return trainable_names


def _decode_from_q(nac: torch.nn.Module, z_q_sum: torch.Tensor) -> torch.Tensor:
    # For this NAC implementation, decode(domain="q") passes through decoder.
    return nac.decode(z_q_sum, domain="q")


def _load_nac_flexible(nac_cfg: str, nac_ckpt: str, device: torch.device) -> torch.nn.Module:
    with open(nac_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    nac = instantiate(cfg["lm"]["generator"]).to(device)

    ckpt = torch.load(nac_ckpt, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

    raw_state = ckpt.get("state_dict", ckpt)
    if not isinstance(raw_state, dict):
        raise RuntimeError("Checkpoint does not contain a valid state dict")

    # Try common key layouts in order.
    candidates = [
        {k.removeprefix("generator."): v for k, v in raw_state.items() if isinstance(k, str) and k.startswith("generator.")},
        {k.removeprefix("nac."): v for k, v in raw_state.items() if isinstance(k, str) and k.startswith("nac.")},
        {k: v for k, v in raw_state.items() if isinstance(k, str) and (k.startswith("encoder.") or k.startswith("decoder.") or k.startswith("quantizer."))},
    ]

    loaded = False
    for i, sd in enumerate(candidates, start=1):
        if not sd:
            continue
        try:
            ret = nac.load_state_dict(sd, strict=False)
            print(
                f"Loaded NAC from checkpoint layout #{i} with {len(sd)} tensors "
                f"(missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)})"
            )
            loaded = True
            break
        except RuntimeError:
            continue

    if not loaded:
        raise RuntimeError("Failed to load NAC weights from checkpoint with known key layouts")

    nac.eval()
    for p in nac.parameters():
        p.requires_grad = False
    return nac


def run_scheme_a_probe() -> dict[str, float]:
    print("--- Start Scheme A decoder-adaptation feasibility probe ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    repo_root = Path(__file__).resolve().parent

    nac_cfg = _resolve_existing([
        str(repo_root / "configs" / "nac.yaml"),
    ])
    nac_ckpt = _resolve_existing([
        str(repo_root / "logs" / "addse-edbase-quick" / "checkpoints" / "last.ckpt"),
        str(repo_root / "logs" / "addse-edbase-quick" / "checkpoints" / "epoch=08-val_loss=5.86.ckpt"),
        str(repo_root / "logs" / "addse-edbase-quick" / "checkpoints" / "addse-s.ckpt"),
    ])

    test_wav = _resolve_existing([
        str(repo_root / "saved_audio_v33" / "edbase-local_000000_clean.wav"),
        str(repo_root / "probe_outputs" / "probe_A_discrete_only.wav"),
    ])

    print(f"Using NAC cfg: {nac_cfg}")
    print(f"Using NAC ckpt: {nac_ckpt}")
    print(f"Using test wav: {test_wav}")

    nac = _load_nac_flexible(nac_cfg, nac_ckpt, device)

    decoder = _decoder_module(nac)
    trainable = _set_trainable_tail(decoder)
    print(f"Trainable decoder params: {len(trainable)}")
    if trainable:
        print(f"  e.g. {trainable[:5]}")

    clean_wav, sr = torchaudio.load(test_wav)
    if clean_wav.ndim == 1:
        clean_wav = clean_wav.unsqueeze(0)
    clean_wav = clean_wav[:1].unsqueeze(0).to(device)  # (B=1,C=1,T)

    n_pad = (nac.downsampling_factor - clean_wav.shape[-1] % nac.downsampling_factor) % nac.downsampling_factor
    clean_wav_pad = torch.nn.functional.pad(clean_wav, (0, n_pad))

    with torch.no_grad():
        z_lat = nac.encoder(clean_wav_pad)
        _, z_q = nac.encode(clean_wav_pad, no_sum=True, domain="q")
        z_q_sum = z_q.sum(dim=2) if z_q.ndim == 4 else z_q
        oracle_residual = (z_lat - z_q_sum).detach()

        y_discrete = _decode_from_q(nac, z_q_sum)[..., : clean_wav.shape[-1]]
        y_oracle_before = _decode_from_q(nac, z_q_sum + oracle_residual)[..., : clean_wav.shape[-1]]

    # Metrics before adaptation.
    pesq_metric = PESQMetric(fs=sr)
    estoi_metric = STOIMetric(fs=sr, extended=True)
    si_sdr_metric = SDRMetric(scale_invariant=True, zero_mean=True)

    base_pesq = pesq_metric(y_discrete[0], clean_wav[0])
    oracle_before_pesq = pesq_metric(y_oracle_before[0], clean_wav[0])

    print(f"[Before] Discrete-only PESQ: {base_pesq:.3f}")
    print(f"[Before] Oracle-fused PESQ: {oracle_before_pesq:.3f}")

    # Adaptation loop.
    params = [p for p in decoder.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable decoder parameters selected")

    optimizer = torch.optim.Adam(params, lr=2e-4)
    discriminator = MSSTFTDiscriminator(in_channels=1, out_channels=1).to(device)
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    mel_loss = MSMelSpecLoss(fs=sr).to(device)
    si_sdr_loss = SDRLoss(scale_invariant=True, zero_mean=True).to(device)

    w_spec = 1.0
    w_si = 0.02
    w_gan = 0.1
    gan_start_step = 40

    out_dir = repo_root / "probe_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "scheme_a_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    decoder.train()
    discriminator.train()
    steps = 150

    # Track best PESQ and save intermediate checkpoints.
    best_pesq = oracle_before_pesq
    best_pesq_step = 0
    best_checkpoint_path = None

    for i in range(1, steps + 1):
        d_loss = torch.zeros((), device=device)
        use_gan = i >= gan_start_step

        # 1) Update discriminator (hinge loss) once GAN is enabled.
        if use_gan:
            with torch.no_grad():
                y_hat_detached = _decode_from_q(nac, z_q_sum + oracle_residual)[..., : clean_wav.shape[-1]]

            real_outs, _ = discriminator(clean_wav)
            fake_outs, _ = discriminator(y_hat_detached.detach())

            d_loss = 0.0
            for real_o, fake_o in zip(real_outs, fake_outs):
                d_loss = d_loss + torch.relu(1.0 - real_o).mean() + torch.relu(1.0 + fake_o).mean()
            d_loss = d_loss / len(real_outs)

            disc_optim.zero_grad(set_to_none=True)
            d_loss.backward()
            disc_optim.step()

        # 2) Update decoder (spec + si-sdr + adversarial).
        optimizer.zero_grad(set_to_none=True)
        y_hat = _decode_from_q(nac, z_q_sum + oracle_residual)[..., : clean_wav.shape[-1]]

        loss_dict = mel_loss(y_hat, clean_wav)
        spec = loss_dict["loss"]
        si = si_sdr_loss(y_hat, clean_wav)["loss"]

        if use_gan:
            fake_for_g, _ = discriminator(y_hat)
            g_adv = 0.0
            for fake_o in fake_for_g:
                g_adv = g_adv - fake_o.mean()
            g_adv = g_adv / len(fake_for_g)
        else:
            g_adv = torch.zeros((), device=device)

        loss = w_spec * spec + w_si * si + w_gan * g_adv
        loss.backward()
        optimizer.step()

        # Every 10 steps, evaluate PESQ and save checkpoint if improved.
        if i % 10 == 0 or i == 1:
            with torch.no_grad():
                y_eval = _decode_from_q(nac, z_q_sum + oracle_residual)[..., : clean_wav.shape[-1]]
                eval_pesq = pesq_metric(y_eval[0], clean_wav[0])

            if eval_pesq > best_pesq:
                best_pesq = eval_pesq
                best_pesq_step = i
                best_checkpoint_path = ckpt_dir / f"decoder_step{i:03d}_pesq{eval_pesq:.3f}.pth"
                torch.save(decoder.state_dict(), str(best_checkpoint_path))

            print(
                f"Step {i:03d} | total: {loss.item():.4f} | "
                f"spec: {spec.item():.4f} | si: {si.item():.4f} | gan: {g_adv.item():.4f} | d: {d_loss.item():.4f} | "
                f"eval_pesq: {eval_pesq:.3f} (best: {best_pesq:.3f} @ step {best_pesq_step})"
            )

    # Restore best decoder checkpoint and re-evaluate.
    if best_checkpoint_path is not None:
        print(f"\nRestoring best checkpoint from {best_checkpoint_path}")
        decoder.load_state_dict(torch.load(str(best_checkpoint_path), map_location=device))

    decoder.eval()
    with torch.no_grad():
        y_oracle_after = _decode_from_q(nac, z_q_sum + oracle_residual)[..., : clean_wav.shape[-1]]

    after_pesq = pesq_metric(y_oracle_after[0], clean_wav[0])
    after_estoi = estoi_metric(y_oracle_after[0], clean_wav[0])
    after_si_sdr = si_sdr_metric(y_oracle_after[0], clean_wav[0])

    out_file = out_dir / "SchemeA_fixed_v3_final.wav"
    torchaudio.save(str(out_file), y_oracle_after[0].detach().cpu(), sr)

    print("\nAdaptation finished")
    print(f"Output saved: {out_file}")
    print(f"[After ] Oracle-fused PESQ:  {after_pesq:.3f}")
    print(f"[After ] Oracle-fused ESTOI: {after_estoi:.3f}")
    print(f"[After ] Oracle-fused SI-SDR:{after_si_sdr:.3f}")
    print(f"[Delta ] PESQ improvement:   {after_pesq - oracle_before_pesq:+.3f}")

    return {
        "pesq_discrete": float(base_pesq),
        "pesq_oracle_before": float(oracle_before_pesq),
        "pesq_oracle_after": float(after_pesq),
        "estoi_oracle_after": float(after_estoi),
        "si_sdr_oracle_after": float(after_si_sdr),
    }


if __name__ == "__main__":
    run_scheme_a_probe()
