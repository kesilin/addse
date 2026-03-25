#!/usr/bin/env python3
"""
NHFAE E2-Revised Stage 2: Partial Freezing Fine-Tuning
加载 tune2_lowsnr (snr_0_5 optimization)，在 snr_5_10 上 fine-tune
冻结离散头（identity_head）和其依赖的Embedding，只训练主干+flow_head
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase9_nhfae_e1 import (
    AttentionTimeBlock,
    MambaLikeBlock,
    PairDataset,
    istft,
    mrstft_loss,
    stft,
    wrap_to_pi,
)
from phase9_nhfae_e1_interact import NHFAE_E1_Interact


@dataclass
class TrainCfg:
    checkpoint_path: Path  # 加载的 Stage 1 checkpoint
    clean_dir: Path
    noisy_dir: Path
    out_dir: Path
    fs: int = 16000
    n_fft: int = 512
    hop: int = 192
    epochs: int = 2
    lr: float = 1e-5  # Stage 2 学习率更低
    lambda_dce: float = 1.0
    lambda_cfm: float = 0.7
    lambda_cycle: float = 0.2
    lambda_mrstft: float = 0.2
    device: str = "cuda"
    frozen_modules: list = None  # 要冻结的模块列表


def freeze_modules(model: nn.Module, module_names: list[str]) -> None:
    """冻结指定的模块，使其不参与梯度更新。"""
    for name, param in model.named_parameters():
        for frozen_name in module_names:
            if frozen_name in name:
                param.requires_grad = False
                print(f"  ✓ Frozen: {name}")


def train(cfg: TrainCfg) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
    
    # 1. 加载 Stage 1 checkpoint
    print(f"[Stage 2] 加载 Stage 1 checkpoint: {cfg.checkpoint_path}")
    pack = torch.load(cfg.checkpoint_path, map_location=device)
    model = NHFAE_E1_Interact(d_model=96, n_layers=8, n_heads=4, n_mag_bins=64).to(device)
    model.load_state_dict(pack["model"])
    print(f"  ✓ Loaded weights from {cfg.checkpoint_path}\n")
    
    # 2. 冻结离散头分支
    print("[Stage 2] 冻结离散头分支（Identity Head）...")
    frozen_list = ["identity_head"]  # 冻结 identity_head
    freeze_modules(model, frozen_list)
    print()
    
    # 3. 加载 Stage 2 数据（snr_5_10）
    ds = PairDataset(cfg.clean_dir, cfg.noisy_dir, cfg.fs)
    if len(ds) < 2:
        raise ValueError("Need at least 2 paired samples")
    print(f"[Stage 2] 加载 snr_5_10 数据集: {len(ds)} 个样本\n")

    # 4. 仅优化未冻结的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=cfg.lr)
    print(f"[Stage 2] 可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    print(f"[Stage 2] 学习率: {cfg.lr}\n")

    ckpt_dir = cfg.out_dir / "ckpt"
    wav_dir = cfg.out_dir / "wav"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for i in range(len(ds)):
            _, clean, noisy = ds.get(i)
            clean = clean.to(device)
            noisy = noisy.to(device)

            S_clean = stft(clean, cfg.n_fft, cfg.hop)
            S_noisy = stft(noisy, cfg.n_fft, cfg.hop)

            out = model(S_noisy.unsqueeze(0))
            mag_n = torch.abs(S_noisy).unsqueeze(0)
            mag_scale = torch.amax(mag_n, dim=(1, 2), keepdim=True) + 1e-8
            mag_out = out["mag_mix"] * mag_scale
            S_out = mag_out * torch.exp(1j * out["phase_out"])
            y = istft(S_out.squeeze(0), length=clean.numel(), n_fft=cfg.n_fft, hop=cfg.hop)

            tgt_mag = torch.abs(S_clean)
            tgt_phase = torch.angle(S_clean)
            d_phase = wrap_to_pi(tgt_phase - torch.angle(S_noisy))

            # ===== 冻结分支时，DCE loss 仍计算但不参与反向传播 =====
            l_dce = out["dce"].detach()  # Stop gradient for frozen identity head
            l_cfm = F.l1_loss(out["v_phase"].squeeze(0), d_phase) + F.l1_loss(
                out["v_mag"].squeeze(0),
                (tgt_mag - torch.abs(S_noisy)) / (mag_scale.squeeze(0) + 1e-8),
            )
            l_cycle = F.l1_loss(stft(y, cfg.n_fft, cfg.hop), S_out.squeeze(0).detach())
            l_mr = mrstft_loss(y, clean)

            # Stage 2: 减弱 DCE 权重（因为身份分类已冻结），专注相位流优化
            loss = 0.1 * cfg.lambda_dce * l_dce + cfg.lambda_cfm * l_cfm + cfg.lambda_cycle * l_cycle + cfg.lambda_mrstft * l_mr

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 5.0)
            opt.step()
            losses.append(float(loss.item()))

        m = float(np.mean(losses))
        print(f"[ep {ep:03d}] loss={m:.6f}")
        if m < best:
            best = m
            torch.save({"model": model.state_dict(), "loss": best}, ckpt_dir / "best.pt")

    print(f"\n✓ Saved best checkpoint to: {ckpt_dir / 'best.pt'}")

    # 推理
    pack = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(pack["model"])
    model.eval()

    with torch.no_grad():
        for i in range(len(ds)):
            name, _, noisy = ds.get(i)
            noisy = noisy.to(device)
            S_noisy = stft(noisy, cfg.n_fft, cfg.hop)
            out = model(S_noisy.unsqueeze(0))
            mag_n = torch.abs(S_noisy).unsqueeze(0)
            mag_scale = torch.amax(mag_n, dim=(1, 2), keepdim=True) + 1e-8
            mag_out = out["mag_mix"] * mag_scale
            S_out = mag_out * torch.exp(1j * out["phase_out"])
            y = istft(S_out.squeeze(0), length=noisy.numel(), n_fft=cfg.n_fft, hop=cfg.hop)
            y = y.detach().cpu().numpy().astype(np.float32, copy=False)
            peak = float(np.max(np.abs(y)) + 1e-8)
            if peak > 1.0:
                y = y / peak
            sf.write(wav_dir / name, y, cfg.fs)

    print(f"✓ Saved NHFAE-E2-Stage2 wavs to: {wav_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NHFAE E2-Revised Stage 2: Partial Freezing Fine-Tune")
    parser.add_argument("--checkpoint-path", required=True, help="Path to Stage 1 checkpoint")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5, help="Fine-tune learning rate (lower by default)")
    parser.add_argument("--lambda-dce", type=float, default=1.0)
    parser.add_argument("--lambda-cfm", type=float, default=0.7)
    parser.add_argument("--lambda-cycle", type=float, default=0.2)
    parser.add_argument("--lambda-mrstft", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = TrainCfg(
        checkpoint_path=Path(args.checkpoint_path),
        clean_dir=Path(args.clean_dir),
        noisy_dir=Path(args.noisy_dir),
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        lr=args.lr,
        lambda_dce=args.lambda_dce,
        lambda_cfm=args.lambda_cfm,
        lambda_cycle=args.lambda_cycle,
        lambda_mrstft=args.lambda_mrstft,
        device=args.device,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    train(cfg)
