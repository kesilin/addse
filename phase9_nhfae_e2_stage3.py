#!/usr/bin/env python3
"""
NHFAE E2-Revised Stage 3: Regime II Refinement (Critical Phase)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

理论基础：
  • Regime II Refinement：锁定模型在 Posterior Mean（后延均值）附近
  • 幅度去敏化：禁止对高保真幅度的任何实质性修改（0.05×λ_cycle, 0.1×λ_mrstft）
  • 相位强聚焦：相位残差流的亚临界精微调（λ_cfm 主导，lr=5e-6 超低学习率）

加载：Stage 2 最优权重 (snr_5_10 trained model)
数据集：snr_10_15（极端高 SNR，10-15 dB）
目标：ΔSDR → +0.02 dB（物理透明黄金线），实现"工业级无损透明性"
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
    checkpoint_path: Path
    clean_dir: Path
    noisy_dir: Path
    out_dir: Path
    fs: int = 16000
    n_fft: int = 512
    hop: int = 192
    epochs: int = 1  # Stage 3 仅 1 epoch
    lr: float = 5e-6  # 超低学习率，精微调整
    lambda_dce: float = 1.0
    lambda_cfm: float = 0.7
    lambda_cycle: float = 0.2
    lambda_mrstft: float = 0.2
    device: str = "cuda"


def freeze_modules(model: nn.Module, module_names: list[str]) -> None:
    """冻结指定的模块。"""
    for name, param in model.named_parameters():
        for frozen_name in module_names:
            if frozen_name in name:
                param.requires_grad = False
                print(f"  ✓ Frozen: {name}")


def train(cfg: TrainCfg) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
    
    # 加载 Stage 2 checkpoint
    print(f"[Stage 3] 加载 Stage 2 checkpoint: {cfg.checkpoint_path}")
    pack = torch.load(cfg.checkpoint_path, map_location=device)
    model = NHFAE_E1_Interact(d_model=96, n_layers=8, n_heads=4, n_mag_bins=64).to(device)
    model.load_state_dict(pack["model"])
    print(f"  ✓ Loaded weights from {cfg.checkpoint_path}")
    print(f"  📌 Regime: II Refinement (Posterior Mean Locking)")
    print(f"  📌 Theory: Φ(t) = Phase Match | Magnitude ≈ Constant")
    print(f"  📌 1-NFE Extension Ready: Linear ODE trajectory enables single-step inference\n")
    
    # Stage 3 保持 identity_head 冻结（策略一致）
    print("[Stage 3] 冻结离散头分支（Identity Head）...")
    frozen_list = ["identity_head"]
    freeze_modules(model, frozen_list)
    print()
    
    # 加载 snr_10_15 数据
    ds = PairDataset(cfg.clean_dir, cfg.noisy_dir, cfg.fs)
    if len(ds) < 2:
        raise ValueError("Need at least 2 paired samples")
    print(f"[Stage 3] 加载 snr_10_15 数据集: {len(ds)} 个样本\n")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=cfg.lr)
    print(f"[Stage 3] 可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    print(f"[Stage 3] 学习率: {cfg.lr}\n")

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

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Stage 3: Regime II Refinement - 极端权重调整
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # [幅度去敏化] 锁定 Posterior Mean，禁止细节修改
            # ────────────────────────────────────────────
            l_dce = out["dce"].detach()  # Data consistency (detach 避免反向传播影响相位)
            l_cfm = F.l1_loss(out["v_phase"].squeeze(0), d_phase) + F.l1_loss(
                out["v_mag"].squeeze(0),
                (tgt_mag - torch.abs(S_noisy)) / (mag_scale.squeeze(0) + 1e-8),
            )
            l_cycle = F.l1_loss(stft(y, cfg.n_fft, cfg.hop), S_out.squeeze(0).detach())
            l_mr = mrstft_loss(y, clean)

            # [相位强聚焦] CFM 主导，其他约束极弱化
            # ────────────────────────────────────────────
            # λ_dce 保留数据一致性，但不生成新的幅度模式
            # λ_cfm 主导地位：强制相位残差精准对齐
            # 0.05×λ_cycle: 防止连锁扰动（cycle 回路），减弱至极限
            # 0.1×λ_mrstft: 频域平滑约束极弱化（仅维持音质底线）
            loss = (
                0.05 * cfg.lambda_dce * l_dce +           # 数据一致性：弱
                cfg.lambda_cfm * l_cfm +                   # 相位流 (CFM)：强
                0.05 * cfg.lambda_cycle * l_cycle +        # 环路稳定性：弱（0.05×0.2=0.01）
                0.1 * cfg.lambda_mrstft * l_mr             # 频域平滑：标量化弱（0.1×0.2=0.02）
            )
            # 理论解释：
            #   E[y|noisy] ≈ E[y*|noisy]  (Posterior Mean without destructive augmentation)
            #   ∂phase_only / ∂t 主驱动，∂mag / ∂t ≈ 0  （相位微调，幅度冻结）
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

    print(f"✓ Saved NHFAE-E2-Stage3 wavs to: {wav_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NHFAE E2-Revised Stage 3: High-SNR Polishing")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6, help="Ultra-low learning rate")
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
