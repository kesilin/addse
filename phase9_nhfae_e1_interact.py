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


class MPICM(nn.Module):
    """Magnitude-Phase Interactive Convolution Module."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mag_gate = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=1),
        )
        self.phase_flow = nn.Sequential(
            nn.Conv2d(2, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=1),
        )

    def forward(self, mag_feat: torch.Tensor, phase_feat: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.mag_gate(mag_feat))
        flow = self.phase_flow(phase_feat)
        return gate * flow


class NHFAE_E1_Interact(nn.Module):
    def __init__(self, d_model: int = 96, n_layers: int = 8, n_heads: int = 4, n_mag_bins: int = 64):
        super().__init__()
        self.n_mag_bins = n_mag_bins

        self.stem = nn.Sequential(
            nn.Conv2d(3, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        blocks = []
        for i in range(n_layers):
            if i % 2 == 0:
                blocks.append(MambaLikeBlock(d_model))
            else:
                blocks.append(AttentionTimeBlock(d_model, n_heads))
        self.blocks = nn.ModuleList(blocks)

        self.interact = MPICM(d_model)

        self.identity_head = nn.Conv2d(d_model, n_mag_bins, kernel_size=1)
        self.flow_head = nn.Conv2d(d_model, 2, kernel_size=1)

        self.snr_est = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(d_model // 2, 1, kernel_size=1),
        )

        centers = torch.linspace(0.0, 1.0, steps=n_mag_bins)
        self.register_buffer("mag_centers", centers)

    def quantize_mag(self, logits: torch.Tensor, mag_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=1)
        soft = torch.sum(probs * self.mag_centers.view(1, -1, 1, 1), dim=1)
        idx = torch.argmax(logits, dim=1)
        hard = self.mag_centers[idx]
        q = soft + (hard - soft).detach()

        tgt = torch.clamp((mag_norm * (self.n_mag_bins - 1)).round().long(), 0, self.n_mag_bins - 1)
        dce = F.cross_entropy(logits, tgt)
        return q, dce

    def forward(self, noisy_stft: torch.Tensor) -> dict[str, torch.Tensor]:
        mag_n = torch.abs(noisy_stft)
        phase_n = torch.angle(noisy_stft)

        mag_log = torch.log1p(mag_n)
        cos_p = torch.cos(phase_n)
        sin_p = torch.sin(phase_n)

        feat = torch.stack([mag_log, cos_p, sin_p], dim=1)
        x = self.stem(feat)

        inter = self.interact(mag_log.unsqueeze(1), torch.stack([cos_p, sin_p], dim=1))
        x = x + inter

        for blk in self.blocks:
            x = blk(x)

        logits = self.identity_head(x)
        flow = self.flow_head(x)
        v_mag = flow[:, 0]
        v_phase = np.pi * torch.tanh(flow[:, 1])

        # ===== SNR 估计 (Global) =====
        snr_map = -5.0 + 25.0 * torch.sigmoid(self.snr_est(x)).squeeze(1)
        snr_global = torch.mean(snr_map, dim=(1, 2))  # 全局 SNR 估计
        
        fatc_gate = torch.sigmoid(-0.9 * (snr_map - 8.0))
        fatc_gate = torch.clamp(fatc_gate, min=0.01, max=0.99)

        mag_norm = mag_n / (torch.amax(mag_n, dim=(1, 2), keepdim=True) + 1e-8)
        _, dce = self.quantize_mag(logits, mag_norm)

        mag_gen = torch.clamp(mag_norm + 0.15 * torch.tanh(v_mag), 0.0, 1.0)
        mag_mix = fatc_gate * mag_gen + (1.0 - fatc_gate) * mag_norm
        phase_out = phase_n + fatc_gate * v_phase

        return {
            "logits": logits,
            "dce": dce,
            "mag_norm": mag_norm,
            "mag_mix": mag_mix,
            "phase_out": phase_out,
            "fatc_gate": fatc_gate,
            "v_mag": v_mag,
            "v_phase": v_phase,
            "snr_global": snr_global,
        }


def sigmoid_regime_switch(snr: torch.Tensor, tau: float = 5.0, k: float = 0.5) -> torch.Tensor:
    """
    动态损失权重函数：α(SNR) = 1 / (1 + exp(-k*(SNR - τ)))
    低 SNR (0 dB): α ≈ 0.2 (降低 cfm 权重，优先 PESQ)
    中 SNR (5 dB): α ≈ 0.5 (平衡)
    高 SNR (15 dB): α ≈ 1.0 (强化 cfm，优先 SDR)
    """
    return 1.0 / (1.0 + torch.exp(-k * (snr - tau)))


class MultiSNRDataset:
    """联合三个 SNR 桶的数据集。"""
    def __init__(self, snr_dirs: list[tuple[str, Path, Path]], fs: int = 16000):
        """
        snr_dirs: [("snr_0_5", clean_path, noisy_path), ...]
        """
        from phase9_nhfae_e1 import PairDataset
        self.datasets = []
        self.snr_names = []
        for name, clean_dir, noisy_dir in snr_dirs:
            ds = PairDataset(clean_dir, noisy_dir, fs)
            self.datasets.append(ds)
            self.snr_names.append(name)
        self.cumulative_sizes = np.cumsum([len(ds) for ds in self.datasets]).tolist()
        
    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def get(self, idx: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        # 找到该 idx 属于哪个数据集
        dataset_idx = 0
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                dataset_idx = i
                break
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get(local_idx)


@dataclass
class TrainCfg:
    clean_dir: Path = None
    noisy_dir: Path = None
    out_dir: Path = None
    fs: int = 16000
    n_fft: int = 512
    hop: int = 192
    epochs: int = 2
    lr: float = 2e-4
    lambda_dce: float = 1.0
    lambda_cfm: float = 0.7
    lambda_cycle: float = 0.2
    lambda_mrstft: float = 0.2
    device: str = "cuda"
    # SNR 动态权重参数
    snr_tau: float = 5.0  # 冲突点
    snr_k: float = 0.5    # 陡峭度
    # 多 SNR 模式
    multi_snr_mode: bool = False
    multi_snr_dirs: list = None


def train(cfg: TrainCfg) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
    
    # 选择单数据集或多数据集模式
    if cfg.multi_snr_mode and cfg.multi_snr_dirs:
        ds = MultiSNRDataset(cfg.multi_snr_dirs, cfg.fs)
        print(f"[Multi-SNR Mode] 加载 {len(ds)} 个样本")
    else:
        from phase9_nhfae_e1 import PairDataset
        ds = PairDataset(cfg.clean_dir, cfg.noisy_dir, cfg.fs)
    
    if len(ds) < 2:
        raise ValueError("Need at least 2 paired samples")

    model = NHFAE_E1_Interact().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

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

            l_dce = out["dce"]
            l_cfm = F.l1_loss(out["v_phase"].squeeze(0), d_phase) + F.l1_loss(
                out["v_mag"].squeeze(0),
                (tgt_mag - torch.abs(S_noisy)) / (mag_scale.squeeze(0) + 1e-8),
            )
            l_cycle = F.l1_loss(stft(y, cfg.n_fft, cfg.hop), S_out.squeeze(0).detach())
            l_mr = mrstft_loss(y, clean)

            # ===== 动态损失权重 α(SNR) =====
            snr_est = out["snr_global"][0]  # 标量
            alpha_snr = sigmoid_regime_switch(snr_est, tau=cfg.snr_tau, k=cfg.snr_k)
            
            # 带动态权重的总损失
            loss = cfg.lambda_dce * l_dce + (alpha_snr * cfg.lambda_cfm) * l_cfm + cfg.lambda_cycle * l_cycle + cfg.lambda_mrstft * l_mr

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))

        m = float(np.mean(losses))
        print(f"[ep {ep:03d}] loss={m:.6f}")
        if m < best:
            best = m
            torch.save({"model": model.state_dict(), "loss": best}, ckpt_dir / "best.pt")

    print(f"Saved best checkpoint to: {ckpt_dir / 'best.pt'}")

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

    print(f"Saved NHFAE-E1-Interact wavs to: {wav_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train NHFAE E1 Interact (MPICM) - Dynamic SNR Regime Switching (E2)")
    parser.add_argument("--clean-dir", default=None, help="Single dataset clean dir")
    parser.add_argument("--noisy-dir", default=None, help="Single dataset noisy dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-dce", type=float, default=1.0)
    parser.add_argument("--lambda-cfm", type=float, default=0.7)
    parser.add_argument("--lambda-cycle", type=float, default=0.2)
    parser.add_argument("--lambda-mrstft", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    # SNR 动态权重参数
    parser.add_argument("--snr-tau", type=float, default=5.0, help="SNR regime switch point (dB)")
    parser.add_argument("--snr-k", type=float, default=0.5, help="SNR sigmoid steepness")
    # 多 SNR 模式
    parser.add_argument("--multi-snr", action="store_true", help="Enable multi-SNR combined training")
    parser.add_argument("--snr-0-5-clean", help="snr_0_5 clean dir")
    parser.add_argument("--snr-0-5-noisy", help="snr_0_5 noisy dir")
    parser.add_argument("--snr-5-10-clean", help="snr_5_10 clean dir")
    parser.add_argument("--snr-5-10-noisy", help="snr_5_10 noisy dir")
    parser.add_argument("--snr-10-15-clean", help="snr_10_15 clean dir")
    parser.add_argument("--snr-10-15-noisy", help="snr_10_15 noisy dir")
    args = parser.parse_args()

    cfg = TrainCfg(
        clean_dir=Path(args.clean_dir) if args.clean_dir else None,
        noisy_dir=Path(args.noisy_dir) if args.noisy_dir else None,
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        lr=args.lr,
        lambda_dce=args.lambda_dce,
        lambda_cfm=args.lambda_cfm,
        lambda_cycle=args.lambda_cycle,
        lambda_mrstft=args.lambda_mrstft,
        device=args.device,
        snr_tau=args.snr_tau,
        snr_k=args.snr_k,
        multi_snr_mode=args.multi_snr,
    )
    
    # 多 SNR 模式配置
    if args.multi_snr:
        cfg.multi_snr_dirs = [
            ("snr_0_5", Path(args.snr_0_5_clean), Path(args.snr_0_5_noisy)),
            ("snr_5_10", Path(args.snr_5_10_clean), Path(args.snr_5_10_noisy)),
            ("snr_10_15", Path(args.snr_10_15_clean), Path(args.snr_10_15_noisy)),
        ]
        print(f"[Multi-SNR Mode] τ={cfg.snr_tau} dB, k={cfg.snr_k}")
    
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    train(cfg)
