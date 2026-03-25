#!/usr/bin/env python3
"""
Phase 10: Autonomous T-CNAC Training (No External Pretrain)

Goal:
- Train T-CNAC + MPICM style NHFAE-E2 fully from scratch.
- Remove dependency on external pretrained encoders.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase9_nhfae_e1 import PairDataset, istft, mrstft_loss, stft, wrap_to_pi
from phase9_tcnac_codec import NHFAE_E2_TCNAC


@dataclass
class TrainCfg:
    clean_dirs: list[Path]
    noisy_dirs: list[Path]
    out_dir: Path
    fs: int = 16000
    n_fft: int = 512
    hop: int = 192
    epochs: int = 10
    lr: float = 3e-4
    lambda_recon: float = 1.0
    lambda_mrstft: float = 0.2
    lambda_topo: float = 0.1
    lambda_tcnac: float = 0.2
    device: str = "cuda"


class MultiPairDataset:
    def __init__(self, clean_dirs: list[Path], noisy_dirs: list[Path], fs: int):
        if len(clean_dirs) != len(noisy_dirs):
            raise ValueError("clean_dirs and noisy_dirs must have equal lengths")

        self.datasets = [PairDataset(c, n, fs) for c, n in zip(clean_dirs, noisy_dirs)]
        self.index: list[tuple[int, int]] = []
        for d_idx, ds in enumerate(self.datasets):
            for i in range(len(ds)):
                self.index.append((d_idx, i))

    def __len__(self) -> int:
        return len(self.index)

    def get(self, i: int):
        d_idx, s_idx = self.index[i]
        name, clean, noisy = self.datasets[d_idx].get(s_idx)
        return f"d{d_idx}_{name}", clean, noisy


def infer_and_save(model: nn.Module, ds: MultiPairDataset, cfg: TrainCfg, wav_dir: Path, device: torch.device):
    model.eval()
    wav_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i in range(len(ds)):
            name, _, noisy = ds.get(i)
            noisy = noisy.to(device)
            S_noisy = stft(noisy, cfg.n_fft, cfg.hop).unsqueeze(0)
            out = model(S_noisy)
            y = istft(out["S_enhanced"].squeeze(0), length=noisy.numel(), n_fft=cfg.n_fft, hop=cfg.hop)
            y_np = y.detach().cpu().numpy().astype(np.float32, copy=False)
            peak = float(np.max(np.abs(y_np)) + 1e-8)
            if peak > 1.0:
                y_np = y_np / peak
            sf.write(wav_dir / name, y_np, cfg.fs)


def train(cfg: TrainCfg):
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")

    model = NHFAE_E2_TCNAC(d_model=96, tcnac_latent=256).to(device)

    ds = MultiPairDataset(cfg.clean_dirs, cfg.noisy_dirs, cfg.fs)
    if len(ds) < 2:
        raise ValueError("Need at least 2 paired samples")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr)

    ckpt_dir = cfg.out_dir / "ckpt"
    wav_dir = cfg.out_dir / "wav"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Autonomous-TCNAC] Samples: {len(ds)}")
    print(f"[Autonomous-TCNAC] Trainable params: {sum(p.numel() for p in trainable)}")

    best = float("inf")
    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []

        for i in range(len(ds)):
            _, clean, noisy = ds.get(i)
            clean = clean.to(device)
            noisy = noisy.to(device)

            S_clean = stft(clean, cfg.n_fft, cfg.hop).unsqueeze(0)
            S_noisy = stft(noisy, cfg.n_fft, cfg.hop).unsqueeze(0)

            out = model(S_noisy)
            S_enh = out["S_enhanced"]
            S_recon = out["S_recon"]

            # Reconstruction losses
            l_mag = F.l1_loss(torch.abs(S_enh), torch.abs(S_clean))
            dphi = wrap_to_pi(torch.angle(S_enh) - torch.angle(S_clean))
            l_phase = torch.mean(torch.abs(dphi))
            l_recon = l_mag + l_phase

            y = istft(S_enh.squeeze(0), length=clean.numel(), n_fft=cfg.n_fft, hop=cfg.hop)
            l_mr = mrstft_loss(y, clean)

            # T-CNAC codec quality losses
            l_tcnac = out["tcnac_magnitude_error"] + out["tcnac_phase_error"]

            # Topology supervision (encourage high topology score)
            l_topo = torch.mean(1.0 - out["topo_score"])

            loss = (
                cfg.lambda_recon * l_recon
                + cfg.lambda_mrstft * l_mr
                + cfg.lambda_tcnac * l_tcnac
                + cfg.lambda_topo * l_topo
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()
            losses.append(float(loss.item()))

        m = float(np.mean(losses))
        print(f"[ep {ep:03d}] loss={m:.6f}")
        if m < best:
            best = m
            torch.save({"model": model.state_dict(), "loss": best}, ckpt_dir / "best.pt")

    print(f"[Autonomous-TCNAC] Saved best checkpoint: {ckpt_dir / 'best.pt'}")

    pack = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(pack["model"])
    infer_and_save(model, ds, cfg, wav_dir, device)
    print(f"[Autonomous-TCNAC] Saved enhanced wavs: {wav_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Phase 10 Autonomous T-CNAC Training")
    parser.add_argument(
        "--clean-dirs",
        default="./outputs/phase6/controlled_snr_test31/snr_0_5/clean,./outputs/phase6/controlled_snr_test31/snr_5_10/clean,./outputs/phase6/controlled_snr_test31/snr_10_15/clean",
    )
    parser.add_argument(
        "--noisy-dirs",
        default="./outputs/phase6/controlled_snr_test31/snr_0_5/noisy,./outputs/phase6/controlled_snr_test31/snr_5_10/noisy,./outputs/phase6/controlled_snr_test31/snr_10_15/noisy",
    )
    parser.add_argument("--out-dir", default="./outputs/phase10/autonomous_tcnac")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-mrstft", type=float, default=0.2)
    parser.add_argument("--lambda-topo", type=float, default=0.1)
    parser.add_argument("--lambda-tcnac", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    clean_dirs = [Path(p.strip()) for p in args.clean_dirs.split(",") if p.strip()]
    noisy_dirs = [Path(p.strip()) for p in args.noisy_dirs.split(",") if p.strip()]

    cfg = TrainCfg(
        clean_dirs=clean_dirs,
        noisy_dirs=noisy_dirs,
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        lr=args.lr,
        lambda_recon=args.lambda_recon,
        lambda_mrstft=args.lambda_mrstft,
        lambda_topo=args.lambda_topo,
        lambda_tcnac=args.lambda_tcnac,
        device=args.device,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    train(cfg)
