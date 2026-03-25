import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_mono(path: Path, fs: int) -> np.ndarray:
    x, src_fs = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    if src_fs != fs:
        x = soxr.resample(x, src_fs, fs)
    return x.astype(np.float32, copy=False)


def stft(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    win = torch.hann_window(n_fft, device=x.device)
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        return_complex=True,
        center=True,
    )


def istft(X: torch.Tensor, length: int, n_fft: int, hop: int) -> torch.Tensor:
    win = torch.hann_window(n_fft, device=X.device)
    return torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=True,
        length=length,
    )


class FiLMGateFree(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cond = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.gamma = nn.Conv2d(16, 1, kernel_size=1)
        self.beta = nn.Conv2d(16, 1, kernel_size=1)
        self.out = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, mag_addse: torch.Tensor, mag_pguse: torch.Tensor) -> torch.Tensor:
        x = torch.stack([torch.log1p(mag_addse), torch.log1p(mag_pguse)], dim=0).unsqueeze(0)
        c = self.cond(x)
        h = x[:, :1]
        g = torch.sigmoid(self.gamma(c))
        b = self.beta(c)
        y = g * h + b
        y = self.out(y)
        y = F.softplus(y)
        return y.squeeze(0).squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser("Phase C free FiLM inference")
    parser.add_argument("--addse-dir", required=True)
    parser.add_argument("--pguse-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    addse_dir = Path(args.addse_dir)
    pguse_dir = Path(args.pguse_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FiLMGateFree().to(device)
    pack = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(pack["model"])
    model.eval()

    names = [p.name for p in sorted(addse_dir.glob("*.wav")) if (pguse_dir / p.name).exists()]
    if not names:
        raise ValueError("No aligned wav files found")

    with torch.no_grad():
        for n in names:
            xa = load_mono(addse_dir / n, args.fs)
            xp = load_mono(pguse_dir / n, args.fs)
            L = min(len(xa), len(xp))
            xa = torch.from_numpy(xa[:L]).to(device)
            xp = torch.from_numpy(xp[:L]).to(device)

            Sa = stft(xa, args.n_fft, args.hop)
            Sp = stft(xp, args.n_fft, args.hop)
            mag = model(torch.abs(Sa), torch.abs(Sp))
            phase = Sp / torch.clamp(torch.abs(Sp), min=1e-8)
            y = istft(mag * phase, length=L, n_fft=args.n_fft, hop=args.hop)
            y = y.detach().cpu().numpy().astype(np.float32, copy=False)

            peak = float(np.max(np.abs(y)) + 1e-8)
            if peak > 1.0:
                y = y / peak
            sf.write(out_dir / n, y, args.fs)

    print(f"Saved enhanced wavs: {out_dir}")


if __name__ == "__main__":
    main()
