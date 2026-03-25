import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch


def load_mono(path: Path, fs: int) -> np.ndarray:
    x, src_fs = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    if src_fs != fs:
        x = soxr.resample(x, src_fs, fs)
    return x.astype(np.float32, copy=False)


def stft(x: np.ndarray, n_fft: int, hop: int) -> torch.Tensor:
    tx = torch.from_numpy(x)
    win = torch.hann_window(n_fft)
    return torch.stft(
        tx,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        return_complex=True,
        center=True,
    )


def istft(X: torch.Tensor, length: int, n_fft: int, hop: int) -> np.ndarray:
    win = torch.hann_window(n_fft)
    y = torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=True,
        length=length,
    )
    return y.cpu().numpy().astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser("Parallel Lite V1: ADDSE magnitude + PGUSE phase")
    parser.add_argument("--addse-dir", required=True)
    parser.add_argument("--pguse-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--alpha-addse-mag",
        type=float,
        default=1.0,
        help="Magnitude blend weight: mag = alpha*mag_addse + (1-alpha)*mag_pguse",
    )
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=192)
    args = parser.parse_args()

    if not (0.0 <= args.alpha_addse_mag <= 1.0):
        raise ValueError("--alpha-addse-mag must be in [0,1]")

    addse_dir = Path(args.addse_dir)
    pguse_dir = Path(args.pguse_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(addse_dir.glob("*.wav"))
    if not files:
        raise ValueError(f"No wavs found in {addse_dir}")

    for ap in files:
        pp = pguse_dir / ap.name
        if not pp.exists():
            continue

        xa = load_mono(ap, args.fs)
        xp = load_mono(pp, args.fs)
        L = min(len(xa), len(xp))
        xa, xp = xa[:L], xp[:L]

        Sa = stft(xa, args.n_fft, args.hop)
        Sp = stft(xp, args.n_fft, args.hop)

        mag_a = torch.abs(Sa)
        mag_p = torch.abs(Sp)
        mag_f = args.alpha_addse_mag * mag_a + (1.0 - args.alpha_addse_mag) * mag_p
        phase_p = Sp / torch.clamp(torch.abs(Sp), min=1e-8)
        Sf = mag_f * phase_p

        y = istft(Sf, length=L, n_fft=args.n_fft, hop=args.hop)

        peak = float(np.max(np.abs(y)) + 1e-8)
        if peak > 1.0:
            y = y / peak

        sf.write(out_dir / ap.name, y, args.fs)

    print(f"Saved fused wavs to: {out_dir}")


if __name__ == "__main__":
    main()
