import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch


def parse_method(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise ValueError(f"Invalid --method format: {item}")
    name, path = item.split("=", 1)
    return name.strip(), Path(path.strip())


def load_mono(path: Path, fs: int) -> np.ndarray:
    x, src_fs = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    if src_fs != fs:
        x = soxr.resample(x, src_fs, fs)
    return x.astype(np.float32, copy=False)


def band_pass_by_stft(x: np.ndarray, fs: int, f_lo: float, f_hi: float, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    tx = torch.from_numpy(x)
    win = torch.hann_window(n_fft)
    X = torch.stft(tx, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True, center=True)
    freqs = torch.linspace(0.0, fs / 2.0, steps=X.shape[0])
    mask = ((freqs >= f_lo) & (freqs < f_hi)).float().unsqueeze(1)
    Y = X * mask
    y = torch.istft(Y, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, center=True, length=len(x))
    return y.numpy().astype(np.float32, copy=False)


def sdr(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    n = pred - target
    return float(10.0 * np.log10((np.sum(target ** 2) + eps) / (np.sum(n ** 2) + eps)))


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate full-band and high-band SDR")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--method", action="append", required=True, help="name=wav_dir")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--high-lo", type=float, default=4000.0)
    parser.add_argument("--high-hi", type=float, default=8000.0)
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir)
    methods = [parse_method(x) for x in args.method]

    clean_files = sorted(clean_dir.glob("*.wav"))
    if not clean_files:
        clean_files = sorted(clean_dir.glob("*.WAV"))
    if not clean_files:
        raise ValueError(f"No clean files in {clean_dir}")

    rows = []
    for method_name, pred_dir in methods:
        vals_full = []
        vals_high = []
        vals_dfull = []
        vals_dhigh = []

        for cp in clean_files:
            name = f"{cp.stem}.wav"
            npth = noisy_dir / name
            ppth = pred_dir / name
            if not npth.exists() or not ppth.exists():
                continue

            c = load_mono(cp, args.fs)
            n = load_mono(npth, args.fs)
            p = load_mono(ppth, args.fs)
            L = min(len(c), len(n), len(p))
            c = c[:L]
            n = n[:L]
            p = p[:L]

            s_full = sdr(p, c)
            s_full_n = sdr(n, c)

            c_h = band_pass_by_stft(c, args.fs, args.high_lo, args.high_hi)
            n_h = band_pass_by_stft(n, args.fs, args.high_lo, args.high_hi)
            p_h = band_pass_by_stft(p, args.fs, args.high_lo, args.high_hi)
            s_high = sdr(p_h, c_h)
            s_high_n = sdr(n_h, c_h)

            vals_full.append(s_full)
            vals_high.append(s_high)
            vals_dfull.append(s_full - s_full_n)
            vals_dhigh.append(s_high - s_high_n)

        if not vals_full:
            continue

        rows.append(
            {
                "method": method_name,
                "count": len(vals_full),
                "sdr_full": f"{np.mean(vals_full):.6f}",
                "delta_sdr_full_vs_noisy": f"{np.mean(vals_dfull):.6f}",
                "sdr_high_4k_8k": f"{np.mean(vals_high):.6f}",
                "delta_sdr_high_4k_8k_vs_noisy": f"{np.mean(vals_dhigh):.6f}",
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "count",
                "sdr_full",
                "delta_sdr_full_vs_noisy",
                "sdr_high_4k_8k",
                "delta_sdr_high_4k_8k_vs_noisy",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Saved high-band SDR table: {out_csv}")


if __name__ == "__main__":
    main()
