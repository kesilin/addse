import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import soxr

from addse.metrics import SDRMetric


def load_mono(path: Path, fs: int) -> np.ndarray:
    x, src_fs = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, :1]
    if src_fs != fs:
        x = soxr.resample(x, src_fs, fs)
    return x.T


def snr_db(clean: np.ndarray, noisy: np.ndarray, eps: float = 1e-8) -> float:
    noise = noisy - clean
    return float(10.0 * np.log10((np.sum(clean ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def main() -> None:
    parser = argparse.ArgumentParser("Plot SNR vs delta SDR")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--fs", type=int, default=16000)
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir)
    pred_dir = Path(args.pred_dir)

    sdr = SDRMetric(scale_invariant=False)

    clean_map = {p.stem: p for p in sorted(clean_dir.glob("*.wav"))}
    noisy_map = {p.stem: p for p in sorted(noisy_dir.glob("*.wav"))}
    pred_map = {p.stem: p for p in sorted(pred_dir.glob("*.wav"))}

    rows = []
    for stem, cp in clean_map.items():
        npth = noisy_map.get(stem)
        ppth = pred_map.get(stem)
        if npth is None or ppth is None:
            continue

        clean = load_mono(cp, args.fs)
        noisy = load_mono(npth, args.fs)
        pred = load_mono(ppth, args.fs)

        n = min(clean.shape[-1], noisy.shape[-1], pred.shape[-1])
        clean = clean[:, :n]
        noisy = noisy[:, :n]
        pred = pred[:, :n]

        sdr_pred = float(sdr(pred, clean))
        sdr_noisy = float(sdr(noisy, clean))
        rows.append(
            {
                "name": stem,
                "snr_in": snr_db(clean, noisy),
                "sdr_pred": sdr_pred,
                "sdr_noisy": sdr_noisy,
                "delta_sdr": sdr_pred - sdr_noisy,
            }
        )

    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "snr_in", "sdr_pred", "sdr_noisy", "delta_sdr"])
        w.writeheader()
        w.writerows(rows)

    x = np.asarray([r["snr_in"] for r in rows], dtype=np.float64)
    y = np.asarray([r["delta_sdr"] for r in rows], dtype=np.float64)

    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(x, y, s=16, alpha=0.75)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    plt.xlabel("SNR_in (dB)")
    plt.ylabel("delta_SDR (pred - noisy)")
    plt.title(f"SNR vs delta_SDR: {args.method_name}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

    print(f"Saved csv: {out_csv}")
    print(f"Saved png: {out_png}")


if __name__ == "__main__":
    main()
