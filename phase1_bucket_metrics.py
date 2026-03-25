import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from addse.metrics import PESQMetric, SDRMetric, STOIMetric


BUCKETS = [(-5.0, 0.0), (0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]


def load_mono(path: Path, fs: int) -> np.ndarray:
    x, src_fs = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[1] > 1:
        x = x[:, :1]
    if src_fs != fs:
        x = soxr.resample(x, src_fs, fs)
    return x.T


def snr_db(clean: np.ndarray, noisy: np.ndarray, eps: float = 1e-8) -> float:
    noise = noisy - clean
    p_clean = float(np.sum(clean**2))
    p_noise = float(np.sum(noise**2))
    return 10.0 * np.log10((p_clean + eps) / (p_noise + eps))


def pick_bucket(snr: float) -> str | None:
    for lo, hi in BUCKETS:
        if lo <= snr < hi:
            return f"[{int(lo)},{int(hi)}]"
    return None


def main() -> None:
    parser = argparse.ArgumentParser("Compute bucketed PESQ/ESTOI/SDR for phase-1 outputs")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--pred-dir", required=True, help="Predicted wav dir (e.g., joint N5)")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--method-name", default="joint")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir)
    pred_dir = Path(args.pred_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pesq = PESQMetric(args.fs)
    estoi = STOIMetric(args.fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    bucket_rows: dict[str, list[dict[str, float]]] = {f"[{int(lo)},{int(hi)}]": [] for lo, hi in BUCKETS}

    clean_files = sorted(clean_dir.glob("*.wav"))
    if not clean_files:
        raise ValueError(f"No clean wav files in {clean_dir}")

    for clean_path in clean_files:
        name = clean_path.name
        noisy_path = noisy_dir / name
        pred_path = pred_dir / name
        if not noisy_path.exists() or not pred_path.exists():
            continue

        clean = load_mono(clean_path, args.fs)
        noisy = load_mono(noisy_path, args.fs)
        pred = load_mono(pred_path, args.fs)

        n = min(clean.shape[-1], noisy.shape[-1], pred.shape[-1])
        clean, noisy, pred = clean[:, :n], noisy[:, :n], pred[:, :n]

        snr = snr_db(clean, noisy)
        b = pick_bucket(snr)
        if b is None:
            continue

        row = {
            "snr": snr,
            "pesq_pred": float(pesq(pred, clean)),
            "pesq_noisy": float(pesq(noisy, clean)),
            "estoi_pred": float(estoi(pred, clean)),
            "estoi_noisy": float(estoi(noisy, clean)),
            "sdr_pred": float(sdr(pred, clean)),
            "sdr_noisy": float(sdr(noisy, clean)),
        }
        bucket_rows[b].append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "bucket",
            "count",
            "method",
            "pesq",
            "pesq_noisy",
            "delta_pesq",
            "estoi",
            "estoi_noisy",
            "delta_estoi",
            "sdr",
            "sdr_noisy",
            "delta_sdr",
        ])

        for bucket in [f"[{int(lo)},{int(hi)}]" for lo, hi in BUCKETS]:
            rows = bucket_rows[bucket]
            if not rows:
                w.writerow([bucket, 0, args.method_name, "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"])
                continue

            pesq_pred = np.nanmean([r["pesq_pred"] for r in rows])
            pesq_noisy = np.nanmean([r["pesq_noisy"] for r in rows])
            estoi_pred = np.nanmean([r["estoi_pred"] for r in rows])
            estoi_noisy = np.nanmean([r["estoi_noisy"] for r in rows])
            sdr_pred = np.nanmean([r["sdr_pred"] for r in rows])
            sdr_noisy = np.nanmean([r["sdr_noisy"] for r in rows])

            w.writerow([
                bucket,
                len(rows),
                args.method_name,
                f"{pesq_pred:.4f}",
                f"{pesq_noisy:.4f}",
                f"{(pesq_pred - pesq_noisy):.4f}",
                f"{estoi_pred:.4f}",
                f"{estoi_noisy:.4f}",
                f"{(estoi_pred - estoi_noisy):.4f}",
                f"{sdr_pred:.4f}",
                f"{sdr_noisy:.4f}",
                f"{(sdr_pred - sdr_noisy):.4f}",
            ])

    print(f"Saved bucket metrics: {out_csv}")


if __name__ == "__main__":
    main()
