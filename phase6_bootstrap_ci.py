import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from addse.metrics import PESQMetric, SDRMetric, STOIMetric


def parse_method(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise ValueError(f"Invalid --method format: {item}")
    name, path = item.split("=", 1)
    return name.strip(), Path(path.strip())


def load_mono(path: Path, fs: int) -> np.ndarray:
    x, src_fs = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, :1]
    if src_fs != fs:
        x = soxr.resample(x, src_fs, fs)
    return x.T


def bootstrap_ci(vals: np.ndarray, n_boot: int, seed: int, alpha: float = 0.95) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = vals.shape[0]
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(vals[idx])))
    means_arr = np.asarray(means, dtype=np.float64)
    lo = float(np.quantile(means_arr, (1.0 - alpha) / 2.0))
    hi = float(np.quantile(means_arr, 1.0 - (1.0 - alpha) / 2.0))
    return float(np.mean(vals)), lo, hi


def main() -> None:
    parser = argparse.ArgumentParser("Bootstrap CI for PESQ/ESTOI/SDR on paired wav methods")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--method", action="append", required=True, help="name=wav_dir")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fs", type=int, default=16000)
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir)
    methods = [parse_method(x) for x in args.method]

    pesq = PESQMetric(args.fs)
    estoi = STOIMetric(args.fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    clean_files = sorted(clean_dir.glob("*.wav"))
    if not clean_files:
        raise ValueError(f"No wav found in {clean_dir}")

    rows = []
    for method_name, pred_dir in methods:
        vals_p = []
        vals_e = []
        vals_d = []
        vals_dp = []
        vals_de = []
        vals_dd = []

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
            clean = clean[:, :n]
            noisy = noisy[:, :n]
            pred = pred[:, :n]

            p = float(pesq(pred, clean))
            e = float(estoi(pred, clean))
            d = float(sdr(pred, clean))
            p0 = float(pesq(noisy, clean))
            e0 = float(estoi(noisy, clean))
            d0 = float(sdr(noisy, clean))

            vals_p.append(p)
            vals_e.append(e)
            vals_d.append(d)
            vals_dp.append(p - p0)
            vals_de.append(e - e0)
            vals_dd.append(d - d0)

        if not vals_p:
            continue

        p_m, p_l, p_h = bootstrap_ci(np.asarray(vals_p, dtype=np.float64), args.n_boot, args.seed)
        e_m, e_l, e_h = bootstrap_ci(np.asarray(vals_e, dtype=np.float64), args.n_boot, args.seed + 1)
        d_m, d_l, d_h = bootstrap_ci(np.asarray(vals_d, dtype=np.float64), args.n_boot, args.seed + 2)
        dp_m, dp_l, dp_h = bootstrap_ci(np.asarray(vals_dp, dtype=np.float64), args.n_boot, args.seed + 3)
        de_m, de_l, de_h = bootstrap_ci(np.asarray(vals_de, dtype=np.float64), args.n_boot, args.seed + 4)
        dd_m, dd_l, dd_h = bootstrap_ci(np.asarray(vals_dd, dtype=np.float64), args.n_boot, args.seed + 5)

        rows.append(
            {
                "method": method_name,
                "count": len(vals_p),
                "pesq_mean": f"{p_m:.6f}",
                "pesq_ci95_lo": f"{p_l:.6f}",
                "pesq_ci95_hi": f"{p_h:.6f}",
                "estoi_mean": f"{e_m:.6f}",
                "estoi_ci95_lo": f"{e_l:.6f}",
                "estoi_ci95_hi": f"{e_h:.6f}",
                "sdr_mean": f"{d_m:.6f}",
                "sdr_ci95_lo": f"{d_l:.6f}",
                "sdr_ci95_hi": f"{d_h:.6f}",
                "delta_pesq_mean": f"{dp_m:.6f}",
                "delta_pesq_ci95_lo": f"{dp_l:.6f}",
                "delta_pesq_ci95_hi": f"{dp_h:.6f}",
                "delta_estoi_mean": f"{de_m:.6f}",
                "delta_estoi_ci95_lo": f"{de_l:.6f}",
                "delta_estoi_ci95_hi": f"{de_h:.6f}",
                "delta_sdr_mean": f"{dd_m:.6f}",
                "delta_sdr_ci95_lo": f"{dd_l:.6f}",
                "delta_sdr_ci95_hi": f"{dd_h:.6f}",
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
                "pesq_mean",
                "pesq_ci95_lo",
                "pesq_ci95_hi",
                "estoi_mean",
                "estoi_ci95_lo",
                "estoi_ci95_hi",
                "sdr_mean",
                "sdr_ci95_lo",
                "sdr_ci95_hi",
                "delta_pesq_mean",
                "delta_pesq_ci95_lo",
                "delta_pesq_ci95_hi",
                "delta_estoi_mean",
                "delta_estoi_ci95_lo",
                "delta_estoi_ci95_hi",
                "delta_sdr_mean",
                "delta_sdr_ci95_lo",
                "delta_sdr_ci95_hi",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Saved bootstrap CI table: {out_csv}")


if __name__ == "__main__":
    main()
