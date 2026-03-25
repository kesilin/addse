import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from addse.metrics import PESQMetric, SDRMetric, STOIMetric


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
    root = Path("outputs/final60_snr_m5_10")
    clean_dir = root / "fixed60" / "clean"
    noisy_dir = root / "fixed60" / "noisy"
    out_dir = root / "final_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    method_seed_dirs = {
        "addse_only": root / "addse_only",
        "pguse_only": root / "pguse_only",
        "serial": root / "serial",
    }
    seeds = [3407, 2026, 777]
    fs = 16000

    pesq = PESQMetric(fs)
    estoi = STOIMetric(fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    clean_map = {p.stem: p for p in sorted(clean_dir.glob("*.wav"))}
    noisy_map = {p.stem: p for p in sorted(noisy_dir.glob("*.wav"))}
    stems = sorted(set(clean_map.keys()) & set(noisy_map.keys()))
    if not stems:
        raise RuntimeError("No matched clean/noisy wav files found in fixed60 set")

    noisy_metric = {}
    for stem in stems:
        clean = load_mono(clean_map[stem], fs)
        noisy = load_mono(noisy_map[stem], fs)
        n = min(clean.shape[-1], noisy.shape[-1])
        clean = clean[:, :n]
        noisy = noisy[:, :n]
        noisy_metric[stem] = (
            float(pesq(noisy, clean)),
            float(estoi(noisy, clean)),
            float(sdr(noisy, clean)),
        )

    per_rows = []
    for method, base_dir in method_seed_dirs.items():
        for seed in seeds:
            pred_dir = base_dir / f"seed{seed}_eval_wav"
            if not pred_dir.exists():
                raise RuntimeError(f"Missing prediction dir: {pred_dir}")

            matched = 0
            for stem in stems:
                pred_path = pred_dir / f"{stem}.wav"
                if not pred_path.exists():
                    continue
                matched += 1

                clean = load_mono(clean_map[stem], fs)
                noisy = load_mono(noisy_map[stem], fs)
                pred = load_mono(pred_path, fs)
                n = min(clean.shape[-1], noisy.shape[-1], pred.shape[-1])
                clean = clean[:, :n]
                pred = pred[:, :n]

                p = float(pesq(pred, clean))
                e = float(estoi(pred, clean))
                d = float(sdr(pred, clean))
                p0, e0, d0 = noisy_metric[stem]
                per_rows.append(
                    {
                        "seed": str(seed),
                        "method": method,
                        "name": stem,
                        "pesq": p,
                        "estoi": e,
                        "sdr": d,
                        "dp": p - p0,
                        "de": e - e0,
                        "dd": d - d0,
                    }
                )

            if matched == 0:
                raise RuntimeError(f"No matched wavs for {method} seed {seed}")

    per_csv = out_dir / "per_sample_seed_metrics.csv"
    with per_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "method", "name", "pesq", "estoi", "sdr", "dp", "de", "dd"])
        w.writeheader()
        w.writerows(per_rows)

    by_method_seed = {}
    for r in per_rows:
        key = (r["method"], r["seed"])
        by_method_seed.setdefault(key, {"pesq": [], "estoi": [], "sdr": [], "dp": [], "de": [], "dd": []})
        for k in ["pesq", "estoi", "sdr", "dp", "de", "dd"]:
            by_method_seed[key][k].append(float(r[k]))

    summary_rows = []
    method_order = ["addse_only", "pguse_only", "serial"]
    for method in method_order:
        seed_stats = []
        for seed in seeds:
            vals = by_method_seed.get((method, str(seed)))
            if not vals:
                continue
            seed_stats.append({k: float(np.mean(v)) for k, v in vals.items()})

        if not seed_stats:
            continue

        row = {"method": method, "n_seeds": len(seed_stats)}
        for k in ["pesq", "estoi", "sdr", "dp", "de", "dd"]:
            arr = np.asarray([x[k] for x in seed_stats], dtype=np.float64)
            row[f"{k}_mean"] = float(arr.mean())
            row[f"{k}_std"] = float(arr.std(ddof=0))
        summary_rows.append(row)

    summary_csv = out_dir / "summary_mean_std.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "n_seeds",
                "pesq_mean",
                "pesq_std",
                "estoi_mean",
                "estoi_std",
                "sdr_mean",
                "sdr_std",
                "dp_mean",
                "dp_std",
                "de_mean",
                "de_std",
                "dd_mean",
                "dd_std",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    by_seed_name = {(r["seed"], r["name"], r["method"]): r for r in per_rows}
    pair_metrics = {k: [] for k in ["pesq", "estoi", "sdr", "dp", "de", "dd"]}
    for seed in [str(s) for s in seeds]:
        for stem in stems:
            a = by_seed_name.get((seed, stem, "serial"))
            b = by_seed_name.get((seed, stem, "pguse_only"))
            if not a or not b:
                continue
            for k in pair_metrics:
                pair_metrics[k].append(float(a[k]) - float(b[k]))

    boot_rows = []
    for i, k in enumerate(["pesq", "estoi", "sdr", "dp", "de", "dd"]):
        vals = np.asarray(pair_metrics[k], dtype=np.float64)
        if vals.size == 0:
            raise RuntimeError(f"No paired values for bootstrap metric {k}")
        mean_delta, lo, hi = bootstrap_ci(vals, n_boot=10000, seed=42 + i)
        boot_rows.append({"metric": k, "mean_delta": mean_delta, "ci95_low": lo, "ci95_high": hi})

    boot_csv = out_dir / "bootstrap_serial_minus_pguse.csv"
    with boot_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "mean_delta", "ci95_low", "ci95_high"])
        w.writeheader()
        w.writerows(boot_rows)

    print(f"Saved: {per_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {boot_csv}")


if __name__ == "__main__":
    main()
