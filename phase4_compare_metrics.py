import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from addse.metrics import PESQMetric, SDRMetric, STOIMetric


BUCKETS = [(-5.0, 0.0), (0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]


def parse_buckets(spec: str) -> list[tuple[float, float]]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(vals) < 2:
        raise ValueError("--buckets requires at least two comma-separated numbers")
    out: list[tuple[float, float]] = []
    for i in range(len(vals) - 1):
        lo, hi = vals[i], vals[i + 1]
        if hi <= lo:
            raise ValueError("Bucket edges must be strictly increasing")
        out.append((lo, hi))
    return out


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


def collect_wav_map(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".wav":
            continue
        out[p.stem] = p
    return out


def snr_db(clean: np.ndarray, noisy: np.ndarray, eps: float = 1e-8) -> float:
    noise = noisy - clean
    return float(10.0 * np.log10((np.sum(clean ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def pick_bucket(snr: float, buckets: list[tuple[float, float]]) -> str | None:
    for lo, hi in buckets:
        if lo <= snr < hi:
            return f"[{int(lo)},{int(hi)}]"
    return None


def main() -> None:
    parser = argparse.ArgumentParser("Compare methods with PESQ/ESTOI/SDR overall and by SNR bucket")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--method", action="append", required=True, help="name=wav_dir")
    parser.add_argument("--out-overall-csv", required=True)
    parser.add_argument("--out-bucket-csv", required=True)
    parser.add_argument(
        "--buckets",
        default="-5,0,5,10,15",
        help="Comma-separated bucket edges, e.g. -20,-5,0,5,10,15",
    )
    parser.add_argument("--fs", type=int, default=16000)
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir)
    methods = [parse_method(x) for x in args.method]
    buckets = parse_buckets(args.buckets)

    pesq = PESQMetric(args.fs)
    estoi = STOIMetric(args.fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    clean_map = collect_wav_map(clean_dir)
    noisy_map = collect_wav_map(noisy_dir)
    if not clean_map:
        raise ValueError(f"No clean files in {clean_dir}")
    if not noisy_map:
        raise ValueError(f"No noisy files in {noisy_dir}")

    overall_rows = []
    bucket_rows = []

    for method_name, pred_dir in methods:
        pred_map = collect_wav_map(pred_dir)
        if not pred_map:
            print(f"[warn] {method_name}: no wav files in {pred_dir}")
            continue

        vals = []
        bucket_map = {f"[{int(lo)},{int(hi)}]": [] for lo, hi in buckets}
        matched = 0

        for stem, clean_path in clean_map.items():
            noisy_path = noisy_map.get(stem)
            pred_path = pred_map.get(stem)
            if noisy_path is None or pred_path is None:
                continue
            matched += 1

            clean = load_mono(clean_path, args.fs)
            noisy = load_mono(noisy_path, args.fs)
            pred = load_mono(pred_path, args.fs)

            n = min(clean.shape[-1], noisy.shape[-1], pred.shape[-1])
            clean, noisy, pred = clean[:, :n], noisy[:, :n], pred[:, :n]

            p = float(pesq(pred, clean))
            e = float(estoi(pred, clean))
            d = float(sdr(pred, clean))
            p0 = float(pesq(noisy, clean))
            e0 = float(estoi(noisy, clean))
            d0 = float(sdr(noisy, clean))
            vals.append((p, e, d, p0, e0, d0))

            b = pick_bucket(snr_db(clean, noisy), buckets)
            if b:
                bucket_map[b].append((p, e, d, p0, e0, d0))

        if not vals:
            print(f"[warn] {method_name}: matched 0 files")
            continue

        print(f"[info] {method_name}: matched {matched} files")

        arr = np.asarray(vals, dtype=np.float64)
        overall_rows.append(
            {
                "method": method_name,
                "count": int(arr.shape[0]),
                "pesq": round(float(arr[:, 0].mean()), 6),
                "estoi": round(float(arr[:, 1].mean()), 6),
                "sdr": round(float(arr[:, 2].mean()), 6),
                "delta_pesq_vs_noisy": round(float((arr[:, 0] - arr[:, 3]).mean()), 6),
                "delta_estoi_vs_noisy": round(float((arr[:, 1] - arr[:, 4]).mean()), 6),
                "delta_sdr_vs_noisy": round(float((arr[:, 2] - arr[:, 5]).mean()), 6),
            }
        )

        for lo, hi in buckets:
            key = f"[{int(lo)},{int(hi)}]"
            sub = bucket_map[key]
            if not sub:
                bucket_rows.append(
                    {
                        "bucket": key,
                        "method": method_name,
                        "count": 0,
                        "pesq": "NA",
                        "estoi": "NA",
                        "sdr": "NA",
                        "delta_pesq": "NA",
                        "delta_estoi": "NA",
                        "delta_sdr": "NA",
                    }
                )
                continue

            a = np.asarray(sub, dtype=np.float64)
            bucket_rows.append(
                {
                    "bucket": key,
                    "method": method_name,
                    "count": int(a.shape[0]),
                    "pesq": f"{a[:, 0].mean():.4f}",
                    "estoi": f"{a[:, 1].mean():.4f}",
                    "sdr": f"{a[:, 2].mean():.4f}",
                    "delta_pesq": f"{(a[:, 0] - a[:, 3]).mean():.4f}",
                    "delta_estoi": f"{(a[:, 1] - a[:, 4]).mean():.4f}",
                    "delta_sdr": f"{(a[:, 2] - a[:, 5]).mean():.4f}",
                }
            )

    out_overall = Path(args.out_overall_csv)
    out_bucket = Path(args.out_bucket_csv)
    out_overall.parent.mkdir(parents=True, exist_ok=True)
    out_bucket.parent.mkdir(parents=True, exist_ok=True)

    with out_overall.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "count",
                "pesq",
                "estoi",
                "sdr",
                "delta_pesq_vs_noisy",
                "delta_estoi_vs_noisy",
                "delta_sdr_vs_noisy",
            ],
        )
        w.writeheader()
        w.writerows(overall_rows)

    with out_bucket.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "bucket",
                "method",
                "count",
                "pesq",
                "estoi",
                "sdr",
                "delta_pesq",
                "delta_estoi",
                "delta_sdr",
            ],
        )
        w.writeheader()
        w.writerows(bucket_rows)

    print(f"Saved overall: {out_overall}")
    print(f"Saved bucket: {out_bucket}")


if __name__ == "__main__":
    main()
