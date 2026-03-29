import argparse
import csv
import re
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


def collect_wav_map(root: Path, role: str = "any") -> dict[str, Path]:
    out: dict[str, Path] = {}

    def canonical_stem(stem: str) -> str:
        # eval outputs are typically like:
        # edbase-local_000012_clean
        # edbase-local_000012_noisy
        # edbase-local_000012_addse-s-edbase-parallel60-a005
        # We normalize all to `edbase-local_000012` for robust matching.
        m = re.match(r"^(.+?_\d{6})(?:_.+)?$", stem)
        return m.group(1) if m else stem

    wav_files = [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() == ".wav"]
    has_role_suffix = any(p.stem.endswith("_clean") or p.stem.endswith("_noisy") for p in wav_files)

    for p in wav_files:
        stem = p.stem
        is_clean = stem.endswith("_clean")
        is_noisy = stem.endswith("_noisy")
        # If directory contains explicit role suffixes, filter by role.
        # Otherwise (plain stems like SI453.wav), treat all files as valid for that role.
        if has_role_suffix:
            if role == "clean" and not is_clean:
                continue
            if role == "noisy" and not is_noisy:
                continue
            if role == "pred" and (is_clean or is_noisy):
                continue
        out[canonical_stem(p.stem)] = p
    return out


def snr_db(clean: np.ndarray, noisy: np.ndarray, eps: float = 1e-8) -> float:
    noise = noisy - clean
    return float(10.0 * np.log10((np.sum(clean ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def _frame_signal(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape={x.shape}")
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames <= 0:
        n_frames = 1
        x = np.pad(x, (0, n_fft - len(x)))
    shape = (n_frames, n_fft)
    strides = (x.strides[0] * hop, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)


def _stft_mag_phase(x: np.ndarray, n_fft: int = 512, hop: int = 128) -> tuple[np.ndarray, np.ndarray]:
    frames = _frame_signal(x.astype(np.float64, copy=False), n_fft=n_fft, hop=hop)
    window = np.hanning(n_fft)[None, :]
    spec = np.fft.rfft(frames * window, axis=-1)
    mag = np.abs(spec)
    pha = np.angle(spec)
    return mag, pha


def phase_distance(clean: np.ndarray, test: np.ndarray, n_fft: int = 512, hop: int = 128, eps: float = 1e-8) -> float:
    clean_1d = clean.squeeze(0)
    test_1d = test.squeeze(0)
    n = min(clean_1d.shape[-1], test_1d.shape[-1])
    clean_mag, clean_pha = _stft_mag_phase(clean_1d[:n], n_fft=n_fft, hop=hop)
    test_mag, test_pha = _stft_mag_phase(test_1d[:n], n_fft=n_fft, hop=hop)
    m = min(clean_pha.shape[0], test_pha.shape[0])
    clean_mag, clean_pha = clean_mag[:m], clean_pha[:m]
    test_mag, test_pha = test_mag[:m], test_pha[:m]
    dphi = np.angle(np.exp(1j * (clean_pha - test_pha)))
    weight = clean_mag / (clean_mag.sum() + eps)
    return float(np.sum(np.abs(dphi) * weight))


def log_spectral_distance(
    clean: np.ndarray,
    test: np.ndarray,
    n_fft: int = 512,
    hop: int = 128,
    eps: float = 1e-8,
) -> float:
    clean_1d = clean.squeeze(0)
    test_1d = test.squeeze(0)
    n = min(clean_1d.shape[-1], test_1d.shape[-1])
    clean_mag, _ = _stft_mag_phase(clean_1d[:n], n_fft=n_fft, hop=hop)
    test_mag, _ = _stft_mag_phase(test_1d[:n], n_fft=n_fft, hop=hop)
    m = min(clean_mag.shape[0], test_mag.shape[0])
    clean_mag, test_mag = clean_mag[:m], test_mag[:m]
    clean_log = np.log(clean_mag + eps)
    test_log = np.log(test_mag + eps)
    lsd_frame = np.sqrt(np.mean((clean_log - test_log) ** 2, axis=-1))
    return float(np.mean(lsd_frame))


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

    clean_map = collect_wav_map(clean_dir, role="clean")
    noisy_map = collect_wav_map(noisy_dir, role="noisy")
    if not clean_map:
        raise ValueError(f"No clean files in {clean_dir}")
    if not noisy_map:
        raise ValueError(f"No noisy files in {noisy_dir}")

    overall_rows = []
    bucket_rows = []

    for method_name, pred_dir in methods:
        pred_map = collect_wav_map(pred_dir, role="pred")
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
            pd = phase_distance(clean, pred)
            pd0 = phase_distance(clean, noisy)
            lsd = log_spectral_distance(clean, pred)
            lsd0 = log_spectral_distance(clean, noisy)
            vals.append((p, e, d, pd, lsd, p0, e0, d0, pd0, lsd0))

            b = pick_bucket(snr_db(clean, noisy), buckets)
            if b:
                bucket_map[b].append((p, e, d, pd, lsd, p0, e0, d0, pd0, lsd0))

        if not vals:
            print(f"[warn] {method_name}: matched 0 files")
            continue

        print(f"[info] {method_name}: matched {matched} files")

        arr = np.asarray(vals, dtype=np.float64)

        def nmean(a: np.ndarray) -> float:
            return float(np.nanmean(a))

        overall_rows.append(
            {
                "method": method_name,
                "count": int(arr.shape[0]),
                "pesq": round(nmean(arr[:, 0]), 6),
                "estoi": round(nmean(arr[:, 1]), 6),
                "sdr": round(nmean(arr[:, 2]), 6),
                "pd": round(nmean(arr[:, 3]), 6),
                "lsd": round(nmean(arr[:, 4]), 6),
                "delta_pesq_vs_noisy": round(nmean(arr[:, 0] - arr[:, 5]), 6),
                "delta_estoi_vs_noisy": round(nmean(arr[:, 1] - arr[:, 6]), 6),
                "delta_sdr_vs_noisy": round(nmean(arr[:, 2] - arr[:, 7]), 6),
                "delta_pd_vs_noisy": round(nmean(arr[:, 3] - arr[:, 8]), 6),
                "delta_lsd_vs_noisy": round(nmean(arr[:, 4] - arr[:, 9]), 6),
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
                        "pd": "NA",
                        "lsd": "NA",
                        "delta_pesq": "NA",
                        "delta_estoi": "NA",
                        "delta_sdr": "NA",
                        "delta_pd": "NA",
                        "delta_lsd": "NA",
                    }
                )
                continue

            a = np.asarray(sub, dtype=np.float64)
            bucket_rows.append(
                {
                    "bucket": key,
                    "method": method_name,
                    "count": int(a.shape[0]),
                    "pesq": f"{np.nanmean(a[:, 0]):.4f}",
                    "estoi": f"{np.nanmean(a[:, 1]):.4f}",
                    "sdr": f"{np.nanmean(a[:, 2]):.4f}",
                    "pd": f"{np.nanmean(a[:, 3]):.4f}",
                    "lsd": f"{np.nanmean(a[:, 4]):.4f}",
                    "delta_pesq": f"{np.nanmean(a[:, 0] - a[:, 5]):.4f}",
                    "delta_estoi": f"{np.nanmean(a[:, 1] - a[:, 6]):.4f}",
                    "delta_sdr": f"{np.nanmean(a[:, 2] - a[:, 7]):.4f}",
                    "delta_pd": f"{np.nanmean(a[:, 3] - a[:, 8]):.4f}",
                    "delta_lsd": f"{np.nanmean(a[:, 4] - a[:, 9]):.4f}",
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
                "pd",
                "lsd",
                "delta_pesq_vs_noisy",
                "delta_estoi_vs_noisy",
                "delta_sdr_vs_noisy",
                "delta_pd_vs_noisy",
                "delta_lsd_vs_noisy",
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
                "pd",
                "lsd",
                "delta_pesq",
                "delta_estoi",
                "delta_sdr",
                "delta_pd",
                "delta_lsd",
            ],
        )
        w.writeheader()
        w.writerows(bucket_rows)

    print(f"Saved overall: {out_overall}")
    print(f"Saved bucket: {out_bucket}")


if __name__ == "__main__":
    main()
