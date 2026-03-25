import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def load_mono(path: Path, fs: int) -> np.ndarray:
    x, src_fs = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    if src_fs != fs:
        x = soxr.resample(x, src_fs, fs)
    return x.astype(np.float32, copy=False)


def snr_db(clean: np.ndarray, noisy: np.ndarray, eps: float = 1e-8) -> float:
    noise = noisy - clean
    return float(10.0 * np.log10((np.sum(clean ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def build_sample(clean: np.ndarray, noisy: np.ndarray, target_snr_db: float, eps: float = 1e-8) -> np.ndarray:
    noise_raw = noisy - clean
    p_clean = float(np.sum(clean ** 2) + eps)
    p_noise = float(np.sum(noise_raw ** 2) + eps)
    target_noise_power = p_clean / (10.0 ** (target_snr_db / 10.0))
    scale = float(np.sqrt(target_noise_power / p_noise))
    out = clean + scale * noise_raw
    peak = float(np.max(np.abs(out)) + eps)
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32, copy=False)


def parse_targets(spec: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Each target range must be lo:hi")
        lo_s, hi_s = item.split(":", 1)
        lo = float(lo_s)
        hi = float(hi_s)
        if hi <= lo:
            raise ValueError(f"Invalid range {item}")
        out.append((lo, hi))
    if not out:
        raise ValueError("No target ranges parsed")
    return out


def main() -> None:
    parser = argparse.ArgumentParser("Create controlled-SNR noisy sets from paired clean/noisy")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--targets", default="0:5,5:10,10:15", help="Comma ranges, e.g. 0:5,5:10,10:15")
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    targets = parse_targets(args.targets)

    names = sorted([p.name for p in clean_dir.glob("*.wav")])
    if not names:
        names = sorted([p.name for p in clean_dir.glob("*.WAV")])
    if not names:
        raise ValueError(f"No clean wav files in {clean_dir}")

    summary_rows = []
    for lo, hi in targets:
        tag = f"snr_{int(lo)}_{int(hi)}"
        clean_out = out_root / tag / "clean"
        noisy_out = out_root / tag / "noisy"
        clean_out.mkdir(parents=True, exist_ok=True)
        noisy_out.mkdir(parents=True, exist_ok=True)

        got = []
        n_ok = 0
        for name in names:
            stem = Path(name).stem
            cp = clean_dir / name
            if not cp.exists():
                cp = clean_dir / f"{stem}.WAV"
            npth = noisy_dir / f"{stem}.wav"
            if not npth.exists():
                npth = noisy_dir / f"{stem}.WAV"
            if not cp.exists() or not npth.exists():
                continue

            c = load_mono(cp, args.fs)
            y = load_mono(npth, args.fs)
            L = min(len(c), len(y))
            c = c[:L]
            y = y[:L]

            target_snr = float(rng.uniform(lo, hi))
            new_noisy = build_sample(c, y, target_snr)
            got_snr = snr_db(c, new_noisy)
            got.append(got_snr)

            out_name = f"{stem}.wav"
            sf.write(clean_out / out_name, c, args.fs)
            sf.write(noisy_out / out_name, new_noisy, args.fs)
            n_ok += 1

        if got:
            arr = np.asarray(got, dtype=np.float64)
            summary_rows.append(
                {
                    "set": tag,
                    "count": n_ok,
                    "snr_min": round(float(arr.min()), 6),
                    "snr_mean": round(float(arr.mean()), 6),
                    "snr_max": round(float(arr.max()), 6),
                }
            )
        else:
            summary_rows.append({"set": tag, "count": 0, "snr_min": "NA", "snr_mean": "NA", "snr_max": "NA"})

    out_csv = out_root / "summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["set", "count", "snr_min", "snr_mean", "snr_max"])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Saved controlled SNR sets to: {out_root}")
    print(f"Saved summary: {out_csv}")


if __name__ == "__main__":
    main()
