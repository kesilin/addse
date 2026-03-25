#!/usr/bin/env python3
"""
Phase 10: Full-Scene Hero Plot

- Supports large-scale evaluation (e.g., VoiceBank+DEMAND 824).
- Works with recursive clean/noisy roots and one or more enhanced roots.
"""

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft


def compute_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    if len(ref) == 0 or len(est) == 0:
        return float("nan")
    L = min(len(ref), len(est))
    ref = ref[:L]
    est = est[:L]
    ref_pow = np.sum(ref * ref) + 1e-8
    err_pow = np.sum((ref - est) ** 2) + 1e-8
    return float(10.0 * np.log10(ref_pow / err_pow))


def estimate_snr(clean: np.ndarray, noisy: np.ndarray, fs: int = 16000) -> float:
    n_fft = 512
    hop = 192
    _, _, Xc = stft(clean, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, Xn = stft(noisy, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    mag_c = np.abs(Xc)
    mag_n = np.abs(Xn)
    T = min(mag_c.shape[1], mag_n.shape[1])
    mag_c = mag_c[:, :T]
    mag_n = mag_n[:, :T]
    noise = mag_n - mag_c
    snr = 10.0 * np.log10((np.sum(mag_c**2) + 1e-8) / (np.sum(noise**2) + 1e-8))
    return float(np.clip(snr, -20, 30))


def normalize_bucket(text: str) -> str | None:
    m = re.search(r"snr[_]?(\d+)[_](\d+)", text)
    if not m:
        return None
    return f"snr_{m.group(1)}_{m.group(2)}"


def extract_bucket_from_path(p: Path) -> str | None:
    for part in p.parts:
        b = normalize_bucket(part)
        if b:
            return b
    return normalize_bucket(str(p))


def index_wavs_by_bucket(root: Path, required_marker: str | None = None) -> dict[str, dict[str, Path]]:
    idx: dict[str, dict[str, Path]] = {}
    for ext in ("*.wav", "*.WAV"):
        for p in root.rglob(ext):
            if required_marker:
                parts_lower = [x.lower() for x in p.parts]
                if required_marker.lower() not in parts_lower:
                    continue
            bucket = extract_bucket_from_path(p) or "global"
            if bucket not in idx:
                idx[bucket] = {}
            idx[bucket][p.stem] = p
    return idx


def parse_enhanced_specs(spec_str: str) -> list[tuple[str | None, Path]]:
    specs: list[tuple[str | None, Path]] = []
    for raw in [x.strip() for x in spec_str.split(",") if x.strip()]:
        if "=" in raw:
            bucket_raw, p = raw.split("=", 1)
            specs.append((normalize_bucket(bucket_raw) or bucket_raw.strip(), Path(p.strip())))
        else:
            specs.append((None, Path(raw)))
    return specs


def main():
    parser = argparse.ArgumentParser("Phase 10 Full-Scene Hero Plot")
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--noisy-root", required=True)
    parser.add_argument("--enhanced-roots", required=True, help="Comma-separated roots, optional bucket=path")
    parser.add_argument("--out-dir", default="./outputs/phase10/hero_fullscene")
    parser.add_argument("--target-count", type=int, default=824)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--clean-marker", default="clean")
    parser.add_argument("--noisy-marker", default="noisy")
    args = parser.parse_args()

    clean_root = Path(args.clean_root)
    noisy_root = Path(args.noisy_root)
    enhanced_specs = parse_enhanced_specs(args.enhanced_roots)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_idx = index_wavs_by_bucket(clean_root, required_marker=args.clean_marker)
    noisy_idx = index_wavs_by_bucket(noisy_root, required_marker=args.noisy_marker)

    rows = []
    snr_vals = []
    delta_vals = []

    for bucket_hint, er in enhanced_specs:
        bucket = (normalize_bucket(bucket_hint) if isinstance(bucket_hint, str) else None) or extract_bucket_from_path(er) or "global"
        enh_local = index_wavs_by_bucket(er)
        enh_files = enh_local.get(bucket, {})
        if not enh_files:
            enh_files = enh_local.get("global", {})

        clean_bucket = clean_idx.get(bucket, clean_idx.get("global", {}))
        noisy_bucket = noisy_idx.get(bucket, noisy_idx.get("global", {}))

        for name, enh_path in enh_files.items():
            cpath = clean_bucket.get(name)
            npath = noisy_bucket.get(name)
            if cpath is None or npath is None:
                continue

            y_clean, _ = sf.read(str(cpath), dtype="float32")
            y_noisy, _ = sf.read(str(npath), dtype="float32")
            y_enh, _ = sf.read(str(enh_path), dtype="float32")

            L = min(len(y_clean), len(y_noisy), len(y_enh))
            y_clean = y_clean[:L]
            y_noisy = y_noisy[:L]
            y_enh = y_enh[:L]

            snr_in = estimate_snr(y_clean, y_noisy, fs=args.fs)
            sdr_noisy = compute_sdr(y_clean, y_noisy)
            sdr_enh = compute_sdr(y_clean, y_enh)
            delta = sdr_enh - sdr_noisy

            snr_vals.append(snr_in)
            delta_vals.append(delta)
            rows.append({"name": name, "bucket": bucket, "snr_in": snr_in, "delta_sdr": delta})

    if not rows:
        raise RuntimeError("No matched clean/noisy/enhanced files found")

    snr_np = np.asarray(snr_vals)
    delta_np = np.asarray(delta_vals)

    # Scatter + quadratic trend
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(snr_np, delta_np, s=24, alpha=0.55, color="#2a9d8f", edgecolor="black", linewidth=0.3)
    if len(snr_np) > 3:
        z = np.polyfit(snr_np, delta_np, deg=2)
        p = np.poly1d(z)
        xs = np.linspace(np.min(snr_np), np.max(snr_np), 200)
        ax.plot(xs, p(xs), "--", color="#264653", linewidth=2.5, label="Quadratic trend")

    ax.axhline(0.0, linestyle="--", color="#1b9e77", linewidth=1.5, label="Baseline")
    ax.axhline(0.02, linestyle="--", color="#e9c46a", linewidth=2.0, label="Physics limit (+0.02 dB)")

    ax.set_title("Full-Scene Hero Plot: Input SNR vs ΔSDR", fontsize=15, fontweight="bold")
    ax.set_xlabel("Input SNR (dB)", fontsize=13, fontweight="bold")
    ax.set_ylabel("ΔSDR (dB)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    fig.savefig(out_dir / "hero_plot_fullscene.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    report = {
        "matched_samples": len(rows),
        "target_count": args.target_count,
        "coverage_ratio": float(len(rows) / max(args.target_count, 1)),
        "snr_range": [float(np.min(snr_np)), float(np.max(snr_np))],
        "delta_sdr": {
            "mean": float(np.mean(delta_np)),
            "median": float(np.median(delta_np)),
            "std": float(np.std(delta_np)),
            "min": float(np.min(delta_np)),
            "max": float(np.max(delta_np)),
            "gt_0p02_ratio": float(np.mean(delta_np > 0.02)),
        },
        "status": "target_met" if len(rows) >= args.target_count else "target_not_met",
    }

    with open(out_dir / "hero_plot_fullscene_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(out_dir / "hero_plot_fullscene_rows.json", "w", encoding="utf-8") as f:
        json.dump(rows, f)

    print("[HeroFullScene] saved:", out_dir / "hero_plot_fullscene.png")
    print("[HeroFullScene] matched_samples:", len(rows))
    print("[HeroFullScene] target_count:", args.target_count)
    print("[HeroFullScene] delta_sdr_mean:", f"{np.mean(delta_np):+.6f} dB")


if __name__ == "__main__":
    main()
