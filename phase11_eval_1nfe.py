import os
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from phase9_nhfae_e1 import stft, istft
from phase9_tcnac_codec import NHFAE_E2_TCNAC


def snr_to_tps_gamma(
    snr_db: torch.Tensor,
    snr_min: float = -10.0,
    snr_max: float = 10.0,
    gamma_low_snr: float = 1.0,
    gamma_high_snr: float = 0.02,
    mode: str = "sigmoid",
    tau: float = 0.0,
    beta: float = 1.0,
) -> torch.Tensor:
    if mode == "linear":
        snr_norm = (snr_db - snr_min) / max(1e-6, (snr_max - snr_min))
        snr_norm = torch.clamp(snr_norm, 0.0, 1.0)
        return gamma_low_snr - (gamma_low_snr - gamma_high_snr) * snr_norm
    if mode == "exp":
        dist = torch.clamp(snr_db - snr_min, min=0.0)
        g = torch.exp(-dist / max(1e-6, beta))
        return gamma_high_snr + (gamma_low_snr - gamma_high_snr) * g
    g = torch.sigmoid((snr_db - tau) / max(1e-6, beta))
    return gamma_low_snr * (1.0 - g) + gamma_high_snr * g


def compute_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    diff = est - ref
    return float(10.0 * np.log10(np.sum(ref ** 2) / (np.sum(diff ** 2) + 1e-10) + 1e-10))


def load_pairs(data_dir: str):
    pairs = []
    snr_dirs = sorted(glob.glob(os.path.join(data_dir, "snr_*")))
    for sd in snr_dirs:
        snr_name = Path(sd).name
        try:
            snr_db = float(snr_name.split("_")[1])
        except Exception:
            snr_db = 0.0
        cdir = os.path.join(sd, "clean")
        ndir = os.path.join(sd, "noisy")
        if not os.path.isdir(cdir) or not os.path.isdir(ndir):
            continue
        cfiles = sorted(glob.glob(os.path.join(cdir, "*.wav")))
        for cf in cfiles:
            stem = Path(cf).stem
            nf = os.path.join(ndir, f"{stem}.wav")
            if os.path.exists(nf):
                pairs.append((cf, nf, snr_db, snr_name, stem))
    return pairs


def main():
    parser = argparse.ArgumentParser("1-NFE evaluation with bucket-wise SDR stats")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=128)
    parser.add_argument("--tps-enabled", action="store_true")
    parser.add_argument("--tps-mode", type=str, default="sigmoid", choices=["linear", "sigmoid", "exp"])
    parser.add_argument("--tps-gamma-high-snr", type=float, default=0.02)
    parser.add_argument("--tps-tau", type=float, default=0.0)
    parser.add_argument("--tps-beta", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wav_out = os.path.join(args.out_dir, "wav_1nfe")
    os.makedirs(wav_out, exist_ok=True)

    device = torch.device(args.device)
    model = NHFAE_E2_TCNAC().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    pairs = load_pairs(args.data_dir)
    if not pairs:
        raise RuntimeError("No clean/noisy pairs found")

    rows = []
    with torch.no_grad():
        for i, (cf, nf, snr_db, snr_name, stem) in enumerate(pairs):
            clean, sr = sf.read(cf)
            noisy, _ = sf.read(nf)
            clean_t = torch.from_numpy(clean.astype(np.float32)).to(device).unsqueeze(0)
            noisy_t = torch.from_numpy(noisy.astype(np.float32)).to(device).unsqueeze(0)

            noisy_stft = stft(noisy_t, n_fft=args.n_fft, hop=args.hop)
            outputs = model(noisy_stft)
            enhanced_stft = outputs["S_enhanced"]

            if args.tps_enabled:
                snr_t = torch.tensor([snr_db], device=device, dtype=torch.float32)
                gamma = snr_to_tps_gamma(
                    snr_t,
                    gamma_high_snr=args.tps_gamma_high_snr,
                    mode=args.tps_mode,
                    tau=args.tps_tau,
                    beta=args.tps_beta,
                ).view(-1, 1, 1)
                enhanced_stft = noisy_stft + gamma * (enhanced_stft - noisy_stft)

            enh_t = istft(enhanced_stft, length=noisy_t.shape[-1], n_fft=args.n_fft, hop=args.hop)
            enh = enh_t.squeeze(0).detach().cpu().numpy()

            out_name = f"{snr_name}_{stem}_{i:04d}.wav"
            out_path = os.path.join(wav_out, out_name)
            sf.write(out_path, enh, sr)

            sdr_noisy = compute_sdr(clean, noisy)
            sdr_enh = compute_sdr(clean, enh)
            rows.append(
                {
                    "snr_db": float(snr_db),
                    "snr_name": snr_name,
                    "stem": stem,
                    "sdr_noisy": sdr_noisy,
                    "sdr_enh": sdr_enh,
                    "delta_sdr": float(sdr_enh - sdr_noisy),
                }
            )

    delta = np.array([r["delta_sdr"] for r in rows], dtype=np.float64)
    stats = {
        "num_samples": int(len(rows)),
        "delta_sdr_mean": float(np.mean(delta)),
        "delta_sdr_median": float(np.median(delta)),
        "delta_sdr_std": float(np.std(delta)),
        "delta_sdr_min": float(np.min(delta)),
        "delta_sdr_max": float(np.max(delta)),
        "ratio_positive": float(np.mean(delta > 0.0)),
        "ratio_above_1p5": float(np.mean(delta > 1.5)),
        "ratio_above_0p02": float(np.mean(delta > 0.02)),
        "bucket_stats": {},
        "is_1nfe": True,
    }

    for snr in sorted(set([r["snr_db"] for r in rows])):
        vals = np.array([r["delta_sdr"] for r in rows if r["snr_db"] == snr], dtype=np.float64)
        key = f"snr_{snr:.1f}"
        stats["bucket_stats"][key] = {
            "num_samples": int(len(vals)),
            "delta_sdr_mean": float(np.mean(vals)),
            "delta_sdr_median": float(np.median(vals)),
            "ratio_positive": float(np.mean(vals > 0.0)),
            "ratio_above_0p02": float(np.mean(vals > 0.02)),
        }

    with open(os.path.join(args.out_dir, "rows_1nfe.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "stats_1nfe.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[1-NFE] samples:", stats["num_samples"])
    print("[1-NFE] delta_sdr_mean:", f"{stats['delta_sdr_mean']:.6f}")
    print("[1-NFE] ratio_above_1p5:", f"{stats['ratio_above_1p5']:.4f}")
    print("[1-NFE] ratio_positive:", f"{stats['ratio_positive']:.4f}")


if __name__ == "__main__":
    main()
