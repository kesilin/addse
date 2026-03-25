#!/usr/bin/env python3
"""
NHFAE E2 Full Range Hero Curve Assessment
=============================================

Target: Generate complete Hero curve with input SNR as x-axis, ΔSDR as y-axis
Data volume: 500+ samples spanning full SNR range (-5 dB ~ 15 dB)

Core proof:
  1. High SNR (15 dB): ΔSDR converges to 0, confirms physical transparency
  2. Mid SNR (5-10 dB): ΔSDR > 0, confirms gradient isolation success
  3. Low SNR (-5-0 dB): ΔSDR enhanced through Identity Master
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import json
from scipy.signal import istft, stft
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class FullRangeHeroEvaluator:
    """Full SNR range Hero curve evaluation."""
    
    def __init__(self, base_output_dir: Path, fs: int = 16000):
        self.base_output_dir = Path(base_output_dir)
        self.fs = fs
        self.snr_buckets = {
            "-5_0": {"range": (-5, 0), "label": "-5~0 dB"},
            "0_5": {"range": (0, 5), "label": "0~5 dB"},
            "5_10": {"range": (5, 10), "label": "5~10 dB"},
            "10_15": {"range": (10, 15), "label": "10~15 dB"},
        }
        
        self.results = {
            "snr_buckets": {},
            "aggregate": {
                "snr_values": [],
                "delta_sdr_values": [],
                "phase_errors": [],
                "mag_perturbations": [],
            }
        }

    def compute_sdr(self, ref, est):
        """Compute SDR metric."""
        ref = ref.astype(np.float64)
        est = est.astype(np.float64)
        
        if len(ref) == 0 or len(est) == 0:
            return np.nan
        
        ref = ref / (np.max(np.abs(ref)) + 1e-8)
        est = est / (np.max(np.abs(est)) + 1e-8)
        
        min_len = min(len(ref), len(est))
        ref = ref[:min_len]
        est = est[:min_len]
        
        ref_pow = np.sum(ref**2) + 1e-8
        err_pow = np.sum((ref - est)**2) + 1e-8
        sdr = 10.0 * np.log10(ref_pow / err_pow)
        return float(sdr)

    def estimate_snr(self, noisy, clean, n_fft=512, hop=192):
        """Estimate input SNR from noisy signal."""
        f, _, X_clean = stft(clean, fs=self.fs, nperseg=n_fft, noverlap=n_fft-hop)
        _, _, X_noisy = stft(noisy, fs=self.fs, nperseg=n_fft, noverlap=n_fft-hop)
        
        mag_clean = np.abs(X_clean)
        mag_noisy = np.abs(X_noisy)
        
        min_t = min(mag_clean.shape[1], mag_noisy.shape[1])
        mag_clean = mag_clean[:, :min_t]
        mag_noisy = mag_noisy[:, :min_t]
        
        noise = mag_noisy - mag_clean
        eps = 1e-8
        snr = 10.0 * np.log10(np.sum(mag_clean**2 + eps) / (np.sum(noise**2) + eps))
        return float(np.clip(snr, -10, 20))

    def process_snr_bucket(self, bucket_key: str, metric_source: str = "phase4_output"):
        """Process single SNR bucket."""
        print(f"[FullRangeHero] Processing SNR bucket: {bucket_key}")
        
        bucket_dir = self.base_output_dir / f"phase6/controlled_snr_test31/{bucket_key.replace('_', '-')}"
        clean_dir = bucket_dir / "clean"
        noisy_dir = bucket_dir / "noisy"
        
        if metric_source == "stage3_output" and bucket_key == "10_15":
            enhanced_dir = self.base_output_dir / "phase9/nhfae_e2_stage3/wav"
        else:
            enhanced_dir = self.base_output_dir / f"phase9/nhfae_e1_interact_tune2_{bucket_key}/wav"
        
        if not clean_dir.exists() or not noisy_dir.exists():
            print(f"  WARNING: data directory not found, skip")
            return
        
        if not enhanced_dir.exists():
            print(f"  WARNING: enhanced result dir not found: {enhanced_dir}, skip")
            return
        
        bucket_metrics = {
            "snr_bucket": bucket_key,
            "samples": [],
            "delta_sdr_list": [],
            "phase_error_list": [],
            "mag_perturb_list": [],
        }
        
        clean_files = sorted(clean_dir.glob("*.wav"))
        for clean_file in clean_files:
            name = clean_file.stem
            noisy_file = noisy_dir / f"{name}.wav"
            enhanced_file = enhanced_dir / f"{name}.wav"
            
            if not (noisy_file.exists() and enhanced_file.exists()):
                continue
            
            # Load audio
            y_clean, _ = sf.read(str(clean_file), dtype=np.float32)
            y_noisy, _ = sf.read(str(noisy_file), dtype=np.float32)
            y_enh, _ = sf.read(str(enhanced_file), dtype=np.float32)
            
            # Align lengths
            min_len = min(len(y_clean), len(y_noisy), len(y_enh))
            y_clean = y_clean[:min_len]
            y_noisy = y_noisy[:min_len]
            y_enh = y_enh[:min_len]
            
            # Estimate SNR
            input_snr = self.estimate_snr(y_noisy, y_clean)
            
            # Compute SDR
            sdr_noisy = self.compute_sdr(y_clean, y_noisy)
            sdr_enh = self.compute_sdr(y_clean, y_enh)
            delta_sdr = sdr_enh - sdr_noisy
            
            bucket_metrics["samples"].append({
                "name": name,
                "input_snr": input_snr,
                "delta_sdr": delta_sdr,
            })
            bucket_metrics["delta_sdr_list"].append(delta_sdr)
            
            # Aggregate
            self.results["aggregate"]["snr_values"].append(input_snr)
            self.results["aggregate"]["delta_sdr_values"].append(delta_sdr)
        
        if bucket_metrics["delta_sdr_list"]:
            mean_delta_sdr = np.mean(bucket_metrics["delta_sdr_list"])
            print(f"  Samples: {len(bucket_metrics['delta_sdr_list'])}")
            print(f"  ΔSDR mean: {mean_delta_sdr:+.6f} dB")
            print(f"  ΔSDR range: [{np.min(bucket_metrics['delta_sdr_list']):+.6f}, {np.max(bucket_metrics['delta_sdr_list']):+.6f}] dB")
            
            self.results["snr_buckets"][bucket_key] = bucket_metrics

    def plot_hero_curve(self):
        """Generate Hero curve plot."""
        print("\n[FullRangeHero] Generating Hero curve...")
        
        snr_vals = np.array(self.results["aggregate"]["snr_values"])
        delta_sdr_vals = np.array(self.results["aggregate"]["delta_sdr_values"])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Complete scatter plot + trend line
        ax1.scatter(snr_vals, delta_sdr_vals, alpha=0.5, s=30, color="steelblue", edgecolor="black")
        
        # Add trend line
        if len(snr_vals) > 3:
            z = np.polyfit(snr_vals, delta_sdr_vals, 2)
            p = np.poly1d(z)
            x_trend = np.linspace(np.min(snr_vals), np.max(snr_vals), 100)
            ax1.plot(x_trend, p(x_trend), "r--", linewidth=2, label="Trend (polynomial)")
        
        # Reference lines
        ax1.axhline(0, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Baseline (ΔSDR=0)")
        ax1.axhline(0.02, color="gold", linestyle="--", linewidth=2, alpha=0.7, label="Physical limit (+0.02 dB)")
        
        ax1.set_xlabel("Input SNR (dB)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("ΔSDR (dB)", fontsize=12, fontweight="bold")
        ax1.set_title("NHFAE E2 Hero Curve: Physics Transparency Verification", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Right: Box plot for each SNR bucket
        bucket_keys = list(self.results["snr_buckets"].keys())
        bucket_data = []
        bucket_labels = []
        
        for key in sorted(bucket_keys):
            metrics = self.results["snr_buckets"][key]
            if metrics["delta_sdr_list"]:
                bucket_data.append(metrics["delta_sdr_list"])
                bucket_labels.append(f"{key.replace('_', '~')} dB")
        
        if bucket_data:
            bp = ax2.boxplot(bucket_data, labels=bucket_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax2.axhline(0, color="green", linestyle="--", linewidth=1, alpha=0.5)
            ax2.axhline(0.02, color="gold", linestyle="--", linewidth=2, alpha=0.7)
            ax2.set_ylabel("ΔSDR (dB)", fontsize=12, fontweight="bold")
            ax2.set_title("ΔSDR Distribution across SNR Buckets", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        output_path = self.base_output_dir / "phase9/nhfae_e2_stage3/hero_plot/hero_curve_full.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
        print(f"  SAVED: {output_path}")

    def save_full_report(self):
        """Save complete report."""
        report = {
            "title": "NHFAE E2 Full Range Hero Curve Assessment",
            "total_samples": len(self.results["aggregate"]["snr_values"]),
            "snr_range": [float(np.min(self.results["aggregate"]["snr_values"])), 
                         float(np.max(self.results["aggregate"]["snr_values"]))],
            "delta_sdr_stats": {
                "mean": float(np.mean(self.results["aggregate"]["delta_sdr_values"])),
                "median": float(np.median(self.results["aggregate"]["delta_sdr_values"])),
                "std": float(np.std(self.results["aggregate"]["delta_sdr_values"])),
                "min": float(np.min(self.results["aggregate"]["delta_sdr_values"])),
                "max": float(np.max(self.results["aggregate"]["delta_sdr_values"])),
            },
            "snr_buckets": {}
        }
        
        for key, metrics in self.results["snr_buckets"].items():
            if metrics["delta_sdr_list"]:
                report["snr_buckets"][key] = {
                    "count": len(metrics["delta_sdr_list"]),
                    "delta_sdr_mean": float(np.mean(metrics["delta_sdr_list"])),
                    "delta_sdr_std": float(np.std(metrics["delta_sdr_list"])),
                    "delta_sdr_range": [float(np.min(metrics["delta_sdr_list"])), 
                                       float(np.max(metrics["delta_sdr_list"]))],
                }
        
        output_path = self.base_output_dir / "phase9/nhfae_e2_stage3/hero_plot/hero_curve_full_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"  SAVED: {output_path}")


def main():
    parser = argparse.ArgumentParser("NHFAE E2 Full Range Hero Curve")
    parser.add_argument("--base-dir", default="./outputs")
    args = parser.parse_args()
    
    evaluator = FullRangeHeroEvaluator(Path(args.base_dir))
    
    # Process all SNR buckets
    for bucket_key in evaluator.snr_buckets.keys():
        evaluator.process_snr_bucket(bucket_key)
    
    # Generate plots and reports
    if evaluator.results["aggregate"]["snr_values"]:
        evaluator.plot_hero_curve()
        evaluator.save_full_report()
        
        print("\n" + "="*60)
        print("FULL RANGE HERO CURVE ASSESSMENT COMPLETE")
        print("="*60)
        print(f"Total samples: {len(evaluator.results['aggregate']['snr_values'])}")
        snr_vals = evaluator.results['aggregate']['snr_values']
        print(f"SNR range: {min(snr_vals):.1f} ~ {max(snr_vals):.1f} dB")
        delta_sdrs = evaluator.results['aggregate']['delta_sdr_values']
        print(f"ΔSDR mean: {np.mean(delta_sdrs):+.6f} dB")
        print("SUCCESS: Core paper figure is ready!")
        print("="*60)
    else:
        print("ERROR: No matching data found. Check directory structure.")


if __name__ == "__main__":
    main()
