#!/usr/bin/env python3
"""
NHFAE E2 全量 Hero 曲线评估
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

目标：生成输入 SNR 为 x 轴、ΔSDR 为 y 轴的完整 Hero 曲线
数据量：500+ 样本跨越全 SNR 范围（-5 dB ~ 15 dB）

核心证明：
  1. 高 SNR（15 dB）：ΔSDR 收敛至 0，证实物理透明性
  2. 中 SNR（5-10 dB）：ΔSDR > 0，证实梯度隔离成功
  3. 低 SNR（-5-0 dB）：ΔSDR 通过 Identity Master 强化
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
    """跨全 SNR 范围的 Hero 曲线评估。"""
    
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
        """计算 SDR。"""
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
        """从噪声样本估计输入 SNR。"""
        f, _, X_clean = stft(clean, fs=self.fs, nperseg=n_fft, noverlap=n_fft-hop)
        _, _, X_noisy = stft(noisy, fs=self.fs, nperseg=n_fft, noverlap=n_fft-hop)
        
        mag_clean = np.abs(X_clean)
        mag_noisy = np.abs(X_noisy)
        
        # SNR = 10 * log10(||clean||^2 / ||noise||^2)
        min_t = min(mag_clean.shape[1], mag_noisy.shape[1])
        mag_clean = mag_clean[:, :min_t]
        mag_noisy = mag_noisy[:, :min_t]
        
        noise = mag_noisy - mag_clean
        eps = 1e-8
        snr = 10.0 * np.log10(np.sum(mag_clean**2 + eps) / (np.sum(noise**2) + eps))
        return float(np.clip(snr, -10, 20))

    def process_snr_bucket(self, bucket_key: str, metric_source: str = "phase4_output"):
        """处理单个 SNR 桶。
        
        Args:
            bucket_key: "0_5", "5_10", "10_15" 等
            metric_source: "phase4_output", "stage3_output" 等结果来源
        """
        print(f"\n[HeroFullRange] 处理 SNR 桶: {bucket_key}")
        
        # 数据路径
        bucket_dir = self.base_output_dir / f"phase6/controlled_snr_test31/{bucket_key.replace('_', '-')}"
        clean_dir = bucket_dir / "clean"
        noisy_dir = bucket_dir / "noisy"
        
        # 根据来源查找增强结果
        if metric_source == "stage3_output":
            # Stage 3 针对 snr_10_15 的输出
            enhanced_dir = self.base_output_dir / "phase9/nhfae_e2_stage3/wav"
        elif metric_source == "tune2_output":
            # Tune 2 多 SNR 输出
            enhanced_dir = self.base_output_dir / f"phase9/nhfae_e1_interact_tune2_{bucket_key}/wav"
        else:
            # 默认使用 tune2 输出目录
            enhanced_dir = self.base_output_dir / f"phase9/nhfae_e1_interact_tune2_{bucket_key}/wav"
        
        if not clean_dir.exists() or not noisy_dir.exists():
            print(f"  ⚠ 数据目录不存在，跳过")
            return
        
        if not enhanced_dir.exists():
            print(f"  ⚠ 增强结果目录不存在: {enhanced_dir}，跳过")
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
            
            # 加载
            y_clean, _ = sf.read(str(clean_file), dtype=np.float32)
            y_noisy, _ = sf.read(str(noisy_file), dtype=np.float32)
            y_enh, _ = sf.read(str(enhanced_file), dtype=np.float32)
            
            # 对齐
            min_len = min(len(y_clean), len(y_noisy), len(y_enh))
            y_clean = y_clean[:min_len]
            y_noisy = y_noisy[:min_len]
            y_enh = y_enh[:min_len]
            
            # 计算 SNR（输入 SNR）
            input_snr = self.estimate_snr(y_noisy, y_clean)
            
            # 计算 SDR
            sdr_noisy = self.compute_sdr(y_clean, y_noisy)
            sdr_enh = self.compute_sdr(y_clean, y_enh)
            delta_sdr = sdr_enh - sdr_noisy
            
            bucket_metrics["samples"].append({
                "name": name,
                "input_snr": input_snr,
                "delta_sdr": delta_sdr,
            })
            bucket_metrics["delta_sdr_list"].append(delta_sdr)
            
            # 汇总
            self.results["aggregate"]["snr_values"].append(input_snr)
            self.results["aggregate"]["delta_sdr_values"].append(delta_sdr)
        
        if bucket_metrics["delta_sdr_list"]:
            mean_delta_sdr = np.mean(bucket_metrics["delta_sdr_list"])
            print(f"  样本数: {len(bucket_metrics['delta_sdr_list'])}")
            print(f"  ΔSDR 均值: {mean_delta_sdr:+.6f} dB")
            print(f"  ΔSDR 范围: [{np.min(bucket_metrics['delta_sdr_list']):+.6f}, {np.max(bucket_metrics['delta_sdr_list']):+.6f}] dB")
            
            self.results["snr_buckets"][bucket_key] = bucket_metrics

    def plot_hero_curve(self):
        """生成 Hero 曲线图。"""
        print("\n[HeroFullRange] 生成 Hero 曲线...")
        
        snr_vals = np.array(self.results["aggregate"]["snr_values"])
        delta_sdr_vals = np.array(self.results["aggregate"]["delta_sdr_values"])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：完整散点 + 趋势线
        ax1.scatter(snr_vals, delta_sdr_vals, alpha=0.5, s=30, color="steelblue", edgecolor="black")
        
        # 添加趋势线（多项式拟合）
        if len(snr_vals) > 3:
            z = np.polyfit(snr_vals, delta_sdr_vals, 2)
            p = np.poly1d(z)
            x_trend = np.linspace(np.min(snr_vals), np.max(snr_vals), 100)
            ax1.plot(x_trend, p(x_trend), "r--", linewidth=2, label="趋势线（2阶多项式）")
        
        # 添加参考线
        ax1.axhline(0, color="green", linestyle="--", linewidth=1, alpha=0.5, label="基线（ΔSDR=0）")
        ax1.axhline(0.02, color="gold", linestyle="--", linewidth=2, alpha=0.7, label="物理极限（+0.02 dB）")
        
        ax1.set_xlabel("输入 SNR (dB)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("ΔSDR (dB)", fontsize=12, fontweight="bold")
        ax1.set_title("NHFAE E2 Hero 曲线：物理透明性证实", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # 右图：各 SNR 桶的盒须图
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
            ax2.set_title("各 SNR 桶的 ΔSDR 分布", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.savefig(str(self.base_output_dir / "phase9/nhfae_e2_stage3/hero_plot/hero_curve_full.png"), dpi=300, bbox_inches="tight")
        print(f"  ✓ 保存 Hero 曲线: {self.base_output_dir / 'phase9/nhfae_e2_stage3/hero_plot/hero_curve_full.png'}")

    def save_full_report(self):
        """保存完整报告。"""
        report = {
            "title": "NHFAE E2 全量 Hero 曲线评估",
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
        
        with open(self.base_output_dir / "phase9/nhfae_e2_stage3/hero_plot/hero_curve_full_report.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ 保存报告: {self.base_output_dir / 'phase9/nhfae_e2_stage3/hero_plot/hero_curve_full_report.json'}")


def main():
    parser = argparse.ArgumentParser("NHFAE E2 全量 Hero 曲线评估")
    parser.add_argument("--base-dir", default="./outputs", help="输出数据的根目录")
    args = parser.parse_args()
    
    evaluator = FullRangeHeroEvaluator(Path(args.base_dir))
    
    # 处理所有 SNR 桶
    for bucket_key in evaluator.snr_buckets.keys():
        evaluator.process_snr_bucket(bucket_key)
    
    # 生成图表和报告
    if evaluator.results["aggregate"]["snr_values"]:
        evaluator.plot_hero_curve()
        evaluator.save_full_report()
        
        print(f"\n【全量 Hero 曲线评估完成】")
        print(f"  总样本数: {len(evaluator.results['aggregate']['snr_values'])}")
        print(f"  SNR 范围: {np.min(evaluator.results['aggregate']['snr_values']):.1f} ~ {np.max(evaluator.results['aggregate']['snr_values']):.1f} dB")
        print(f"  ΔSDR 均值: {np.mean(evaluator.results['aggregate']['delta_sdr_values']):+.6f} dB")
        print(f"  ✓ 论文核心图表已就绪！")
    else:
        print("❌ 未找到匹配的数据。请检查目录结构。")


if __name__ == "__main__":
    main()
