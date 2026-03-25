"""
Phase 11: 在均匀SNR (-10~10 dB) 数据集上训练 T-CNAC + MPICM
- 99个样本，均匀分布在11个SNR bucket
- 完整的训练→推理→Hero Plot评估流程
"""

import os
import glob
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from datetime import datetime

# 导入现有模块
import sys
sys.path.insert(0, str(Path(__file__).parent))

from phase9_nhfae_e1 import (
    stft, istft, 
    mrstft_loss, NHFAE_E1
)
from phase9_tcnac_codec import NHFAE_E2_TCNAC


class UniformSNRDataset(Dataset):
    """均匀SNR数据集加载器"""
    
    def __init__(self, data_dir: str, sr: int = 16000):
        """
        Args:
            data_dir: 数据根目录（包含多个snr_*.* 子目录）
            sr: 采样率
        """
        self.sr = sr
        self.samples = []
        self.snr_buckets = {}
        
        # 收集所有SNR bucket
        snr_dirs = sorted(glob.glob(os.path.join(data_dir, "snr_*")))
        print(f"[UniformSNRDataset] 找到 {len(snr_dirs)} 个SNR bucket")
        
        for snr_dir in snr_dirs:
            snr_name = Path(snr_dir).name
            try:
                snr_val = float(snr_name.split("_")[1])
            except:
                snr_val = 0.0
            
            clean_dir = os.path.join(snr_dir, "clean")
            noisy_dir = os.path.join(snr_dir, "noisy")
            
            if not os.path.exists(clean_dir) or not os.path.exists(noisy_dir):
                continue
            
            clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
            noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
            
            for cf, nf in zip(clean_files, noisy_files):
                self.samples.append({
                    "clean_path": cf,
                    "noisy_path": nf,
                    "snr_db": snr_val,
                    "snr_name": snr_name,
                })
            
            self.snr_buckets[snr_name] = {
                "snr_db": snr_val,
                "count": len(clean_files),
                "clean_dir": clean_dir,
                "noisy_dir": noisy_dir,
            }
            
            print(f"  {snr_name}: {len(clean_files)} samples (SNR={snr_val:.1f} dB)")
        
        print(f"[UniformSNRDataset] 总共 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载音频
        clean, _ = sf.read(sample["clean_path"])
        noisy, _ = sf.read(sample["noisy_path"])
        
        clean = torch.from_numpy(clean.astype(np.float32))
        noisy = torch.from_numpy(noisy.astype(np.float32))
        
        return {
            "clean": clean.unsqueeze(0),  # [1, T]
            "noisy": noisy.unsqueeze(0),  # [1, T]
            "snr_db": sample["snr_db"],
            "snr_name": sample["snr_name"],
            "clean_path": sample["clean_path"],
            "noisy_path": sample["noisy_path"],
        }


def train_and_evaluate(
    data_dir: str,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """
    完整的训练和评估流程
    """
    
    # 初始化
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    
    # 数据加载
    print("\n[Phase11] 加载数据...")
    dataset = UniformSNRDataset(data_dir, sr=16000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 模型初始化
    print("[Phase11] 初始化模型...")
    model = NHFAE_E2_TCNAC().to(device)
    
    # 损失和优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 统计用
    train_log = {
        "epochs": [],
        "losses": [],
        "lrs": [],
    }
    
    print(f"[Phase11] 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Phase11] 开始训练 {epochs} 个epoch，batch_size={batch_size}...\n")
    
    # 训练循环
    best_loss = float("inf")
    best_ckpt = os.path.join(output_dir, "ckpt", "best.pt")
    Path(os.path.dirname(best_ckpt)).mkdir(parents=True, exist_ok=True)
    
    # STFT参数
    n_fft = 512
    hop_len = 128
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            clean = batch["clean"].to(device)  # [B, 1, T]
            noisy = batch["noisy"].to(device)  # [B, 1, T]
            
            # 前向传播
            optimizer.zero_grad()
            
            # STFT变换
            noisy_stft = stft(noisy.squeeze(1), n_fft=n_fft, hop=hop_len)  # [B, F, T]
            clean_stft = stft(clean.squeeze(1), n_fft=n_fft, hop=hop_len)  # [B, F, T]
            
            # 模型处理（将STFT输入转换为复数张量）
            # noisy_stft 是 [B, F, T]，需要转为 [B, F, T, 2]
            noisy_stft_complex = torch.stack([noisy_stft.real, noisy_stft.imag], dim=-1).to(device)
            
            enhanced_stft, mag_pred, phase_pred = model(noisy_stft_complex)  # 各[B, F, T, 2] / [B, F, T]
            
            # iSTFT变换
            enhanced_wav = istft(enhanced_stft, length=noisy.shape[-1], n_fft=n_fft, hop=hop_len).unsqueeze(1)  # [B, 1, T]
            
            # 损失计算
            recon_loss = nn.L1Loss()(enhanced_wav, clean)
            mrstft_l = mrstft_loss(enhanced_wav.squeeze(1), clean.squeeze(1))
            
            # 总损失
            total_loss = recon_loss + 0.2 * mrstft_l
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            batch_count += 1
            
            if (batch_idx + 1) % max(1, len(dataloader) // 2) == 0:
                print(f"  [Ep {epoch:2d}] Batch {batch_idx+1:3d}/{len(dataloader)} | Loss={total_loss.item():.6f}")
        
        # 周期统计
        avg_loss = epoch_loss / batch_count
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        
        train_log["epochs"].append(epoch)
        train_log["losses"].append(avg_loss)
        train_log["lrs"].append(current_lr)
        
        print(f"[Ep {epoch:2d}] AvgLoss={avg_loss:.6f} | LR={current_lr:.2e}")
        
        # 保存最佳checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_ckpt)
            print(f"       ✓ 保存最佳checkpoint")
    
    # 保存训练日志
    log_path = os.path.join(output_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"\n[Phase11] ✓ 训练完成！最佳Loss={best_loss:.6f}")
    print(f"[Phase11] 日志: {log_path}")
    
    # 推理和评估
    print("\n[Phase11] 开始推理...")
    model.eval()
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    
    wav_out_dir = os.path.join(output_dir, "wav")
    Path(wav_out_dir).mkdir(parents=True, exist_ok=True)
    
    inference_results = []
    
    # STFT参数
    n_fft = 512
    hop_len = 128
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            clean = batch["clean"].to(device)
            noisy = batch["noisy"].to(device)
            snr_names = batch["snr_name"]
            snr_dbs = batch["snr_db"].cpu().numpy()
            
            # 推理
            noisy_stft = stft(noisy.squeeze(1), n_fft=n_fft, hop=hop_len)
            noisy_stft_complex = torch.stack([noisy_stft.real, noisy_stft.imag], dim=-1).to(device)
            
            enhanced_stft, _, _ = model(noisy_stft_complex)
            
            enhanced_wav = istft(enhanced_stft, length=noisy.shape[-1], n_fft=n_fft, hop=hop_len).unsqueeze(1)  # [B, 1, T]
            
            # 保存增强音频
            for b in range(enhanced_wav.shape[0]):
                snr_name = snr_names[b]
                snr_db = snr_dbs[b]
                wav = enhanced_wav[b, 0, :].cpu().numpy()
                
                # 文件名
                sample_idx = batch_idx * batch_size + b
                out_name = f"{snr_name}_sample_{sample_idx:03d}"
                out_path = os.path.join(wav_out_dir, f"{out_name}.wav")
                
                sf.write(out_path, wav, 16000)
                
                inference_results.append({
                    "sample_id": sample_idx,
                    "snr_name": snr_name,
                    "snr_db": float(snr_db),
                    "output_path": out_path,
                })
            
            if (batch_idx + 1) % max(1, len(dataloader) // 2) == 0:
                print(f"  推理进度: {batch_idx + 1}/{len(dataloader)}")
    
    # 保存推理结果
    result_path = os.path.join(output_dir, "inference_results.json")
    with open(result_path, "w") as f:
        json.dump(inference_results, f, indent=2)
    
    print(f"[Phase11] ✓ 推理完成！生成 {len(inference_results)} 个增强音频")
    print(f"[Phase11] 输出目录: {wav_out_dir}")
    
    # 生成Hero Plot
    print("\n[Phase11] 生成Hero Plot...")
    generate_hero_plot(
        output_dir=output_dir,
        data_dir=data_dir,
        enhanced_wav_dir=wav_out_dir,
    )
    
    print("\n" + "="*60)
    print("[Phase11] ✓ 任务完成！")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print(f"最佳Loss: {best_loss:.6f}")
    print(f"推理样本: {len(inference_results)}")


def generate_hero_plot(output_dir: str, data_dir: str, enhanced_wav_dir: str):
    """生成Hero Plot评估图"""
    
    try:
        import matplotlib.pyplot as plt
        from scipy.io import wavfile
    except ImportError:
        print("[警告] matplotlib/scipy未安装，跳过Hero Plot生成")
        return
    
    # 简单的SDR计算
    def compute_sdr(ref, est):
        """计算SDR"""
        ref = ref.astype(np.float64)
        est = est.astype(np.float64)
        
        diff = est - ref
        sdr = 10 * np.log10(np.sum(ref**2) / (np.sum(diff**2) + 1e-10) + 1e-10)
        return float(sdr)
    
    # 收集信息
    results = []
    
    # 遍历SNR bucket
    snr_dirs = sorted(glob.glob(os.path.join(data_dir, "snr_*")))
    
    for snr_dir in snr_dirs:
        snr_name = Path(snr_dir).name
        try:
            snr_db = float(snr_name.split("_")[1])
        except:
            snr_db = 0.0
        
        clean_dir = os.path.join(snr_dir, "clean")
        noisy_dir = os.path.join(snr_dir, "noisy")
        
        if not os.path.exists(clean_dir):
            continue
        
        clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
        
        for cf in clean_files:
            fname = Path(cf).stem
            
            # 找到对应的增强音频
            pattern = f"*{fname}*"
            enh_files = glob.glob(os.path.join(enhanced_wav_dir, f"*{fname}*.wav"))
            
            if not enh_files:
                continue
            
            enh_path = enh_files[0]
            noisy_path = os.path.join(noisy_dir, f"{fname}.wav")
            
            # 加载音频
            try:
                clean, _ = sf.read(cf)
                noisy, _ = sf.read(noisy_path)
                enhanced, _ = sf.read(enh_path)
                
                # 计算SDR
                sdr_noisy = compute_sdr(clean, noisy)
                sdr_enh = compute_sdr(clean, enhanced)
                delta_sdr = sdr_enh - sdr_noisy
                
                results.append({
                    "snr_db": snr_db,
                    "sdr_noisy": sdr_noisy,
                    "sdr_enh": sdr_enh,
                    "delta_sdr": delta_sdr,
                })
            except Exception as e:
                print(f"[警告] 计算SDR失败 {fname}: {e}")
    
    if not results:
        print("[警告] 没有可用的结果用于Hero Plot")
        return
    
    # 绘制
    snrs = np.array([r["snr_db"] for r in results])
    delta_sdrs = np.array([r["delta_sdr"] for r in results])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(snrs, delta_sdrs, alpha=0.6, s=100, label="ΔSDRi")
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="零线")
    plt.axhline(y=0.02, color="g", linestyle="--", alpha=0.5, label="目标 (+0.02dB)")
    
    # 拟合曲线
    z = np.polyfit(snrs, delta_sdrs, 2)
    p = np.poly1d(z)
    x_fit = np.linspace(snrs.min(), snrs.max(), 100)
    plt.plot(x_fit, p(x_fit), "b-", alpha=0.5, label="趋势")
    
    plt.xlabel("输入 SNR (dB)")
    plt.ylabel("ΔSDRi (dB)")
    plt.title("Phase11: 均匀SNR (-10~10 dB) Hero Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, "hero_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # 统计
    stats = {
        "num_samples": len(results),
        "delta_sdr_mean": float(np.mean(delta_sdrs)),
        "delta_sdr_std": float(np.std(delta_sdrs)),
        "delta_sdr_median": float(np.median(delta_sdrs)),
        "delta_sdr_min": float(np.min(delta_sdrs)),
        "delta_sdr_max": float(np.max(delta_sdrs)),
        "snr_range": [float(snrs.min()), float(snrs.max())],
        "ratio_positive": float(np.mean(delta_sdrs > 0)),
        "ratio_above_0p02": float(np.mean(delta_sdrs > 0.02)),
    }
    
    stats_path = os.path.join(output_dir, "hero_plot_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"[Phase11] ✓ Hero Plot 生成完成！")
    print(f"[Phase11]   样本数: {stats['num_samples']}")
    print(f"[Phase11]   ΔSDRi 均值: {stats['delta_sdr_mean']:.6f} dB")
    print(f"[Phase11]   ΔSDRi 中值: {stats['delta_sdr_median']:.6f} dB")
    print(f"[Phase11]   正增益比例: {stats['ratio_positive']*100:.1f}%")
    print(f"[Phase11]   超过 +0.02dB 比例: {stats['ratio_above_0p02']*100:.1f}%")
    print(f"[Phase11]   绘图: {plot_path}")


def main():
    parser = argparse.ArgumentParser("Phase11: 均匀SNR训练与评估")
    parser.add_argument("--data-dir", type=str, default="./outputs/phase11/uniform_snr_minus10_plus10",
                        help="均匀SNR数据目录")
    parser.add_argument("--output-dir", type=str, default="./outputs/phase11/uniform_snr_train",
                        help="输出目录")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="batch大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Phase11: 均匀SNR (-10~10) 内完整训练评估")
    print("="*60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Device: {args.device}")
    print("="*60 + "\n")
    
    train_and_evaluate(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
