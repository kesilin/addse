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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from datetime import datetime

# 导入现有模块
import sys
sys.path.insert(0, str(Path(__file__).parent))

from phase9_nhfae_e1 import stft, istft, mrstft_loss
from phase9_tcnac_codec import NHFAE_E2_TCNAC


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """将角度差包裹到 [-pi, pi)。"""
    return torch.atan2(torch.sin(x), torch.cos(x))


def snr_to_alpha(
    snr_db: torch.Tensor,
    snr_min: float = -10.0,
    snr_max: float = 10.0,
    alpha_low_snr: float = 1.0,
    alpha_high_snr: float = 0.2,
    mode: str = "linear",
    tau: float = 0.0,
    beta: float = 2.0,
) -> torch.Tensor:
    """SNR 动态权重：低 SNR 给更高权重，高 SNR 逼近透明输运。"""
    if mode == "sigmoid":
        g = torch.sigmoid((snr_db - tau) / max(1e-6, beta))
        alpha = alpha_low_snr * (1.0 - g) + alpha_high_snr * g
        return alpha
    if mode == "exp":
        dist = torch.clamp(snr_db - snr_min, min=0.0)
        g = torch.exp(-dist / max(1e-6, beta))
        alpha = alpha_high_snr + (alpha_low_snr - alpha_high_snr) * g
        return alpha
    snr_norm = (snr_db - snr_min) / max(1e-6, (snr_max - snr_min))
    snr_norm = torch.clamp(snr_norm, 0.0, 1.0)
    alpha = alpha_low_snr - (alpha_low_snr - alpha_high_snr) * snr_norm
    return alpha


def circular_smooth_cross_entropy(
    pred_phase: torch.Tensor,
    target_phase: torch.Tensor,
    num_bins: int = 72,
    sigma_bins: float = 1.5,
) -> torch.Tensor:
    """
    Circular Smooth Cross-Entropy (CSCE).
    通过单位圆距离构建软标签，解决相位边界跳变问题。
    """
    device = pred_phase.device
    pi = float(np.pi)

    bin_centers = torch.linspace(-pi, pi, steps=num_bins + 1, device=device)[:-1]
    sigma_rad = (2.0 * pi / num_bins) * sigma_bins

    # 由连续相位生成可导 logits
    dist_phase = wrap_to_pi(pred_phase.unsqueeze(-1) - bin_centers.view(1, 1, 1, -1))
    logits = -(dist_phase ** 2) / max(1e-6, 2.0 * sigma_rad * sigma_rad)

    # 目标 bin 索引
    target_idx = torch.floor((target_phase + pi) / (2.0 * pi) * num_bins).long()
    target_idx = torch.clamp(target_idx, min=0, max=num_bins - 1)

    # 圆环软标签
    bin_ids = torch.arange(num_bins, device=device).view(1, 1, 1, -1)
    delta = torch.abs(bin_ids - target_idx.unsqueeze(-1))
    circ_delta = torch.minimum(delta, num_bins - delta).float()
    soft_target = torch.exp(-0.5 * (circ_delta / max(1e-6, sigma_bins)) ** 2)
    soft_target = soft_target / soft_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    log_prob = F.log_softmax(logits, dim=-1)
    return -(soft_target * log_prob).sum(dim=-1).mean()


def normalize_csce_loss(csce_raw: torch.Tensor, num_bins: int, mode: str = "logk") -> torch.Tensor:
    """将 CSCE 量纲归一到与重建项同数量级。"""
    if mode == "k":
        return csce_raw / float(max(1, num_bins))
    if mode == "logk":
        return csce_raw / float(max(1e-6, np.log(max(2, num_bins))))
    return csce_raw


def csce_weight_schedule(
    epoch: int,
    target_lambda: float,
    stage1_epochs: int,
    warmup_epochs: int,
) -> float:
    """
    阶段式损失诱导：
    - Stage 1: lambda=0（只训骨架）
    - Stage 2: 线性 warmup 到 target
    - Stage 3: 维持 target
    """
    if epoch <= stage1_epochs:
        return 0.0
    if warmup_epochs <= 0:
        return float(target_lambda)
    ramp_pos = epoch - stage1_epochs
    if ramp_pos >= warmup_epochs:
        return float(target_lambda)
    return float(target_lambda) * float(ramp_pos) / float(warmup_epochs)


def stage3_weight_schedule(
    epoch: int,
    target_lambda: float,
    start_epoch: int,
    warmup_epochs: int,
) -> float:
    """Stage 3 透明正则权重日程。"""
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return float(target_lambda)
    ramp_pos = epoch - start_epoch + 1
    if ramp_pos >= warmup_epochs:
        return float(target_lambda)
    return float(target_lambda) * float(ramp_pos) / float(warmup_epochs)


def snr_to_tps_gamma(
    snr_db: torch.Tensor,
    snr_min: float = -10.0,
    snr_max: float = 10.0,
    gamma_low_snr: float = 1.0,
    gamma_high_snr: float = 0.05,
    mode: str = "sigmoid",
    tau: float = 0.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """SNR 驱动的截断采样强度：低 SNR 强生成，高 SNR 近 Identity。"""
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


class UniformSNRDataset(Dataset):
    """均匀SNR数据集加载器"""
    
    def __init__(self, data_dir: str, sr: int = 16000):
        self.sr = sr
        self.samples = []
        snr_dirs = sorted(glob.glob(os.path.join(data_dir, "snr_*")))
        
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
            
            print(f"  {snr_name}: {len(clean_files)} samples (SNR={snr_val:.1f} dB)")
        
        print(f"[UniformSNRDataset] 总共 {len(self.samples)} 个样本\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        clean, _ = sf.read(sample["clean_path"])
        noisy, _ = sf.read(sample["noisy_path"])
        
        clean = torch.from_numpy(clean.astype(np.float32))
        noisy = torch.from_numpy(noisy.astype(np.float32))
        
        return {
            "clean": clean.unsqueeze(0),  # [1, T]
            "noisy": noisy.unsqueeze(0),  # [1, T]
            "snr_db": sample["snr_db"],
            "snr_name": sample["snr_name"],
            "file_stem": Path(sample["clean_path"]).stem,
        }


def train_and_evaluate(
    data_dir: str,
    output_dir: str,
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: str = "cuda",
    lambda_csce: float = 0.20,
    lambda_cfm: float = 0.10,
    phase_bins: int = 72,
    phase_sigma_bins: float = 1.5,
    alpha_low_snr: float = 1.0,
    alpha_high_snr: float = 0.2,
    csce_norm_mode: str = "logk",
    stage1_epochs: int = 10,
    csce_warmup_epochs: int = 10,
    alpha_mode: str = "linear",
    alpha_tau: float = 0.0,
    alpha_beta: float = 2.0,
    lambda_trans: float = 0.1,
    stage3_start_epoch: int = 21,
    trans_warmup_epochs: int = 5,
    tps_enabled: bool = True,
    tps_mode: str = "sigmoid",
    tps_gamma_high_snr: float = 0.05,
    tps_tau: float = 0.0,
    tps_beta: float = 1.0,
):
    """完整的训练和评估流程"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    
    print("[Phase11] 加载数据...")
    dataset = UniformSNRDataset(data_dir, sr=16000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print("[Phase11] 初始化模型...")
    model = NHFAE_E2_TCNAC().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_log = {
        "epochs": [],
        "losses": [],
        "recon_losses": [],
        "mrstft_losses": [],
        "csce_losses": [],
        "cfm_losses": [],
        "alpha_mean": [],
        "lambda_csce_eff": [],
        "csce_raw_losses": [],
        "trans_losses": [],
        "lambda_trans_eff": [],
    }
    
    print(f"[Phase11] 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Phase11] 开始训练 {epochs} 个epoch，batch_size={batch_size}...\n")
    
    best_loss = float("inf")
    best_ckpt = os.path.join(output_dir, "ckpt", "best.pt")
    Path(os.path.dirname(best_ckpt)).mkdir(parents=True, exist_ok=True)
    
    n_fft = 512
    hop_len = 128
    
    for epoch in range(1, epochs + 1):
        model.train()
        lambda_csce_eff = csce_weight_schedule(
            epoch=epoch,
            target_lambda=lambda_csce,
            stage1_epochs=stage1_epochs,
            warmup_epochs=csce_warmup_epochs,
        )
        lambda_trans_eff = stage3_weight_schedule(
            epoch=epoch,
            target_lambda=lambda_trans,
            start_epoch=stage3_start_epoch,
            warmup_epochs=trans_warmup_epochs,
        )
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_mrstft = 0.0
        epoch_csce = 0.0
        epoch_cfm = 0.0
        epoch_alpha = 0.0
        epoch_csce_raw = 0.0
        epoch_trans = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            clean = batch["clean"].to(device)  # [B, 1, T]
            noisy = batch["noisy"].to(device)  # [B, 1, T]
            snr_db = batch["snr_db"].to(device=device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            # STFT变换
            noisy_stft = stft(noisy.squeeze(1), n_fft=n_fft, hop=hop_len)  # [B, F, T] complex
            clean_stft = stft(clean.squeeze(1), n_fft=n_fft, hop=hop_len)  # [B, F, T] complex
            
            # 模型前向（返回字典）
            outputs = model(noisy_stft)
            enhanced_stft = outputs["S_enhanced"]
            
            # iSTFT变换
            enhanced_wav = istft(enhanced_stft, length=noisy.shape[-1], n_fft=n_fft, hop=hop_len).unsqueeze(1)
            
            # 损失计算
            recon_loss = nn.L1Loss()(enhanced_wav, clean)
            mrstft_l = mrstft_loss(enhanced_wav.squeeze(1), clean.squeeze(1))

            # 相位拓扑：CSCE + 动态残差输运
            pred_phase = torch.angle(enhanced_stft)
            target_phase = torch.angle(clean_stft)

            csce_raw = circular_smooth_cross_entropy(
                pred_phase,
                target_phase,
                num_bins=phase_bins,
                sigma_bins=phase_sigma_bins,
            )
            csce_l = normalize_csce_loss(csce_raw, num_bins=phase_bins, mode=csce_norm_mode)

            phase_err = wrap_to_pi(pred_phase - target_phase).abs()
            phase_err_per_sample = phase_err.mean(dim=(1, 2))
            alpha = snr_to_alpha(
                snr_db,
                snr_min=-10.0,
                snr_max=10.0,
                alpha_low_snr=alpha_low_snr,
                alpha_high_snr=alpha_high_snr,
                mode=alpha_mode,
                tau=alpha_tau,
                beta=alpha_beta,
            )
            cfm_l = (alpha * phase_err_per_sample).mean()

            # 高 SNR 透明锚定：SNR>=0 样本惩罚过度生成
            trans_gate = (snr_db >= 0.0).float()
            trans_per_sample = torch.abs(enhanced_stft - noisy_stft).mean(dim=(1, 2))
            if trans_gate.sum() > 0:
                trans_l = (trans_gate * trans_per_sample).sum() / (trans_gate.sum() + 1e-8)
            else:
                trans_l = torch.zeros((), device=device)
            
            total_loss = (
                recon_loss
                + 0.2 * mrstft_l
                + lambda_csce_eff * csce_l
                + lambda_cfm * cfm_l
                + lambda_trans_eff * trans_l
            )
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_mrstft += mrstft_l.item()
            epoch_csce += csce_l.item()
            epoch_cfm += cfm_l.item()
            epoch_alpha += alpha.mean().item()
            epoch_csce_raw += csce_raw.item()
            epoch_trans += trans_l.item()
            batch_count += 1
            
            if (batch_idx + 1) % max(1, len(dataloader) // 2) == 0:
                print(
                    f"  [Ep {epoch:2d}] Batch {batch_idx+1:3d}/{len(dataloader)} "
                    f"| Loss={total_loss.item():.6f} | CSCE={csce_l.item():.6f} "
                    f"| λcsce={lambda_csce_eff:.4f} | λtrans={lambda_trans_eff:.4f} | alpha={alpha.mean().item():.3f}"
                )
        
        avg_loss = epoch_loss / batch_count
        avg_recon = epoch_recon / batch_count
        avg_mrstft = epoch_mrstft / batch_count
        avg_csce = epoch_csce / batch_count
        avg_cfm = epoch_cfm / batch_count
        avg_alpha = epoch_alpha / batch_count
        avg_csce_raw = epoch_csce_raw / batch_count
        avg_trans = epoch_trans / batch_count
        scheduler.step()
        
        train_log["epochs"].append(epoch)
        train_log["losses"].append(avg_loss)
        train_log["recon_losses"].append(avg_recon)
        train_log["mrstft_losses"].append(avg_mrstft)
        train_log["csce_losses"].append(avg_csce)
        train_log["cfm_losses"].append(avg_cfm)
        train_log["alpha_mean"].append(avg_alpha)
        train_log["lambda_csce_eff"].append(lambda_csce_eff)
        train_log["csce_raw_losses"].append(avg_csce_raw)
        train_log["trans_losses"].append(avg_trans)
        train_log["lambda_trans_eff"].append(lambda_trans_eff)
        
        print(
            f"[Ep {epoch:2d}] AvgLoss={avg_loss:.6f} | Recon={avg_recon:.6f} "
            f"| MRSTFT={avg_mrstft:.6f} | CSCE(raw/norm)={avg_csce_raw:.3f}/{avg_csce:.6f} "
            f"| CFM={avg_cfm:.6f} | Trans={avg_trans:.6f} "
            f"| λcsce={lambda_csce_eff:.4f} | λtrans={lambda_trans_eff:.4f} | alpha={avg_alpha:.3f}"
        )
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_ckpt)
            print(f"       ✓ 保存最佳checkpoint\n")
    
    log_path = os.path.join(output_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    
    print(f"\n[Phase11] ✓ 训练完成！最佳Loss={best_loss:.6f}\n")
    
    # 推理
    print("[Phase11] 开始推理...")
    model.eval()
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    
    wav_out_dir = os.path.join(output_dir, "wav")
    Path(wav_out_dir).mkdir(parents=True, exist_ok=True)
    
    inference_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            clean = batch["clean"].to(device)
            noisy = batch["noisy"].to(device)
            snr_names = batch["snr_name"]
            snr_dbs = batch["snr_db"].cpu().numpy()
            snr_db_t = batch["snr_db"].to(device=device, dtype=torch.float32)
            file_stems = batch["file_stem"]
            
            noisy_stft = stft(noisy.squeeze(1), n_fft=n_fft, hop=hop_len)
            
            outputs = model(noisy_stft)
            enhanced_stft = outputs["S_enhanced"]

            if tps_enabled:
                gamma = snr_to_tps_gamma(
                    snr_db_t,
                    snr_min=-10.0,
                    snr_max=10.0,
                    gamma_low_snr=1.0,
                    gamma_high_snr=tps_gamma_high_snr,
                    mode=tps_mode,
                    tau=tps_tau,
                    beta=tps_beta,
                ).view(-1, 1, 1)
                enhanced_stft = noisy_stft + gamma * (enhanced_stft - noisy_stft)
            enhanced_wav = istft(enhanced_stft, length=noisy.shape[-1], n_fft=n_fft, hop=hop_len).unsqueeze(1)
            
            for b in range(enhanced_wav.shape[0]):
                snr_name = snr_names[b]
                snr_db = snr_dbs[b]
                wav = enhanced_wav[b, 0, :].cpu().numpy()
                stem = file_stems[b]
                
                sample_idx = batch_idx * batch_size + b
                out_name = f"{snr_name}_{stem}_{sample_idx:03d}"
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
    
    result_path = os.path.join(output_dir, "inference_results.json")
    with open(result_path, "w") as f:
        json.dump(inference_results, f, indent=2)
    
    print(f"[Phase11] ✓ 推理完成！生成 {len(inference_results)} 个增强音频\n")
    
    # Hero Plot
    print("[Phase11] 生成Hero Plot...")
    generate_hero_plot(output_dir, data_dir, wav_out_dir)
    
    print("\n" + "="*60)
    print("[Phase11] ✓ 全流程完成！")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print(f"最佳Loss: {best_loss:.6f}")


def generate_hero_plot(output_dir: str, data_dir: str, enhanced_wav_dir: str):
    """生成Hero Plot评估图"""
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[警告] matplotlib未安装，跳过Hero Plot生成")
        return
    
    def compute_sdr(ref, est):
        ref = ref.astype(np.float64)
        est = est.astype(np.float64)
        diff = est - ref
        sdr = 10 * np.log10(np.sum(ref**2) / (np.sum(diff**2) + 1e-10) + 1e-10)
        return float(sdr)
    
    results = []
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
            matched = glob.glob(os.path.join(enhanced_wav_dir, f"*_{fname}_*.wav"))
            if not matched:
                matched = glob.glob(os.path.join(enhanced_wav_dir, f"*{fname}*.wav"))

            if not matched:
                continue
            
            try:
                clean, _ = sf.read(cf)
                noisy, _ = sf.read(os.path.join(noisy_dir, f"{fname}.wav"))
                enhanced, _ = sf.read(matched[0])
                
                sdr_noisy = compute_sdr(clean, noisy)
                sdr_enh = compute_sdr(clean, enhanced)
                delta_sdr = sdr_enh - sdr_noisy
                
                results.append({
                    "snr_db": snr_db,
                    "delta_sdr": delta_sdr,
                })
            except Exception as e:
                print(f"[警告] 计算SDR失败 {fname}: {e}")
    
    if not results:
        print("[警告] 没有可用结果用于Hero Plot")
        return
    
    snrs = np.array([r["snr_db"] for r in results])
    delta_sdrs = np.array([r["delta_sdr"] for r in results])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(snrs, delta_sdrs, alpha=0.6, s=100, label="ΔSDRi")
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="零线")
    plt.axhline(y=0.02, color="g", linestyle="--", alpha=0.5, label="目标 (+0.02dB)")
    
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

    # 分桶诊断：定位是低 SNR 未提上来还是高 SNR 被投毒
    bucket_stats = {}
    for snr_v in sorted(set([float(r["snr_db"]) for r in results])):
        vals = np.array([r["delta_sdr"] for r in results if float(r["snr_db"]) == snr_v], dtype=np.float64)
        if len(vals) == 0:
            continue
        bucket_stats[f"snr_{snr_v:.1f}"] = {
            "num_samples": int(len(vals)),
            "delta_sdr_mean": float(np.mean(vals)),
            "delta_sdr_median": float(np.median(vals)),
            "delta_sdr_std": float(np.std(vals)),
            "ratio_positive": float(np.mean(vals > 0)),
            "ratio_above_0p02": float(np.mean(vals > 0.02)),
        }
    stats["bucket_stats"] = bucket_stats
    
    stats_path = os.path.join(output_dir, "hero_plot_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"[Phase11] ✓ Hero Plot 生成完成！")
    print(f"[Phase11]   样本数: {stats['num_samples']}")
    print(f"[Phase11]   ΔSDRi 均值: {stats['delta_sdr_mean']:.6f} dB")
    print(f"[Phase11]   ΔSDRi 中值: {stats['delta_sdr_median']:.6f} dB")
    print(f"[Phase11]   正增益比例: {stats['ratio_positive']*100:.1f}%")
    print(f"[Phase11]   超过 +0.02dB 比例: {stats['ratio_above_0p02']*100:.1f}%")
    for bucket_name, b in stats["bucket_stats"].items():
        print(
            f"[Phase11]   {bucket_name}: mean={b['delta_sdr_mean']:.4f} dB, "
            f"pos={b['ratio_positive']*100:.1f}% ({b['num_samples']} samples)"
        )
    print(f"[Phase11]   绘图: {plot_path}\n")


def main():
    parser = argparse.ArgumentParser("Phase11: 均匀SNR训练与评估")
    parser.add_argument("--data-dir", type=str, default="./outputs/phase11/uniform_snr_minus10_plus10",
                        help="均匀SNR数据目录")
    parser.add_argument("--output-dir", type=str, default="./outputs/phase11/uniform_snr_train",
                        help="输出目录")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="batch大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--lambda-csce", type=float, default=0.20, help="CSCE损失权重")
    parser.add_argument("--lambda-cfm", type=float, default=0.10, help="残差输运损失权重")
    parser.add_argument("--phase-bins", type=int, default=72, help="离散相位bin数量")
    parser.add_argument("--phase-sigma-bins", type=float, default=1.5, help="CSCE圆环平滑sigma（bin单位）")
    parser.add_argument("--alpha-low-snr", type=float, default=1.0, help="低SNR时alpha")
    parser.add_argument("--alpha-high-snr", type=float, default=0.2, help="高SNR时alpha")
    parser.add_argument("--alpha-mode", type=str, default="linear", choices=["linear", "sigmoid", "exp"],
                        help="alpha调度模式")
    parser.add_argument("--alpha-tau", type=float, default=0.0,
                        help="alpha sigmoid模式阈值tau")
    parser.add_argument("--alpha-beta", type=float, default=2.0,
                        help="alpha sigmoid/exp模式平滑参数beta")
    parser.add_argument("--csce-norm-mode", type=str, default="logk", choices=["none", "logk", "k"],
                        help="CSCE归一化方式")
    parser.add_argument("--stage1-epochs", type=int, default=10,
                        help="Stage1骨架训练轮数（CSCE=0）")
    parser.add_argument("--csce-warmup-epochs", type=int, default=10,
                        help="CSCE线性升权warmup轮数")
    parser.add_argument("--lambda-trans", type=float, default=0.1,
                        help="高SNR透明锚定损失权重")
    parser.add_argument("--stage3-start-epoch", type=int, default=21,
                        help="Stage3透明锚定起始epoch")
    parser.add_argument("--trans-warmup-epochs", type=int, default=5,
                        help="透明锚定升权warmup轮数")
    parser.add_argument("--tps-enabled", action="store_true",
                        help="推理启用SNR截断采样（TPS）")
    parser.add_argument("--tps-mode", type=str, default="sigmoid", choices=["linear", "sigmoid", "exp"],
                        help="TPS强度曲线模式")
    parser.add_argument("--tps-gamma-high-snr", type=float, default=0.05,
                        help="高SNR时TPS最小保留比例")
    parser.add_argument("--tps-tau", type=float, default=0.0,
                        help="TPS sigmoid阈值tau")
    parser.add_argument("--tps-beta", type=float, default=1.0,
                        help="TPS平滑参数beta")
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
    print(f"lambda_csce: {args.lambda_csce}")
    print(f"lambda_cfm: {args.lambda_cfm}")
    print(f"phase_bins: {args.phase_bins}, phase_sigma_bins: {args.phase_sigma_bins}")
    print(f"alpha(low->high SNR): {args.alpha_low_snr} -> {args.alpha_high_snr}")
    print(f"alpha_mode/tau/beta: {args.alpha_mode}/{args.alpha_tau}/{args.alpha_beta}")
    print(f"csce_norm_mode: {args.csce_norm_mode}")
    print(f"curriculum(stage1/warmup): {args.stage1_epochs}/{args.csce_warmup_epochs}")
    print(f"lambda_trans: {args.lambda_trans}")
    print(f"stage3/trans_warmup: {args.stage3_start_epoch}/{args.trans_warmup_epochs}")
    print(f"TPS enabled/mode/gamma_high: {args.tps_enabled}/{args.tps_mode}/{args.tps_gamma_high_snr}")
    print(f"Device: {args.device}")
    print("="*60 + "\n")
    
    train_and_evaluate(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        lambda_csce=args.lambda_csce,
        lambda_cfm=args.lambda_cfm,
        phase_bins=args.phase_bins,
        phase_sigma_bins=args.phase_sigma_bins,
        alpha_low_snr=args.alpha_low_snr,
        alpha_high_snr=args.alpha_high_snr,
        csce_norm_mode=args.csce_norm_mode,
        stage1_epochs=args.stage1_epochs,
        csce_warmup_epochs=args.csce_warmup_epochs,
        alpha_mode=args.alpha_mode,
        alpha_tau=args.alpha_tau,
        alpha_beta=args.alpha_beta,
        lambda_trans=args.lambda_trans,
        stage3_start_epoch=args.stage3_start_epoch,
        trans_warmup_epochs=args.trans_warmup_epochs,
        tps_enabled=args.tps_enabled,
        tps_mode=args.tps_mode,
        tps_gamma_high_snr=args.tps_gamma_high_snr,
        tps_tau=args.tps_tau,
        tps_beta=args.tps_beta,
    )


if __name__ == "__main__":
    main()
