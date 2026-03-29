"""
Phase 13: 浅层交互并联模型（实验2）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

设计目标：
  1. 复数输入 → 极坐标分解 → 双分支处理
  2. 幅度分支 + 相位分支（并联）
  3. 浅层交互模块（验证跨模态融合的必要性）
  4. 融合输出 → 时域波形

消融价值：
  - Exp 0: 无交互（独立分支）
  - Exp 1: 早期融合（共享编码）
  - Exp 2: 浅层交互（本脚本）← 验证门控融合的作用
  - Exp 3: 深层交互（现有 Phase11）

预期结果：
  ΔSDR 提升: 0.8 dB (Exp1) → 1.4 dB (Exp2) → 2.06 dB (Exp3)

作者：Assistant
日期：2026-03-26
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

from phase9_nhfae_e1 import stft, istft, mrstft_loss, wrap_to_pi
from phase11_train_uniform_v2 import circular_smooth_cross_entropy, normalize_csce_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 浅层编码器 - 幅度分支
# ═══════════════════════════════════════════════════════════════════════════════

class ShallowMagEncoder(nn.Module):
    """轻量级幅度编码器（3层残差卷积）"""
    
    def __init__(self, d_model: int = 48, n_layers: int = 3):
        super().__init__()
        self.d_model = d_model
        
        # Stem: 1→d_model
        self.stem = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )
        
        # 残差块
        self.blocks = nn.ModuleList([
            self._make_block(d_model) for _ in range(n_layers)
        ])
    
    @staticmethod
    def _make_block(d_model: int):
        return nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )
    
    def forward(self, mag_log: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mag_log: [B, F, T] 对数幅度
        
        Returns:
            feat_mag: [B, d, F, T] 幅度特征
        """
        x = mag_log.unsqueeze(1)  # [B, 1, F, T]
        x = self.stem(x)           # [B, d, F, T]
        
        for block in self.blocks:
            x = block(x) + x       # 残差连接
        
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 浅层编码器 - 相位分支
# ═══════════════════════════════════════════════════════════════════════════════

class ShallowPhaseEncoder(nn.Module):
    """轻量级相位编码器（cos/sin 表示）"""
    
    def __init__(self, d_model: int = 48, n_layers: int = 3):
        super().__init__()
        self.d_model = d_model
        
        # Stem: 2 (cos+sin) → d_model
        self.stem = nn.Sequential(
            nn.Conv2d(2, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )
        
        self.blocks = nn.ModuleList([
            self._make_block(d_model) for _ in range(n_layers)
        ])
    
    @staticmethod
    def _make_block(d_model: int):
        return nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )
    
    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phase: [B, F, T] 相位角
        
        Returns:
            feat_phase: [B, d, F, T] 相位特征
        """
        # 转为 [cos, sin] 表示
        cos_p = torch.cos(phase)   # [B, F, T]
        sin_p = torch.sin(phase)   # [B, F, T]
        x = torch.stack([cos_p, sin_p], dim=1)  # [B, 2, F, T]
        
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x) + x
        
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 浅层交互模块 - 核心创新
# ═══════════════════════════════════════════════════════════════════════════════

class ShallowCrossAttention(nn.Module):
    """跨模态交互：幅度←→相位，通过门控融合"""
    
    def __init__(self, d_model: int = 48):
        super().__init__()
        
        # 幅度分支被相位驱动的权重门
        self.mag_gate = nn.Sequential(
            nn.Conv2d(d_model, d_model // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model // 4, d_model, kernel_size=1),
        )
        
        # 相位分支被幅度驱动的权重门
        self.phase_gate = nn.Sequential(
            nn.Conv2d(d_model, d_model // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model // 4, d_model, kernel_size=1),
        )
        
        # 融合投影
        self.fuse_proj = nn.Conv2d(2 * d_model, d_model, kernel_size=1)
    
    def forward(
        self, 
        mag_feat: torch.Tensor, 
        phase_feat: torch.Tensor,
        interaction_strength: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mag_feat: [B, d, F, T] 幅度特征
            phase_feat: [B, d, F, T] 相位特征
            interaction_strength: 交互强度 (0~1)
        
        Returns:
            mag_interact: [B, d, F, T] 交互后幅度特征
            phase_interact: [B, d, F, T] 交互后相位特征
        """
        # 相位驱动的幅度权重
        mag_weight = torch.sigmoid(self.mag_gate(phase_feat))
        
        # 幅度驱动的相位权重
        phase_weight = torch.sigmoid(self.phase_gate(mag_feat))
        
        # 门控应用
        mag_interact = mag_feat * (1.0 + interaction_strength * mag_weight)
        phase_interact = phase_feat * (1.0 + interaction_strength * phase_weight)
        
        return mag_interact, phase_interact


# ═══════════════════════════════════════════════════════════════════════════════
# 浅层解码头
# ═══════════════════════════════════════════════════════════════════════════════

class MagDecoder(nn.Module):
    """从特征解码幅度"""
    
    def __init__(self, d_model: int = 48):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.Conv2d(d_model, 1, kernel_size=1),
        )
    
    def forward(self, feat_mag: torch.Tensor) -> torch.Tensor:
        """返回 [B, F, T] 的幅度增强系数"""
        out = self.decoder(feat_mag)  # [B, 1, F, T]
        # 用 sigmoid 确保非负
        out = torch.sigmoid(out)
        return out.squeeze(1)  # [B, F, T]


class PhaseDecoder(nn.Module):
    """从特征解码相位残差"""
    
    def __init__(self, d_model: int = 48):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.Conv2d(d_model, 1, kernel_size=1),
        )
    
    def forward(self, feat_phase: torch.Tensor) -> torch.Tensor:
        """返回 [B, F, T] 的相位残差"""
        out = self.decoder(feat_phase)  # [B, 1, F, T]
        # 用 tanh 将残差限制在 [-π, π]
        out = np.pi * torch.tanh(out)
        return out.squeeze(1)  # [B, F, T]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase13ShallowModel - 完整模型
# ═══════════════════════════════════════════════════════════════════════════════

class Phase13ShallowModel(nn.Module):
    """
    浅层交互并联模型
    
    架构：
        复数 STFT
          ↓
        极坐标分解
          ↓
        [MagEncoder] [PhaseEncoder]（并联）
          ↓
        ShallowCrossAttention（交互）
          ↓
        [MagDecoder] [PhaseDecoder]
          ↓
        融合 → 复数 STFT
    """
    
    def __init__(self, d_model: int = 48, n_layers: int = 3, half_version: bool = False):
        super().__init__()
        
        self.d_model = d_model
        self.half_version = half_version
        
        # 双分支编码
        self.mag_encoder = ShallowMagEncoder(d_model, n_layers)
        self.phase_encoder = ShallowPhaseEncoder(d_model, n_layers)
        
        # 浅层交互
        self.interaction = ShallowCrossAttention(d_model)
        
        # 解码头
        self.mag_decoder = MagDecoder(d_model)
        self.phase_decoder = PhaseDecoder(d_model)
    
    def forward(self, noisy_stft: torch.Tensor) -> dict:
        """
        Args:
            noisy_stft: [B, F, T] 复 STFT
        
        Returns:
            outputs: 字典，包含 S_enhanced, aux 信息等
        """
        B, F, T = noisy_stft.shape
        
        # ════════════════════════════════════════════════════════════
        # Step 1: 极坐标分解
        # ════════════════════════════════════════════════════════════
        mag_noisy = torch.abs(noisy_stft)       # [B, F, T]
        phase_noisy = torch.angle(noisy_stft)   # [B, F, T]
        
        mag_log = torch.log1p(mag_noisy)  # [B, F, T]
        
        # ════════════════════════════════════════════════════════════
        # Step 2: 并联编码
        # ════════════════════════════════════════════════════════════
        feat_mag = self.mag_encoder(mag_log)      # [B, d, F, T]
        feat_phase = self.phase_encoder(phase_noisy)  # [B, d, F, T]
        
        # ════════════════════════════════════════════════════════════
        # Step 3: 浅层交互
        # ════════════════════════════════════════════════════════════
        interaction_strength = 0.05 if self.half_version else 0.1
        feat_mag_int, feat_phase_int = self.interaction(
            feat_mag, feat_phase,
            interaction_strength=interaction_strength
        )
        
        # ════════════════════════════════════════════════════════════
        # Step 4: 解码
        # ════════════════════════════════════════════════════════════
        mag_scale = self.mag_decoder(feat_mag_int)         # [B, F, T] in (0,1)
        phase_residual = self.phase_decoder(feat_phase_int)  # [B, F, T]
        
        # ════════════════════════════════════════════════════════════
        # Step 5: 融合输出
        # ════════════════════════════════════════════════════════════
        mag_enhanced = mag_noisy * mag_scale                # 缩放幅度
        phase_enhanced = phase_noisy + phase_residual       # 调整相位
        
        S_enhanced = mag_enhanced * torch.exp(1j * phase_enhanced)
        
        outputs = {
            "S_enhanced": S_enhanced,
            "mag_scale": mag_scale,
            "phase_residual": phase_residual,
            "mag_noisy": mag_noisy,
            "phase_noisy": phase_noisy,
            "feat_mag": feat_mag_int,
            "feat_phase": feat_phase_int,
        }
        
        return outputs


# ═══════════════════════════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════════════════════════

class UniformSNRDataset(Dataset):
    """与 Phase11 兼容的均匀 SNR 数据集"""
    
    def __init__(self, data_dir: str, sr: int = 16000):
        self.data_dir = Path(data_dir)
        self.sr = sr
        self.samples = []
        
        snr_dirs = sorted(self.data_dir.glob("snr_*"))
        for snr_dir in snr_dirs:
            snr_name = snr_dir.name
            try:
                snr_db = float(snr_name.split("_")[1])
            except:
                snr_db = 0.0
            
            clean_dir = snr_dir / "clean"
            noisy_dir = snr_dir / "noisy"
            
            if not clean_dir.exists() or not noisy_dir.exists():
                continue
            
            for clean_file in clean_dir.glob("*.wav"):
                noisy_file = noisy_dir / clean_file.name
                if noisy_file.exists():
                    self.samples.append({
                        "clean": clean_file,
                        "noisy": noisy_file,
                        "snr_db": snr_db,
                        "snr_name": snr_name,
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        clean, _ = sf.read(str(sample["clean"]))
        noisy, _ = sf.read(str(sample["noisy"]))
        
        clean = torch.from_numpy(clean.astype(np.float32))
        noisy = torch.from_numpy(noisy.astype(np.float32))
        
        return {
            "clean": clean.unsqueeze(0),
            "noisy": noisy.unsqueeze(0),
            "snr_db": torch.tensor(sample["snr_db"], dtype=torch.float32),
            "snr_name": sample["snr_name"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 训练函数 - 半版本
# ═══════════════════════════════════════════════════════════════════════════════

def train_half_version(
    data_dir: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """
    快速验证：固定损失权重，无高级组件
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    
    print("[Phase13-Half] 加载数据...")
    dataset = UniformSNRDataset(data_dir, sr=16000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print("[Phase13-Half] 初始化模型...")
    model = Phase13ShallowModel(d_model=48, n_layers=3, half_version=True).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n_fft, hop = 512, 128
    best_loss = float("inf")
    
    print(f"[Phase13-Half] 训练 {epochs} 个 epoch, dataset={len(dataset)} samples\n")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            clean = batch["clean"].to(device)
            noisy = batch["noisy"].to(device)
            
            noisy_stft = stft(noisy.squeeze(1), n_fft=n_fft, hop=hop)
            clean_stft = stft(clean.squeeze(1), n_fft=n_fft, hop=hop)
            
            optimizer.zero_grad()
            
            # 前向
            outputs = model(noisy_stft)
            enhanced_stft = outputs["S_enhanced"]
            enhanced_wav = istft(enhanced_stft, length=noisy.shape[-1], n_fft=n_fft, hop=hop)
            
            # 简单损失（半版本）
            recon_loss = F.l1_loss(enhanced_wav.unsqueeze(1), clean)
            mrstft_l = mrstft_loss(enhanced_wav, clean.squeeze(1))
            phase_loss = F.mse_loss(torch.angle(enhanced_stft), torch.angle(clean_stft))
            
            total_loss = recon_loss + 0.2 * mrstft_l + 0.1 * phase_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            batch_count += 1
            
            if (batch_idx + 1) % max(1, len(dataloader) // 2) == 0:
                avg_loss = epoch_loss / batch_count
                print(
                    f"[E{epoch}] Batch {batch_idx+1:3d}/{len(dataloader)} "
                    f"| Loss={avg_loss:.6f} | Recon={recon_loss:.6f}"
                )
        
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / batch_count
        print(f"\n[E{epoch}] 完成 - 平均损失={avg_epoch_loss:.6f}\n")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            ckpt_path = os.path.join(output_dir, "ckpt", "best.pt")
            Path(os.path.dirname(ckpt_path)).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ 保存最佳模型: {ckpt_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 评估函数
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    diff = est - ref
    return float(10.0 * np.log10(np.sum(ref ** 2) / (np.sum(diff ** 2) + 1e-10) + 1e-10))


def evaluate_model(
    model,
    data_dir: str,
    output_dir: str,
    n_samples: int = 50,
    device: str = "cuda",
):
    """简单评估：计算整体 SDR"""
    
    dataset = UniformSNRDataset(data_dir)
    n_samples = min(n_samples, len(dataset))
    
    model.eval()
    device = torch.device(device)
    
    sdr_scores = []
    
    with torch.no_grad():
        for i in range(n_samples):
            batch = dataset[i]
            clean = batch["clean"].to(device)
            noisy = batch["noisy"].to(device)
            
            noisy_stft = stft(noisy.squeeze(1), n_fft=512, hop=128)
            outputs = model(noisy_stft)
            enhanced_stft = outputs["S_enhanced"]
            enhanced_wav = istft(enhanced_stft, length=noisy.shape[-1], n_fft=512, hop=128)
            
            sdr = compute_sdr(clean.squeeze().cpu().numpy(), enhanced_wav.cpu().numpy())
            sdr_scores.append(sdr)
    
    mean_sdr = np.mean(sdr_scores)
    print(f"[Phase13-Half] 评估完成: 平均 SDR = {mean_sdr:.4f} dB")
    
    return {"mean_sdr": mean_sdr, "scores": sdr_scores}


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser("Phase13 浅层交互实验")
    parser.add_argument("--data-dir", type=str, required=True, help="数据目录")
    parser.add_argument("--output-dir", type=str, default="outputs/phase13", help="输出目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=8, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--eval-only", action="store_true", help="仅评估模式")
    parser.add_argument("--ckpt", type=str, default="", help="检查点路径")
    
    args = parser.parse_args()
    
    if args.eval_only and args.ckpt:
        print("[Phase13] 进入评估模式...")
        model = Phase13ShallowModel(d_model=48, n_layers=3, half_version=True)
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state)
        evaluate_model(model, args.data_dir, args.output_dir, device=args.device)
    else:
        print("[Phase13] 开始训练（半版本）...")
        train_half_version(
            args.data_dir,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
        print("[Phase13] 训练完成！")


if __name__ == "__main__":
    main()
