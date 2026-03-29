"""
Phase12: Discrete-Domain Magnitude-Phase Shallow Interaction Experiment (Half-Version)
======================================================================================

实验2（递进）：浅层交互机制的快速验证
- 基于 ADDSE 离散域（STFT 时频域）
- 极坐标分解 + 双分支并联 + 乘性门控交互
- 对标 MPICM，参数量从 50K 削减至 8K
- 预期 ΔSDR: +1.2 ~ 1.4 dB

关键设计：
  1. MagnitudeBranch: 对数幅度谱 → 重建能量显著性
  2. PhaseBranch: cos/sin 相位表示 → 相位一致性  
  3. ShallowCrossAttention: 通道级乘性门控交互
  4. 简化损失函数（L1 重建 + 固定权重 MR-STFT）
  
目标数据：outputs/phase11/uniform_snr_330 (330 样本，11 个 SNR 桶)
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import MelScale

# ============================================================================
# 日志配置
# ============================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ============================================================================
# 数据加载与预处理
# ============================================================================

class DiscretePhase12Dataset(Dataset):
    """
    加载 Phase11 生成的均匀 SNR 数据集
    
    数据结构：
    outputs/phase11/uniform_snr_330/
    ├── snr_-10/ → clean/, noisy/
    ├── snr_-8/  → clean/, noisy/
    ├── ...
    └── snr_10/  → clean/, noisy/
    """
    
    def __init__(self, data_dir, n_fft=512, hop_length=128):
        self.data_dir = Path(data_dir)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = 16000
        
        # 扫描数据集
        self.samples = []
        snr_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for snr_dir in snr_dirs:
            clean_dir = snr_dir / "clean"
            noisy_dir = snr_dir / "noisy"
            
            if clean_dir.exists() and noisy_dir.exists():
                clean_files = sorted(clean_dir.glob("*.wav"))
                for clean_file in clean_files:
                    noisy_file = noisy_dir / clean_file.name
                    if noisy_file.exists():
                        self.samples.append({
                            "clean": str(clean_file),
                            "noisy": str(noisy_file),
                            "snr_db": float(snr_dir.name.split("_")[-1])
                        })
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载音频
        clean, sr = torchaudio.load(sample["clean"])
        noisy, _ = torchaudio.load(sample["noisy"])
        
        assert sr == self.sample_rate, f"Sample rate mismatch: {sr} vs {self.sample_rate}"
        
        # 强制转换为单通道
        if clean.shape[0] > 1:
            clean = clean.mean(dim=0, keepdim=True)
        if noisy.shape[0] > 1:
            noisy = noisy.mean(dim=0, keepdim=True)
        
        # STFT 变换
        clean_stft = torch.stft(
            clean.squeeze(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )  # [F, T]
        
        noisy_stft = torch.stft(
            noisy.squeeze(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )  # [F, T]
        
        return {
            "clean_stft": clean_stft,
            "noisy_stft": noisy_stft,
            "snr_db": torch.tensor(sample["snr_db"], dtype=torch.float32)
        }


def collate_phase12_batch(batch):
    """
    将不同长度的 STFT 补零对齐到批次中最长的
    """
    max_time = max([item["clean_stft"].shape[1] for item in batch])
    
    batch_data = {
        "clean_stft": [],
        "noisy_stft": [],
        "snr_db": [],
        "masks": []  # 记录哪些时间帧是有效的
    }
    
    for item in batch:
        clean = item["clean_stft"]  # [F, T]
        noisy = item["noisy_stft"]  # [F, T]
        t_len = clean.shape[1]
        
        # 补零
        pad_len = max_time - t_len
        if pad_len > 0:
            clean = F.pad(clean, (0, pad_len))
            noisy = F.pad(noisy, (0, pad_len))
        
        batch_data["clean_stft"].append(clean)
        batch_data["noisy_stft"].append(noisy)
        batch_data["snr_db"].append(item["snr_db"])
        
        # 掩蔽：标记有效帧
        mask = torch.ones(max_time, device=clean.device)
        mask[t_len:] = 0
        batch_data["masks"].append(mask)
    
    batch_data["clean_stft"] = torch.stack(batch_data["clean_stft"])  # [B, F, T]
    batch_data["noisy_stft"] = torch.stack(batch_data["noisy_stft"])  # [B, F, T]
    batch_data["snr_db"] = torch.stack(batch_data["snr_db"])  # [B]
    batch_data["masks"] = torch.stack(batch_data["masks"])  # [B, T]
    
    return batch_data


# ============================================================================
# 模型模块
# ============================================================================

class MagnitudeBranch(nn.Module):
    """
    幅度分支：处理对数幅度谱
    
    输入：log(|STFT|) ∈ [0, log(max_mag)]，形状 [B, 1, F, T]
    输出：幅度特征 [B, d_model, F, T]
    
    设计理由：
    - 幅度是欧几里得空间，采用标准卷积
    - Dilated 卷积扩大时间感受野
    - 残差连接稳定梯度流
    """
    
    def __init__(self, d_model=48):
        super().__init__()
        self.d_model = d_model
        
        # 特征提取
        self.stem = nn.Conv2d(1, d_model // 2, kernel_size=3, padding=1)
        
        # 三层残差块（捕捉跨频率和跨时间的连续性）
        self.blocks = nn.ModuleList([
            self._make_dilated_block(d_model // 2, d_model // 2, dilation=1),
            self._make_dilated_block(d_model // 2, d_model // 2, dilation=2),
            self._make_dilated_block(d_model // 2, d_model, dilation=1),
        ])
    
    @staticmethod
    def _make_dilated_block(in_channels, out_channels, dilation=1):
        """构建 Dilated 卷积残差块"""
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=dilation, dilation=dilation
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=dilation, dilation=dilation
            ),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        """
        x: [B, 1, F, T] (对数幅度谱)
        return: [B, d_model, F, T]
        """
        feat = self.stem(x)  # [B, d_model//2, F, T]
        
        for block in self.blocks:
            feat_skip = feat if feat.shape[1] == block[0].in_channels else feat
            feat = block(feat) + F.pad(feat_skip, (0, 0, 0, 0, 0, 
                                                    block[0].out_channels - block[0].in_channels))
        
        return feat  # [B, d_model, F, T]


class PhaseBranch(nn.Module):
    """
    相位分支：处理三角函数相位表示
    
    输入：[cos(phase), sin(phase)] ∈ (-1, 1)，形状 [B, 2, F, T]
    输出：相位特征 [B, d_model, F, T]
    
    设计理由：
    - 相位是圆环流形（S^1），三角函数表示保证周期性和连续性
    - 与幅度分支相同的架构，便于对称交互
    - Dilated 卷积捕捉谐波的长程相关性（相邻谐波的相位约束）
    """
    
    def __init__(self, d_model=48):
        super().__init__()
        self.d_model = d_model
        
        # 特征提取（输入2通道：cos 和 sin）
        self.stem = nn.Conv2d(2, d_model // 2, kernel_size=3, padding=1)
        
        # 三层残差块
        self.blocks = nn.ModuleList([
            self._make_dilated_block(d_model // 2, d_model // 2, dilation=1),
            self._make_dilated_block(d_model // 2, d_model // 2, dilation=2),
            self._make_dilated_block(d_model // 2, d_model, dilation=1),
        ])
    
    @staticmethod
    def _make_dilated_block(in_channels, out_channels, dilation=1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=dilation, dilation=dilation
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=dilation, dilation=dilation
            ),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        """
        x: [B, 2, F, T] (cos/sin 相位表示)
        return: [B, d_model, F, T]
        """
        feat = self.stem(x)  # [B, d_model//2, F, T]
        
        for block in self.blocks:
            feat_skip = feat if feat.shape[1] == block[0].in_channels else feat
            feat = block(feat) + F.pad(feat_skip, (0, 0, 0, 0, 0, 
                                                    block[0].out_channels - block[0].in_channels))
        
        return feat  # [B, d_model, F, T]


class ShallowCrossAttention(nn.Module):
    """
    核心创新：通道级乘性门控交互
    
    数学形式：
      H'_pha = H_pha ⊙ σ(W_mag H_mag + b_mag) = H_pha ⊙ (1 + α*gate)
      H'_mag = H_mag ⊙ σ(W_pha H_pha + b_pha) = H_mag ⊙ (1 + β*gate)
    
    其中：
    - ⊙ 表示逐元素相乘（通道维度）
    - σ(·) 是 Sigmoid 激活
    - (1 + α*gate) 是残差形式，防止梯度消失
    - W_mag, W_pha 是线性变换（FC）
    
    参数量：
    - 每个分路：d_model² ≈ (48 × 48) = 2,304 参数
    - 总计：约 4,608 参数（还有偏置）
    """
    
    def __init__(self, d_model=48, interaction_strength=0.1):
        super().__init__()
        self.d_model = d_model
        self.alpha = interaction_strength  # 交互强度（半版本中固定）
        
        # 相位驱动的幅度门（通道级）
        # 输入：相位特征通道数，输出：幅度特征通道数
        self.phase_to_mag_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        
        # 幅度驱动的相位门（通道级）
        self.mag_to_phase_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, feat_mag, feat_phase):
        """
        feat_mag: [B, d_model, F, T]
        feat_phase: [B, d_model, F, T]
        
        return:
            feat_mag_interact: [B, d_model, F, T]
            feat_phase_interact: [B, d_model, F, T]
        """
        B, d, F, T = feat_mag.shape
        
        # 全局汇聚（通道维度的平均）
        mag_global = feat_mag.mean(dim=(2, 3))  # [B, d_model]
        phase_global = feat_phase.mean(dim=(2, 3))  # [B, d_model]
        
        # 生成交互权重
        mag_gate = torch.sigmoid(self.mag_to_phase_gate(phase_global))  # [B, d_model]
        phase_gate = torch.sigmoid(self.phase_to_mag_gate(mag_global))  # [B, d_model]
        
        # 应用乘性调制（残差形式）
        mag_gate = mag_gate.view(B, d, 1, 1)  # [B, d_model, 1, 1]
        phase_gate = phase_gate.view(B, d, 1, 1)  # [B, d_model, 1, 1]
        
        feat_mag_interact = feat_mag * (1.0 + self.alpha * mag_gate)
        feat_phase_interact = feat_phase * (1.0 + self.alpha * phase_gate)
        
        return feat_mag_interact, feat_phase_interact


class MagnitudeDecoder(nn.Module):
    """
    幅度解码器：将幅度特征转换为掩蔽函数
    
    输入：幅度特征 [B, d_model, F, T]
    输出：幅度掩蔽 [B, 1, F, T] ∈ [0, 1]
    """
    
    def __init__(self, d_model=48):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 掩蔽约束在 [0, 1]
        )
    
    def forward(self, feat):
        """
        feat: [B, d_model, F, T]
        return: mask [B, 1, F, T]
        """
        return self.decoder(feat)


class PhaseDecoder(nn.Module):
    """
    相位解码器：将相位特征转换为相位残差修正
    
    输入：相位特征 [B, d_model, F, T]
    输出：相位残差 [B, 2, F, T]（cos/sin 形式）
    
    相位残差表示为 cos/sin 便于学习圆环约束
    """
    
    def __init__(self, d_model=48):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, 2, kernel_size=3, padding=1),
            nn.Tanh()  # 相位残差输出到 [-1, 1]（隐式 [-π, π]）
        )
    
    def forward(self, feat):
        """
        feat: [B, d_model, F, T]
        return: phase_residual [B, 2, F, T]
        """
        return self.decoder(feat)


class Phase12ShallowModel(nn.Module):
    """
    完整模型：极坐标分解 + 双分支并联 + 浅层交互 + 融合输出
    
    架构流程：
    1. 极坐标分解：复数 STFT → 幅度（log 压缩）+ 相位（cos/sin）
    2. 幅度分支编码 → 幅度特征 [B, d, F, T]
    3. 相位分支编码 → 相位特征 [B, d, F, T]
    4. 浅层交互：乘性门控调制两个特征
    5. 幅度解码 → 掩蔽函数 [B, 1, F, T]
    6. 相位解码 → 相位残差 [B, 2, F, T]
    7. 融合重建：带掩蔽的幅度 + 调整后的相位 + iSTFT
    """
    
    def __init__(self, d_model=48, interaction_strength=0.1, n_fft=512, hop_length=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # 编码器
        self.mag_encoder = MagnitudeBranch(d_model)
        self.phase_encoder = PhaseBranch(d_model)
        
        # 浅层交互
        self.interaction = ShallowCrossAttention(d_model, interaction_strength)
        
        # 解码器
        self.mag_decoder = MagnitudeDecoder(d_model)
        self.phase_decoder = PhaseDecoder(d_model)
    
    def forward(self, noisy_stft):
        """
        noisy_stft: [B, F, T] (复数)
        return: enhanced_stft [B, F, T] (复数复制)
        """
        B, F, T = noisy_stft.shape
        
        # ========== 步骤 1: 极坐标分解 ==========
        mag_noisy = torch.abs(noisy_stft)  # [B, F, T]
        phase_noisy = torch.angle(noisy_stft)  # [B, F, T]
        
        # 幅度：对数压缩到 [0, log(max_mag)) 范围
        mag_log = torch.log1p(mag_noisy)  # log(1 + mag)
        mag_log = mag_log.unsqueeze(1)  # [B, 1, F, T]
        
        # 相位：转换为三角函数表示 [B, 2, F, T]
        phase_cos = torch.cos(phase_noisy).unsqueeze(1)  # [B, 1, F, T]
        phase_sin = torch.sin(phase_noisy).unsqueeze(1)  # [B, 1, F, T]
        phase_trig = torch.cat([phase_cos, phase_sin], dim=1)  # [B, 2, F, T]
        
        # ========== 步骤 2-3: 双分支编码 ==========
        feat_mag = self.mag_encoder(mag_log)  # [B, d_model, F, T]
        feat_phase = self.phase_encoder(phase_trig)  # [B, d_model, F, T]
        
        # ========== 步骤 4: 浅层交互 ==========
        feat_mag_interact, feat_phase_interact = self.interaction(feat_mag, feat_phase)
        
        # ========== 步骤 5-6: 解码 ==========
        mag_mask = self.mag_decoder(feat_mag_interact)  # [B, 1, F, T]
        phase_residual = self.phase_decoder(feat_phase_interact)  # [B, 2, F, T]
        
        # ========== 步骤 7: 融合输出 ==========
        # 幅度融合：原幅度 × 掩蔽函数
        mag_enhanced = mag_noisy * mag_mask.squeeze(1)  # [B, F, T]
        
        # 相位融合：
        # phase_residual [B, 2, F, T] → 转换为弧度值
        # 简化版本：直接使用 residual 作为相位修正增量
        residual_angle = torch.atan2(phase_residual[:, 1, :, :], phase_residual[:, 0, :, :])  # [B, F, T]
        phase_enhanced = phase_noisy + 0.1 * residual_angle  # 弱修正，系数 0.1
        
        # 转换回复数形式
        enhanced_stft = mag_enhanced * torch.exp(1j * phase_enhanced)
        
        return enhanced_stft


# ============================================================================
# 损失函数
# ============================================================================

def compute_mrstft_loss(pred_stft, target_stft, n_ffts=[256, 512, 1024], hop_lengths=None):
    """
    多分辨率 STFT 损失
    
    计算多个尺度上的幅度和相位误差
    """
    if hop_lengths is None:
        hop_lengths = [n // 4 for n in n_ffts]
    
    loss = 0.0
    
    for n_fft, hop_length in zip(n_ffts, hop_lengths):
        # 提取幅度和相位
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        pred_phase = torch.angle(pred_stft)
        target_phase = torch.angle(target_stft)
        
        # 对数幅度误差（避免小幅度的主导）
        mag_diff = torch.log1p(pred_mag) - torch.log1p(target_mag)
        loss_mag = torch.mean(torch.abs(mag_diff))
        
        # 相位一致性（使用三角函数距离）
        phase_diff = torch.sin((pred_phase - target_phase) / 2.0)
        loss_phase = torch.mean(torch.abs(phase_diff))
        
        loss += loss_mag + 0.1 * loss_phase  # 相位权重较低
    
    return loss / len(n_ffts)


def compute_time_domain_loss(pred_wav, target_wav):
    """
    时域 L1 重建损失
    """
    return torch.mean(torch.abs(pred_wav - target_wav))


# ============================================================================
# 训练循环
# ============================================================================

def train_phase12(args):
    """主训练函数"""
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据加载
    logger.info(f"Loading dataset from {args.data_dir}")
    dataset = DiscretePhase12Dataset(args.data_dir, n_fft=512, hop_length=128)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_phase12_batch,
        num_workers=0
    )
    
    # 模型
    model = Phase12ShallowModel(d_model=48, interaction_strength=0.1).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            clean_stft = batch["clean_stft"].to(device)  # [B, F, T]
            noisy_stft = batch["noisy_stft"].to(device)  # [B, F, T]
            masks = batch["masks"].to(device)  # [B, T]
            
            # 前向传播
            enhanced_stft = model(noisy_stft)  # [B, F, T]
            
            # 损失计算
            loss_mrstft = compute_mrstft_loss(enhanced_stft, clean_stft)
            
            # ISTFT 转换到时域
            enhanced_wav = torch.istft(
                enhanced_stft,
                n_fft=512,
                hop_length=128,
                return_complex=False
            )
            clean_wav = torch.istft(
                clean_stft,
                n_fft=512,
                hop_length=128,
                return_complex=False
            )
            
            # 应用时间掩蔽（去除补零部分）
            T_wav = enhanced_wav.shape[-1]
            masks_wav = F.interpolate(
                masks.unsqueeze(1),
                size=T_wav,
                mode='nearest'
            ).squeeze(1)  # [B, T_wav]
            
            enhanced_wav = enhanced_wav[:, :T_wav] * masks_wav
            clean_wav = clean_wav[:, :T_wav] * masks_wav
            
            loss_time = compute_time_domain_loss(enhanced_wav, clean_wav)
            
            # 总损失
            loss = loss_time + 0.2 * loss_mrstft
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(dataloader)}: "
                    f"Loss={loss.item():.4f}, L_time={loss_time.item():.4f}, "
                    f"L_mrstft={loss_mrstft.item():.4f}"
                )
        
        # 每个 epoch 的平均损失
        epoch_loss /= len(dataloader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {epoch_loss:.4f}")
        
        scheduler.step()
        
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_loss
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
    
    logger.info("Training finished!")
    return model, output_dir


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase12 Discrete Magnitude-Phase Shallow Interaction")
    parser.add_argument("--data-dir", type=str, default="outputs/phase11/uniform_snr_330",
                        help="Path to Phase11 dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/phase12_half",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Phase12: Discrete-Domain Shallow Interaction (Half-Version)")
    logger.info("=" * 80)
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    
    train_phase12(args)


if __name__ == "__main__":
    main()
