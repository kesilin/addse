"""
均匀SNR混合脚本：
- 从controlled_snr_test31的干净数据出发
- 使用ED_BASE的噪声库
- 生成SNR均匀分布在 -10 ~ 10 dB 的数据集
- 分为 11 个 SNR bucket: [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
- 每个bucket产生 8-10 个样本（共~100样本）
"""

import os
import glob
import random
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Dict
import argparse
import json
from datetime import datetime


def load_wav(path: str, sr: int = 16000) -> np.ndarray:
    """加载wav文件，自动重采样到目标采样率"""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def normalize_audio(x: np.ndarray) -> np.ndarray:
    """归一化音频到 [-1, 1] 范围"""
    max_val = np.max(np.abs(x))
    if max_val > 0:
        x = x / max_val
    return x.astype(np.float32)


def crop_or_pad(x: np.ndarray, target_len: int, random_crop: bool = True) -> np.ndarray:
    """裁剪或补零到目标长度"""
    if len(x) > target_len:
        if random_crop:
            start = random.randint(0, len(x) - target_len)
        else:
            start = 0
        return x[start:start + target_len]
    elif len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)))
    return x


def mix_clean_and_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float, sr: int = 16000) -> np.ndarray:
    """按目标SNR混合干净音频和噪声"""
    # 确保长度匹配
    clean_len = len(clean)
    noise_len = len(noise)
    
    # 噪声不足则循环拼接
    if noise_len < clean_len:
        num_reps = (clean_len // noise_len) + 1
        noise = np.tile(noise, num_reps)
    
    # 从噪声中随机截取一段
    start = random.randint(0, len(noise) - clean_len)
    noise_seg = noise[start:start + clean_len]
    
    # 计算SNR并混合
    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise_seg ** 2) + 1e-10
    
    # SNR = 10 * log10(P_clean / P_noise)
    # => P_noise_target = P_clean / 10^(SNR/10)
    snr_linear = 10 ** (snr_db / 10)
    noise_target_power = clean_power / snr_linear
    noise_scale = np.sqrt(noise_target_power / noise_power)
    
    noisy = clean + noise_scale * noise_seg
    
    # 防止过度裁剪
    max_val = np.max(np.abs(noisy))
    if max_val > 1.0:
        noisy = noisy / max_val
    
    return noisy.astype(np.float32)


def build_uniform_snr_dataset(
    clean_dir: str,
    noise_dir: str,
    output_dir: str,
    snr_range: Tuple[float, float] = (-10.0, 10.0),
    num_snr_buckets: int = 11,
    samples_per_bucket: int = 8,
    segment_length: float = 1.0,  # 秒
    sr: int = 16000,
) -> Dict:
    """
    构建均匀SNR分布数据集
    
    Args:
        clean_dir: 干净音频目录路径
        noise_dir: 噪声文件目录
        output_dir: 输出目录
        snr_range: SNR范围 (min, max)
        num_snr_buckets: SNR bucket数量
        samples_per_bucket: 每个bucket的样本数
        segment_length: 每个样本的长度（秒）
        sr: 采样率
    """
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 收集干净音频和噪声
    print("[UniformSNR] 收集干净音频...")
    clean_files = glob.glob(os.path.join(clean_dir, "**/*.wav"), recursive=True)
    clean_files += glob.glob(os.path.join(clean_dir, "**/*.WAV"), recursive=True)
    print(f"[UniformSNR] 找到 {len(clean_files)} 个干净音频")
    
    print("[UniformSNR] 收集噪声...")
    noise_files = glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)
    noise_files += glob.glob(os.path.join(noise_dir, "**/*.WAV"), recursive=True)
    print(f"[UniformSNR] 找到 {len(noise_files)} 个噪声文件")
    
    if not clean_files:
        raise ValueError(f"未找到干净音频文件: {clean_dir}")
    if not noise_files:
        raise ValueError(f"未找到噪声文件: {noise_dir}")
    
    # 生成SNR分布
    snr_db_list = np.linspace(snr_range[0], snr_range[1], num_snr_buckets).tolist()
    target_len = int(segment_length * sr)
    
    print(f"[UniformSNR] SNR buckets: {[f'{s:.1f}' for s in snr_db_list]}")
    print(f"[UniformSNR] 每个bucket {samples_per_bucket} 个样本")
    print(f"[UniformSNR] 总样本数: {len(snr_db_list) * samples_per_bucket}")
    
    # 预加载噪声（加速）
    print("[UniformSNR] 预加载噪声...")
    noises = {}
    for nf in noise_files:
        noise_name = Path(nf).stem
        try:
            noises[noise_name] = load_wav(nf, sr=sr)
            if (len(noises) % 5 == 0):
                print(f"  加载进度: {len(noises)}/{len(noise_files)}")
        except Exception as e:
            print(f"[警告] 加载噪声失败 {nf}: {e}")
    
    print(f"[UniformSNR] 成功加载 {len(noises)} 个噪声")
    
    # 生成混合数据
    metadata = {
        "snr_buckets": snr_db_list,
        "samples_per_bucket": samples_per_bucket,
        "total_samples": len(snr_db_list) * samples_per_bucket,
        "segment_length_sec": segment_length,
        "sr": sr,
        "created_at": datetime.now().isoformat(),
        "samples": []
    }
    
    total_count = 0
    
    for bucket_idx, snr_db in enumerate(snr_db_list):
        bucket_dir = os.path.join(output_dir, f"snr_{snr_db:.1f}")
        clean_out = os.path.join(bucket_dir, "clean")
        noisy_out = os.path.join(bucket_dir, "noisy")
        
        Path(clean_out).mkdir(parents=True, exist_ok=True)
        Path(noisy_out).mkdir(parents=True, exist_ok=True)
        
        print(f"\n[UniformSNR] 处理 SNR={snr_db:.1f} dB ...")
        
        for sample_idx in range(samples_per_bucket):
            # 随机选择干净音频和噪声
            clean_file = random.choice(clean_files)
            noise_name = random.choice(list(noises.keys()))
            noise_wav = noises[noise_name]
            
            try:
                # 加载干净音频
                clean = load_wav(clean_file, sr=sr)
                clean = crop_or_pad(clean, target_len, random_crop=True)
                clean = normalize_audio(clean)
                
                # 混合
                noisy = mix_clean_and_noise(clean, noise_wav, snr_db, sr=sr)
                noisy = normalize_audio(noisy)
                
                # 保存
                sample_name = f"sample_{bucket_idx:02d}_{sample_idx:02d}"
                clean_path = os.path.join(clean_out, f"{sample_name}.wav")
                noisy_path = os.path.join(noisy_out, f"{sample_name}.wav")
                
                sf.write(clean_path, clean, sr)
                sf.write(noisy_path, noisy, sr)
                
                metadata["samples"].append({
                    "name": sample_name,
                    "snr_db": snr_db,
                    "clean_path": clean_path,
                    "noisy_path": noisy_path,
                    "source_clean": Path(clean_file).name,
                    "source_noise": noise_name,
                })
                
                total_count += 1
                if (sample_idx + 1) % 3 == 0 or sample_idx == samples_per_bucket - 1:
                    print(f"  [{bucket_idx+1}/{len(snr_db_list)}] 已生成 {sample_idx + 1}/{samples_per_bucket} 个样本")
            
            except Exception as e:
                print(f"[错误] 生成样本失败 SNR={snr_db:.1f} idx={sample_idx}: {e}")
    
    # 保存metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n[UniformSNR] ✓ 完成！")
    print(f"[UniformSNR] 生成了 {total_count} 个样本")
    print(f"[UniformSNR] 输出目录: {output_dir}")
    print(f"[UniformSNR] Metadata: {meta_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser("均匀SNR数据混合脚本")
    parser.add_argument("--clean-dir", type=str, default="./outputs/phase6/controlled_snr_test31/snr_0_5/clean",
                        help="干净音频目录")
    parser.add_argument("--noise-dir", type=str, default="../ED_BASE/processed_16kHz",
                        help="噪声文件目录")
    parser.add_argument("--output-dir", type=str, default="./outputs/phase11/uniform_snr_minus10_plus10",
                        help="输出目录")
    parser.add_argument("--snr-min", type=float, default=-10.0, help="最小SNR (dB)")
    parser.add_argument("--snr-max", type=float, default=10.0, help="最大SNR (dB)")
    parser.add_argument("--num-buckets", type=int, default=11, help="SNR bucket数量")
    parser.add_argument("--samples-per-bucket", type=int, default=8, help="每个bucket的样本数")
    parser.add_argument("--segment-length", type=float, default=1.0, help="分段长度（秒）")
    parser.add_argument("--sr", type=int, default=16000, help="采样率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    build_uniform_snr_dataset(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        output_dir=args.output_dir,
        snr_range=(args.snr_min, args.snr_max),
        num_snr_buckets=args.num_buckets,
        samples_per_bucket=args.samples_per_bucket,
        segment_length=args.segment_length,
        sr=args.sr,
    )


if __name__ == "__main__":
    main()
