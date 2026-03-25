#!/usr/bin/env python3
"""
E2-Stage2 跨桶推理脚本
评估冻结离散头后的 snr_5_10 微调模型在所有三个 SNR 桶上的表现
"""

import torch
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from phase9_nhfae_e1_interact import NHFAE_E1_Interact

device = 'cuda'
n_fft = 512
hop_length = 192

def istft_as_waveform(stft_matrix):
    """Convert STFT matrix back to waveform."""
    waveform = librosa.istft(stft_matrix, hop_length=hop_length, window='hann')
    return waveform

# 加载 Stage 2 checkpoint（snr_5_10微调后）
ckpt_path = Path('outputs/phase9/nhfae_e2_stage2_snr5_10/ckpt/best.pt')
pack = torch.load(ckpt_path, map_location=device)
model = NHFAE_E1_Interact(d_model=96, n_layers=8, n_heads=4, n_mag_bins=64)
model.load_state_dict(pack['model'])
model.eval()
model = model.to(device)

print(f"✅ Loaded E2-Stage2 checkpoint from {ckpt_path}\n")

def inference_on_bucket(bucket_name: str, noisy_dir_path: Path):
    """推理指定 SNR 桶中的所有音频。"""
    out_dir = Path(f'outputs/phase9/nhfae_e2_stage2/wav_{bucket_name}')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[{bucket_name}] 推理中...")
    count = 0
    with torch.no_grad():
        for wav_file in sorted(noisy_dir_path.glob('*.wav')):
            noisy, sr = librosa.load(str(wav_file), sr=16000)
            # 计算 STFT
            noisy_stft = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length, window='hann')
            noisy_stft_t = torch.from_numpy(noisy_stft).cfloat().to(device).unsqueeze(0)
            
            # 模型推理
            outputs = model(noisy_stft_t)
            mag_n = torch.abs(noisy_stft_t)
            mag_scale = torch.amax(mag_n, dim=(1, 2), keepdim=True) + 1e-8
            
            # ===== 修正：magnitude 缩放 =====
            mag_enhanced = outputs['mag_mix'] * mag_scale
            phase_enhanced = outputs['phase_out']
            
            # 重构 STFT
            enh_stft = mag_enhanced * torch.exp(1j * phase_enhanced)
            enh_stft_np = enh_stft.squeeze().cpu().numpy()
            
            # 转换回 waveform
            enh = istft_as_waveform(enh_stft_np)
            sf.write(out_dir / wav_file.name, enh, sr)
            count += 1
    
    print(f"[{bucket_name}] 完成: {count} files → {out_dir}\n")
    return out_dir

# 推理三个 SNR 桶
inference_on_bucket('snr_0_5', Path('outputs/phase6/controlled_snr_test31/snr_0_5/noisy'))
inference_on_bucket('snr_5_10', Path('outputs/phase6/controlled_snr_test31/snr_5_10/noisy'))
inference_on_bucket('snr_10_15', Path('outputs/phase6/controlled_snr_test31/snr_10_15/noisy'))

print("✅ E2-Stage2 全桶推理完成！")
