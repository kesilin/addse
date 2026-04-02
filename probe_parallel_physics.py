#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADDSE 并联架构物理特性诊断脚本

目标：解开"并联能否Work"的生死谜题
通过三个关键实验判断冻结解码器是否支持隐空间加法。
"""

import os
import sys
import io

# 设置UTF-8编码
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import torch
import torchaudio
from pathlib import Path

from addse.lightning import load_nac
from addse.metrics import PESQMetric, STOIMetric, SDRMetric
from hydra.utils import instantiate
import yaml

@torch.no_grad()
def run_physics_probe(
    nac_cfg: str = "configs/nac.yaml",
    nac_ckpt: str = "logs/addse-edbase-quick/checkpoints/addse-s.ckpt",
    output_dir: str = "probe_outputs",
    clean_wav: str = "saved_audio_v33/edbase-local_000000_clean.wav",
) -> dict:
    """执行隐空间并联物理特性探测。
    
    Args:
        nac_cfg: NAC配置文件路径
        nac_ckpt: NAC检查点路径（或ADDSE模型路径）
        output_dir: 输出目录
        
    Returns:
        诊断结果字典
    """
    print("=" * 80)
    print("[START] 开始执行隐空间并联物理特性探测")
    print("=" * 80)
    
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # ===== 1. 加载冻结的编解码器 =====
    print("\n[1/5] 加载 NAC Codec...")
    try:
        # 尝试先加载ADDSE完整模型，从中提取NAC
        model_path = "logs/addse-s-edbase-parallel60-a008-p02-spec/checkpoints/last.ckpt"
        if os.path.exists(model_path):
            print(f"  从ADDSE模型 {model_path} 加载NAC...")
            ckpt = torch.load(model_path, map_location=device)
            
            # 首先加载nac配置
            with open(nac_cfg) as f:
                cfg = yaml.safe_load(f)
            nac = instantiate(cfg["lm"]["generator"])
            nac = nac.to(device)
            
            # 从ADDSE checkpoint中提取NAC的state_dict
            nac_state_dict = {}
            for k, v in ckpt["state_dict"].items():
                if k.startswith("nac."):
                    # 移除 "nac." 前缀
                    new_k = k[4:]
                    nac_state_dict[new_k] = v
            
            if nac_state_dict:
                nac.load_state_dict(nac_state_dict, strict=False)
                print(f"[OK] 从ADDSE模型提取NAC成功 (加载 {len(nac_state_dict)} 个参数)")
            else:
                print("  未找到NAC参数，尝试直接加载NAC检查点...")
                nac, mask_token = load_nac(nac_cfg, nac_ckpt)
        else:
            nac, mask_token = load_nac(nac_cfg, nac_ckpt)
        
        nac = nac.to(device)
        nac.eval()
        for param in nac.parameters():
            param.requires_grad = False
        print(f"[OK] NAC 加载成功")
        print(f"  - Downsampling Factor: {nac.downsampling_factor}")
    except Exception as e:
        print(f"[FAIL] NAC 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
    # ===== 2. 准备可复现的干净语音样本 =====
    print("\n[2/5] 准备测试音频...")
    test_samples = []

    if os.path.exists(clean_wav):
        wav, sr = torchaudio.load(clean_wav)
        test_samples.append({"wav": wav, "sr": sr, "path": clean_wav})
        print(f"[OK] 使用指定 clean 样本: {clean_wav} (sr={sr}, len={wav.shape[-1]})")
    else:
        print(f"  指定样本不存在，回退到自动搜索: {clean_wav}")
        for wav_file in Path("saved_audio_v33").rglob("*_clean.wav"):
            wav, sr = torchaudio.load(str(wav_file))
            test_samples.append({"wav": wav, "sr": sr, "path": str(wav_file)})
            print(f"[OK] 自动选择 clean 样本: {wav_file.name} (sr={sr}, len={wav.shape[-1]})")
            break

    if len(test_samples) == 0:
        print("  未找到 clean 语音，生成合成数据...")
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # 干净信号: 440 Hz + 880 Hz
        clean = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        clean = clean / (np.max(np.abs(clean)) + 1e-8)  # 归一化
        
        test_samples.append({
            'wav': torch.from_numpy(clean).float().unsqueeze(0),
            'sr': sr,
            'path': '<synthetic>'
        })
        print(f"[OK] 生成合成音频 (sr={sr}, duration={duration}s)")
    
    test_wav = test_samples[0]['wav'].to(device)
    sr = test_samples[0]['sr']

    # 强制形状为 (B, C, T)
    if test_wav.ndim == 1:
        test_wav = test_wav.unsqueeze(0).unsqueeze(0)
    elif test_wav.ndim == 2:
        test_wav = test_wav.unsqueeze(0)
    if test_wav.shape[1] > 1:
        test_wav = test_wav[:, :1, :]
    print(f"[OK] 输入波形形状: {tuple(test_wav.shape)}")
    
    # 初始化指标计算器
    pesq_metric = PESQMetric(fs=sr)
    stoi_metric = STOIMetric(fs=sr, extended=True)
    sdr_metric = SDRMetric(scale_invariant=False, zero_mean=False)
    si_sdr_metric = SDRMetric(scale_invariant=True, zero_mean=True)
    
    # ===== 3. 编码并提取关键特征 =====
    print("\n[3/5] 提取 Latent 特征与离散 Token...")
    
    # 填充到采样因子对齐
    n_pad = (nac.downsampling_factor - test_wav.shape[-1]) % nac.downsampling_factor
    test_wav_pad = torch.nn.functional.pad(test_wav, (0, n_pad))
    
    try:
        # 获取干净的连续隐特征
        z_clean_lat = nac.encoder(test_wav_pad)
        print(f"✓ 干净隐特征: shape={z_clean_lat.shape}, dtype={z_clean_lat.dtype}")
        
        # 获取量化后的离散特征 (带 no_sum=True 保留多codebook)
        _, z_clean_q = nac.encode(test_wav_pad, no_sum=True, domain="q")
        print(f"✓ 离散量化特征: shape={z_clean_q.shape}")
        
        # 计算离散求和特征：z_clean_q 形状为 (B, C, K, L)，应在 codebook 维 K 上求和。
        z_clean_q_sum = z_clean_q.sum(dim=2) if z_clean_q.ndim == 4 else z_clean_q
        print(f"✓ 离散求和特征: shape={z_clean_q_sum.shape}")
        
    except Exception as e:
        print(f"✗ 编码失败: {e}")
        return {"error": f"编码失败: {e}"}
    
    results = {}
    
    # ===== 实验 A: 离散天花板 =====
    print("\n[4/5] 执行三个关键实验...")
    print("\n  [A] 离散天花板 (纯骨架, 无残差)")
    try:
        wav_discrete_only = nac.decode(z_clean_q_sum, domain="q")
        wav_discrete_only = wav_discrete_only[..., :test_wav.shape[-1]]
        
        # 保存输出
        output_path_a = os.path.join(output_dir, "probe_A_discrete_only.wav")
        torchaudio.save(output_path_a, wav_discrete_only[0].cpu(), sr)
        
        # 计算指标 (指标实现期望输入为 (C, T))
        pesq_a = pesq_metric(wav_discrete_only[0], test_wav[0])
        estoi_a = stoi_metric(wav_discrete_only[0], test_wav[0])
        sdr_a = sdr_metric(wav_discrete_only[0], test_wav[0])
        si_sdr_a = si_sdr_metric(wav_discrete_only[0], test_wav[0])
        
        results['A'] = {
            'pesq': pesq_a,
            'estoi': estoi_a,
            'sdr': sdr_a,
            'si_sdr': si_sdr_a,
            'path': output_path_a,
            'description': '纯离散解码 (基线)'
        }
        
        print(f"  ✓ PESQ={pesq_a:.3f}, ESTOI={estoi_a:.3f}, SDR={sdr_a:.3f}, SI-SDR={si_sdr_a:.3f}")
        print(f"    💾 保存到: {output_path_a}")
        
    except Exception as e:
        print(f"  ✗ 实验A失败: {e}")
        results['A'] = {'error': str(e)}
    
    # ===== 实验 B: 神之残差 =====
    print("\n  [B] 神之残差 (离散 + 完美真实残差)")
    try:
        # 计算物理上丢失的真实残差
        oracle_residual = z_clean_lat - z_clean_q_sum
        print(f"    Oracle残差幅度: mean={oracle_residual.abs().mean():.6f}, max={oracle_residual.abs().max():.6f}")
        
        # 物理相加
        z_oracle_fused = z_clean_q_sum + oracle_residual
        
        wav_oracle_fused = nac.decode(z_oracle_fused, domain="q")
        wav_oracle_fused = wav_oracle_fused[..., :test_wav.shape[-1]]
        
        output_path_b = os.path.join(output_dir, "probe_B_oracle_fused.wav")
        torchaudio.save(output_path_b, wav_oracle_fused[0].cpu(), sr)
        
        # 计算指标 (指标实现期望输入为 (C, T))
        pesq_b = pesq_metric(wav_oracle_fused[0], test_wav[0])
        estoi_b = stoi_metric(wav_oracle_fused[0], test_wav[0])
        sdr_b = sdr_metric(wav_oracle_fused[0], test_wav[0])
        si_sdr_b = si_sdr_metric(wav_oracle_fused[0], test_wav[0])
        
        results['B'] = {
            'pesq': pesq_b,
            'estoi': estoi_b,
            'sdr': sdr_b,
            'si_sdr': si_sdr_b,
            'path': output_path_b,
            'description': '离散 + 真实残差 (神之上限)'
        }
        
        print(f"  ✓ PESQ={pesq_b:.3f}, ESTOI={estoi_b:.3f}, SDR={sdr_b:.3f}, SI-SDR={si_sdr_b:.3f}")
        print(f"    💾 保存到: {output_path_b}")
        
    except Exception as e:
        print(f"  ✗ 实验B失败: {e}")
        results['B'] = {'error': str(e)}
    
    # ===== 实验 C: 随机噪声破坏程度 =====
    print("\n  [C] 随机噪声破坏 (模拟未训练网络)")
    try:
        # 生成与oracle_residual同尺度的随机噪声
        scale = oracle_residual.std().item()
        random_residual = torch.randn_like(oracle_residual) * scale * 0.5
        
        z_random_fused = z_clean_q_sum + random_residual
        
        wav_random_fused = nac.decode(z_random_fused, domain="q")
        wav_random_fused = wav_random_fused[..., :test_wav.shape[-1]]
        
        output_path_c = os.path.join(output_dir, "probe_C_random_fused.wav")
        torchaudio.save(output_path_c, wav_random_fused[0].cpu(), sr)
        
        # 计算指标 (指标实现期望输入为 (C, T))
        pesq_c = pesq_metric(wav_random_fused[0], test_wav[0])
        estoi_c = stoi_metric(wav_random_fused[0], test_wav[0])
        sdr_c = sdr_metric(wav_random_fused[0], test_wav[0])
        si_sdr_c = si_sdr_metric(wav_random_fused[0], test_wav[0])
        
        results['C'] = {
            'pesq': pesq_c,
            'estoi': estoi_c,
            'sdr': sdr_c,
            'si_sdr': si_sdr_c,
            'path': output_path_c,
            'description': '离散 + 随机高斯残差 (噪声破坏)'
        }
        
        print(f"  ✓ PESQ={pesq_c:.3f}, ESTOI={estoi_c:.3f}, SDR={sdr_c:.3f}, SI-SDR={si_sdr_c:.3f}")
        print(f"    💾 保存到: {output_path_c}")
        
    except Exception as e:
        print(f"  ✗ 实验C失败: {e}")
        results['C'] = {'error': str(e)}
    
    # ===== 5. 诊断分析 =====
    print("\n[5/5] 诊断分析")
    print("=" * 80)
    
    if 'A' in results and 'error' not in results['A'] and \
       'B' in results and 'error' not in results['B'] and \
       'C' in results and 'error' not in results['C']:
        
        pesq_a = results['A']['pesq']
        pesq_b = results['B']['pesq']
        pesq_c = results['C']['pesq']
        
        si_sdr_a = results['A']['si_sdr']
        si_sdr_b = results['B']['si_sdr']
        si_sdr_c = results['C']['si_sdr']
        
        print(f"\n[METRICS] 结果对比:")
        print(f"  (+) A (纯离散):      PESQ={pesq_a:.3f}, SI-SDR={si_sdr_a:+.2f}")
        print(f"  (✓) B (神之残差):    PESQ={pesq_b:.3f}, SI-SDR={si_sdr_b:+.2f}  Δ={pesq_b-pesq_a:+.3f}")
        print(f"  (-) C (随机噪声):    PESQ={pesq_c:.3f}, SI-SDR={si_sdr_c:+.2f}  Δ={pesq_c-pesq_a:+.3f}")
        
        print(f"\n[DIAG] 关键诊断:")
        
        # 判断 B 是否接近完美
        if pesq_b < 3.5:
            print(f"\n  [CRITICAL] 神之残差依然效果差 (PESQ={pesq_b:.3f} < 3.5)")
            print(f"     这说明冻结解码器拒绝隐空间加法操作！")
            print(f"     [FAIL] 当前的纯隐空间并联架构不可行！")
            print(f"\n     建议方案:")
            print(f"     1. 尝试解冻解码器的最后几层并动态微调")
            print(f"     2. 转向波形级别的后处理 (Post-Net + Concat)")
            print(f"     3. 考虑使用其他特征空间 (e.g., STFTLoss)")
            results['diagnosis'] = 'FAILED_ARCHITECTURE_INCOMPATIBLE'
        else:
            print(f"\n  [PASS] 架构可行：神之残差效果极好 (PESQ={pesq_b:.3f})")
            print(f"     隐空间加法物理上是可行的！")
            print(f"     [INFO] 问题出在连续分支训练不足上！")
            print(f"\n     建议方案:")
            print(f"     1. 增加训练轮次 (当前只有5个epoch)")
            print(f"     2. 调整residual_l1_loss权重和学习对象")
            print(f"     3. 改进损失函数 (考虑SI-SDR而非L1)")
            results['diagnosis'] = 'PASSED_ARCHITECTURE_VIABLE'
        
        # C与A的差异表示噪声的破坏程度
        noise_impact = pesq_a - pesq_c
        print(f"\n  [NOISE] 噪声破坏程度: {noise_impact:.3f} PESQ点")
        if noise_impact > 0.5:
            print(f"     隐空间对噪声很敏感 (需要精准控制梯度)")
        
        results['noise_impact'] = noise_impact
    
    # ===== 保存完整结果 =====
    print(f"\n[DONE] 诊断完成！")
    print(f"输出文件位置: {os.path.abspath(output_dir)}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_physics_probe()
    
    # 打印诊断结果摘要
    print("\n" + "=" * 80)
    print("[SUMMARY] 诊断结果摘要")
    print("=" * 80)
    if 'diagnosis' in results:
        print(f"\n最终诊断: {results['diagnosis']}")
        if results['diagnosis'] == 'PASSED_ARCHITECTURE_VIABLE':
            print("\n[PASS] 并联架构本身是可行的！现在需要:")
            print("  1. 更长的训练周期")
            print("  2. 更好的损失函数设计")
            print("  3. 校正学习率和权重配置")
        else:
            print("\n[FAIL] 并联架构存在根本性问题，需要重新设计")
    
    print("\n实验结果:")
    for exp in ['A', 'B', 'C']:
        if exp in results and 'error' not in results[exp]:
            r = results[exp]
            print(f"\n{exp}. {r['description']}")
            print(f"   PESQ={r['pesq']:.3f}, ESTOI={r['estoi']:.3f}, SDR={r['sdr']:.3f}, SI-SDR={r['si_sdr']:+.2f}")
