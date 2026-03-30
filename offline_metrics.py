import os
import glob
import soundfile as sf
from pesq import pesq
from pystoi import stoi
import numpy as np

def run_offline_test(folder="saved_audio_v33"):
    """直接读取保存的 wav 文件计算 PESQ/STOI"""
    # 查找所有的增强音频
    enhanced_files = glob.glob(os.path.join(folder, "*_v33.wav"))
    
    results = []
    print(f"--- 正在对 {folder} 进行离线指标分析 ---")

    for enh_path in enhanced_files:
        # 寻找对应的 clean 音频
        clean_path = enh_path.replace("_v33.wav", "_clean.wav")
        if not os.path.exists(clean_path): continue

        ref, fs = sf.read(clean_path)
        deg, _ = sf.read(enh_path)
        
        # 确保长度一致
        min_len = min(len(ref), len(deg))
        p_score = pesq(fs, ref[:min_len], deg[:min_len], 'wb')
        s_score = stoi(ref[:min_len], deg[:min_len], fs, extended=False)
        
        results.append((p_score, s_score))

    if results:
        avg_pesq = np.mean([r[0] for r in results])
        avg_stoi = np.mean([r[1] for r in results])
        print(f"\n[评估结果]")
        print(f"平均 PESQ: {avg_pesq:.3f}")
        print(f"平均 STOI: {avg_stoi:.3f}")
    else:
        print("未找到配对音频。")

if __name__ == "__main__":
    run_offline_test()