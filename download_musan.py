"""生成合成噪声数据集（MUSAN-like）- 100MB级"""
import numpy as np
import soundfile as sf
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 配置
OUTPUT_DIR = Path("data/chunks/musan_noise_raw")
FS = 16000  # 采样率
DURATION_PER_FILE = 120  # 每个噪声文件120秒
NUM_FILES = 36  # 生成36个文件，约100MB

def generate_synthetic_noise(duration_sec, noise_type, fs=16000):
    """生成不同类型的合成噪声
    
    Args:
        duration_sec: 持续时间（秒）
        noise_type: 噪声类型 (white, pink, brown, speech_like, machine, traffic)
        fs: 采样率
    
    Returns:
        numpy array with shape (num_samples,)
    """
    num_samples = int(duration_sec * fs)
    
    if noise_type == "white":
        # 白噪声
        noise = np.random.normal(0, 0.05, num_samples)
    
    elif noise_type == "pink":
        # 粉噪声 - 低频更强
        white = np.random.normal(0, 0.05, num_samples)
        noise = np.zeros_like(white)
        for i in range(1, len(white)):
            noise[i] = 0.99 * noise[i-1] + white[i] * 0.01
        noise = noise / np.max(np.abs(noise)) * 0.05
    
    elif noise_type == "brown":
        # 棕噪声 - 更低频
        white = np.random.normal(0, 0.05, num_samples)
        noise = np.zeros_like(white)
        for i in range(1, len(white)):
            noise[i] = 0.98 * noise[i-1] + white[i] * 0.02
        noise = noise / np.max(np.abs(noise)) * 0.05
    
    elif noise_type == "speech_like":
        # 类似语音的噪声
        noise = np.sin(2 * np.pi * np.random.uniform(100, 500, num_samples) * 
                      np.arange(num_samples) / fs) * 0.03
        noise += 0.02 * np.random.normal(0, 1, num_samples)
    
    elif noise_type == "machine":
        # 机器噪声 - 特定频率
        t = np.arange(num_samples) / fs
        fundamentals = [100, 250, 400]  # Hz
        noise = np.zeros(num_samples)
        for freq in fundamentals:
            noise += 0.02 * np.sin(2 * np.pi * freq * t)
        noise += 0.015 * np.random.normal(0, 1, num_samples)
        noise = noise / np.max(np.abs(noise)) * 0.05
    
    elif noise_type == "traffic":
        # 交通噪声 - 混合多个频率
        t = np.arange(num_samples) / fs
        noise = 0.02 * np.sin(2 * np.pi * 70 * t)  # 低频
        noise += 0.015 * np.sin(2 * np.pi * 150 * t)  # 中频
        noise += 0.01 * np.random.randint(-1, 2, num_samples) * np.sin(2 * np.pi * 300 * t)  # 高频
        noise += 0.02 * np.random.normal(0, 1, num_samples)
        noise = noise / np.max(np.abs(noise)) * 0.05
    
    else:
        noise = np.random.normal(0, 0.01, num_samples)
    
    return noise.astype(np.float32)

def generate_synthetic_dataset():
    """生成合成噪声数据集"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    noise_types = ["white", "pink", "brown", "speech_like", "machine", "traffic"]
    files_per_type = NUM_FILES // len(noise_types)
    
    total_size = 0
    file_count = 0
    
    print("生成合成噪声数据集...")
    print(f"目标: {NUM_FILES} 个文件, ~100MB")
    print(f"每个文件: {DURATION_PER_FILE}秒 @ {FS}Hz")
    print()
    
    for noise_idx, noise_type in enumerate(noise_types):
        print(f"生成 {noise_type} 噪声... ", end="", flush=True)
        
        for file_idx in range(files_per_type):
            # 生成噪声
            noise = generate_synthetic_noise(DURATION_PER_FILE, noise_type, FS)
            
            # 增加一些变化：随机调制
            t = np.arange(len(noise)) / FS
            modulation = 0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * t)  # 0.5-0.8的调制
            noise = noise * modulation
            
            # 保存
            output_file = OUTPUT_DIR / f"noise_{noise_type}_{file_idx:02d}.wav"
            sf.write(str(output_file), noise, FS)
            
            file_size = output_file.stat().st_size
            total_size += file_size
            file_count += 1
        
        print(f"✓ 完成 ({files_per_type} 文件)")
    
    print()
    print(f"✓ 生成完成")
    print(f"  总文件数: {file_count}")
    print(f"  总大小: {total_size / 1024 / 1024:.1f} MB")
    print(f"  输出路径: {OUTPUT_DIR}")
    
    return total_size / 1024 / 1024

if __name__ == "__main__":
    generate_synthetic_dataset()
