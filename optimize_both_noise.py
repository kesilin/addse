"""优化噪声数据"""
from litdata import optimize
from pathlib import Path

def identity_fn(x):
    """恒等函数，用于LitData优化"""
    return x

def optimize_noise_source(input_dir, output_dir):
    """优化单个噪声源"""
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    
    print(f"优化 {input_path} -> {output_path}")
    
    if not input_path.exists():
        print(f"✗ 输入目录不存在: {input_path}")
        return False
    
    wav_files = sorted(input_path.glob("*.wav"))
    if not wav_files:
        print(f"✗ 没有找到WAV文件: {input_path}")
        return False
    
    print(f"  找到 {len(wav_files)} 个文件，总大小: {sum(f.stat().st_size for f in wav_files) / 1024 / 1024:.1f} MB")
    
    try:
        optimize(
            fn=identity_fn,
            inputs=wav_files,
            output_dir=str(output_path),
            chunk_bytes="10MB",
            num_workers=0,
        )
        print(f"✓ {output_path} 优化完成")
        return True
    except Exception as e:
        print(f"✗ 优化失败: {e}")
        return False

if __name__ == "__main__":
    base_path = Path(__file__).parent
    
    # 优化musan_noise
    optimize_noise_source(
        base_path / "data/chunks/musan_noise",
        base_path / "data/chunks/musan_noise_optimized"
    )
    
    print()
    
    # 优化edbase_noise_original
    optimize_noise_source(
        base_path / "data/chunks/edbase_noise_original",
        base_path / "data/chunks/edbase_noise_original_optimized"
    )
