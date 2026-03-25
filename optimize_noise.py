"""用LitData优化噪声数据"""
import subprocess
from pathlib import Path

# 创建LitData优化脚本
script_template = '''
from litdata import optimize
from pathlib import Path

input_dir = "{input_dir}"
output_dir = "{output_dir}"

print(f"优化 {{input_dir}} -> {{output_dir}}")
optimize(
    fn=lambda x: x,
    inputs=sorted(Path(input_dir).glob("*.wav")),
    output_dir=output_dir,
    chunk_bytes="10MB",
    num_workers=0,
)
print("✓ 完成")
'''

def optimize_noise_sources():
    """优化两个噪声源"""
    noise_sources = [
        ("data/chunks/musan_noise", "data/chunks/musan_noise_optimized"),
        ("data/chunks/edbase_noise_original", "data/chunks/edbase_noise_original_optimized"),
    ]
    
    for input_dir, output_dir in noise_sources:
        output_path = Path(output_dir)
        
        # 检查是否已存在
        if output_path.exists() and list(output_path.glob("*")):
            print(f"✓ {output_dir} 已优化（跳过）")
            continue
        
        print(f"\n开始优化: {input_dir}")
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"✗ 输入目录不存在: {input_dir}")
            continue
        
        # 创建临时脚本
        script_content = script_template.format(
            input_dir=str(input_path.absolute()),
            output_dir=str(output_path.absolute())
        )
        
        script_file = Path("optimize_temp.py")
        script_file.write_text(script_content)
        
        try:
            # 运行优化
            result = subprocess.run(
                ["D:/Users/KSL/PycharmProjects/the_sound/.venv/Scripts/python.exe", 
                 str(script_file)],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            if result.returncode != 0:
                print(f"✗ 优化失败，错误码: {result.returncode}")
            else:
                print(f"✓ {output_dir} 优化完成")
                
        finally:
            script_file.unlink(missing_ok=True)

if __name__ == "__main__":
    optimize_noise_sources()
