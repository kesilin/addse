#!/usr/bin/env python3
"""快速对比两种噪声源的训练效果"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_training(config_name, description):
    """运行单个训练配置"""
    print(f"\n{'='*60}")
    print(f"🎯 {description}")
    print(f"{'='*60}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置文件: configs/{config_name}.yaml")
    print(f"训练时长: 10 epochs × 150 batches (~8-10分钟)")
    print()
    
    cmd = [
        sys.executable, "-m", "addse.app", "train",
        f"configs/{config_name}.yaml",
        "--init-ckpt", "logs/addse-edbase-quick/checkpoints/addse-s.ckpt",
        "--overwrite"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        return False

def run_evaluation(config_name, checkpoint_pattern, db_name, description):
    """评估模型"""
    print(f"\n{'='*60}")
    print(f"📊 评估: {description}")
    print(f"{'='*60}")
    
    # 查找最新检查点
    log_dir = Path(f"logs/{config_name}/checkpoints")
    if not log_dir.exists():
        print(f"✗ 日志目录不存在: {log_dir}")
        return None
    
    checkpoints = sorted(log_dir.glob("epoch=*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not checkpoints:
        print(f"✗ 未找到检查点")
        return None
    
    checkpoint = checkpoints[0]
    print(f"使用检查点: {checkpoint.name}")
    
    cmd = [
        sys.executable, "-m", "addse.app", "eval",
        f"configs/{config_name}.yaml",
        str(checkpoint),
        "--device", "cuda",
        "--output-db", db_name,
        "--overwrite",
        "--num-consumers", "0",
        "--num-examples", "30",
        "lm.num_steps=64",
        "--noisy"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # 提取结果
            output = result.stderr + result.stdout
            print(output)
            return checkpoint
        else:
            print(f"✗ 评估失败")
            return None
    except Exception as e:
        print(f"✗ 评估错误: {e}")
        return None

def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          ADDSE 噪声源对比实验                                  ║
║  比较两种方案对模型效果的影响 (约20-40分钟)                   ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    configs = [
        ("addse-s-original-ft", "方案1️⃣: 原有噪声源 (5个LitData优化文件)"),
        ("addse-s-musan-ft", "方案2️⃣: 新增MUSAN合成噪声 (36个WAV文件)"),
    ]
    
    results = {}
    
    for config_name, description in configs:
        print(f"\n[{configs.index((config_name, description)) + 1}/{len(configs)}]", end=" ")
        
        # 训练
        success = run_training(config_name, description)
        if not success:
            print(f"跳过 {config_name} 的评估")
            continue
        
        # 评估
        db_name = f"eval_{config_name}_comparison.db"
        run_evaluation(
            config_name,
            "epoch=*.ckpt",
            db_name,
            f"{config_name} - 最佳模型"
        )
        
        results[config_name] = {
            "description": description,
            "db": db_name,
        }
    
    # 总结
    print(f"\n{'='*60}")
    print("📈 实验总结")
    print(f"{'='*60}")
    for config_name, info in results.items():
        print(f"✓ {info['description']}")
        print(f"  结果数据库: {info['db']}")

if __name__ == "__main__":
    main()
