#!/usr/bin/env python3
"""
NHFAE E2 Stage 3 + Hero Plot 一键启动
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

此脚本自动执行：
  1. Stage 3 训练（Regime II Refinement）
  2. Hero Plot 评估（工业级无损透明性验证）
  3. 生成论文核心图表和报告
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime


class Stage3Pipeline:
    """Stage 3 完整流程管理。"""
    
    def __init__(self, config):
        self.config = config
        self.out_dir = Path(config["out_dir"])
        self.log_dir = self.out_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def log(self, message):
        """记录日志信息。"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_dir / "stage3_pipeline.log", "a") as f:
            f.write(log_msg + "\n")
    
    def run_stage3(self):
        """运行 Stage 3 训练。"""
        self.log("="*60)
        self.log("开始 Stage 3：Regime II Refinement")
        self.log("="*60)
        
        cmd = [
            sys.executable,
            "./addse/phase9_nhfae_e2_stage3.py",
            "--checkpoint-path", str(self.config["stage2_checkpoint"]),
            "--clean-dir", str(self.config["clean_dir"]),
            "--noisy-dir", str(self.config["noisy_dir"]),
            "--out-dir", str(self.config["out_dir"]),
            "--epochs", str(self.config.get("epochs", 1)),
            "--lr", str(self.config.get("lr", 5e-6)),
            "--lambda-dce", str(self.config.get("lambda_dce", 1.0)),
            "--lambda-cfm", str(self.config.get("lambda_cfm", 0.7)),
            "--lambda-cycle", str(self.config.get("lambda_cycle", 0.2)),
            "--lambda-mrstft", str(self.config.get("lambda_mrstft", 0.2)),
            "--device", self.config.get("device", "cuda"),
        ]
        
        self.log(f"执行命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, cwd=str(Path.cwd()))
            self.log("✓ Stage 3 训练完成")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"✗ Stage 3 训练失败: {e}")
            return False
    
    def run_hero_plot(self):
        """运行 Hero Plot 评估。"""
        self.log("="*60)
        self.log("开始 Hero Plot 评估：工业级无损透明性验证")
        self.log("="*60)
        
        enhanced_dir = self.out_dir / "wav"
        hero_dir = self.out_dir / "hero_plot"
        
        cmd = [
            sys.executable,
            "./addse/phase9_nhfae_e2_hero_plot.py",
            "--clean-dir", str(self.config["clean_dir"]),
            "--noisy-dir", str(self.config["noisy_dir"]),
            "--enhanced-dir", str(enhanced_dir),
            "--out-dir", str(hero_dir),
        ]
        
        self.log(f"执行命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, cwd=str(Path.cwd()))
            self.log("✓ Hero Plot 评估完成")
            
            # 读取和显示 Hero Plot 指标
            metrics_file = hero_dir / "hero_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                self._display_hero_metrics(metrics)
            
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"✗ Hero Plot 评估失败: {e}")
            return False
    
    def _display_hero_metrics(self, metrics):
        """显示 Hero Plot 指标摘要。"""
        stats = metrics["statistics"]
        
        self.log("\n" + "="*60)
        self.log("📊 Hero Plot 指标摘要")
        self.log("="*60)
        
        self.log(f"\n【ΔSDR 性能】")
        self.log(f"  平均值: {stats['delta_sdr']['mean']:+.6f} dB")
        self.log(f"  中位数: {stats['delta_sdr']['median']:+.6f} dB")
        self.log(f"  标准差: {stats['delta_sdr']['std']:.6f} dB")
        self.log(f"  范围:   [{stats['delta_sdr']['min']:+.6f}, {stats['delta_sdr']['max']:+.6f}] dB")
        
        target_exceeded = "✓" if stats['delta_sdr']['mean'] > 0.01 else "⚠"
        self.log(f"  {target_exceeded} 超越 +0.01 dB 阈值: {stats['delta_sdr']['mean'] > 0.01}")
        
        self.log(f"\n【相位对齐精度】")
        self.log(f"  平均误差: {stats['phase_error']['mean']:.6f} rad")
        self.log(f"  中位数:   {stats['phase_error']['median']:.6f} rad")
        self.log(f"  标准差:   {stats['phase_error']['std']:.6f} rad")
        
        phase_ok = "✓" if stats['phase_error']['mean'] < 0.05 else "⚠"
        self.log(f"  {phase_ok} 1-NFE 就绪性: {'优秀' if stats['phase_error']['mean'] < 0.05 else '需改进'}")
        
        self.log(f"\n【幅度锁定指标】")
        self.log(f"  平均扰动: {stats['mag_perturbation']['mean']:.6f}")
        self.log(f"  中位数:   {stats['mag_perturbation']['median']:.6f}")
        self.log(f"  标准差:   {stats['mag_perturbation']['std']:.6f}")
        
        mag_ok = "✓" if stats['mag_perturbation']['mean'] < 0.01 else "⚠"
        self.log(f"  {mag_ok} Posterior Mean 锁定: {'成功' if stats['mag_perturbation']['mean'] < 0.01 else '可接受' if stats['mag_perturbation']['mean'] < 0.05 else '需改进'}")
        
        self.log("\n" + "="*60)
    
    def run_pipeline(self):
        """执行完整 Stage 3 流程。"""
        self.log("开始 NHFAE E2 Stage 3 完整流程")
        self.log(f"配置: {json.dumps(self.config, indent=2, ensure_ascii=False)}")
        
        # Stage 3 训练
        if not self.run_stage3():
            self.log("✗ 流程中止: Stage 3 训练失败")
            return False
        
        # Hero Plot 评估
        if not self.run_hero_plot():
            self.log("⚠ 警告: Hero Plot 评估失败")
            # 不中止，提示用户查看日志
        
        # 最终总结
        self._summarize()
        
        return True
    
    def _summarize(self):
        """生成最终总结。"""
        self.log("\n" + "="*60)
        self.log("✓ Stage 3 流程完成")
        self.log("="*60)
        
        ckpt_path = self.out_dir / "ckpt" / "best.pt"
        wav_dir = self.out_dir / "wav"
        hero_dir = self.out_dir / "hero_plot"
        
        self.log(f"\n【输出位置】")
        self.log(f"  权重: {ckpt_path}")
        self.log(f"  波形: {wav_dir}")
        self.log(f"  Hero Plot: {hero_dir}")
        self.log(f"  日志: {self.log_dir}")
        
        self.log(f"\n【下一步】")
        self.log(f"  1. 查看 Hero Plot 图表: {hero_dir / 'hero_plot.png'}")
        self.log(f"  2. 检查指标 JSON: {hero_dir / 'hero_metrics.json'}")
        self.log(f"  3. 如果指标满足 1-NFE 条件，开始 Stage 4 preparation")
        self.log(f"\n【1-NFE 条件检查表】")
        self.log(f"  □ Phase Error Mean < 0.05 rad")
        self.log(f"  □ Mag Perturbation Mean < 0.01")
        self.log(f"  □ ΔSDR Mean > +0.01 dB")
        
        self.log("\n论文核心贡献已就绪！ 🚀")


def main():
    parser = argparse.ArgumentParser("NHFAE E2 Stage 3 + Hero Plot 一键启动")
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"✗ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = json.load(f)
    
    # 验证配置
    required_keys = ["stage2_checkpoint", "clean_dir", "noisy_dir", "out_dir"]
    for key in required_keys:
        if key not in config:
            print(f"✗ 配置缺失: {key}")
            sys.exit(1)
        
        if key != "out_dir":  # out_dir 无需存在
            path = Path(config[key])
            if not path.exists():
                print(f"✗ 路径不存在: {key} = {path}")
                sys.exit(1)
    
    # 执行流程
    pipeline = Stage3Pipeline(config)
    success = pipeline.run_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
