import os
import sys
import torch
from pathlib import Path
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# 1. 锁定项目根目录
project_root = Path(__file__).parent.resolve()
os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from addse.app.train import train as train_func
from addse.app.eval import eval as eval_func

# 2. 配置文件路径
config_path = project_root / "configs" / "addse-s-edbase-parallel60-a008-p02-spec.yaml"
config_str = str(config_path.resolve())

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print(f"--- ADDSE V3.3 任务：小数据验证 (彻底修复版) ---")

    # A. 训练阶段
    try:
        # 使用更稳健的 overrides，避开导致 "multiple values" 的 dataset 嵌套
        train_func(
            config_file=config_str,
            overrides=[
                "++trainer.max_epochs=20",
                "++dm.train_dataloader.batch_size=10",
                # [修正] 不要设为 6。设为 5 留一点余地，或者设为 1.0 (如果数据量够大)
                "++trainer.limit_train_batches=5", 
                "++dm.train_dataloader.num_workers=0",
                # 强制关闭 shuffle 可能有助于小样本下 litdata 的稳定
                "++dm.train_dataloader.dataset.shuffle=false", 
                "++model.metrics=true"
            ],
            overwrite=True,
            wandb=False
        )
    except Exception as e:
        import traceback
        print(f"\n[训练中断]: {e}")
        traceback.print_exc()

    # B. 自动生成增强音频并评估
    GlobalHydra.instance().clear() 
    ckpt_dir = project_root / "logs" / "addse-s-edbase-parallel60-a008-p02-spec" / "checkpoints"
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    
    if ckpts:
        best_ckpt = str(max(ckpts, key=lambda p: os.path.getmtime(p)))
        print(f"--- 训练完成，正在离线生成音频: {best_ckpt} ---")
        eval_func(
            config_file=config_str,
            checkpoint=best_ckpt,
            output_dir="saved_audio_v33", 
            num_examples=60, # 对应你数据集的全量
            clean=True,                   
            device="cuda"
        )
    else:
        print("\n[致命提示] 训练未生成权重，请检查上方报错信息。")