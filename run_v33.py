import os, sys, torch, shutil
from pathlib import Path
from hydra.core.global_hydra import GlobalHydra

project_root = Path(__file__).parent.resolve()
os.chdir(project_root)
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))

# 【关键操作】清理可能导致污染的旧日志目录
bad_log_dir = project_root / "logs" / "addse-s-edbase-parallel60-a008-p02-spec"
if bad_log_dir.exists():
    print(f"--- 正在清理损坏的日志目录以防 PESQ 1.05 污染 ---")
    shutil.rmtree(bad_log_dir)

from addse.app.train import train as train_func
from addse.app.eval import eval as eval_func

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print(f"--- ADDSE V3.3 进化实验 (主干保护版) ---")

    try:
        train_func(
            config_file="configs/addse-s-edbase-parallel60-a008-p02-spec.yaml",
            overrides=[
                "++trainer.max_epochs=20",
                "++dm.train_dataloader.batch_size=8",
                "++trainer.limit_train_batches=6", 
                "++dm.train_dataloader.num_workers=0",
                "++trainer.check_val_every_n_epoch=1",
                # 解决小数据集 Epoch 报错
                "++dm.train_dataset.resume=false",
                "++dm.val_dataset.resume=false",
                "++model.metrics=true",
                "++model.interaction_alpha=0.01" # 初始轻微介入，保护 1.7 分主干
            ],
            overwrite=True, wandb=False
        )
    except Exception as e:
        import traceback; traceback.print_exc()

    GlobalHydra.instance().clear() 
    ckpt_dir = project_root / "logs" / "addse-s-edbase-parallel60-a008-p02-spec" / "checkpoints"
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if ckpts:
        best_ckpt = str(max(ckpts, key=lambda p: os.path.getmtime(p)))
        eval_func(config_file="configs/addse-s-edbase-parallel60-a008-p02-spec.yaml", 
                  checkpoint=best_ckpt, output_dir="saved_audio_v33", 
                  output_db="v33_decoupled.db", num_examples=60, clean=True, device="cuda")