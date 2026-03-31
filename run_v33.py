import os, sys, torch, shutil
from pathlib import Path
from hydra.core.global_hydra import GlobalHydra

project_root = Path(__file__).parent.resolve()
os.chdir(project_root)
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))

from addse.app.train import train as train_func
from addse.app.eval import eval as eval_func

if __name__ == "__main__":
    # ====================================================================
    # 【关键修复】必须放在 __main__ 里面！防止 Windows 多进程反复删除文件夹！
    # ====================================================================
    bad_log_dir = project_root / "logs" / "addse-s-edbase-parallel60-a008-p02-spec"
    if bad_log_dir.exists():
        print(f"--- 正在清理损坏的日志目录以防 PESQ 1.05 污染 ---")
        shutil.rmtree(bad_log_dir)

    # 强制删除旧的跑分数据库，彻底防止缓存污染
    bad_db = project_root / "v33_decoupled.db"
    if bad_db.exists():
        print(f"--- 正在删除旧的跑分数据库 {bad_db.name} ---")
        os.remove(bad_db)
    # ====================================================================

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print(f"--- ADDSE V3.3 进化实验 (主干保护版) ---")

    try:
        train_func(
            config_file="configs/addse-s-edbase-parallel60-a008-p02-spec.yaml",
            overrides=[
                # ==== 1. 训练轮次与数据量精准控制 ====
                "++trainer.max_epochs=10",           # 跑 20 轮结束
                "++trainer.limit_train_batches=20", # 100 个 batch * 8 = 800 组音频数据！
                "++dm.train_dataloader.batch_size=8", 
                
                # ==== 2. 硬件加速（榨干性能，帮你省时间） ====
                "++dm.train_dataloader.num_workers=8",   # 开启多线程读数据，别让 GPU 等 CPU
                "++dm.train_dataloader.pin_memory=true", # 内存锁页，加速传输
                
                # ==== 3. 验证与杂项配置 ====
                "++trainer.check_val_every_n_epoch=2",   # 每 2 轮跑一次测试（省点测试的时间）
                "++dm.train_dataset.resume=false",
                "++dm.val_dataset.resume=false",
                "++model.metrics=true",
                "++model.interaction_alpha=0.01", 
                
                # ==== 4. 学习率控制 ====
                "++optimizer.lr=5e-5", # 既然数据少了，学习率可以稍微放大一点点，加速收敛
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
        print(f"--- 正在使用最佳权重进行评估: {best_ckpt} ---")
        eval_func(config_file="configs/addse-s-edbase-parallel60-a008-p02-spec.yaml", 
                  checkpoint=best_ckpt, output_dir="saved_audio_v33", 
                  output_db="v33_decoupled.db", num_examples=60, clean=True, device="cuda", 
                  overwrite=True)