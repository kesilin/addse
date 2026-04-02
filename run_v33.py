import argparse
import os
import shutil
import sys
import torch
from pathlib import Path
from hydra.core.global_hydra import GlobalHydra

project_root = Path(__file__).parent.resolve()
os.chdir(project_root)
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))

from addse.app.train import train as train_func
from addse.app.eval import eval as eval_func

if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_v33 short/full pipeline")
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--train-batches", type=int, default=5)
    parser.add_argument("--train-groups", type=int, default=150, help="Synthetic training groups / dataset length")
    parser.add_argument("--eval-examples", type=int, default=60)
    parser.add_argument("--eval-steps", type=int, default=200, help="Diffusion steps during eval; use 200 for strict evaluation")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--reset-log-dir", action="store_true", help="Delete log dir before training")
    parser.add_argument("--reset-db", action="store_true", help="Delete output DB before evaluation")
    parser.add_argument("--deterministic-eval", action="store_true", help="Use deterministic token decoding during eval")
    parser.add_argument("--output-db", type=str, default="v33_decoupled.db")
    parser.add_argument("--output-dir", type=str, default="saved_audio_v33")
    parser.add_argument("--disable-wave-residual", action="store_true", help="Disable wave residual side-stream branch")
    parser.add_argument("--direct-residual-weight", type=float, default=1.0, help="Weight for explicit DEX residual supervision")
    parser.add_argument("--residual-l1-weight", type=float, default=0.0, help="Fallback latent residual L1 weight")
    parser.add_argument("--si-sdr-weight", type=float, default=0.2, help="Wave-domain SI-SDR weight for this run")
    parser.add_argument("--alpha-init-prob", type=float, default=0.05, help="Initial sigmoid probability for alpha gates")
    parser.add_argument("--print-alpha", action="store_true", help="Print alpha logs in progress bar (handled in lightning logs)")
    args = parser.parse_args()

    # ====================================================================
    # 【关键修复】必须放在 __main__ 里面！防止 Windows 多进程反复删除文件夹！
    # ====================================================================
    bad_log_dir = project_root / "logs" / "addse-s-edbase-parallel60-a008-p02-spec"
    if args.reset_log_dir and bad_log_dir.exists():
        print(f"--- 正在清理损坏的日志目录以防 PESQ 1.05 污染 ---")
        shutil.rmtree(bad_log_dir)

    # 强制删除旧的跑分数据库，彻底防止缓存污染
    bad_db = project_root / args.output_db
    if args.reset_db and bad_db.exists():
        print(f"--- 正在删除旧的跑分数据库 {bad_db.name} ---")
        os.remove(bad_db)
    # ====================================================================

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print(f"--- ADDSE V3.3 进化实验 (主干保护版) ---")

    train_func(
        config_file="configs/addse-s-edbase-parallel60-a008-p02-spec.yaml",
        overrides=[
            # ==== 1. 训练轮次与数据量精准控制 ====
            f"++trainer.max_epochs={args.train_epochs}",
            f"++trainer.limit_train_batches={args.train_batches}",
            "++dm.train_dataloader.batch_size=8",
            f"++dm.train_dataset.length={max(args.train_groups, args.train_batches * 8)}",

            # ==== 1.1 P-SSA 训练超参（从 Oracle 过渡到真实训练） ====
            "++lm.spec_loss_weight=1.0",
            "++lm.wave_l1_weight=2.0",
            f"++lm.si_sdr_weight={args.si_sdr_weight}",
            f"++lm.residual_l1_weight={args.residual_l1_weight}",
            f"++lm.direct_residual_weight={args.direct_residual_weight}",
            "++lm.metricgan_plus_enabled=false",
            "++lm.metricgan_weight=0.0",
            "++lm.wave_residual_multiscale=true",
            "++lm.wave_residual_low_stride=8",
            f"++lm.alpha_init_prob={args.alpha_init_prob}",

            # ==== 1.2 实验开关 ====
            f"++lm.wave_residual_enabled={str((not args.disable_wave_residual)).lower()}",

            # ==== 2. 硬件加速（榨干性能，帮你省时间） ====
            "++dm.train_dataloader.num_workers=8",   # 开启多线程读数据，别让 GPU 等 CPU
            "++dm.train_dataloader.pin_memory=true", # 内存锁页，加速传输

            # ==== 3. 验证与杂项配置 ====
            f"++trainer.check_val_every_n_epoch={args.val_every}",
            "++dm.train_dataset.resume=false",
            "++dm.val_dataset.resume=false",
        ],
        overwrite=args.reset_log_dir, wandb=False
    )

    GlobalHydra.instance().clear() 
    ckpt_dir = project_root / "logs" / "addse-s-edbase-parallel60-a008-p02-spec" / "checkpoints"
    ckpts = list(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
    if ckpts:
        best_ckpt = str(max(ckpts, key=lambda p: os.path.getmtime(p)))
        print(f"--- 正在使用最佳权重进行评估: {best_ckpt} ---")
        eval_func(
            config_file="configs/addse-s-edbase-parallel60-a008-p02-spec.yaml",
            checkpoint=best_ckpt,
            overrides=[
                f"lm.num_steps={args.eval_steps}",
                f"lm.wave_residual_enabled={str((not args.disable_wave_residual)).lower()}",
                f"eval.dsets.edbase-local.length={args.eval_examples}",
                f"lm.deterministic_eval={str(args.deterministic_eval).lower()}",
            ],
            output_dir=args.output_dir,
            output_db=args.output_db,
            num_examples=args.eval_examples,
            noisy=True,
            clean=True,
            device=args.device,
            overwrite=True,
            num_consumers=0,
        )
    else:
        print("--- 未找到 checkpoint，跳过评估。请检查 trainer 回调与验证频率。---")