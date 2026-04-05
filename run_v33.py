import argparse
import os
import shutil
import sys
import random
import numpy as np
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-shuffle", action="store_true", help="Enable train dataloader shuffle")
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
    parser.add_argument("--optimizer-lr", type=float, default=6.0e-5, help="Optimizer LR override")
    parser.add_argument("--spec-loss-weight", type=float, default=1.0, help="Spectral loss weight")
    parser.add_argument("--wave-l1-weight", type=float, default=2.0, help="Wave L1 loss weight")
    parser.add_argument("--alpha-init-prob", type=float, default=0.05, help="Initial sigmoid probability for alpha gates")
    parser.add_argument("--print-alpha", action="store_true", help="Print alpha logs in progress bar (handled in lightning logs)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducible runs")

    # ===== SAD-RVQ Scheme Selection =====
    parser.add_argument("--sad-rvq-scheme", type=str, default="baseline", choices=["baseline", "a", "b", "c", "d", "e", "f", "g", "h"],
                       help="Which SAD-RVQ scheme to use (baseline, a=entropy, b=dual-head, c=progressive, d=gating)")
    parser.add_argument("--sad-rvq-scheme-a-weight", type=float, default=0.5,
                       help="Weight for Scheme A entropy boost loss")
    parser.add_argument("--sad-rvq-scheme-d-gate-entropy-weight", type=float, default=0.1,
                       help="Weight for Scheme D gate entropy regularization")
    parser.add_argument("--sad-rvq-scheme-d-acoustic-weight", type=float, default=0.5,
                       help="Weight for direct acoustic logits supervision (Scheme D)")
    parser.add_argument("--sad-rvq-scheme-d-gate-polar-weight", type=float, default=0.1,
                       help="Weight for gate polarization penalty (Scheme D)")
    parser.add_argument("--sad-rvq-scheme-d-acoustic-lr-scale", type=float, default=5.0,
                       help="Gradient scale for Scheme D acoustic branch to emulate higher LR")
    parser.add_argument("--sad-rvq-scheme-d-head-lr-scale", type=float, default=1.0,
                       help="Extra gradient scale for Scheme D acoustic token head")
    parser.add_argument("--sad-rvq-scheme-d-final-weight", type=float, default=1.0,
                       help="Weight for fused post-3 CE in Scheme D/H")
    parser.add_argument("--sad-rvq-scheme-d-reg-weight", type=float, default=1.0,
                       help="Weight for latent regression loss in Scheme D/H")
    parser.add_argument("--sad-rvq-scheme-d-ce-aux-weight", type=float, default=0.1,
                       help="Auxiliary CE weight when acoustic_only mode is enabled")
    parser.add_argument("--sad-rvq-scheme-d-codebook-consistency-weight", type=float, default=0.0,
                       help="Weight for codebook consistency loss on regressed latent")
    parser.add_argument("--sad-rvq-scheme-d-codebook-consistency-books", type=int, default=12,
                       help="Number of NAC codebooks used in codebook consistency loss")
    parser.add_argument("--sad-rvq-scheme-d-post3-head-hidden-mult", type=int, default=3,
                       help="Hidden dimension multiplier for post-3 acoustic heads")
    parser.add_argument("--sad-rvq-scheme-d-use-prototype-objective", action="store_true",
                       help="Use prototype alignment + residual correction objective for post-3 acoustic branch")
    parser.add_argument("--sad-rvq-scheme-d-prototype-weight", type=float, default=1.0,
                       help="Weight for prototype alignment loss")
    parser.add_argument("--sad-rvq-scheme-d-residual-correction-weight", type=float, default=0.1,
                       help="Weight for residual correction loss")
    parser.add_argument("--sad-rvq-scheme-d-use-candidate-objective", action="store_true",
                       help="Use candidate-set contrastive objective for post-3 acoustic branch")
    parser.add_argument("--sad-rvq-scheme-d-candidate-size", type=int, default=32,
                       help="Number of candidate tokens for contrastive selection")
    parser.add_argument("--sad-rvq-scheme-d-candidate-ce-weight", type=float, default=0.3,
                       help="Weight for candidate-set contrastive CE loss")
    parser.add_argument("--sad-rvq-scheme-d-candidate-query-from-front-tokens", action="store_true",
                       help="Build candidate set using front token embeddings instead of projected latent")
    parser.add_argument("--sad-rvq-scheme-d-use-multimodal-query", action="store_true",
                       help="Enable multimodal query branch (baseline keeps this disabled)")
    parser.add_argument("--sad-rvq-scheme-d-distribution-alignment-weight", type=float, default=1.0,
                       help="Weight for projector-space distribution alignment loss")
    parser.add_argument("--sad-rvq-scheme-d-warmup-steps", type=int, default=10000,
                       help="Prototype-only warmup steps before enabling full post-3 objective")
    parser.add_argument("--sad-rvq-scheme-d-distribution-only-l4", action="store_true",
                       help="Align distribution stats using only layer-4 codebook")
    parser.add_argument("--sad-rvq-scheme-d-log-l4-dist", action="store_true",
                       help="Log layer-4 correct/wrong token distance means during validation")
    parser.add_argument("--sad-rvq-scheme-d-continuous-residual-mode", action="store_true",
                       help="Bypass token CE objectives and train wave-domain continuous residual refinement")
    parser.add_argument("--sad-rvq-scheme-d-continuous-coarse-weight", type=float, default=0.3,
                       help="Weight for coarse-wave SI-SDR loss in continuous residual mode")
    parser.add_argument("--sad-rvq-scheme-d-continuous-enhanced-weight", type=float, default=1.0,
                       help="Weight for enhanced-wave SI-SDR loss in continuous residual mode")
    parser.add_argument("--sad-rvq-scheme-d-continuous-front3-ce-weight", type=float, default=1.0,
                       help="Auxiliary CE weight for first 3 codebooks in continuous residual mode")
    parser.add_argument("--sad-rvq-scheme-d-continuous-coarse-source", type=str, default="front3", choices=["front3", "noisy"],
                       help="Coarse waveform source in continuous residual mode")
    parser.add_argument("--sad-rvq-scheme-d-continuous-dump-audio", action="store_true",
                       help="Dump coarse/noisy/clean wav triplet for val batch-0 diagnosis")
    parser.add_argument("--sad-rvq-scheme-d-continuous-use-stft-predictor", action="store_true",
                       help="Use STFT-conditioned residual predictor in continuous residual mode")
    parser.add_argument("--sad-rvq-scheme-d-continuous-stft-nfft", type=int, default=512,
                       help="NFFT for STFT-conditioned residual predictor")
    parser.add_argument("--sad-rvq-scheme-d-continuous-stft-hop", type=int, default=128,
                       help="Hop length for STFT-conditioned residual predictor")
    parser.add_argument("--sad-rvq-scheme-d-continuous-stft-win", type=int, default=512,
                       help="Window length for STFT-conditioned residual predictor")
    parser.add_argument("--sad-rvq-scheme-d-continuous-multiscale-spec-weight", type=float, default=0.0,
                       help="Weight for multiscale STFT magnitude loss in continuous mode")
    parser.add_argument("--sad-rvq-scheme-d-projector-hidden-mult", type=float, default=1.5,
                       help="Hidden multiplier for codebook projector MLP")
    parser.add_argument("--sad-rvq-scheme-d-train-post3-only", action="store_true",
                       help="Freeze non-post3 parameters and train only post3 prototype branch")
    parser.add_argument("--sad-rvq-scheme-d-init-from-codebook", action="store_true",
                       help="Initialize Scheme D acoustic token head from NAC codebook")
    parser.add_argument("--sad-rvq-scheme-train-mode", type=str, default="normal", choices=["normal", "acoustic_only"],
                       help="Training mode for Scheme D/H")
    parser.add_argument("--sad-rvq-freeze-main-model", action="store_true",
                       help="Freeze main DiT model parameters and train acoustic branch only")
    parser.add_argument("--sad-rvq-scheme-g-entropy-quantile", type=float, default=0.5,
                       help="Quantile threshold for entropy-guided hard routing (Scheme G)")
    parser.add_argument("--sad-rvq-scheme-h-min-temp", type=float, default=0.1,
                       help="Minimum temperature for annealed routing (Scheme H)")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
            f"++dm.train_dataloader.batch_size={args.batch_size}",
            f"++dm.train_dataloader.shuffle={str(args.train_shuffle).lower()}",
            f"++dm.train_dataset.length={max(args.train_groups, args.train_batches * args.batch_size)}",

            # ==== 1.1 P-SSA 训练超参（从 Oracle 过渡到真实训练） ====
            f"++lm.spec_loss_weight={args.spec_loss_weight}",
            f"++lm.wave_l1_weight={args.wave_l1_weight}",
            f"++lm.si_sdr_weight={args.si_sdr_weight}",
            f"++lm.residual_l1_weight={args.residual_l1_weight}",
            f"++lm.direct_residual_weight={args.direct_residual_weight}",
            f"++lm.optimizer.lr={args.optimizer_lr}",
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

            # ==== 4. SAD-RVQ 方案选择 ====
            f"++lm.sad_rvq_scheme={args.sad_rvq_scheme}",
            f"++lm.sad_rvq_scheme_d_enabled={str(args.sad_rvq_scheme == 'd').lower()}",
            f"++lm.sad_rvq_scheme_a_weight={args.sad_rvq_scheme_a_weight}",
            f"++lm.sad_rvq_scheme_d_gate_entropy_weight={args.sad_rvq_scheme_d_gate_entropy_weight}",
            f"++lm.sad_rvq_scheme_d_acoustic_weight={args.sad_rvq_scheme_d_acoustic_weight}",
            f"++lm.sad_rvq_scheme_d_gate_polar_weight={args.sad_rvq_scheme_d_gate_polar_weight}",
            f"++lm.sad_rvq_scheme_d_acoustic_lr_scale={args.sad_rvq_scheme_d_acoustic_lr_scale}",
            f"++lm.sad_rvq_scheme_d_head_lr_scale={args.sad_rvq_scheme_d_head_lr_scale}",
            f"++lm.sad_rvq_scheme_d_final_weight={args.sad_rvq_scheme_d_final_weight}",
            f"++lm.sad_rvq_scheme_d_reg_weight={args.sad_rvq_scheme_d_reg_weight}",
            f"++lm.sad_rvq_scheme_d_ce_aux_weight={args.sad_rvq_scheme_d_ce_aux_weight}",
            f"++lm.sad_rvq_scheme_d_codebook_consistency_weight={args.sad_rvq_scheme_d_codebook_consistency_weight}",
            f"++lm.sad_rvq_scheme_d_codebook_consistency_books={args.sad_rvq_scheme_d_codebook_consistency_books}",
            f"++lm.sad_rvq_scheme_d_post3_head_hidden_mult={args.sad_rvq_scheme_d_post3_head_hidden_mult}",
            f"++lm.sad_rvq_scheme_d_use_prototype_objective={str(args.sad_rvq_scheme_d_use_prototype_objective).lower()}",
            f"++lm.sad_rvq_scheme_d_prototype_weight={args.sad_rvq_scheme_d_prototype_weight}",
            f"++lm.sad_rvq_scheme_d_residual_correction_weight={args.sad_rvq_scheme_d_residual_correction_weight}",
            f"++lm.sad_rvq_scheme_d_use_candidate_objective={str(args.sad_rvq_scheme_d_use_candidate_objective).lower()}",
            f"++lm.sad_rvq_scheme_d_candidate_size={args.sad_rvq_scheme_d_candidate_size}",
            f"++lm.sad_rvq_scheme_d_candidate_ce_weight={args.sad_rvq_scheme_d_candidate_ce_weight}",
            f"++lm.sad_rvq_scheme_d_candidate_query_from_front_tokens={str(args.sad_rvq_scheme_d_candidate_query_from_front_tokens).lower()}",
            f"++lm.sad_rvq_scheme_d_use_multimodal_query={str(args.sad_rvq_scheme_d_use_multimodal_query).lower()}",
            f"++lm.sad_rvq_scheme_d_distribution_alignment_weight={args.sad_rvq_scheme_d_distribution_alignment_weight}",
            f"++lm.sad_rvq_scheme_d_warmup_steps={args.sad_rvq_scheme_d_warmup_steps}",
            f"++lm.sad_rvq_scheme_d_distribution_only_l4={str(args.sad_rvq_scheme_d_distribution_only_l4).lower()}",
            f"++lm.sad_rvq_scheme_d_log_l4_dist={str(args.sad_rvq_scheme_d_log_l4_dist).lower()}",
            f"++lm.sad_rvq_scheme_d_continuous_residual_mode={str(args.sad_rvq_scheme_d_continuous_residual_mode).lower()}",
            f"++lm.sad_rvq_scheme_d_continuous_coarse_weight={args.sad_rvq_scheme_d_continuous_coarse_weight}",
            f"++lm.sad_rvq_scheme_d_continuous_enhanced_weight={args.sad_rvq_scheme_d_continuous_enhanced_weight}",
            f"++lm.sad_rvq_scheme_d_continuous_front3_ce_weight={args.sad_rvq_scheme_d_continuous_front3_ce_weight}",
            f"++lm.sad_rvq_scheme_d_continuous_coarse_source={args.sad_rvq_scheme_d_continuous_coarse_source}",
            f"++lm.sad_rvq_scheme_d_continuous_dump_audio={str(args.sad_rvq_scheme_d_continuous_dump_audio).lower()}",
            f"++lm.sad_rvq_scheme_d_continuous_use_stft_predictor={str(args.sad_rvq_scheme_d_continuous_use_stft_predictor).lower()}",
            f"++lm.sad_rvq_scheme_d_continuous_stft_nfft={args.sad_rvq_scheme_d_continuous_stft_nfft}",
            f"++lm.sad_rvq_scheme_d_continuous_stft_hop={args.sad_rvq_scheme_d_continuous_stft_hop}",
            f"++lm.sad_rvq_scheme_d_continuous_stft_win={args.sad_rvq_scheme_d_continuous_stft_win}",
            f"++lm.sad_rvq_scheme_d_continuous_multiscale_spec_weight={args.sad_rvq_scheme_d_continuous_multiscale_spec_weight}",
            f"++lm.sad_rvq_scheme_d_projector_hidden_mult={args.sad_rvq_scheme_d_projector_hidden_mult}",
            f"++lm.sad_rvq_scheme_d_train_post3_only={str(args.sad_rvq_scheme_d_train_post3_only).lower()}",
            f"++lm.sad_rvq_scheme_d_init_from_codebook={str(args.sad_rvq_scheme_d_init_from_codebook).lower()}",
            f"++lm.sad_rvq_scheme_train_mode={args.sad_rvq_scheme_train_mode}",
            f"++lm.sad_rvq_freeze_main_model={str(args.sad_rvq_freeze_main_model).lower()}",
            f"++lm.sad_rvq_scheme_g_entropy_quantile={args.sad_rvq_scheme_g_entropy_quantile}",
            f"++lm.sad_rvq_scheme_h_min_temp={args.sad_rvq_scheme_h_min_temp}",
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
                f"++lm.sad_rvq_scheme={args.sad_rvq_scheme}",
                f"++lm.sad_rvq_scheme_d_enabled={str(args.sad_rvq_scheme == 'd').lower()}",
                f"++lm.sad_rvq_scheme_a_weight={args.sad_rvq_scheme_a_weight}",
                f"++lm.sad_rvq_scheme_d_gate_entropy_weight={args.sad_rvq_scheme_d_gate_entropy_weight}",
                f"++lm.sad_rvq_scheme_d_acoustic_weight={args.sad_rvq_scheme_d_acoustic_weight}",
                f"++lm.sad_rvq_scheme_d_gate_polar_weight={args.sad_rvq_scheme_d_gate_polar_weight}",
                f"++lm.sad_rvq_scheme_d_acoustic_lr_scale={args.sad_rvq_scheme_d_acoustic_lr_scale}",
                f"++lm.sad_rvq_scheme_d_head_lr_scale={args.sad_rvq_scheme_d_head_lr_scale}",
                f"++lm.sad_rvq_scheme_d_final_weight={args.sad_rvq_scheme_d_final_weight}",
                f"++lm.sad_rvq_scheme_d_reg_weight={args.sad_rvq_scheme_d_reg_weight}",
                f"++lm.sad_rvq_scheme_d_ce_aux_weight={args.sad_rvq_scheme_d_ce_aux_weight}",
                f"++lm.sad_rvq_scheme_d_codebook_consistency_weight={args.sad_rvq_scheme_d_codebook_consistency_weight}",
                f"++lm.sad_rvq_scheme_d_codebook_consistency_books={args.sad_rvq_scheme_d_codebook_consistency_books}",
                f"++lm.sad_rvq_scheme_d_post3_head_hidden_mult={args.sad_rvq_scheme_d_post3_head_hidden_mult}",
                f"++lm.sad_rvq_scheme_d_use_prototype_objective={str(args.sad_rvq_scheme_d_use_prototype_objective).lower()}",
                f"++lm.sad_rvq_scheme_d_prototype_weight={args.sad_rvq_scheme_d_prototype_weight}",
                f"++lm.sad_rvq_scheme_d_residual_correction_weight={args.sad_rvq_scheme_d_residual_correction_weight}",
                f"++lm.sad_rvq_scheme_d_use_candidate_objective={str(args.sad_rvq_scheme_d_use_candidate_objective).lower()}",
                f"++lm.sad_rvq_scheme_d_candidate_size={args.sad_rvq_scheme_d_candidate_size}",
                f"++lm.sad_rvq_scheme_d_candidate_ce_weight={args.sad_rvq_scheme_d_candidate_ce_weight}",
                f"++lm.sad_rvq_scheme_d_candidate_query_from_front_tokens={str(args.sad_rvq_scheme_d_candidate_query_from_front_tokens).lower()}",
                f"++lm.sad_rvq_scheme_d_use_multimodal_query={str(args.sad_rvq_scheme_d_use_multimodal_query).lower()}",
                f"++lm.sad_rvq_scheme_d_distribution_alignment_weight={args.sad_rvq_scheme_d_distribution_alignment_weight}",
                f"++lm.sad_rvq_scheme_d_warmup_steps={args.sad_rvq_scheme_d_warmup_steps}",
                f"++lm.sad_rvq_scheme_d_distribution_only_l4={str(args.sad_rvq_scheme_d_distribution_only_l4).lower()}",
                f"++lm.sad_rvq_scheme_d_log_l4_dist={str(args.sad_rvq_scheme_d_log_l4_dist).lower()}",
                f"++lm.sad_rvq_scheme_d_continuous_residual_mode={str(args.sad_rvq_scheme_d_continuous_residual_mode).lower()}",
                f"++lm.sad_rvq_scheme_d_continuous_coarse_weight={args.sad_rvq_scheme_d_continuous_coarse_weight}",
                f"++lm.sad_rvq_scheme_d_continuous_enhanced_weight={args.sad_rvq_scheme_d_continuous_enhanced_weight}",
                f"++lm.sad_rvq_scheme_d_continuous_front3_ce_weight={args.sad_rvq_scheme_d_continuous_front3_ce_weight}",
                f"++lm.sad_rvq_scheme_d_continuous_coarse_source={args.sad_rvq_scheme_d_continuous_coarse_source}",
                f"++lm.sad_rvq_scheme_d_continuous_dump_audio={str(args.sad_rvq_scheme_d_continuous_dump_audio).lower()}",
                f"++lm.sad_rvq_scheme_d_continuous_use_stft_predictor={str(args.sad_rvq_scheme_d_continuous_use_stft_predictor).lower()}",
                f"++lm.sad_rvq_scheme_d_continuous_stft_nfft={args.sad_rvq_scheme_d_continuous_stft_nfft}",
                f"++lm.sad_rvq_scheme_d_continuous_stft_hop={args.sad_rvq_scheme_d_continuous_stft_hop}",
                f"++lm.sad_rvq_scheme_d_continuous_stft_win={args.sad_rvq_scheme_d_continuous_stft_win}",
                f"++lm.sad_rvq_scheme_d_continuous_multiscale_spec_weight={args.sad_rvq_scheme_d_continuous_multiscale_spec_weight}",
                f"++lm.sad_rvq_scheme_d_projector_hidden_mult={args.sad_rvq_scheme_d_projector_hidden_mult}",
                f"++lm.sad_rvq_scheme_d_train_post3_only={str(args.sad_rvq_scheme_d_train_post3_only).lower()}",
                f"++lm.sad_rvq_scheme_d_init_from_codebook={str(args.sad_rvq_scheme_d_init_from_codebook).lower()}",
                f"++lm.sad_rvq_scheme_train_mode={args.sad_rvq_scheme_train_mode}",
                f"++lm.sad_rvq_freeze_main_model={str(args.sad_rvq_freeze_main_model).lower()}",
                f"++lm.sad_rvq_scheme_g_entropy_quantile={args.sad_rvq_scheme_g_entropy_quantile}",
                f"++lm.sad_rvq_scheme_h_min_temp={args.sad_rvq_scheme_h_min_temp}",
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