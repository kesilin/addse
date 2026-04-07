import argparse
import os
import re
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
    parser.add_argument("--config", type=str, default="configs/addse-s-edbase-parallel60-a008-p02-spec.yaml", help="Hydra config file path")
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--train-batches", type=int, default=5)
    parser.add_argument("--total-steps", type=int, default=0, help="If >0, override to single-epoch training with this many batches")
    parser.add_argument("--train-groups", type=int, default=150, help="Synthetic training groups / dataset length")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-shuffle", action="store_true", help="Enable train dataloader shuffle")
    parser.add_argument("--eval-examples", type=int, default=60)
    parser.add_argument("--eval-min-duration", type=float, default=1.0, help="Minimum duration in seconds for evaluation items")
    parser.add_argument("--eval-min-active-duration", type=float, default=0.5, help="Minimum active-speech duration in seconds")
    parser.add_argument("--eval-active-threshold", type=float, default=0.01, help="Amplitude threshold for active-speech masking")
    parser.add_argument("--eval-seed", type=int, default=42, help="Seed to keep evaluation sampling deterministic")
    parser.add_argument("--eval-steps", type=int, default=200, help="Diffusion steps during eval; use 200 for strict evaluation")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--train-num-workers", type=int, default=8, help="Number of train dataloader workers")
    parser.add_argument("--reset-log-dir", action="store_true", help="Delete log dir before training")
    parser.add_argument("--reset-db", action="store_true", help="Delete output DB before evaluation")
    parser.add_argument("--deterministic-eval", action="store_true", help="Use deterministic token decoding during eval")
    parser.add_argument("--output-db", type=str, default="v33_decoupled.db")
    parser.add_argument("--output-dir", type=str, default="saved_audio_v33")
    parser.add_argument("--disable-wave-residual", action="store_true", help="Disable wave residual side-stream branch")
    parser.add_argument("--continuous-residual-train-only", action="store_true", help="Freeze all modules except continuous STFT residual branch")
    parser.add_argument("--continuous-residual-joint-train", action="store_true", help="Train backbone and residual branch jointly with separate learning rates")
    parser.add_argument("--continuous-residual-lr-scale", type=float, default=3.0, help="LR scale when continuous residual train-only is enabled")
    parser.add_argument("--continuous-residual-backbone-lr-scale", type=float, default=0.1, help="Backbone LR scale for joint training")
    parser.add_argument("--continuous-residual-probe-scale", type=float, default=1.0, help="Scale factor for residual probe injection")
    parser.add_argument("--continuous-residual-init-bias", type=float, default=0.0, help="Initial bias for the residual predictor output conv")
    parser.add_argument("--continuous-residual-zero-mean", action="store_true", help="Force residual prediction to zero-mean per sample")
    parser.add_argument("--continuous-residual-warmup-steps", type=int, default=0, help="Warmup steps using direct residual L1 objective")
    parser.add_argument("--continuous-residual-warmup-weight", type=float, default=1.0, help="Weight for direct residual L1 warmup objective")
    parser.add_argument("--continuous-residual-warmup-residual-only", action="store_true", help="Use residual warmup objective only during warmup window")
    parser.add_argument("--continuous-residual-warmup-mag-weight", type=float, default=0.0, help="STFT magnitude loss weight during residual warmup")
    parser.add_argument("--continuous-residual-warmup-phase-weight", type=float, default=0.0, help="STFT phase consistency loss weight during residual warmup")
    parser.add_argument("--continuous-residual-warmup-stft-nfft", type=int, default=512, help="NFFT used by warmup STFT mag/phase loss")
    parser.add_argument("--continuous-residual-warmup-stft-hop", type=int, default=128, help="Hop length used by warmup STFT mag/phase loss")
    parser.add_argument("--continuous-residual-warmup-stft-win", type=int, default=512, help="Window length used by warmup STFT mag/phase loss")
    parser.add_argument("--continuous-gain-margin", type=float, default=0.0, help="Target minimum gain_vs_noisy margin")
    parser.add_argument("--continuous-gain-penalty-weight", type=float, default=0.0, help="Penalty weight when gain_vs_noisy is below margin")
    parser.add_argument("--continuous-residual-use-multiscale-stft", action="store_true", help="Use multiscale STFT features in the residual branch")
    parser.add_argument("--continuous-residual-predictor-channels", type=int, default=256, help="Hidden channels for the residual predictor")
    parser.add_argument("--continuous-residual-predictor-blocks", type=int, default=3, help="Number of temporal residual blocks in the predictor")
    parser.add_argument("--continuous-residual-predictor-dilation-rates", type=str, default="1,2,4,8", help="Comma-separated dilation rates for the atrous pyramid in the predictor")
    parser.add_argument("--continuous-residual-stft-low-stride", type=int, default=8, help="Pooling stride for the low-frequency refinement head")
    parser.add_argument("--continuous-residual-stft-input-mode", type=str, default="noisy", choices=["noisy", "noisy_base_residual"], help="Input feature mode for STFT residual branch")
    parser.add_argument("--continuous-residual-direct-clean-target", action="store_true", help="Let STFT branch directly predict clean waveform instead of residual")
    parser.add_argument("--continuous-branch-wave-l1-weight", type=float, default=0.0, help="Independent L1 supervision weight for branch output")
    parser.add_argument("--continuous-branch-si-sdr-weight", type=float, default=0.0, help="Independent SI-SDR supervision weight for branch output (direct clean mode)")
    parser.add_argument("--continuous-residual-multiscale-nffts", type=str, default="512,1024,2048", help="Comma-separated FFT sizes for multiscale STFT")
    parser.add_argument("--continuous-residual-multiscale-hops", type=str, default="128,256,512", help="Comma-separated hop sizes for multiscale STFT")
    parser.add_argument("--continuous-residual-multiscale-wins", type=str, default="512,1024,2048", help="Comma-separated window sizes for multiscale STFT")
    parser.add_argument("--direct-residual-weight", type=float, default=1.0, help="Weight for explicit DEX residual supervision")
    parser.add_argument("--residual-l1-weight", type=float, default=0.0, help="Fallback latent residual L1 weight")
    parser.add_argument("--si-sdr-weight", type=float, default=0.2, help="Wave-domain SI-SDR weight for this run")
    parser.add_argument("--optimizer-lr", type=float, default=6.0e-5, help="Optimizer LR override")
    parser.add_argument("--spec-loss-weight", type=float, default=1.0, help="Spectral loss weight")
    parser.add_argument("--wave-l1-weight", type=float, default=2.0, help="Wave L1 loss weight")
    parser.add_argument("--alpha-init-prob", type=float, default=0.05, help="Initial sigmoid probability for alpha gates")
    parser.add_argument("--print-alpha", action="store_true", help="Print alpha logs in progress bar (handled in lightning logs)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducible runs")
    parser.add_argument("--disable-pretrained-ckpt", action="store_true", help="Disable lm.pretrained_ckpt loading to avoid mismatch interference")
    parser.add_argument("--pretrained-ckpt", type=str, default=None, help="Override lm.pretrained_ckpt; empty string disables preload")
    parser.add_argument("--trainable-param-patterns", type=str, default=None, help="Comma-separated lm.trainable_param_patterns; empty string clears it")
    parser.add_argument("--audit-pretrained-load", action="store_true", help="Print pretrained missing/unexpected key audit")
    parser.add_argument("--discrete-audit-log-interval", type=int, default=20, help="Print interval for force-discrete CE/accuracy audit")
    parser.add_argument("--eval-ckpt-policy", type=str, default="best", choices=["best", "last"], help="Checkpoint policy for evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate with --checkpoint-path")
    parser.add_argument("--checkpoint-path", type=str, default="", help="Checkpoint path used when --eval-only is enabled")
    parser.add_argument("--force-discrete-only", action="store_true", help="Force discrete trunk only and disable continuous branch in inference/training path")
    parser.add_argument("--late-fusion-cont-weight", type=float, default=1.0, help="Late-fusion weight applied to continuous residual branch")
    parser.add_argument("--continuous-residual-discrete-prior-weight", type=float, default=0.3, help="Weight of discrete coarse prior injected into continuous branch input")
    parser.add_argument("--continuous-residual-explicit-l1-weight", type=float, default=0.5, help="Explicit residual L1 supervision weight in continuous enhanced loss")
    parser.add_argument("--discrete-branch-lr-scale", type=float, default=1.0, help="LR scale applied to discrete trunk params under force_discrete_only")

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
    parser.add_argument("--sad-rvq-scheme-d-continuous-full-ce-weight", type=float, default=0.0,
                       help="Full discrete CE weight in continuous residual mode (for serial+parallel joint optimization)")
    parser.add_argument("--sad-rvq-scheme-d-continuous-loss-scale", type=float, default=1.0,
                       help="Global scale for continuous coarse/enhanced SI-SDR primary loss")
    parser.add_argument("--sad-rvq-scheme-d-continuous-stage1-epochs", type=int, default=0,
                       help="Early epochs to prioritize discrete CE in continuous mode")
    parser.add_argument("--sad-rvq-scheme-d-continuous-stage1-steps", "--stage1_steps", dest="sad_rvq_scheme_d_continuous_stage1_steps", type=int, default=0,
                       help="Early train steps to prioritize discrete CE in continuous mode")
    parser.add_argument("--sad-rvq-scheme-d-continuous-stage1-cont-weight-scale", type=float, default=0.6,
                       help="Scale factor on continuous coarse/enhanced weights during stage1")
    parser.add_argument("--sad-rvq-scheme-d-continuous-stage1-full-ce-scale", type=float, default=1.8,
                       help="Scale factor on full CE weight during stage1")
    parser.add_argument("--sad-rvq-scheme-d-continuous-stage1-front3-ce-scale", type=float, default=1.2,
                       help="Scale factor on front3 CE weight during stage1")
    parser.add_argument("--sad-rvq-scheme-d-continuous-stage1-freeze-continuous", action="store_true",
                       help="Freeze continuous branch grads during stage1 to prioritize discrete trunk")
    parser.add_argument("--sad-rvq-scheme-d-continuous-stage2-discrete-weight", type=float, default=0.7,
                       help="Stage2 weight for discrete auxiliary CE loss in continuous mode")
    parser.add_argument("--sad-rvq-scheme-d-continuous-stage2-continuous-weight", type=float, default=0.3,
                       help="Stage2 weight for continuous residual objectives in continuous mode")
    parser.add_argument("--continuous-residual-discrete-grad-clip", type=float, default=1.0,
                       help="Gradient clipping norm for discrete/backbone params in continuous training")
    parser.add_argument("--continuous-residual-continuous-grad-clip", type=float, default=0.6,
                       help="Gradient clipping norm for continuous branch params in continuous training")
    parser.add_argument("--continuous-residual-grad-log-interval", type=int, default=20,
                       help="Log interval (steps) for discrete/continuous gradient mean diagnostics")
    parser.add_argument("--stage2-dis-weight", type=float, default=None,
                       help="Alias of --sad-rvq-scheme-d-continuous-stage2-discrete-weight")
    parser.add_argument("--stage2-cont-weight", type=float, default=None,
                       help="Alias of --sad-rvq-scheme-d-continuous-stage2-continuous-weight")
    parser.add_argument("--disable-logits-interaction", action="store_true", default=True,
                       help="Force disable continuous branch interaction on discrete logits")
    parser.add_argument("--enable-logits-interaction", action="store_true",
                       help="Explicitly enable continuous branch interaction on discrete logits")
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

    if args.total_steps > 0:
        args.train_epochs = 1
        args.train_batches = int(args.total_steps)

    if args.stage2_dis_weight is not None:
        args.sad_rvq_scheme_d_continuous_stage2_discrete_weight = float(args.stage2_dis_weight)
    if args.stage2_cont_weight is not None:
        args.sad_rvq_scheme_d_continuous_stage2_continuous_weight = float(args.stage2_cont_weight)

    disable_logits_interaction = bool(args.disable_logits_interaction and not args.enable_logits_interaction)
    print(f"--- logits交互状态: {str((not disable_logits_interaction)).lower()} ---")
    trainable_patterns_override = "++lm.trainable_param_patterns=[]" if args.force_discrete_only else None

    pretrained_ckpt_override = "++lm.pretrained_ckpt=null" if args.disable_pretrained_ckpt else None
    if args.pretrained_ckpt is not None:
        if args.pretrained_ckpt.strip() == "":
            pretrained_ckpt_override = "++lm.pretrained_ckpt=null"
        else:
            pretrained_ckpt_override = f"++lm.pretrained_ckpt={args.pretrained_ckpt.strip()}"

    if args.trainable_param_patterns is not None:
        if args.trainable_param_patterns.strip() == "":
            trainable_patterns_override = "++lm.trainable_param_patterns=[]"
        else:
            patterns = [p.strip() for p in args.trainable_param_patterns.split(",") if p.strip()]
            trainable_patterns_override = f"++lm.trainable_param_patterns={patterns}"

    multiscale_nffts = [int(x) for x in args.continuous_residual_multiscale_nffts.split(",") if x.strip()]
    multiscale_hops = [int(x) for x in args.continuous_residual_multiscale_hops.split(",") if x.strip()]
    multiscale_wins = [int(x) for x in args.continuous_residual_multiscale_wins.split(",") if x.strip()]
    predictor_dilation_rates = [int(x) for x in args.continuous_residual_predictor_dilation_rates.split(",") if x.strip()]

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

    selected_ckpt = ""

    if not args.eval_only:
        train_func(
            config_file=args.config,
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
            f"++lm.continuous_residual_train_only={str(args.continuous_residual_train_only).lower()}",
            f"++lm.continuous_residual_joint_train={str(args.continuous_residual_joint_train).lower()}",
            f"++lm.continuous_residual_lr_scale={args.continuous_residual_lr_scale}",
            f"++lm.continuous_residual_backbone_lr_scale={args.continuous_residual_backbone_lr_scale}",
            f"++lm.continuous_residual_probe_scale={args.continuous_residual_probe_scale}",
            f"++lm.continuous_residual_init_bias={args.continuous_residual_init_bias}",
            f"++lm.continuous_residual_zero_mean={str(args.continuous_residual_zero_mean).lower()}",
            f"++lm.continuous_residual_warmup_steps={args.continuous_residual_warmup_steps}",
            f"++lm.continuous_residual_warmup_weight={args.continuous_residual_warmup_weight}",
            f"++lm.continuous_residual_warmup_residual_only={str(args.continuous_residual_warmup_residual_only).lower()}",
            f"++lm.continuous_residual_warmup_mag_weight={args.continuous_residual_warmup_mag_weight}",
            f"++lm.continuous_residual_warmup_phase_weight={args.continuous_residual_warmup_phase_weight}",
            f"++lm.continuous_residual_warmup_stft_nfft={args.continuous_residual_warmup_stft_nfft}",
            f"++lm.continuous_residual_warmup_stft_hop={args.continuous_residual_warmup_stft_hop}",
            f"++lm.continuous_residual_warmup_stft_win={args.continuous_residual_warmup_stft_win}",
            f"++lm.continuous_gain_margin={args.continuous_gain_margin}",
            f"++lm.continuous_gain_penalty_weight={args.continuous_gain_penalty_weight}",
            f"++lm.force_discrete_only={str(args.force_discrete_only).lower()}",
            f"++lm.audit_pretrained_load={str(args.audit_pretrained_load).lower()}",
            f"++lm.discrete_audit_log_interval={int(args.discrete_audit_log_interval)}",
            f"++lm.discrete_branch_lr_scale={args.discrete_branch_lr_scale}",
            f"++lm.late_fusion_cont_weight={args.late_fusion_cont_weight}",
            f"++lm.continuous_residual_discrete_prior_weight={args.continuous_residual_discrete_prior_weight}",
            f"++lm.continuous_residual_explicit_l1_weight={args.continuous_residual_explicit_l1_weight}",
            *([trainable_patterns_override] if trainable_patterns_override is not None else []),
            f"++lm.continuous_residual_use_multiscale_stft={str(args.continuous_residual_use_multiscale_stft).lower()}",
            f"++lm.continuous_residual_predictor_channels={args.continuous_residual_predictor_channels}",
            f"++lm.continuous_residual_predictor_blocks={args.continuous_residual_predictor_blocks}",
            f"++lm.continuous_residual_predictor_dilation_rates={predictor_dilation_rates}",
            f"++lm.continuous_residual_stft_low_stride={args.continuous_residual_stft_low_stride}",
            f"++lm.continuous_residual_stft_input_mode={args.continuous_residual_stft_input_mode}",
            f"++lm.continuous_residual_direct_clean_target={str(args.continuous_residual_direct_clean_target).lower()}",
            f"++lm.continuous_branch_wave_l1_weight={args.continuous_branch_wave_l1_weight}",
            f"++lm.continuous_branch_si_sdr_weight={args.continuous_branch_si_sdr_weight}",
            f"++lm.continuous_residual_multiscale_nffts={multiscale_nffts}",
            f"++lm.continuous_residual_multiscale_hops={multiscale_hops}",
            f"++lm.continuous_residual_multiscale_wins={multiscale_wins}",

            # ==== 2. 硬件加速（榨干性能，帮你省时间） ====
            f"++dm.train_dataloader.num_workers={args.train_num_workers}",   # 开启多线程读数据，别让 GPU 等 CPU
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
            f"++lm.sad_rvq_scheme_d_continuous_full_ce_weight={args.sad_rvq_scheme_d_continuous_full_ce_weight}",
            f"++lm.sad_rvq_scheme_d_continuous_loss_scale={args.sad_rvq_scheme_d_continuous_loss_scale}",
            f"++lm.sad_rvq_scheme_d_continuous_stage1_epochs={args.sad_rvq_scheme_d_continuous_stage1_epochs}",
            f"++lm.sad_rvq_scheme_d_continuous_stage1_steps={args.sad_rvq_scheme_d_continuous_stage1_steps}",
            f"++lm.sad_rvq_scheme_d_continuous_stage1_cont_weight_scale={args.sad_rvq_scheme_d_continuous_stage1_cont_weight_scale}",
            f"++lm.sad_rvq_scheme_d_continuous_stage1_full_ce_scale={args.sad_rvq_scheme_d_continuous_stage1_full_ce_scale}",
            f"++lm.sad_rvq_scheme_d_continuous_stage1_front3_ce_scale={args.sad_rvq_scheme_d_continuous_stage1_front3_ce_scale}",
            f"++lm.sad_rvq_scheme_d_continuous_stage1_freeze_continuous={str(args.sad_rvq_scheme_d_continuous_stage1_freeze_continuous).lower()}",
            f"++lm.sad_rvq_scheme_d_continuous_stage2_discrete_weight={args.sad_rvq_scheme_d_continuous_stage2_discrete_weight}",
            f"++lm.sad_rvq_scheme_d_continuous_stage2_continuous_weight={args.sad_rvq_scheme_d_continuous_stage2_continuous_weight}",
            f"++lm.continuous_residual_discrete_grad_clip={args.continuous_residual_discrete_grad_clip}",
            f"++lm.continuous_residual_continuous_grad_clip={args.continuous_residual_continuous_grad_clip}",
            f"++lm.continuous_residual_grad_log_interval={args.continuous_residual_grad_log_interval}",
            f"++lm.model.interaction_on_logits={str((not disable_logits_interaction)).lower()}",
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
            *([pretrained_ckpt_override] if pretrained_ckpt_override is not None else []),
            ],
            overwrite=args.reset_log_dir, wandb=False
        )

        GlobalHydra.instance().clear()
        ckpt_dir = project_root / "logs" / "addse-s-edbase-parallel60-a008-p02-spec" / "checkpoints"
        ckpts = list(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
        if ckpts:
            def _parse_val_loss(path: Path) -> float | None:
                m = re.search(r"val_loss=([0-9]+\.[0-9]+)", path.name)
                return float(m.group(1)) if m else None

            if args.eval_ckpt_policy == "last":
                selected_ckpt = str(max(ckpts, key=lambda p: os.path.getmtime(p)))
            else:
                val_loss_ckpts = [(p, _parse_val_loss(p)) for p in ckpts]
                val_loss_ckpts = [(p, v) for p, v in val_loss_ckpts if v is not None]
                if val_loss_ckpts:
                    selected_ckpt = str(min(val_loss_ckpts, key=lambda x: x[1])[0])
                else:
                    selected_ckpt = str(max(ckpts, key=lambda p: os.path.getmtime(p)))
        else:
            print("--- 未找到 checkpoint，跳过评估。请检查 trainer 回调与验证频率。---")
            sys.exit(1)
    else:
        selected_ckpt = args.checkpoint_path.strip()
        if not selected_ckpt:
            print("--- eval-only 模式需要提供 --checkpoint-path ---")
            sys.exit(1)
        if not os.path.exists(selected_ckpt):
            print(f"--- checkpoint 不存在: {selected_ckpt} ---")
            sys.exit(1)
        print(f"--- eval-only: 使用指定 checkpoint 评估: {selected_ckpt} ---")

    print(f"--- 正在使用权重进行评估: {selected_ckpt} ---")
    eval_func(
            config_file=args.config,
            checkpoint=selected_ckpt,
            overrides=[
                f"lm.num_steps={args.eval_steps}",
                f"lm.wave_residual_enabled={str((not args.disable_wave_residual)).lower()}",
                f"eval.dsets.edbase-local.length={args.eval_examples}",
                f"++eval_min_duration={args.eval_min_duration}",
                f"lm.deterministic_eval={str(args.deterministic_eval).lower()}",
                f"++lm.force_discrete_only={str(args.force_discrete_only).lower()}",
                f"++lm.audit_pretrained_load={str(args.audit_pretrained_load).lower()}",
                f"++lm.discrete_audit_log_interval={int(args.discrete_audit_log_interval)}",
                f"++lm.discrete_branch_lr_scale={args.discrete_branch_lr_scale}",
                f"++lm.late_fusion_cont_weight={args.late_fusion_cont_weight}",
                f"++lm.continuous_residual_discrete_prior_weight={args.continuous_residual_discrete_prior_weight}",
                f"++lm.continuous_residual_explicit_l1_weight={args.continuous_residual_explicit_l1_weight}",
                *([trainable_patterns_override] if trainable_patterns_override is not None else []),
                f"++lm.continuous_residual_train_only={str(args.continuous_residual_train_only).lower()}",
                f"++lm.continuous_residual_joint_train={str(args.continuous_residual_joint_train).lower()}",
                f"++lm.continuous_residual_lr_scale={args.continuous_residual_lr_scale}",
                f"++lm.continuous_residual_backbone_lr_scale={args.continuous_residual_backbone_lr_scale}",
                f"++lm.continuous_residual_probe_scale={args.continuous_residual_probe_scale}",
                f"++lm.continuous_residual_init_bias={args.continuous_residual_init_bias}",
                f"++lm.continuous_residual_zero_mean={str(args.continuous_residual_zero_mean).lower()}",
                f"++lm.continuous_residual_warmup_steps={args.continuous_residual_warmup_steps}",
                f"++lm.continuous_residual_warmup_weight={args.continuous_residual_warmup_weight}",
                f"++lm.continuous_residual_warmup_residual_only={str(args.continuous_residual_warmup_residual_only).lower()}",
                f"++lm.continuous_residual_warmup_mag_weight={args.continuous_residual_warmup_mag_weight}",
                f"++lm.continuous_residual_warmup_phase_weight={args.continuous_residual_warmup_phase_weight}",
                f"++lm.continuous_residual_warmup_stft_nfft={args.continuous_residual_warmup_stft_nfft}",
                f"++lm.continuous_residual_warmup_stft_hop={args.continuous_residual_warmup_stft_hop}",
                f"++lm.continuous_residual_warmup_stft_win={args.continuous_residual_warmup_stft_win}",
                f"++lm.continuous_gain_margin={args.continuous_gain_margin}",
                f"++lm.continuous_gain_penalty_weight={args.continuous_gain_penalty_weight}",
                f"++lm.continuous_residual_use_multiscale_stft={str(args.continuous_residual_use_multiscale_stft).lower()}",
                f"++lm.continuous_residual_predictor_channels={args.continuous_residual_predictor_channels}",
                f"++lm.continuous_residual_predictor_blocks={args.continuous_residual_predictor_blocks}",
                f"++lm.continuous_residual_stft_low_stride={args.continuous_residual_stft_low_stride}",
                f"++lm.continuous_residual_stft_input_mode={args.continuous_residual_stft_input_mode}",
                f"++lm.continuous_residual_direct_clean_target={str(args.continuous_residual_direct_clean_target).lower()}",
                f"++lm.continuous_branch_wave_l1_weight={args.continuous_branch_wave_l1_weight}",
                f"++lm.continuous_branch_si_sdr_weight={args.continuous_branch_si_sdr_weight}",
                f"++lm.continuous_residual_multiscale_nffts={multiscale_nffts}",
                f"++lm.continuous_residual_multiscale_hops={multiscale_hops}",
                f"++lm.continuous_residual_multiscale_wins={multiscale_wins}",
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
                f"++lm.sad_rvq_scheme_d_continuous_full_ce_weight={args.sad_rvq_scheme_d_continuous_full_ce_weight}",
                f"++lm.sad_rvq_scheme_d_continuous_loss_scale={args.sad_rvq_scheme_d_continuous_loss_scale}",
                f"++lm.sad_rvq_scheme_d_continuous_stage1_epochs={args.sad_rvq_scheme_d_continuous_stage1_epochs}",
                f"++lm.sad_rvq_scheme_d_continuous_stage1_steps={args.sad_rvq_scheme_d_continuous_stage1_steps}",
                f"++lm.sad_rvq_scheme_d_continuous_stage1_cont_weight_scale={args.sad_rvq_scheme_d_continuous_stage1_cont_weight_scale}",
                f"++lm.sad_rvq_scheme_d_continuous_stage1_full_ce_scale={args.sad_rvq_scheme_d_continuous_stage1_full_ce_scale}",
                f"++lm.sad_rvq_scheme_d_continuous_stage1_front3_ce_scale={args.sad_rvq_scheme_d_continuous_stage1_front3_ce_scale}",
                f"++lm.sad_rvq_scheme_d_continuous_stage1_freeze_continuous={str(args.sad_rvq_scheme_d_continuous_stage1_freeze_continuous).lower()}",
                f"++lm.sad_rvq_scheme_d_continuous_stage2_discrete_weight={args.sad_rvq_scheme_d_continuous_stage2_discrete_weight}",
                f"++lm.sad_rvq_scheme_d_continuous_stage2_continuous_weight={args.sad_rvq_scheme_d_continuous_stage2_continuous_weight}",
                f"++lm.continuous_residual_discrete_grad_clip={args.continuous_residual_discrete_grad_clip}",
                f"++lm.continuous_residual_continuous_grad_clip={args.continuous_residual_continuous_grad_clip}",
                f"++lm.continuous_residual_grad_log_interval={args.continuous_residual_grad_log_interval}",
                f"++lm.model.interaction_on_logits={str((not disable_logits_interaction)).lower()}",
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
                *([pretrained_ckpt_override] if pretrained_ckpt_override is not None else []),
            ],
            output_dir=args.output_dir,
            output_db=args.output_db,
            num_examples=args.eval_examples,
            eval_min_duration=args.eval_min_duration,
            eval_min_active_duration=args.eval_min_active_duration,
            eval_active_threshold=args.eval_active_threshold,
            eval_seed=args.eval_seed,
            noisy=True,
            clean=True,
            device=args.device,
            overwrite=True,
            num_consumers=0,
        )