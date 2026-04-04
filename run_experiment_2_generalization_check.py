"""
Experiment 2: Generalization Check (Validation Alignment)
==========================================================

Purpose:
  - If Experiment 1 passed, verify the scheme generalizes to validation data
  - Check that PESQ improves from baseline 1.591 to >1.65 (ideally 1.7+)
  - Confirm gradient flow works on real data with natural variation

Setup:
  - batch_size=8 (normal)
  - shuffle=true (normal data loading)
  - max_batches=1000 or 1 epoch (whichever comes first)
  - scheme='d' (if sanity check passed)

Success Indicators:
  ✅ Validation PESQ: 1.591 → 1.65+ (at least +2.5% improvement)
  ✅ No abnormal loss spikes (smooth training curve)
  ✅ SI-SDR: improves from -5.9 baseline

Timeline:
  - ~30 minutes for 1000 batches
  - ~1-2 hours for 1 full epoch
"""

import argparse
import os
import shutil
import sys
import torch
from pathlib import Path
from hydra.core.global_hydra import GlobalHydra

project_root = Path(__file__).parent.resolve()
os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from addse.app.train import train as train_func
from addse.app.eval import eval as eval_func


def run_generalization_check(scheme="d", max_batches=500):
    """Run validation alignment test with real data"""
    
    print("\n" + "="*80)
    print("EXPERIMENT 2: GENERALIZATION CHECK (VALIDATION ALIGNMENT)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Scheme: {scheme.upper()}")
    print(f"  Batch Size: 8 (normal)")
    print(f"  Shuffle: ON (normal data loading)")
    print(f"  Max Batches: {max_batches}")
    print(f"  Expected: PESQ 1.591 → 1.65+ (improvement ≥ +2.5%)")
    print("="*80 + "\n")
    
    # Clean up old logs
    bad_log_dir = project_root / "logs" / "addse-s-edbase-parallel60-a008-p02-spec"
    if bad_log_dir.exists():
        print(f"Cleaning old log directory: {bad_log_dir}")
        shutil.rmtree(bad_log_dir)
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    
    # Build Hydra overrides for generalization check
    overrides = [
        # ========== 1. TRAINING CONFIG ==========
        f"++trainer.max_epochs=1",
        f"++trainer.limit_train_batches={max_batches}",
        f"++dm.train_dataloader.batch_size=8",  # Normal batch size
        f"++dm.train_dataloader.shuffle=true",  # WITH shuffle (normal operation)
        f"++dm.train_dataset.length=150",  # Normal training group count
        
        # ========== 2. NORMAL DATA LOADING ==========
        f"++dm.train_dataloader.num_workers=8",  # Multi-worker loading
        f"++dm.train_dataloader.pin_memory=true",
        f"++dm.train_dataset.resume=false",
        f"++dm.val_dataset.resume=false",
        
        # ========== 3. SCHEME SELECTION ==========
        f"++lm.sad_rvq_scheme={scheme}",
        f"++lm.sad_rvq_scheme_d_enabled=true",
        f"++lm.sad_rvq_scheme_a_weight=0.5",
        f"++lm.sad_rvq_scheme_d_gate_entropy_weight=0.1",
        
        # ========== 4. LOSS WEIGHTS (STANDARD) ==========
        "++lm.spec_loss_weight=1.0",
        "++lm.wave_l1_weight=2.0",
        "++lm.si_sdr_weight=0.2",
        "++lm.residual_l1_weight=0.0",
        "++lm.direct_residual_weight=1.0",
        "++lm.metricgan_plus_enabled=false",
        "++lm.metricgan_weight=0.0",
        
        # ========== 5. VALIDATION EVERY 100 BATCHES ==========
        f"++trainer.check_val_every_n_epoch=1",  # Validate every epoch
        
        # ========== 6. LOGGING ==========
        "++trainer.log_every_n_steps=10",  # Log every 10 steps
    ]
    
    print(f"\nStarting training with Scheme: {scheme.upper()}")
    print(f"Monitor metrics:")
    print(f"  - train/ce_loss (should be stable, ~1-2)")
    print(f"  - val/pesq_score (target: >1.65, baseline 1.591)")
    if scheme.lower() == "d":
        print(f"  - train/scheme_d_gate_mean (should show meaningful variation)")
    print("\n")
    
    train_func(
        config_file="configs/addse-s-edbase-parallel60-a008-p02-spec.yaml",
        overrides=overrides,
        overwrite=True,
        wandb=False
    )
    
    GlobalHydra.instance().clear()
    
    # Evaluate on validation set
    print("\n" + "="*80)
    print("Running final evaluation on validation set...")
    print("="*80 + "\n")
    
    ckpt_dir = project_root / "logs" / "addse-s-edbase-parallel60-a008-p02-spec" / "checkpoints"
    ckpts = list(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
    
    if ckpts:
        best_ckpt = str(max(ckpts, key=lambda p: os.path.getmtime(p)))
        print(f"Using checkpoint: {best_ckpt}")
        
        eval_func(
            config_file="configs/addse-s-edbase-parallel60-a008-p02-spec.yaml",
            checkpoint=best_ckpt,
            overrides=[
                f"lm.num_steps=200",
                f"eval.dsets.edbase-local.length=60",
                f"lm.deterministic_eval=false",
            ],
            output_dir="saved_audio_exp2",
            output_db="v33_exp2.db",
            num_examples=60,
            noisy=True,
            clean=True,
            device="auto",
            overwrite=True,
            num_consumers=0,
        )
    else:
        print("❌ No checkpoint found - evaluation skipped")
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT 2 COMPLETED")
    print("="*80)
    print("\nResult Analysis:")
    print("  📊 Check logs/addse-s-edbase-parallel60-a008-p02-spec/metrics.csv for:")
    print(f"     - Final val/pesq_score vs baseline 1.591")
    print(f"     - If PESQ improved by ≥2.5% → Scheme {scheme.upper()} is working! 🎉")
    print(f"     - If PESQ stalled or degraded → Debug the loss landscape\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Experiment 2: Generalization Check")
    parser.add_argument("--scheme", type=str, default="d", choices=["d", "a", "baseline"],
                       help="Which scheme to test (d=learnable gating, a=entropy boost)")
    parser.add_argument("--max-batches", type=int, default=500,
                       help="Max training batches (default 500 ≈ 30 min)")
    args = parser.parse_args()
    
    run_generalization_check(scheme=args.scheme, max_batches=args.max_batches)
