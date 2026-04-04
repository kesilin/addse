"""
Experiment 1: Single Batch Overfitting (Sanity Check)
======================================================

Purpose:
  - Verify that gradients flow correctly through the entire architecture
  - Confirm router gate network learns and oscillates in [0, 1]
  - Verify post-5-layer CE_Loss drops from ~6.0 to <0.1 in a waterfall pattern

Setup:
  - batch_size=2 (fixed - only one batch)
  - no_shuffle=True (always feed same batch)
  - max_steps=200
  - scheme='d' (enable learnable gating)

Indicators of Success:
  ✅ CE_Loss: 6.0 → quickly drops to 0.1 (waterfall descent)
  ✅ Gate mean: oscillates in range [0.3, 0.7] (not stuck at 0.0 or 1.0)
  ✅ Gate std: >0.05 (not constant)
  ✅ No NaN/Inf in loss or gate
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


def run_sanity_check(scheme="d", max_steps=200):
    """Run single-batch overfitting test"""
    
    print("\n" + "="*80)
    print("EXPERIMENT 1: SINGLE BATCH OVERFITTING (SANITY CHECK)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Scheme: {scheme.upper()}")
    print(f"  Batch Size: 2 (fixed)")
    print(f"  Shuffle: OFF (same batch every step)")
    print(f"  Max Steps: {max_steps}")
    print(f"  Expected: CE_Loss [6.0 → <0.1], gate_mean ∈ [0.3, 0.7]")
    print("="*80 + "\n")
    
    # Clean up old logs to avoid interference
    bad_log_dir = project_root / "logs" / "addse-s-edbase-parallel60-a008-p02-spec"
    if bad_log_dir.exists():
        print(f"Cleaning old log directory: {bad_log_dir}")
        shutil.rmtree(bad_log_dir)
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    
    # Build Hydra overrides for single-batch overfitting
    overrides = [
        # ========== 1. SINGLE BATCH CONFIG ==========
        f"++trainer.max_epochs=1",  # Run for 1 epoch
        f"++trainer.limit_train_batches={max_steps}",  # Limit to N steps
        f"++dm.train_dataloader.batch_size=2",  # Small batch for memory
        f"++dm.train_dataloader.shuffle=false",  # NO SHUFFLE - same batch always
        f"++dm.train_dataloader.num_workers=0",  # Single worker to ensure same batch
        f"++dm.train_dataset.length={max_steps * 2}",  # max_steps batches × batch_size=2
        
        # ========== 2. PREVENT DATA VARIATION ==========
        f"++dm.train_dataset.resume=false",
        f"++dm.val_dataset.resume=false",
        f"++num_workers=0",
        
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
        
        # ========== 5. NO VALIDATION (speed up) ==========
        f"++trainer.check_val_every_n_epoch=10000",  # Disable validation
        
        # ========== 6. LOGGING ==========
        "++trainer.log_every_n_steps=1",  # Log every step for monitoring
    ]
    
    print(f"\nStarting training with Scheme: {scheme.upper()}")
    print(f"Monitor metrics:")
    print(f"  - train/ce_loss (should drop from ~6.0 to <0.1)")
    if scheme.lower() == "d":
        print(f"  - train/scheme_d_gate_mean (should be in [0.0, 1.0])")
        print(f"  - train/scheme_d_gate_std (should be >0.05)")
    print("\n")
    
    train_func(
        config_file="configs/addse-s-edbase-parallel60-a008-p02-spec.yaml",
        overrides=overrides,
        overwrite=True,
        wandb=False
    )
    
    GlobalHydra.instance().clear()
    print("\n" + "="*80)
    print("✅ EXPERIMENT 1 COMPLETED")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Check logs/addse-s-edbase-parallel60-a008-p02-spec/logs.txt")
    print("  2. Look for:")
    print(f"     - train/ce_loss: Started ~6.0, ended <0.1 (waterfall)?")
    print(f"     - train/scheme_d_gate_mean: Oscillates in [0.3, 0.7]?")
    print(f"     - train/scheme_d_gate_std: >0.05 (not frozen)?")
    print("\n  If ✅ YES to all → Run Experiment 2 (Generalization Check)")
    print("  If ❌ NO to any → Check error logs and debug\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Experiment 1: Sanity Check (Single Batch Overfitting)")
    parser.add_argument("--scheme", type=str, default="d", choices=["d", "a", "baseline"],
                       help="Which scheme to test (d=learnable gating, a=entropy boost, baseline=none)")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Number of steps to run (default 100 for speed, 200 for thorough)")
    args = parser.parse_args()
    
    run_sanity_check(scheme=args.scheme, max_steps=args.max_steps)
