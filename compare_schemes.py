#!/usr/bin/env python
"""
Quick Comparison: Scheme A vs D
================================

Run both schemes with same configuration to compare:
- Scheme A: Post-5-layer entropy boost
- Scheme D: Learnable gating router (soft fusion)

Usage:
  python compare_schemes.py --epochs 1 --batches 200

This will:
1. Run Scheme A for 1 epoch, 200 batches
2. Save metrics to logs/
3. Run Scheme D for 1 epoch, 200 batches
4. Compare metrics side-by-side
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Scheme A vs D Quick Comparison")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs per scheme")
    parser.add_argument("--batches", type=int, default=200, help="Max batches per epoch")
    parser.add_argument("--reset-logs", action="store_true", help="Clean logs before each run")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.resolve()
    os.chdir(project_root / "addse")
    
    schemes = [
        ("a", "🟢 Scheme A: Entropy Boost (Post-5-Layer)"),
        ("d", "🔵 Scheme D: Learnable Gating (Soft Fusion)"),
    ]
    
    results = {}
    
    print("\n" + "="*80)
    print("📊 SAD-RVQ SCHEME COMPARISON")
    print("="*80)
    
    for scheme_id, scheme_name in schemes:
        print(f"\n\n{'█'*80}")
        print(f"Running: {scheme_name}")
        print(f"Configuration: epochs={args.epochs}, batches={args.batches}")
        print(f"{'█'*80}\n")
        
        cmd = [
            "python", "run_v33.py",
            f"--train-epochs={args.epochs}",
            f"--train-batches={args.batches}",
            "--val-every=1",
            f"--sad-rvq-scheme={scheme_id}",
            "--reset-log-dir",
            "--reset-db",
        ]
        
        print(f"Command: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, cwd=str(project_root / "addse"))
        
        log_file = project_root / "addse" / "logs" / "addse-s-edbase-parallel60-a008-p02-spec" / "metrics.csv"
        if log_file.exists():
            # Extract final metrics
            import pandas as pd
            try:
                df = pd.read_csv(log_file)
                last_train = df[df['train/ce_loss'].notna()].iloc[-1] if any(df['train/ce_loss'].notna()) else None
                last_val = df[df['val/ce_loss'].notna()].iloc[-1] if any(df['val/ce_loss'].notna()) else None
                
                results[scheme_id] = {
                    "name": scheme_name,
                    "train_ce_loss": float(last_train['train/ce_loss']) if last_train is not None else None,
                    "val_ce_loss": float(last_val['val/ce_loss']) if last_val is not None else None,
                }
                
                print(f"\n✅ {scheme_name} completed")
                print(f"   Final train CE_loss: {results[scheme_id]['train_ce_loss']:.4f}")
                print(f"   Final val CE_loss: {results[scheme_id]['val_ce_loss']:.4f}")
            except Exception as e:
                print(f"⚠️ Error extracting metrics: {e}")
        else:
            print(f"⚠️ Metrics file not found: {log_file}")
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📈 COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    if results:
        for scheme_id, metrics in results.items():
            print(f"{metrics['name']}")
            print(f"  Train CE_loss: {metrics['train_ce_loss']:.4f if metrics['train_ce_loss'] else 'N/A'}")
            print(f"  Val CE_loss: {metrics['val_ce_loss']:.4f if metrics['val_ce_loss'] else 'N/A'}\n")
    
    print("\n✅ Comparison complete. Check logs/ directory for detailed metrics.")
