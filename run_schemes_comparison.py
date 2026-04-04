#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start: Run SAD-RVQ Schemes A, B, C, D
This script orchestrates training runs for all 4 schemes using run_v33.py
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent


def run_scheme(scheme: str, epochs: int = 5, batch_size: int = 100) -> bool:
    """
    Run a single scheme training
    
    Returns: True if successful, False otherwise
    """
    print(f"\n{'='*100}")
    print(f"[TRAINING] Scheme {scheme.upper()} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "addse/run_v33.py"),
        f"--epochs={epochs}",
        f"--batch-size={batch_size}",
        f"--sad-rvq-scheme={scheme}",
        "--enable-logging",
    ]
    
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        success = result.returncode == 0
        scheme_status = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n[RESULT] Scheme {scheme.upper()}: {scheme_status}")
        return success
    except Exception as e:
        print(f"[ERROR] Failed to run scheme {scheme}: {e}")
        return False


def run_all_schemes(epochs: int = 5, batch_size: int = 100, schemes: list[str] | None = None):
    """
    Run all specified schemes sequentially
    """
    if schemes is None:
        schemes = ["baseline", "a", "d", "c", "b"]  # Priority order
    
    results = {}
    
    for scheme in schemes:
        results[scheme] = run_scheme(scheme, epochs, batch_size)
    
    print(f"\n{'='*100}")
    print("[FINAL REPORT]")
    print(f"{'='*100}")
    for scheme, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"Scheme {scheme.upper():8} | {status}")
    
    # Generate comparison command
    print(f"\n[NEXT STEP] Compare results:")
    print(f"  python addse/probe_architecture_surgery.py --compare-schemes")
    
    return all(results.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SAD-RVQ Schemes A/B/C/D comparison experiments"
    )
    parser.add_argument(
        "--schemes",
        type=str,
        default="baseline,a,d",
        help="Comma-separated list of schemes to run (baseline,a,b,c,d)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs per scheme (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size (default: 100)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: baseline + A only (for fast testing)",
    )
    
    args = parser.parse_args()
    
    if args.quick:
        schemes = ["baseline", "a"]
        epochs = 2
    else:
        schemes = [s.strip().lower() for s in args.schemes.split(",")]
        epochs = args.epochs
    
    print(f"\n{'='*100}")
    print(f"[SAD-RVQ SCHEMES] Comparison Experiment")
    print(f"{'='*100}")
    print(f"Schemes to run: {', '.join([s.upper() for s in schemes])}")
    print(f"Epochs per scheme: {epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*100}\n")
    
    success = run_all_schemes(epochs=epochs, batch_size=args.batch_size, schemes=schemes)
    sys.exit(0 if success else 1)
