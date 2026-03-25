import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def load_mono(path: Path, fs: int) -> np.ndarray:
    x, src_fs = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    if src_fs != fs:
        x = soxr.resample(x, src_fs, fs)
    return x.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser("Create serial mix input: x = beta*addse + (1-beta)*noisy")
    parser.add_argument("--addse-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--fs", type=int, default=16000)
    args = parser.parse_args()

    if not (0.0 <= args.beta <= 1.0):
        raise ValueError("--beta must be in [0,1]")

    addse_dir = Path(args.addse_dir)
    noisy_dir = Path(args.noisy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(addse_dir.glob("*.wav"))
    if not files:
        raise ValueError(f"No wav found in {addse_dir}")

    kept = 0
    for ap in files:
        npy = noisy_dir / ap.name
        if not npy.exists():
            continue

        xa = load_mono(ap, args.fs)
        xn = load_mono(npy, args.fs)
        L = min(len(xa), len(xn))
        xa = xa[:L]
        xn = xn[:L]

        y = args.beta * xa + (1.0 - args.beta) * xn
        peak = float(np.max(np.abs(y)) + 1e-8)
        if peak > 1.0:
            y = y / peak

        sf.write(out_dir / ap.name, y, args.fs)
        kept += 1

    print(f"Saved {kept} mixed wavs to {out_dir}")


if __name__ == "__main__":
    main()
