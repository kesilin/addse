import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch


def load_mono(path: Path, fs: int) -> np.ndarray:
    x, src_fs = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    if src_fs != fs:
        x = soxr.resample(x, src_fs, fs)
    return x.astype(np.float32, copy=False)


def stft(x: np.ndarray, n_fft: int, hop: int) -> torch.Tensor:
    tx = torch.from_numpy(x)
    win = torch.hann_window(n_fft)
    return torch.stft(
        tx,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        return_complex=True,
        center=True,
    )


def istft(X: torch.Tensor, length: int, n_fft: int, hop: int) -> np.ndarray:
    win = torch.hann_window(n_fft)
    y = torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=True,
        length=length,
    )
    return y.cpu().numpy().astype(np.float32, copy=False)


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def main() -> None:
    parser = argparse.ArgumentParser("Phase D.3: Phase-First Flow with frequency-adaptive transparency")
    parser.add_argument("--addse-dir", required=True)
    parser.add_argument("--pguse-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=192)
    parser.add_argument("--g-low", type=float, default=0.03, help="Low-frequency phase correction gain")
    parser.add_argument("--g-high", type=float, default=0.45, help="High-frequency phase correction gain")
    parser.add_argument("--gamma", type=float, default=1.8, help="Frequency gain curvature")
    parser.add_argument("--conf-thr", type=float, default=0.25, help="Log-mag disagreement threshold")
    parser.add_argument("--conf-slope", type=float, default=10.0, help="Confidence sigmoid slope")
    args = parser.parse_args()

    addse_dir = Path(args.addse_dir)
    pguse_dir = Path(args.pguse_dir)
    noisy_dir = Path(args.noisy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(noisy_dir.glob("*.wav"))
    if not files:
        files = sorted(noisy_dir.glob("*.WAV"))
    if not files:
        raise ValueError(f"No noisy wav found in {noisy_dir}")

    for npth in files:
        name = f"{npth.stem}.wav"
        apth = addse_dir / name
        ppth = pguse_dir / name
        if not apth.exists() or not ppth.exists():
            continue

        xa = load_mono(apth, args.fs)
        xp = load_mono(ppth, args.fs)
        xn = load_mono(npth, args.fs)
        L = min(len(xa), len(xp), len(xn))
        xa = xa[:L]
        xp = xp[:L]
        xn = xn[:L]

        Sa = stft(xa, args.n_fft, args.hop)
        Sp = stft(xp, args.n_fft, args.hop)
        Sn = stft(xn, args.n_fft, args.hop)

        mag_n = torch.abs(Sn)
        phase_n = torch.angle(Sn)
        phase_p = torch.angle(Sp)
        phase_delta = wrap_to_pi(phase_p - phase_n)

        # Frequency-adaptive transparency: allow more phase correction at high frequencies.
        F = mag_n.shape[0]
        f = torch.linspace(0.0, 1.0, steps=F).unsqueeze(1)
        g_freq = args.g_low + (args.g_high - args.g_low) * torch.pow(f, args.gamma)

        # Confidence from cross-branch agreement; strong disagreement reduces phase transport.
        log_diff = torch.abs(torch.log1p(torch.abs(Sa)) - torch.log1p(torch.abs(Sp)))
        conf = torch.sigmoid(args.conf_slope * (args.conf_thr - log_diff))

        phase_out = phase_n + g_freq * conf * phase_delta

        # Phase-first mode: keep noisy magnitude, only transport phase residual.
        Sout = mag_n * torch.exp(1j * phase_out)
        y = istft(Sout, length=L, n_fft=args.n_fft, hop=args.hop)

        peak = float(np.max(np.abs(y)) + 1e-8)
        if peak > 1.0:
            y = y / peak
        sf.write(out_dir / name, y, args.fs)

    print(f"Saved Phase-First outputs to: {out_dir}")


if __name__ == "__main__":
    main()
