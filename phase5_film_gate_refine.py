import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_mono(path: Path, fs: int) -> np.ndarray:
    x, src_fs = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    if src_fs != fs:
        x = soxr.resample(x, src_fs, fs)
    return x.astype(np.float32, copy=False)


def stft(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    win = torch.hann_window(n_fft, device=x.device)
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        return_complex=True,
        center=True,
    )


def istft(X: torch.Tensor, length: int, n_fft: int, hop: int) -> torch.Tensor:
    win = torch.hann_window(n_fft, device=X.device)
    return torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=True,
        length=length,
    )


class FiLMGateRefiner(nn.Module):
    """Phase D refiner: residual FiLM with high-confidence bypass."""

    def __init__(
        self,
        base_alpha: float = 0.5,
        max_delta: float = 0.2,
        bypass_center: float = 0.88,
        bypass_sharpness: float = 12.0,
        bypass_strength: float = 0.75,
    ) -> None:
        super().__init__()
        self.base_alpha = float(base_alpha)
        self.max_delta = float(max_delta)
        self.bypass_center = float(bypass_center)
        self.bypass_sharpness = float(bypass_sharpness)
        self.bypass_strength = float(bypass_strength)
        # Small trainable interaction module; backbone is implicit/fixed by inputs.
        self.cond = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.gamma = nn.Conv2d(16, 1, kernel_size=1)
        self.beta = nn.Conv2d(16, 1, kernel_size=1)
        self.out = nn.Conv2d(1, 1, kernel_size=1)

    def forward(
        self,
        mag_addse: torch.Tensor,
        mag_pguse: torch.Tensor,
        mag_noisy: torch.Tensor,
        snr_est_db: torch.Tensor,
        truncate_ratio: float,
    ) -> torch.Tensor:
        # Inputs: [F, T]
        x = torch.stack([torch.log1p(mag_addse), torch.log1p(mag_pguse)], dim=0).unsqueeze(0)
        c = self.cond(x)
        h = x[:, :1]
        g = torch.sigmoid(self.gamma(c))
        b = self.beta(c)
        y = g * h + b
        delta = torch.tanh(self.out(y)) * self.max_delta

        # SNR-driven truncation: high SNR -> only tiny latent movement.
        delta = delta * float(np.clip(truncate_ratio, 0.0, 1.0))

        base_mag = self.base_alpha * mag_addse + (1.0 - self.base_alpha) * mag_pguse
        refined = torch.clamp(base_mag * (1.0 + delta.squeeze(0).squeeze(0)), min=1e-8)

        # High-confidence bypass: when two branches agree, reduce aggressive modification.
        diff = torch.mean(torch.abs(torch.log1p(mag_addse) - torch.log1p(mag_pguse)))
        agreement = torch.exp(-diff)
        bypass = torch.sigmoid(self.bypass_sharpness * (agreement - self.bypass_center)) * self.bypass_strength
        z_gen = bypass * base_mag + (1.0 - bypass) * refined

        # Residual transport in latent magnitude space.
        # High SNR should approach identity on noisy latent, preventing SDR collapse.
        gmf = torch.sigmoid(-0.55 * (snr_est_db - 8.0))
        gmf = torch.clamp(gmf, min=0.02, max=0.98)
        out = mag_noisy + gmf * (z_gen - mag_noisy)
        return torch.clamp(out, min=1e-8)


class WavTripletDataset:
    def __init__(
        self,
        names: Iterable[str],
        addse_dir: Path,
        pguse_dir: Path,
        clean_dir: Path,
        noisy_dir: Path | None,
        fs: int,
    ):
        self.names = list(names)
        self.addse_dir = addse_dir
        self.pguse_dir = pguse_dir
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.fs = fs

    def __len__(self) -> int:
        return len(self.names)

    def get(self, i: int) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.names[i]
        xa = load_mono(self.addse_dir / n, self.fs)
        xp = load_mono(self.pguse_dir / n, self.fs)
        xc = load_mono(self.clean_dir / n, self.fs)
        if self.noisy_dir is None:
            xn = xp
        else:
            xn = load_mono(self.noisy_dir / n, self.fs)

        L = min(len(xa), len(xp), len(xc), len(xn))
        xa = torch.from_numpy(xa[:L])
        xp = torch.from_numpy(xp[:L])
        xc = torch.from_numpy(xc[:L])
        xn = torch.from_numpy(xn[:L])
        return n, xa, xp, xc, xn


def mr_stft_complex_loss(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    scales = [(256, 64), (512, 128), (1024, 256)]
    loss = torch.tensor(0.0, device=y.device)
    for n_fft, hop in scales:
        Y = stft(y, n_fft=n_fft, hop=hop)
        T = stft(t, n_fft=n_fft, hop=hop)
        mag_l1 = torch.mean(torch.abs(torch.log1p(torch.abs(Y)) - torch.log1p(torch.abs(T))))
        cmp_l1 = torch.mean(torch.abs(torch.view_as_real(Y) - torch.view_as_real(T)))
        loss = loss + mag_l1 + cmp_l1
    return loss / len(scales)


def cycle_consistency_loss(
    model: FiLMGateRefiner,
    y: torch.Tensor,
    mag_ref: torch.Tensor,
    mag_noisy: torch.Tensor,
    phase_ref: torch.Tensor,
    snr_est_db: torch.Tensor,
    truncate_ratio: float,
    n_fft: int,
    hop: int,
) -> torch.Tensor:
    Sy = stft(y, n_fft=n_fft, hop=hop)
    mag_y = torch.abs(Sy)
    mag_2 = model(mag_y, mag_ref, mag_noisy, snr_est_db=snr_est_db, truncate_ratio=truncate_ratio)
    y_2 = istft(mag_2 * phase_ref, length=y.numel(), n_fft=n_fft, hop=hop)
    l_time = F.l1_loss(y_2, y)
    l_mag = F.l1_loss(torch.log1p(mag_2), torch.log1p(torch.clamp(mag_y, min=1e-8)))
    return l_time + l_mag


def estimate_snr_db(noisy: torch.Tensor, clean: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p_clean = torch.sum(clean ** 2)
    p_noise = torch.sum((noisy - clean) ** 2)
    return 10.0 * torch.log10((p_clean + eps) / (p_noise + eps))


def truncation_ratio_from_snr(snr_db: torch.Tensor) -> float:
    # <= 0 dB: full correction, >=10 dB: small correction budget around 5%.
    s = float(snr_db.detach().cpu().item())
    if s <= 0.0:
        return 1.0
    if s >= 10.0:
        return 0.05
    return float(1.0 - 0.95 * (s / 10.0))


def complex_loop_constraint(y: torch.Tensor, noisy: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    Sy = stft(y, n_fft=n_fft, hop=hop)
    Sn = stft(noisy, n_fft=n_fft, hop=hop)

    mag_y = torch.abs(Sy)
    mag_n = torch.abs(Sn)

    # Keep generated spectral envelope close to noisy envelope for high-SNR transparency.
    l_env = F.l1_loss(torch.log1p(mag_y), torch.log1p(torch.clamp(mag_n, min=1e-8)))

    # Phase-manifold regularization: magnitude changes should imply bounded phase drift.
    phase_y = torch.angle(Sy)
    phase_n = torch.angle(Sn)
    dphi = torch.atan2(torch.sin(phase_y - phase_n), torch.cos(phase_y - phase_n)).abs()
    dmag = torch.clamp(torch.abs(mag_y - mag_n) / torch.clamp(mag_n, min=1e-6), max=1.0)
    l_phase = F.l1_loss(dphi, 1.2 * dmag)

    return l_env + l_phase


def collect_clean_map(clean_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(clean_root.rglob("*.wav")):
        out[p.name] = p
    return out


def main() -> None:
    parser = argparse.ArgumentParser("Phase C: FiLM-gated refinement from ParallelLite B+")
    parser.add_argument("--addse-dir", required=True)
    parser.add_argument("--pguse-dir", required=True)
    parser.add_argument("--noisy-dir", default="", help="Optional noisy reference dir for SNR-aware transparent transport")
    parser.add_argument("--clean-root", required=True, help="Can contain nested split folders")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ckpt", default="", help="Checkpoint path to save best model")
    parser.add_argument("--init-ckpt", default="", help="Optional checkpoint path for weight initialization")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda-mrstft", type=float, default=0.5)
    parser.add_argument("--lambda-cycle", type=float, default=0.1)
    parser.add_argument("--lambda-complex-loop", type=float, default=0.15)
    parser.add_argument("--base-alpha", type=float, default=0.5)
    parser.add_argument("--max-delta", type=float, default=0.2)
    parser.add_argument("--bypass-center", type=float, default=0.88)
    parser.add_argument("--bypass-sharpness", type=float, default=12.0)
    parser.add_argument("--bypass-strength", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    addse_dir = Path(args.addse_dir)
    pguse_dir = Path(args.pguse_dir)
    noisy_dir = Path(args.noisy_dir) if args.noisy_dir else None
    clean_root = Path(args.clean_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_map = collect_clean_map(clean_root)
    names = [p.name for p in sorted(addse_dir.glob("*.wav")) if (pguse_dir / p.name).exists() and p.name in clean_map]
    if len(names) < 2:
        raise ValueError("Need at least 2 aligned wav files")

    train_names = names[:-2]
    val_names = names[-2:-1]
    test_names = names[-1:]

    def to_dataset(ns: list[str]) -> WavTripletDataset:
        cdir = out_dir / "_tmp_clean_flat"
        cdir.mkdir(parents=True, exist_ok=True)
        for n in ns:
            if not (cdir / n).exists():
                x = load_mono(clean_map[n], args.fs)
                sf.write(cdir / n, x, args.fs)
        return WavTripletDataset(ns, addse_dir, pguse_dir, cdir, noisy_dir, args.fs)

    train_ds = to_dataset(train_names)
    val_ds = to_dataset(val_names)
    infer_ds = to_dataset(names)

    model = FiLMGateRefiner(
        base_alpha=args.base_alpha,
        max_delta=args.max_delta,
        bypass_center=args.bypass_center,
        bypass_sharpness=args.bypass_sharpness,
        bypass_strength=args.bypass_strength,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    ckpt_path = Path(args.ckpt) if args.ckpt else (out_dir / "film_gate_best.pt")

    init_ckpt = args.init_ckpt or args.ckpt
    if init_ckpt:
        init_pack = torch.load(init_ckpt, map_location=device)
        model.load_state_dict(init_pack["model"], strict=False)
        print(f"Loaded init weights from: {init_ckpt}")

    for ep in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for i in range(len(train_ds)):
            _, xa, xp, xc, xn = train_ds.get(i)
            xa = xa.to(device)
            xp = xp.to(device)
            xc = xc.to(device)
            xn = xn.to(device)

            Sa = stft(xa, args.n_fft, args.hop)
            Sp = stft(xp, args.n_fft, args.hop)
            Sn = stft(xn, args.n_fft, args.hop)
            mag_a = torch.abs(Sa)
            mag_p = torch.abs(Sp)
            mag_n = torch.abs(Sn)
            phase_p = Sp / torch.clamp(torch.abs(Sp), min=1e-8)
            snr_est = estimate_snr_db(noisy=xn, clean=xc)
            tr_ratio = truncation_ratio_from_snr(snr_est)

            mag_f = model(mag_a, mag_p, mag_n, snr_est_db=snr_est, truncate_ratio=tr_ratio)
            Sf = mag_f * phase_p
            y = istft(Sf, length=xc.numel(), n_fft=args.n_fft, hop=args.hop)

            l_time = F.l1_loss(y, xc)
            l_spec = mr_stft_complex_loss(y, xc)
            l_cycle = cycle_consistency_loss(
                model=model,
                y=y,
                mag_ref=mag_p,
                mag_noisy=mag_n,
                phase_ref=phase_p,
                snr_est_db=snr_est,
                truncate_ratio=tr_ratio,
                n_fft=args.n_fft,
                hop=args.hop,
            )
            l_cplx = complex_loop_constraint(y=y, noisy=xn, n_fft=args.n_fft, hop=args.hop)
            loss = l_time + args.lambda_mrstft * l_spec + args.lambda_cycle * l_cycle + args.lambda_complex_loop * l_cplx

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(len(val_ds)):
                _, xa, xp, xc, xn = val_ds.get(i)
                xa = xa.to(device)
                xp = xp.to(device)
                xc = xc.to(device)
                xn = xn.to(device)

                Sa = stft(xa, args.n_fft, args.hop)
                Sp = stft(xp, args.n_fft, args.hop)
                Sn = stft(xn, args.n_fft, args.hop)
                snr_est = estimate_snr_db(noisy=xn, clean=xc)
                tr_ratio = truncation_ratio_from_snr(snr_est)
                mag_f = model(torch.abs(Sa), torch.abs(Sp), torch.abs(Sn), snr_est_db=snr_est, truncate_ratio=tr_ratio)
                phase_p = Sp / torch.clamp(torch.abs(Sp), min=1e-8)
                y = istft(mag_f * phase_p, length=xc.numel(), n_fft=args.n_fft, hop=args.hop)

                val_l = F.l1_loss(y, xc) + args.lambda_mrstft * mr_stft_complex_loss(y, xc)
                val_l = val_l + args.lambda_cycle * cycle_consistency_loss(
                    model=model,
                    y=y,
                    mag_ref=torch.abs(Sp),
                    mag_noisy=torch.abs(Sn),
                    phase_ref=phase_p,
                    snr_est_db=snr_est,
                    truncate_ratio=tr_ratio,
                    n_fft=args.n_fft,
                    hop=args.hop,
                )
                val_l = val_l + args.lambda_complex_loop * complex_loop_constraint(
                    y=y,
                    noisy=xn,
                    n_fft=args.n_fft,
                    hop=args.hop,
                )
                val_losses.append(float(val_l.item()))

            mean_val = float(np.mean(val_losses)) if val_losses else float(np.mean(train_losses))
            if mean_val < best_val:
                best_val = mean_val
                torch.save({"model": model.state_dict(), "val_loss": best_val, "epoch": ep}, ckpt_path)

        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(f"[ep {ep:03d}] train={np.mean(train_losses):.6f} val={mean_val:.6f} best={best_val:.6f}")

    print(f"Saved best checkpoint: {ckpt_path}")

    pack = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(pack["model"])
    model.eval()

    wav_out = out_dir / "wav"
    wav_out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i in range(len(infer_ds)):
            n, xa, xp, _, xn = infer_ds.get(i)
            xa = xa.to(device)
            xp = xp.to(device)
            xn = xn.to(device)
            S_a = stft(xa, args.n_fft, args.hop)
            S_p = stft(xp, args.n_fft, args.hop)
            S_n = stft(xn, args.n_fft, args.hop)
            snr_est = estimate_snr_db(noisy=xn, clean=xa)
            tr_ratio = truncation_ratio_from_snr(snr_est)
            mag = model(
                torch.abs(S_a),
                torch.abs(S_p),
                torch.abs(S_n),
                snr_est_db=snr_est,
                truncate_ratio=tr_ratio,
            )
            phase = S_p / torch.clamp(torch.abs(S_p), min=1e-8)
            y = istft(mag * phase, length=xa.numel(), n_fft=args.n_fft, hop=args.hop)
            y = y.detach().cpu().numpy().astype(np.float32, copy=False)
            peak = float(np.max(np.abs(y)) + 1e-8)
            if peak > 1.0:
                y = y / peak
            sf.write(wav_out / n, y, args.fs)

    print(f"Saved enhanced wavs: {wav_out}")
    print(f"Train/Val/Test split sizes: {len(train_names)}/{len(val_names)}/{len(test_names)}")


if __name__ == "__main__":
    main()
