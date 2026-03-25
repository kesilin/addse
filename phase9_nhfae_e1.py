import argparse
from dataclasses import dataclass
from pathlib import Path

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


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def mrstft_loss(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    scales = [(256, 64), (512, 128), (1024, 256)]
    loss = torch.tensor(0.0, device=y.device)
    for n_fft, hop in scales:
        Y = stft(y, n_fft=n_fft, hop=hop)
        T = stft(t, n_fft=n_fft, hop=hop)
        loss = loss + torch.mean(torch.abs(torch.log1p(torch.abs(Y)) - torch.log1p(torch.abs(T))))
        loss = loss + torch.mean(torch.abs(torch.view_as_real(Y) - torch.view_as_real(T)))
    return loss / len(scales)


class PairDataset:
    def __init__(self, clean_dir: Path, noisy_dir: Path, fs: int):
        self.fs = fs
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        clean = sorted(clean_dir.glob("*.wav"))
        if not clean:
            clean = sorted(clean_dir.glob("*.WAV"))
        self.names = [f"{p.stem}.wav" for p in clean if (noisy_dir / f"{p.stem}.wav").exists() or (noisy_dir / f"{p.stem}.WAV").exists()]

    def __len__(self) -> int:
        return len(self.names)

    def get(self, i: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        name = self.names[i]
        cpath = self.clean_dir / name
        if not cpath.exists():
            cpath = self.clean_dir / f"{Path(name).stem}.WAV"
        npath = self.noisy_dir / name
        if not npath.exists():
            npath = self.noisy_dir / f"{Path(name).stem}.WAV"

        c = load_mono(cpath, self.fs)
        n = load_mono(npath, self.fs)
        L = min(len(c), len(n))
        c = torch.from_numpy(c[:L])
        n = torch.from_numpy(n[:L])
        return name, c, n


class MambaLikeBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, d_model)
        self.dw = nn.Conv2d(d_model, d_model, kernel_size=(1, 7), padding=(0, 3), groups=d_model)
        self.pw1 = nn.Conv2d(d_model, 2 * d_model, kernel_size=1)
        self.pw2 = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.dw(h)
        h = self.pw1(h)
        a, b = torch.chunk(h, 2, dim=1)
        h = F.silu(a) * torch.sigmoid(b)
        h = self.pw2(h)
        return x + h


class AttentionTimeBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.proj = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        tok = x.mean(dim=2).transpose(1, 2)  # [B,T,C]
        tok2 = self.norm(tok)
        out, _ = self.attn(tok2, tok2, tok2, need_weights=False)
        out = out.transpose(1, 2).unsqueeze(2).expand(-1, -1, f, -1)
        out = self.proj(out)
        return x + out


class ResidualConstrainedFiLM(nn.Module):
    def __init__(self, d_model: int, max_delta: float = 0.2):
        super().__init__()
        self.max_delta = max_delta
        self.gate = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.bias = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(cond))
        b = self.bias(cond)
        y = g * x + b
        d = torch.tanh(y - x) * self.max_delta
        return x + d


class NHFAE_E1(nn.Module):
    def __init__(self, d_model: int = 96, n_layers: int = 8, n_heads: int = 4, n_mag_bins: int = 64):
        super().__init__()
        self.n_mag_bins = n_mag_bins

        self.stem = nn.Sequential(
            nn.Conv2d(3, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        blocks = []
        films = []
        for i in range(n_layers):
            if i % 2 == 0:
                blocks.append(MambaLikeBlock(d_model))
            else:
                blocks.append(AttentionTimeBlock(d_model, n_heads))
            films.append(ResidualConstrainedFiLM(d_model, max_delta=0.15))
        self.blocks = nn.ModuleList(blocks)
        self.films = nn.ModuleList(films)

        self.geom_cond = nn.Conv2d(d_model, d_model, kernel_size=1)

        self.identity_head = nn.Conv2d(d_model, n_mag_bins, kernel_size=1)
        self.flow_head = nn.Conv2d(d_model, 2, kernel_size=1)  # [v_mag, v_phase]

        self.snr_est = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(d_model // 2, 1, kernel_size=1),
        )

        centers = torch.linspace(0.0, 1.0, steps=n_mag_bins)
        self.register_buffer("mag_centers", centers)

    def quantize_mag(self, logits: torch.Tensor, mag_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # logits: [B,K,F,T], mag_norm in [0,1]
        probs = F.softmax(logits, dim=1)
        soft = torch.sum(probs * self.mag_centers.view(1, -1, 1, 1), dim=1)
        idx = torch.argmax(logits, dim=1)
        hard = self.mag_centers[idx]
        q = soft + (hard - soft).detach()  # STE

        tgt = torch.clamp((mag_norm * (self.n_mag_bins - 1)).round().long(), 0, self.n_mag_bins - 1)
        dce = F.cross_entropy(logits, tgt)
        return q, dce

    def forward(self, noisy_stft: torch.Tensor) -> dict[str, torch.Tensor]:
        mag_n = torch.abs(noisy_stft)
        phase_n = torch.angle(noisy_stft)

        feat = torch.stack([
            torch.log1p(mag_n),
            torch.cos(phase_n),
            torch.sin(phase_n),
        ], dim=1)
        x = self.stem(feat)

        for blk, film in zip(self.blocks, self.films):
            x = blk(x)
            cond = self.geom_cond(x)
            x = film(x, cond)

        logits = self.identity_head(x)
        flow = self.flow_head(x)
        v_mag = flow[:, 0]
        v_phase = np.pi * torch.tanh(flow[:, 1])

        # Learned local SNR estimator mapped to [-5, 20] dB.
        snr_map = -5.0 + 25.0 * torch.sigmoid(self.snr_est(x)).squeeze(1)
        fatc_gate = torch.sigmoid(-0.9 * (snr_map - 8.0))
        fatc_gate = torch.clamp(fatc_gate, min=0.01, max=0.99)

        mag_norm = mag_n / (torch.amax(mag_n, dim=(1, 2), keepdim=True) + 1e-8)
        q_mag, dce = self.quantize_mag(logits, mag_norm)

        mag_gen = torch.clamp(mag_norm + 0.15 * torch.tanh(v_mag), 0.0, 1.0)
        mag_mix = fatc_gate * mag_gen + (1.0 - fatc_gate) * mag_norm

        # In high-SNR region, identity path dominates magnitude and only phase gets slight correction.
        phase_out = phase_n + fatc_gate * v_phase

        return {
            "logits": logits,
            "dce": dce,
            "mag_norm": mag_norm,
            "mag_mix": mag_mix,
            "phase_out": phase_out,
            "fatc_gate": fatc_gate,
            "v_mag": v_mag,
            "v_phase": v_phase,
        }


@dataclass
class TrainCfg:
    clean_dir: Path
    noisy_dir: Path
    out_dir: Path
    fs: int = 16000
    n_fft: int = 512
    hop: int = 192
    epochs: int = 2
    lr: float = 2e-4
    lambda_dce: float = 1.0
    lambda_cfm: float = 0.5
    lambda_cycle: float = 0.2
    lambda_mrstft: float = 0.2
    device: str = "cuda"


def train(cfg: TrainCfg) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
    ds = PairDataset(cfg.clean_dir, cfg.noisy_dir, cfg.fs)
    if len(ds) < 2:
        raise ValueError("Need at least 2 paired samples")

    model = NHFAE_E1().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    ckpt_dir = cfg.out_dir / "ckpt"
    wav_dir = cfg.out_dir / "wav"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for i in range(len(ds)):
            _, clean, noisy = ds.get(i)
            clean = clean.to(device)
            noisy = noisy.to(device)

            S_clean = stft(clean, cfg.n_fft, cfg.hop)
            S_noisy = stft(noisy, cfg.n_fft, cfg.hop)

            out = model(S_noisy.unsqueeze(0))
            mag_n = torch.abs(S_noisy).unsqueeze(0)
            mag_scale = torch.amax(mag_n, dim=(1, 2), keepdim=True) + 1e-8
            mag_out = out["mag_mix"] * mag_scale
            phase_out = out["phase_out"]
            S_out = mag_out * torch.exp(1j * phase_out)
            y = istft(S_out.squeeze(0), length=clean.numel(), n_fft=cfg.n_fft, hop=cfg.hop)

            tgt_mag = torch.abs(S_clean)
            tgt_phase = torch.angle(S_clean)
            d_phase = wrap_to_pi(tgt_phase - torch.angle(S_noisy))

            l_dce = out["dce"]
            l_cfm = F.l1_loss(out["v_phase"].squeeze(0), d_phase) + F.l1_loss(out["v_mag"].squeeze(0), (tgt_mag - torch.abs(S_noisy)) / (mag_scale.squeeze(0) + 1e-8))
            l_cycle = F.l1_loss(stft(y, cfg.n_fft, cfg.hop), S_out.squeeze(0).detach())
            l_mr = mrstft_loss(y, clean)

            loss = cfg.lambda_dce * l_dce + cfg.lambda_cfm * l_cfm + cfg.lambda_cycle * l_cycle + cfg.lambda_mrstft * l_mr

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))

        m = float(np.mean(losses))
        print(f"[ep {ep:03d}] loss={m:.6f}")
        if m < best:
            best = m
            torch.save({"model": model.state_dict(), "loss": best}, ckpt_dir / "best.pt")

    print(f"Saved best checkpoint to: {ckpt_dir / 'best.pt'}")

    # Export inference wavs with best ckpt.
    pack = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(pack["model"])
    model.eval()

    with torch.no_grad():
        for i in range(len(ds)):
            name, _, noisy = ds.get(i)
            noisy = noisy.to(device)
            S_noisy = stft(noisy, cfg.n_fft, cfg.hop)
            out = model(S_noisy.unsqueeze(0))
            mag_n = torch.abs(S_noisy).unsqueeze(0)
            mag_scale = torch.amax(mag_n, dim=(1, 2), keepdim=True) + 1e-8
            mag_out = out["mag_mix"] * mag_scale
            S_out = mag_out * torch.exp(1j * out["phase_out"])
            y = istft(S_out.squeeze(0), length=noisy.numel(), n_fft=cfg.n_fft, hop=cfg.hop)
            y = y.detach().cpu().numpy().astype(np.float32, copy=False)
            peak = float(np.max(np.abs(y)) + 1e-8)
            if peak > 1.0:
                y = y / peak
            sf.write(wav_dir / name, y, cfg.fs)

    print(f"Saved NHFAE-E1 wavs to: {wav_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train NHFAE E1 (single-backbone dual-head)")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = TrainCfg(
        clean_dir=Path(args.clean_dir),
        noisy_dir=Path(args.noisy_dir),
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    train(cfg)
