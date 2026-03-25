import argparse
from pathlib import Path

import soundfile as sf
import soxr
import torch
from hydra.utils import instantiate

from addse.lightning import BaseLightningModule
from addse.utils import load_hydra_config


def load_audio(path: Path, target_fs: int) -> torch.Tensor:
    x, fs = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[1] > 1:
        x = x[:, :1]
    if fs != target_fs:
        x = soxr.resample(x, fs, target_fs)
    x = torch.from_numpy(x.T).unsqueeze(0)
    return x


def main() -> None:
    parser = argparse.ArgumentParser("Export ADDSE-enhanced wavs for phase-1 joint testing")
    parser.add_argument("--config", required=True, help="Path to ADDSE yaml config")
    parser.add_argument("--ckpt", required=True, help="Path to ADDSE checkpoint")
    parser.add_argument("--noisy-dir", required=True, help="Input noisy wav directory")
    parser.add_argument("--out-dir", required=True, help="Output enhanced wav directory")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--suffix", default="", help="Optional suffix added before .wav")
    args = parser.parse_args()

    noisy_dir = Path(args.noisy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not noisy_dir.exists():
        raise FileNotFoundError(f"Noisy dir not found: {noisy_dir}")

    base_cfg, _ = load_hydra_config(args.config, overrides=None)
    fs = int(base_cfg.get("fs", 16000))

    lm: BaseLightningModule = instantiate(base_cfg.lm)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(state_dict, dict):
        raise ValueError("Unsupported checkpoint format")
    lm.load_state_dict(state_dict, strict=False)
    lm.eval().to(args.device)

    wavs = sorted(noisy_dir.glob("*.wav"))
    if not wavs:
        raise ValueError(f"No wav found in {noisy_dir}")

    with torch.no_grad():
        for i, wav_path in enumerate(wavs, 1):
            x = load_audio(wav_path, fs).to(args.device)
            y_hat = lm(x)
            y_np = y_hat.squeeze(0).detach().cpu().numpy().T
            out_name = f"{wav_path.stem}{args.suffix}.wav"
            sf.write(out_dir / out_name, y_np, fs)
            if i % 20 == 0 or i == len(wavs):
                print(f"[{i}/{len(wavs)}] exported: {out_name}")

    print(f"Done. Exported {len(wavs)} files to {out_dir}")


if __name__ == "__main__":
    main()
