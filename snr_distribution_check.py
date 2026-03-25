import argparse
from typing import Sequence

import torch
from hydra.utils import instantiate

from addse.lightning import DataModule
from addse.utils import load_hydra_config, seed_all


def parse_bins(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) < 1:
        raise ValueError("bins must contain at least one value")
    values = sorted(values)
    return values


def compute_snr_db(clean: torch.Tensor, noisy: torch.Tensor, eps: float = 1e-12) -> float:
    noise = noisy - clean
    p_clean = clean.pow(2).mean()
    p_noise = noise.pow(2).mean()
    snr = 10.0 * torch.log10((p_clean + eps) / (p_noise + eps))
    return float(snr.item())


def bucketize(snrs: torch.Tensor, edges: Sequence[float]) -> list[tuple[str, int, float]]:
    edges_t = torch.tensor(edges, dtype=snrs.dtype)
    n = int(snrs.numel())
    out: list[tuple[str, int, float]] = []

    left_count = int((snrs < edges_t[0]).sum().item())
    out.append((f"(-inf, {edges_t[0].item():.1f})", left_count, 100.0 * left_count / n))

    for i in range(len(edges_t) - 1):
        lo = edges_t[i].item()
        hi = edges_t[i + 1].item()
        count = int(((snrs >= lo) & (snrs < hi)).sum().item())
        out.append((f"[{lo:.1f}, {hi:.1f})", count, 100.0 * count / n))

    right_count = int((snrs >= edges_t[-1]).sum().item())
    out.append((f"[{edges_t[-1].item():.1f}, +inf)", right_count, 100.0 * right_count / n))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SNR distribution from ADDSE dynamic mixing dataset")
    parser.add_argument("--config", default="configs/addse-edbase-quick.yaml", help="Path to ADDSE YAML config")
    parser.add_argument("--num-samples", type=int, default=800, help="Number of mixed samples to draw")
    parser.add_argument(
        "--bins",
        default="-3,0,5,10,15",
        help="Comma-separated SNR bin edges in dB, e.g. -3,0,5,10,15",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override, can be used multiple times",
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    edges = parse_bins(args.bins)

    overrides = [f"dm.train_dataset.length={args.num_samples}"] + args.override
    cfg, _ = load_hydra_config(args.config, overrides=overrides)

    seed_all(int(cfg.seed))

    dm: DataModule = instantiate(cfg.dm)
    dm.setup("fit")
    if dm.train_dset is None:
        raise RuntimeError("train dataset is not initialized")

    iterator = iter(dm.train_dset)
    snrs = []
    for _ in range(args.num_samples):
        noisy, clean, _ = next(iterator)
        snrs.append(compute_snr_db(clean, noisy))

    snr_t = torch.tensor(snrs, dtype=torch.float32)

    print("=== ADDSE Dynamic Mixing SNR Distribution ===")
    print(f"config: {args.config}")
    print(f"samples: {args.num_samples}")
    print(f"snr_range in config: {cfg.dm.train_dataset.snr_range}")
    print(f"min/mean/max: {snr_t.min().item():.3f} / {snr_t.mean().item():.3f} / {snr_t.max().item():.3f} dB")
    print(f"std: {snr_t.std(unbiased=False).item():.3f} dB")

    q = torch.quantile(snr_t, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    print("quantiles p10/p25/p50/p75/p90:", ", ".join([f"{x.item():.3f}" for x in q]), "dB")

    print("\n-- bucket ratios --")
    for label, count, pct in bucketize(snr_t, edges):
        print(f"{label:>18}: {count:4d} ({pct:6.2f}%)")

    low = int((snr_t < 0.0).sum().item())
    mid = int(((snr_t >= 0.0) & (snr_t < 10.0)).sum().item())
    high = int((snr_t >= 10.0).sum().item())
    n = int(snr_t.numel())
    print("\n-- grouped summary --")
    print(f"low  (<0 dB):      {low:4d} ({100.0 * low / n:6.2f}%)")
    print(f"mid  [0,10) dB:    {mid:4d} ({100.0 * mid / n:6.2f}%)")
    print(f"high (>=10 dB):    {high:4d} ({100.0 * high / n:6.2f}%)")


if __name__ == "__main__":
    main()
