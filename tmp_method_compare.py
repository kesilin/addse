import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from addse.metrics import PESQMetric, SDRMetric, STOIMetric

fs = 16000
root = Path("D:/Users/KSL/PycharmProjects/the_sound")
clean_dir = root / "PGUSE-main/AudioSamples/clean"
noisy_dir = root / "PGUSE-main/AudioSamples/degraded"
methods = {
    "ADDSE_only": root / "addse/outputs/phase1/addse_wav",
    "PGUSE_only_round2_N1_A10": root / "addse/outputs/phase2/grid/wav/PGUSE_only_N1_A10",
    "ADDSE_plus_PGUSE_round2_N1_A10": root / "addse/outputs/phase2/grid/wav/N1_A10",
}

buckets = [(-5, 0), (0, 5), (5, 10), (10, 15)]


def bucket_of(v: float) -> str | None:
    for lo, hi in buckets:
        if lo <= v < hi:
            return f"[{lo},{hi}]"
    return None


def load(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[1] > 1:
        x = x[:, :1]
    if sr != fs:
        x = soxr.resample(x, sr, fs)
    return x.T


def snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    n = noisy - clean
    return float(10 * np.log10((np.sum(clean**2) + 1e-8) / (np.sum(n**2) + 1e-8)))


pesq = PESQMetric(fs)
estoi = STOIMetric(fs, extended=True)
sdr = SDRMetric(scale_invariant=False)

names = sorted([p.name for p in clean_dir.glob("*.wav")])
all_rows: list[dict] = []

for mname, mdir in methods.items():
    vals = []
    bmap = {f"[{lo},{hi}]": [] for lo, hi in buckets}

    for n in names:
        cp, yp, pp = clean_dir / n, noisy_dir / n, mdir / n
        if not (cp.exists() and yp.exists() and pp.exists()):
            continue

        c, y, p = load(cp), load(yp), load(pp)
        L = min(c.shape[-1], y.shape[-1], p.shape[-1])
        c, y, p = c[:, :L], y[:, :L], p[:, :L]

        row = {
            "pesq": float(pesq(p, c)),
            "estoi": float(estoi(p, c)),
            "sdr": float(sdr(p, c)),
            "pesq_noisy": float(pesq(y, c)),
            "estoi_noisy": float(estoi(y, c)),
            "sdr_noisy": float(sdr(y, c)),
        }
        vals.append(row)

        b = bucket_of(snr(c, y))
        if b:
            bmap[b].append(row)

    if not vals:
        continue

    all_rows.append(
        {
            "method": mname,
            "count": len(vals),
            "pesq": float(np.mean([v["pesq"] for v in vals])),
            "estoi": float(np.mean([v["estoi"] for v in vals])),
            "sdr": float(np.mean([v["sdr"] for v in vals])),
            "delta_pesq_vs_noisy": float(np.mean([v["pesq"] - v["pesq_noisy"] for v in vals])),
            "delta_estoi_vs_noisy": float(np.mean([v["estoi"] - v["estoi_noisy"] for v in vals])),
            "delta_sdr_vs_noisy": float(np.mean([v["sdr"] - v["sdr_noisy"] for v in vals])),
        }
    )

    bout = root / "addse/outputs/phase2/grid/tables" / f"compare_buckets_{mname}.csv"
    bout.parent.mkdir(parents=True, exist_ok=True)
    with bout.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "count", "method", "pesq", "estoi", "sdr", "delta_pesq", "delta_estoi", "delta_sdr"])
        for lo, hi in buckets:
            k = f"[{lo},{hi}]"
            r = bmap[k]
            if not r:
                w.writerow([k, 0, mname, "NA", "NA", "NA", "NA", "NA", "NA"])
                continue
            w.writerow(
                [
                    k,
                    len(r),
                    mname,
                    f"{np.mean([x['pesq'] for x in r]):.4f}",
                    f"{np.mean([x['estoi'] for x in r]):.4f}",
                    f"{np.mean([x['sdr'] for x in r]):.4f}",
                    f"{np.mean([x['pesq'] - x['pesq_noisy'] for x in r]):.4f}",
                    f"{np.mean([x['estoi'] - x['estoi_noisy'] for x in r]):.4f}",
                    f"{np.mean([x['sdr'] - x['sdr_noisy'] for x in r]):.4f}",
                ]
            )

out = root / "addse/outputs/phase2/grid/tables/method_compare_overall.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "method",
            "count",
            "pesq",
            "estoi",
            "sdr",
            "delta_pesq_vs_noisy",
            "delta_estoi_vs_noisy",
            "delta_sdr_vs_noisy",
        ],
    )
    w.writeheader()
    w.writerows(all_rows)

print(f"saved {out}")
for r in all_rows:
    print(r)
