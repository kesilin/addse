import sqlite3
import subprocess
import sys
from pathlib import Path

CONFIG = "configs/addse-s-edbase-ft.yaml"
CKPT = "logs/addse-s-edbase-ft/checkpoints/epoch=07-val_loss=3.54.ckpt"
NOISE_DIR = "data/chunks/edbase_noise_original/"
NUM_EXAMPLES = 20

BUCKETS = [
    (-5.0, 0.0),
    (0.0, 5.0),
    (5.0, 10.0),
    (10.0, 15.0),
]


def fetch_metrics(db_path: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT metric, name, AVG(value)
        FROM results
        GROUP BY metric, name
        """
    )
    rows = cur.fetchall()
    conn.close()

    data = {}
    for metric, name, avg_value in rows:
        data.setdefault(name, {})[metric] = float(avg_value)
    return data


def run_bucket(low: float, high: float):
    bucket_name = f"{int(low)}_{int(high)}".replace("-", "m")
    db_name = f"eval_bucket_{bucket_name}.db"
    db_path = Path(db_name)

    cmd = [
        sys.executable,
        "-m",
        "addse.app",
        "eval",
        CONFIG,
        CKPT,
        "--device",
        "cuda",
        "--output-db",
        db_name,
        "--overwrite",
        "--num-consumers",
        "0",
        "--num-examples",
        str(NUM_EXAMPLES),
        "lm.num_steps=64",
        "--noisy",
        f"train_noise_chunks={NOISE_DIR}",
        f"eval.dsets.edbase-local.noise_dataset.input_dir={NOISE_DIR}",
        f"eval.dsets.edbase-local.snr_range=[{low:.1f},{high:.1f}]",
    ]

    print(f"\n=== Running bucket [{low}, {high}] ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Bucket [{low}, {high}] failed with code {result.returncode}")

    metrics = fetch_metrics(db_path)
    model_name = "addse-s-edbase-ft"
    model = metrics.get(model_name, {})
    noisy = metrics.get("noisy", {})

    return {
        "bucket": f"[{low:.0f},{high:.0f}]",
        "model_pesq": model.get("pesq"),
        "noisy_pesq": noisy.get("pesq"),
        "delta_pesq": (model.get("pesq") - noisy.get("pesq")) if model.get("pesq") is not None and noisy.get("pesq") is not None else None,
        "model_estoi": model.get("estoi"),
        "noisy_estoi": noisy.get("estoi"),
        "delta_estoi": (model.get("estoi") - noisy.get("estoi")) if model.get("estoi") is not None and noisy.get("estoi") is not None else None,
        "model_sdr": model.get("sdr"),
        "noisy_sdr": noisy.get("sdr"),
        "delta_sdr": (model.get("sdr") - noisy.get("sdr")) if model.get("sdr") is not None and noisy.get("sdr") is not None else None,
        "db": db_name,
    }


def fmt(x):
    return "NA" if x is None else f"{x:.2f}"


def main():
    all_rows = []
    for low, high in BUCKETS:
        row = run_bucket(low, high)
        all_rows.append(row)

    print("\n\n===== SNR Bucket Summary =====")
    print("bucket | pesq(model/noisy/delta) | estoi(model/noisy/delta) | sdr(model/noisy/delta) | db")
    for r in all_rows:
        print(
            f"{r['bucket']} | "
            f"{fmt(r['model_pesq'])}/{fmt(r['noisy_pesq'])}/{fmt(r['delta_pesq'])} | "
            f"{fmt(r['model_estoi'])}/{fmt(r['noisy_estoi'])}/{fmt(r['delta_estoi'])} | "
            f"{fmt(r['model_sdr'])}/{fmt(r['noisy_sdr'])}/{fmt(r['delta_sdr'])} | "
            f"{r['db']}"
        )


if __name__ == "__main__":
    main()
