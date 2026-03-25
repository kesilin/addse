import argparse
import csv
import subprocess
from pathlib import Path

import yaml


def run_cmd(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_overall_row(csv_path: Path, method_name: str) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("method") == method_name:
                return row
    raise ValueError(f"Method {method_name} not found in {csv_path}")


def append_summary(summary_csv: Path, set_name: str, row: dict[str, str]) -> None:
    rows = []
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

    normalized = []
    for r in rows:
        normalized.append(
            {
                "set": r.get("set", r.get("\ufeffset", "")),
                "method": r.get("method", ""),
                "pesq": r.get("pesq", ""),
                "estoi": r.get("estoi", ""),
                "sdr": r.get("sdr", ""),
                "delta_pesq_vs_noisy": r.get("delta_pesq_vs_noisy", ""),
                "delta_estoi_vs_noisy": r.get("delta_estoi_vs_noisy", ""),
                "delta_sdr_vs_noisy": r.get("delta_sdr_vs_noisy", ""),
            }
        )

    new_row = {
        "set": set_name,
        "method": row["method"],
        "pesq": row["pesq"],
        "estoi": row["estoi"],
        "sdr": row["sdr"],
        "delta_pesq_vs_noisy": row["delta_pesq_vs_noisy"],
        "delta_estoi_vs_noisy": row["delta_estoi_vs_noisy"],
        "delta_sdr_vs_noisy": row["delta_sdr_vs_noisy"],
    }

    normalized = [r for r in normalized if not (r["set"] == set_name and r["method"] == new_row["method"])]
    normalized.append(new_row)
    normalized = sorted(normalized, key=lambda x: (x["set"], -float(x["pesq"])))

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "set",
                "method",
                "pesq",
                "estoi",
                "sdr",
                "delta_pesq_vs_noisy",
                "delta_estoi_vs_noisy",
                "delta_sdr_vs_noisy",
            ],
        )
        w.writeheader()
        w.writerows(normalized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run NHFAE E0 ablations")
    parser.add_argument("--config", default="configs/nhfae_phase_e0.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    data = cfg["data"]
    fatc = cfg["fatc"]
    abl = cfg["ablation"]
    out_root = Path(cfg["output"]["root"])
    summary_csv = Path(cfg["output"]["summary_csv"])

    py = "D:/Users/KSL/PycharmProjects/the_sound/.venv/Scripts/python.exe"

    variants = [
        ("PhaseD3_E0_base", fatc["g_low"], fatc["g_high"], fatc["gamma"], fatc["conf_thr"], fatc["conf_slope"]),
    ]
    if abl.get("run_no_freq_adapt", False):
        variants.append(("PhaseD3_E0_no_freq_adapt", fatc["g_low"], fatc["g_low"], fatc["gamma"], fatc["conf_thr"], fatc["conf_slope"]))
    if abl.get("run_no_confidence_gate", False):
        variants.append(("PhaseD3_E0_no_conf_gate", fatc["g_low"], fatc["g_high"], fatc["gamma"], 999.0, 1.0))

    for method_name, g_low, g_high, gamma, conf_thr, conf_slope in variants:
        run_dir = out_root / method_name
        wav_dir = run_dir / "wav"

        run_cmd(
            [
                py,
                "phase7_phase_first_flow.py",
                "--addse-dir",
                data["addse_dir"],
                "--pguse-dir",
                data["pguse_dir"],
                "--noisy-dir",
                data["noisy_dir"],
                "--out-dir",
                str(wav_dir),
                "--g-low",
                str(g_low),
                "--g-high",
                str(g_high),
                "--gamma",
                str(gamma),
                "--conf-thr",
                str(conf_thr),
                "--conf-slope",
                str(conf_slope),
            ]
        )

        overall_csv = run_dir / "overall.csv"
        bucket_csv = run_dir / "bucket.csv"
        run_cmd(
            [
                py,
                "phase4_compare_metrics.py",
                "--clean-dir",
                data["clean_dir"],
                "--noisy-dir",
                data["noisy_dir"],
                "--method",
                f"{method_name}={wav_dir}",
                "--out-overall-csv",
                str(overall_csv),
                "--out-bucket-csv",
                str(bucket_csv),
                "--buckets=-5,0,5,10,15,20",
            ]
        )

        row = read_overall_row(overall_csv, method_name)
        append_summary(summary_csv, "snr_10_15", row)
        print(f"Updated summary with {method_name}")

    print("E0 ablation finished")
