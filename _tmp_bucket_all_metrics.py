import csv
import pathlib

root = pathlib.Path(r"D:/Users/KSL/PycharmProjects/the_sound/addse/outputs/final60_snr_m5_10")

bucket_map = {}
with open(root / "fixed60" / "manifest.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        bucket_map[pathlib.Path(row["name"]).stem] = row["bucket"]

rows = []
with open(root / "final_stats" / "per_sample_seed_metrics.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        row["bucket"] = bucket_map.get(row["name"])
        for k in ["pesq", "estoi", "sdr", "dp", "de", "dd"]:
            row[k] = float(row[k])
        rows.append(row)

for b in ["[-5,0)", "[0,5)", "[5,10)", "ALL"]:
    print(f"===== {b} =====")
    for method in ["addse_only", "pguse_only", "serial"]:
        subset = [x for x in rows if x["method"] == method and (b == "ALL" or x["bucket"] == b)]
        n = len(subset)
        print(method, "n=", n,
              "PESQ=", round(sum(x["pesq"] for x in subset) / n, 6),
              "ESTOI=", round(sum(x["estoi"] for x in subset) / n, 6),
              "SDR=", round(sum(x["sdr"] for x in subset) / n, 6))
