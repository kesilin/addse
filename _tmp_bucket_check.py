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
        rows.append(row)

for b in ["[-5,0)", "[0,5)", "[5,10)"]:
    print(f"BUCKET {b}")
    for method in ["addse_only", "pguse_only", "serial"]:
        vals = [float(x["pesq"]) for x in rows if x["bucket"] == b and x["method"] == method]
        print(method, round(sum(vals) / len(vals), 6), f"n={len(vals)}")
