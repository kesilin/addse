import argparse
from pathlib import Path

import yaml


def parse_steps(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser("Create PGUSE phase-1 test configs for multiple step counts")
    parser.add_argument("--template", required=True, help="Template PGUSE config yaml")
    parser.add_argument("--ckpt", required=True, help="PGUSE checkpoint path")
    parser.add_argument("--test-src-dir", required=True, help="Input wav directory for PGUSE test")
    parser.add_argument("--test-tgt-dir", required=True, help="Clean wav directory for PGUSE test")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated configs")
    parser.add_argument("--steps", default="1,3,5,16,64", help="Comma-separated step list")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.template, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    base["ckpt_path"] = str(Path(args.ckpt).as_posix())
    base["dataset_config"]["test_src_dir"] = str(Path(args.test_src_dir).as_posix())
    base["dataset_config"]["test_tgt_dir"] = str(Path(args.test_tgt_dir).as_posix())

    for n in parse_steps(args.steps):
        cfg = dict(base)
        cfg["test_sde_config"] = dict(base["test_sde_config"])
        cfg["test_sde_config"]["N"] = int(n)
        out_path = out_dir / f"config_phase1_N{n}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
        print(f"generated: {out_path}")


if __name__ == "__main__":
    main()
