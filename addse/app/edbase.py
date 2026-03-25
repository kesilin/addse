import importlib.util
import logging
import os
from typing import Annotated, Any

import typer

from .ldopt import ldopt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = typer.Typer()


def _load_edbase_config(edbase_config_py: str) -> dict[str, Any]:
    if not os.path.isfile(edbase_config_py):
        raise FileNotFoundError(f"ED_BASE config not found: {edbase_config_py}")

    spec = importlib.util.spec_from_file_location("edbase_config", edbase_config_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import ED_BASE config: {edbase_config_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "load_config"):
        raise AttributeError(f"ED_BASE config module does not expose load_config(): {edbase_config_py}")

    cfg = module.load_config()
    if not isinstance(cfg, dict):
        raise TypeError("ED_BASE load_config() must return a dict")
    return cfg


@app.command("prepare-edbase-data")
def prepare_edbase_data(
    edbase_config_py: Annotated[
        str,
        typer.Option(help="Path to ED_BASE config.py (or core/config.py)."),
    ] = "../ED_BASE/config.py",
    speech_output_dir: Annotated[
        str,
        typer.Option(help="LitData output dir for speech chunks."),
    ] = "data/chunks/edbase_speech/",
    noise_output_dir: Annotated[
        str,
        typer.Option(help="LitData output dir for noise chunks."),
    ] = "data/chunks/edbase_noise/",
    speech_regex: Annotated[
        str,
        typer.Option(help=r"Regex for speech files."),
    ] = r"^.*\.(wav|flac)$",
    noise_regex: Annotated[
        str,
        typer.Option(help=r"Regex for noise files."),
    ] = r"^.*\.(wav|flac|mp3)$",
    noise_seglen: Annotated[
        float,
        typer.Option(help="Optional segmentation length for noise files (seconds). 0 means no segmentation."),
    ] = 10.0,
    num_workers: Annotated[int, typer.Option(help="Number of workers for LitData optimize.")] = 4,
    seed: Annotated[int | None, typer.Option(help="Shuffle seed. If omitted, no shuffle seed is set.")] = 42,
) -> None:
    """Prepare ADDSE litdata chunks from ED_BASE data paths."""
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    cfg = _load_edbase_config(edbase_config_py)
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}

    clean_root = data_cfg.get("clean_root")
    noise_root = data_cfg.get("noise_root")
    sample_rate = data_cfg.get("sample_rate")
    segment_seconds = data_cfg.get("segment_seconds")

    if not isinstance(clean_root, str) or not clean_root:
        raise ValueError("ED_BASE config missing valid data.clean_root")
    if not isinstance(noise_root, str) or not noise_root:
        raise ValueError("ED_BASE config missing valid data.noise_root")

    logger.info(
        "Loaded ED_BASE config: clean_root=%s noise_root=%s sample_rate=%s segment_seconds=%s",
        clean_root,
        noise_root,
        sample_rate,
        segment_seconds,
    )

    ldopt(
        input_dirs=[clean_root],
        output_dir=speech_output_dir,
        regexes=[speech_regex],
        num_workers=num_workers,
        seglens=[0.0],
        labels=["edbase_clean"],
        seed=seed,
    )

    ldopt(
        input_dirs=[noise_root],
        output_dir=noise_output_dir,
        regexes=[noise_regex],
        num_workers=num_workers,
        seglens=[noise_seglen],
        labels=["edbase_noise"],
        seed=seed,
    )

    logger.info("Prepared chunks for ADDSE:")
    logger.info("  speech: %s", speech_output_dir)
    logger.info("  noise : %s", noise_output_dir)
