import itertools
import os
import random
import re
import sqlite3
from collections.abc import Mapping
from multiprocessing import JoinableQueue, Process, Queue
from typing import Annotated

import numpy as np
import polars as pl
import soundfile as sf
import torch
import typer
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from ..data import DynamicMixingDataset
from ..lightning import BaseLightningModule
from ..metrics import MP_METRICS, BaseMetric
from ..utils import load_hydra_config

app = typer.Typer()


@app.command()
def eval(
    config_file: Annotated[str, typer.Argument(help="Path to YAML config file.")],
    checkpoint: Annotated[
        str, typer.Argument(help="Path to model checkpoint with optional '{}' placeholder for the model name.")
    ],
    overrides: Annotated[list[str] | None, typer.Argument(help="Overrides for the config file.")] = None,
    device: Annotated[str, typer.Option(help="Device to run inference on ('cpu', 'cuda', or 'auto').")] = "auto",
    select: Annotated[str | None, typer.Option(help="Name of sweep configuration to evaluate.")] = None,
    regex: Annotated[str | None, typer.Option(help="Regex to filter sweep configurations.")] = None,
    noisy: Annotated[bool, typer.Option(help="Whether to compute metrics for the noisy input.")] = False,
    clean: Annotated[bool, typer.Option(help="Whether to compute metrics for the clean target.")] = False,
    output_dir: Annotated[str | None, typer.Option(help="Directory for audio outputs.")] = None,
    output_db: Annotated[str, typer.Option(help="Path to output SQLite database file.")] = "eval.db",
    num_consumers: Annotated[int, typer.Option(help="Number of consumer processes for calculating metrics.")] = 0,
    overwrite: Annotated[bool, typer.Option(help="Whether to overwrite existing results in the database.")] = False,
    num_examples: Annotated[int | None, typer.Option(help="Number of examples to evaluate per dataset.")] = None,
    clean_input: Annotated[bool, typer.Option(help="Whether to use clean signal as input instead of noisy.")] = False,
    return_nfe: Annotated[bool, typer.Option(help="Whether to return number of function evaluations.")] = False,
    no_lm: Annotated[bool, typer.Option(help="Whether to skip the model.")] = False,
    compute_loss: Annotated[bool, typer.Option(help="Whether to compute and store the loss.")] = False,
    no_metrics: Annotated[bool, typer.Option(help="Whether to not compute any metric")] = False,
    eval_min_duration: Annotated[float, typer.Option(help="Minimum duration in seconds for evaluation items.")] = 0.0,
    eval_min_active_duration: Annotated[float, typer.Option(help="Minimum active-speech duration in seconds.")] = 0.0,
    eval_active_threshold: Annotated[float, typer.Option(help="Energy threshold for active-speech masking.")] = 0.01,
    eval_seed: Annotated[int, typer.Option(help="Random seed for deterministic evaluation sampling.")] = 42,
) -> None:
    """Evaluate a model."""
    if not os.path.exists(config_file):
        raise FileNotFoundError("Config file does not exist.")
    if not os.path.isfile(config_file):
        raise ValueError("Input config must be a file.")
    if not config_file.endswith(".yaml"):
        raise ValueError("Config file must be a YAML file.")

    if num_consumers < 0:
        raise ValueError("Number of consumers must be non-negative.")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device not in ["cpu", "cuda"]:
        raise ValueError(f"Device must be 'auto', 'cpu', or 'cuda'. Got '{device}'.")
    print(f"Using device: {device}")

    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)

    load_dotenv()

    base_cfg, config_name = load_hydra_config(config_file, overrides=overrides)

    if isinstance(base_cfg.get("eval"), DictConfig) and isinstance(base_cfg.eval.get("dsets"), DictConfig):
        with open_dict(base_cfg):
            for _, dcfg in base_cfg.eval.dsets.items():
                if isinstance(dcfg, DictConfig):
                    dcfg.reset_rngs = True
                    dcfg.resume = False
                    dcfg.seed = int(eval_seed)
                    if isinstance(dcfg.get("speech_dataset"), DictConfig):
                        dcfg.speech_dataset.shuffle = False
                        dcfg.speech_dataset.seed = int(eval_seed + 1)
                    if isinstance(dcfg.get("noise_dataset"), DictConfig):
                        dcfg.noise_dataset.shuffle = False
                        dcfg.noise_dataset.seed = int(eval_seed + 2)

    config_name = base_cfg.get("name", config_name)

    cfgs: dict[str, DictConfig] = {}
    if (sweep_list := base_cfg.get("sweep")) is None:
        cfgs[config_name] = base_cfg
    else:
        if select is None and "{}" not in checkpoint:
            raise ValueError(
                "Sweep configuration detected. Please specify the name of the parameter set to evaluate with --select, "
                "or provide a checkpoint path with a '{}' placeholder for the model name."
            )
        product_sweeps = {}
        for sweep_items in itertools.product(*[s.items() for s in sweep_list]):
            product_name = "-".join(f"{s[0]}" for s in sweep_items)
            product_update = {k: v for d in sweep_items for k, v in d[1].items()}
            product_sweeps[product_name] = DictConfig(product_update)
        if len(product_sweeps.values()) != len(set(product_sweeps.values())):
            raise ValueError("Found duplicate sweep parameters sets after product.")
        for sweep_name, sweep_update in product_sweeps.items():
            merged_cfg = OmegaConf.merge(base_cfg, sweep_update)
            assert isinstance(merged_cfg, DictConfig)
            cfgs[f"{config_name}-{sweep_name}"] = merged_cfg
        print(f"Found {len(cfgs)} sweep configurations: {', '.join(cfgs.keys())}")

    if select is not None:
        if select not in cfgs:
            raise ValueError(f"Selected configuration '{select}' not found in sweep configurations.")
        cfgs = {select: cfgs[select]}

    if regex is not None:
        cfgs = {name: cfg for name, cfg in cfgs.items() if re.search(regex, name)}

    print(f"Evaluating {len(cfgs)} configurations: {', '.join(cfgs.keys())}")

    db = sqlite3.connect(output_db)
    db.execute("""
        CREATE TABLE IF NOT EXISTS results (
            dset TEXT NOT NULL,
            idx INT NOT NULL,
            metric TEXT NOT NULL,
            name TEXT NOT NULL,
            value REAL,
            PRIMARY KEY (dset, idx, metric, name)
        )
    """)

    metrics: Mapping[str, BaseMetric] = {} if no_metrics else instantiate(base_cfg.eval.metrics)
    metrics_mp = {name: metric for name, metric in metrics.items() if isinstance(metric, MP_METRICS)}
    metrics_nomp = {name: metric for name, metric in metrics.items() if not isinstance(metric, MP_METRICS)}

    dsets: Mapping[str, DynamicMixingDataset] = instantiate(base_cfg.eval.dsets)

    torch.set_grad_enabled(False)

    in_q = None  # type: ignore
    out_q = None  # type: ignore
    if num_consumers > 0 and metrics_mp:
        in_q = JoinableQueue(maxsize=10)
        out_q = Queue()
        for _ in range(num_consumers):
            Process(target=consumer, args=(in_q, out_q, metrics_mp), daemon=True).start()

    if not no_lm:
        for name, cfg in cfgs.items():
            ckpt = checkpoint.format(name) if "{}" in checkpoint else checkpoint
            if not os.path.exists(ckpt):
                print(f"Checkpoint '{ckpt}' does not exist. Skipping evaluation for {name}.")
                continue

            # When a full model checkpoint is explicitly provided, avoid redundant base preload.
            if isinstance(cfg.lm, DictConfig) and cfg.lm.get("pretrained_ckpt"):
                with open_dict(cfg):
                    cfg.lm.pretrained_ckpt = None

            lm: BaseLightningModule = instantiate(cfg.lm)
            lm.to(device)
            lm.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
            lm.eval()

            iterate_and_compute_metrics(
                dsets,
                device,
                metrics_mp,
                metrics_nomp,
                name,
                db,
                lm,
                output_dir,
                in_q,
                out_q,
                overwrite,
                num_examples,
                eval_min_duration,
                eval_min_active_duration,
                eval_active_threshold,
                clean_input,
                return_nfe,
                compute_loss,
            )

    if noisy:
        iterate_and_compute_metrics(
            dsets,
            device,
            metrics_mp,
            metrics_nomp,
            "noisy",
            db,
            None,
            output_dir,
            in_q,
            out_q,
            overwrite,
            num_examples,
            eval_min_duration,
            eval_min_active_duration,
            eval_active_threshold,
            clean_input,
            return_nfe,
            compute_loss,
        )

    if clean:
        iterate_and_compute_metrics(
            dsets,
            device,
            metrics_mp,
            metrics_nomp,
            "clean",
            db,
            None,
            output_dir,
            in_q,
            out_q,
            overwrite,
            num_examples,
            eval_min_duration,
            eval_min_active_duration,
            eval_active_threshold,
            clean_input,
            return_nfe,
            compute_loss,
        )

    names = list(cfgs.keys()) + (["noisy"] if noisy else []) + (["clean"] if clean else [])
    metrics_str = list(metrics.keys()) + (["nfe"] if return_nfe else []) + (["loss"] if compute_loss else [])
    query = f"""
        SELECT dset, metric, name, ROUND(AVG(value), 2) AS mean_value
        FROM results
        WHERE dset IN ({", ".join(f"'{dset}'" for dset in dsets)})
        AND name IN ({", ".join(f"'{name}'" for name in names)})
        AND metric IN ({", ".join(f"'{metric}'" for metric in metrics_str)})
        GROUP BY dset, metric, name
        ORDER BY dset, metric, name
    """  # noqa: S608
    df = pl.read_database(query, db)
    pl.Config.set_tbl_rows(100)
    print(df)

    db.close()


def iterate_and_compute_metrics(
    dsets: Mapping[str, DynamicMixingDataset],
    device: str,
    metrics_mp: Mapping[str, BaseMetric],
    metrics_nomp: Mapping[str, BaseMetric],
    name: str,
    db: sqlite3.Connection,
    lm: BaseLightningModule | None,
    output_dir: str | None,
    in_q: "JoinableQueue | None",
    out_q: "Queue | None",
    overwrite: bool,
    num_examples: int | None,
    eval_min_duration: float,
    eval_min_active_duration: float,
    eval_active_threshold: float,
    clean_input: bool,
    return_nfe: bool,
    compute_loss: bool,
) -> None:
    """Run inference and compute metrics for each item in each dataset and store the results."""
    assert (in_q is None and out_q is None) or (in_q is not None and out_q is not None)

    for dset_name, dset in dsets.items():
        target_examples = len(dset) if num_examples is None else int(num_examples)
        print(f"{name} | {dset_name}: target examples = {target_examples}")
        existing_results = db.execute(
            "SELECT COUNT(*) FROM results WHERE dset = ? AND name = ?", (dset_name, name)
        ).fetchone()[0]
        assert isinstance(existing_results, int)
        if existing_results > 0 and not overwrite:
            print(f"Skipping {dset_name} for {name} as results already exist in the database.")
            continue

        num_accepted = 0
        num_seen = 0
        for idx, (x, y, fs) in tqdm(enumerate(dset), desc=f"{name} | {dset_name}"):
            num_seen += 1
            if eval_min_duration > 0:
                duration = float(y.shape[-1]) / float(fs)
                if duration < eval_min_duration:
                    continue
            if eval_min_active_duration > 0:
                active = (y.abs() > eval_active_threshold).float().sum().item()
                active_duration = active / float(fs)
                if active_duration < eval_min_active_duration:
                    continue

            y_hat_torch: torch.Tensor | None = None
            nfe: int | None = None
            loss: dict[str, torch.Tensor] | None = None
            if lm is None:
                if name == "noisy":
                    y_hat_torch = x
                elif name == "clean":
                    y_hat_torch = y
                else:
                    raise ValueError(f"When no model is provided, name must be 'noisy' or 'clean'. Got '{name}'.")
            else:
                if metrics_nomp or metrics_mp or return_nfe or output_dir is not None:
                    lm_input = (y if clean_input else x).to(device).unsqueeze(0)
                    y_hat_torch, nfe = lm(lm_input, return_nfe=return_nfe) if return_nfe else (lm(lm_input), None)
                    y_hat_torch = y_hat_torch.squeeze(0)
                if compute_loss:
                    batch = (x.unsqueeze(0).to(device), y.unsqueeze(0).to(device), torch.tensor([fs]).to(device))
                    loss, _, _ = lm.step(batch, "test", 0, None)
            y_hat_np: np.ndarray | None = None if y_hat_torch is None else y_hat_torch.cpu().numpy()
            y_np: np.ndarray | None = None if y is None else y.cpu().numpy()

            if output_dir is not None and y_hat_np is not None:
                peak = max(np.abs(y).max(), np.abs(y_hat_np).max())
                os.makedirs(output_dir, exist_ok=True)
                y_hat_path = os.path.join(output_dir, f"{dset_name}_{idx:06d}_{name}.wav")
                sf.write(y_hat_path, y_hat_np.T / peak, fs)

            metric_values_nomp = {}
            if metrics_nomp:
                assert y_hat_np is not None
                assert y_np is not None
                metric_values_nomp = compute_metrics(y_hat_np, y_np, metrics_nomp)
            if return_nfe:
                assert nfe is not None
                metric_values_nomp["nfe"] = nfe
            if lm is not None and compute_loss:
                assert loss is not None
                metric_values_nomp["loss"] = loss["loss"].item()
            update_db(metric_values_nomp, dset_name, idx, name, db)

            if in_q is None:
                if metrics_mp:
                    assert y_hat_np is not None
                    assert y_np is not None
                    metric_values_mp = compute_metrics(y_hat_np, y_np, metrics_mp)
                    update_db(metric_values_mp, dset_name, idx, name, db)

            else:
                assert metrics_mp
                assert y_hat_np is not None
                assert y_np is not None
                in_q.put((idx, y_hat_np, y_np))

            num_accepted += 1
            if num_examples is not None and num_accepted >= num_examples:
                break

        filtered = max(0, num_seen - num_accepted)
        print(f"{name} | {dset_name}: accepted={num_accepted}, filtered={filtered}, min_duration={eval_min_duration}, min_active_duration={eval_min_active_duration}")

        if in_q is not None:
            in_q.join()

            assert out_q is not None
            while not out_q.empty():
                idx, metric_values_mp = out_q.get()
                update_db(metric_values_mp, dset_name, idx, name, db)

        db.commit()


def update_db(
    metric_values: dict[str, float],
    dset_name: str,
    idx: int,
    name: str,
    db: sqlite3.Connection,
) -> None:
    """Update the database with the computed metric values."""
    for metric_name, metric_value in metric_values.items():
        db.execute(
            "INSERT OR REPLACE INTO results (dset, idx, metric, name, value) VALUES (?, ?, ?, ?, ?)",
            (dset_name, idx, metric_name, name, metric_value),
        )


def consumer(in_q: JoinableQueue, out_q: Queue, metrics: Mapping[str, BaseMetric]) -> None:
    """Process items from the input queue and put results in the output queue."""
    while True:
        idx, y_hat, y = in_q.get()
        metric_values = compute_metrics(y_hat, y, metrics)
        out_q.put((idx, metric_values))
        in_q.task_done()


def compute_metrics(y_hat: np.ndarray, y: np.ndarray, metrics: Mapping[str, BaseMetric]) -> dict[str, float]:
    """Compute metrics for the given prediction and target."""
    metric_values = {}
    for metric_name, metric in metrics.items():
        try:
            metric_value = metric(y_hat, y)
        except Exception as e:
            print(f"Error computing {metric_name}: {e}")
            metric_value = float("nan")
        metric_values[metric_name] = metric_value
    return metric_values
