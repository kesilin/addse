import sys
import os
import traceback

# ensure package imports resolve
sys.path.insert(0, os.path.abspath('.'))

from addse.utils import load_hydra_config, seed_all
from hydra.utils import instantiate
import torch
from omegaconf import OmegaConf

cfg_path = 'configs/addse-s-edbase-parallel60-a008-p02-spec.yaml'


def run_with_cfg(overrides):
    cfg, name = load_hydra_config(cfg_path, overrides=overrides)
    # If no explicit test dataset is configured, use eval.dsets.edbase-local
    try:
        from omegaconf import open_dict, OmegaConf
        if getattr(cfg, "eval", None) is not None and getattr(cfg.eval, "dsets", None) is not None:
            ds = cfg.eval.dsets.get("edbase-local")
            if ds is not None and getattr(cfg.dm, "test_dataset", None) is None:
                # ensure we pass a factory (partial) rather than an already-instantiated object
                ds_copy = OmegaConf.create(OmegaConf.to_container(ds, resolve=True))
                ds_copy.setdefault("_partial_", True)
                with open_dict(cfg):
                    cfg.dm.test_dataset = ds_copy
                    # reuse val_dataloader settings for test if not provided
                    if cfg.get("val_dataloader") is not None:
                        cfg.dm.test_dataloader = cfg.get("val_dataloader")
    except Exception:
        pass
    seed_all(cfg.seed)
    lm = instantiate(cfg.lm)
    # Force a small eval/test dataset based on cfg.eval.dsets.edbase-local
    try:
        if getattr(cfg, "eval", None) is not None and getattr(cfg.eval, "dsets", None) is not None:
            ed = OmegaConf.to_container(cfg.eval.dsets.get("edbase-local"), resolve=True)
            # ensure we create a short dataset for quick test (increased per user request)
            ed["length"] = 100
            ed.setdefault("_partial_", True)
            # place into dm.test_dataset
            from omegaconf import open_dict
            with open_dict(cfg):
                cfg.dm.test_dataset = ed
                # reusing val_dataloader settings for test if not provided
                if cfg.get("val_dataloader") is not None:
                    cfg.dm.test_dataloader = cfg.get("val_dataloader")
    except Exception:
        pass

    dm = instantiate(cfg.dm)
    trainer = instantiate(cfg.trainer)
    # ensure model is in eval mode for sampling/inference
    try:
        lm.eval()
    except Exception:
        pass
    # If instantiated DataModule's test_dataloader is empty, inject a small local dataset
    try:
        test_loader = dm.test_dataloader()
    except Exception:
        test_loader = []
    print(f"Instantiated LM: {type(lm).__name__}")
    print(f"Instantiated Trainer: {type(trainer).__name__}")
    try:
        # Instead of trainer.test, manually step through first 2 batches to collect alpha and metrics
        dm.setup(stage="test")
        loader = dm.test_dataloader()
        metrics_acc = []
        alpha_means = []
        running_sdr_initial = float(lm.running_sdr.detach().cpu().item()) if hasattr(lm, "running_sdr") else None
        # prepare metrics instances
        metrics_cfg = cfg.get('eval', {}).get('metrics', {})
        metrics_inst = {k: instantiate(v) for k, v in metrics_cfg.items()} if metrics_cfg else None

        # attempt to get iterator from loader; if empty, build concrete small DynamicMixingDataset
        try:
            it = iter(loader)
            try:
                first = next(it)
                has_first = True
            except StopIteration:
                has_first = False
        except Exception:
            has_first = False

        if not has_first:
            print('dm.test_dataloader empty -> building concrete DynamicMixingDataset from chunks')
            from addse.data import AudioStreamingDataset, DynamicMixingDataset, AudioStreamingDataLoader
            speech_path = os.path.join(os.path.dirname(__file__), 'data', 'chunks', 'edbase_speech')
            noise_path = os.path.join(os.path.dirname(__file__), 'data', 'chunks', 'edbase_noise')
            speech_ds = AudioStreamingDataset(input_dir=speech_path, fs=cfg.fs, segment_length=1.0, max_dynamic_range=cfg.max_dynamic_range, shuffle=True, seed=cfg.seed)
            noise_ds = AudioStreamingDataset(input_dir=noise_path, fs=cfg.fs, segment_length=1.0, max_dynamic_range=cfg.max_dynamic_range, shuffle=True, seed=cfg.seed)
            mix_ds = DynamicMixingDataset(speech_ds, noise_ds, snr_range=tuple(cfg.eval.dsets['edbase-local'].get('snr_range', [0.0, 10.0])), rms_range=tuple(cfg.eval.dsets['edbase-local'].get('rms_range', [0.0, 0.0])), length=100, resume=False, reset_rngs=True)
            loader = AudioStreamingDataLoader(mix_ds, batch_size=4, num_workers=0)
            it = iter(loader)

        # process up to first 2 batches (including the pre-fetched `first` if present)
        i = 0
        if 'first' in locals() and has_first:
            batch = first
            loss_dict, metric_vals, debug = lm.step(batch, stage="test", batch_idx=i, metrics=metrics_inst)
            metrics_acc.append(metric_vals)
            try:
                alpha_batch = lm.model._last_alpha_total_batch
                alpha_means.append(float(alpha_batch.mean().item()))
            except Exception:
                alpha_means.append(None)
            i += 1

        for batch in it:
            if i >= 2:
                break
            loss_dict, metric_vals, debug = lm.step(batch, stage="test", batch_idx=i, metrics=metrics_inst)
            metrics_acc.append(metric_vals)
            try:
                alpha_batch = lm.model._last_alpha_total_batch
                alpha_means.append(float(alpha_batch.mean().item()))
            except Exception:
                alpha_means.append(None)
            i += 1
        # print aggregated results
        if metrics_acc:
            # take last metrics as test result proxy
            print('TEST_METRICS_FIRST_BATCHES:', metrics_acc)
        print('running_sdr_initial:', running_sdr_initial)
        print('alpha_means_first_2_batches:', alpha_means)
        return metrics_acc
    except RuntimeError as e:
        msg = str(e).lower()
        if 'out of memory' in msg or 'oom' in msg:
            print("OOM detected during test run.")
            raise
        else:
            traceback.print_exc()
            raise


if __name__ == '__main__':
    # first attempt: strict config, no override except stage=test
    try:
        print('Running test with strict config (stage=test) with samples=100, batch_size=4, max_epochs=5')
        # run with requested overrides: sample count 100, batch size 4, epochs 5
        run_with_cfg(['+stage=test', 'dm.train_dataloader.batch_size=4', 'dm.val_dataloader.batch_size=4', 'trainer.max_epochs=5'])
    except RuntimeError:
        # On OOM, retry with batch_size=1
        print('Retrying with dm.batch_size=1')
        try:
            run_with_cfg(['+stage=test', '+dm.batch_size=1'])
        except Exception:
            print('Retry failed:')
            traceback.print_exc()
            sys.exit(2)

    except Exception:
        print('Fatal error:')
        traceback.print_exc()
        sys.exit(1)

    print('Done')
