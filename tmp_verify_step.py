import sys
import os
import math
sys.path.insert(0, os.path.abspath('.'))
import torch
from addse.lightning import ADDSELightningModule
from hydra.utils import instantiate
from omegaconf import OmegaConf

# 1. Simulate a batch: 2 examples, 1 channel, 16k samples (~1s)
B, C, L = 2, 1, 16000
clean = torch.randn(B, C, L)
noisy = clean + 5.0 * torch.randn(B, C, L)  # very low SNR
batch = (noisy, clean, torch.tensor([16000, 16000]))

# 2. Instantiate model from config
cfg = OmegaConf.load('configs/addse-s-edbase-parallel60-a008-p02-spec.yaml')
# ensure NAC checkpoint path is present
cfg.lm.nac_ckpt = 'logs/nac/checkpoints/last.ckpt'

lm = instantiate(cfg.lm)

# 3. Provide a simple SDR metric to drive running_sdr update
def sdr_metric(x: torch.Tensor, y: torch.Tensor) -> float:
    # compute simple SDR in dB per-sample
    num = float((y ** 2).sum().item())
    den = float(((y - x) ** 2).sum().item()) + 1e-8
    return 10.0 * math.log10(num / den)

lm.test_metrics = {'sdr': sdr_metric}

# 4. Run one test step
lm.eval()
with torch.no_grad():
    loss, metric_vals, debug = lm.step(batch, stage='test', batch_idx=0, metrics=lm.test_metrics)

# 5. Report
print('--- Logic Verification ---')
print(f"Current EMA SDR: {lm.running_sdr.item():.2f} dB")
spec_guard_active = float(lm.running_sdr.item()) < float(lm.sdr_threshold)
print(f"Spec Guard Active: {spec_guard_active}")
# debug['output'] may contain model output; print a simple proxy
if isinstance(debug, dict) and 'output' in debug and isinstance(debug['output'], torch.Tensor):
    print(f"Output peak (proxy): {debug['output'].abs().max().item():.4f}")
else:
    print('No debug output available')

print('Metrics returned:', metric_vals)
print('Loss keys:', list(loss.keys()))
