import functools
import math
import re
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, override

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset

from .data import AudioStreamingDataLoader, DynamicMixingDataset
from .losses import BaseLoss
from .metrics import BaseMetric
from .models import ADM, NAC, ADDSERQDiT, SGMSEUNet
from .stft import STFT

@dataclass
class LogConfig:
    on_train_step: bool = False
    on_train_epoch: bool = True
    on_val_step: bool = False
    on_val_epoch: bool = True
    on_test_step: bool = False
    on_test_epoch: bool = True

class BaseLightningModule(L.LightningModule):
    val_metrics: Mapping[str, BaseMetric] | None
    test_metrics: Mapping[str, BaseMetric] | None
    log_cfg: LogConfig
    debug_sample: tuple[int, int] | None

    @abstractmethod
    def step(self, batch, stage, batch_idx, metrics=None):
        pass

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> dict[str, Tensor]:
        loss, metrics, _ = self.step(batch, "train", batch_idx)
        self.log_metrics(loss, metrics, "train", self.log_cfg.on_train_step, self.log_cfg.on_train_epoch)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> dict[str, Tensor]:
        loss, metrics, debug_samples = self.step(batch, "val", batch_idx, self.val_metrics)
        self.log_debug_samples(batch, batch_idx, debug_samples)
        self.log_metrics(loss, metrics, "val", self.log_cfg.on_val_step, self.log_cfg.on_val_epoch)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> dict[str, Tensor]:
        loss, metrics, _ = self.step(batch, "test", batch_idx, self.test_metrics)
        self.log_metrics(loss, metrics, "test", self.log_cfg.on_test_step, self.log_cfg.on_test_epoch)
        return loss

    def log_metrics(self, loss, metrics, stage, on_step, on_epoch):
        for key, value in {**loss, **metrics}.items():
            self.log(f"{stage}_{key}", value, on_step=on_step, on_epoch=on_epoch)

    def log_debug_samples(self, batch, batch_idx, debug_samples):
        wandb_logger = next((logger for logger in self.loggers if isinstance(logger, WandbLogger)), None)
        if wandb_logger is None or self.debug_sample is None or batch_idx != self.debug_sample[0]:
            return
        for name, x in debug_samples.items():
            x_cpu = x[self.debug_sample[1], 0, :].cpu().float().numpy()
            fs = batch[2][self.debug_sample[1]].item()
            wandb_logger.log_audio(key=name, audios=[x_cpu / max(abs(x_cpu))], step=self.global_step, sample_rate=[fs])

class ConfigureOptimizersMixin(L.LightningModule):
    optimizer: Callable[[Iterator[nn.Parameter]], Optimizer]
    lr_scheduler: Mapping[str, Any] | None

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer(self.parameters())
        output: dict[str, Any] = {"optimizer": optimizer}
        if self.lr_scheduler is not None:
            output["lr_scheduler"] = {k: v(optimizer) if k == "scheduler" else v for k, v in self.lr_scheduler.items()}
        return output

# ======================================================================
# ADDSELightningModule (QRC 正交解耦版)
# ======================================================================
class ADDSELightningModule(BaseLightningModule, ConfigureOptimizersMixin):
    def __init__(self, nac_cfg: str, nac_ckpt: str, model: ADDSERQDiT, num_steps: int, block_size: int, **kwargs) -> None:
        super().__init__()
        self.nac, self.mask_token = load_nac(nac_cfg, nac_ckpt)
        self.model, self.num_steps, self.block_size = model, num_steps, block_size
        self.optimizer, self.lr_scheduler = kwargs.get("optimizer"), kwargs.get("lr_scheduler")
        
        self.val_metrics = kwargs.get("val_metrics")
        self.test_metrics = kwargs.get("test_metrics")
        log_cfg = kwargs.get("log_cfg")
        self.log_cfg = LogConfig() if log_cfg is None else log_cfg
        self.debug_sample = kwargs.get("debug_sample")
        
        # 加载 1.7 分主干权重
        pretrained_ckpt = "logs/addse-edbase-quick/checkpoints/addse-s.ckpt" 
        import os
        if os.path.exists(pretrained_ckpt):
            print(f"--- 正在加载 1.7 分主干权重: {pretrained_ckpt} ---")
            ckpt_data = torch.load(pretrained_ckpt, map_location="cpu")
            state_dict = ckpt_data.get("state_dict", ckpt_data)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"--- 权重加载完毕！连续并联残差网络已随机初始化，等待黄金特征监督 ---")
        else:
            print(f"⚠️ 未找到预训练权重 {pretrained_ckpt}，模型将从零开始！")

    def loss(self, x_q, y_q, y_tok, return_intermediates=False, x_cont=None):
        B, K, L = y_tok.shape
        lambd = torch.rand(B, device=y_tok.device)
        mask = torch.rand(y_tok.shape, device=y_tok.device) < lambd[:, None, None]
        y_lambda_q = y_q.clone()
        y_lambda_q.masked_fill_(mask[:, None], 0)
        
        log_score = self.log_score(y_lambda_q, x_q, x_cont=x_cont)
        V = log_score.shape[-1]
        loss = F.cross_entropy(log_score.reshape(-1, V), y_tok.reshape(-1), reduction="none")
        loss = (loss.reshape(B, K, L) * mask).sum() / (mask.sum() + 1e-8)
        return (loss, log_score, mask) if return_intermediates else loss

    def log_score(self, y_q, x_q, x_cont=None):
        x_c_in = x_cont if x_cont is not None else torch.zeros_like(x_q)
        def model_wrapper(y_s, x_s, c_s):
            return self.model(y_s, x_s, None, c_cont=c_s)
        score = process_in_blocks((y_q, x_q, x_c_in), self.block_size, model_wrapper)
        return score.moveaxis(1, -1).log_softmax(dim=-1)

    @override
    def step(self, batch, stage, batch_idx, metrics=None):
        x, y, _ = batch
        
        # 获取带噪和干净的原始连续隐特征 (x_lat & y_lat)
        x_lat = self.nac.encoder(x)
        with torch.no_grad():
            y_lat = self.nac.encoder(y) # 黄金标尺：真正完美的高频连续特征！

        xy_tok, xy_q = self.nac.encode(torch.cat([x, y]), no_sum=True, domain="q")
        x_tok, y_tok = xy_tok.chunk(2); x_q, y_q = xy_q.chunk(2)

        # =========================================================
        # 支路一 (离散主干): 专注猜词，计算分类损失 (CE Loss)
        # =========================================================
        ce_loss, log_score, mask = self.loss(x_q, y_q, y_tok, return_intermediates=True, x_cont=x_lat)
        total_loss = ce_loss
        self.log(f"{stage}/ce_loss", ce_loss, prog_bar=True, sync_dist=True)

        # =========================================================
        # 支路二 (连续并联): 专注弥补高频，计算残差损失 (Res L1 Loss)
        # =========================================================
        if hasattr(self.model, "predict_residual"):
            residual_pred = self.model.predict_residual(x_lat) # (B, C, L)
            
            # 计算被量化器无情抛弃的 "黄金残差" = 真实连续 - 真实量化
            K = y_tok.shape[1]
            y_q_sum = y_q.sum(dim=2) if y_q.ndim == 4 and y_q.shape[2] == K else y_q.sum(dim=1)
            true_residual = (y_lat - y_q_sum).detach()
            
            # 强迫并联支路学会提取高频残差！
            res_loss = F.l1_loss(residual_pred, true_residual)
            total_loss = total_loss + 10.0 * res_loss # 权重可以给大点
            self.log(f"{stage}/res_loss", res_loss, prog_bar=True, sync_dist=True)

        self.log(f"{stage}/loss", total_loss, prog_bar=False, sync_dist=True)

        if metrics or (stage == "val" and batch_idx == 0):
            y_lat_fused = self.solve(x_tok, x_q, self.num_steps, x_cont=x_lat)
            
            # 直接将最终完美的连续特征送给生成器
            if hasattr(self.nac, "generator"):
                y_hat = self.nac.generator(y_lat_fused)
            else:
                y_hat = self.nac.decode(y_lat_fused, domain="q")
                
            y_hat = y_hat[..., :y.shape[-1]]
            return {"loss": total_loss}, compute_metrics(y_hat, y, metrics), {"output": y_hat, "input": x, "clean": y}
            
        return {"loss": total_loss}, {}, {}

    @torch.no_grad()
    def solve(self, x_tok, x_q, num_steps, x_cont=None):
        B, K, L = x_tok.shape
        
        # =========================================================
        # 1. 连续支路：一键预测出丢失的高频相位残差！
        # =========================================================
        if hasattr(self.model, "predict_residual") and x_cont is not None:
            residual_pred = self.model.predict_residual(x_cont)
        else:
            residual_pred = 0

        # =========================================================
        # 2. 离散主干：顽强地从噪声中还原出鲁棒的语义 Token
        # =========================================================
        y_tok = torch.full_like(x_tok, self.mask_token)
        import math
        for i in range(num_steps):
            mask = y_tok == self.mask_token
            if not mask.any(): break
            
            y_q_step = self.nac.quantizer.decode(y_tok.masked_fill(mask, 0), output_no_sum=True, domain="code")
            if y_q_step.shape != x_q.shape:
                if y_q_step.ndim == 4:
                    if y_q_step.shape[2] == x_q.shape[1]: y_q_step = y_q_step.transpose(1, 2)
                    elif y_q_step.shape[1] != x_q.shape[1]: y_q_step = y_q_step.reshape(x_q.shape)
                    
            y_q_step = y_q_step.masked_fill(mask.unsqueeze(1), 0)
            log_p = self.log_score(y_q_step, x_q, x_cont=x_cont)
            probs = log_p.exp()
            sampled_tokens = torch.multinomial(probs[mask], 1).squeeze(-1)
            y_tok_new = y_tok.clone()
            y_tok_new[mask] = sampled_tokens
            
            if i == num_steps - 1:
                y_tok = y_tok_new
                break
                
            confidence = probs[mask].gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
            conf_full = torch.full_like(y_tok, 1e9, dtype=torch.float32)
            conf_full[mask] = confidence
            
            ratio = math.cos(math.pi / 2 * (i + 1) / num_steps)
            num_mask = int(ratio * K * L)
            if num_mask > 0:
                conf_flat = conf_full.view(B, -1)
                cutoff_vals, _ = torch.topk(conf_flat, num_mask, dim=-1, largest=False)
                cutoff = cutoff_vals[:, -1:]
                new_mask = (conf_flat <= cutoff).view(B, K, L)
                y_tok_new[new_mask] = self.mask_token
            y_tok = y_tok_new

        # =========================================================
        # 3. 顶会级正交融合 (Orthogonal Parallel Fusion)
        # =========================================================
        # 把最终猜出的离散 Token 解码成连续的基底特征
        y_q_discrete = self.nac.quantizer.decode(y_tok, output_no_sum=False, domain="code")
        if y_q_discrete.ndim == 4:
            y_q_discrete = y_q_discrete.sum(dim=1) if y_q_discrete.shape[1] == K else y_q_discrete.sum(dim=2)

        # 【核心魔法】：离散骨架 + 高频残差 = 突破天花板的完美特征
        y_lat_fused = y_q_discrete + residual_pred
        return y_lat_fused

    def forward(self, x: torch.Tensor, return_nfe: bool = False, **kwargs):
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional"
        n_pad = (self.nac.downsampling_factor - x.shape[-1]) % self.nac.downsampling_factor
        x_pad = F.pad(x, (0, n_pad))
        
        x_lat = self.nac.encoder(x_pad)
        x_tok, x_q = self.nac.encode(x_pad, no_sum=True, domain="q")
        
        # solve 现在返回的是完美的结合特征 (B, C, L)
        y_lat_fused = self.solve(x_tok, x_q, self.num_steps, x_cont=x_lat)
        
        # 直接跨越量化器，把特征喂给最终的声码器/生成器！
        if hasattr(self.nac, "generator"):
            y_hat_pad = self.nac.generator(y_lat_fused)
        else:
            y_hat_pad = self.nac.decode(y_lat_fused, domain="q")
            
        y_hat = y_hat_pad[..., : y_hat_pad.shape[-1] - n_pad]
        if return_nfe: return y_hat, self.num_steps
        return y_hat

# ======================================================================
# 其他辅助类（EDMMixin, DataModule 等）原样保留
# ======================================================================
class EDMMixin(L.LightningModule):
    model: nn.Module
    num_steps: int
    norm_factor: float
    sigma_data: float
    p_mean: float
    p_sigma: float
    s_churn: float
    s_min: float
    s_max: float
    s_noise: float
    sigma_min: float
    sigma_max: float
    rho: float

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        log_sigma = self.p_mean + self.p_sigma * torch.randn(y.shape[0], dtype=y.real.dtype, device=y.device)
        sigma = log_sigma.exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        noise = sigma.view(-1, *(1,) * (y.ndim - 1)) * torch.randn_like(y)
        loss = weight.view(-1, *(1,) * (y.ndim - 1)) * (self.denoiser(y + noise, x, sigma) - y).abs().pow(2)
        return loss.mean()

    def denoiser(self, y: Tensor, x: Tensor, sigma: Tensor) -> Tensor:
        sigma_broad = sigma.view(-1, *(1,) * (y.ndim - 1))
        c_skip = self.sigma_data**2 / (sigma_broad**2 + self.sigma_data**2)
        c_out = self.sigma_data * sigma_broad / (self.sigma_data**2 + sigma_broad**2).sqrt()
        c_in_y = 1 / (self.sigma_data**2 + sigma_broad**2).sqrt()
        c_in_x = 1 / self.sigma_data
        c_noise = 0.25 * sigma.log()
        return c_skip * y + c_out * self.model(c_in_y * y, c_in_x * x, c_noise)

    @torch.no_grad()
    def solve(self, x: Tensor, num_steps: int) -> Tensor:
        t = torch.tensor([self.sampling_step(i) if i < num_steps else 0.0 for i in range(num_steps + 1)], device=x.device, dtype=x.real.dtype)
        y = t[0] * torch.randn_like(x)
        for i in range(num_steps):
            if self.s_churn > 0 and self.s_min <= t[i] <= self.s_max:
                gamma = min(self.s_churn / num_steps, math.sqrt(2) - 1)
                t_hat = t[i] * (1 + gamma)
                y_hat = y + (t_hat**2 - t[i] ** 2).sqrt() * torch.randn_like(y) * self.s_noise
            else:
                t_hat = t[i]; y_hat = y
            d = (y_hat - self.denoiser(y_hat, x, t_hat[None])) / t_hat
            y = y_hat + (t[i + 1] - t_hat) * d
            if i < num_steps - 1:
                d_next = (y - self.denoiser(y, x, t[i + 1, None])) / t[i + 1]
                y = y_hat + 0.5 * (t[i + 1] - t_hat) * (d + d_next)
        return y

    def sampling_step(self, i: int) -> float:
        return (self.sigma_max ** (1 / self.rho) + i / (self.num_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho

class DataModule(L.LightningDataModule):
    def __init__(self, train_dataset, train_dataloader, val_dataset=None, val_dataloader=None, test_dataset=None, test_dataloader=None):
        super().__init__()
        self.train_dataset_fn = train_dataset
        self.val_dataset_fn = val_dataset
        self.test_dataset_fn = test_dataset
        self.train_dataloader_fn = train_dataloader
        self.val_dataloader_fn = val_dataloader
        self.test_dataloader_fn = test_dataloader
        self.train_dset = None; self.val_dset = None; self.test_dset = None; self._state_dict = None

    def setup(self, stage: str) -> None:
        self.train_dset = self.train_dataset_fn()
        self.val_dset = None if self.val_dataset_fn is None else self.val_dataset_fn()
        self.test_dset = None if self.test_dataset_fn is None else self.test_dataset_fn()

    def train_dataloader(self) -> DataLoader:
        train_dataloader = self.train_dataloader_fn(self.train_dset)
        if self._state_dict is not None and isinstance(train_dataloader, AudioStreamingDataLoader):
            train_dataloader.load_state_dict(self._state_dict)
            self._state_dict = None
        return train_dataloader

    def val_dataloader(self) -> DataLoader | list:
        return [] if self.val_dataloader_fn is None or self.val_dset is None else self.val_dataloader_fn(self.val_dset)

    def test_dataloader(self) -> DataLoader | list:
        return [] if self.test_dataloader_fn is None or self.test_dset is None else self.test_dataloader_fn(self.test_dset)

    def state_dict(self) -> dict:
        return self.trainer.train_dataloader.state_dict() if self.trainer is not None and isinstance(self.trainer.train_dataloader, AudioStreamingDataLoader) else {}

    def load_state_dict(self, state_dict: dict) -> None:
        self._state_dict = state_dict

def compute_metrics(x: Tensor, y: Tensor, metrics: Mapping[str, BaseMetric] | None = None) -> dict[str, float]:
    return {k: sum(v(x_i, y_i) for x_i, y_i in zip(x, y)) / y.shape[0] for k, v in (metrics or {}).items()}

def load_nac(cfg_path: str, ckpt_path: str) -> tuple[NAC, int]:
    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    nac: NAC = instantiate(cfg["lm"]["generator"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {k.removeprefix("generator."): v for k, v in ckpt["state_dict"].items() if k.startswith("generator.")}
    nac.load_state_dict(state_dict)
    nac.eval()
    for param in nac.parameters(): param.requires_grad = False
    return nac, nac.quantizer.codebooks[0].codebook.weight.shape[0]

def process_in_blocks(args: tuple[Tensor, ...], block_size: int, fn: Callable[..., Tensor]) -> Tensor:
    blocks = [fn(*(arg[..., i : i + block_size] for arg in args)) for i in range(0, args[0].shape[-1], block_size)]
    return torch.cat(blocks, dim=-1)