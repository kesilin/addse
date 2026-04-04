import functools
import math
import os
import re
import warnings
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, override

import lightning as L
import numpy as np
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
from .models import ADM, NAC, ADDSERQDiT, ConvTasNet, SGMSEUNet
from .stft import STFT

try:
    from pesq import BufferTooShortError, NoUtterancesError, pesq
except Exception:  # pragma: no cover
    BufferTooShortError = NoUtterancesError = Exception
    pesq = None


class SchemeDLearnableGatingRouter(nn.Module):
    """
    Scheme D: Learnable Gating for Soft Fusion
    
    Learns frame-level and codebook-level gating to decide between:
    - DiT (base_logits): semantic + pre-trained priors
    - STFT (acoustic_logits): acoustic ground truth
    
    gate ∈ [0, 1]: 0 = trust DiT, 1 = trust STFT
    """
    def __init__(self, num_codebooks: int = 8, feature_dim: int = 256):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.feature_dim = feature_dim
        
        # Simple router: maps acoustic features to per-codebook gates
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
    
    def forward(self, acoustic_features: torch.Tensor) -> torch.Tensor:
        """
        acoustic_features: [B, K, L, F] or [B, L, F]
        returns: gate [B, K, L, 1] ∈ [0, 1]
        """
        # Reshape to [B*K*L, F]
        shape = acoustic_features.shape
        feat_flat = acoustic_features.reshape(-1, shape[-1])
        
        # Compute gate logits
        gate_logits = self.gate_net(feat_flat)  # [B*K*L, 1]
        
        # Reshape back and apply sigmoid
        if len(shape) == 4:
            B, K, L, _ = shape
            gate_logits = gate_logits.reshape(B, K, L, -1)
            gate = torch.sigmoid(gate_logits[:, :, :, 0:1])  # [B, K, L, 1]
        else:
            # Fallback for [B, L, F]
            B, L, _ = shape
            gate_logits = gate_logits.reshape(B, L, -1)
            gate = torch.sigmoid(gate_logits[:, :, 0:1])  # [B, L, 1]
        
        return gate


class MetricDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 2, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden * 2, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, y_hat: Tensor, y_ref: Tensor) -> Tensor:
        if y_hat.ndim == 2:
            y_hat = y_hat.unsqueeze(1)
        if y_ref.ndim == 2:
            y_ref = y_ref.unsqueeze(1)
        h = torch.cat([y_hat, y_ref], dim=1)
        h = self.net(h).squeeze(-1)
        return torch.sigmoid(self.head(h))

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

# 支持返回 Tuple 的 process_in_blocks
def process_in_blocks(args: tuple[Tensor, ...], block_size: int, fn: Callable[..., Any]) -> Any:
    blocks = [fn(*(arg[..., i : i + block_size] for arg in args)) for i in range(0, args[0].shape[-1], block_size)]
    if isinstance(blocks[0], tuple):
        num_returns = len(blocks[0])
        result = []
        for j in range(num_returns):
            if isinstance(blocks[0][j], Tensor):
                result.append(torch.cat([b[j] for b in blocks], dim=-1))
            else:
                result.append(blocks[0][j])
        return tuple(result)
    return torch.cat(blocks, dim=-1)

# ======================================================================
# ADDSELightningModule (QRC 残差补偿彻底解耦版)
# ======================================================================
class ADDSELightningModule(BaseLightningModule, ConfigureOptimizersMixin):
    def __init__(self, nac_cfg: str, nac_ckpt: str, model: ADDSERQDiT, num_steps: int, block_size: int, **kwargs) -> None:
        super().__init__()
        self.nac, self.mask_token = load_nac(nac_cfg, nac_ckpt)
        self.model, self.num_steps, self.block_size = model, num_steps, block_size
        self.optimizer, self.lr_scheduler = kwargs.get("optimizer"), kwargs.get("lr_scheduler")
        self.spec_loss: BaseLoss | None = kwargs.get("spec_loss")
        self.spec_loss_weight = float(kwargs.get("spec_loss_weight", 0.0))
        self.si_sdr_weight = float(kwargs.get("si_sdr_weight", 0.0))
        self.residual_l1_weight = float(kwargs.get("residual_l1_weight", 2.0))
        self.direct_residual_weight = float(kwargs.get("direct_residual_weight", 0.0))
        self.wave_l1_weight = float(kwargs.get("wave_l1_weight", 1.0))
        self.residual_cos_weight = float(kwargs.get("residual_cos_weight", 0.0))
        self.residual_std_weight = float(kwargs.get("residual_std_weight", 0.0))
        self.residual_ema_decay = float(kwargs.get("residual_ema_decay", 0.8))
        self.quality_branch_weight = float(kwargs.get("quality_branch_weight", 0.0))
        self.quality_smooth_weight = float(kwargs.get("quality_smooth_weight", 0.0))
        self.enable_latent_residual_add = bool(kwargs.get("enable_latent_residual_add", False))
        self.max_residual_ratio = float(kwargs.get("max_residual_ratio", 0.05))
        self.fidelity_gate_enabled = bool(kwargs.get("fidelity_gate_enabled", True))
        self.fidelity_gate_min = float(kwargs.get("fidelity_gate_min", 0.15))
        self.fidelity_gate_max = float(kwargs.get("fidelity_gate_max", 1.0))
        self.fidelity_conf_threshold = float(kwargs.get("fidelity_conf_threshold", 0.75))
        self.fidelity_conf_sharpness = float(kwargs.get("fidelity_conf_sharpness", 10.0))
        self.fidelity_energy_ref = float(kwargs.get("fidelity_energy_ref", 0.25))
        self.snr_adaptive_fusion = bool(kwargs.get("snr_adaptive_fusion", True))
        self.snr_adaptive_threshold = float(kwargs.get("snr_adaptive_threshold", 0.8))
        self.snr_adaptive_sharpness = float(kwargs.get("snr_adaptive_sharpness", 5.0))
        self.deterministic_eval = bool(kwargs.get("deterministic_eval", False))
        self.metricgan_plus_enabled = bool(kwargs.get("metricgan_plus_enabled", False))
        self.metricgan_weight = float(kwargs.get("metricgan_weight", 0.0))
        self.metricgan_disc_lr = float(kwargs.get("metricgan_disc_lr", 1e-4))
        self.metricgan_update_every = int(kwargs.get("metricgan_update_every", 8))
        self.metricgan_start_step = int(kwargs.get("metricgan_start_step", 100))
        self.metricgan_batch_items = int(kwargs.get("metricgan_batch_items", 1))
        self.wave_residual_enabled = bool(kwargs.get("wave_residual_enabled", False))
        self.wave_residual_num_filters = int(kwargs.get("wave_residual_num_filters", 256))
        self.wave_residual_filter_size = int(kwargs.get("wave_residual_filter_size", 16))
        self.wave_residual_hop_size = int(kwargs.get("wave_residual_hop_size", self.wave_residual_filter_size // 2))
        self.wave_residual_bottleneck_channels = int(kwargs.get("wave_residual_bottleneck_channels", 64))
        self.wave_residual_hidden_channels = int(kwargs.get("wave_residual_hidden_channels", 256))
        self.wave_residual_skip_channels = int(kwargs.get("wave_residual_skip_channels", 64))
        self.wave_residual_kernel_size = int(kwargs.get("wave_residual_kernel_size", 3))
        self.wave_residual_layers = int(kwargs.get("wave_residual_layers", 4))
        self.wave_residual_repeats = int(kwargs.get("wave_residual_repeats", 2))
        self.wave_residual_causal = bool(kwargs.get("wave_residual_causal", False))
        self.wave_residual_multiscale = bool(kwargs.get("wave_residual_multiscale", True))
        self.wave_residual_low_stride = int(kwargs.get("wave_residual_low_stride", 8))
        self.wave_residual_net: nn.Module | None = None
        self.wave_residual_low_head: nn.Conv1d | None = None
        self.wave_residual_high_head: nn.Conv1d | None = None
        self.alphas: nn.Parameter | None = None
        
        # SAD-RVQ Scheme Selection
        self.sad_rvq_scheme = str(kwargs.get("sad_rvq_scheme", "baseline")).lower()
        self.sad_rvq_scheme_a_weight = float(kwargs.get("sad_rvq_scheme_a_weight", 0.5))
        self.sad_rvq_scheme_d_enabled = bool(kwargs.get("sad_rvq_scheme_d_enabled", False))
        self.sad_rvq_scheme_d_gate_entropy_weight = float(kwargs.get("sad_rvq_scheme_d_gate_entropy_weight", 0.1))
        self.sad_rvq_scheme_d_acoustic_weight = float(kwargs.get("sad_rvq_scheme_d_acoustic_weight", 0.5))
        self.sad_rvq_scheme_d_gate_polar_weight = float(kwargs.get("sad_rvq_scheme_d_gate_polar_weight", 0.1))
        self.sad_rvq_scheme_d_acoustic_lr_scale = float(kwargs.get("sad_rvq_scheme_d_acoustic_lr_scale", 1.0))
        self.sad_rvq_scheme_g_entropy_quantile = float(kwargs.get("sad_rvq_scheme_g_entropy_quantile", 0.5))
        self.sad_rvq_scheme_h_min_temp = float(kwargs.get("sad_rvq_scheme_h_min_temp", 0.1))
        self.sad_rvq_scheme_d_final_weight = float(kwargs.get("sad_rvq_scheme_d_final_weight", 1.0))
        self.sad_rvq_scheme_train_mode = str(kwargs.get("sad_rvq_scheme_train_mode", "normal")).lower()
        self.sad_rvq_freeze_main_model = bool(kwargs.get("sad_rvq_freeze_main_model", False))
        
        # Scheme D: Initialize learnable router
        scheme_with_acoustic = self.sad_rvq_scheme in {"d", "e", "f", "g", "h"} or self.sad_rvq_scheme_d_enabled
        if scheme_with_acoustic:
            self.scheme_d_router = SchemeDLearnableGatingRouter(num_codebooks=8, feature_dim=256)
            self.scheme_d_acoustic_head = nn.Sequential(
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Linear(256, int(getattr(self.model, "output_channels", 1024))),
            )

            # Optional gradient scaling to give Scheme-D acoustic branch an effectively higher LR.
            if self.sad_rvq_scheme_d_acoustic_lr_scale > 1.0:
                scale = self.sad_rvq_scheme_d_acoustic_lr_scale
                for p in list(self.scheme_d_router.parameters()) + list(self.scheme_d_acoustic_head.parameters()):
                    if p.requires_grad:
                        p.register_hook(lambda g, s=scale: g * s)
        else:
            self.scheme_d_router = None
            self.scheme_d_acoustic_head = None

        if self.sad_rvq_freeze_main_model:
            for p in self.model.parameters():
                p.requires_grad = False
        
        if self.wave_residual_enabled:
            wave_norm = functools.partial(nn.BatchNorm1d, momentum=0.1)
            self.wave_residual_net = ConvTasNet(
                input_channels=3,
                output_channels=1,
                num_filters=self.wave_residual_num_filters,
                filter_size=self.wave_residual_filter_size,
                hop_size=self.wave_residual_hop_size,
                bottleneck_channels=self.wave_residual_bottleneck_channels,
                hidden_channels=self.wave_residual_hidden_channels,
                skip_channels=self.wave_residual_skip_channels,
                kernel_size=self.wave_residual_kernel_size,
                layers=self.wave_residual_layers,
                repeats=self.wave_residual_repeats,
                causal=self.wave_residual_causal,
                norm=wave_norm,
            )
            # Multi-scale zero-init adapter heads.
            # Step 1 output is exactly zero to preserve the discrete 3.0 baseline.
            self.wave_residual_low_head = nn.Conv1d(1, 1, kernel_size=1)
            self.wave_residual_high_head = nn.Conv1d(1, 1, kernel_size=1)
            nn.init.zeros_(self.wave_residual_low_head.weight)
            nn.init.zeros_(self.wave_residual_low_head.bias)
            nn.init.zeros_(self.wave_residual_high_head.weight)
            nn.init.zeros_(self.wave_residual_high_head.bias)
            alpha_init_prob = float(kwargs.get("alpha_init_prob", 0.05))
            alpha_init_prob = min(max(alpha_init_prob, 1e-4), 1.0 - 1e-4)
            alpha_init_logit = math.log(alpha_init_prob / (1.0 - alpha_init_prob))
            self.alphas = nn.Parameter(torch.full((2,), alpha_init_logit, dtype=torch.float32))
        
        self.val_metrics = kwargs.get("val_metrics")
        self.test_metrics = kwargs.get("test_metrics")
        log_cfg = kwargs.get("log_cfg")
        self.log_cfg = LogConfig() if log_cfg is None else log_cfg
        self.debug_sample = kwargs.get("debug_sample")

        self.metric_discriminator: MetricDiscriminator | None = None
        self.metric_disc_optimizer: Adam | None = None
        if self.metricgan_plus_enabled:
            self.metric_discriminator = MetricDiscriminator()
            self.metric_disc_optimizer = Adam(self.metric_discriminator.parameters(), lr=self.metricgan_disc_lr)

        trainable_param_patterns = kwargs.get("trainable_param_patterns")
        if trainable_param_patterns:
            self._apply_trainable_param_patterns(trainable_param_patterns)

        pretrained_ckpt = kwargs.get("pretrained_ckpt")
        if pretrained_ckpt:
            if not os.path.exists(pretrained_ckpt):
                raise FileNotFoundError(f"Configured pretrained_ckpt does not exist: {pretrained_ckpt}")
            print(f"--- 显式预加载主干权重: {pretrained_ckpt} ---")
            ckpt_data = torch.load(pretrained_ckpt, map_location="cpu")
            state_dict = ckpt_data.get("state_dict", ckpt_data)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"--- 预加载完成 (missing={len(missing)}, unexpected={len(unexpected)}) ---")

    @staticmethod
    def _limit_residual_energy(base_lat: Tensor, residual: Tensor, max_ratio: float, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
        """Clamp residual RMS relative to base latent RMS to avoid decoder phase collapse."""
        base_rms = base_lat.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(eps)
        res_rms = residual.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(eps)
        ratio = res_rms / base_rms
        scale = torch.clamp(max_ratio / ratio, max=1.0)
        return residual * scale, ratio

    @staticmethod
    def _sum_quantized_latent(latent: Tensor, num_codebooks: int) -> Tensor:
        if latent.ndim != 4:
            return latent
        if latent.shape[1] == num_codebooks:
            return latent.sum(dim=1)
        if latent.shape[2] == num_codebooks:
            return latent.sum(dim=2)
        return latent.sum(dim=1)

    def _decode_latent_to_wave(self, latent: Tensor, target_length: int | None = None) -> Tensor:
        if hasattr(self.nac, "generator"):
            wave = self.nac.generator(latent)
        else:
            wave = self.nac.decode(latent, no_sum=False, domain="q")
        if target_length is not None:
            wave = wave[..., :target_length]
        return wave

    def _predict_wave_residual(
        self,
        x_wave: Tensor,
        base_wave: Tensor,
        residual_hint: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if not self.wave_residual_enabled or self.wave_residual_net is None:
            return base_wave, None
        target_len = base_wave.shape[-1]
        
        if residual_hint is None:
            residual_hint = torch.zeros((base_wave.shape[0], 1, target_len), 
                                       dtype=base_wave.dtype, device=base_wave.device).detach()
        else:
            if hasattr(self.model, "summarize_continuous_hint"):
                residual_hint = self.model.summarize_continuous_hint(residual_hint)
            elif residual_hint.ndim == 3:
                residual_hint = residual_hint.mean(dim=1, keepdim=True)
            elif residual_hint.ndim == 4:
                residual_hint = residual_hint.mean(dim=1)
                if residual_hint.ndim == 3 and residual_hint.shape[1] != 1:
                    residual_hint = residual_hint.mean(dim=1, keepdim=True)
            residual_hint = residual_hint.detach()
            
            if residual_hint.shape[-1] != target_len:
                residual_hint = F.interpolate(
                    residual_hint, size=target_len, mode='nearest'
                )

        x_wave = x_wave[..., :target_len]
        residual_hint = residual_hint[..., :target_len]
        wave_input = torch.cat([x_wave, base_wave, residual_hint], dim=1)
        wave_delta_raw = self.wave_residual_net(wave_input)
        wave_delta_raw = wave_delta_raw[..., :target_len]

        if (
            self.wave_residual_multiscale
            and self.wave_residual_low_head is not None
            and self.wave_residual_high_head is not None
            and self.alphas is not None
        ):
            # Injection A (low-frequency envelope): pooled coarse branch.
            stride = max(2, int(self.wave_residual_low_stride))
            low_feat = F.avg_pool1d(wave_delta_raw, kernel_size=stride, stride=stride, ceil_mode=True)
            low_delta = self.wave_residual_low_head(low_feat)
            low_delta = F.interpolate(low_delta, size=target_len, mode="linear", align_corners=False)

            # Injection B (high-frequency details): full-resolution branch.
            high_delta = self.wave_residual_high_head(wave_delta_raw)

            alpha = torch.sigmoid(self.alphas).to(wave_delta_raw.dtype)
            wave_delta = alpha[0] * low_delta + alpha[1] * high_delta
        else:
            wave_delta = wave_delta_raw

        fused_wave = base_wave + wave_delta
        return fused_wave, wave_delta

    def _apply_trainable_param_patterns(self, patterns: Iterable[str]) -> None:
        normalized = [p for p in patterns if isinstance(p, str) and p.strip()]
        if not normalized:
            return
        for _, p in self.model.named_parameters():
            p.requires_grad = False
        matched = 0
        for name, p in self.model.named_parameters():
            if any(pattern in name for pattern in normalized):
                p.requires_grad = True
                matched += 1
        if matched == 0:
            raise ValueError(f"No model parameters matched trainable_param_patterns: {normalized}")

    def log_score(self, y_q, x_q, x_cont=None, return_raw_logits: bool = False):
        x_c_in = x_cont if x_cont is not None else torch.zeros_like(x_q)
        def model_wrapper(y_s, x_s, c_s):
            return self.model(y_s, x_s, None, c_cont=c_s)
        
        result = process_in_blocks((y_q, x_q, x_c_in), self.block_size, model_wrapper)
        
        if isinstance(result, tuple):
            if len(result) == 3:
                logits, residual, quality_map = result
            else:
                logits, residual = result
                quality_map = None
            logits_raw = logits.moveaxis(1, -1)
            log_p = logits_raw.log_softmax(dim=-1)
            if return_raw_logits:
                return log_p, residual, quality_map, logits_raw
            return log_p, residual, quality_map
        else:
            logits_raw = result.moveaxis(1, -1)
            log_p = logits_raw.log_softmax(dim=-1)
            if return_raw_logits:
                return log_p, None, None, logits_raw
            return log_p, None, None

    def _build_scheme_d_acoustic_logits(self, x_lat: Tensor, k_4plus: int, L: int) -> Tensor:
        feat_dim = 256
        acoustic_feat = F.interpolate(x_lat, size=L, mode="nearest").transpose(1, 2)  # [B, L, C]
        if acoustic_feat.shape[-1] < feat_dim:
            acoustic_feat = F.pad(acoustic_feat, (0, feat_dim - acoustic_feat.shape[-1]))
        elif acoustic_feat.shape[-1] > feat_dim:
            acoustic_feat = acoustic_feat[..., :feat_dim]
        acoustic_logits = self.scheme_d_acoustic_head(acoustic_feat)  # [B, L, V]
        return acoustic_logits.unsqueeze(1).expand(-1, k_4plus, -1, -1)  # [B, K4, L, V]

    def _scheme_d_fuse_logits(
        self,
        base_logits: Tensor,
        x_lat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # base_logits: [B, K, L, V]
        B, K, L, _ = base_logits.shape
        if K <= 3 or self.scheme_d_router is None or self.scheme_d_acoustic_head is None:
            return base_logits, None, None

        # Router gate for all codebooks, use only post-3 layers for enhancement
        feat_dim = 256
        acoustic_feat = F.interpolate(x_lat, size=L, mode="nearest").transpose(1, 2)
        if acoustic_feat.shape[-1] < feat_dim:
            acoustic_feat = F.pad(acoustic_feat, (0, feat_dim - acoustic_feat.shape[-1]))
        elif acoustic_feat.shape[-1] > feat_dim:
            acoustic_feat = acoustic_feat[..., :feat_dim]
        acoustic_feat_expanded = acoustic_feat.unsqueeze(1).expand(-1, K, -1, -1)
        acoustic_logits_4plus = self._build_scheme_d_acoustic_logits(x_lat, K - 3, L)
        fused_logits = base_logits.clone()

        scheme = self.sad_rvq_scheme
        router_gate = None

        # Scheme E: static takeover on post-3 layers
        if scheme == "e":
            fused_logits[:, 3:, :, :] = acoustic_logits_4plus
            return fused_logits, acoustic_logits_4plus, router_gate

        # Scheme F: residual logit addition on post-3 layers
        if scheme == "f":
            fused_logits[:, 3:, :, :] = base_logits[:, 3:, :, :] + acoustic_logits_4plus
            return fused_logits, acoustic_logits_4plus, router_gate

        # Scheme G: entropy-guided hard routing (non-learnable gate)
        if scheme == "g":
            probs = F.softmax(base_logits[:, 3:, :, :], dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1, keepdim=True)
            q = min(max(self.sad_rvq_scheme_g_entropy_quantile, 0.0), 1.0)
            threshold = torch.quantile(entropy.detach().reshape(-1), q)
            router_gate = (entropy > threshold).float()
            fused_logits[:, 3:, :, :] = base_logits[:, 3:, :, :] * (1.0 - router_gate) + acoustic_logits_4plus * router_gate
            return fused_logits, acoustic_logits_4plus, router_gate

        # Scheme H: temperature annealed soft routing (learnable gate)
        if scheme == "h":
            raw_gate_logits = self.scheme_d_router.gate_net(acoustic_feat_expanded.reshape(-1, acoustic_feat_expanded.shape[-1]))
            raw_gate_logits = raw_gate_logits.reshape(B, K, L, 1)[:, 3:, :, :]

            est_steps = 1
            trainer_obj = getattr(self, "_trainer", None)
            if trainer_obj is not None:
                est_steps = int(getattr(trainer_obj, "estimated_stepping_batches", 0) or 0)
            est_steps = max(est_steps, 1)
            progress = min(max(float(self.global_step) / float(est_steps), 0.0), 1.0)
            temp = max(1.0 - 0.9 * progress, self.sad_rvq_scheme_h_min_temp)
            router_gate = torch.sigmoid(raw_gate_logits / temp)
            if trainer_obj is not None:
                self.log("train/scheme_h_temp" if self.training else "val/scheme_h_temp", temp, prog_bar=False, sync_dist=True)
            fused_logits[:, 3:, :, :] = base_logits[:, 3:, :, :] * (1.0 - router_gate) + acoustic_logits_4plus * router_gate
            return fused_logits, acoustic_logits_4plus, router_gate

        # Scheme D(default): soft learnable routing
        router_gate = self.scheme_d_router(acoustic_feat_expanded)[:, 3:, :, :]  # [B, K4, L, 1]
        fused_logits[:, 3:, :, :] = base_logits[:, 3:, :, :] * (1.0 - router_gate) + acoustic_logits_4plus * router_gate
        return fused_logits, acoustic_logits_4plus, router_gate

    @staticmethod
    def _si_sdr_loss(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        proj = (pred * target).sum(dim=-1, keepdim=True) * target / (target.pow(2).sum(dim=-1, keepdim=True) + eps)
        noise = pred - proj
        ratio = proj.pow(2).sum(dim=-1) / (noise.pow(2).sum(dim=-1) + eps)
        si_sdr = 10.0 * torch.log10(ratio + eps)
        return -si_sdr.mean()

    def _pesq_target(self, pred: Tensor, target: Tensor, max_items: int = 1) -> Tensor:
        if pesq is None:
            return torch.full((min(pred.shape[0], max_items), 1), 0.5, device=pred.device, dtype=pred.dtype)

        scores: list[float] = []
        n = min(pred.shape[0], max_items)
        pred_np = pred[:n].detach().cpu().numpy()
        target_np = target[:n].detach().cpu().numpy()
        for i in range(n):
            try:
                s = pesq(16000, target_np[i, 0], pred_np[i, 0], "wb")
            except (BufferTooShortError, NoUtterancesError, ValueError) as e:
                warnings.warn(f"MetricGAN+ PESQ target fallback: {e}")
                s = 1.0
            # Normalize PESQ(roughly 1.0~4.5) to [0,1]
            s_norm = max(0.0, min(1.0, (float(s) - 1.0) / 3.5))
            scores.append(s_norm)
        return torch.tensor(scores, device=pred.device, dtype=pred.dtype).view(-1, 1)

    @override
    def step(self, batch, stage, batch_idx, metrics=None):
        x, y, _ = batch
        x_lat = self.nac.encoder(x)
        with torch.no_grad():
            y_lat_clean = self.nac.encoder(y) # 真正的黄金无损特征！

        xy_tok, xy_q = self.nac.encode(torch.cat([x, y]), no_sum=True, domain="q")
        x_tok, y_tok = xy_tok.chunk(2); x_q, y_q = xy_q.chunk(2)

        B, K, L = y_tok.shape
        lambd = torch.rand(B, device=y_tok.device)
        mask = torch.rand(y_tok.shape, device=y_tok.device) < lambd[:, None, None]
        y_lambda_q = y_q.clone()
        y_lambda_q.masked_fill_(mask.unsqueeze(1), 0)

        # 核心前向传播
        log_p, residual_pred, quality_pred, logits_raw = self.log_score(y_lambda_q, x_q, x_cont=x_lat, return_raw_logits=True)
        logits_for_ce = logits_raw
        
        # 1. 离散 CE Loss (猜词)
        V = logits_for_ce.shape[-1]
        ce_per_token = F.cross_entropy(logits_for_ce.reshape(-1, V), y_tok.reshape(-1), reduction="none").reshape(B, K, L)
        ce_loss = (ce_per_token * mask).sum() / (mask.sum() + 1e-8)
        total_loss = ce_loss
        self.log(f"{stage}/ce_loss", ce_loss, prog_bar=True, sync_dist=True)
        if K > 3:
            mask_4plus = mask[:, 3:, :]
            ce_4plus = (ce_per_token[:, 3:, :] * mask_4plus).sum() / (mask_4plus.sum() + 1e-8)
            self.log(f"{stage}/ce_loss_l4plus", ce_4plus, prog_bar=True, sync_dist=True)
            self.log(f"{stage}/ce_l4plus_base", ce_4plus, prog_bar=False, sync_dist=True)
        
        # SAD-RVQ Scheme A: Post-5-Layer Enhancement
        if self.sad_rvq_scheme == "a":
            if K > 3:
                # Boost entropy on Layer 4+ to encourage diversity
                probs = F.softmax(log_p, dim=-1)
                entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)
                entropy_layers_4plus = entropy[:, 3:, :]  # Layer 4 onwards
                if entropy_layers_4plus.numel() > 0:
                    entropy_boost_loss = -entropy_layers_4plus.mean()  # Negative entropy = boost diversity
                    total_loss = total_loss + self.sad_rvq_scheme_a_weight * entropy_boost_loss
                    self.log(f"{stage}/scheme_a_entropy_boost", entropy_boost_loss, prog_bar=True, sync_dist=True)
        
        # SAD-RVQ Scheme D/E/F/G/H: post-3-layer fusion family
        if self.sad_rvq_scheme in {"d", "e", "f", "g", "h"} and self.scheme_d_acoustic_head is not None and K > 3:
            fused_logits, acoustic_logits_4plus, router_gate = self._scheme_d_fuse_logits(logits_raw, x_lat)

            mask_4plus = mask[:, 3:, :]
            loss_final_4plus = F.cross_entropy(
                fused_logits[:, 3:, :, :].reshape(-1, V),
                y_tok[:, 3:, :].reshape(-1),
                reduction="none",
            ).reshape(B, K - 3, L)
            loss_final_4plus = (loss_final_4plus * mask_4plus).sum() / (mask_4plus.sum() + 1e-8)

            # For E/F/G: single clean objective on post-3 layers.
            if self.sad_rvq_scheme in {"e", "f", "g"}:
                total_loss = loss_final_4plus
                self.log(f"{stage}/scheme_post3_loss", loss_final_4plus, prog_bar=True, sync_dist=True)
            else:
                # D/H: dual supervision.
                loss_acoustic_4plus = F.cross_entropy(
                    acoustic_logits_4plus.reshape(-1, V),
                    y_tok[:, 3:, :].reshape(-1),
                    reduction="none",
                ).reshape(B, K - 3, L)
                loss_acoustic_4plus = (loss_acoustic_4plus * mask_4plus).sum() / (mask_4plus.sum() + 1e-8)

                if self.sad_rvq_scheme_train_mode == "acoustic_only":
                    total_loss = self.sad_rvq_scheme_d_acoustic_weight * loss_acoustic_4plus
                else:
                    total_loss = (
                        self.sad_rvq_scheme_d_final_weight * loss_final_4plus
                        + self.sad_rvq_scheme_d_acoustic_weight * loss_acoustic_4plus
                    )
                self.log(f"{stage}/scheme_d_loss_acoustic_l4plus", loss_acoustic_4plus, prog_bar=True, sync_dist=True)
            self.log(f"{stage}/scheme_d_loss_final_l4plus", loss_final_4plus, prog_bar=True, sync_dist=True)
            self.log(f"{stage}/ce_l4plus_fused", loss_final_4plus, prog_bar=False, sync_dist=True)

            # Gate-dependent regularization for schemes with gates.
            if router_gate is not None:
                gate_penalty = (router_gate * (1.0 - router_gate)).mean()
                if self.sad_rvq_scheme in {"d", "h"} and self.sad_rvq_scheme_train_mode != "acoustic_only":
                    total_loss = total_loss + self.sad_rvq_scheme_d_gate_polar_weight * gate_penalty
                self.log(f"{stage}/scheme_d_gate_polar_penalty", gate_penalty, prog_bar=True, sync_dist=True)

                gate_mean = router_gate.mean().detach()
                gate_std = router_gate.std().detach()
                self.log(f"{stage}/scheme_d_gate_mean", gate_mean, prog_bar=True, sync_dist=True)
                self.log(f"{stage}/scheme_d_gate_std", gate_std, prog_bar=False, sync_dist=True)

                gate_entropy = -(router_gate * torch.log(router_gate.clamp(min=1e-8)) + (1 - router_gate) * torch.log((1 - router_gate).clamp(min=1e-8)))
                gate_entropy_loss = -gate_entropy.mean()
                if self.sad_rvq_scheme in {"d", "h"} and self.sad_rvq_scheme_train_mode != "acoustic_only":
                    total_loss = total_loss + self.sad_rvq_scheme_d_gate_entropy_weight * gate_entropy_loss
                self.log(f"{stage}/scheme_d_gate_entropy", gate_entropy_loss, prog_bar=False, sync_dist=True)


        y_q_sum = self._sum_quantized_latent(y_q, K)

        if self.wave_residual_enabled:
            base_wave = self._decode_latent_to_wave(y_q_sum.detach(), target_length=y.shape[-1])
            y_pred, wave_delta = self._predict_wave_residual(x, base_wave, residual_pred)
            
            min_len = min(base_wave.shape[-1], y.shape[-1])
            base_wave = base_wave[..., :min_len]
            y_pred = y_pred[..., :min_len]
            y_loss_target = y[..., :min_len]
            wave_residual_target = (y_loss_target - base_wave).detach()
            true_residual = (y_lat_clean - y_q_sum).detach()

            if residual_pred is not None and self.direct_residual_weight > 0:
                direct_res_loss = F.l1_loss(residual_pred, true_residual)
                total_loss = total_loss + self.direct_residual_weight * direct_res_loss
                self.log(f"{stage}/direct_res_loss", direct_res_loss, prog_bar=True, sync_dist=True)

            if wave_delta is not None:
                wave_delta = wave_delta[..., :min_len]
                
                if self.residual_l1_weight > 0:
                    res_loss = F.l1_loss(wave_delta, wave_residual_target)
                    total_loss = total_loss + self.residual_l1_weight * res_loss
                    self.log(f"{stage}/res_loss", res_loss, prog_bar=True, sync_dist=True)

                if self.residual_cos_weight > 0:
                    res_cos = 1.0 - F.cosine_similarity(
                        wave_delta.flatten(1), wave_residual_target.flatten(1), dim=1
                    ).mean()
                    total_loss = total_loss + self.residual_cos_weight * res_cos
                    self.log(f"{stage}/res_cos_loss", res_cos, prog_bar=False, sync_dist=True)

                if self.residual_std_weight > 0:
                    pred_std = wave_delta.std(dim=-1).mean(dim=1)
                    tgt_std = wave_residual_target.std(dim=-1).mean(dim=1)
                    res_std = F.l1_loss(pred_std, tgt_std)
                    total_loss = total_loss + self.residual_std_weight * res_std
                    self.log(f"{stage}/res_std_loss", res_std, prog_bar=False, sync_dist=True)

                if quality_pred is not None and self.quality_branch_weight > 0:
                    local_err = (wave_delta.detach() - wave_residual_target).abs().mean(dim=1, keepdim=True)
                    err_ref = local_err.mean(dim=-1, keepdim=True).clamp_min(1e-6)
                    quality_target = torch.exp(-local_err / err_ref).clamp(min=0.0, max=1.0)
                    quality_loss = F.l1_loss(quality_pred, quality_target)
                    total_loss = total_loss + self.quality_branch_weight * quality_loss
                    self.log(f"{stage}/quality_loss", quality_loss, prog_bar=False, sync_dist=True)

                    if self.quality_smooth_weight > 0 and quality_pred.shape[-1] > 1:
                        q_smooth = (quality_pred[..., 1:] - quality_pred[..., :-1]).abs().mean()
                        total_loss = total_loss + self.quality_smooth_weight * q_smooth
                        self.log(f"{stage}/quality_smooth", q_smooth, prog_bar=False, sync_dist=True)

            if self.spec_loss is not None and self.spec_loss_weight > 0:
                spec_losses = self.spec_loss(y_pred, y_loss_target)
                spec_main = spec_losses["loss"]
                total_loss = total_loss + self.spec_loss_weight * spec_main
                self.log(f"{stage}/spec_loss", spec_main, prog_bar=True, sync_dist=True)

            if self.wave_l1_weight > 0:
                wave_l1 = F.l1_loss(y_pred, y_loss_target)
                total_loss = total_loss + self.wave_l1_weight * wave_l1
                self.log(f"{stage}/wave_l1", wave_l1, prog_bar=False, sync_dist=True)

            if self.alphas is not None:
                alpha_vals = torch.sigmoid(self.alphas)
                self.log(f"{stage}/alpha_low", alpha_vals[0], prog_bar=(stage == "train"), sync_dist=True)
                self.log(f"{stage}/alpha_high", alpha_vals[1], prog_bar=(stage == "train"), sync_dist=True)

            if self.si_sdr_weight > 0:
                si_sdr_loss = self._si_sdr_loss(y_pred, y_loss_target)
                total_loss = total_loss + self.si_sdr_weight * si_sdr_loss
                self.log(f"{stage}/si_sdr_loss", si_sdr_loss, prog_bar=True, sync_dist=True)

            if (
                stage == "train"
                and self.metricgan_plus_enabled
                and self.metric_discriminator is not None
                and self.metric_disc_optimizer is not None
                and self.metricgan_weight > 0
            ):
                self.metric_discriminator.to(y_pred.device)
                should_update_disc = (
                    self.global_step >= self.metricgan_start_step
                    and self.metricgan_update_every > 0
                    and (self.global_step % self.metricgan_update_every == 0)
                )

                metric_items = max(1, self.metricgan_batch_items)
                y_sub = y[:metric_items]
                y_pred_sub = y_pred[:metric_items]

                if should_update_disc:
                    with torch.no_grad():
                        metric_target = self._pesq_target(y_pred_sub, y_sub, max_items=metric_items)
                    disc_pred = self.metric_discriminator(y_pred_sub.detach(), y_sub.detach())
                    disc_loss = F.mse_loss(disc_pred, metric_target)
                    self.metric_disc_optimizer.zero_grad(set_to_none=True)
                    disc_loss.backward()
                    self.metric_disc_optimizer.step()
                    self.log(f"{stage}/metric_disc_loss", disc_loss.detach(), prog_bar=False, sync_dist=True)

                for p in self.metric_discriminator.parameters():
                    p.requires_grad_(False)
                metric_reward = self.metric_discriminator(y_pred_sub, y_sub).mean()
                metric_gen_loss = -metric_reward
                total_loss = total_loss + self.metricgan_weight * metric_gen_loss
                self.log(f"{stage}/metric_reward", metric_reward.detach(), prog_bar=True, sync_dist=True)
                self.log(f"{stage}/metric_gen_loss", metric_gen_loss.detach(), prog_bar=False, sync_dist=True)
                for p in self.metric_discriminator.parameters():
                    p.requires_grad_(True)
        elif residual_pred is not None:
            # 旧的 latent 残差路径保留作 fallback。
            true_residual = (y_lat_clean - y_q_sum).detach()

            if self.direct_residual_weight > 0:
                res_loss = F.l1_loss(residual_pred, true_residual)
                total_loss = total_loss + self.direct_residual_weight * res_loss
                self.log(f"{stage}/direct_res_loss", res_loss, prog_bar=True, sync_dist=True)
            elif self.residual_l1_weight > 0:
                res_loss = F.l1_loss(residual_pred, true_residual)
                total_loss = total_loss + self.residual_l1_weight * res_loss
                self.log(f"{stage}/res_loss", res_loss, prog_bar=True, sync_dist=True)

            if self.residual_cos_weight > 0:
                res_cos = 1.0 - F.cosine_similarity(
                    residual_pred.flatten(1), true_residual.flatten(1), dim=1
                ).mean()
                total_loss = total_loss + self.residual_cos_weight * res_cos
                self.log(f"{stage}/res_cos_loss", res_cos, prog_bar=False, sync_dist=True)

            if self.residual_std_weight > 0:
                pred_std = residual_pred.std(dim=-1).mean(dim=1)
                tgt_std = true_residual.std(dim=-1).mean(dim=1)
                res_std = F.l1_loss(pred_std, tgt_std)
                total_loss = total_loss + self.residual_std_weight * res_std
                self.log(f"{stage}/res_std_loss", res_std, prog_bar=False, sync_dist=True)

            if quality_pred is not None and self.quality_branch_weight > 0:
                local_err = (residual_pred.detach() - true_residual).abs().mean(dim=1, keepdim=True)
                err_ref = local_err.mean(dim=-1, keepdim=True).clamp_min(1e-6)
                quality_target = torch.exp(-local_err / err_ref).clamp(min=0.0, max=1.0)
                quality_loss = F.l1_loss(quality_pred, quality_target)
                total_loss = total_loss + self.quality_branch_weight * quality_loss
                self.log(f"{stage}/quality_loss", quality_loss, prog_bar=False, sync_dist=True)

                if self.quality_smooth_weight > 0 and quality_pred.shape[-1] > 1:
                    q_smooth = (quality_pred[..., 1:] - quality_pred[..., :-1]).abs().mean()
                    total_loss = total_loss + self.quality_smooth_weight * q_smooth
                    self.log(f"{stage}/quality_smooth", q_smooth, prog_bar=False, sync_dist=True)

            if (self.spec_loss is not None and self.spec_loss_weight > 0) or self.wave_l1_weight > 0 or self.si_sdr_weight > 0:
                if self.enable_latent_residual_add:
                    safe_residual, train_ratio = self._limit_residual_energy(
                        y_q_sum.detach(), residual_pred, max(self.max_residual_ratio, 1e-6)
                    )
                    pred_lat = y_q_sum.detach() + safe_residual
                    self.log(f"{stage}/res_ratio_train", train_ratio.mean(), prog_bar=False, sync_dist=True)
                else:
                    pred_lat = y_q_sum.detach()
                y_pred = self._decode_latent_to_wave(pred_lat, target_length=y.shape[-1])

                if self.spec_loss is not None and self.spec_loss_weight > 0:
                    spec_losses = self.spec_loss(y_pred, y)
                    spec_main = spec_losses["loss"]
                    total_loss = total_loss + self.spec_loss_weight * spec_main
                    self.log(f"{stage}/spec_loss", spec_main, prog_bar=True, sync_dist=True)

                if self.wave_l1_weight > 0:
                    wave_l1 = F.l1_loss(y_pred, y)
                    total_loss = total_loss + self.wave_l1_weight * wave_l1
                    self.log(f"{stage}/wave_l1", wave_l1, prog_bar=False, sync_dist=True)

                if self.si_sdr_weight > 0:
                    si_sdr_loss = self._si_sdr_loss(y_pred, y)
                    total_loss = total_loss + self.si_sdr_weight * si_sdr_loss
                    self.log(f"{stage}/si_sdr_loss", si_sdr_loss, prog_bar=True, sync_dist=True)

                if (
                    stage == "train"
                    and self.metricgan_plus_enabled
                    and self.metric_discriminator is not None
                    and self.metric_disc_optimizer is not None
                    and self.metricgan_weight > 0
                ):
                    self.metric_discriminator.to(y_pred.device)
                    should_update_disc = (
                        self.global_step >= self.metricgan_start_step
                        and self.metricgan_update_every > 0
                        and (self.global_step % self.metricgan_update_every == 0)
                    )

                    metric_items = max(1, self.metricgan_batch_items)
                    y_sub = y[:metric_items]
                    y_pred_sub = y_pred[:metric_items]

                    if should_update_disc:
                        with torch.no_grad():
                            metric_target = self._pesq_target(y_pred_sub, y_sub, max_items=metric_items)
                        disc_pred = self.metric_discriminator(y_pred_sub.detach(), y_sub.detach())
                        disc_loss = F.mse_loss(disc_pred, metric_target)
                        self.metric_disc_optimizer.zero_grad(set_to_none=True)
                        disc_loss.backward()
                        self.metric_disc_optimizer.step()
                        self.log(f"{stage}/metric_disc_loss", disc_loss.detach(), prog_bar=False, sync_dist=True)

                    for p in self.metric_discriminator.parameters():
                        p.requires_grad_(False)
                    metric_reward = self.metric_discriminator(y_pred_sub, y_sub).mean()
                    metric_gen_loss = -metric_reward
                    total_loss = total_loss + self.metricgan_weight * metric_gen_loss
                    self.log(f"{stage}/metric_reward", metric_reward.detach(), prog_bar=True, sync_dist=True)
                    self.log(f"{stage}/metric_gen_loss", metric_gen_loss.detach(), prog_bar=False, sync_dist=True)
                    for p in self.metric_discriminator.parameters():
                        p.requires_grad_(True)

        self.log(f"{stage}/loss", total_loss, prog_bar=False, sync_dist=True)

        if metrics or (stage == "val" and batch_idx == 0):
            solve_out = self.solve(x_tok, x_q, self.num_steps, x_cont=x_lat, x_wave=x)
            if self.wave_residual_enabled:
                y_hat = solve_out
            else:
                y_lat_fused = solve_out
                y_hat = self._decode_latent_to_wave(y_lat_fused, target_length=y.shape[-1])

            y_hat = y_hat[..., :y.shape[-1]]
            return {"loss": total_loss}, compute_metrics(y_hat, y, metrics), {"output": y_hat, "input": x, "clean": y}
            
        return {"loss": total_loss}, {}, {}

    @torch.no_grad()
    def solve(self, x_tok, x_q, num_steps, x_cont=None, x_wave=None):
        B, K, L = x_tok.shape
        y_tok = torch.full_like(x_tok, self.mask_token)
        import math
        final_residual = None
        final_confidence = None

        # 主干扩散：猜词
        for i in range(num_steps):
            mask = y_tok == self.mask_token
            if not mask.any(): break
            
            y_q_step = self.nac.quantizer.decode(y_tok.masked_fill(mask, 0), output_no_sum=True, domain="code")
            if y_q_step.shape != x_q.shape:
                if y_q_step.ndim == 4:
                    if y_q_step.shape[2] == x_q.shape[1]: y_q_step = y_q_step.transpose(1, 2)
                    elif y_q_step.shape[1] != x_q.shape[1]: y_q_step = y_q_step.reshape(x_q.shape)
            y_q_step = y_q_step.masked_fill(mask.unsqueeze(1), 0)
            
            log_p, residual_pred, _, logits_raw = self.log_score(y_q_step, x_q, x_cont=x_cont, return_raw_logits=True)

            # Apply post-3-layer fusion schemes in inference sampling loop as well.
            if self.sad_rvq_scheme in {"d", "e", "f", "g", "h"} and self.scheme_d_acoustic_head is not None:
                fused_logits, _, _ = self._scheme_d_fuse_logits(logits_raw, x_cont if x_cont is not None else x_q.mean(dim=2))
                log_p = fused_logits.log_softmax(dim=-1)
            
            # 用 EMA 聚合多步连续残差，减少只取最后一步导致的细节丢失。
            if residual_pred is not None:
                if final_residual is None:
                    final_residual = residual_pred
                else:
                    decay = min(max(self.residual_ema_decay, 0.0), 0.99)
                    final_residual = decay * final_residual + (1.0 - decay) * residual_pred

            probs = log_p.exp()
            final_confidence = probs.max(dim=-1).values.mean(dim=(1, 2), keepdim=True)
            if self.deterministic_eval and not self.training:
                sampled_tokens = probs.argmax(dim=-1)[mask]
            else:
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

        # ======================================================================
        # 物理级终极融合 (Orthogonal Fusion)
        # ======================================================================
        y_q_discrete = self.nac.quantizer.decode(y_tok, output_no_sum=False, domain="code")
        if y_q_discrete.ndim == 4:
            y_q_discrete = y_q_discrete.sum(dim=1) if y_q_discrete.shape[1] == K else y_q_discrete.sum(dim=2)

        if self.wave_residual_enabled and x_wave is not None:
            base_wave = self._decode_latent_to_wave(y_q_discrete, target_length=x_wave.shape[-1])
            wave_fused, _ = self._predict_wave_residual(x_wave, base_wave, final_residual)
            return wave_fused[..., : x_wave.shape[-1]]

        if isinstance(final_residual, Tensor) and self.fidelity_gate_enabled:
            if not self.enable_latent_residual_add:
                self._last_fidelity_gate_mean = 0.0
                return y_q_discrete

            # 保真门：置信度越高、残差能量越低，连续分支注入越强；否则自动收缩，保护主干语义与时域保真。
            base_rms = y_q_discrete.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
            res_rms = final_residual.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
            energy_ratio = res_rms / base_rms
            energy_gate = torch.exp(-energy_ratio / max(self.fidelity_energy_ref, 1e-6))

            if final_confidence is None:
                conf_gate = torch.ones((B, 1, 1), device=y_q_discrete.device, dtype=y_q_discrete.dtype)
            else:
                conf_gate = torch.sigmoid(
                    self.fidelity_conf_sharpness * (final_confidence.to(y_q_discrete.dtype) - self.fidelity_conf_threshold)
                )

            fidelity_gate = torch.clamp(
                conf_gate * energy_gate,
                min=self.fidelity_gate_min,
                max=self.fidelity_gate_max,
            )

            if self.snr_adaptive_fusion and x_cont is not None:
                sem_rms = y_q_discrete.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
                noi_rms = (x_cont - y_q_discrete).pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
                noise_ratio = noi_rms / sem_rms
                snr_gate = torch.sigmoid(
                    self.snr_adaptive_sharpness * (noise_ratio - self.snr_adaptive_threshold)
                ).to(fidelity_gate.dtype)
                fidelity_gate = torch.clamp(
                    fidelity_gate * snr_gate,
                    min=self.fidelity_gate_min,
                    max=self.fidelity_gate_max,
                )

            self._last_fidelity_gate_mean = float(fidelity_gate.mean().item())
            safe_residual, infer_ratio = self._limit_residual_energy(
                y_q_discrete, fidelity_gate * final_residual, max(self.max_residual_ratio, 1e-6)
            )
            self._last_residual_ratio_mean = float(infer_ratio.mean().item())
            return y_q_discrete + safe_residual

        # 干净的离散语义骨架 + 连续网络提取的高清相位残差
        if isinstance(final_residual, Tensor):
            if not self.enable_latent_residual_add:
                return y_q_discrete
            safe_residual, infer_ratio = self._limit_residual_energy(
                y_q_discrete, final_residual, max(self.max_residual_ratio, 1e-6)
            )
            self._last_residual_ratio_mean = float(infer_ratio.mean().item())
            return y_q_discrete + safe_residual
        return y_q_discrete

    def forward(self, x: torch.Tensor, return_nfe: bool = False, **kwargs):
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional"
        n_pad = (self.nac.downsampling_factor - x.shape[-1]) % self.nac.downsampling_factor
        x_pad = F.pad(x, (0, n_pad))
        x_lat = self.nac.encoder(x_pad)
        x_tok, x_q = self.nac.encode(x_pad, no_sum=True, domain="q")
        
        # solve 在 wave_residual_enabled 时直接返回波形晚期融合结果。
        solve_out = self.solve(x_tok, x_q, self.num_steps, x_cont=x_lat, x_wave=x_pad)
        if self.wave_residual_enabled:
            y_hat_pad = solve_out
        else:
            y_hat_pad = self._decode_latent_to_wave(solve_out, target_length=x_pad.shape[-1])
            
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