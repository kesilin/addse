from collections.abc import Callable, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F

class Snake1d(nn.Module):
    """Snake 激活函数，用于增强周期性信号（如语音）的拟合能力。"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 形状自适应处理 3D/4D
        alpha = self.alpha.view(1, -1, *([1] * (x.ndim - 2)))
        return x + (1.0 / (alpha + 1e-9)) * torch.pow(torch.sin(alpha * x), 2)

class GroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, causal: bool = False) -> None:
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(f"`num_channels` must be divisible by `num_groups`")
        self.num_groups, self.num_channels, self.eps, self.causal = num_groups, num_channels, eps, causal
        self.weight, self.bias = nn.Parameter(torch.zeros(num_channels)), nn.Parameter(torch.zeros(num_channels))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return group_norm(x, self.num_groups, self.weight, self.bias, self.eps, self.causal, False)

class LayerNorm(nn.Module):
    def __init__(self, num_channels: int, element_wise: bool = False, frame_wise: bool = False, causal: bool = False, center: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_channels, self.element_wise, self.frame_wise, self.causal, self.eps = element_wise, frame_wise, causal, eps
        self.weight = nn.Parameter(torch.zeros(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels)) if center else None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.element_wise:
            x = x.moveaxis(1, -1)
            x = F.layer_norm(x, x.shape[-1:], 1 + self.weight, self.bias, self.eps)
            return x.moveaxis(-1, 1)
        return group_norm(x, 1, self.weight, self.bias, self.eps, self.causal, self.frame_wise)

class InstanceNorm(GroupNorm):
    def __init__(self, num_channels: int, eps: float = 1e-5, causal: bool = False) -> None:
        super().__init__(num_groups=num_channels, num_channels=num_channels, eps=eps, causal=causal)

class BatchNorm(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, track_running_stats: bool = True, momentum: float | None = 0.1) -> None:
        super().__init__()
        self.num_channels, self.eps, self.track_running_stats, self.momentum = num_channels, eps, track_running_stats, momentum
        self.weight, self.bias = nn.Parameter(torch.zeros(num_channels)), nn.Parameter(torch.zeros(num_channels))
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_channels)); self.register_buffer("running_var", torch.ones(num_channels))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.reshape(x.shape[0], x.shape[1], -1)
        h = F.batch_norm(h, self.running_mean, self.running_var, 1 + self.weight, self.bias, self.training, self.momentum or 0.1, self.eps)
        return h.reshape(x.shape)

class BandSplit(nn.Module):
    def __init__(self, subband_idx, input_channels, output_channels, norm):
        super().__init__()
        self.subband_idx = subband_idx
        self.norm = nn.ModuleList([norm(2 * input_channels * (end - start)) for start, end in subband_idx])
        self.fc = nn.ModuleList([nn.Conv1d(2 * input_channels * (end - start), output_channels, 1) for start, end in subband_idx])
    def forward(self, x):
        B, _, _, T = x.shape; x = torch.view_as_real(x); out = []
        for i, (start, end) in enumerate(self.subband_idx):
            h = x[:, :, start:end, :, :].moveaxis(-1, 1).reshape(B, -1, T)
            out.append(self.fc[i](self.norm[i](h)))
        return torch.stack(out, dim=2)

class BandMerge(nn.Module):
    def __init__(self, subband_idx, input_channels, output_channels, num_channels, norm, mlp, residual):
        super().__init__()
        self.mlp_mask = nn.ModuleList([mlp(num_channels, 2 * input_channels * output_channels * (end - start), norm) for start, end in subband_idx])
        self.mlp_res = nn.ModuleList([mlp(num_channels, 2 * output_channels * (end - start), norm) for start, end in subband_idx]) if residual else None
        self.input_channels, self.output_channels = input_channels, output_channels
    def forward(self, x):
        B, _, K, T = x.shape
        submasks = [self.mlp_mask[i](x[:, :, i, :]).reshape(B, 2, self.input_channels, self.output_channels, -1, T) for i in range(K)]
        mask = torch.complex(*torch.cat(submasks, dim=-2).unbind(1))
        if self.mlp_res is None: return mask, None
        subres = [self.mlp_res[i](x[:, :, i, :]).reshape(B, 2, self.output_channels, -1, T) for i in range(K)]
        res = torch.complex(*torch.cat(subres, dim=-2).unbind(1))
        return mask, res

def group_norm(x, num_groups, weight, bias, eps, causal, frame_wise):
    if causal:
        h = x.reshape(x.shape[0], num_groups, x.shape[1] // num_groups, -1, x.shape[-1])
        mu_2 = h.pow(2).mean((2, 3), keepdim=True).cumsum(-1) / torch.arange(1, h.shape[-1] + 1, device=h.device)
        return (h / (mu_2 + eps).sqrt()).reshape(x.shape[0], x.shape[1], -1) * (1 + weight).unsqueeze(1)
    if frame_wise:
        h = x.moveaxis(-1, 1).flatten(0, 1)
        h = F.group_norm(h, num_groups, (1 + weight), bias, eps)
        return h.unflatten(0, (x.shape[0], x.shape[-1])).moveaxis(1, -1)
    return F.group_norm(x, num_groups, (1 + weight), bias, eps)