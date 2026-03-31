import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from addse.layers import Snake1d

def get_rot_emb(dim: int, max_seq_len: int) -> tuple[Tensor, Tensor]:
    """Compute rotary embeddings."""
    assert dim % 2 == 0, "dim should be divisible by 2"
    pos = torch.arange(max_seq_len)
    omega = 1 / 10000 ** (torch.arange(0, dim, 2) / dim)
    angles = pos[:, None] * omega[None, :]
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    return cos, sin

class ADDSERQDiT(nn.Module):
    def __init__(self, input_channels, output_channels, num_codebooks, hidden_dim, num_layers, num_heads, max_seq_len, conditional, time_independent):
        super().__init__()
        elementwise_affine = not conditional and time_independent
        self.input_proj_x = nn.Linear(input_channels, hidden_dim)
        self.input_proj_c = nn.Linear(input_channels, hidden_dim) if conditional else None
        self.time_dit = ADDSEDiT(hidden_dim, num_layers, num_heads, max_seq_len, elementwise_affine)
        self.dep_dit = ADDSEDiT(hidden_dim, num_layers, num_heads, num_codebooks, elementwise_affine) if num_codebooks > 1 else None
        self.output_adaln = None if elementwise_affine else nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim, elementwise_affine=elementwise_affine, eps=1e-6)
        self.output_proj = nn.Linear(hidden_dim, output_channels)
        self.t_emb = None if time_independent else ADDSEEmbeddingBlock(hidden_dim)
        self.skip_scale, self.dep_scale = 2**-0.5, num_codebooks**-0.5

    def forward(self, x: Tensor, c: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        if squeeze_output := x.ndim == 3:
            x, c = x[:, :, None, :], (None if c is None else c[:, :, None, :])
        B, _, K, L = x.shape
        if t is not None: t = self.t_emb(t)
        
        if c is None:
            c_time, c_dep = None, None
        else:
            c_p = self.input_proj_c(c.moveaxis(1, -1))
            c_time, c_dep = c_p.sum(dim=1) * self.dep_scale, c_p.transpose(1, 2).reshape(B * L, K, -1)
            t_dep = None if t is None else t.expand(B, -1).repeat_interleave(L, dim=0)

        x = self.input_proj_x(x.moveaxis(1, -1))
        h = self.time_dit(x.sum(dim=1) * self.dep_scale, c_time, t)
        if self.dep_dit is not None:
            x = (x + h[:, None]) * self.skip_scale
            x = self.dep_dit(x.transpose(1, 2).reshape(B * L, K, -1), c_dep, t_dep if t is not None else None).reshape(B, L, K, -1).transpose(1, 2)
        else: x = h[:, None]
        
        c_final = c if t is None else (t[:, None, None] if c is None else (c + t[:, None, None]) * self.skip_scale)
        if c_final is None:
            output_shift, output_scale = 0, 0
        else:
            # 兼容性修正：如果 c_final 是原始维度，需映射到 hidden_dim
            if c_final.shape[-1] != self.output_norm.normalized_shape[0]:
                c_final = self.input_proj_c(c_final.moveaxis(1, -1))
            output_shift, output_scale = self.output_adaln(c_final).chunk(2, dim=-1)
            
        x = (self.output_norm(x) * (1 + output_scale) + output_shift) * self.skip_scale
        x = self.output_proj(x).moveaxis(-1, 1)
        return x.squeeze(2) if squeeze_output else x

class ADDSERQDiTParallel(ADDSERQDiT):
    """ADDSE V3.3 Evolved: 特征解耦 + 4层空洞卷积架构。"""
    def __init__(self, **kwargs) -> None:
        self.input_channels, self.adapter_hidden = kwargs.get("input_channels"), kwargs.get("adapter_hidden", 256)
        self.alpha_max, self.interaction_alpha, self.dynamic_alpha = kwargs.get("alpha_max", 0.08), kwargs.get("interaction_alpha", 0.1), kwargs.get("dynamic_alpha", False)
        
        f_kwargs = {k: v for k, v in kwargs.items() if k in ["input_channels", "output_channels", "num_codebooks", "hidden_dim", "num_layers", "num_heads", "max_seq_len", "conditional", "time_independent"]}
        super().__init__(**f_kwargs)

        self.refiner = nn.Sequential(
            nn.Conv2d(self.input_channels, self.adapter_hidden, kernel_size=(3, 1), padding=(1, 0), dilation=1),
            Snake1d(self.adapter_hidden), 
            nn.Conv2d(self.adapter_hidden, self.adapter_hidden, kernel_size=(3, 1), padding=(2, 0), dilation=2),
            Snake1d(self.adapter_hidden),
            nn.Conv2d(self.adapter_hidden, self.adapter_hidden, kernel_size=(3, 1), padding=(4, 0), dilation=4),
            Snake1d(self.adapter_hidden),
            nn.Conv2d(self.adapter_hidden, self.adapter_hidden, kernel_size=(3, 1), padding=(8, 0), dilation=8),
            Snake1d(self.adapter_hidden),
            nn.Conv2d(self.adapter_hidden, self.input_channels * 2, kernel_size=1),
        )
        self.gate_proj = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1)
        self.alpha_proj = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1) if self.dynamic_alpha else None
        self._reset_parallel_parameters()

    def _reset_parallel_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.refiner[-1].weight); nn.init.zeros_(self.refiner[-1].bias)
            nn.init.constant_(self.gate_proj.bias, -4.0)

    def forward(self, x: Tensor, c: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        # 【核心修正】调用主干时 c=None，防止新特征干扰扩散模型底座
        # 满足 backbone 对 Tensor 的需求，主干将使用 time_emb 进行稳定预测
        z_base = super().forward(x, c=None, t=t) 
        
        if c is not None:
            c_4d = c[:, :, None, :] if c.ndim == 3 else c
            gamma, beta = self.refiner(c_4d).chunk(2, dim=1)
            gate = torch.sigmoid(self.gate_proj(c_4d))
            alpha = self.alpha_max * torch.sigmoid(self.alpha_proj(c_4d)) if self.alpha_proj else self.interaction_alpha
            alpha_total = alpha * gate
            if z_base.ndim == 3:
                alpha_total, gamma, beta = alpha_total.squeeze(2), gamma.squeeze(2), beta.squeeze(2)
            # 在主干 ~1.7 分的基础上进行相位细节修补
            return z_base + alpha_total * (torch.tanh(gamma) * z_base + beta)
        return z_base

class ADDSEDiT(nn.Module):
    def __init__(self, dim, num_layers, num_heads, max_seq_len, elementwise_affine):
        super().__init__()
        self.blocks = nn.ModuleList([ADDSEDiTBlock(dim, num_heads, elementwise_affine) for _ in range(num_layers)])
        cos, sin = get_rot_emb(dim // num_heads, max_seq_len)
        self.register_buffer("cos_emb", cos); self.register_buffer("sin_emb", sin)
        self.skip_scale = 2**-0.5
    def forward(self, x, c, t):
        c = c if t is None else (t[:, None] if c is None else (c + t[:, None]) * self.skip_scale)
        for block in self.blocks: x = block(x, c, self.cos_emb, self.sin_emb)
        return x

class ADDSEDiTBlock(nn.Module):
    def __init__(self, dim, num_heads, elementwise_affine):
        super().__init__()
        self.norm_1, self.msa = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=1e-6), ADDSESelfAttentionBlock(dim, num_heads)
        self.norm_2, self.mlp = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=1e-6), nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU("tanh"), nn.Linear(4 * dim, dim))
        self.adaln = None if elementwise_affine else nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.skip_scale = 2**-0.5
    def forward(self, x, c, cos, sin):
        sh1, sc1, g1, sh2, sc2, g2 = self.adaln(c).chunk(6, dim=-1) if (self.adaln and c is not None) else (0, 0, 1, 0, 0, 1)
        x = (x + g1 * self.msa((self.norm_1(x) * (1 + sc1) + sh1) * self.skip_scale, cos, sin)) * self.skip_scale
        return (x + g2 * self.mlp((self.norm_2(x) * (1 + sc2) + sh2) * self.skip_scale)) * self.skip_scale

class ADDSESelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
    def forward(self, x, cos, sin):
        B, L, C = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_rot = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape)
        k_rot = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape)
        q = q * cos[None, None, :L] + q_rot * sin[None, None, :L]
        k = k * cos[None, None, :L] + k_rot * sin[None, None, :L]
        return self.proj(((self.scale * q @ k.transpose(-2, -1)).softmax(dim=-1) @ v).transpose(1, 2).reshape(B, L, C))

class ADDSEEmbeddingBlock(nn.Module):
    def __init__(self, dim, emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(emb_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.register_buffer("freqs", 2 * torch.pi * torch.randn(emb_dim))
        self.register_buffer("phases", 2 * torch.pi * torch.rand(emb_dim))
    def forward(self, x):
        return self.mlp(2.0**0.5 * torch.cos(x[:, None] * self.freqs[None, :] + self.phases[None, :]))