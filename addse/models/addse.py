import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from addse.layers import Snake1d

def get_rot_emb(dim: int, max_seq_len: int) -> tuple[Tensor, Tensor]:
    """Compute rotary embeddings."""
    assert dim % 2 == 0
    pos = torch.arange(max_seq_len)
    omega = 1 / 10000 ** (torch.arange(0, dim, 2) / dim)
    angles = pos[:, None] * omega[None, :]
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    return cos, sin

class ADDSERQDiT(nn.Module):
    def __init__(self, input_channels, output_channels, num_codebooks, hidden_dim, num_layers, num_heads, max_seq_len, conditional, time_independent):
        super().__init__()
        self.output_channels = output_channels # 关键：保存此属性
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
        if c is not None:
            c_p = self.input_proj_c(c.moveaxis(1, -1))
            c_time, c_dep = c_p.sum(dim=1) * self.dep_scale, c_p.transpose(1, 2).reshape(B * L, K, -1)
            t_dep = None if t is None else t.expand(B, -1).repeat_interleave(L, dim=0)
        else: c_time = c_dep = t_dep = None
        
        x_p = self.input_proj_x(x.moveaxis(1, -1))
        h = self.time_dit(x_p.sum(dim=1) * self.dep_scale, c_time, t)
        if self.dep_dit is not None:
            x_p = (x_p + h[:, None]) * self.skip_scale
            x_p = self.dep_dit(x_p.transpose(1, 2).reshape(B * L, K, -1), c_dep, t_dep).reshape(B, L, K, -1).transpose(1, 2)
        else: x_p = h[:, None]
        
        cond = c if t is None else (t[:, None, None] if c is None else (c + t[:, None, None]) * self.skip_scale)
        shift, scale = self.output_adaln(self.input_proj_c(cond.moveaxis(1, -1)) if self.input_proj_c and cond is not None else cond).chunk(2, dim=-1) if self.output_adaln and cond is not None else (0, 0)
        x_p = (self.output_norm(x_p) * (1 + scale) + shift) * self.skip_scale
        x_out = self.output_proj(x_p).moveaxis(-1, 1)
        return x_out.squeeze(2) if squeeze_output else x_out

class ADDSERQDiTParallel(ADDSERQDiT):
    """V3.4 QRC (Quantization Residual Compensation) 顶会架构
    彻底解耦离散语义与连续高频残差，真正实现双轨并联。
    """
    def __init__(self, **kwargs) -> None:
        self.input_channels = kwargs.get("input_channels")
        self.adapter_hidden = kwargs.get("adapter_hidden", 256)
        
        # 过滤 kwargs 传给父类
        base_kwargs = {k: v for k, v in kwargs.items() if k in ["input_channels", "output_channels", "num_codebooks", "hidden_dim", "num_layers", "num_heads", "max_seq_len", "conditional", "time_independent"]}
        super().__init__(**base_kwargs)
        
        # 独立的连续残差预测网络 (1D CNN, 感受野指数级扩大以捕获相位)
        self.refiner = nn.Sequential(
            nn.Conv1d(self.input_channels, self.adapter_hidden, kernel_size=3, padding=1),
            Snake1d(self.adapter_hidden),
            nn.Conv1d(self.adapter_hidden, self.adapter_hidden, kernel_size=3, padding=2, dilation=2),
            Snake1d(self.adapter_hidden),
            nn.Conv1d(self.adapter_hidden, self.adapter_hidden, kernel_size=3, padding=4, dilation=4),
            Snake1d(self.adapter_hidden),
            nn.Conv1d(self.adapter_hidden, self.adapter_hidden, kernel_size=3, padding=8, dilation=8),
            Snake1d(self.adapter_hidden),
            nn.Conv1d(self.adapter_hidden, self.input_channels, kernel_size=1),
        )
        self._reset_parallel_parameters()

    def _reset_parallel_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.refiner[-1].weight)
            nn.init.zeros_(self.refiner[-1].bias)

    def predict_residual(self, c_cont):
        """并联支路使命：预测被离散量化器无情丢弃的高频残差"""
        # 确保输入是连续特征 (B, C, L)
        if c_cont.ndim == 4:
            c_cont = c_cont.sum(dim=2) if c_cont.shape[2] > 1 else c_cont.squeeze(2)
        return self.refiner(c_cont)

    def forward(self, x, c, t, c_cont=None):
        # 离散主干绝对纯净！只负责粗粒度语义猜词，彻底摒弃交互污染
        z_base = super().forward(x, c, t) 
        return z_base

# 后续辅助类 ADDSEDiT, ADDSESelfAttentionBlock 等保持 Git 源码完整...
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
        q_rot, k_rot = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape), torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape)
        q, k = q * cos[None, None, :L] + q_rot * sin[None, None, :L], k * cos[None, None, :L] + k_rot * sin[None, None, :L]
        attn = (self.scale * q @ k.transpose(-2, -1)).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, L, C))

class ADDSEEmbeddingBlock(nn.Module):
    def __init__(self, dim, emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(emb_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.register_buffer("freqs", 2 * torch.pi * torch.randn(emb_dim)); self.register_buffer("phases", 2 * torch.pi * torch.rand(emb_dim))
    def forward(self, x):
        return self.mlp(2.0**0.5 * torch.cos(x[:, None] * self.freqs[None, :] + self.phases[None, :]))