import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from addse.layers import Snake1d
class ADDSERQDiT(nn.Module):
    """Residual Quantized Diffusion Transformer (RQDiT) backbone used in ADDSE."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_codebooks: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        conditional: bool,
        time_independent: bool,
    ) -> None:
        """Initialize the ADDSE RQDiT backbone.

        Args:
            input_channels: Number of input channels.
            output_channels: Number of output channels.
            num_codebooks: Number of codebooks.
            hidden_dim: Number of DiT hidden channels.
            num_layers: Number of DiT layers.
            num_heads: Number of DiT attention heads.
            max_seq_len: Maximum sequence length.
            conditional: Whether the model is conditional.
            time_independent: Whether the model is time-independent.
        """
        super().__init__()
        elementwise_affine = not conditional and time_independent
        self.input_proj_x = nn.Linear(input_channels, hidden_dim)
        self.input_proj_c = nn.Linear(input_channels, hidden_dim) if conditional else None
        self.time_dit = ADDSEDiT(hidden_dim, num_layers, num_heads, max_seq_len, elementwise_affine)
        self.dep_dit = (
            ADDSEDiT(hidden_dim, num_layers, num_heads, num_codebooks, elementwise_affine)
            if num_codebooks > 1
            else None
        )
        self.output_adaln = (
            None if elementwise_affine else nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        )
        self.output_norm = nn.LayerNorm(hidden_dim, elementwise_affine=elementwise_affine, eps=1e-6)
        self.output_proj = nn.Linear(hidden_dim, output_channels)
        self.t_emb = None if time_independent else ADDSEEmbeddingBlock(hidden_dim)
        self.skip_scale = 2**-0.5
        self.dep_scale = num_codebooks**-0.5

    def forward(self, x: Tensor, c: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: Diffused embeddings. Shape `(batch_size, input_channels, num_codebooks, seq_len)` or `(batch_size,
                input_channels, seq_len)`.
            c: Conditioning embeddings. Same shape as `x`.
            t: Time step or noise level. Shape `(batch_size,)`.

        Returns:
            Output tensor. Shape `(batch_size, output_channels, num_codebooks, seq_len)`.
        """
        assert x.ndim in (3, 4), f"Input must be 3- or 4-dimensional. Got shape {x.shape}."
        assert c is None or c.shape == x.shape, f"Conditioning input shape {c.shape} must match input shape {x.shape}."
        if squeeze_output := x.ndim == 3:
            x = x[:, :, None, :]
            c = None if c is None else c[:, :, None, :]
        B, _, K, L = x.shape
        if t is None:
            assert self.t_emb is None, "Got no time input, but time embedding layer was initialized."
        else:
            assert self.t_emb is not None, "Got time input, but time embedding layer was not initialized."
            t = self.t_emb(t)
        if c is None:
            assert self.input_proj_c is None, "Got no conditioning input, but conditioning layer was initialized."
            c_time = None
            c_dep = None
        else:
            assert self.input_proj_c is not None, "Got conditioning input, but conditioning layer was not initialized."
            c = c.moveaxis(1, -1)  # (B, K, L, C)
            c = self.input_proj_c(c)  # (B, K, L, C)
            c_time = c.sum(dim=1) * self.dep_scale  # (B, L, C)
            c_dep = c.transpose(1, 2).reshape(B * L, K, -1)  # (B * L, K, C)
            t_dep = None if t is None else t.expand(B, -1).repeat_interleave(L, dim=0)  # (B * L, C)
        x = x.moveaxis(1, -1)  # (B, K, L, C)
        x = self.input_proj_x(x)  # (B, K, L, C)
        h = self.time_dit(x.sum(dim=1) * self.dep_scale, c_time, t)  # (B, L, C)
        if self.dep_dit is None:
            assert K == 1, "Got multiple codebooks, but depth DiT was not initialized."
            x = h[:, None]  # (B, K, L, C)
        else:
            assert K > 1, "Got single codebook, but depth DiT was initialized."
            x = (x + h[:, None]) * self.skip_scale  # (B, K, L, C)
            x = x.transpose(1, 2).reshape(B * L, K, -1)  # (B * L, K, C)
            x = self.dep_dit(x, c_dep, t_dep)  # (B * L, K, C)
            x = x.reshape(B, L, K, -1).transpose(1, 2)  # (B, K, L, C)
        c_final = c if t is None else (t[:, None, None] if c is None else (c + t[:, None, None]) * self.skip_scale)
        if c_final is None:
            assert self.output_adaln is None, "Got no conditioning input, but AdaLN layer was initialized."
            output_shift, output_scale = 0, 0
        else:
            assert self.output_adaln is not None, "Got conditioning input, but AdaLN layer was not initialized."
            output_shift, output_scale = self.output_adaln(c_final).chunk(2, dim=-1)  # (B, K, L, C)
        x = (self.output_norm(x) * (1 + output_scale) + output_shift) * self.skip_scale
        x = self.output_proj(x).moveaxis(-1, 1)  # (B, C, K, L)
        return x.squeeze(2) if squeeze_output else x


class ADDSERQDiTParallel(ADDSERQDiT):
    """ADDSE RQDiT 并联连续自适应模型 (V3.3 Evolved)。
    
    核心修复：
    1. 注入位置：改为“输出端残差修正”，确保不破坏离散主干的去噪分布，保住 1.4+ 底分。
    2. Identity Start：通过零初始化和负偏置，确保训练初始状态为恒等映射。
    3. 感受野：4层空洞卷积 [1, 2, 4, 8] 提供约 620ms 的上下文参考。
    """
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_codebooks: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        conditional: bool,
        time_independent: bool,
        interaction_alpha: float = 0.1,
        adapter_hidden: int = 256,
        use_fusion_norm: bool = False,
        dynamic_alpha: bool = False,
        alpha_max: float = 0.08,
        fusion_mode: str = "mul",
        use_adain_align: bool = False,
        use_freq_dynamic_gate: bool = False,
        gate_kernel_size: int = 3,
        use_pitch_aware_gate: bool = False,
        pitch_gain: float = 0.1,
    ) -> None:
        # 调用父类初始化
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            num_codebooks=num_codebooks,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            conditional=conditional,
            time_independent=time_independent,
        )
        
        # 记录关键参数
        self.input_channels = input_channels
        self.adapter_hidden = adapter_hidden
        self.dynamic_alpha = dynamic_alpha
        self.alpha_max = alpha_max
        self.interaction_alpha = interaction_alpha
        self.pitch_gain = pitch_gain
        self.use_pitch_aware_gate = use_pitch_aware_gate
        self.use_freq_dynamic_gate = use_freq_dynamic_gate
        self.use_adain_align = use_adain_align
        self.use_fusion_norm = use_fusion_norm

        # 并联精炼器 (Refiner): 4层空洞卷积，空洞率 [1, 2, 4, 8]
        self.refiner = nn.Sequential(
            nn.Conv2d(input_channels, adapter_hidden, kernel_size=(3, 1), padding=(1, 0), dilation=1),
            nn.SiLU(),
            nn.Conv2d(adapter_hidden, adapter_hidden, kernel_size=(3, 1), padding=(2, 0), dilation=2),
            nn.SiLU(),
            nn.Conv2d(adapter_hidden, adapter_hidden, kernel_size=(3, 1), padding=(4, 0), dilation=4),
            nn.SiLU(),
            nn.Conv2d(adapter_hidden, adapter_hidden, kernel_size=(3, 1), padding=(8, 0), dilation=8),
            nn.SiLU(),
            nn.Conv2d(adapter_hidden, input_channels * 2, kernel_size=1),
        )

        self.gate_proj = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.alpha_proj = nn.Conv2d(input_channels, input_channels, kernel_size=1) if dynamic_alpha else None
        
        if use_freq_dynamic_gate:
            self.freq_gate = nn.Conv2d(input_channels, input_channels, kernel_size=(gate_kernel_size, 1), 
                                      padding=(gate_kernel_size // 2, 0), groups=input_channels)
        else:
            self.freq_gate = None

        if use_fusion_norm:
            self.x_norm = nn.LayerNorm(input_channels)
            self.c_norm = nn.LayerNorm(input_channels)
        else:
            self.x_norm, self.c_norm = None, None

        # 核心：初始化保护逻辑
        self._reset_parallel_parameters()

    def _reset_parallel_parameters(self) -> None:
        """强制实现恒等映射初始化，防止干扰主干信号。"""
        with torch.no_grad():
            # 1. 最后一层卷积设为 0，使初始 gamma=0, beta=0
            nn.init.zeros_(self.refiner[-1].weight)
            nn.init.zeros_(self.refiner[-1].bias)
            
            # 2. 门控偏置设为 -4.0，使 sigmoid 输出接近 0，初始关闭支路
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, -4.0)
            
            if self.alpha_proj is not None:
                nn.init.zeros_(self.alpha_proj.weight)
                nn.init.constant_(self.alpha_proj.bias, -4.0)

    def _to_4d(self, x: Tensor) -> tuple[Tensor, bool]:
        if x.ndim == 3: return x[:, :, None, :], True
        return x, False

    def forward(self, x: Tensor, c: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        # 1. 首先运行主干 DiT 推理，得到基础预测 z_base (PESQ ~1.4)
        z_base = super().forward(x, c, t) 
        
        if c is not None:
            # 2. 准备条件特征
            c_4d, _ = self._to_4d(c)
            
            # 3. 提取 Refiner 修正参数
            h_ref = self.refiner(c_4d)
            gamma, beta = h_ref.chunk(2, dim=1)
            
            # 4. 计算门控 (对应汇报中的自适应语音补偿逻辑)
            gate = torch.sigmoid(self.gate_proj(c_4d))
            alpha = self.alpha_max * torch.sigmoid(self.alpha_proj(c_4d)) if self.alpha_proj else self.interaction_alpha
            alpha_total = alpha * gate
            
            # 5. 在输出端进行残差修正 (z_base 为 3D 或 4D，需注意维度匹配)
            # 公式: y = z_base + alpha * (tanh(gamma) * z_base + beta)
            if z_base.ndim == 3:
                alpha_total = alpha_total.squeeze(2)
                gamma = gamma.squeeze(2)
                beta = beta.squeeze(2)
            
            # 使用 tanh 限制 gamma 范围，增强数值稳定性
            return z_base + alpha_total * (torch.tanh(gamma) * z_base + beta)
        
        return z_base
class ADDSEDiT(nn.Module):
    """ADDSE DiT."""

    cos_emb: Tensor
    sin_emb: Tensor

    def __init__(self, dim: int, num_layers: int, num_heads: int, max_seq_len: int, elementwise_affine: bool) -> None:
        """Initialize the ADDSE DiT."""
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.blocks = nn.ModuleList([ADDSEDiTBlock(dim, num_heads, elementwise_affine) for _ in range(num_layers)])
        cos_emb, sin_emb = get_rot_emb(dim // num_heads, max_seq_len)
        self.register_buffer("cos_emb", cos_emb)
        self.register_buffer("sin_emb", sin_emb)
        self.skip_scale = 2**-0.5

    def forward(self, x: Tensor, c: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        """Forward pass."""
        c = c if t is None else (t[:, None] if c is None else (c + t[:, None]) * self.skip_scale)
        for block in self.blocks:
            x = block(x, c, self.cos_emb, self.sin_emb)
        return x


class ADDSEDiTBlock(nn.Module):
    """ADDSE DiT block."""

    def __init__(self, dim: int, num_heads: int, elementwise_affine: bool) -> None:
        """Initialize the ADDSE DiT block."""
        super().__init__()
        self.norm_1 = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=1e-6)
        self.msa = ADDSESelfAttentionBlock(dim, num_heads)
        self.norm_2 = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU("tanh"), nn.Linear(4 * dim, dim))
        self.adaln = None if elementwise_affine else nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.skip_scale = 2**-0.5

    def forward(self, x: Tensor, c: Tensor | None, cos_emb: Tensor, sin_emb: Tensor) -> Tensor:
        """Forward pass."""
        if c is None:
            assert self.adaln is None, "Got no conditioning input, but AdaLN layer was initialized."
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = 0, 0, 1, 0, 0, 1
        else:
            assert self.adaln is not None, "Got conditioning input, but AdaLN layer was not initialized."
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(c).chunk(6, dim=-1)
        msa_in = (self.norm_1(x) * (1 + scale_msa) + shift_msa) * self.skip_scale
        msa_out = self.msa(msa_in, cos_emb, sin_emb)
        x = (x + gate_msa * msa_out) * self.skip_scale
        mlp_in = (self.norm_2(x) * (1 + scale_mlp) + shift_mlp) * self.skip_scale
        mlp_out = self.mlp(mlp_in)
        return (x + gate_mlp * mlp_out) * self.skip_scale


class ADDSESelfAttentionBlock(nn.Module):
    """ADDSE self-attention block."""

    def __init__(self, dim: int, num_heads: int) -> None:
        """Initialize the ADDSE self-attention block."""
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, cos_emb: Tensor, sin_emb: Tensor) -> Tensor:
        """Forward pass."""
        B, L, C = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_rot = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape)
        k_rot = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape)
        q = q * cos_emb[None, None, :L] + q_rot * sin_emb[None, None, :L]
        k = k * cos_emb[None, None, :L] + k_rot * sin_emb[None, None, :L]
        attn = self.scale * q @ k.transpose(-2, -1)
        x = attn.softmax(dim=-1) @ v
        x = x.transpose(1, 2).reshape(B, L, C)
        return self.proj(x)


class ADDSEEmbeddingBlock(nn.Module):
    """ADDSE noise embedding block with Fourier features."""

    freqs: Tensor
    phases: Tensor

    def __init__(self, dim: int, emb_dim: int = 256) -> None:
        """Initialize the ADDSE time embedding block."""
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(emb_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.register_buffer("freqs", 2 * torch.pi * torch.randn(emb_dim))
        self.register_buffer("phases", 2 * torch.pi * torch.rand(emb_dim))
        self.scale = 2**0.5

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.scale * torch.cos(x[:, None] * self.freqs[None, :] + self.phases[None, :])
        return self.mlp(x)


def get_rot_emb(dim: int, max_seq_len: int) -> tuple[Tensor, Tensor]:
    """Compute rotary embeddings. Shape `(max_seq_len, dim)`."""
    assert dim % 2 == 0, "dim should be divisible by 2"
    pos = torch.arange(max_seq_len)
    omega = 1 / 10000 ** (torch.arange(0, dim, 2) / dim)
    angles = pos[:, None] * omega[None, :]
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    return cos, sin
