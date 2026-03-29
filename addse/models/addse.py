import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
    """ADDSE RQDiT with lightweight continuous adaptation in parallel.

    The discrete diffusion backbone is unchanged. A continuous branch derived from
    conditioning embeddings is used as a calibration gate before the backbone:

        x' = x * (1 + alpha * sigmoid(W_g(Adapter(c))))

    This keeps pretrained ADDSE behavior largely intact while enabling
    low-risk parallel fusion experiments.
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
        self.interaction_alpha = interaction_alpha
        self.use_fusion_norm = use_fusion_norm
        self.dynamic_alpha = dynamic_alpha
        self.alpha_max = alpha_max
        self.fusion_mode = fusion_mode
        self.use_adain_align = use_adain_align
        self.use_freq_dynamic_gate = use_freq_dynamic_gate
        self.use_pitch_aware_gate = use_pitch_aware_gate
        self.pitch_gain = pitch_gain
        # Adapter -> Refiner: 4-layer dilated residual conv "refinery" producing FiLM params
        # dilation schedule chosen to expand temporal receptive field: [1,2,4,8]
        self.refiner = nn.Sequential(
            nn.Conv2d(input_channels, adapter_hidden, kernel_size=(3, 1), padding=(1, 0), dilation=1),
            nn.SiLU(),
            nn.Conv2d(adapter_hidden, adapter_hidden, kernel_size=(3, 1), padding=(2, 0), dilation=2),
            nn.SiLU(),
            nn.Conv2d(adapter_hidden, adapter_hidden, kernel_size=(3, 1), padding=(4, 0), dilation=4),
            nn.SiLU(),
            nn.Conv2d(adapter_hidden, input_channels * 2, kernel_size=1),
        )
        # Initialize final refiner conv to zero so adapter starts as identity
        try:
            last_conv = self.refiner[-1]
            if isinstance(last_conv, nn.Conv2d):
                nn.init.zeros_(last_conv.weight)
                if last_conv.bias is not None:
                    nn.init.zeros_(last_conv.bias)
        except Exception:
            pass
        self.gate_proj = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.parallel_proj = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.alpha_proj = nn.Conv2d(input_channels, input_channels, kernel_size=1) if dynamic_alpha else None
        if use_freq_dynamic_gate:
            pad_k = max(1, gate_kernel_size) // 2
            self.freq_gate = nn.Conv2d(
                input_channels,
                input_channels,
                # (K, 1): operate along codebook axis (pseudo-frequency), not time axis.
                kernel_size=(max(1, gate_kernel_size), 1),
                padding=(pad_k, 0),
                groups=input_channels,
            )
        else:
            self.freq_gate = None

        if self.fusion_mode not in ("mul", "add"):
            raise ValueError(f"Unsupported fusion_mode={fusion_mode}. Use 'mul' or 'add'.")

        if use_fusion_norm:
            self.x_norm = nn.LayerNorm(input_channels)
            self.c_norm = nn.LayerNorm(input_channels)
            # Learnable channel-wise scale, initialized close to interaction_alpha.
            self.gamma = nn.Parameter(torch.full((1, input_channels, 1, 1), interaction_alpha))
        else:
            self.x_norm = None
            self.c_norm = None
            self.gamma = None

    def _to_4d(self, x: Tensor) -> tuple[Tensor, bool]:
        if x.ndim == 3:
            return x[:, :, None, :], True
        if x.ndim == 4:
            return x, False
        raise ValueError(f"Input must be 3D/4D. Got shape {x.shape}.")

    def _apply_channel_norm(self, x: Tensor, ln: nn.LayerNorm | None) -> Tensor:
        if ln is None:
            return x
        x_n = x.moveaxis(1, -1)  # (B, K, L, C)
        x_n = ln(x_n)
        return x_n.moveaxis(-1, 1)  # (B, C, K, L)

    def _adain_align(self, src: Tensor, ref: Tensor, eps: float = 1e-5) -> Tensor:
        # AdaIN over (K, L) dimensions for each (B, C).
        src_mean = src.mean(dim=(2, 3), keepdim=True)
        src_std = src.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        ref_mean = ref.mean(dim=(2, 3), keepdim=True)
        ref_std = ref.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        src_norm = (src - src_mean) / src_std
        return src_norm * ref_std + ref_mean

    def _pitch_proxy(self, c_4d: Tensor) -> Tensor:
        # Energy-variance proxy: compute per-frame variance across channel dim
        # (lower variance -> voiced/tonal; higher variance -> noisy).
        # Input `c_4d` shape: (B, C, K, L). We average over codebook axis K,
        # compute variance across channels, smooth and return shape (B,1,1,L).
        # Output shape: (B, 1, 1, L), broadcastable to gate logits.
        # average over codebooks to reduce quantizer artifacts
        z = c_4d.mean(dim=2)  # (B, C, L)
        # variance across channels
        var = z.var(dim=1, unbiased=False)  # (B, L)
        # normalize per-sample
        var = var - var.mean(dim=-1, keepdim=True)
        denom = (var.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        if denom.abs().min() == 0:
            return torch.zeros(z.shape[0], 1, 1, z.shape[-1], device=z.device, dtype=z.dtype)
        rho = var / denom
        rho = rho.unsqueeze(1)  # (B,1,L)
        rho = F.avg_pool1d(rho, kernel_size=9, stride=1, padding=4)
        rho = torch.relu(rho)
        return rho[:, :, None, :]

    def forward(self, x: Tensor, c: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        x_4d, squeezed = self._to_4d(x)
        if c is not None:
            c_4d, _ = self._to_4d(c)
            x_in = self._apply_channel_norm(x_4d, self.x_norm)
            c_in = self._apply_channel_norm(c_4d, self.c_norm)
            # refine conditioning to produce FiLM parameters (gamma, beta)
            h_ref = self.refiner(c_in)  # (B, 2*C, K, L)

            # split into FiLM params first, then optionally align each to x_in
            gamma, beta = h_ref.chunk(2, dim=1)  # each: (B, C, K, L)
            if self.use_adain_align:
                # align gamma and beta separately to avoid channel-size collisions
                gamma = self._adain_align(gamma, x_in)
                beta = self._adain_align(beta, x_in)

            # compute gating / alpha dynamic coefficients from original conditioning
            gate_logits = self.gate_proj(c_in)
            if self.freq_gate is not None:
                gate_logits = gate_logits + self.freq_gate(c_in)
            if self.use_pitch_aware_gate:
                gate_logits = gate_logits + self.pitch_gain * self._pitch_proxy(c_in)
            gate = torch.sigmoid(gate_logits)

            # compute base alpha map (can be scalar or per-channel map)
            # Fully dynamic alpha: let `alpha_proj` predict the per-channel map and
            # use `alpha_max` as a hard upper bound to avoid explosion. If no
            # `alpha_proj` is provided we fall back to the configured interaction_alpha.
            if self.alpha_proj is not None:
                # completely remove fixed-base interaction_alpha; alpha comes from network
                alpha_map = self.alpha_max * torch.sigmoid(self.alpha_proj(c_in))
            else:
                # degenerate fallback: preserve some dynamic behaviour via the scalar
                alpha_map = self.interaction_alpha

            # Final injection weight = predicted map * frequency/pitch gate
            alpha_total = alpha_map * gate

            # store batch-wise mean for external inspection (debug/analysis)
            try:
                # mean over (C, K, L) -> per-sample mean of alpha_total
                alpha_mean_per_sample = alpha_total.mean(dim=(1, 2, 3))
                # detach and move to cpu for safe access from training loop
                self._last_alpha_total_batch = alpha_mean_per_sample.detach().cpu()
            except Exception:
                pass
            # FiLM-style affine modulation using tanh for stable +/- modulation
            # F_fused = F_backbone + alpha_total * (tanh(gamma) * F_backbone + beta)
            x_4d = x_in + alpha_total * (torch.tanh(gamma) * x_in + beta)
        x_in = x_4d.squeeze(2) if squeezed else x_4d
        return super().forward(x_in, c, t)


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
