import math
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
    """V3.5 QRC (Quantization Residual Compensation) 顶会级架构
    采用 WaveNet 风格的空洞卷积，指数级扩大感受野，精准修复离散量化导致的高频与相位丢失。
    """
    def __init__(self, **kwargs) -> None:
        # 过滤 kwargs 传递给父类
        base_kwargs = {k: v for k, v in kwargs.items() if k in ["input_channels", "output_channels", "num_codebooks", "hidden_dim", "num_layers", "num_heads", "max_seq_len", "conditional", "time_independent"]}
        super().__init__(**base_kwargs)
        
        in_channels = kwargs.get("input_channels", 1024)
        hidden = kwargs.get("adapter_hidden", 256)
        num_codebooks = kwargs.get("num_codebooks", 4)
        
        # 感受野 (Receptive Field) 指数级扩大的残差网络
        layers = []
        layers.append(nn.Conv1d(in_channels, hidden, 3, padding=1))
        layers.append(nn.GELU())
        # 使用空洞卷积 (Dilation) 捕捉长距离相位和上下文信息
        for dilation in [1, 2, 4, 8, 16, 32]:
            layers.append(nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation))
            layers.append(nn.GELU())
        layers.append(nn.Conv1d(hidden, in_channels, 1))
        self.refiner = nn.Sequential(*layers)
        
        # 交互参数：让连续残差动态辅助离散猜词
        interaction_alpha = float(kwargs.get("interaction_alpha", 0.05))
        self.interaction_alpha = nn.Parameter(torch.tensor([interaction_alpha], dtype=torch.float32))

        # 并联交互开关
        self.fusion_mode = kwargs.get("fusion_mode", "add")
        self.use_adain_align = bool(kwargs.get("use_adain_align", False))
        self.use_freq_dynamic_gate = bool(kwargs.get("use_freq_dynamic_gate", False))
        self.use_pitch_aware_gate = bool(kwargs.get("use_pitch_aware_gate", False))
        self.dynamic_alpha = bool(kwargs.get("dynamic_alpha", False))
        self.alpha_max = float(kwargs.get("alpha_max", 0.08))
        self.pitch_gain = float(kwargs.get("pitch_gain", 0.1))
        self.use_interaction_alignment_gate = bool(kwargs.get("use_interaction_alignment_gate", True))
        self.use_interaction_confidence_gate = bool(kwargs.get("use_interaction_confidence_gate", True))
        self.use_band_quality_gate = bool(kwargs.get("use_band_quality_gate", True))
        self.interaction_gate_floor = float(kwargs.get("interaction_gate_floor", 0.2))

        # IMPORTANT: 张量布局是 (B, C, K, L)，对 codebook 维 K 的卷积核必须是 (k, 1)，避免误作用到时间轴。
        gate_kernel_size = int(kwargs.get("gate_kernel_size", 3))
        gate_kernel_size = max(1, gate_kernel_size)
        if gate_kernel_size % 2 == 0:
            gate_kernel_size += 1
        self.freq_gate = nn.Conv2d(1, 1, kernel_size=(gate_kernel_size, 1), padding=(gate_kernel_size // 2, 0))

        self.pitch_gate = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=max(1, min(in_channels, 64))),
            nn.GELU(),
            nn.Conv1d(in_channels, 1, kernel_size=1),
        )

        self.alpha_proj = nn.Sequential(
            nn.Conv1d(2 * in_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

        quality_hidden = int(kwargs.get("quality_head_hidden", max(64, hidden // 2)))
        self.quality_head = nn.Sequential(
            nn.Conv1d(in_channels, quality_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(quality_hidden, 1, kernel_size=1),
        )

        # 将连续残差先投影到离散 logits 语义空间，避免“不同语义空间直接相加”导致交互失真。
        self.interaction_proj = nn.Conv1d(in_channels, self.output_channels, kernel_size=1)

        if self.fusion_mode == "film":
            self.parallel_proj = nn.Conv1d(in_channels, 2 * self.output_channels, kernel_size=1)
        else:
            self.parallel_proj = None
        
        self._reset_parallel_parameters()

    def _reset_parallel_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.refiner[-1].weight)
            nn.init.zeros_(self.refiner[-1].bias)
            nn.init.zeros_(self.freq_gate.weight)
            nn.init.zeros_(self.freq_gate.bias)
            nn.init.zeros_(self.interaction_proj.weight)
            nn.init.zeros_(self.interaction_proj.bias)
            if self.parallel_proj is not None:
                nn.init.zeros_(self.parallel_proj.weight)
                nn.init.zeros_(self.parallel_proj.bias)
            nn.init.zeros_(self.quality_head[-1].weight)
            nn.init.zeros_(self.quality_head[-1].bias)

    @staticmethod
    def _adain_align(source: Tensor, target: Tensor, eps: float = 1e-5) -> Tensor:
        source_mean = source.mean(dim=-1, keepdim=True)
        source_std = source.std(dim=-1, keepdim=True).clamp_min(eps)
        target_mean = target.mean(dim=-1, keepdim=True)
        target_std = target.std(dim=-1, keepdim=True).clamp_min(eps)
        return (source - source_mean) / source_std * target_std + target_mean

    def forward(self, x, c, t, c_cont=None):
        # 1. 主干离散网络预测语义 (猜词)
        z_base = super().forward(x, c, t) 
        
        if c_cont is not None:
            if c_cont.ndim == 4:
                c_cont = c_cont.sum(dim=2) if c_cont.shape[2] > 1 else c_cont.squeeze(2)
            
            # 2. 连续支路预测高频量化残差！
            residual_pred = self.refiner(c_cont) # (B, C, L)
            quality_map = torch.sigmoid(self.quality_head(residual_pred))

            base_cont = z_base.mean(dim=2)
            residual_logit = self.interaction_proj(residual_pred)

            if self.use_adain_align:
                residual_logit = self._adain_align(residual_logit, base_cont)

            residual_4d = residual_logit.unsqueeze(2).expand_as(z_base)

            if self.use_freq_dynamic_gate:
                gate_in = z_base.detach().mean(dim=1, keepdim=True)
                freq_gate = torch.sigmoid(self.freq_gate(gate_in))
                residual_4d = residual_4d * freq_gate

            if self.use_pitch_aware_gate:
                pitch_like = torch.sigmoid(self.pitch_gate(c_cont)).unsqueeze(2)
                residual_4d = residual_4d * (1.0 + self.pitch_gain * pitch_like)

            if self.dynamic_alpha:
                alpha_feat = torch.cat([c_cont, residual_pred], dim=1)
                alpha_t = torch.sigmoid(self.alpha_proj(alpha_feat)).unsqueeze(2)
                alpha_t = alpha_t * self.alpha_max
                alpha = alpha_t
            else:
                alpha = self.interaction_alpha

            interaction_gate = 1.0
            if self.use_interaction_alignment_gate:
                base_norm = F.normalize(base_cont.detach(), dim=1)
                residual_norm = F.normalize(residual_logit, dim=1)
                cos_sim = (base_norm * residual_norm).sum(dim=1, keepdim=True)
                align_gate = ((cos_sim + 1.0) * 0.5).clamp(min=self.interaction_gate_floor, max=1.0).unsqueeze(2)
                interaction_gate = interaction_gate * align_gate

            if self.use_interaction_confidence_gate:
                probs = z_base.detach().softmax(dim=1)
                entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
                entropy = entropy / math.log(max(2, z_base.shape[1]))
                conf_gate = (1.0 - entropy).clamp(min=self.interaction_gate_floor, max=1.0)
                interaction_gate = interaction_gate * conf_gate

            if self.use_band_quality_gate:
                band_energy = z_base.detach().pow(2).mean(dim=1)  # (B, K, L)
                band_ref = band_energy.mean(dim=1, keepdim=True).clamp_min(1e-6)
                band_ratio = (band_energy / band_ref).clamp(min=0.25, max=4.0)
                band_gate = torch.sigmoid(2.0 * (band_ratio - 1.0)).unsqueeze(1)
                quality_gate = quality_map.unsqueeze(2)
                local_quality_gate = (band_gate * quality_gate).clamp(min=self.interaction_gate_floor, max=1.0)
                interaction_gate = interaction_gate * local_quality_gate

            residual_4d = residual_4d * interaction_gate
            
            # 3. 平滑交互：连续残差引导离散 logits
            if self.fusion_mode == "film" and self.parallel_proj is not None:
                gamma, beta = self.parallel_proj(residual_pred).chunk(2, dim=1)
                gamma = torch.tanh(gamma).unsqueeze(2)
                beta = beta.unsqueeze(2)
                logits = z_base * (1.0 + alpha * gamma) + alpha * beta
            else:
                logits = z_base + alpha * residual_4d
            
            return logits, residual_pred, quality_map
        else:
            return z_base, None, None

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