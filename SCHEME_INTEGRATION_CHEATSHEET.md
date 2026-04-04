# SAD-RVQ Schemes 代码集成速查表

## 1️⃣ 修改 run_v33.py - 添加 CLI 参数

```python
# 在 main() 函数的 argparse 部分添加

parser.add_argument(
    "--sad-rvq-scheme",
    type=str,
    choices=["baseline", "a", "b", "c", "d"],
    default="baseline",
    help="SAD-RVQ improvement scheme (A: L4+ boost, D: learnable gating)"
)

parser.add_argument(
    "--sad-rvq-scheme-a-weight",
    type=float,
    default=0.5,
    help="Scheme A entropy boost weight"
)

parser.add_argument(
    "--sad-rvq-scheme-d-enabled",
    action="store_true",
    help="Enable Scheme D learnable gating"
)

# 在创建 LightningModule 时传递参数
lightning_module = ADDSELightningModule(
    ...,
    sad_rvq_scheme=args.sad_rvq_scheme,
    sad_rvq_scheme_a_weight=args.sad_rvq_scheme_a_weight,
    sad_rvq_scheme_d_enabled=args.sad_rvq_scheme_d_enabled,
    ...
)
```

---

## 2️⃣ Scheme A 集成 - 已完成 ✅

位置: `addse/addse/lightning.py` 的 `step()` 方法中 CE loss 后

```python
# Scheme A: Post-5-Layer Enhancement
if self.sad_rvq_scheme == "a":
    if K > 3:
        probs = F.softmax(log_p, dim=-1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)
        entropy_layers_4plus = entropy[3:, :, :]  # Layer 4 onwards
        if entropy_layers_4plus.numel() > 0:
            entropy_boost_loss = -entropy_layers_4plus.mean()
            total_loss = total_loss + self.sad_rvq_scheme_a_weight * entropy_boost_loss
            self.log(f"{stage}/scheme_a_entropy_boost", entropy_boost_loss, prog_bar=True, sync_dist=True)
```

---

## 3️⃣ Scheme D 集成 - 需要完成 (Router 已定义)

### Step 1: 在 step() 中计算 acoustic_logits

```python
# 在主模型前向传播后，计算 acoustic_logits
# 伪代码 - 实际需要根据你的 STFT 支路实现

if self.sad_rvq_scheme == "d":
    # 假设有 self.stft_branch(x_q) → acoustic_probs
    acoustic_probs = self.stft_branch(x_q)  # [B, K, L, V]
    acoustic_logits = torch.log(acoustic_probs.clamp(min=1e-8))
    
    # 使用 Router 计算 gate
    acoustic_features = ... # 从 STFT 支路提取 [B, K, L, F]
    gate = self.scheme_d_router(acoustic_features)  # [B, K, L, 1]
    
    # 软融合
    probs_base = F.softmax(log_p, dim=-1)
    probs_acoustic = F.softmax(acoustic_logits, dim=-1)
    probs_fused = probs_base * (1 - gate) + probs_acoustic * gate
    
    # 使用融合的 logits 计算 CE loss
    log_p_fused = torch.log(probs_fused.clamp(min=1e-8))
    ce_loss_fused = F.cross_entropy(log_p_fused.reshape(-1, V), y_tok.reshape(-1), reduction="none")
    ce_loss = (ce_loss_fused.reshape(B, K, L) * mask).sum() / (mask.sum() + 1e-8)
    
    # Gate 正则化（防止退化）
    gate_entropy = -(gate * torch.log(gate + 1e-8) + (1-gate) * torch.log(1-gate + 1e-8)).mean()
    total_loss = ce_loss + self.sad_rvq_scheme_d_gate_entropy_weight * (-gate_entropy)
```

### Step 2: 在 __init__ 中创建 Router

```python
# 在 ADDSELightningModule.__init__ 的初始化部分

if self.sad_rvq_scheme == "d" or self.sad_rvq_scheme_d_enabled:
    from addse.lightning import SchemeDLearnableGatingRouter
    self.scheme_d_router = SchemeDLearnableGatingRouter(
        num_codebooks=8,  # 根据你的设置调整
        feature_dim=256,  # STFT 特征维度
    )
```

---

## 4️⃣ Scheme C 集成 - 简单

```python
# 在 step() 方法中，CE loss 计算前

if self.sad_rvq_scheme == "c":
    current_epoch = self.current_epoch
    
    # 根据 epoch 冻结/解冻特定层
    if current_epoch < 2:
        # Epoch 0-1: baseline (冻结所有层)
        for name, p in self.model.named_parameters():
            if "adapter" in name:
                p.requires_grad = False
    elif current_epoch < 4:
        # Epoch 2-3: 解冻 Layer 3
        for name, p in self.model.named_parameters():
            if "adapter" in name and ("layer_3" in name or "mid" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        # Epoch 4+: 解冻 Layer 4+
        for name, p in self.model.named_parameters():
            if "adapter" in name and ("layer_4" in name or "layer_5" in name or "high" in name):
                p.requires_grad = True
```

---

## 5️⃣ Scheme B 集成 - 中等

```python
# 需要在 model 中添加两个输出头

class ADDSERQDiT_DualHead(ADDSERQDiT):
    def __init__(self, ...):
        super().__init__(...)
        # Head-A: Semantic layers 0-3
        self.head_a_proj = nn.Linear(vocab_size, vocab_size)
        # Head-B: Detail layers 4+
        self.head_b_proj = nn.Linear(vocab_size, vocab_size)
    
    def forward_with_heads(self, y_q, x_q, x_cont=None):
        base_logits = self.forward(y_q, x_q, x_cont)  # [B, K, L, V]
        
        logits_a = base_logits[:, :3, :, :]
        logits_b = base_logits[:, 3:, :, :]
        
        logits_a_proj = self.head_a_proj(logits_a)
        logits_b_proj = self.head_b_proj(logits_b)
        
        return torch.cat([logits_a_proj, logits_b_proj], dim=1)

# 在 lightning.py 中使用
if self.sad_rvq_scheme == "b":
    log_p_b, _ = self.model.forward_with_heads(y_lambda_q, x_q, x_cont=x_lat)
    # ... 为 Head-A 和 Head-B 分别计算 CE loss，可用不同权重
```

---

## 6️⃣ 运行命令示例

### Quick Test (Scheme A, 2 epochs)
```bash
cd d:/Users/KSL/PycharmProjects/the_sound
python addse/run_v33.py \
  --epochs 2 \
  --batch-size 100 \
  --sad-rvq-scheme a \
  --sad-rvq-scheme-a-weight 0.5
```

### Full Training (Scheme D, 5 epochs)
```bash
python addse/run_v33.py \
  --epochs 5 \
  --batch-size 100 \
  --sad-rvq-scheme d \
  --sad-rvq-scheme-d-enabled
```

### Comparison Run
```bash
python addse/run_schemes_comparison.py \
  --schemes baseline,a,d \
  --epochs 3
```

---

## 7️⃣ 监控指标

### Scheme A 应该看到
```
step/scheme_a_entropy_boost: 逐渐降低（negative entropy 增加）
val/pesq: 应该从 1.591 → 1.8+ 逐步提升
val/si_sdr: 应该从 -5.897 → -3.0 逐步改善
```

### Scheme D 应该看到
```
gate_weights: 分布应在 [0.3, 0.7] 之间（不应是 0 或 1）
val/pesq: 应该从 1.591 → 2.0+ 
val/si_sdr: 应该从 -5.897 → -1.0 ~ 0.0
gate_entropy: 应该保持正值（防止 gate 退化）
```

---

## 8️⃣ 调试技巧

### 问题 1: Scheme A 没有改进
- 检查 entropy_boost_loss 是否计算正确
- 尝试调整权重：0.1 → 1.0
- 查看 entropy 的初值：应该在 log(V) 附近

### 问题 2: Scheme D 的 gate 总是 0.5
- 检查 Router 网络是否正确初始化
- 添加 gate 的梯度监控：`print(scheme_d_router.gate_net[0].weight.grad)`
- 确保 acoustic_features 不是 NaN

### 问题 3: SI-SDR 恶化
- 可能是 acoustic_logits 质量差
- 降低其权重：`gate_target_weight = 0.3` 而非 0.5
- 检查 acoustic_features 的方差

---

## 9️⃣ 性能优化

### 内存节省
```python
# 如果 OOM，考虑不同时计算两个 logits
if self.sad_rvq_scheme == "d":
    # 分离计算 base_logits 和 acoustic_logits
    with torch.no_grad():
        acoustic_logits = self.stft_branch(x_q)
    # 只更新 base_logits
    base_logits = self.model(...)
```

### 加速
```python
# 使用低精度计算 gate
gate = self.scheme_d_router(acoustic_features.half()).float()
```

---

## 🔟 最终检查清单

- [ ] 修改 run_v33.py 添加 CLI 参数
- [ ] 在 __init__ 中初始化 Scheme 对应的模块
- [ ] 在 step() 中添加 Scheme 逻辑
- [ ] 快速测试 Scheme A (1 epoch)
- [ ] 验证 PESQ/SI-SDR 趋势
- [ ] 对比 A vs D
- [ ] 选定最优方案
- [ ] 运行最终训练

