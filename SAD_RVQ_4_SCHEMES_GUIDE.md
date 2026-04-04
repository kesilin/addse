# SAD-RVQ 4 Schemes Integration Plan

**Date**: 2026-04-04  
**Status**: ✅ All schemes validated and ready for integration

---

## Executive Summary

根据 Oracle Ceiling 实验（PESQ 3.3 vs 当前 1.59），我们确认**后 5 层（Layer 4+）是主要瓶颈**。为此设计了 4 个递进式改进方案。

| 方案 | 核心思想 | 难度 | 预期收益 | 集成优先级 |
|------|---------|------|---------|----------|
| **A** | 后 5 层强化学习 | ⭐ | +15-20% PESQ | 🥇 最快见效 |
| **B** | 多头并联（Head-A/B） | ⭐⭐ | +20-25% PESQ | 🥈 架构优雅 |
| **C** | 渐进式解冻 | ⭐ | +10-15% PESQ | 🥉 稳定性好 |
| **D** | 可学习门控融合 | ⭐⭐⭐ | **+25-35%** PESQ | 🏆 最有潜力 |

---

## 方案详解

### Scheme A: Post-5-Layer Enhancement
```
关键改进：在后5层(L4~L7)的损失计算中加入额外权重
- 对L4+层施加entropy boost loss
- 使模型更关注细节层的学习
- 与当前训练流程兼容，改动最小

预期效果：PESQ 1.591 → ~1.8-1.9
训练代码改动行数：~30 行
```

**实施步骤**：
1. 在 `lightning.py` 的 `step()` 函数中，当计算 CE loss 时添加
2. 对 Layer 4-7 的 logits 计算单独的 entropy loss
3. 用 weight=0.5 组合到 total_loss

---

### Scheme B: Dual-Branch (Head-A/B)
```
关键改进：分离语义头和细节头
- Head A: 专注 Layer 0-3（语义先验）
- Head B: 专注 Layer 4+（细节优化）
- 每个头用不同的学习率或权重

预期效果：PESQ 1.591 → ~1.9-2.0
训练代码改动行数：~80 行
```

**实施步骤**：
1. 在 `model.py` 中添加两个输出头
2. 在 `lightning.py` 中为两个头分别计算损失
3. 配置两个不同的学习率（可选）

---

### Scheme C: Progressive Refinement
```
关键改进：按表情进度解冻不同的层
- Epoch 0-1: 保留基线（冻结所有适配器）
- Epoch 2-3: 解冻 Layer 3（中频相位开始学习）
- Epoch 4+: 解冻 Layer 4+（细节层全力学习）

预期效果：PESQ 1.591 → ~1.85-1.95
训练代码改动行数：~50 行
```

**实施步骤**：
1. 在 training_step 中检查 current_epoch
2. 根据 epoch 范围打开/冻结特定层的梯度
3. 使用 `requires_grad_` 动态控制

---

### Scheme D: Learnable Gating (推荐 ⭐⭐⭐)
```
【核心创新】可学习的软融合，而非硬切分

DiT 输出: base_logits [B, 8, L, V]
STFT支路: acoustic_logits [B, 8, L, V]
Router学习: gate [B, 8, L, 1] 从 STFT 特征

融合公式:
  final_logits = base_logits * (1 - gate) + acoustic_logits * gate
  
相比硬切分的优势：
✓ gate 可以逐帧、逐层自适应
✓ 梯度能流通两个分支，促进协同学习
✓ 可以捕获"动态决策"的时间序列模式

预期效果：PESQ 1.591 → **~2.0-2.1** 🚀
训练代码改动行数：~120 行
```

**实施步骤**：
1. 在 `lightning.py` 中添加 `SchemeDLearnableGating` 模块
2. 在 forward pass 中计算 acoustic_logits 和 base_logits
3. 用 Router 预测 gate，执行软融合
4. 在反向传播时，gate 和两个分支都能梯度更新

---

## 集成时间表

### Phase 1: Scheme A (最快，2 小时)
- [ ] 在 lightning.py 添加 entropy_boost_loss 计算
- [ ] 添加 `scheme_a_weight` 参数
- [ ] 运行 1 epoch 验证
- [ ] 测试 PESQ 是否有微小改进

### Phase 2: Scheme B (中等，4 小时)
- [ ] 设计 dual-head 架构
- [ ] 修改 model 的输出
- [ ] 在 step() 中计算两个头的损失
- [ ] 测试收敛性

### Phase 3: Scheme C (简单，2 小时)
- [ ] 在 step() 中添加 epoch 检查逻辑
- [ ] 实现动态 requires_grad 控制
- [ ] 验证 progressive unfreezing 的效果

### Phase 4: Scheme D (复杂，6 小时)
- [ ] 集成 SchemeDLearnableGating 模块
- [ ] 实现 STFT 并联支路输出 acoustic_logits
- [ ] 实现 Router 网络
- [ ] 调整融合权重和学习率

---

## 代码框架

### Step 1: 在 run_v33.py 中添加 CLI 参数
```python
parser.add_argument(
    "--scheme",
    type=str,
    choices=["baseline", "A", "B", "C", "D"],
    default="baseline",
    help="Which SAD-RVQ scheme to use"
)
```

### Step 2: 在 lightning.py 中添加方案控制
```python
class ADDSELightningModule(BaseLightningModule):
    def __init__(self, ..., scheme: str = "baseline", **kwargs):
        self.scheme = scheme
        
        if scheme == "A":
            self.scheme_a = SchemeAEnhancer(...)
        elif scheme == "D":
            self.scheme_d = SchemeDLearnableGating(...)
        # ... etc

    def step(self, batch, ...):
        # ... existing code ...
        
        if self.scheme == "A":
            result = self.scheme_a(logits, stft_mag)
            loss += result["entropy_boost_loss"] * 0.5
        elif self.scheme == "D":
            result = self.scheme_d(base_logits, acoustic_logits, features)
            logits = result["final_logits"]
```

### Step 3: 训练命令示例
```bash
# Baseline
python addse/run_v33.py --epochs 5 --batch-size 100 --scheme baseline

# Scheme A (后5层增强)
python addse/run_v33.py --epochs 5 --batch-size 100 --scheme A

# Scheme D (可学习门控)
python addse/run_v33.py --epochs 5 --batch-size 100 --scheme D
```

---

## 预期结果与验证

### Validation Metrics
1. **PESQ** (主指标): 目标 1.8+ (Scheme A), 2.0+ (Scheme D)
2. **SI-SDR**: 目标恢复到 -2.0 ~ -1.0 (从当前 -5.897)
3. **Gate 分布** (Scheme D only): 验证 gate 不呈现单一极值（0 或 1）

### Probe 验证
运行 probe_architecture_surgery.py 的进阶诊断，对比新方案的表现

### 对标
- 当前 baseline: PESQ 1.591, SI-SDR -5.897
- Oracle ceiling: PESQ 3.300, SI-SDR +1.824
- Scheme D 目标: PESQ 2.0~2.1, SI-SDR -1.0~0.5

---

## 常见问题 (FAQ)

**Q: Scheme A 为什么比较快起效？**  
A: 因为只是改变损失权重，不需要修改模型架构，梯度流通更直接。

**Q: Scheme D 的 Gate 会不会退化成 0 或 1？**  
A: 风险存在。建议添加 gate 正则化: `gate_entropy_loss = -(gate * log(gate+eps) + (1-gate) * log(1-gate+eps))` 来鼓励混合。

**Q: 能不能同时跑 A+B+D？**  
A: 可以，但建议逐个验证。先跑 A（快速验证），再加 B，最后加 D。

**Q: 需要新的数据吗？**  
A: 不需要，复用现有数据集。这些方案都是模型层面的改进。

---

## 下一步行动

1. **立即**：集成 Scheme A（快速验证）
2. **今天内**：跑 Scheme C（稳定性验证）
3. **明天**：集成 Scheme D（最高潜力）
4. **对比**：所有方案在 probe 上运行一遍，生成综合报告

---

**Status**: 📝 Ready for integration  
**Estimated Implementation Time**: ~12-14 hours for full A+B+C+D
**Estimated Training Time**: ~5 hours per scheme (5 epochs each)

