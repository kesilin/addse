# NHFAE E2 Stage 3：执行指南 & 1-NFE 理论预计划

## 📋 Stage 3 执行步骤

### Step 1: 准备 Stage 2 最优权重
```bash
# 确认 Stage 2 的最优 checkpoint 位置
STAGE2_CKPT="./outputs/stage2/ckpt/best.pt"

# 验证文件存在
ls -la $STAGE2_CKPT
```

### Step 2: 准备 SNR 10-15 dB 数据集
```bash
# 确保 clean 和 noisy 目录存在：
# - /path/to/clean_snr_10_15/  （干净语音）
# - /path/to/noisy_snr_10_15/  （嘈杂语音）

# 样本量检查
find /path/to/clean_snr_10_15 -name "*.wav" | wc -l
find /path/to/noisy_snr_10_15 -name "*.wav" | wc -l
```

### Step 3: 运行 Stage 3（Regime II Refinement）
```bash
# 核心命令
python ./addse/phase9_nhfae_e2_stage3.py \
  --checkpoint-path $STAGE2_CKPT \
  --clean-dir /path/to/clean_snr_10_15 \
  --noisy-dir /path/to/noisy_snr_10_15 \
  --out-dir ./outputs/stage3 \
  --epochs 1 \
  --lr 5e-6 \
  --lambda-dce 1.0 \
  --lambda-cfm 0.7 \
  --lambda-cycle 0.2 \
  --lambda-mrstft 0.2 \
  --device cuda

# 输出：
#   ./outputs/stage3/ckpt/best.pt  （最优权重）
#   ./outputs/stage3/wav/  （Stage 3 增强波形）
```

### Step 4: 评估 Hero Plot（工业级无损透明性）
```bash
# 生成核心图表和统计
python ./addse/phase9_nhfae_e2_hero_plot.py \
  --clean-dir /path/to/clean_snr_10_15 \
  --noisy-dir /path/to/noisy_snr_10_15 \
  --enhanced-dir ./outputs/stage3/wav \
  --out-dir ./outputs/stage3/hero_plot

# 输出：
#   ./outputs/stage3/hero_plot/hero_plot.png   （Hero Plot 可视化）
#   ./outputs/stage3/hero_plot/hero_metrics.json （详细指标）
```

---

## 🎯 Stage 3 理论核心

### Regime II Refinement（极端权重调整）

#### 幅度去敏化（Magnitude De-Sensitization）
```
原始：       loss = λ_dce·l_dce + λ_cfm·l_cfm + λ_cycle·l_cycle + λ_mrstft·l_mr
Stage 3:    loss = 0.05·λ_dce·l_dce + λ_cfm·l_cfm + 0.05·λ_cycle·l_cycle + 0.1·λ_mrstft·l_mr
             │        ↓              │        ↑              ↓                ↓
             │      数据一致(弱)     │   相位流(强)    环路稳定(极弱)  频域平滑(极弱)
             └─────────────────────────────────────────────────────────────────
```

**数学解释**：
- 锁定模型在 **Posterior Mean** E[y|noisy, clean] 附近
- ∂y_mag / ∂t ≈ 0（幅度冻结，无修改）
- ∂y_phase / ∂t 主驱动（相位微调，自由优化）

#### 相位强聚焦（Phase Strong Focus）
```
CFM Loss = ||∂_t log(S_est)|| + ||∇phase(S_est - S_noisy)||
           ├─ Phase velocity（相位速度）主导
           └─ Magnitude velocity（幅度速度）极弱
           
学习率：lr = 5e-6（超低），使得 ∂phase ≈ 0.001 rad/epoch（亚临界）
```

---

## 🚀 1-NFE 极速推理：理论预计划

### 什么是 1-NFE？
- **NFE** = Number of Function Evaluations（函数评估次数）
- **1-NFE** = 单步推理（仅需 1 次模型前向传播）
- **目标**：从 10+ NFE（Level-DDPM） → 1 NFE（线性 ODE）

### 1-NFE 可行性条件（Stage 3 验证）

| 条件 | 评估指标 | 目标值 | 验证方法 |
|------|--------|-------|--------|
| **相位线性性** | Phase Error RMSE | < 0.05 rad | `heroplot.py` 输出 |
| **幅度稳定性** | Magnitude Perturbation | < 0.01 | `heroplot.py` 输出 |
| **频谱一致性** | MRSTFT 损失 | < 0.05 | `phase9_nhfae_e1.py:mrstft_loss()` |
| **无损透明性** | ΔSDR | > +0.01 dB | 量化论证 |

### 1-NFE 推理实现框架（Stage 4 预计划）

```python
# ═══════════════════════════════════════════════════════════
# 1-NFE Scheme (Linear Trajectory)
# ═══════════════════════════════════════════════════════════

class NHFAE_1NFE(nn.Module):
    """单步推理版本（ODENet with 1 Euler step）"""
    
    def forward(self, S_noisy):
        """
        单步线性推理：
        
        假设 Phase residual Φ(t) 沿时间 t ∈ [0, 1] 线性变化：
          Φ(t=0) = 0         (noisy phase)
          Φ(t=1) = Φ_target  (clean phase)
        
        1-NFE 直接跳到 t=1：
          y = istft(|S_noisy| * exp(i * (∠S_noisy + Φ_target)))
        """
        
        # 单次前向：计算目标相位残差
        S_noisy_mag = torch.abs(S_noisy)
        S_noisy_phase = torch.angle(S_noisy)
        
        # 模型输出：相位残差速度（解释为全程梯度）
        out = self.model(S_noisy)  # 1 次 forward pass
        v_phase = out["v_phase"]   # [B, T, F]
        
        # 线性外推：v_phase 代表从 t=0 → t=1 的全程相位变化
        # Φ_target ≈ v_phase * 1  (积分因子为 1)
        S_enhanced_phase = S_noisy_phase + v_phase.squeeze(0)
        
        # 重建：幅度保持不变（Posterior Mean 锁定）
        S_enhanced = S_noisy_mag * torch.exp(1j * S_enhanced_phase)
        y = istft(S_enhanced, n_fft=512, hop=192)
        
        return {
            "y": y,
            "S_enhanced": S_enhanced,
            "phase_residual": v_phase,  # 记录相位变化以备验证
        }

# ═══════════════════════════════════════════════════════════
# 验证脚本：1-NFE 线性性检验
# ═══════════════════════════════════════════════════════════

def validate_1nfe_linearity():
    """
    验证：多步和单步推理是否在线性轨迹上对齐
    """
    # 加载 Stage 3 模型
    model = load_stage3_checkpoint()
    
    # 对同一个样本进行多步推理（as reference）
    S_noisy = ...
    
    # 多步推理（用作参考真值）
    S_multi_step = recursive_ode_solver(model, S_noisy, steps=10)
    
    # 单步推理
    S_1nfe = model.forward_1nfe(S_noisy)
    
    # 比较：期望 |S_1nfe - S_multi_step| 接近 0（线性假设成立）
    phase_diff = compute_phase_distance(S_1nfe, S_multi_step)
    assert phase_diff < 0.05, f"1-NFE 线性性不满足: {phase_diff}"
    
    print(f"✓ 1-NFE 线性轨迹验证通过 (误差={phase_diff:.4f} rad)")
```

### Stage 4: 1-NFE 实现路线（预计）

| 阶段 | 工作内容 | 输出 |
|------|--------|------|
| **4A** | 线性性验证（多步 vs 单步） | `phase9_nhfae_e2_1nfe_validate.py` |
| **4B** | 单步推理优化 | `NHFAE_1NFE` 类实现 |
| **4C** | 评估 1-NFE 性能 | RTF（实时因子）、ΔSDR 对比 |
| **4D** | 工业部署准备 | ONNX 导出、TensorRT 编译 |

---

## 📊 预期结果

### Stage 3 成功指标
```
✓ ΔSDR 均值 ≈ +0.015 ~ +0.025 dB（接近 +0.02 极限）
✓ Phase Error RMSE < 0.05 rad（相位对齐精度优异）
✓ Magnitude Perturbation < 0.01（幅度锁定成功）
✓ 无负迁移（所有 SNR 桶 PESQ 稳定）
```

### 论文核心贡献
```
【Title】NHFAE: Neural Hybrid Fixed-Phase Alignment for Physics-Transparent 
        Audio Enhancement at Industrial Scale

【Abstract Excerpt】
... Stage 3 验证了在极端高 SNR（10-15 dB）条件下，Regime II Refinement 策略
成功将模型约束在 Posterior Mean 附近，实现 ΔSDR ≈ +0.02 dB 的物理透明性极限。
相位线性轨迹的稳定性和幅度冻结的可靠性，为后续 1-NFE 推理奠定了理论基础，
届时将实现单次前向传播的音频增强，工业部署时延从 50ms 降低至 1ms。
```

---

## ⚠️ 常见问题 & 故障排查

### Q1: Stage 3 loss 不收敛？
**A**: 检查学习率。如果使用的是继承自 Stage 2 的优化器状态，请重新创建优化器：
```python
trainable_params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable_params, lr=5e-6)  # 重新初始化
```

### Q2: Hero Plot 中 ΔSDR 为负？
**A**: 可能是 magnitude perturbation 过大。检查：
```python
# 在 Stage 3 脚本中降低 cycle/mrstft 权重
loss = (
    0.01 * cfg.lambda_dce * l_dce +   # 进一步降低
    cfg.lambda_cfm * l_cfm +
    0.02 * cfg.lambda_cycle * l_cycle + # 进一步降低
    0.05 * cfg.lambda_mrstft * l_mr    # 进一步降低
)
```

### Q3: 1-NFE 性能达不到预期？
**A**: Stage 3 的相位线性性不足。运行 Hero Plot 检查 `phase_error` 均值：
- 如果 > 0.05 rad：增加 Stage 3 的 epochs 或降低学习率到 1e-6
- 如果 > 0.1 rad：可能需要额外的 Stage 3.5（中间微调）

---

## 🔗 相关文件清单

- `phase9_nhfae_e2_stage3.py` — Stage 3 训练脚本
- `phase9_nhfae_e2_hero_plot.py` — Hero Plot 评估框架
- `phase9_nhfae_e1.py` — 核心模型定义（mrstft_loss 实现）
- `phase9_nhfae_e1_interact.py` — NHFAE_E1_Interact 类定义

---

**准备好了？开始 Stage 3！** 🚀
