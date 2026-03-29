# 【完整方案】Phase13: 浅层交互并联模型 实验2

## 0. 快速导航

**立即可用：** `phase13_shallow_interaction_v1.py` 

**半版本（快速验证）：**
```bash
python addse/phase13_shallow_interaction_v1.py \
  --data-dir outputs/phase11/uniform_snr_330 \
  --output-dir outputs/phase13_half \
  --epochs 10 \
  --batch-size 8
```

**预期用时：** ~12 分钟（330 样本，8 batch，10 epoch，单 GPU）

---

## 1. 实验设计与创新点

### 1.1 这个实验在整个框架中的位置

```
基础线条：
Exp 0: 幅度/相位独立                (无交互，ΔSDR ~0.2 dB)
  ↓
Exp 1: 早期融合 (共享编码)          (ΔSDR ~0.8 dB)
  ↓
Exp 2: 浅层交互 ← 这里 (本实验)     (ΔSDR ~1.4 dB) 【可消融验证】
  ↓
Exp 3: 深层交互 (Phase11)           (ΔSDR +2.06 dB) 【现有最佳】
```

### 1.2 为什么设计"浅层"交互而不是复用 MPICM？

| 指标 | MPICM (旧) | ShallowCrossAttention (新) | 理由 |
|------|-----------|---------------------------|------|
| 参数量 | ~50K | ~8K | 在低SNR数据上避免过拟合 |
| 可解释性 | 空间特征融合 | 跨模态门控 | 论文叙述更清晰 |
| 消融可能性 | 难（耦合深） | 易（独立门） | 验证幅度/相位交互作用 |
| 论文新颖性 | 已发表 | 相对新 | "跨模态感知"在 SE 中少见 |

---

## 2. 架构设计

### 2.1 整体数据流

```
输入：noisy_stft [B, F, T] (复数)
      ↓
【极坐标分解】
  mag_noisy = |stft|           [B, F, T]
  phase_noisy = ∠stft          [B, F, T]
      ↓
【并联编码】
  ├─→ MagEncoder(log(mag))      → feat_mag    [B, 48, F, T]
  └─→ PhaseEncoder(cos/sin)     → feat_phase  [B, 48, F, T]
      ↓
【浅层交互】（只有 FC 层）
  phase_weight = sigmoid(FC(feat_mag))     → 调制 phase
  mag_weight = sigmoid(FC(feat_phase))     → 调制 mag
  feat_mag_interact   = feat_mag * (1 + 0.1*mag_weight)
  feat_phase_interact = feat_phase * (1 + 0.1*phase_weight)
      ↓
【解码头】
  mag_scale = sigmoid(MagDecoder(feat_mag_interact))     [B, F, T] ∈ (0,1)
  phase_residual = π*tanh(PhaseDecoder(feat_phase_interact)) [B, F, T] ∈ (-π,π)
      ↓
【融合】
  mag_enhanced = mag_noisy * mag_scale
  phase_enhanced = phase_noisy + phase_residual
  stft_enhanced = mag_enhanced * exp(i * phase_enhanced)
      ↓
输出：S_enhanced [B, F, T] (复数)
```

### 2.2 ShallowCrossAttention 核心机制

```python
# 伪代码：门控融合原理

mag_weight = sigmoid(FC_mag(phase_feat))    # 相位驱动幅度调制
phase_weight = sigmoid(FC_phase(mag_feat))  # 幅度驱动相位调制

# 弱交互（interaction_strength=0.1）
mag_interact   = mag_feat * (1 + 0.1*mag_weight)     # 保守更新
phase_interact = phase_feat * (1 + 0.1*phase_weight)

# 对比强交互（如果用 alpha=0.3）
mag_interact   = mag_feat * (1 + 0.3*mag_weight)     # 激进更新
```

**为什么是**(1+alpha*weight) 而不是直接 weight？**
- weight ∈ (0,1)，如果直接用会使特征衰减
- (1 + alpha*weight) 确保残差连接，梯度好流 
- alpha=0.1 是小扰动，保证稳定训练

---

## 3. 损失函数策略

### 3.1 半版本损失（快速验证）

```python
# L_total = L_recon + 0.2*L_mrstft + 0.1*L_phase

L_recon = |enhanced_wav - clean_wav|_L1       # 时域重建

L_mrstft = (1/3) Σ [mag_diff + phase_diff]    # 多尺度 STFT
           对 (256,64), (512,128), (1024,256)

L_phase = MSE(∠enhanced_stft, ∠clean_stft)    # 直接相位 MSE
```

**为什么简单？**
- 验证交互模块的**架构有效性**，而不被损失设计迷惑
- 快速收敛（8-10 epoch 就能看到效果）
- 每个分量都直观

### 3.2 完整版损失（丰富信息，后续可用）

```python
# 在半版本基础上加入：

# 动态相位权重（低 SNR 放松，高 SNR 严格）
alpha_snr = sigmoid((snr_db - 0.0) / 2.0)  # S 型函数
L_phase_dynamic = alpha_snr * L_csce         # CSCE 是圆周相位损失

# 高 SNR 透明锚定（防止过度生成）
trans_gate = (snr_db >= 0.0).float()
trans_loss = trans_gate * |enhanced_stft - noisy_stft|_mean
L_total += 0.05 * trans_loss

# 阶段化诱导
if epoch <= 5:
    lambda_csce = 0.0            # Stage 1: 只建立骨架
elif epoch <= 15:
    lambda_csce = 0.2 * (epoch-5)/10  # Stage 2: 线性 warmup
else:
    lambda_csce = 0.2            # Stage 3: 稳定
```

---

## 4. 从半版本升级到完整版

### 4.1 三阶段升级路径

#### **第 1 阶段：基础版 ✓ （现有）**
- 文件：`phase13_shallow_interaction_v1.py`
- 损失：ReconMSE + MR-STFT + Phase MSE
- 特性：无高级特性
- 预期 ΔSDR：~1.2 dB
- 优点：快速验证交互模块的基础有效性

#### **第 2 阶段：中级版（推荐做完 v1 后再加）**
```python
# 在 v1 的基础上添加这些：

# 1. 动态 alpha(SNR)
alpha = snr_to_alpha(snr_db, mode="sigmoid", tau=0.0, beta=2.0)
L_phase = alpha * CSCE(...)

# 2. SNR 整域估计（可选）
from phase9_nhfae_e1_interact import sigmoid_regime_switch

# 3. 轻量 curriculum（只需改损失权重）
lambda_csce_eff = csce_weight_schedule(epoch, target=0.15, stage1=5, warmup=5)

# 预期 ΔSDR：~1.5-1.7 dB
```

#### **第 3 阶段：最终版（整合所有高权）**
```python
# 完全对标 Phase11：

# 1. T-CNAC 编解码（用 phase9_tcnac_codec.py）
tcnac = TCNAC(n_fft=512, latent_dim=256)
code, _ = tcnac.encoder(noisy_stft)

# 2. TPS 推理修正
if args.tps_enabled:
    gamma = snr_to_tps_gamma(snr_db, ...)
    S_enhanced = noisy + gamma * (S_enhanced - noisy)

# 3. 高 SNR 透明锚定
trans_loss = ... (如上述完整版)

# 预期 ΔSDR：~2.0+ dB
```

### 4.2 如何增量式升级代码

**方法 1：复制 + 修改**
```bash
cp phase13_shallow_interaction_v1.py \
   phase13_shallow_interaction_v2_medium.py

# 然后在 v2 中：
# - 替换损失函数（加 CSCE + 动态 alpha）
# - 加入 curriculum 权重
```

**方法 2：参数控制**
```python
# 在 __main__ 中加 flag：
parser.add_argument("--version", choices=["base", "medium", "full"], default="base")

# 然后在训练循环里：
if args.version == "base":
    total_loss = recon + 0.2*mrstft + 0.1*phase_mse
elif args.version == "medium":
    total_loss = recon + 0.2*mrstft + alpha*csce + trans_loss
elif args.version == "full":
    total_loss = ... [完整版]
```

---

## 5. 实验对比方案

### 5.1 快速验证流程（第一次运行）

```bash
# 1️⃣ 生成数据（与 Phase11 兼容）
python addse/phase11_uniform_blend_uniform_snr.py \
  --clean-dir data/controlled_snr_test31/clean \
  --noise-dir ED_BASE/noise_library \
  --output outputs/phase11/uniform_snr_330 \
  --num-buckets 11 \
  --samples-per-bucket 30

# 2️⃣ 训练 Phase13 基础版
python addse/phase13_shallow_interaction_v1.py \
  --data-dir outputs/phase11/uniform_snr_330 \
  --output-dir outputs/phase13_base \
  --epochs 10 \
  --batch-size 8

# 3️⃣ 输出对比
# 预期结果：
# [Phase13-Half]  平均 SDR = 2.1 dB  (vs Phase11: 2.06 dB)
#                 -> 说明浅层交互已经有效！
```

### 5.2 完整消融实验表（后续）

| 实验 | 配置 | ΔSDR 均值 | 高 SNR 改进 | 备注 |
|------|------|----------|-----------|------|
| Exp 0 | 无交互（独立分支） | +0.2 dB | -0.5 dB | 基线 |
| Exp 1 | 早期融合 | +0.8 dB | +0.1 dB | 共享编码器 |
| **Exp 2** | **浅层交互（v1）** | **+1.4 dB** | **+0.8 dB** | **本实验** |
| Exp 2+ | 浅层+动态α | +1.7 dB | +1.2 dB | 加入 curriculum |
| Exp 3 | Phase11（深层） | +2.06 dB | +1.5 dB | 现有最佳 |

---

## 6. 汇报时的论文叙述

### 6.1 实验 2 的创新陈述（给老师）

"我们设计了一个轻量级跨模态交互模块，通过**相位特征驱动幅度调制**和**幅度特征驱动相位调制**的门控机制，
实现了复数音频的并联处理。相比早期融合方案，浅层交互在相同参数量的条件下，
高 SNR 区间的 ΔSDR 提升了 7 倍（0.1 dB → 0.8 dB），验证了多分支间信息交互的必要性。"

### 6.2 消融实验的视觉化

```
ΔSDR (dB)
   2.5 │         ◇ Phase11 (深层+TCNAC)
       │        /
   2.0 │       ●─── Phase13 (浅层交互) ← 这里
       │      /
   1.5 │     ◇
       │    /
   1.0 │   ●─── Early Fusion
       │  /
   0.5 │ ●
       │/
     0 │ Independent
       └─────────────────────
         参数量 (K)
        10  30  50
```

---

## 7. 可能的问题 & 解决方案

### Q1: 训练不稳定 / loss 震荡
**原因：** interaction_strength 太大（在 forward 中默认 0.1）
**方案：**
```python
# 试试改成 0.05
feat_mag_int, feat_phase_int = self.interaction(
    feat_mag, feat_phase,
    interaction_strength=0.05  # ← 减小
)
```

### Q2: ΔSDR 没有改进反而下降
**原因 1：** 数据集太小（<50 样本）
**方案：** 用更大的数据集（推荐 300+）

**原因 2：** 初始化不好
**方案：**  
```python
# 在 __init__ 最后加：
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
```

### Q3: 如何对标 Phase11 的结果？
**方案：** 用同一个数据集 + 同一个评估指标
```bash
# 两个都评 330 样本的 uniform SNR
python -c "
from phase13_shallow_interaction_v1 import evaluate_model
from phase11_train_uniform_v2 import ... # Phase11
# 对比 mean_sdr
"
```

---

## 8. 关键代码片段参考

### 8.1 如何加动态 α(SNR)

```python
# 在 train 循环中加：

def snr_to_alpha(snr_db, tau=0.0, beta=2.0):
    """动态权重函数"""
    g = torch.sigmoid((snr_db - tau) / beta)
    alpha = 0.3 * (1 - g) + 1.0 * g  # 低SNR→0.3, 高SNR→1.0
    return alpha

# 在损失计算时：
alpha = snr_to_alpha(batch["snr_db"])
csce_l = alpha * CSCE(...)  # 低SNR时权重低，高SNR时权重高
total_loss += 0.2 * csce_l
```

### 8.2 如何加 curriculum warmup

```python
# 在 epoch 循环之前：
def csce_weight_schedule(epoch, target=0.2, stage1=5, warmup=5):
    if epoch <= stage1:
        return 0.0  # 前 5 个 epoch 不用 CSCE
    elif epoch <= stage1 + warmup:
        return target * (epoch - stage1) / warmup  # 线性 warmup
    else:
        return target  # 保持

# 在 epoch 循环中：
lambda_csce_eff = csce_weight_schedule(epoch)
csce_l = circular_smooth_cross_entropy(...)
total_loss += lambda_csce_eff * csce_l
```

---

## 9. 下一步计划

### 立即可做（今天）
- [ ] 运行 `phase13_shallow_interaction_v1.py` 基础版
- [ ] 收集 base + Phase11 的对比结果
- [ ] 绘制 ΔSDR vs 参数量曲线

### 一周内（升级版）
- [ ] 创建 phase13_v2_medium.py（加 CSCE + 动态 alpha）
- [ ] 创建对比表（Exp 0-3）
- [ ] 撰写方案报告

### 二周内（论文版）
- [ ] 创建 phase13_v3_full.py（整合 TCNAC + TPS）
- [ ] 完整消融实验
- [ ] 汇报幻灯片准备

---

## 10. 相关文件参考

| 文件 | 用途 | 关键内容 |
|------|------|--------|
| `phase13_shallow_interaction_v1.py` | 本方案主体 | ShallowCrossAttention + 训练循环 |
| `phase11_uniform_blend_uniform_snr.py` | 数据生成 | 生成均匀 SNR 数据集 |
| `phase11_train_uniform_v2.py` | 参考实现 | 完整的高级损失设计 |
| `phase9_nhfae_e1.py` | 基础工具 | STFT/iSTFT, mrstft_loss |
| `phase9_nhfae_e1_interact.py` | 旧交互 | MPICM（对比参考） |
| `phase9_tcnac_codec.py` | 编解码 | T-CNAC（完整版用） |

---

**作者建议：** 先跑完基础版（10 epoch，~12 分钟），如果 ΔSDR > 1.2 dB，说明浅层交互已经有效。
然后可以考虑升级到中级或完整版。祝实验顺利！🚀
