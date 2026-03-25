# 🚀 NHFAE E2 Stage 3 快速启动指南

## 方式 A：直接运行（推荐快速测试）

### Step 1: 修改配置
编辑 `stage3_config.json`：
```json
{
  "stage2_checkpoint": "./outputs/stage2/ckpt/best.pt",  // ← Stage 2最优权重路径
  "clean_dir": "/path/to/clean_snr_10_15",             // ← 干净语音目录
  "noisy_dir": "/path/to/noisy_snr_10_15",             // ← 嘈杂语音目录
  "out_dir": "./outputs/stage3",                       // ← 输出目录
  "epochs": 1,                                          // ← Epoch数
  "lr": 5e-6,                                           // ← 学习率
  "device": "cuda"                                      // ← GPU/CPU
}
```

### Step 2: 运行 Stage 3 + Hero Plot（一键启动）
```bash
cd /path/to/the_sound

# 一键启动（推荐）
python ./addse/run_stage3_pipeline.py --config ./addse/stage3_config.json
```

**输出**：
```
./outputs/stage3/
├── ckpt/best.pt              # Stage 3 最优权重
├── wav/                       # 增强波形
├── hero_plot/
│   ├── hero_plot.png         # 核心图表
│   └── hero_metrics.json      # 详细指标
└── logs/stage3_pipeline.log   # 完整日志
```

---

## 方式 B：分步运行（完整控制）

### Step 3A: 仅运行 Stage 3 训练
```bash
python ./addse/phase9_nhfae_e2_stage3.py \
  --checkpoint-path ./outputs/stage2/ckpt/best.pt \
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
```

### Step 3B: 运行 Hero Plot 评估
```bash
python ./addse/phase9_nhfae_e2_hero_plot.py \
  --clean-dir /path/to/clean_snr_10_15 \
  --noisy-dir /path/to/noisy_snr_10_15 \
  --enhanced-dir ./outputs/stage3/wav \
  --out-dir ./outputs/stage3/hero_plot
```

---

## 📋 关键参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `epochs` | 1 | Epoch数（Stage 3仅1个） |
| `lr` | 5e-6 | 超低学习率（亚临界精微调） |
| `lambda_dce` | 1.0 | 数据一致性权重（与清洁信号匹配）|
| `lambda_cfm` | 0.7 | 相位流权重（CFM主导） |
| `lambda_cycle` | 0.2 | 环路权重（乘以0.05后=0.01）|
| `lambda_mrstft` | 0.2 | 频域平滑权重（乘以0.1后=0.02）|

**关键理解**：
```
Stage 3 Loss = 0.05·λ_dce + λ_cfm + 0.05·λ_cycle + 0.1·λ_mrstft
               ↓           ↑      ↓                ↓
            数据一致    相位强   环路极弱      频域极弱
            (弱)       聚焦(强)  (Posterior Mean 锁定)
```

---

## 🎯 Hero Plot 输出解读

### hero_plot.png（5个关键子图）

#### 1️⃣ ΔSDR 分布（左上）
- **物理透明线**：+0.02 dB 黄金线
- **目标**：Mean ≥ +0.01 dB
- **解释**：正值表示增强音质，接近金线证明无损透明性

#### 2️⃣ 相位对齐误差（右上）
- **单位**：弧度 (rad)
- **目标**：Mean < 0.05 rad（1-NFE 就绪条件）
- **解释**：越小越好，反映相位流的线性性

#### 3️⃣ 幅度扰动（左下）
- **指标**：相对扰动 RMSE
- **目标**：Mean < 0.01（Posterior Mean 锁定成功）
- **解释**：证明幅度完全冻结，无实质修改

#### 4️⃣ 散点图：相位 ↔ SDR（右下）
- **关键相关性**：相位精度越高 → ΔSDR 越大
- **线性性验证**：点应沿直线分布

#### 5️⃣ 统计表（底部）
- **汇总所有指标**
- **1-NFE 可扩展性评估**
- **论文核心贡献陈述**

---

## ✅ 成功检查表

运行完成后，检查以下指标：

- [ ] **ΔSDR 均值** ≥ 0.01 dB（超越 +0.005 dB 即可）
- [ ] **相位误差均值** < 0.05 rad（1-NFE 就绪 ✓）
- [ ] **幅度扰动均值** < 0.01（Posterior Mean 锁定 ✓）
- [ ] **无负迁移** PESQ 稳定（查看日志）
- [ ] **Hero Plot 图表** 生成成功

如果全部通过 ✓，即可宣布：
```
🏆 "物理透明黄金线"（ΔSDR ≈ +0.02 dB）已达成！
🚀 1-NFE 推理就绪条件已满足！
📝 论文核心贡献图表已就绪！
```

---

## ⚡ 加速建议

如果数据集很大，可以：

1. **减少样本数**（仅用前 100 个文件测试）
   ```bash
   find /path/to/clean_snr_10_15 -name "*.wav" | head -100 > /tmp/subset.txt
   ```

2. **降低计算精度**（实验阶段）
   - 在脚本中改为 `float16`：`model.half()`
   - GTX 1080/2080 可快速验证

3. **使用更少 Epoch**
   ```bash
   --epochs 1  # 已是最小值
   ```

---

## 🔗 相关资源

- [STAGE3_EXECUTION_GUIDE.md](./STAGE3_EXECUTION_GUIDE.md) — 详细理论和 1-NFE 预计划
- [phase9_nhfae_e2_stage3.py](./phase9_nhfae_e2_stage3.py) — Stage 3 训练脚本
- [phase9_nhfae_e2_hero_plot.py](./phase9_nhfae_e2_hero_plot.py) — Hero Plot 评估框架
- [run_stage3_pipeline.py](./run_stage3_pipeline.py) — 一键启动脚本

---

## 💬 常见问题

### Q: Stage 3 需要多长时间？
**A**: 1 Epoch，取决于数据集大小和GPU：
- 100 样本 + A100：~2-5 分钟
- 1000 样本 + RTX 3090：~15-30 分钟

### Q: 可以在 CPU 上运行吗？
**A**: 可以，但非常慢（10-100 倍）。配置：
```bash
--device cpu
```

### Q: Stage 3 结果不好怎么办？
**A**: 查看执行指南中的"故障排查"部分。通常原因：
1. Stage 2 权重不是最优的
2. SNR 10-15 数据集质量不好
3. 学习率调整（试试 1e-6 或 1e-5）

### Q: 如何对比 Stage 2 vs Stage 3？
**A**: 使用同一个测试集，分别用两个模型推理，然后用 `phase4_compare_metrics.py` 对比。

---

## 🎬 一句话启动

```bash
python ./addse/run_stage3_pipeline.py --config ./addse/stage3_config.json && echo "✓ Stage 3 完成！检查 ./outputs/stage3/hero_plot/hero_plot.png"
```

**现在就开始吧！** 期待看到 ΔSDR 接近物理透明线！🚀
