# 噪声数据源配置指南

## 目录结构

已为你创建了两个独立的噪声数据源：

```
data/chunks/
├── edbase_noise_original/      # 原有5个噪声文件 (LitData优化格式)
│   ├── chunk-0-0.bin
│   ├── chunk-1-0.bin
│   └── index.json
│
├── musan_noise/                # 新生成的36个合成噪声 (132MB, WAV格式)
│   ├── noise_white_00.wav
│   ├── noise_pink_00.wav
│   ├── noise_brown_00.wav
│   ├── noise_speech_like_00.wav
│   ├── noise_machine_00.wav
│   ├── noise_traffic_00.wav
│   └── ... (共36个文件，涵盖6种噪声类型)
│
├── musan_noise_raw/            # 原始生成文件备份
```

## 配置文件

### 1. 使用原有噪声源 (推荐用于快速测试)
```bash
python -m addse.app train configs/addse-s-original-ft.yaml --init-ckpt logs/addse-edbase-quick/checkpoints/addse-s.ckpt --overwrite
```
- **噪声源**: `edbase_noise_original/` (5个文件)
- **格式**: LitData优化格式（加载快）
- **用时**: ~6分钟 (10 epochs × 120 batches)

### 2. 使用新生成的MUSAN合成噪声 (推荐用于扩展)
```bash
python -m addse.app train configs/addse-s-musan-ft.yaml --init-ckpt logs/addse-edbase-quick/checkpoints/addse-s.ckpt --overwrite
```
- **噪声源**: `musan_noise/` (36个合成噪声文件)
- **格式**: WAV原始文件 (首次加载需要处理)
- **性质**: 6种噪声类型 (White, Pink, Brown, Speech-like, Machine, Traffic)
- **用时**: ~8-10分钟 (10 epochs，WAV处理略慢)

## 下一步：扩展训练

### 使用更长的训练周期
```bash
# 50 epochs (~30分钟)
python -m addse.app train configs/addse-s-musan-ft.yaml \
  --init-ckpt logs/addse-edbase-quick/checkpoints/addse-s.ckpt \
  --overwrite \
  trainer.max_epochs=50

# 100 epochs (~60分钟)
python -m addse.app train configs/addse-s-musan-ft.yaml \
  --init-ckpt logs/addse-edbase-quick/checkpoints/addse-s.ckpt \
  --overwrite \
  trainer.max_epochs=100 \
  trainer.limit_train_batches=150
```

### 评估结果
```bash
# 评估最佳模型
python -m addse.app eval configs/addse-s-musan-ft.yaml \
  logs/addse-s-musan-ft/checkpoints/epoch=XX-val_loss=X.XX.ckpt \
  --device cuda --output-db eval_musan_noisy.db --overwrite --num-examples 30 \
  lm.num_steps=64 --noisy
```

## 评测规程（强制执行）

从现在开始，不再只看总平均分，必须做 SNR 分桶评测。

### 分桶区间

- `[-5, 0]`
- `[0, 5]`
- `[5, 10]`
- `[10, 15]`

### 汇报口径

- 每个桶都要报告 `PESQ / ESTOI / SDR`。
- 每个桶都要报告 `ADDSE(or ADDSE+PGUSE) - noisy` 的增量。
- 若出现 `PESQ 升, SDR 大幅降`，需要在结论中明确标注为质量-保真 trade-off。

### 最低通过线（当前项目）

- `PESQ`：4 个桶中至少 3 个桶为正增量。
- `SDR`：中高 SNR（`[5,10]` 与 `[10,15]`）不应出现大幅负增量。
- `ESTOI`：高 SNR 桶不应持续回落。

## 技术细节

### 合成噪声类型

| 噪声类型 | 描述 | 应用场景 |
|---------|------|--------|
| **White** | 均匀分布，所有频率能量相等 | 评估宽带抑制 |
| **Pink** | 低频更强，频率越低能量越大 | 环境噪声模拟 |
| **Brown** | 更低频主导（$f^{-2}$滚降） | 远处交通/风声 |
| **Speech-like** | 100-500Hz调制，模拟背景语音 | 鸡尾酒会问题 |
| **Machine** | 特定频率基音（100/250/400Hz） | 机械设备噪声 |
| **Traffic** | 混合70/150/300Hz，离散脉冲 | 城市设备噪音 |

### 数据规模对比

| 源 | 文件数 | 时长 | 大小 | 加载格式 |
|----|-------|------|-----|--------|
| edbase_original | 4 chunks | ~2h | ~40MB | LitData (.bin) |
| musan_synthetic | 36 files | ~72min | 132MB | WAV 原始 |

## 常见问题

### Q: WAV文件加载需要多长时间？
A: 首次加载36个WAV文件需要~30秒，之后会缓存到内存。

### Q: 能否合并两个噪声源？
A: 目前配置只支持单一噪声源。要合并，需要手动复制WAV文件到同一文件夹，或修改数据加载逻辑。

### Q: MUSAN合成噪声效果如何？
A: 多种类型的合成噪声可以增加模型的强化性。与真实噪声相比，合成噪声提供了更好的数据多样性而无需下载大量真实数据。

### Q: 我可以添加更多噪声文件吗？
A: 是的！在相应的文件夹中添加 `.wav` 文件即可自动被加载。

## 后续可选优化

### 0. 结构改版（建议优先）

- 将单阶段 ADDSE 改为两阶段：`ADDSE 粗增强 + PGUSE 细化恢复`。
- 训练目标从单纯感知指标，改为 `感知质量 + 保真约束` 的联合目标。
- 推理时引入按 SNR 的策略：低 SNR 强增强，高 SNR 轻增强以降低过处理。

### 1. 使用LitData优化（如果WAV加载变慢）
```python
from litdata import optimize
optimize(
    fn=lambda x: x,
    inputs=Path("data/chunks/musan_noise").glob("*.wav"),
    output_dir="data/chunks/musan_noise_optimized",
    chunk_bytes="10MB"
)
```

### 2. 下载真实噪声数据（如需要）
- **DEMAND**: http://www.openslr.org/17 (需要替代来源)
- **FSD50K**: https://zenodo.org/record/4060432 (精选音效库)

### 3. 混合配置
创建包含50%原有噪声 + 50%合成噪声的混合数据集。

### 4. 指标导向策略

- 继续保留 PESQ 作为主要感知指标。
- 同时引入 SDR/ESTOI 作为约束监控项，避免只追求 PESQ 导致高 SNR 退化。
