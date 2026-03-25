# ✅ 噪声数据源配置完成总结

## 📊 已完成的工作

### 1. 目录结构重组
```
data/chunks/
├── edbase_noise_original/      ← 原有 5 个噪声文件（保留）
├── musan_noise/                ← 新增 36 个合成噪声（132MB）
├── musan_noise_raw/            ← 生成过程备份
├── edbase_speech/              ← 语音数据（不变）
```

### 2. 新增数据
- **36个合成噪声文件**，140秒每个,16kHz采样率，总容量**132MB**
- **6种噪声类型**：White/ Pink/ Brown/ Speech-like/ Machine/ Traffic
- **完全本地生成**，无需外部下载

### 3. 配置文件
创建了两个新的训练配置文件：

| 配置文件 | 噪声源 | 文件数 | 大小 | 推荐场景 |
|---------|-------|------|-----|--------|
| `addse-s-original-ft.yaml` | 原有edbase | 4 chunks | ~40MB | 快速验证、对标 |
| `addse-s-musan-ft.yaml` | 新增合成 | 36 WAV | 132MB | 扩展训练、多样性 |

### 4. 文档
- `NOISE_SETUP.md` - 详细使用指南
- `compare_noise_sources.py` - 自动对比脚本

---

## 🚀 立即开始使用

### 快速测试（选一个運行）

**方案 A：使用原有数据（最快）**
```bash
cd d:\Users\KSL\PycharmProjects\the_sound
.\.venv\Scripts\python.exe -m addse.app train ^
  addse/configs/addse-s-original-ft.yaml ^
  --init-ckpt addse/logs/addse-edbase-quick/checkpoints/addse-s.ckpt ^
  --overwrite
```
⏱️ 预计时间：~6-8 分钟 (10 epochs)

**方案 B：使用新合成噪声（推荐）**
```bash
cd d:\Users\KSL\PycharmProjects\the_sound
.\.venv\Scripts\python.exe -m addse.app train ^
  addse/configs/addse-s-musan-ft.yaml ^
  --init-ckpt addse/logs/addse-edbase-quick/checkpoints/addse-s.ckpt ^
  --overwrite
```
⏱️ 预计时间：~8-10 分钟 (10 epochs，WAV处理稍慢)

### 自动对比两种方案（可选）
```bash
cd d:\Users\KSL\PycharmProjects\the_sound\addse
.\.venv\Scripts\python.exe compare_noise_sources.py
```
⏱️ 预计时间：~40 分钟 (顺序训练评估两个方案)

---

## 📈 当前实测结果（分桶评测）

基于已完成的 4 个 SNR 桶评测（ADDSE-S-FT vs noisy）：

| SNR桶 | PESQ增量 | ESTOI增量 | SDR增量 |
|-----|--------|--------|--------|
| **[-5,0]** | **+0.12** | **+0.10** | **+2.04** |
| **[0,5]** | **+0.19** | **+0.10** | **-2.54** |
| **[5,10]** | **+0.31** | **+0.06** | **-6.74** |
| **[10,15]** | **+0.30** | **-0.03** | **-11.64** |

结论：

- `PESQ` 在四个桶全部提升，说明感知质量路径有效。
- `SDR` 在中高 SNR 桶显著下降，说明单阶段 ADDSE 存在保真副作用。
- 高 SNR 桶 `ESTOI` 略回落，存在过处理风险。

---

## ⚙️ 进阶选项

### 0. 路线升级（推荐）

从单阶段 ADDSE 升级为：

1. `阶段A：ADDSE 粗增强`（负责低 SNR 去噪增益）
2. `阶段B：PGUSE 细化恢复`（负责中高 SNR 的保真与可懂度修复）

该路线目标：保持 PESQ 优势，同时修复中高 SNR 的 SDR 负增量。

### 延长训练（更好的效果）
```bash
# 50 epochs (~30 分钟)
trainer.max_epochs=50

# 100 epochs (~60 分钟)
trainer.max_epochs=100 trainer.limit_train_batches=150 trainer.limit_val_batches=5
```

### 评估最佳模型
```bash
python -m addse.app eval addse/configs/addse-s-musan-ft.yaml ^
  "addse/logs/addse-s-musan-ft/checkpoints/epoch=XX-val_loss=X.XX.ckpt" ^
  --device cuda --output-db eval_musan.db --overwrite ^
  --num-examples 30 lm.num_steps=64 --noisy
```

### 未来：真实噪声合并
如果需要真实噪声：
1. 下载 DEMAND 等数据集
2. 复制到 `data/chunks/mixed_noise/`
3. 创建新配置文件使用它

---

## 🧭 迭代节奏（边跑边调）

当前采用渐进式优化，不追求一次定型：

1. 固定分桶评测协议（4桶 + noisy基线）
2. 每次只改一个关键因素（结构/损失/步数/数据）
3. 达标则保留，退化则回退

本阶段优先级：

1. 完成 ADDSE + PGUSE 联合基线
2. 观察中高 SNR 桶 SDR 是否回升
3. 再决定是否需要更大规模重训练或编码器解冻

---

## 📋 检查清单

- ✅ 噪声文件夹重组（2个分离源）
- ✅ 132MB 合成噪声生成（36个文件）
- ✅ 两个配置文件创建（原有 + 新增）
- ✅ 使用文档编写
- ✅ 对比脚本准备

**下一步：在现有预训练基础上做 ADDSE+PGUSE 联合尝试，并继续按分桶标准迭代。** 🎯
