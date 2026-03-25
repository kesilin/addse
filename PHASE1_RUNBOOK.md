# 第一阶段执行任务单（先验证，不训练）

## 0. 阶段目标

本阶段只验证一件事：

- 在不重训 ADDSE 的前提下，接入 PGUSE 细化后，是否能修复中高 SNR 桶的 SDR 下滑，同时尽量保持 PESQ 优势。

本阶段不做：

- 不重跑预训练
- 不改联合损失
- 不解冻编码器

---

## 1. 目录约定

工作根目录：

- `d:/Users/KSL/PycharmProjects/the_sound`

建议输出目录：

- `addse/outputs/phase1/addse_wav/`（ADDSE 中间结果）
- `addse/outputs/phase1/joint_wav/`（ADDSE+PGUSE 最终结果）
- `addse/outputs/phase1/tables/`（评测结果表）

---

## 2. 前置检查

### 2.1 确认 ADDSE checkpoint

- 当前使用：`addse/logs/addse-s-edbase-ft/checkpoints/epoch=07-val_loss=3.54.ckpt`

### 2.2 确认 PGUSE checkpoint

你需要准备一个可用的 PGUSE ckpt（如果还没有，先用你已有的最优 ckpt 路径填进去）。

- 示例占位：`PGUSE-main/log/ckpts/version_xxx/epoch=...-pesq=....ckpt`

### 2.3 确认数据目录（PGUSE test 需要 wav 对）

PGUSE test 读取方式：

- `test_src_dir`：输入 wav 目录
- `test_tgt_dir`：参考 clean wav 目录（用于计算 PESQ）

---

## 3. 实验设计（第一阶段）

## 3.1 实验组

1. Baseline-A：Noisy
2. Baseline-B：ADDSE（你已跑过）
3. Joint：ADDSE + PGUSE（本阶段新增）

## 3.2 步数消融（仅 Joint）

固定其它参数，只改推理步数：

- `N = 1, 3, 5, 16, 64`

## 3.3 分桶评测（强制）

- `[-5,0]`
- `[0,5]`
- `[5,10]`
- `[10,15]`

每桶记录：

- PESQ
- ESTOI
- SDR
- 相对 noisy 增量
- 推理时延或 RTF

---

## 4. 执行步骤

## Step 1: 生成 ADDSE 中间增强 wav

说明：先把需要评测的样本统一导出为 ADDSE 增强 wav，作为 PGUSE 输入。

建议做法：

1. 使用你当前评测样本集（和分桶一致）
2. 将每个样本的 ADDSE 输出保存到 `addse_wav`

备注：如果现有 eval 命令未直接保存 wav，先写一个小导出脚本（推荐后续由我补一个脚本）。

## Step 2: 配置 PGUSE test config（针对 ADDSE 输出）

复制：

- `PGUSE-main/config/config.yaml`

另存为：

- `PGUSE-main/config/config_phase1_joint.yaml`

关键字段改成：

1. `ckpt_path`：你的 PGUSE ckpt
2. `dataset_config.test_src_dir`：`addse/outputs/phase1/addse_wav`
3. `dataset_config.test_tgt_dir`：对应 clean wav 目录
4. `devices`：`[0]`

## Step 3: 跑 Joint 推理（按步数）

你需要分别运行 5 次（N=1/3/5/16/64）。

建议每次修改 `config_phase1_joint.yaml` 里的测试 SDE 步数相关参数，然后执行：

```bash
cd d:/Users/KSL/PycharmProjects/the_sound/PGUSE-main
D:/Users/KSL/PycharmProjects/the_sound/.venv/Scripts/python.exe test.py --config ./config/config_phase1_joint.yaml --save_enhanced ../addse/outputs/phase1/joint_wav/N_XX
```

其中 `N_XX` 替换为 `N_1/N_3/N_5/N_16/N_64`。

## Step 4: 对 Joint 输出做分桶评测

对每个 `N_xx` 目录分别评测 4 个 SNR 桶。

建议沿用 addse 现有分桶评测流程，输出到不同数据库：

- `eval_joint_N1_bucket_*.db`
- `eval_joint_N3_bucket_*.db`
- ...

## Step 5: 汇总结果表

输出两张总表：

1. 质量表：
- 行：SNR 桶
- 列：Noisy / ADDSE / Joint-N1 / Joint-N3 / Joint-N5 / Joint-N16 / Joint-N64

2. 效率表：
- 行：N=1/3/5/16/64
- 列：平均延迟、RTF、显存峰值（可选）

---

## 5. 通过标准（是否进入第二阶段训练）

满足以下至少 3 条即可进入第二阶段：

1. 中高 SNR 桶（[5,10]、[10,15]）的 SDR 相比 ADDSE 明显回升。
2. PESQ 在 4 桶中至少 3 桶不劣于 ADDSE（允许微小波动）。
3. 高 SNR 桶 ESTOI 不再持续回落。
4. 存在一个低步数配置（优先 1/3/5）在质量可接受时显著提速。

---

## 6. 止损标准

出现以下任一情况，暂停进入训练：

1. 所有步数配置在高 SNR 桶 SDR 都无改善。
2. 低步数虽然提速，但 PESQ/ESTOI 明显崩塌。
3. Joint 相对 ADDSE 没有形成可复现收益。

---

## 7. 记录模板（建议直接复制到汇报）

### 7.1 分桶质量表模板

| SNR桶 | 方法 | PESQ | ESTOI | SDR | Delta-PESQ(vs noisy) | Delta-SDR(vs noisy) |
|---|---|---:|---:|---:|---:|---:|
| [-5,0] | noisy |  |  |  |  |  |
| [-5,0] | addse |  |  |  |  |  |
| [-5,0] | joint-N5 |  |  |  |  |  |

### 7.2 步数-效率模板

| 步数N | 平均延迟(ms) | RTF | 备注 |
|---:|---:|---:|---|
| 1 |  |  |  |
| 3 |  |  |  |
| 5 |  |  |  |
| 16 |  |  |  |
| 64 |  |  |  |

---

## 8. 本阶段交付物

1. `N=1/3/5/16/64` 的 Joint 输出 wav
2. 4 桶评测数据库与结果总表
3. 一页结论：是否进入第二阶段（冻结 ADDSE 训练 PGUSE）
