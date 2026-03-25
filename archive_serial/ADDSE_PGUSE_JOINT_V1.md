# ADDSE + PGUSE 联合改进方案 V1（工程+公式+实验）

## 1. 当前结论与问题定义

基于你已经完成的 4 个 SNR 桶评测，当前单阶段 ADDSE-S 的特征是：

- PESQ：四个桶均提升（有效）
- SDR：中高 SNR 桶显著下降（保真不足）
- ESTOI：高 SNR 桶出现回落（过处理风险）

因此目标不再是“单模型全包”，而是：

- ADDSE 负责低 SNR 粗增强（提升感知质量）
- PGUSE 负责中高 SNR 细化恢复（修复保真与可懂度）

同时明确一个核心瓶颈假设（后续要做实验证明）：

- 在高 SNR 场景，NAC 离散量化误差可能成为主要失真来源（“量化地板”效应）
- 这会导致“感知指标上升但 SDR 回落”的帕累托冲突


## 2. 代码工程信息（现有仓库可复用部分）

### 2.1 ADDSE 侧

- 训练入口：addse/addse/app/train.py
- 核心模块：addse/addse/lightning.py 中的 ADDSELightningModule
- 关键能力：
  - `--init-ckpt` 已支持从预训练权重初始化（strict=False）
  - 现有训练/评测链路可直接复用

ADDSE 的核心损失在 `ADDSELightningModule.loss`：

- 基于随机掩码比例 $\lambda$ 的 denoising cross-entropy
- 输入是 NAC 编码后的 noisy/clean 量化表示

对应工程位置：

- addse/addse/lightning.py（`loss`, `solve`, `log_score`）

### 2.2 PGUSE 侧

- 训练入口：PGUSE-main/train.py
- 主模型：PGUSE-main/model/model.py 的 `Model`
- SDE 采样：
  - Predictor：PGUSE-main/model/sde/predictors.py
  - SDE 定义：PGUSE-main/model/sde/sdes.py（BBED）
- 损失：PGUSE-main/model/loss/loss.py

PGUSE 现有训练逻辑：

- `training_step` 中同时训练预测支路 + 生成支路
- 验证指标直接记录 PESQ


## 3. 联合方案（V1，最小可行）

## 3.1 联合结构

采用两阶段串联，并引入“后验均值锚点”：

1. 阶段 A（ADDSE）：输出粗增强波形 $\hat y_A$
2. 阶段 B（PGUSE）：以 $\hat y_A$ 作为主输入做细化，输出最终波形 $\hat y_B$

在结构上优先采用“预测分支引导 + 残差生成”：

- 预测分支给出高保真骨架（后验均值近似）
- 生成分支只预测残差细节，而不是重建全量信号

训练阶段建议先冻结 ADDSE，仅训练 PGUSE 细化器（稳定、低成本）。

推理阶段：

- 低 SNR（<=5 dB）：ADDSE + PGUSE 全流程
- 中高 SNR（>5 dB）：PGUSE 轻推理步数（降低过处理）


## 3.2 数学描述（与你当前代码一致）

### A. ADDSE 损失（离散掩码去噪）

采样掩码率：

$$
\lambda \sim \mathcal U(0,1), \quad m_{k,l} \sim \text{Bernoulli}(\lambda)
$$

构造掩码后的 clean 量化表示：

$$
y_{\lambda,q} = (1-m) \odot y_q
$$

优化目标（按代码中 masked token 的负对数似然）：

$$
\mathcal L_{\text{ADDSE}} = -\frac{1}{\lambda}\,\mathbb E_{k,l}\left[m_{k,l}\log p_\theta(y^{tok}_{k,l}\mid y_{\lambda,q},x_q)\right]
$$

### B. PGUSE 损失（预测+生成）

训练时：

$$
x_t = \mu_t(x_0,y)+\sigma_t z,
$$

代码中联合损失为：

$$
\mathcal L_{\text{PGUSE}} = \underbrace{\left\|\frac{\hat{\sigma z}}{\sigma}-z\right\|_2^2}_{\mathcal L_g} +
\underbrace{0.5\|\hat r-r\|_2^2 + 0.5\|\,|\hat r|-|r|\,\|_2^2}_{\mathcal L_p}
$$

其中 $r$ 表示复频谱实虚部拼接表示（对应 `est_ri`, `tgt_ri`）。

### C. PGUSE 反向 SDE 更新（Euler-Maruyama）

由代码：

$$
\mathbf x_{t-\Delta t}=\mathbf x_t + \left(-f(\mathbf x_t,t)+g(t)^2 s_\theta(\mathbf x_t,t)\right)\Delta t + g(t)\sqrt{\Delta t}\,\epsilon
$$

且分数函数采用：

$$
s_\theta(x,t)= -\frac{\text{forward\_g}(x,t)}{\text{std}(t)^2}
$$

### D. 联合总目标（建议）

V1 训练先不端到端，仅训练 PGUSE 细化器时可用：

$$
\mathcal L_{\text{total}} = \mathcal L_{\text{PGUSE}} + \beta\,\mathcal L_{\text{cons}}
$$

其中一致性项可取：

$$
\mathcal L_{\text{cons}} = \|\text{STFT}(\hat y_B)-\text{STFT}(\hat y_A)\|_1
$$

用于避免 PGUSE 细化阶段把 ADDSE 的有效去噪结果破坏掉。

### E. 后验均值锚点与门控残差（本阶段优先）

在不大改工程骨架前提下，建议采用：

$$
\hat x_{final}=\hat x_{pred}+\eta(\widehat{\mathrm{SNR}})\cdot G(\tilde c,\hat x_{pred})
$$

其中：

- $\hat x_{pred}$：预测分支的高保真估计
- $G(\cdot)$：生成分支残差
- $\eta(\widehat{\mathrm{SNR}})$：随 SNR 增大而减小的门控系数

作用：

- 低 SNR：保持生成能力，延续 PESQ 优势
- 高 SNR：约束过处理，优先保护 SDR/ESTOI

### F. 动态截断采样（本阶段优先）

从固定步数改为 SNR 感知步数，建议使用连续映射：

$$
t_{start}=\mathrm{clip}(a-b\cdot \widehat{\mathrm{SNR}},\ t_{min},\ t_{max})
$$

或等价地映射采样步数 $N(\widehat{\mathrm{SNR}})$。

建议先做三档：

- 低 SNR：较多步
- 中 SNR：中等步
- 高 SNR：少步（1-5 步优先验证）

目标：在高 SNR 桶减少随机采样引入的包络扰动，抑制 ESTOI 回落。


## 4. 工程对接实现建议（不重构大工程）

## 4.1 数据与接口

- 继续沿用 addse 的数据准备与分桶评测流程
- 新增一个“联合推理脚本”：
  - 先调用 ADDSE checkpoint 输出增强波形
  - 将增强波形作为 PGUSE 输入进行细化
  - 输出最终波形并写入同一评测数据库

## 4.2 训练策略（三阶段）

1. Stage-0：复现基线
- 固定你当前 ADDSE-S best ckpt
- 固定 PGUSE 原训练配置（少量 epoch）

2. Stage-1：冻结 ADDSE，仅训 PGUSE
- 输入从 noisy 改为 ADDSE 输出（或混合输入）
- 观察中高 SNR 桶 SDR 是否回升

3. Stage-2：部分解冻联合微调（可选）
- 仅在 Stage-1 成功后尝试
- 只解冻 ADDSE 后层或桥接层，避免训练不稳定

4. Stage-3：离散-连续联合头（论文强化项）
- 在 RQDiT 末端增加连续回归头，联合优化离散码与连续潜向量
- 作为 1 区冲刺方向，不作为首轮必做


## 4.3 量化地板验证实验（必须补）

为避免“纯解释缺证据”，增加一项定量验证：

1. 取 clean 语音，经过 NAC 编码解码得到 $\tilde y_{nac}$
2. 计算上限保真：$\mathrm{SDR}(\tilde y_{nac}, y)$
3. 与模型输出 SDR 对比

若上限本身在高 SNR 已明显受限，可直接支撑“量化地板”论证。


## 5. 分桶评测协议（强制）

固定 4 个桶：

- [-5, 0]
- [0, 5]
- [5, 10]
- [10, 15]

每桶报告：

- PESQ / ESTOI / SDR
- 相对 noisy 的增量

判定门槛（V1）：

1. PESQ：至少 3 个桶正增量
2. SDR：中高 SNR 桶不再出现大幅负增量
3. ESTOI：高 SNR 桶不持续回落

新增效率指标（联合方案必报）：

4. 推理耗时：每桶报告平均时延或 RTF
5. 步数曲线：报告 1/3/5/16/64 步的质量-速度对比


## 6. 投稿导向（2区/1区）

## 2区优先目标

- 在预训练权重基础上完成联合方案改进（可接受）
- 给出分桶稳定增益 + 消融（有/无 PGUSE，有/无一致性项）
- 给出速度和显存开销（至少单卡 RTX 4060 可复现）

## 1区加分目标

- 跨数据集验证
- 给出质量-效率 Pareto 曲线
- 补一组“非预训练或弱预训练”对照，证明方法贡献而不是纯迁移收益
- 补“量化地板”上限实验与“离散-连续联合头”消融


## 7. 本周可执行清单（建议）

1. 先做联合推理（不训练）
- 开启“后验锚点 + 残差生成”，先看高 SNR 桶 SDR 是否回升

2. 立刻做动态步数实验
- 固定其它参数，仅比较 1/3/5/16/64 步
- 每步数都做 4 桶评测 + 速度记录

3. 再做 Stage-1 训练（冻结 ADDSE，训 PGUSE）
- 输入替换为 ADDSE 输出
- 先不加复杂损失，只训稳定基线

4. 分阶段加入损失
- 先加 $\mathcal L_{MR\text{-}STFT}$（小权重）
- 再加 $\mathcal L_{SI\text{-}SDR}$（更小权重）

5. 同步跑“量化地板”验证
- 给论文动机提供硬证据

---

这版 V1 的核心是：

- 不重跑全量预训练
- 不大改工程骨架
- 先用后验锚点与动态步数修复高 SNR 保真问题
- 再逐步引入高复杂度创新项冲击更高分区
