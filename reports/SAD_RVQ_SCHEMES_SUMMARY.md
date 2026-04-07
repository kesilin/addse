# 🚀 SAD-RVQ 进阶消融实验完整总结

**日期**: 2026-04-04  
**项目**: The Sound - Advanced Audio Enhancement via SAD-RVQ  
**阶段**: Phase 3 - Architecture Surgery & Scheme Design

---

## 📊 核心发现回顾

### Oracle Ceiling 实验结果
前 3 层 DiT 预测 + 后 5 层完美 Clean Token：
- **PESQ: 3.300** (vs 当前 1.591，**提升 107%**)
- **SI-SDR: +1.824** (vs 当前 -5.897，**提升 7.7 分**)
- **结论**: ✅ 后 5 层（Layer 4~7）是主要瓶颈

### 其他关键发现
| 方法 | PESQ | SI-SDR | 收益 |
|------|------|--------|------|
| Dynamic Routing | 1.726 | -3.737 | +8.5% PESQ |
| VAD Masking | 1.658 | -4.664 | +4.2% PESQ |
| Current Baseline (rvq_split) | 1.591 | -5.897 | - |

---

## 🎯 4 大改进方案

### Scheme A: 后 5 层学习增强 ⭐
```
核心机制: 对 Layer 4+ 的 CE loss 添加 entropy boost
实现难度: ★☆☆ (最简单)
预期效果: PESQ 1.591 → 1.8-1.9
集成行数: ~30 行
优先级: 🥇 最快验证
```

✅ 代码已集成到 `addse/addse/lightning.py`

```python
# 伪代码
if self.sad_rvq_scheme == "a":
    entropy_boost_loss = -entropy[3:, :, :].mean()
    total_loss += 0.5 * entropy_boost_loss
```

---

### Scheme B: 多头并联 (Head-A/B) ⭐⭐
```
核心机制: 分离语义头(L0-3)和细节头(L4+)
实现难度: ★★☆
预期效果: PESQ 1.591 → 1.9-2.0
集成行数: ~80 行
优先级: 🥉 架构最优雅
```

**Head-A (L0-3)**: 负责语义层，学习速率标准  
**Head-B (L4+)**: 负责细节层，可用更高学习率

---

### Scheme C: 渐进式精化 ⭐
```
核心机制: 按 epoch 解冻不同层
- Epoch 0-1: 冻结所有层（baseline）
- Epoch 2-3: 解冻 Layer 3
- Epoch 4+: 解冻 Layer 4+
实现难度: ★☆☆
预期效果: PESQ 1.591 → 1.85-1.95
集成行数: ~50 行
优先级: 🥈 稳定性最好
```

---

### Scheme D: 可学习门控软融合 ⭐⭐⭐ **【推荐】**
```
【核心创新】不再硬切分，改为学习的概率融合

DiT 输出:      base_logits [B, K, L, V]
STFT 支路:     acoustic_logits [B, K, L, V]
Router 学习:   gate [B, K, L, 1] ∈ [0, 1]

融合公式:
  final_logits = base_logits * (1 - gate) 
               + acoustic_logits * gate

优势：
✓ 每一帧、每一层都能动态决策
✓ gate 梯度双向流通，促进协同学习
✓ 可捕获"寻找最佳融合比例"的学习过程

实现难度: ★★★
预期效果: PESQ 1.591 → 2.0-2.1 🚀
集成行数: ~120 行
优先级: 🏆 最高潜力
```

✅ `SchemeDLearnableGatingRouter` 已在 lightning.py 中定义

---

## 📁 生成的文件清单

### 核心代码
1. **`addse/run_sad_rvq_schemes.py`** - 4 个方案的模块定义与验证
2. **`addse/addse/lightning.py`** - 集成了 Scheme A 和 Scheme D 框架
3. **`addse/run_schemes_comparison.py`** - 方案对比实验运行脚本
4. **`addse/probe_architecture_surgery.py`** - 扩展了 4 个新的诊断 probe

### 文档
1. **`reports/SAD_RVQ_4_SCHEMES_GUIDE.md`** - 完整集成指南与时间表
2. **`addse/probe_outputs/sad_rvq_advanced_diagnosis.txt`** - 诊断实验结果报告

---

## 🔧 快速开始

### 立即验证 Scheme A（最快，2 小时）
```bash
# 在 run_v33.py 中添加 --sad-rvq-scheme 参数，然后运行：
python addse/run_v33.py --epochs 5 --batch-size 100 --sad-rvq-scheme a
```

### 测试 Scheme D（最高潜力，6 小时）
```bash
python addse/run_v33.py --epochs 5 --batch-size 100 --sad-rvq-scheme d
```

### 一次性对比所有方案
```bash
python addse/run_schemes_comparison.py --schemes baseline,a,d,c,b --epochs 5
```

### 快速测试模式
```bash
python addse/run_schemes_comparison.py --quick  # 只跑 baseline + A，2 epoch
```

---

## 📈 预期实验时间表

| 方案 | 集成时间 | 训练时间 | 总耗时 | 优先级 |
|------|---------|---------|--------|--------|
| **A** | 0.5h | 1.5h | **2h** | 🥇 |
| **D** | 2h | 1.5h | **3.5h** | 🏆 |
| **C** | 1h | 1.5h | **2.5h** | 🥈 |
| **B** | 2h | 1.5h | **3.5h** | 🥉 |
| **All** | - | - | **~12h** | - |

---

## ✅ 验证清单

### 已完成 ✅
- [x] Oracle Ceiling 实验确认后 5 层是瓶颈
- [x] 4 个方案的设计、模块化与验证
- [x] Scheme A/D 在 lightning.py 中的初步集成
- [x] 诊断 probe 的扩展（Oracle, Dynamic Routing, VAD, Smoothing）
- [x] 完整的集成指南与文档

### 待执行 ⏳
- [ ] 修改 run_v33.py 添加 CLI 参数映射
- [ ] 运行 Scheme A 验证（快速反馈）
- [ ] 运行 Scheme D 验证（最高潜力）
- [ ] 对比 Scheme A vs D 的收益与成本
- [ ] 选定最优方案进行最终训练

---

## 💡 关键洞察

### 为什么 Scheme D 最有潜力？
1. **梯度双向流**：gate 的梯度同时优化两个分支，而非选择一个
2. **动态自适应**：每一帧可以有不同的融合比例，分布式决策而非全局策略
3. **可解释性**：gate 权重可视化能显示模型在何时何处信任哪个分支
4. **理论上界**：Oracle (3.3) 意味着完美融合能达到 3.3，soft fusion 距离更近

### 为什么先跑 Scheme A？
1. 改动最小，风险最低
2. 5 分钟内能知道方向是否正确
3. 作为 baseline，为其他方案的收益奠定基准

---

## 🎬 建议的执行步骤

### Day 1 (今天)
1. ✅ 修改 run_v33.py 集成 CLI 参数
2. ✅ 运行 Scheme A 快速验证 (1 epoch)
3. ✅ 观察 entropy_boost 是否能推高 PESQ

### Day 2
1. ✅ 基于 A 的结果调整权重
2. ✅ 集成 Scheme D 的 acoustic_logits 计算
3. ✅ 运行 Scheme D 短期实验 (1-2 epoch)

### Day 3
1. ✅ 对比 A vs D 的完整结果
2. ✅ 选定最优方案
3. ✅ 运行最终训练 (5+ epoch)

---

## 📚 参考资源

- [Oracle Ceiling Results](addse/probe_outputs/sad_rvq_advanced_diagnosis.txt)
- [Architecture Surgery Guide](SAD_RVQ_4_SCHEMES_GUIDE.md)
- [Scheme Modules](addse/run_sad_rvq_schemes.py)
- [Integration Guide](addse/addse/lightning.py) - 搜索 `sad_rvq_scheme`

---

## 🏁 最终目标

| 指标 | 当前 | Scheme A 目标 | Scheme D 目标 | Oracle 上界 |
|------|------|-------------|-------------|----------|
| PESQ | 1.591 | **1.8+** | **2.0+** | 3.300 |
| SI-SDR | -5.897 | **-3.0** | **-1.0** | +1.824 |
| 收益幅度 | - | +13% | +26% | +107% |

---

**Status**: 🟢 **Ready for Execution**  
**Created**: 2026-04-04 GMT+8  
**Owner**: ADDSE Research Team
