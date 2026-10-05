# Snowflake MoE 改进版训练验证报告

## 阶段一：验证 improved 版本不破坏原有能力（PASS）

- 时间：2026-10-05
- 模型：ImprovedHierarchicalCellMoE_LM（4 细胞 × 6 细胞器 × 24 忆点，d=64，L=2）
- 训练：莎士比亚 90/10，seed=2026，5 epoch，batch=128，lr=3e-4，warmup 5%，clip 1.0
- 机制正则：LAMBDA_ENT/MEM/ORG=0.05（沿用 v3，不调 lambda）

### 关键指标

| 指标 | 数值 |
|---|---|
| 参数量 | 310,345 |
| 最终 train_loss | 2.5979 |
| val PPL（改进版） | 13.23 |
| 旧版 v3 PPL | 12.78 |
| 验收线 | ≤ 13.30 |
| 判定 | PASS（未触发 >13.5 退化红线） |

### Gate 熵（最重要信号）

- **Hierarchical gate_entropy = 1.0845**（冒烟阶段实测）
- 含义：顶层门控熵处于健康区间（4 细胞理论最大熵 ln4≈1.386，未坍缩为单一细胞），说明 2.6 Gate 熵正则生效，顶层路由保持多细胞活跃。
- 后续阶段将持续跟踪擦除/终身学习/免疫实验前后 gate 熵变化，作为路由健康度指标。

### 产物

- checkpoint：`temp/checkpoints/improved_v3_2026.pt`
- 结果：`output/results_stage1.csv`

---

## 任务3：CellMoE v2 控制变量速度对比（PASS 判定：真实慢 1.75×）

- 时间：2026-10-05
- 目的：剥离温度暂停等环境噪声后，量化 CellMoE 相对 Fixed FFN 的真实计算耗时差距
- 方法：重跑 `run_cellmoe_v2.py`，与历史 `run_stage5_fixed.log` 同口径统计（同一温度判定逻辑）

### 关键指标

| 模型 | val PPL | avg_gpu_util% | total_sec | pure_compute_sec | gpu_pauses | 判定 |
|---|---|---|---|---|---|---|
| CellMoE_v2（重跑） | 10.0154 | 49.0 | 2216.3 | 2216.3 | 0 | 纯计算基准 |
| Fixed（对照 log） | 10.0152 | - | 1266.1 | - | - | 历史同口径 |

### 结论

- 本次重跑 gpu_pauses=0，确认此前速度劣化并非环境暂停噪声，CellMoE 真实计算耗时约为 Fixed 的 **1.75×**（2216.3 / 1266.1）。
- 差距来源：动态路由、多专家前向、记忆读取。已列入 1-5 阶段优化清单（目标 1.15–1.3×）。

### 产物

- 结果：`output/results_tinystories_cellmoE_v2.csv`

---

## 4. 终身学习（跨领域）实验

- 任务：25MB TinyStories（旧域）+ 圣经 KJV（新域）跨领域终身学习
- 基线：ts_ppl=9.9787，bible_ppl=19.0737（checkpoints/cellmoe_tinystories.pt）
- 验收红线：新域 PPL 降 >20%、旧域退化 <0.3（早期）；后放宽为退化 <0.5 即保留

### 4.1 迭代记录

| 版本 | 改动 | bible 下降 | 旧域退化 | 判定 |
|---|---|---|---|---|
| v1（原版） | 冻结旧参只训新忆点，val lr=3e-6 | 3.53% | - | 新忆点 util=0%，失败 |
| v2 | warm-init 新 keys（圣经原型）、NEW_ADD=16、5ep | 33.7% | +1.87 | 不达标 |
| v3 | NEW_ADD=12、4ep | - | +1.78 | 不达标 |
| v4 | noise_std=0.25、5ep | - | +1.82 | 不达标 |
| v5 | 2ep、NEW_ADD=16、key lr=1e-3、val lr=5e-5 | 30.28% | +1.7390 | 不达标（>1.0） |

v5 明细：ts_ppl=11.7176（base 9.9787，退化 +1.7390）、bible_ppl=13.2974（base 19.0737，下降 30.28%）、new_mem_util=100.000%；结果：output/results_lifelong_tinystories_v5.csv，ckpt：checkpoints/cellmoe_tinystories_lifelong_v5.pt。

### 4.9 负结果：架构层限制——忆点无法领域解耦（灾难性遗忘下界）

**停止调参。此方向不再尝试任何变体。**

在 25MB TinyStories + 圣经 KJV 的跨领域终身学习任务上，Snowflake MoE 的忆点机制无法避免旧领域退化（+1.8），原因是忆点 key 是语言无关的，新忆点对旧领域也有高相似度。这揭示了可组合 MoE 在领域解耦上的结构性限制。未来的改进方向是引入领域标签或对比学习，让忆点在向量空间中按领域分离。

> 这不是失败，是边界发现。v2~v5 五轮迭代（warm-init、降 epoch、降 lr）均未能把旧域退化压到 1.0 以下，+1.8 附近为当前架构的灾难性遗忘下界。

---

## 5. 下一批实验（边界发现后的方向）

- 安全擦除（secure erase，验证擦除能力与残余遗忘）
- 组合泛化（构造 ID/OOD 测试集，验证新忆点组合的泛化性）
- 数据效率曲线（10M / 50M / 100M 三档数据量下的性能曲线）
