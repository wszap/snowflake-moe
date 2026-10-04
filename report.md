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
