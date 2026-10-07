# 贡献指南

## 铁律：一切改动必须先登记

**任何参数改动、任何实验，必须先写进 `EXPERIMENT_LOG.md`，再动手。**

原因：本项目已发生多次「同一指标在不同文档给出不同数字」的事故。
日志是唯一真相来源，论文和 README 中的数字必须能回溯到日志条目。

### 改动流程

```
1. 在 EXPERIMENT_LOG.md §2 登记新参数（或标记旧参数废弃）
2. 跑验证脚本（verify_lock60.py / verify_lowrank.py）
3. 把结果写进 §3 验证实验记录表
4. 若有发现，写进 §4 关键发现（L-0xx）
5. 若失败，写进 NEGATIVE_RESULTS.md（N-0xx）
6. 更新 §8 Changelog
```

---

## 七条红线（不可违反）

来自项目核心设计，违反任何一条的 PR 会被拒绝：

1. **不要"优化"掉核心机制** —— `relu` / `detach` / `clamp` 都是设计，不是冗余
2. **不要把内生约束改成外挂惩罚** —— 稳态（行和归一化）在前向里，不是 loss 项
3. **不要把高斯带通改成帐篷函数** —— 帐篷在 |u|>2 处梯度恒 0，器官会永久死亡
4. **不要合并 `wiring` 和 `connect` 的计算** —— 分步是设计
5. **不要删除监控指标** —— `.detach()` 是防显存累积
6. **不要改器官初始化尺度** —— `base(0.5√d) + randn*0.5` 是打破对称的关键
7. **不要跳过 permuted 对照** —— 唯一能验证"分工真实存在"的实验

---

## 新增实验的要求

### 统计严谨性（论文级）

| 要求 | 标准 |
|---|---|
| seeds | 主表 ≥10（功率分析：n=5 只有 0.61 power） |
| 报告 | mean ± std + 95% CI（t 分布） |
| 架构对比 | 必须**配对**（同 seed、同数据、同初始化协议） |
| 效应量 | 必报 Cohen's d_z，不能只报 p 值 |
| 多重比较 | 同族假设用 Benjamini-Hochberg FDR 校正 |
| "打平" | 用 **TOST 等价性检验** + 预设等价界。`p>0.05` 不等于"证明等价" |

### 用记录器

采集已内置在训练脚本里，不需要手工调用：

```bash
python train_lock60.py --scale tiny --data <TinyStories> --delta-scale 0.5
# => output/runs/<run_id>/record.json + steps.csv，并追加 master.csv
```

若要接入自己的脚本：

```python
from run_metrics import RunMetrics
rm = RunMetrics(vars(args), script=__file__, seed=seed, tags=["main"])
rm.attach_data("data.txt", tokenizer=..., vocab_size=V, split_method="sequential")

for step, (xb, yb) in enumerate(train_iter):
    ...
    rm.log_step(step, model, loss=loss.item(), lr=lr, tokens_seen=n)

abl = rm.ablation(model, val_iter, loss_fn, n_perm=20)      # 三变体，必须跑
rm.finalize(model, final=abl, verdicts=RunMetrics.auto_verdicts(abl))
```

**新增指标必须【三处】同步改，缺一处整条证据链就断**（见 L-024，实测踩过）：

| # | 位置 | 作用 |
|---|---|---|
| 1 | `snowflake_B.stats()` 的 `_diag` | 产生字段 |
| 2 | `SnowflakeLM.agg_diag()` 的 `keys` | 跨 cell 聚合 |
| 3 | `run_metrics.STEP_FIELDS` | 允许落盘（未注册会抛错） |

这条防的是 `wiring_ent` → `wiring_entropy` 这类改名导致的口径丢失，
也防"加了指标但采不到"的静默失败。

强制 schema：未注册的指标名直接抛错（防止 `wiring_ent` → `wiring_entropy` 造成口径丢失）。

---

## 验证脚本（改代码前后都要跑）

```bash
python verify_lock60.py     # 17 项规格验收，纯 numpy，无需 GPU
python verify_lowrank.py    # 低秩两项结论
python train_lock60.py --smoke   # 合成数据端到端，判据阳性对照
```

`--smoke` 是**判据的阳性对照**：在已知有分工的合成任务上，核心判据必须 > 0。
如果它测出 0，是判据坏了或构造错了，不是"证明没有分工"。

---

## 提交规范

```
<类型>: <简述>

类型：feat / fix / exp / docs / refactor
exp = 实验（必须同时更新 EXPERIMENT_LOG.md）
```

示例：
```
exp: 扫描 delta_scale 0.5 vs 1.0

- EXPERIMENT_LOG.md §3 新增 V-019
- 结果：0.5 的 wiring_variance 更高，val_ppl 低 0.03
```

---

## 负结果同样重要

失败、失效、被推翻的设计写进 `NEGATIVE_RESULTS.md`，走同一条落盘路径。

本项目已登记的负结果包括：判据失效（eff_rank / 条件数 / permute 器官）、
乘性铰链是衰减加强器、满秩 delta 拿不到 FLOPs 优势等 13 条。
它们标记了不该再走的路，价值不低于正结果。
