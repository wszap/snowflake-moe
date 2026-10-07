# 实验数据协议（论文 / 开源用）

> 目的：把"跑出来什么"变成"别人能复现、能证伪、能引用"。
>
> 本文件针对已发生的三类事故制定：
> - 同一指标在不同文档给出不同数字（擦除 18.1251 vs 22.83；终身学习 +0.0098 vs +1.8 vs −0.57；基线 9.9787 vs 10.0157）
> - 跑完才决定哪条是贡献（HARKing）
> - 漂亮结果进 README、失败结果进角落（file-drawer）
>
> **规则**：论文/文档中出现的每一个数字，必须能回溯到一个 `run_id`；每一个 `run_id` 的记录由 `experiment_logger.py` 自动落盘，不可手改。

---

## 0. 采集器

```python
from experiment_logger import RunLogger
log = RunLogger(cfg, script="run_cellmoe_ckpt_lock47.py",
                tags=["main","tinystories"], root="output/runs", repo_root=".")
log.attach_data("tinystories_25mb.txt", tokenizer=..., vocab_size=...,
                split_method="doc_boundary", split_seed=2026, doc_boundary_aware=True)
...
log.log_step(step, epoch=..., train_loss=..., val_ppl=..., in_band=..., mix_var=...)
log.finalize({"test_ppl":..., "mix_gain":...}, verdicts={"H1_noninferior": "pass"})
```

自动采集：`run_id` / 时间戳(UTC+CST) / git commit + dirty / config sha256 / python+torch+cuda 版本 / GPU 型号 / 数据 sha256+bytes+mtime。
强制 schema：未注册的指标名直接抛错（防止 `wiring_ent` → `wiring_entropy` 这种改名造成口径丢失）。

---

## A. 可复现性数据（缺失即不得进主表）

| 类别 | 字段 | 来源 |
|---|---|---|
| 代码 | git commit、dirty 标志、branch | 自动 |
| 配置 | 全部超参 + config sha256 | 自动 |
| 环境 | python / torch / cuda / cudnn / GPU 型号 / 驱动 | 自动 |
| 数据 | 文件名、字节数、sha256、mtime、下载来源与日期 | `attach_data` |
| 切分 | 切分方法、切分 seed、是否按文档边界（锁10 泄漏修复） | 手填 |
| 词表 | tokenizer 类型、vocab_size、train/val/test token 数 | 手填 |
| 成本 | 总壁钟、单 epoch 壁钟、峰值显存 | 自动/手填 |

**硬要求**：论文报告的每个数字，其 run 必须 `git_dirty == False`。脏仓库跑出来的结果只能进附录。

---

## B. 数据集指纹（每个数据集一张表）

| 数据集 | 用途 | 必记字段 |
|---|---|---|
| TinyStories | 主实验 | 版本/下载日期、bytes、sha256、token 数、vocab |
| Shakespeare | 次数据集 | 同上 |
| Bible KJV | 终身学习新域 | 同上 + 与 TinyStories 的域距离（如词表重叠率） |
| 合成 d=16 / 4096 | 容量惩罚 | **必须注明这是玩具任务**，结论不得外推到主模型 |

> ⚠️ 现有最强结果「可导容量惩罚 0.8955 vs Token Dropping 0.7383」出自合成任务。**若要用它做论文主线，必须在真实 LM 任务上重做**；否则只能作为辅助实验并明确标注。

---

## C. 预注册判据（**跑之前定死**，跑完不得修改）

这是本协议最重要的一节。判据先定，是为了杜绝"看哪个数字好看就说哪个是贡献"。

### 主假设

| ID | 假设 | 判据（可证伪） | 统计方法 | 最小 seeds |
|---|---|---|---|---|
| **H1** | 拼专家不劣于等参数 Fixed | test PPL ≤ Fixed × 1.05 | 配对 t + TOST 等价性检验（等价界 ±5%） | 5 |
| **H2** | 同等质量下更省算力 | 在 PPL 差 ≤2% 的配置对上，FLOPs/token ≤ Top-K × 0.5 | 配对比较，chunk P ∈ {16,32,128} | 5 |
| **H3** | 配方真的在做事 | `mix_gain > 0`（常数配方 − 学出配方的损失差），95% CI 下界 > 0 | 配对 t，5 seeds | 5 |
| **H4** | 分工不是掷骰子 | `mix_gain(learned)` 显著 **>** `mix_gain(permuted)` | 配对 t + Wilcoxon | 5 |
| **H5** | 训练不坍塌 | 全程 `in_band ≥ 0.5` 且 `cond_ema ≤ 阈值` | 时间序列最小值/最大值 | 3 |

### H4 是关键的、也是最容易被漏掉的对照

`eff_rank` / `mix_var` / `mix_sharp` **无法**区分"按内容分工"与"每个 chunk 随机掷骰子"——实测两者数值几乎相同（7.822 vs 6.926，噪声反而更高）。

正确的对照是 **permuted mix**：保持 mix 的边缘分布完全不变，只打乱 (chunk → mix) 的配对关系。

```
mix_gain(learned)  = loss(constant mix) − loss(learned mix)
mix_gain(permuted) = loss(constant mix) − loss(shuffled learned mix)
```

若 `mix_gain(learned) − mix_gain(permuted)` 的 CI 含 0，则"组合/分工"主张**不成立**，无论 eff_rank 多漂亮。

### 擦除 / 终身学习（若仍要主张）

| ID | 判据 | 现状 |
|---|---|---|
| **H6** 擦除 | 判据数值**待你确认**：验收线是"擦除后 PPL ≤ 30（仍可用）"还是"擦除后须显著退化以证明知识真的被擦除"？两种口径下 22.83 的结论相反 | 现有数字：100% 擦除 → PPL **22.83**（另一处记 18.1251）。判定前先定死判据方向 |
| **H7** 终身学习 | 旧域回归 ≤ 0.3（后放宽 0.5），新域下降 ≥ 20% | v2–v5 回归 +1.8 不合格；v4' 报 −0.57 待 scale=1.0 复跑 |

> ⚠️ H6/H7 的现有数据在两条链路上互相矛盾，**重跑前不得写入论文**。重跑必须锁定：`memory_read_scale` 是 25.0（legacy）还是可学习参数（lock43+）——这两条链路上"改成 1.0"是完全不同的两件事。

---

## D. 实验矩阵

| 组 | 配置 | seeds | 产出 |
|---|---|---|---|
| G1 主实验 | pz47 / Fixed / Top-K MoE，等参数 | 5 | H1 |
| G2 效率 | pz47 × chunk P ∈ {1,8,16,32,128} vs Top-K k∈{2,4} | 5 | H2 + 帕累托图 |
| G3 机制 | learned / constant / permuted mix 三个变体 | 5 | H3, H4 |
| G4 稳定性 | 训练全程 step 级日志 | 3 | H5 曲线 |
| G5 缩放 | 参数量 0.5M / 1.8M / 10M（若有算力） | 3 | 容量饱和曲线 |
| G6 次数据集 | Shakespeare 重复 G1 | 5 | 跨数据集一致性 |
| G7 擦除 | erase ∈ {0,25,50,75,100}% | 3 | H6 |
| G8 终身学习 | TinyStories → Bible，scale=1.0 重跑 | 3 | H7 |

**G2 的 P=1 必须跑**：它是"拼专家其实更贵 2.25×"的诚实对照，缺了它整个效率主张不可信。

---

## E. 统计严谨性

1. **seeds**：主表所有数字 ≥ 5 seeds，报 `mean ± std` 与 **95% CI（t 分布）**。
2. **配对**：所有架构对比必须同 seed、同数据、同初始化协议下配对比较。
3. **效应量**：必报 Cohen's d_z，不只报 p 值。n<5 时 p 值不可靠，以 CI 与效应量为准。
4. **多重比较**：同一族假设用 Benjamini-Hochberg FDR 校正（`bh_correct`）。
5. **"打平"要用 TOST**，不能用 p>0.05。`p>0.05` 是"没证据表明不同"，不是"证明等价"。声称"拼专家 ≈ Fixed"必须做等价性检验并预设等价界。
6. **负结果必报**：走同一条落盘路径，`verdicts` 里写 `"fail"`，汇总进 `NEGATIVE_RESULTS.md`。

---

## F. 图表清单（论文需要的最小集）

| # | 图 | 数据来自 |
|---|---|---|
| 1 | PPL vs FLOPs/token 帕累托前沿（pz47 多 P、Top-K 多 k、Dense） | G2 |
| 2 | 训练全程 `in_band` / `cond_ema` / `heal_term` 曲线 | G4 |
| 3 | `mix_gain` 三变体柱状图（learned / constant / permuted）+ CI | G3 |
| 4 | 参数量 vs PPL 缩放曲线（两架构） | G5 |
| 5 | 擦除比例 vs PPL | G7 |
| 6 | 帐篷+铰链机制示意图（带内/带外梯度） | 解析，非实验 |

每张图的数据源必须是 `output/runs/**/steps.csv` 或 `master.csv`，**不接受手抄数字**。

---

## G. 开源仓库清单

```
LICENSE                      Apache-2.0
README.md                    架构 + 诚实状态表（已有）
EXPERIMENT_PROTOCOL.md       本文件
NEGATIVE_RESULTS.md          负结果登记（必填）
environment.yml              版本锁定（python/torch/cuda）
requirements.txt
configs/*.yaml               每个实验一份
scripts/download_data.sh     含 sha256 校验
src/                         模型代码
output/runs/                 原始记录（JSON + steps.csv + master.csv）
scripts/make_figures.py      从 master.csv 直接出图，不手改
checkpoints/                 权重 + sha256
```

**开源前自检**：
- [ ] 每个论文数字都能 `grep` 到对应 run_id
- [ ] 主表所有 run 的 `git_dirty == False`
- [ ] 数据 sha256 与下载脚本一致
- [ ] 负结果文件非空，且包含 RL 治理触顶、MNIST p>0.05、KD 反向污染
- [ ] README 与 idea_evolution 口径已统一（当前未统一）

---

## H. 审稿人会攻击的点（提前备好数据）

| 攻击 | 需要的数据 |
|---|---|
| "基线太弱" | G1 含 ≥2 个基线（Fixed + Top-K MoE），且等参数/等 FLOPs 各一套 |
| "规模太小，结论外推无效" | G5 缩放曲线 ≥3 个点；或明确限定 claim 范围 |
| "只在 TinyStories 上" | G6 第二个数据集 |
| "没报方差 / cherry-pick seed" | ≥5 seeds + CI + 预注册 |
| "你怎么证明不是随机噪声" | **G3 的 permuted 对照**（最关键） |
| "效率优势是理论 FLOPs，实际不快" | 实测吞吐 token/s + 延迟 + 峰值显存，与理论并列 |
| "擦除/终身学习互相矛盾" | 先统一口径再写，或撤下该 claim |

---

## I. 批量跑法（真机）

```bash
# 单配置 5 seeds
for s in 2026 1 2 3 4; do
  python run_cellmoe_ckpt_lock47.py --seed $s --tag main 2>&1 | tee logs/main_$s.log
done

# 效率扫描（重点：P=1 是诚实对照，必须跑）
for P in 1 8 16 32 128; do
  python run_cellmoe_ckpt_lock47.py --chunk-size $P --tag eff_P$P
done
```

跑完：

```python
from experiment_logger import load_runs, seeds_aggregate, paired_compare
a = load_runs("output/runs", {"cfg.arch": "pz47"})
b = load_runs("output/runs", {"cfg.arch": "fixed"})
print(seeds_aggregate(a, "final.test_ppl"))
print(paired_compare(a, b, "final.test_ppl"))
```

---

## J. 当前已具备 / 仍缺

**已具备**：PPL 主表（1 seed）、擦除曲线、终身学习 v1–v5、容量惩罚合成实验、完整负结果记录。

**仍缺（阻塞论文）**：
1. ≥5 seeds 的主表（当前多为单 seed）
2. G3 的 permuted 对照 —— **没有它，"组合"主张不可证伪**
3. G2 的实测吞吐/延迟/显存（当前只有理论 FLOPs）
4. 第二个数据集的一致性
5. 三处数字冲突的统一（擦除、终身学习、基线 PPL）
