# Snowflake MoE v1.0 — Memory Points

> **本版本已冻结**，不再改动架构。仅维护文档。
>
> **忆点定义（v1.0）**：独立的 memory-value slot（值槽），与路由权重**分离**。
> ⚠️ 与 v2.0 的「忆点即连接本体」是**不同架构下的不同定义**，请勿混用。

---

## 架构

Fine-grained experts + shared experts + sparse activation routing，
外加一套与路由**解耦**的记忆值槽。

## 核心结果

### PPL（词级 TinyStories，5 epochs）

| Model | Params | val_ce | val_ppl |
|---|---|---|---|
| CellMoE (ckpt run) | 1,823,077 | 2.3042 | **10.0157** |
| CellMoE (MR2 fixed) | 1,823,077 | 2.3043 | 10.0169 |

### Memory Erasure

| Erase | val_ce | val_ppl |
|---|---|---|
| 0% | 2.3041 | 10.0148 |
| 50% | 2.3151 | 10.1261 |
| 100% | 2.8973 | 18.1251 |

单调恶化，数据内部自洽（exp(val_ce) 与 val_ppl 逐行核对一致）。

### Lifelong Learning（TinyStories + Bible）

| Stage | ts_base | ts_life | regression | bible_base | bible_life |
|---|---|---|---|---|---|
| lifelong | 10.0148 | 10.0246 | **+0.0098** | 19.4129 | 18.2200 |

### OOD Robustness — Relative Degradation（Shakespeare → OOD）

| Model | id_ppl | ood_ppl | ratio | 退化 |
|---|---|---|---|---|
| CellMoE v3 | 15.0317 | 13.9253 | 0.9264 | **7.4%** |
| Fixed MoE | 26.5113 | 18.1065 | 0.6830 | 31.7% |

> **注意**：两个模型 ratio 均 < 1，说明 OOD 集本身比 ID 更简单。
> 本表衡量的是**同等分布偏移下的相对退化**，不是绝对 OOD 泛化能力。

### Sparse Activation Ratio（Snowflake v3, Shakespeare）

| Mode | Active | Fixed | Ratio | val_ppl (3 seeds) |
|---|---|---|---|---|
| snowflake_v3 | 203,713 | 259,009 | **0.7865** | 13.33–14.39 |

> **注意**：Param Ratio = Active / Fixed（稀疏激活比），
> **不是**与 dense baseline 的对比。本仓库尚无 dense 对照（v2.0 正在补充）。

---

## Reproduction

```bash
python run_cellmoe_ckpt.py          # 基线 PPL ~10.01
python run_cellmoe_mr2.py           # MR2 fixed ~10.02
python run_erase_tinystories.py     # 擦除 0/50/100%
python run_erase_mr2.py
python run_lifelong_tinystories.py  # 终身学习
python run_ood_v2.py                # OOD
python run_stage1.py                # 5 阶段流水线
python train_mnist.py               # MNIST 路由消融
python run_experiments.py           # LM scaling & ablation sweep
```

## Checkpoint 规范

保存位置：仓库根目录 `checkpoints/`（已 gitignore，不入库）。
命名：`checkpoints/{实验名}_{seed}.pt`，seed 取脚本内 `SEED` 常量（默认 2026）。

保存内容为 `torch.save` 的 dict：`state_dict` + 训练配置 + 关键指标（如 `final_ppl`）。

完整表格见主 [README](../README.md)。

## 已知口径分歧

以下三处存在不同数值来源，README 采用了数值更好的一侧，**未经口径统一**：

| 项 | 值 A | 值 B | 采用 |
|---|---|---|---|
| 基线 PPL | 9.9787 | 10.0157 | 10.0157 |
| 100% 擦除 | 18.1251 | 22.83 | 18.1251 |
| 终身学习 | +0.0098 | +1.8 / −0.57 | +0.0098 |

建议后续补一张「不同 tokenization / 数据切分口径对照表」。
