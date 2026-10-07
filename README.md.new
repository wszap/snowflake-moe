# Snowflake MoE

A composable Mixture-of-Experts architecture with memory points (independent research).

> **本仓库包含两个架构版本，核心定义不同，请勿混用数字。**
> 见下方「版本导航」。

---

## 版本导航

| | **v1.0 — Memory Points** | **v2.0 — Compositional** |
|---|---|---|
| 目录 | [`v1_memory_points/`](v1_memory_points/) | [`v2_compositional/`](v2_compositional/) |
| 忆点定义 | 独立 **memory-value slot**（值槽），与路由**分离** | **忆点即连接本体**，与路由**耦合** |
| 组合方式 | 稀疏路由选择专家 | 参数空间合成（`bmm(wiring, W_U)`），非线性在求和之后 |
| 任务口径 | **词级** TinyStories | **字符级** TinyStories（vocab=98） |
| 参数量 | 1,823,077 | 5,163,618 |
| 代表结果 | PPL **10.0157**，可擦除记忆，终身学习 | eff≈26 **自发组合**，FLOPs **0.664x** |

**⚠️ 两个版本的 PPL 不可比**（词级 vs 字符级，差近一倍量级）。

---

## v1.0 核心结果（词级 TinyStories，5 epochs）

| Model | Params | val_ce | val_ppl |
|---|---|---|---|
| CellMoE (ckpt run) | 1,823,077 | 2.3042 | **10.0157** |
| CellMoE (MR2 fixed) | 1,823,077 | 2.3043 | 10.0169 |

**Memory Erasure**（0% → 50% → 100% 单调恶化，数据内部自洽）

| Erase | val_ce | val_ppl |
|---|---|---|
| 0% | 2.3041 | 10.0148 |
| 50% | 2.3151 | 10.1261 |
| 100% | 2.8973 | 18.1251 |

**Lifelong Learning**（TinyStories + Bible）：regression **+0.0098**

**OOD Robustness**（相对退化，非绝对泛化）

| Model | id_ppl | ood_ppl | ratio | 退化 |
|---|---|---|---|---|
| CellMoE v3 | 15.0317 | 13.9253 | 0.9264 | **7.4%** |
| Fixed MoE | 26.5113 | 18.1065 | 0.6830 | 31.7% |

> 两个模型 ratio 均 < 1 ⇒ OOD 评测集本身比 ID 更简单。
> 这里衡量的是**同等分布偏移下的相对退化**，不是绝对泛化能力。

---

## v2.0 核心结果（字符级 TinyStories，10 seeds）

| 假设 | 数值 | 统计 | 判定 |
|---|---|---|---|
| H1 架构是拼专家（非线性在求和之后） | 数学事实 | — | ✅ |
| H2 自发组合 | eff = **25.73 ± 2.06** | 10 seeds，全部 > 23 | ✅ |
| H3 涌现而非强制 | loss **1.779**（无约束）vs 1.784 | 无约束更优 | ✅ |
| H4 分工真实存在 | core_stat = **+1.363 ± 0.029** | CI[1.342, 1.384]，d_z≈37 | ✅ |
| H5 效率碾压 | FLOPs **0.664x** | vs Top-K MoE | ✅ |

**最关键的一点**：`--min-eff` 约束**根本没激活**（eff 25.7 ≫ K=8 ⇒ a=0）。
模型在真实 LM 任务上**自发**使用了约 26 个器官，且不加约束时 loss 更低。

详见 [`v2_compositional/RESULTS.md`](v2_compositional/RESULTS.md)。

---

## 快速开始

### v1.0（词级）

```bash
cd v1_memory_points
pip install -r ../requirements.txt

python run_cellmoe_ckpt.py          # 基线 PPL ~10.01
python run_erase_tinystories.py     # 擦除 0/50/100%
python run_lifelong_tinystories.py  # 终身学习
python run_ood_v2.py                # OOD
python run_stage1.py                # 5 阶段训练
```

### v2.0（字符级）

```bash
cd v2_compositional
python train_lock60.py --scale tiny \
    --data <TinyStories.txt> \
    --delta-scale 0.5 --no-bf16 --fix-mojibake \
    --max-steps 3000 --seed 0

# dense baseline 对照（补 v1.0 README 缺失的 claim）
python dense_baseline.py --data <TinyStories.txt> --scale-sweep
python compare_dense.py --dense output/dense
```

---

## 负结果与经验教训

独立研究最容易缺失、也最有价值的部分。本仓库公开全部失败：

- [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) —— 13 条负结果
- [`docs/blog_moe_six_traps.md`](docs/blog_moe_six_traps.md) —— 六个陷阱（机制全部验证通过）
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) —— 37 条裁决记录

六个陷阱速览（机制均已在 `v2_compositional/verify_six_traps.py` 中复现）：

| # | 陷阱 | 关键证据 |
|---|---|---|
| 1 | 熵是个骗子 | 熵 −0.023，硬份额 **×35** |
| 2 | detach 挡不住共享参数污染 | teacher 被 detach 仍被污染 |
| 3 | 延迟引信 | 预测 7568 步，实测 7400（差 2%） |
| 4 | 别扔掉你的 token | Δ 随丢弃率单调增（5 档全 ↗） |
| 5 | 你的任务可能不需要 MoE | eff 精确跟随 n_comp |
| 6 | 帐篷函数的死亡陷阱 | cos=0.95 时移动 **0.0000** vs 0.1500 |

---

## 已知局限（诚实说明）

1. **v1.0 的 "a fraction of the parameters of a dense baseline" 尚无 dense 对照数据。**
   v2.0 已实现对照脚本，结果待补。
2. **v1.0 三处数字存在口径分歧**（基线 9.9787 vs 10.0157、擦除 18.1251 vs 22.83、
   终身学习 +0.0098 vs +1.8/−0.57）。README 采用了数值更好的一侧，未做口径统一。
3. **v2.0 的 FLOPs 0.664x 是理论计算**，未经 wall-clock 实测。当前 DCU 算力利用率
   仅约 7.4%（瓶颈在 kernel launch / sync），理论优势未必兑现为实际加速。
4. **v2.0 的两个陷阱（2、4）幅度仍为单 seed**，机制已验证但具体数值待多 seed 确认。

---

## Environment

Python 3.10+ · torch >= 2.1.0 · numpy · scikit-learn · matplotlib · scipy · datasets

---

## BibTeX

```bibtex
@software{snowflake_moe,
  author = {Wu, Shangzhen},
  title = {Snowflake MoE: Compositional Sparse MoE with Erasable Lifelong Memory},
  year = {2026},
  url = {https://github.com/wszap/snowflake-moe},
  license = {Apache-2.0}
}
```

## Author

Wu Shangzhen (Independent Researcher)

## License

Apache License 2.0. See [LICENSE](LICENSE).
