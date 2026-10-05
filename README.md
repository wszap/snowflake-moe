# Snowflake MoE

A lightweight Sparse Mixture-of-Experts (MoE) language model with compositional experts, memory points, and erasable lifelong learning, built from scratch in PyTorch.

## Project Intro

Snowflake MoE is a from-scratch implementation of a sparse MoE architecture that combines fine-grained expert routing with a shared-expert "memory point" mechanism, enabling **compositional generalization**, **targeted memory erasure**, and **lifelong learning** in a compact parameter budget.

## Core Contributions

1. **Compositional MoE (组合 MoE)**: Fine-grained experts + shared experts + sparse activation routing. The router learns to compose reusable expert combinations, achieving strong generalization under a fraction of the parameters of a dense baseline.
2. **Memory Points (忆点)**: A dedicated memory-value slot mechanism that stores decoupled factual knowledge, separate from routing weights.
3. **Erasable Memory (可擦除)**: Memory values can be selectively erased at inference time. Erasing 50% of memory slots causes only mild PPL degradation, while erasing 100% degrades the model significantly — demonstrating that knowledge is localized and erasable on demand.
4. **Lifelong Learning (终身学习)**: New domains can be added with negligible regression on previously learned domains.

## Environment Dependencies

- Python 3.10+
- torch >= 2.1.0 (CUDA recommended)
- numpy
- scikit-learn
- matplotlib
- scipy
- datasets

Install with:

```bash
pip install -r requirements.txt
```

## Reproduction Commands

Main experiments (run from the repository root):

```bash
# TinyStories baseline (CellMoE, ckpt) — PPL ~10.01
python run_cellmoe_ckpt.py

# TinyStories baseline (CellMoE, MR2, fixed) — PPL ~10.02
python run_cellmoe_mr2.py

# Memory erasure study (erase 0% / 50% / 100% of memory slots)
python run_erase_tinystories.py
python run_erase_mr2.py

# Lifelong learning: TinyStories + Bible, check regression
python run_lifelong_tinystories.py

# Out-of-distribution generalization (ID vs OOD PPL)
python run_ood_v2.py

# Stage-wise training pipeline (stages 1-5)
python run_stage1.py
python run_stage2.py
python run_stage3.py
python run_stage4.py
python run_stage5.py

# MNIST routing-behavior ablation (fixed / rules / learned)
python train_mnist.py

# LM scaling & ablation sweep
python run_lm.py  # via run_experiments.py
```

## Checkpoint 保存规范

所有 `run_*.py` 训练脚本统一按以下规范保存模型，**只加保存、不改变训练逻辑**。

**保存位置**：仓库根目录 `checkpoints/`（该目录已被 `.gitignore` 排除，模型文件**不会** push 到 GitHub，只保留本地；代码与 CSV 正常提交）。

**命名规则**：`checkpoints/{实验名}_{seed}.pt`，seed 取脚本内 `SEED` 常量（默认 2026）。

| 脚本 | 保存内容 | 命名 |
|---|---|---|
| `run_cellmoe_ckpt.py` | 每个 epoch 结束保存一次，最终 epoch 覆盖为最终版 | `cellmoe_tinystories_epoch{N}.pt` / `cellmoe_tinystories.pt` |
| `run_cellmoe_v2.py` | 最终模型 | `cellmoe_v2_{seed}.pt` |
| `run_cellmoe_mr2.py` | 最终模型 | `cellmoe_mr2_{seed}.pt` |
| `run_stage1.py` | 最终模型 | `improved_v3_{seed}.pt` |
| `run_stage2.py` | 最终模型 | `stage2_{seed}.pt` |
| `run_stage3.py` | 持续学习后模型 | `improved_v3_lifelong_{seed}.pt` |
| `run_stage4.py` | 免疫组 A / 对照组 B | `stage4_immune_{seed}.pt` / `stage4_control_{seed}.pt` |
| `run_stage5.py` | CellMoE / Fixed 最终模型 | `stage5_cellmoe_{seed}.pt` / `stage5_fixed_{seed}.pt` |
| `run_stage5_fast.py` | 最终模型 | `stage5_fast_{seed}.pt` |
| `run_stage5_fixed.py` | 最终模型 | `stage5_fixed_{seed}.pt` |
| `run_lifelong.py` | 持续学习后模型 | `v3_lifelong_{seed}.pt` |
| `run_lifelong_tinystories.py` | 持续学习后模型 | `cellmoe_lifelong_{seed}.pt` |
| `run_ood.py` | CellMoE / Fixed 最终模型 | `ood_cellmoe_{seed}.pt` / `ood_fixed_{seed}.pt` |
| `run_ood_v2.py` | CellMoE / Fixed 最终模型 | `ood_v2_cellmoe_{seed}.pt` / `ood_v2_fixed_{seed}.pt` |
| `run_erase.py` | 100% 擦除并恢复后模型 | `erase_{seed}.pt` |
| `run_erase_mr2.py` | 擦除后模型（50%/100% 覆盖保存） | `erase_mr2_{seed}.pt` |
| `run_erase_tinystories.py` | 擦除后模型（50%/100% 覆盖保存） | `erase_tinystories_{seed}.pt` |
| `run_experiments.py` | 每个 (seed, mode) 组合最终模型 | `experiments_{mode}_{seed}.pt` |

保存内容统一为 `torch.save` 的 dict：`state_dict` + 训练配置（`cfg`/`config`）+ 关键指标（如 `final_ppl`），便于后续加载复现。

## Core Results

### PPL Comparison (TinyStories, 5 epochs)

| Model | Params | val_ce | val_ppl | Note |
|---|---|---|---|---|
| CellMoE (ckpt run) | 1,823,077 | 2.3042 | **10.0157** | baseline |
| CellMoE (MR2 fixed) | 1,823,077 | 2.3043 | **10.0169** | fixed run |

### Memory Erasure (TinyStories)

| Erase Fraction | val_ce | val_ppl |
|---|---|---|
| 0% (baseline) | 2.3041 | 10.0148 |
| 50% (256 slots) | 2.3151 | 10.1261 |
| 100% (512 slots) | 2.8973 | 18.1251 |

### Lifelong Learning (TinyStories + Bible)

| Stage | ts_base_ppl | ts_life_ppl | regression | bible_base_ppl | bible_life_ppl |
|---|---|---|---|---|---|
| lifelong | 10.0148 | 10.0246 | +0.0098 | 19.4129 | 18.2200 |

### OOD Generalization (Shakespeare → OOD)

| Model | id_ppl | ood_ppl | ood/id ratio |
|---|---|---|---|
| CellMoE v3 | 15.0317 | 13.9253 | 0.9264 |
| Fixed MoE | 26.5113 | 18.1065 | 0.6830 |

### Parameter Efficiency (Snowflake v3, Shakespeare)

| Mode | Active Params | Fixed Params | Param Ratio | val_ppl (3 seeds) |
|---|---|---|---|---|
| snowflake_v3 | 203,713 | 259,009 | 0.7865 | 13.33–14.39 |

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

**Wu Shangzhen** (Independent Researcher)

## License

Apache License 2.0. See [LICENSE](LICENSE).
