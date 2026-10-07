# MARVIS 执行任务书 — snowflake-moe 仓库整改

> 目标仓库：`https://github.com/wszap/snowflake-moe`
> 当前 main：`75ea953`（Oct 6, 2026）
> 任务：把 v2.0（lock6.x）成果并入，并修掉 v1.0 README 的五处硬伤

---

## 0. 前置（必做）

```bash
git clone https://github.com/wszap/snowflake-moe.git
cd snowflake-moe
git checkout -b restructure/v2-integration
export REPO_ROOT=$(pwd)
```

**安全**：所有改动都在新分支。主脚本会自动打 `pre-restructure-*` tag 快照。

---

## 1. 环境检查

```bash
bash restructure_repo.sh --check
```

预期输出 5 条 `⚠️`（坏命令 / OOD 标题 / Param Efficiency 标题 / dense claim / 忆点定义）。

---

## 2. 阶段 1 — 修 v1.0 文档硬伤（纯文本，零风险）

```bash
bash restructure_repo.sh --phase 1
```

自动完成 5 处修正：

| # | 修正 |
|---|---|
| ① | `python run_lm.py` → `run_experiments.py` / `train_lm.py`（**原文件不存在，新用户第一条命令就失败**） |
| ② | `### OOD Generalization` → `### OOD Robustness — Relative Degradation` + 加 ratio<1 说明 |
| ③ | `### Parameter Efficiency` → `### Sparse Activation Ratio` + 注明是 active/fixed 非 vs dense |
| ④ | "a fraction of the parameters of a dense baseline" 加「尚无 dense 数据」标注 |
| ⑤ | 忆点定义加版本标注（v1.0 值槽 vs v2.0 连接本体） |

---

## 3. 阶段 2 — 归档 v1.0 + 建目录

```bash
bash restructure_repo.sh --phase 2
```

- 33 个 v1.0 脚本 → `v1_memory_points/`（用 `git mv`，保留历史）
- `report.md`、`CONFIG_NOTES` → `v1_memory_points/`
- 建 `v2_compositional/`、`docs/`
- `early_experiments/README.md` 加「历史存档，不保证可跑」说明

---

## 4. 阶段 3 — 文件投放

从交付包 `snowflake_open.zip` 解压后，按此映射投放：

### 4.1 主 README

```bash
cp README_main.md README.md
```

### 4.2 v1.0 文档

```bash
cp README_v1.md v1_memory_points/README.md
```

### 4.3 v2.0 目录（`v2_compositional/`）

**代码**
```
snowflake_B.py
train_lock60.py
run_metrics.py
experiment_logger.py
run_smoke_gpu.py
dense_baseline.py          ★ dense 对照
compare_dense.py           ★ 配对统计检验
verify_six_traps.py        ★ 六陷阱验证（可复跑）
probe_data.py
snowflake_lowrank.py
snowflake_spec.py
verify_min_eff.py
verify_lock60.py
verify_lowrank.py
permute_check.py
permute_sensitivity.py
ablate_permute.py
temp_scan.py
scale_plan.py
spec_check.py
```

**文档**
```
RESULTS.md          → v2_compositional/RESULTS.md      ★ 论文主表
README_v2.md        → v2_compositional/README.md
RUN.md              → v2_compositional/RUN.md
```

### 4.4 docs/

```
博客_MoE的六个陷阱.md → docs/blog_moe_six_traps.md
NEGATIVE_RESULTS.md   → docs/NEGATIVE_RESULTS.md
EXPERIMENT_LOG.md     → docs/EXPERIMENT_LOG.md
MoE六个失败模式.md     → docs/MoE_six_failure_modes.md   （报告版，表格密集）
```

### 4.5 证据目录（可选，检查体积后决定）

远端 `output/runs/` 10 seeds 的 `record.json` + `steps.csv`
→ `v2_compositional/output/runs/`

```bash
du -sh output/    # 超过 50MB 建议不入库，改用 .gitignore + 说明
```

---

## 5. 阶段 4 — 跑 dense baseline（补齐 P0 证据）

**这是 v1.0 README 唯一还没被证据支撑的 claim。**

```bash
cd v2_compositional

# ① 规模扫描：找 dense 达到 Snowflake loss=1.788 的临界参数量
python dense_baseline.py --data <TinyStories.txt> --scale-sweep --max-steps 3000

# ② 临界配置跑 10 seeds
python dense_baseline.py --data <TinyStories.txt> --L 5 --d 288 --sweep

# ③ 配对检验 + claim 自动判定
python compare_dense.py --dense output/dense
```

**判据**（`compare_dense.py` 会自动打印）：

```
临界 dense 参数 / 5,163,618 < 1  → "a fraction" 成立，README 保持
                            > 1  → 必须改，脚本会给出替代表述：
                                   "achieves comparable perplexity
                                    with X.XXx the parameters of a dense baseline"
```

**关键配置**（已校准，勿改）：
- dense 默认 `L=5, d=288, ff=1152` → 5,024,736（**0.973x** Snowflake 5,163,618）
- **必须不加 pos embedding** —— SnowflakeLM 也没有；加了参数量虚高 9.9%，对比不公平
- 数据用 `--fix-mojibake`，与 Snowflake 侧一致

---

## 6. 验收清单

- [ ] `bash restructure_repo.sh --check` 五项全绿
- [ ] `README.md` 已替换为双版本导航版
- [ ] `v1_memory_points/` 含 33 个脚本 + README_v1.md
- [ ] `v2_compositional/` 含全部代码 + RESULTS.md
- [ ] `docs/` 含博客 + NEGATIVE_RESULTS + EXPERIMENT_LOG
- [ ] `early_experiments/README.md` 有说明
- [ ] dense baseline 已跑，`compare_dense.py` 输出 claim 判定
- [ ] 所有数字标注了口径（词级 / 字符级）
- [ ] `du -sh output/` 检查体积，超大则 gitignore
- [ ] `git status` 无意外文件

---

## 7. 提交

```bash
git add -A
git commit -m "restructure: split v1.0 (memory points) and v2.0 (compositional)

- Fix broken repro command run_lm.py -> run_experiments.py
- Rename OOD Generalization -> OOD Robustness (ratio<1 explained)
- Rename Parameter Efficiency -> Sparse Activation Ratio (active/fixed, not vs dense)
- Add version labels to memory-point definition (v1.0 slot vs v2.0 connection)
- Archive v1.0 scripts to v1_memory_points/
- Add v2_compositional/ with 10-seed results
- Add docs: blog (six traps), negative results, experiment log"

git push origin restructure/v2-integration
# 然后在 GitHub 开 PR，检查无误后合并到 main
```

---

## 8. 回滚

```bash
git tag                          # 找 pre-restructure-*
git reset --hard pre-restructure-XXXXXXXX
```

---

## 9. 注意事项（给执行者）

1. **不要**把 v2.0 的结论改到 v1.0 README 里 —— 两个架构核心定义互斥。
2. **参数量别搞混**：
   - v1.0 CellMoE = **1,823,077**（词级）
   - v2.0 SnowflakeB tiny = **5,163,618**（字符级）
   - 早期文档的「3.6M」是**错的**
3. **PPL 不可比**：词级 10.0157 vs 字符级 ≈5.95，差近一倍量级。
4. **dense 不要加 pos embedding**，否则对比不公平。
5. 阶段 1 可单独先推（零风险），dense baseline 跑完再决定 README 那句 claim。
