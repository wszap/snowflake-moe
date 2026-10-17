# snowflake-moe 仓库整合方案

> 基于仓库当前 main 分支（18 commits，最新 75ea953，Oct 6 2026）
> 整合目标：把 v2.0（lock6.x）成果并入，同时修掉 v1.0 README 的硬伤

---

## 一、现状盘点

### 仓库已有（v1.0）

| 类别 | 内容 |
|---|---|
| 模型 | CellMoE 1,823,077 params，**词级** TinyStories，PPL ~10.01 |
| 核心故事 | 忆点 = **memory-value slot**（值槽），与路由**分离** |
| 实验 | 擦除 0/50/100%、终身学习、OOD、5 阶段训练、MNIST 路由消融 |
| 工程 | 18 个脚本统一 ckpt 保存规范，`.gitignore` 排除 checkpoints |
| 文档 | `report.md`、`CONFIG_NOTES`、`docs/`（含 blog） |
| 工具 | `tools/` 离线合成数据生成器 |

### 本地待整合（v2.0 lock6.x）

| 内容 | 状态 |
|---|---|
| 架构 | SnowflakeB 低秩拼专家，**忆点即连接**，N=384 连续组合空间 |
| 主表 | 5 条假设全 PASS，含 **10 seeds** 统计 |
| 负结果 | 六个陷阱（机制全部验证通过）+ `NEGATIVE_RESULTS.md` |
| 博客 | `博客_MoE的六个陷阱.md` |
| 实验日志 | `EXPERIMENT_LOG.md`（37 条裁决） |
| dense baseline | 脚本已就绪，待跑 |

---

## 二、P0 硬伤（必须修，否则会被审稿人/读者抓）

### 🔴 1. 忆点定义自相矛盾

| | 仓库 README (v1.0) | v2.0 (lock6.x) |
|---|---|---|
| 定义 | memory-value slot，**与路由分离** | **忆点就是连接本身** |
| 机制 | 可擦除的值槽 | 高斯带通 + 稳态 + 参数合成 |

**同一术语两种互斥定义。** 必须版本切分。

**⚠️ 我此前说"README 的擦除 claim 是假的"——这句话不准确，需修正：**

v1.0 架构下忆点与路由**分离**，擦除数据**内部自洽**（0% → 50% → 100% 单调恶化）：

| 擦除 | val_ce | val_ppl |
|---|---|---|
| 0% | 2.3041 | 10.0148 |
| 50% | 2.3151 | 10.1261 |
| 100% | 2.8973 | 18.1251 |

**这是 v1.0 架构下的真实数据，成立。** 我 v2.0 发现的"清零后权重上升 6 倍"是**新架构**的性质——因为忆点变成了连接本体，与路由耦合。两者不矛盾，但**必须标明适用版本**。

### 🔴 2. "a fraction of the parameters of a dense baseline" 无数据

README 两处提到参数效率，但**全仓库没有一个 dense baseline**：

- Core Contributions: "achieving strong generalization under **a fraction of the parameters of a dense baseline**"
- Parameter Efficiency 表：`snowflake_v3` active 203,713 / fixed 259,009 / ratio **0.7865**

**注意**：0.7865 是 active/fixed（稀疏激活比），**不是 vs dense**。把它放在 "Parameter Efficiency" 标题下会误导读者以为对比过 dense。

**已解决**：`dense_baseline.py` + `compare_dense.py` 已写好，待跑。

### 🔴 3. 复现命令是坏的

```bash
python run_lm.py   # ❌ 文件列表里根本没有 run_lm.py
```

实际有的是 `train_lm.py` 和 `run_experiments.py`。**新用户第一条命令就失败。**

### 🟠 4. OOD 表的 ratio < 1 需要解释

| Model | id_ppl | ood_ppl | ratio |
|---|---|---|---|
| CellMoE v3 | 15.0317 | 13.9253 | **0.9264** |
| Fixed MoE | 26.5113 | 18.1065 | **0.6830** |

OOD PPL 比 ID 还低——**两个模型都是**，说明 **OOD 数据集本身比 ID 简单**（不是模型的问题）。

**正确的 claim 不是"泛化好"，而是**：

> CellMoE 的 OOD/ID 比值（0.9264）远高于 Fixed MoE（0.6830），
> 说明在同等分布偏移下 CellMoE 的**相对退化更小**。

建议把标题从 "OOD Generalization" 改为 **"OOD Robustness (relative degradation)"**，并加一句说明 OOD 集更简单。

### 🟠 5. 数字挑选嫌疑

| 冲突项 | 值 A | 值 B | README 采用 |
|---|---|---|---|
| 基线 PPL | 9.9787 | 10.0157 | **10.0157** |
| 100% 擦除 | 18.1251 | 22.83 | **18.1251** |
| 终身学习 | +0.0098 | +1.8 / −0.57 | **+0.0098** |

**三次都选了数值更好的一侧。** 若源于 tokenization / 切分口径差异，未经统一的数字不应直接发布。

**建议**：在 README 加一句口径说明，或补一张"不同口径对照表"。

---

## 三、版本策略（关键决策）

**推荐：双轨并存，v1.0 冻结 + v2.0 独立目录**

理由：两个架构的核心定义互斥，混在一起两边都站不住。

```
v1.0（冻结，不再改架构）
    忆点 = 值槽，可擦除、可终身学习
    PPL 10.0157 / 擦除 / OOD 表
    → 只修文档硬伤，不动代码

v2.0（新增目录）
    忆点 = 连接本体，参数空间合成
    N=384 连续组合空间，eff≈26 自发组合
    → 全部新成果放这里
```

**不推荐**的做法：把 v2.0 的结论直接改到 v1.0 的 README 里（会造成术语混乱）。

---

## 四、目录结构建议

```
snowflake-moe/
├── README.md                    # 修改为：总览 + 双版本导航
├── report.md                    # v1.0 报告（保留）
├── CONFIG_NOTES                 # 保留
│
├── v1_memory_points/            # ← 新建，把 v1.0 代码归档进去
│   ├── README.md                # v1.0 专属文档（从主 README 拆出）
│   ├── snowflake_moe.py
│   ├── run_cellmoe_ckpt.py
│   ├── run_erase_*.py
│   ├── run_lifelong_*.py
│   └── ...（其余 v1.0 脚本）
│
├── v2_compositional/            # ← 新建，v2.0 全部内容
│   ├── README.md                # v2.0 专属文档
│   ├── RESULTS.md               # ★ 论文主表（5 条假设全 PASS）
│   ├── snowflake_B.py
│   ├── train_lock60.py
│   ├── run_metrics.py
│   ├── experiment_logger.py
│   ├── dense_baseline.py        # ★ dense 对照
│   ├── compare_dense.py         # ★ 配对统计检验
│   ├── verify_six_traps.py      # ★ 六陷阱验证（可复跑）
│   └── output/                  # 10 seeds 证据
│
├── docs/
│   ├── blog_moe_six_traps.md    # ★ 中文博客
│   ├── NEGATIVE_RESULTS.md      # ★ 13 条负结果（独立研究强可信度信号）
│   └── EXPERIMENT_LOG.md        # ★ 37 条裁决
│
├── tools/                       # 保留
├── early_experiments/           # 加 README 说明"历史存档，不保证可跑"
└── checkpoints/                 # 已 gitignore
```

---

## 五、分步操作清单

### 阶段 1：修 v1.0 文档硬伤（不涉及代码，风险最低）

- [ ] 修 `python run_lm.py` → `python run_experiments.py`（或 `train_lm.py`）
- [ ] OOD 表改标题 + 加"OOD 集更简单"的说明
- [ ] Parameter Efficiency 表改标题（active/fixed，非 vs dense）
- [ ] 加忆点定义的**版本标注**

### 阶段 2：跑 dense baseline（补齐 P0 证据）

```bash
# ① 规模扫描，找 loss=1.788 的临界参数量
python dense_baseline.py --data <TinyStories> --scale-sweep --max-steps 3000

# ② 临界配置跑 10 seeds
python dense_baseline.py --data <TinyStories> --L 5 --d 288 --sweep

# ③ 配对检验 + claim 自动判定
python compare_dense.py --dense output/dense
```

判据：
```
临界 dense 参数 / 5,163,618 < 1  → "a fraction" 成立
                            > 1  → 必须改 README
```

### 阶段 3：归档 v1.0 + 新建 v2.0 目录

```bash
git mv snowflake_moe.py v1_memory_points/
git mv run_cellmoe_*.py run_erase_*.py run_lifelong_*.py run_ood_*.py v1_memory_points/
git mv run_stage*.py run_experiments.py v1_memory_points/
mkdir -p v2_compositional
# 拷入 v2.0 全部文件
```

### 阶段 4：重写主 README

结构建议：
1. **一句话定位**（双版本）
2. **版本导航表**（v1.0 vs v2.0 的对比，一眼看清区别）
3. 各版本核心结果（各自独立，不混用数字）
4. 快速开始（分版本给命令）
5. 引用

### 阶段 5：文档整合

- [ ] 博客放 `docs/blog_moe_six_traps.md`
- [ ] `NEGATIVE_RESULTS.md` 放 `docs/`
- [ ] `EXPERIMENT_LOG.md` 放 `docs/`
- [ ] `early_experiments/` 加 README 说明

---

## 六、参数量对照（易混淆，务必写清）

| 模型 | 参数量 | 任务 | 备注 |
|---|---|---|---|
| **v1.0 CellMoE** | **1,823,077** | 词级 TinyStories | README 现状 |
| **v2.0 SnowflakeB tiny** | **5,163,618** | 字符级 TinyStories(vocab=98) | 本次实验 |
| dense 对照（待跑） | ~5,024,736 | 同上 | L=5 d=288, 0.973x |

**这两者不可比**：
- 词级 vs 字符级（PPL 10.01 vs 5.95 完全不同量级）
- 架构不同（值槽 vs 连接本体）

**README 里必须分开列，且明确标注口径。**

### v2.0 tiny 参数量逐项（之前一直算错，已修正）

| 组件 | 计算 | 参数量 |
|---|---|---|
| embed | 98×128 | 12,544 |
| in_proj | 128×128 | 16,384 |
| head | 128×98+98 | 12,642 |
| **每个 SnowflakeB** | | **320,128** |
| └ memory | 384×128 | 49,152 |
| └ W1_U | 384×128×4 | **196,608** ← 最大头 |
| └ W2_U | 384×32×4 | 49,152 |
| └ 其余 | | 25,216 |
| 细胞层 | 16 × 320,128 | 5,122,048 |
| **合计** | | **5,163,618** |

---

## 七、发布前检查清单

- [ ] `python run_lm.py` 已修正
- [ ] OOD 表已加说明
- [ ] Parameter Efficiency 标题已改
- [ ] 忆点定义已标注版本
- [ ] dense baseline 已跑完，claim 已判定
- [ ] v1.0 / v2.0 目录已分离
- [ ] 主 README 已重写为双版本导航
- [ ] 所有数字都标注了口径（词级/字符级）
- [ ] `early_experiments/` 有说明
- [ ] `output/` 无超大文件（检查体积）
