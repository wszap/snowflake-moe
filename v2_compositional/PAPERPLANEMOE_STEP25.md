# paperplaneMoe / step25 —— 回到「拼专家」主线的下一步改动

## 0. 这次的代码来源（按「根据最新的仓库」执行）

- `github.com/wszap/snowflake-moe` 网页与 `git ls-remote` 返回 **403**（未登录会话）；
  但 `codeload.github.com` 的 tarball 与 `api.github.com` 的 tree/commit 接口可公开读。
- 已拉取 **最新 main = `0934ab31`**（2026-11-07T12:59:10Z，commit message 已是
  `paperplaneMoe: flipper 3-seed lossless + clip-grad inert dummy attribution`），
  解压到工作副本 `/data/workspace/paperplaneMoe_latest/`（482 个 blob，其中 288 个在
  `v2_compositional/`）。**所有改动都在这份最新副本上做，不是凭记忆写。**
- ⚠️ 我**不能 push 到你的 GitHub 仓库**（无写凭据）。交付的是下面 6 个新文件 +
  `install_step25.sh`，在 DSW 上 `git pull` 后放进 `v2_compositional/` 即可；
  这一步由你执行，不是我「已经改好线上仓库」。

## 1. 真读过的文件（不是从文件名反推）

`composable_expert.py`、`snowflake_B.py`、`snowflake_v4.py`、`train_split.py`、
`train_v3.py`、`scaffold_v3.py`（前 120 行）、`organelle_registry.py`、
`organelle_groups.py`、`verify_composable_expert.py`、`step7_compose_vs_select.py`、
`RESULTS.md`、`v2_compositional/README.md`，以及 GitHub API 的完整 tree。

## 2. 查出来的关键事实：主训练根本没走「真拼专家」

| | `snowflake_B.py`（B5 / v4 全部实验在用） | `composable_expert.py`（v8，仓库里已有，未接入） |
|---|---|---|
| 前向 | `U1 = Σ wᵢ W1_U[i]` → **一个共享 silu** | `Σ wᵢ · silu(x E1ᵢ) · E2ᵢ` → **每个专家各自非线性** |
| 容量 | 秩 ≤ `rank=4`，**拼 k 个 = 拼 1 个**（文件头自己写「32 人挤 4 人道」） | 恒等式 `Σ wᵢσ(xE1ᵢ)E2ᵢ ≡ [σ(xE1₁),…] @ blkdiag(wᵢE2ᵢ)`，**等效隐层宽度 = k×h_cell，线性增长** |
| 证据 | `step7_compose_vs_select.py` Q1/Q3 | `verify_composable_expert.py` 用 numpy 证明 Jacobian 秩随 k 从 1→32 增长 |

`train_split.py:8` import 的是 `snowflake_v4.SnowflakeLMV4`，其 `make_cell()` 返回
`SnowflakeB(**kw)`；`composable_expert.py` 只被单元验证脚本引用过，**从未进过主训练循环**。

> 所以「结合仓库里所有代码、一定走拼专家路线」的下一步不是再调 wiring / 再补一个模块，
> 而是**把仓库里已经写好、已被数学验证的真拼核心接进主训练** —— 这是架构级改动，
> 也是目前最可能产生足够大效应量、把 `p≈0.08` 推进到显著的一刀。

## 3. 同时落地的既有实测结论（不另起炉灶）

| 结论 | 出处 | 本步怎么落 |
|---|---|---|
| 全局 `clip_grad_norm_(model.parameters(),1.0)` 因新增参数一致有害（`D0−F0` 3/3 > 0） | `step23` / `flip_attrib.log` | `--clip-mode grouped`：`layers.*` 与主干各自独立裁剪，主干 `clip_coef` 不再被专家库规模绑定 |
| 新细胞走正锥份额、不抢 softmax 名额 | B5 `E3−E1 = +0.1780` loss 净扰动 | 沿用 `_split_n0` 截断 + `_split_child_w` 追加，`expand()` 里 E1/E2 新行零初始化 |
| 不复制母体已学知识、旧 c 块冻结 | `train_split.do_split` / registry | `ComposableExpertV10.expand()` 逐位照搬 |
| 全 n 权重恒非零（拼不是选），eff/var_norm 监控 | `RESULTS.md` H2 / `organelle_groups.py` | 诊断一律用**全 n** wiring；top-k 只作计算稀疏 |
| 翻盖（flipper）当前等于没用 | 三 seed + inert dummy 归因 | **不接** `flipper`；`variant` 保持 `shared` |
| H2「按输入组合」显著 FAIL（`var_norm=0.0405`） | `RESULTS.md` | 提供 `--wiring-mod {none,norm,center}` 作为第二批实验变量，本批先固定 `none` 不混变量 |

## 4. 参数量对账（d=256, h=128, n=384, L=4, cells=4, vocab=100）

每个 cell 的可训练参数：

```
B 型   = n·d + n·d·r + r·h + h·d + n·h·r + r·d + d·h + d²
       = 98,304 + 393,216 + 512 + 32,768 + 196,608 + 1,024 + 32,768 + 65,536
       = 820,736      ×16 = 13,131,776  → 全模型 ≈ 17,225,573（与 B5 实测一致）

v10    = n·d + n·d·h_cell + n·h_cell·d + d·h + h·d + d + d² + 1
h_cell=3: 98,304 + 294,912 + 294,912 + 32,768 + 32,768 + 256 + 65,536 + 1
       = 819,457      ×16 = 13,111,312  → 全模型 ≈ 17,205,109
```

**`h_cell=3, k=16` 与 B 档参数量差约 −20,464（−0.12%），可视为对齐。**
`composed_width = k·h_cell = 48`（B 型共享隐层是 128，但秩被 `rank=4` 锁死）。
可选容量档：`h_cell=4` → +18% 参数、`h_cell=8` → 约 +100%，用于排除「赢在容量」的替代解释。

## 5. 交付文件

| 文件 | 作用 |
|---|---|
| `compose_v10.py` | 真拼 cell：继承 `ComposableExpert`，补 `_split_c/_split_n0/_split_child_w/expand()`，接口与 `SnowflakeB` 一致 |
| `train_compose.py` | 主训练：**零改动现有源码**，模块级替换 `snowflake_v4.SnowflakeB`；复用 `train_v3.load_data/evaluate`、`train_split.do_split`（B 档）、registry/soft groups；新增分组裁剪与扩展落盘字段 |
| `step25_compose_ab.sh` | CRN 配对三档 × seed 0/1/2（可扩到 8） |
| `step25_compose_summary.py` | 配对 t + 三族 Holm + CUPED（探索性）+ Cohen's d |
| `PAPERPLANEMOE_STEP25.md` | 本文件 |
| `install_step25.sh` | 落地脚本 |

## 6. 实验矩阵与预注册判据

```
seed s ∈ {0,1,2}（效应量确认后扩到 0..7）
  A  st25_s{S}_B       cell=B    clip=global    历史基线复现
  C  st25_s{S}_V10w48  cell=v10  clip=global    h_cell=3 k=16
  D  st25_s{S}_V10iso  cell=v10  clip=grouped   h_cell=3 k=16  ★主处理档
```

**三族比较**（同 seed 配对）：`C−A`（架构）、`D−A`（架构+裁剪）、`D−C`（裁剪净效应）。

**主判据**：配对 t（df=n−1），Holm 校正 α=0.05，且至少 3/3 同号；同时必须报告
`params_start` / `composed_width` / `eff` / `var_norm` / `elapsed_sec`。

- 若 `D−A` Holm PASS 且参数量对齐 → 真拼专家 + 分组裁剪的收益可写进主结论。
- 若只有 `C−A` PASS → 收益来自架构，裁剪无关，如实拆开写。
- 若三族都 fail → 记负结果：真拼扩容在 3000 步 / 该数据上未兑现，回查 `gate` 是否学起来、
  `composed_width=48` 是否太小、`temp=5.0` 是否压死了按输入组合。
- **不写「加速」**：到达步数需用 history 插值另验，且前面五个阈值四个 Δ<1.3 步。

**显著的前提是效应量**：n=8 时需 `|d| ≳ 0.84` 才能进 p<0.05。换核心 cell 是架构级改动，
效应量有机会远大于翻盖的 0.0168；但**这不是保证**，先跑 seed0 三档看量级再决定扩不扩。

## 7. 在 DSW 上落地

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional
git pull --ff-only                     # 应到 0934ab31
# 把补丁包里的 6 个文件放到本目录，然后：
bash install_step25.sh

# ① 20 步 smoke（先看 v10 链路）
python3 train_compose.py --data data_25mb.txt --d 256 --h 128 --L 4 \
  --n-cells 4 --variant shared --rank 4 --temp 5.0 --bs 8 --seq-len 128 \
  --lr 3e-4 --max-steps 20 --split-k 8 --split-at 10 --seed 0 \
  --cell-type v10 --h-cell 3 --compose-k 16 --clip-mode grouped \
  --tag st25_smoke > st25_smoke.log 2>&1
echo "smoke exit=$?"; grep -E "\[split\]|\[saved\]|params " st25_smoke.log | tail -4

# ② 三档 × 3 seed
nohup bash step25_compose_ab.sh > step25.log 2>&1 & echo "PID=$!"

# ③ 判据
python3 step25_compose_summary.py 2>&1 | tee step25_summary.log
```

`smoke` 验收线：`exit=0` + `[split] step 10 384→392` + `[saved]` + `params` 与对账表一致
（v10 约 17,205,109 → 分裂后 +16×K×h_cell×d×2 ≈ +98,304）。

## 8. 诚实边界

- 沙盒**没有 torch**，所以本轮验证的是：`ast.parse` 全部 Python、`bash -n` 全部 shell、
  numpy 等价代数（真拼恒等式 / 零门 / 正锥份额全非零 / 分组裁剪系数 / CUPED 公式）、
  以及用模拟 json 实跑 `step25_compose_summary.py`。**真 smoke 与正式训练必须在 DSW 上跑**，
  我不把「预期输出」写成事实。
- 现有源码一个字节没改（全是新文件），历史 json / 日志 / 结论不受影响。
- `h_cell=3, k=16` 时一次性 gather 的 `need ≈ 384 MB > mem_budget_mb=256`，
  `compose_v10` 会自动走逐 slot 低显存路径（16 次 bmm）。**v10 单档 wall-clock
  预计明显慢于 B 档**，`elapsed_sec` 必须如实报；本批只比 val，不写「更高效」。
- GitHub 网页那 6 条解析失败（首条标题 `Sign in to GitHub`、末条 URL 尾巴 `:String`）
  是未登录会话与解析服务拼接错误，与仓库无关；远端状态以 `git ls-remote` 为准，PR 不用开。
- 名字：对外与 commit message 用 **paperplaneMoe**；目录/类名/import 一律不动，
  避免这十几轮所有 `cd` 路径、json 路径、checkpoint key 对不上。批量改名仍等你一句话。
