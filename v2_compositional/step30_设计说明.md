# V1.005 设计说明 · 把酶的作用面从闸门挪到主干

## 0 · 先纠正一个我上轮写错的代码事实

上一轮根因报告写「主干 `W1_base/W2_base` 是 shared、不按 cell 切分，
所以 `cat_all` 结构上不可实现」。**逐行核完代码，这是错的**：

```
snowflake_v4.py:225-229
    self.layers = nn.ModuleList([
        nn.ModuleList([ SnowflakeB(...) for ci in range(n_cells) ])
        for li in range(L) ])
train_compose.py:82-83
    def all_cells(model): return [c for layer in model.layers for c in layer]

snowflake_B.py:102-108
    self.W1_base = nn.Parameter(...)      # 每个实例各一份
    self.W1_U / W1_V / W2_base / W2_U / W2_V / organelle_memory 同理
```

`--variant shared` 只是额外加 `self.shared = ModuleList([SharedExpert(...) for _ in range(L)])`
（每层一个共享全秩专家），**不是主干共享**。16 个 cell（L=4 × n_cells=4）本来就各有
一份独立主干参数，总共约 17.2M 里的绝大部分。

**所以 V1.005 不需要改架构、不需要拆主干、不需要重跑历史对照。**
只需要改 optimizer 的分组方式。

## 1 · 真正的根因（精确版）

```
AdamW(model.parameters(), lr=a.lr)          ← 全部参数塞进 1 个 param_group
rebind_groups()  [step29]                    ← 只把 16 个 gate_comp 拆出来
                                                打上 gate_cell 键
observe_step():
    if "gate_cell" in pg:  lr = base·k_i     ← 16 个 gate 标量
    elif mode=="lrx":      lr = base·k̄      ← 全部参数（故意不分组）
    else:                  lr = base        ← 主干 17M 原样
```

16 份主干一直存在，却全留在同一个默认组、用同一个 lr。`k_i` 只落在
16 个 `gate_comp` 上 ⇒ step29e 六档 AUC 全 736.1 逐位相同；唯一变快的
`lrx` 是把 k̄ 乘到全部参数的等效调大学习率。

**结论修正**：不是「屋子里没给酶留岗位」，是「岗位一直在，只是没给它们
各自开门」。`step29f` 的 toy 里 `cat_all−lrx = −7.097 (5/5)` 那一档，
在真模型里其实是可实现的 —— 只要把主干按 cell 拆组。

## 2 · V1.005 改了什么（11 处锚点，全部精确匹配）

| 文件 | 改动 |
|---|---|
| `enzyme_rate.py` | `scope="gate"\|"backbone"`、`warmup` 参数 |
| | `rebind_groups`：backbone 档把每个 cell 的主干参数也拆成 `param_group`，打同一个 `gate_cell` 键 + `enz_scope="backbone"` |
| | `observe_step`：backbone 档 `[S]` = 该 cell 主干参数梯度均方（**剩余未转化量**），不再用 gate 的 `g/H` |
| | 诱导延迟 `step >= warmup` 才更新 e ⇒ **step0 严格 k≡1** |
| | `corr` 按 scope 切换：backbone 看 `corr(k,S)`，gate 保持 `corr(k,1/H)` |
| | trace 增记 `scope / bb_lr / n_bb_groups`（G2 作用面实证，不靠推理背书） |
| `train_bio.py` | CLI `--enz-scope` / `--enz-warmup`；构造 controller 传参；json 记 `enz_scope` |

**前向、参数量、RNG、wiring、E1/E2 buffer 全部一比特不动。**
`observe_step` 里现有的 `if "gate_cell" in pg` 分支会自动给主干组设
`lr = base·k_i` —— 这是最小补丁，不是重写。

AdamW 的 state 按 parameter 对象索引，重新分组不丢动量（step29 已实证）。
`do_split_B` 重建 optimizer 后，原有两处 `rebind_groups` 调用点自动覆盖新分组。

## 3 · 为什么这次符合生物学本意

| 生物学 | V1.005 |
|---|---|
| 酶降 ΔG‡，只改速率常数、不改 Keq / 不动点 | k 乘在 `param_group.lr`（AdamW 里唯一不被 `m̂/√v̂` 洗掉的速率位置） |
| `[S]` = 还没转化的反应物 | 该 cell 主干梯度均方（离不动点还有多远），**不是产物、不是通量** |
| `Km` 归一化比值（Atkinson 能荷式） | `Km = median(S)` |
| 诱导表达改酶量 `[E]ₜ`，慢变量 | `e ← e + (f−e)/τ_E`，`f=S/(Km+S)` |
| 转录→翻译有延迟，酶不是出生就满 | `warmup=1` ⇒ step0 严格 k≡1，之后全自主长起来 |
| 自噬 / 产物抑制 | 长期 f 低 ⇒ e 自动衰减，k→1 |
| 各种各样的酶 | `cat_het` 的 `Km_i` 漂变保留（本轮主实验不进，step29e 已证为阴性对照） |
| **酶和细胞是整体** | 主干真的读酶了：k_i 决定每个细胞走多快，不是可选支路的总闸 |

**没有路由、没有筛选、w 全正、Σwᵢ·silu(xE1ᵢ)E2ᵢ 底线不动。**

## 4 · 档位（单变量：只改 `--enz-scope`）

| tag | mode | scope | 作用 |
|---|---|---|---|
| `st30_A0_B` | off | — | G1 零污染基线 |
| `st30_gate_cat` | cat | gate | step29 现状，阴性对照（应打平） |
| **`st30_bb_cat`** | cat | backbone | ★ 主实验：16 份主干各吃 k_i |
| `st30_bb_const` | const | backbone | 分组但 k_i=k̄（排除「只是分组」） |
| `st30_lrx` | lrx | gate | 不分组、全参数 lr×k̄（旧等效 lr 对照） |
| `st30_bb_freeze` | cat | backbone | FULL 关 rescue：单 cell k≡1，不删专家 |
| `st30_ref*_nopatch` | off | — | 用 `train_bio.py.bak_step30` 跑的同配置逐位基线 |

`--enz-beta-const 0.0` ⇒ k 从 1 起步，无预置分化；所有差异只能来自底物诱导。

## 5 · 三条门禁（G1/G2/G3，跑前注册）

- **G1 零污染**：off 档零 trace；同配置 nopatch 基线 31/301 点逐位全 0。
  配置不同只 SKIP 不判停线；真 301 点留给 FULL `--zero-check`。
- **G2 作用面实证（承重）**：`bb_*` 每步 `n_bb_groups=16` 且 `bb_lr` 真分化；
  `gate_cat/lrx/A0` 为 0 组。**这条不过 = 实验根本没做**，比任何收益都优先。
- **G3 可分辨性**：`bb_cat` 必须同时赢过 `bb_const` 与 `lrx`（T_ε 或 AUC），
  且 `gate_cat−A0` 打平。主指标 T_ε@1.30 / 渐近率 / AUC / 双线成本；
  终态 val 降为门禁。

## 6 · 沙盒验证过的（toy，5 seeds）

```
P1 gate_only 效应 < 主干分化的 10%      PASS   −1.063 vs −70.378
P2 backbone_cat 同时赢 const 与 lrx      PASS   −17.885 (5/5)
P3 k0≡1 / 中段分化 / 活跃窗 corr>0.3    PASS   kmaxstd=1.234 corrA=1.000
P4 分组但不分化 ≡ 等效 lr               PASS   +0.000
```

**toy 是 16-block 确定性 GD，只验逻辑，不外推 DSW 真模型。**

## 7 · 可证伪预测

若 G2 过了而 G3 仍输给等效 lr，就废弃「酶催化 ↔ 拼专家主干 lr」这条映射，
把酶当普通可学习层处理，不再声称与生物同构；**不调 β/Km/τ_E 硬凑**。

## 8 · 边界

沙盒无 torch，真 301 点硬前置与真收益只能在 DSW 出；我也不能 push 到你的 GitHub。
`V10−B=+0.1099`、`B−A=+0.0002`、`C−B≈−0.0003`、`grouped−global=+0.0059`、
`warm−zero=−0.0003`、B5 `p≈0.09` / 翻盖 `p≈0.083` 一字不翻案；
**显著成效未达成，永远不写「加速」**。
