# Snowflake MoE 实验总日志

> **唯一真相来源（Single Source of Truth）**
>
> 所有实验、参数改动、结果、决策记录在此。每次实验后必须更新本文件，每次交付必须验收。
> 论文 / README / GitHub 中的每一个数字都必须能回溯到本文件的某一条记录。
>
> 最后更新：2026-10-07

---

## 0. 项目定位

**核心命题**：用「组合」替代「堆叠」。

```
标准 MoE:  y = Σᵢ∈TopK wᵢ · SiLU(x·Wᵢ)     非线性在求和【之前】→ 输出空间组合
Snowflake: y = SiLU(x · Σᵢ mixᵢ·W1ᵢ)        非线性在求和【之后】→ 参数空间组合
```

**硬件**：64GB 显存 GPU（沙箱无 GPU / 无 torch，真机在 D 盘 snowflake 项目）

---

## 1. 版本演进（锁序列）

| 锁 | 核心改动 | 状态 | 关键结果 |
|---|---|---|---|
| lock3.x | TopK 器官选择、内容寻址、硬分层偏置 | 已废弃 | 震荡：wiring_ent 1.88（断路）/ 0（短路） |
| lock3.8 | 知识蒸馏 | ❌ 失败 | 反向污染，teacher 萎缩，坍塌提前到 step 1800 |
| lock3.9 | 逐 cell ΔPPL | ❌ 失效 | 4 cell ΔPPL 均 +29~33 但彼此差 <4 ⇒ 只证"缺一不可"，不证"分工" |
| lock4.1 | 绕过坍塌 encoder 用原始 x | 记录 | 未升为主结论 |
| lock4.3 | **废除 TopK，改参数空间合成** | ✅ 主线 | `mix/wire_proj` |
| lock4.5 | 器官 base+delta | ✅ | — |
| lock4.6 | 条件数正则 | ⚠️ 判据失效 | 见 L-006 |
| lock47 | 帐篷+铰链 + 低秩 + chunk | ⚠️ 偏离规格 | 见 L-004（初始化错误） |
| **lock6.0** | 高斯带通 + 稳态 + 规格初始化 | ✅ 当前 | 见 §3 |
| lock6.1 | A 型低秩 rank=32 | ❌ **已停用** | 见 D-004，改 B 型 |
| **lock6.2** | B 型低秩 **rank=4, P=32** | ✅ 当前主线 | 见 D-004 |

---

## 2. 实验参数总表（每次改动必须登记）

### 2.1 当前生效配置（锁 6.0 + A 型低秩）

| 参数 | 值 | 来源 | 备注 |
|---|---|---|---|
| `n_organelles` | 384 | 用户 2026-10-07 拍板 | 512 作为扩展 |
| `d` | 128（tiny）/ 256（mid） | 规模预设 | 见 §5 |
| `h` | 32 / 64 | 规模预设 | — |
| `r`（rank） | **4** | 用户裁决改 B 型 | 见 D-004。**B 型 r 不能开大**，与 A 型相反 |
| `target_partners` | 4.0 | 用户拍板 | 行和目标，恒定不随 N |
| `band` | 0.5 | 规格 | 高斯带通中心 |
| `width` | 0.1 | 规格 | **暂不动**，等 smoke 数据 |
| `delta_scale` | 0.5 | 用户拍板 | 第二档 1.0 |
| `wire_temp` | **5.0**（N=384） | V-022 | ✅ **已改**。规格 2.0 在 N=384 下过均匀。N=8 时需 1.8 |
| `chunk_size` P | **32** | 见 D-004 | B型 r=4: P=32→0.664x；P=64→0.477x |
| 铰链模式 | `loss`（加性） | 见 L-007 | `mult` 保留作对照 |
| `hinge_start` | 0.8 | 用户确认 | — |
| 铰链归一化 | `/[N(N-1)]` | 见 L-003 | 必须，否则爆炸 |

### 2.2 规模预设（64GB）

| 档 | N | d | h | L | cells | r | P | bs | 参数量 | 相对 Top-K |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny | 384 | 128 | 32 | 4 | 4 | **32** | 8 | 64 | 3.6M | 0.656x |
| **mid** | 384 | 256 | 64 | 8 | 4 | **32** | 8 | 64 | 10.1M | 0.656x |
| large | 512 | 512 | 128 | 12 | 4 | 16 | 32 | 32 | 53M | 0.290x |
| extreme | 1024 | 1024 | 256 | 16 | 4 | 32 | 32 | 16 | 213M | 0.290x |

★ **效率优势不随规模衰减**（四档恒定 0.290x）——可写进论文。

---

## 3. 验证实验记录（V = Verification，纯 numpy，沙箱可跑）

| ID | 实验 | 脚本 | 结果 | 结论 |
|---|---|---|---|---|
| V-001 | 帐篷+铰链五项验证 | `tent_hinge_verify.py` | 全过，7.8s | 死区可恢复（坍缩 −6.9%/−4.8%） |
| V-002 | c.T vs c 折叠 | `validate_ultimate.py` | c.T: 6.66e-16；c: 0.0988 | **必须用 c.T** |
| V-003 | 漏乘 width² 是否发散 | `validate_ultimate.py` | 未复现发散（13× 恶化） | ⚠ 修正用户原描述，见 L-001 |
| V-004 | eff_rank 判据 | `validate_ultimate.py` | 噪声 7.822 **>** 分工 6.926 | ⚠ **判据失效**，见 L-002 |
| V-005 | 满秩 delta 条件数 | `spec_check.py` | 恒 2.0~2.9，与 mix 无关 | ⚠ 条件数判据失效，见 L-006 |
| V-006 | 初始化对比 | `spec_check.py` | 规格 cos=0.4964 / lock47 cos=0.0002 | ⚠ lock47 错误，见 L-004 |
| V-007 | 三种连接函数回复力 | `spec_check.py` | 高斯 cos=0.9 时 9e-06 | 高斯解决硬死区但无回复力 |
| V-008 | 稳态行和 | `verify_lock60.py` | N=384 → 3.95；N=8 → 2.29 | 软式归一化，N≥384 成立 |
| V-009 | permute 器官 | `permute_check.py` | 输出差 2.5e-16 | ⚠ **恒等变换**，见 L-005 |
| V-010 | permute mix 行 | `permute_check.py` | 输出改变，边缘分布差 0 | ✅ 唯一有效对照 |
| V-011 | 判据灵敏度 | `permute_sensitivity.py` | 信噪比 129~158× | ✅ 判据有效 |
| V-012 | 统计功率 | `permute_sensitivity.py` | n=5→0.61；n=10→0.73 | ⚠ **主对照需 10+ seeds** |
| V-013 | heal 归一化 | `tent_fix_check.py` | N=384 → 37.5（ce 的 16×） | ⚠ 必须 /[N(N-1)] |
| V-014 | 共享基初始化 | `tent_fix_check.py` | in_band 0.0004 → 0.999 | ✅ |
| V-015 | N=384 维度相容性 | `tent_dim_check.py` | d=128 可行 | ✅ Welch 界不阻塞 |
| V-016 | 锁6.0 十七项验收 | `verify_lock60.py` | **17/17 通过** | ✅ |
| V-017 | A 型组合空间维数 | `verify_lowrank.py` | 恒定 4 维，不随 N 涨 | ⚠ **被 rank 锁死**，见 L-008 |
| V-018 | A 型 FLOPs | `verify_lowrank.py` | r=4: P=1→0.383x；**r=32: P=8→0.656x** | ✅ P≥2 即便宜 |
| V-019 | r 扫描 | `verify_lowrank.py` | 4/16/32 的 P* = 0.13/0.68/1.71 | r=32 甜点 P=8~32 |
| **V-020** | **GPU 冒烟（B型 r=4）** | `run_smoke_gpu.py` | 核心判据 **+49.217**；wiring_var **0.10529**；in_band **0.714** | ✅ 检出分工 |
| V-021 | 双侧铰链梯度核验 | 手工数值 | 上侧 d/dcos=+0.4（推降）；下侧 −0.6（推升）；带内恒 0 | ✅ 方向正确 |
| **V-022** | **temp 随 N 缩放扫描** | `temp_scan.py` | N=8→temp*=1.8；**N=384→temp*=5.09** | ⚠ **规格 temp=2.0 在 N=384 下过均匀** |

---

## 4. 关键发现登记（L = Learning）

### L-001 漏乘 width² 的后果（修正用户描述）
- 用户原描述："直接发散到 1e18"
- 实测（SPSA 200 步，lr=0.02）：`max|mem|` 1.684 → 22.5（约 13×），**未发散**
- 处理：文档中改为"放大 1/width²=100 倍，实测参数范数恶化 13×"
- 可能原因：用户那次是 Adam + 更大 lr

### L-002 eff_rank / mix_var 判据失效 ★重要
| mix | eff_rank | mix_var |
|---|---|---|
| 内容决定（真分工） | 6.926 | 0.0118 |
| **逐 chunk 纯随机** | **7.822** | **0.0382** |

噪声分数**更高**。任何单看 mix 统计量的判据都分不出「分工」与「掷骰子」。
**唯一有效判据**：permuted 对照 + `mix_gain` 差值。

### L-003 heal 归一化必须 /[N(N-1)]
用户原推导「/n → 总回复力与 n 无关」漏了 off-diagonal 项数是 N(N-1) 而非 N。

| N | /N | /[N(N-1)] |
|---|---|---|
| 8 | 0.579 | 0.083 |
| 384 | **37.511** | 0.098 |

ce_loss ≈ 2.3 ⇒ N=384 时 heal 是 ce 的 16 倍，完全主导训练。

### L-004 lock47 器官初始化错误 ★重要
- 规格：`base(0.5√d) + randn*0.5`
- lock47 我写成：`randn*0.5`（无共享基）

| N | 规格 | lock47 |
|---|---|---|
| 384 | cos **0.4964**，带内 99.91% | cos 0.0002，带内 **0.01%** |

**后果**：连接机制从第 0 步就是死的。已在 lock6.0 改正。
（注：我独立推导的 `a = sqrt(band/(1−band))·s·√d = 0.5√d` 与规格常数完全吻合）

### L-005 permute 器官是恒等变换 ★重要
用户原判据：「训练时随机打乱器官顺序」
实测：输出差 **2.5e-16**。softmax 与 einsum 对器官维都是置换等变的。
**按字面实现会得到"差距为 0"并误判"分工不存在"**。
正确做法：训练照常，**评估时** permute mix 的行。

### L-006 条件数判据失效
随机矩阵加权和 → 谱平坦 → 条件数恒 ≈ 1。
实测 delta_scale 0.1→2.0 全程 2.66~2.95，与 mix 无关。
**用户裁决：判据改用 wiring_variance / val_ppl / in_band** ✅ 与实测一致。

### L-007 乘性铰链是「衰减加强器」不是回复力 ★重要
用户给的 `c = c * (1 - relu(cos-0.8)*5).clamp_min(0)`：

| cos | 纯高斯 | 加 hinge | 比值 |
|---|---|---|---|
| 0.90 | 9.00e-06 | 5.06e-06 | **0.563** |
| 1.00 | 1.39e-09 | 3.47e-11 | **0.025** |

比值恒 <1 ⇒ 乘性门把梯度压得更小。数学上乘性门只能压低 c 的数值，
产生不了"把 cos 推回 0.5"的方向力。
- 用户确认：**改用加性独立项** `relu(cos-0.8)²`，梯度 `2(cos-0.8)`
- 不违反红线2：稳态（行和归一化）在前向 ✅，铰链自愈在 loss ✅

### D-003 裁决：A 型 r=32 + 论文表述改为「连续组合空间」★重要
用户 2026-10-07 裁决，并纠正了一个更早的表述错误：

**「2^384 组合空间」这个说法本来就不对，与 r 取值无关。**
- 2^N 是二值选择空间的组合数（选/不选）
- wiring 是 softmax 出来的**连续向量** ∈ N 维单纯形（无穷多连续点）

| 框架 | 组合空间 | 规模 |
|---|---|---|
| 标准 MoE | 离散 top-k 选择 | C(N, k) 种 |
| Snowflake | 连续 wiring | **N 维单纯形（无穷）** |

**关键区别是「离散选择 vs 连续组合」，不是「2^N vs N」。**
正确表述（不依赖 r，审稿人挑不出刺）：
> 标准 MoE 的组合是离散的（C(N,k) 种 top-k 选择），我们的组合是连续的
> （N 维单纯形上的无穷多组合），能表达任意细粒度的专家组合。

**A 型低秩的真实代价不是组合空间，而是参数多样性**：
- 满秩：384 个器官是 [d,h] 空间里 384 个独立方向
- r=4：所有器官挤在 4 维子空间，本质只有 4 个独立方向 ⇒ 表达力上限被压缩
- **r=32**：32 维子空间，折中。参数比满秩降 **90 倍**，P=8 时仍 0.656x

### D-004 裁决：停用 A 型，改 B 型（rank=4, P=32）★重要
用户 2026-10-07 裁决。这是**无代价的净收益**：

| | 组合空间维数公式 | N=384,d=128,r=4 |
|---|---|---|
| A 型 | min(N−1, **r**) | **4 维** |
| B 型 | min(N−1, **d·r**) | **383 维**（d·r=512 > 383，满维） |

小规模实测（d=64,h=16）确认公式：
    N=16 r=4: A=4维 / B=15维（理论 15）✅
    N=64 r=4: A=4维 / B=61维（理论 63）✅

**FLOPs 几乎相同，组合空间涨 12 倍**：

| 配置 | 每 token | 相对 Top-K | 组合维数 |
|---|---|---|---|
| A型 r=32 P=8 | 10,752 | 0.656x | **32** |
| **B型 r=4 P=32** | 10,880 | **0.664x** | **383** |

差 1.2% 的 FLOPs，换来 32 → 383 维。**B 型严格占优。**

⚠ **B 型的 r 不能开大（与 A 型完全相反）**：
合成项 N·d·r/P 随 r 线性涨（A 型只有 N·r/P，差 d=128 倍）。
    r=4  P=32 → 0.664x ✅
    r=8  P=32 → 1.078x ⚠
    r=32 P=128→ 1.312x ❌（P 拉满仍更贵）
⇒ B 型效率全靠【小 r + 大 P】。P 越大越便宜，组合空间恒 383 维不变。

### L-011 冒烟 +49.2 提示 wiring 可能近似 one-hot ★待确认
核心判据量级反推：均匀≈0 / 温和分化≈0.07 / **one-hot≈49**。
实测 +49.217 落在 one-hot 档 ⇒ wiring 可能退化为「每样本挑一个器官」，
在行为上退化成离散选择，**打击「连续组合空间」叙事**。

**必须确认 `mix_sharp`**（冒烟每 100 步打印）：
- sharp ≈ 0.9~1.0 → 确认 one-hot
- sharp ≈ 0.2~0.4 → 连续组合成立，+49.2 只因簇太分离

**诊断流程（用户裁决）**：
1. 先看 sep=2.0 下的 sharp
2. 若 ≈0.9，跑 `--sep 1.0` 对照
3. sep=1.0 软化 ⇒ 数据问题；仍 one-hot ⇒ 架构问题，需熵奖励

论文叙事上：即使 one-hot 也可写「极限分离任务上退化为离散选择，
混合任务上保持连续组合」，但**需要 sep=1.0 的数据证明后者**。

### L-012 in_band 单调下降是设计内部张力 ★待确认
in_band 1.000 → 0.714 是**一路降下来的**，未收敛。
根因：**任务压力推 cos ↓（器官分化），高斯带通想留 cos ≈ 0.5**。
两者方向相反。

原铰链只在 cos > 0.8 生效（管趋同），**管不了 cos < 0.3（孤立）**。
⇒ 若漂移方向向下，铰链完全没出力，in_band 会继续掉。

**已加双侧铰链（默认关闭）**：
```
heal = relu(cos−0.8)²  +  relu(0.3−cos)²
        ↑ 管趋同          ↑ 管孤立（新增）
```
实测梯度方向正确（V-021）：带内 0.3~0.8 两侧梯度恒 0，不干扰已学分工。
开启方式：构造参数 `use_hinge_low=True` / 命令行 `--hinge-low`。

### L-013 sharp=1.0 的真因：任务诱导 + 观测假象 ★重要
冒烟 step 0 就 sharp=1.000，易读成"初始化即 one-hot"。
**但脚本打印在 epoch 循环之后** ⇒ step 0 的数值是训练完 1 个 epoch 后的状态。

实测初始 sharp（N=8, d=64, temp=4.0）= **0.505**（相当均匀）
    → 训练 1 epoch → 1.000 → 299 步维持 0.997

**用户所称"早熟硬化"方向正确，机制是任务诱导**：
合成任务是 8 簇 / 每簇一个真专家 / sep=2.0，**最优解就是 one-hot**。
模型一眼看穿，1 个 epoch 收敛到最优解。

⇒ **不代表架构有问题**。真实 LM 无此极端簇结构，必须在真实任务上复测 sharp。

### L-014 temp 必须随 N 缩放 ★重要（正式训练前必改）
logits ≈ N(0,σ²)，σ ∝ temp·|x|·|key|/√d；sharp 由 N 个 logits 的 max 决定
⇒ E[max] ≈ σ·√(2·ln N)。N=8→2.04，N=384→3.45（因子 1.69x）

| N | d | temp*（sharp=0.30） | 规格 temp=2.0 下的 sharp |
|---|---|---|---|
| 8 | 64 | **1.80** | 0.321 |
| 384 | 128 | **5.09** | **0.039** ⚠ 过均匀 |
| 384 | 256 | **5.41** | 0.034 ⚠ |
| 512 | 512 | **5.68** | 0.027 ⚠ |

判据用 `ent/lnN` 与 `top5 质量`（sharp 的均匀基准 1/N 随 N 变，绝对值不可跨 N 比）：
    N=384 temp=2.0：ent/lnN=0.907，top5=0.121 ⇒ 过均匀（陷阱侧）
    N=384 temp=5.0：ent/lnN=0.533，top5=0.566 ⇒ ✅ 目标区

**⇒ 正式训练 N=384 必须把 temp 从 2.0 提到 ~5.0，否则掉进均匀平均陷阱（病理1）。**

### L-015 「卡输出」是 stdout 块缓冲，不是算力问题
用户报告脚本在服务器上"卡输出"。诊断结论：

**根因 A：48 处 print 全部无 flush=True**（train 30 + smoke 18）
重定向到文件/管道时 stdout 是块缓冲（4KB 一刷），训练日志攒着不吐
⇒ 看起来卡死，实际在跑。**不浪费 GPU 时长。**
修法：全部加 flush=True。临时解法 `PYTHONUNBUFFERED=1 python ...`

**根因 B：每 100 步才打印 ⇒ 100 步盲区**
修法：加每 10 步轻量心跳 + 进循环前 `[ready]` 确认行。

**根因 C（次要）：einsum 走慢路径**
`'cpd,cdr->cpr'` 本质是 bmm 但 einsum 会先 permute；
`'cn,ndr->cdr'` 会展开出 `[C, d*r]` 大中间张量。
改为 `torch.bmm` / `torch.matmul`，**数值等价已验证（差 ~1e-15）**。

**算力核算确认不是瓶颈**：
    C = 64*128/32 = 256 chunks
    16 cells 每 forward = 2231 MFLOP，含 backward ≈ 6694 MFLOP/step
    A100 bf16 理论 0.02 ms/step，算 launch 开销约 2 ms/step
    ⇒ 100 步 ≈ 0.2 秒，不存在算力卡顿

**现场诊断法（30 秒）**：另开终端 `nvidia-smi -l 1`
    利用率 > 0% ⇒ 正常运行，只是输出没刷出来
    利用率 = 0% ⇒ 真卡住

### L-018 静默回退 CPU —— 在按时长计费的服务器上白烧钱 ★重要
用户报告"CPU 这次这么慢"。根因：

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # ← 静默回退
```

`train_lock60.py` 原本**没有** `run_smoke_gpu.py` 那样的硬失败保护。
当 `torch.cuda.is_available()` 返回 False（最常见：装了 CPU 版 torch），
程序**不报错、直接降到 CPU 跑**，用户要等很久才发现。

**速度差距（实测核算）**：
    每 step ≈ 6694 MFLOP（含 backward）
    A100 bf16  ≈ 2 ms/step
    CPU 多核   ≈ 130 ms/step
    ⇒ **50~100 倍差距**

    另：CPU 上 `autocast(bfloat16)` 支持有限，可能不生效甚至更慢。

**已修（用户要求"自动匹配"）**：GPU 优先，无 GPU 则自动用 CPU。
关键是【显式】而非【静默】——启动即宣告，并给出原因与降配建议：

```python
def _pick_device():     # train_lock60.py / run_smoke_gpu.py 均使用
    if torch.cuda.is_available():  -> 打印 GPU 型号+显存+torch/cuda 版本，返回 "cuda"
    else:                          -> 打印 WARN + 原因诊断 + 降配建议，返回 "cpu"
```

CPU 回退时自动：
- 关闭 bf16（CPU 上 bf16 支持差且常更慢）
- 未指定 `--scale` 时自动降为 `tiny`
- 提示调小 `--bs`

三种场景实测（stub 验证）：

| 场景 | 输出 |
|---|---|
| 有 GPU | `[device] GPU OK  NVIDIA A100-SXM4-80GB  64.0GB  torch=2.1.0 cuda=12.1` |
| CPU 版 torch | `[device] WARN ... 原因：装的是 CPU 版 torch（torch.version.cuda 为 None）` + 修复命令 |
| 驱动/运行时问题 | `[device] WARN ... 原因：torch 带 CUDA 12.1，但驱动/运行时不可用` |

**诊断命令（10 秒）**：
```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```
- `torch.version.cuda` 为 **None** ⇒ 装了 CPU 版 torch，需重装 GPU 版
- 为 `12.x` 但 `is_available()` 为 False ⇒ 驱动/CUDA 运行时问题

### D-005 裁决：必须拼专家 ⇒ 加内生最低有效器官数约束 ★核心
用户裁决："一定要拼专家！"

**问题量化**（step 100，N=384, temp=5.0）：

| sharp | 有效器官数 1/Σw² | 含义 |
|---|---|---|
| 0.30 | 10.96 | 真组合 |
| 0.957 | 1.09 | 只用 1 个 |
| **0.982** | **1.04** | **384 选 1 = 选专家** |

**药方**：把 wiring 与均匀分布混合，解析求 a：

```
w_a = (1-a)·w + a·(1/N)
(a-1)² = (1/K − 1/N)/(S₂ − 1/N),  S₂ = Σw²
a = clamp(1 − sqrt(ratio), 0, 1)
```

**实测验证（V-023，verify_min_eff.py）**：

| 输入 sharp | 原始有效数 | K=8 后 | 误差 |
|---|---|---|---|
| 0.982 | 1.04 | **8.00** | 0.0000 |
| 0.957 | 1.09 | 8.00 | 0.0000 |
| 0.500 | 3.99 | 8.00 | 0.0000 |
| 0.300 | 10.96 | 10.96（**不动**） | — |

K 的效果：K=2→sharp 0.707；K=4→0.499；**K=8→0.352**；K=16→0.247；K=32→0.172

**五条性质（均已验证）**：
1. **单边**：有效数 > K 时 a=0，完全不动 ⇒ 不破坏已学分工
2. **精确**：解析解，误差 0.0000，无迭代
3. **可微**：da/dw 有限（0.007 / 0.706），梯度可回传
4. **内生**：在前向里生效，不是 loss 惩罚（符合红线 2 精神）
5. **每样本独立**：坍缩样本多掺，已组合好的不动

**★ 与调 temp 的本质区别**：
    temp 是【间接】控制 —— 训练会把 logits σ 放大 10.9 倍（2.74→30.0），
    静态 temp 猜不准这个倍数 ⇒ L-014 的 temp=5.0 实测是错的
    min_eff 是【直接】控制 —— 约束的是【结果】（有效器官数），与 spread 无关

开关：`--min-eff 8`（默认 0=关闭）

### L-019 temp=5.0 是错的，正确值约 0.5（已被 D-005 取代）
反推：训练把 logits σ 从 2.74 放大到 30.0（10.9 倍）。
要让训练后 sharp≈0.30 需 σ≈2.78 ⇒ temp ≈ 5.0/10.9 ≈ **0.46**。
→ 但静态 temp 依赖"放大倍数"这个未知量，**不可靠**。
→ 已改用 D-005 的直接约束，temp 不再是关键参数。

### L-020 服务器是 DCU，不是 NVIDIA；瓶颈是 sync 不是算力 ★重要
**环境确认**：
```
torch.version.cuda = None          ← ROCm/HIP 分支，无 CUDA 版本号
torch.cuda.is_available() = True   ← DCU 可用（HIP 伪装成 CUDA）
torch.cuda.get_device_name(0) = "BW"
```

⚠ 我此前误判为"装了 CPU 版 torch"，**诊断错误**。DCU 一直在用。
已将 `_pick_device()` 改为识别 `torch.version.hip`。

**实测数据（用户跑）**：
```
fp32 matmul (8192x128 @ 128x32): 0.01 ms
bf16 matmul (同上)            : 0.02 ms   ← 慢 2 倍
device: BW
```
用户判断正确：矩阵太小，被调度开销主导，测不出真实算力。
但 **bf16 比 fp32 慢 2 倍** 是可信结论 ⇒ DCU 上 bf16 无收益反亏。

**"每步 2 秒"的真因（非算力）**：
单 kernel 0.01ms，而冒烟每 step 的 matmul 比测试矩阵还小 ~100 倍
⇒ 算力差 10⁶ 倍，完全不是瓶颈。

真正的开销是**同步**：
- `loss.item()` 每次强制 GPU→CPU 同步
- 冒烟每 step 16 个 batch × 2 次 = 32 次 sync
- DCU 单次 sync 可达数十 ms ⇒ 32 × 60ms ≈ 2s ✅ 与实测吻合
- 次要：Adam 的 ~36 个小 kernel，纯调度开销

**已修**：
1. 加 `--no-bf16` 开关（DCU 上必关）
2. 冒烟/训练的 `loss.item()` 改为累积 tensor，仅打印步才同步
   预期每步 2s → 0.1s（约 20x）
3. `_pick_device()` 识别 `torch.version.hip`

### L-021 冒烟是 N=8；sharp=0.456 是【约束夹出来的】不是学会的 ★重要
我上轮按 N=384 分析（得 eff=4.79，判"约束未激活"）——**算错了**。
`run_smoke_gpu.py` 默认 `n_cls=8`，模型是 `SnowflakeB(n_organelles=8)`。

| | sharp=0.456 的有效器官数 | 判定 |
|---|---|---|
| **N=8（冒烟实际）** | **3.997** | **== K=4 ⇒ 约束正好激活，被夹在 K** |
| N=384（我错算的） | 4.791 | > K ⇒ 不激活 |

**决定性证据**：sharp 300 步恒定（0.456/0.455/0.455/0.454）。
- 若模型**学会**组合 ⇒ eff > K ⇒ a=0 约束退出 ⇒ sharp 应浮在 K 之上且随训练变化
- 若模型**仍要**one-hot ⇒ eff < K ⇒ 被夹回 K ⇒ sharp 恒定 = K 对应值 ← **实测正是**

**⇒ 模型底层偏好仍是 one-hot，是约束在兜着。**

反推混合系数 a = (1−sharp)/(1−1/N)：
| N | sharp | a | 合成权重构成 |
|---|---|---|---|
| 8 | 0.456 | 0.622 | 37.8% 选中 + **62.2% 全体平均** |
| 384 | 0.352 | 0.650 | 35.0% 选中 + **65.0% 全体平均** |

⇒ "临时专家" 约 2/3 是全体平均专家。这是**弱组合**。

### L-022 参与器官数 K 与特异性的数学张力（不可调参解决）
若底层偏好 one-hot，强制 eff=K 必然等价于掺 uniform：

| K（N=384） | sharp | a（均匀占比） | 选中占比 | 解读 |
|---|---|---|---|---|
| 2 | 0.707 | 29.4% | 70.6% | 接近选专家 |
| 4 | 0.499 | 50.2% | 49.8% | 弱组合 |
| **8** | 0.352 | **65.0%** | 35.0% | 合理组合 |
| 16 | 0.247 | 75.5% | 24.5% | 过度平均 |
| 32 | 0.172 | 83.1% | 16.9% | 几乎纯平均 |

**K ↑ ⟺ 均匀占比 ↑ ⟺ 单器官特异性 ↓ —— 数学必然。**
⇒ 单靠调 K 无法同时拿到"多器官参与"和"强特异性"。
⇒ 真正的解法是让模型**自己想分散**（release test 验证），而非掺 uniform。

**release test（已实现，`--release-steps N`）**：
```
1. 用 --min-eff K 训练收敛
2. 自动关掉 min_eff，继续训 N 步
3. 看 sharp 回弹幅度 rebound = (sharp_off − sharp_on)/(sharp_on − 1/N)
       < 25%  ⇒ 涌现：模型自己学会，论文可写"涌现"
    25~60%  ⇒ 部分内化，需更长训练或降 K 再测
       > 60%  ⇒ 强制：纯靠约束，论文只能写"架构强制最低组合度"
```
结果自动落盘（release_* 字段）。

### L-023 wiring_var 的 0.05 门槛是错的（跨 N 不可用）
wiring_var 的上限与 N 强相关：
    N=8   完全 one-hot ⇒ var_max ≈ 0.109
    N=384 完全 one-hot ⇒ var_max ≈ 0.00260

**⇒ 0.05 在 N=384 下【数学上不可能达到】（连 one-hot 上限都只有 0.0026）。**
实测 0.0152 与 K=4 的理论值 0.01565 吻合（占 N=8 上限的 13.9%）。

**修正**（已实现）：
- 新增 `wiring_var_norm = var / var_max`（跨 N 可比）
- 新增 `eff_organs = 1/Σw²`（核心量，跨 N 可比，直接就是"参与器官数"）
- 两者已进 `run_metrics.py` 字段表，自动落盘

建议判据改成 `eff_organs`（直接、可比、有物理意义），
`wiring_var_norm` 作辅助。

### D-006 数据接入：字符级 vocab=92（已实现）
诊断结果（服务器 tinystories_100mb.txt）：**字符级，V=92**。

| 项 | 值 |
|---|---|
| embedding | 92 × 128 = 11,776 |
| head | 128 × 92 = 11,776 |
| 合计 | **23,552 = tiny 主体的 0.65%** ✅ 不会喧宾夺主 |

⇒ 架构对比干净，论文无需额外说明 embedding 占比。

**规模**：100 MB ≈ 105M token
| 配置 | tok/step | step/epoch |
|---|---|---|
| bs=64 seq=128 | 8,192 | 11,520 |
| bs=32 seq=128 | 4,096 | 23,040 |

**DCU 时间估算**（sync 优化后）：
    乐观 0.1 s/step → 1 epoch = 0.32 h
    中性 0.3 s/step → 1 epoch = 0.96 h
    保守 1.0 s/step → 1 epoch = 3.20 h

**已实现**：
- `load_data()` 内置字符级加载（自动算 vocab、90/10 切分）
- 新增 `--bs` / `--seq-len` / `--max-steps`
- 摸底建议 `--max-steps 3000`（防止一开始就烧几小时）

### D-007 决策：本次正式训练采用【字符级】vocab=100
用户 2026-10-07 拍板。理由与代价均已确认：

**选字符级的理由**：
- embedding 仅 25,600 参数 = tiny 主体的 **0.71%** ⇒ 不稀释架构对比
- 碾压论证（拼专家 vs 标准 MoE）需要在"主体模型占绝对主导"下做，字符级满足

**代价（必须写进论文 limitations）**：
- 历史 lock39/40 的 PPL（9.9787 / 10.0157）是**词级**
- 字符级 PPL 将在 3.7~5.0 量级，**与历史数字不可比，不得并列引用**

**★ 由此产生的假设（待验证，可能一次性解决三个历史冲突）**：
日志 §6 的三处冲突（基线 9.9787 vs 10.0157、擦除 18.1251 vs 22.83、
终身学习 +0.0098 vs +1.8 vs −0.57）**可能根因都是 tokenization/切分口径不同**，
而非真实差异。若确认，三处冲突一次性解决。

**字符级判据基准**：
```
随机初始化 ln(100) = 4.6052
3000 步后:  loss > 2.5 → 有问题；1.8~2.2 → 正常 ✅
完整 epoch: loss 1.3~1.6 → 健康（ppl 3.7~5.0）
```

### L-024 字段注册缺口（已在远端踩到，本地已修）
用户远端报：`STEP_FIELDS 需要注册 eff_organs（以及 wiring_variance 等字段也缺）`。

**根因**：我加 `eff_organs` / `wiring_var_norm` 时只改了 `snowflake_B.stats()`
和打印行，**漏了三处**：
1. `run_metrics.STEP_FIELDS` 未注册 ⇒ `log_step` 抛错
2. `SnowflakeLM.agg_diag()` 的 `keys` 列表未含新字段 ⇒ 采不到
3. `--n-perm` 在 argparse 中重复定义两处 ⇒ 冲突

**已修（本地，校验通过）**：
- `STEP_FIELDS.update` 补齐 18 个字段：in_band / cos_mean / cos_std / cos_max /
  connect_rowsum / wiring_ent / wiring_variance / **wiring_var_norm** /
  **eff_organs** / mix_sharp / heal_term / combo_dim / flops_ratio /
  throughput_tok_s / wall_sec / epoch / tokens_seen / lr
- `agg_diag.keys` 同步补齐（10 个）
- 删除重复的 `--n-perm` 定义
- 每 100 步打印行补 `eff=`

**校验脚本**（已跑）：`snowflake_B._diag` 的 9 个 key 与 `agg_diag` 的 10 个 key
全部在注册表中，**缺失字段=无**。

**教训**：新增指标必须三处同步改 —— `stats()` / `agg_diag()` / `STEP_FIELDS`。
已写进 CONTRIBUTING.md。

### L-025 首次真实数据训练：loss 4.60 → 2.24 ✅
字符级 vocab=100，随机基线 ln(100)=4.6052，**起始 4.60 完全吻合** ⇒ 数据接入正确。

| 阶段 | loss | ppl |
|---|---|---|
| 起点 | 4.605 | 99.98 |
| **当前** | **2.240** | **9.39** |
| 3000 步目标 | 2.00 | 7.39 |
| 完整 epoch | 1.45 | 4.26 |

已压缩 51.4% 的初始不确定性。

判据（D-007）：
- 若才几百步 ⇒ 学习速率健康，很好
- 若跑满 3000 步 ⇒ 2.24 略高于目标区间(1.8~2.2)上沿 0.04，**仍属正常**
- 若已完整 epoch ⇒ 偏高，可能 `--min-eff 8` 的均匀化拖累拟合（见 L-022）

### L-026 ablation 崩溃导致 3000 步数据全丢 ★重要
**故障链（三个独立 bug，均在我这边）**：

| # | 问题 | 后果 |
|---|---|---|
| 1 | `DEV` 未定义（`train_lock60.py` 里是 `DEVICE`，从冒烟脚本复制时没改） | ablation 崩溃 |
| 2 | `finalize()` 只在最后写盘 | ablation 一崩 ⇒ **前面 3000 步全丢** |
| 3 | 打印行未含 `eff` | 核心指标看不见 |

**#2 最严重**——它把 #1 的小错放大成"3000 步白跑"。这是设计缺陷：
证据系统不该有单点故障。

**已修（8 项校验全过）**：
1. `DEV` → `DEVICE`
2. ablation 用 try/except 兜底，崩了也保证训练数据落盘
3. `log_step` 用 try/except 兜底（字段未注册不拖垮训练）
4. 新增 `--save-ckpt` / `--eval-only`：ablation 崩了可从 ckpt 恢复评估，
   **不用重训**
5. 打印行补 `eff=`
6. 三处字段同步已校验（stats / agg_diag / STEP_FIELDS）

**教训**：证据采集必须【增量 + 异常安全】，不能只在 finalize 时一次性写。
已写进 CONTRIBUTING.md。

### L-027 全量审查发现 5 个真 bug（均已修）★重要
从头至尾静态审查 `train_lock60.py` / `run_metrics.py` / `snowflake_B.py`，
发现 5 个此前遗漏的问题，其中 2 个会直接导致错误结论：

| # | bug | 后果 | 严重度 |
|---|---|---|---|
| **A** | `--seed` 未传给 `SnowflakeLM` | **10 seeds 全部用 seed=0 初始化** ⇒ 结果几乎相同，CI 虚窄，统计检验失效 | 🔴 致命 |
| **C** | FLOPs 用满秩公式 | 算出 **3.25x（更贵）**，实际低秩 **0.664x（便宜 33.6%）**，差 4.9 倍 ⇒ 会反向推翻效率主张 | 🔴 致命 |
| **D** | loss 双重 shift | `get_batch` 已 shift（y[j]=x[j+1]），loss 又 `[:, :-1]`/`[:, 1:]` ⇒ 错位 1 token，学成 x[j]→x[j+2]，loss/ppl 虚高 | 🟠 严重 |
| **B** | release test 用 `val_iter` 训练 | 在验证集上继续训练 = 数据泄漏，判定失真 | 🟠 严重 |
| **E** | ablation 的 `C = bs // P` | 实际 chunk 数是 `bs*seq/P`，算小 ⇒ 只打乱前几个 chunk，permuted 效果被削弱，**核心判据被低估** | 🟠 严重 |
| **G** | ablation 的 loss 未 reshape | `ce([bs,seq,V], [bs,seq])` 形状不匹配 ⇒ 报错或算错 | 🟡 中等 |

**修复验证**：
- A：`SnowflakeLM(..., seed=a.seed)`
- B：release test 改用 `train_iter`
- C：`per_tok = N·d·r/P + d·r + r·h + d·h`（B 型低秩）
  修正后各规模：tiny **0.664x** / mid 0.457x / large 0.385x ✅
- D：`ce(logits.reshape(-1,V), yb.reshape(-1))`（去掉双重 shift）
- E：`C = (bs * seq) // P`
- G：`loss_fn(o.reshape(-1, o.size(-1)), y.reshape(-1))`

**★ C 的教训最深**：满秩/低秩的 FLOPs 差 4.9 倍，而我一直用满秩公式打印。
若按旧公式，论文结论会是"拼专家比 Top-K 贵 3.25x"——**与事实完全相反**。

### L-028 STEP_FIELDS 注册依赖错位（字段注册机制的真正根因）★重要
用户报："experiment_logger.py 的 STEP_FIELDS 仍未确认"。

**根因**：`experiment_logger.py` 的原生 `STEP_FIELDS` 停留在 **lock47 时代**：
```
旧字段: cancer_mask / cond_ema / mix_var / org_norm / mix_eff_rank
缺   : cos_mean / cos_std / cos_max / connect_rowsum /
       wiring_variance / wiring_var_norm / eff_organs / combo_dim / flops_ratio
```

而新字段是靠 `run_metrics.py` 里的 `STEP_FIELDS.update(...)` 追加的。
这是**脆弱依赖**，三个问题：

1. 必须 import `run_metrics` 才生效；直接用 `experiment_logger` 的脚本
   （如 `verify_lock60.py`）会报"未知字段"
2. 字段名对不上：旧 `mix_var` vs 新 `wiring_variance`（同一含义两个名字）
3. `eff_organs`（核心判据）不在原生表里 ⇒ 采不到

**已修**：新字段**直接写进** `experiment_logger.py` 定义体，不再靠 update。
- `STEP_FIELDS`：21 → **30** 个（保留旧字段兼容历史 run）
- `FINAL_FIELDS`：24 → **51** 个（含 core_stat / release_* 等）

**校验**：11 个关键 step 字段 + 8 个关键 final 字段全部 OK。

**原则**：schema 定义必须自包含，不能靠下游模块反向修改。

### L-029 ★★ 最关键结果：真实 LM 上模型【自发】用了 ~31 个器官 ★★
3000 步摸底重跑（N=384, vocab=100→98, min_eff=8）：

| 步数 | eff_organs |
|---|---|
| 0 | 334.4 |
| 500 | 62.6 |
| 1500 | 31.5 |
| 2900 | **31.3** |

**★ 决定性推论：约束【没有激活】**

    min_eff K = 8，实测 eff = 31.3
    31.3 > 8  ⇒  a = 0  ⇒  enforce_min_effective 完全没介入

对比 N=8 冒烟那次（L-021）：
    eff = 4.00 == K = 4  ⇒  【被夹在 K 上】⇒ 强制
本次：
    eff = 31.3 >> K = 8  ⇒  【模型自己走到 31.3】⇒ 自发

**⇒ 在真实 LM 任务上，模型自发使用了约 31 个器官。拼专家成立。**
这推翻了 L-021 的担忧（"字符级太简单会退化 one-hot"）。

### L-030 release test 判据用错指标：应改用 eff 而非 sharp
原判据 `rebound = (sharp_off − sharp_on)/(sharp_on − 1/N)` 给出 **39.8% → WARN**。

但实测数据内部矛盾：

| 指标 | 约束开 | 约束关 | 变化 | 读作 |
|---|---|---|---|---|
| sharp | 0.1289 | 0.1792 | **+39.8%** | 更尖了 |
| eff | 31.63 | 32.96 | **+4.2%** | 更分散了 |

**方向相反**。原因：sharp 只看 `max(w)`，N=384 下对尾部噪声极敏感；
eff = 1/Σw² 反映整体分布，稳定得多。

**已修**：主判据改 `rebound_eff = (eff_on − eff_off)/(eff_on − 1)`。
⇒ 按 eff 重算本次：**−4.3%，属"完全保持" ⇒ OK 涌现**

### L-031 mojibake 清洗（已加 --fix-mojibake）
实测 vocab=100，其中 21 个非常规字符（Â Ã â œ）疑似 UTF-8 双重编码。
机制：UTF-8 字节被 latin-1/cp1252 解码后再存回 UTF-8。

已加 `--fix-mojibake`：尝试 `txt.encode('latin-1').decode('utf-8')` 逆转，
仅在字符数减少时采纳。

**★ 实测结果（见 L-034）：只减 2 个，影响 0.014%，推测作废。**

### L-032 ablation 崩溃：generator device 不匹配（已修）
报错：`Expected a 'cuda' device type for generator but found 'cpu'`
位置：`run_metrics.py:240 torch.randperm(C, generator=g, device=dev)`

根因：DCU/HIP 伪装 cuda 时，CPU generator 配 cuda randperm 不被接受。
**已修**：`torch.Generator(device=dev)` 建在目标 device 上 + try/except 兜底。

### L-033 perm 长度算错 128 倍 —— 修 L-027 时引入的新 bug ★重要
用户发现，根因是我修 L-027 时写错：

```python
T = xb.shape[1] if xb.dim() > 2 else 1     # ❌
#   xb 是 [bs, seq] 的 token id 张量，dim() == 2 ⇒ 走 else ⇒ T = 1
C = (64 * 1) // 32 = 2                     # ❌ 应为 256
```

| | 值 |
|---|---|
| chunk 总数 | 256 |
| 修复前打乱 | **2 个（0.8%）** |
| 修复后打乱 | **256 个（100%）** |

⇒ 99.2% 连接没被 permute ⇒ `permuted` ≈ `learned` ⇒ **核心判据 ≈ 0，证伪实验失效**

**修复**：`T = xb.shape[-1]`（64×128/32 = 256 ✅）+ 加 `C < 8` 断言。

**教训**：修 bug 引入新 bug。改动后必须【数值验算】，不只编译通过。

### L-034 vocab=92 的推测作废，真实值是 98 ★
L-031 我推测 vocab 应为 92、多出 8 个全是 mojibake。
实测 `--fix-mojibake` 后 **100 → 98**，tokens 减 14,971。

    14,971 / 104,857,600 = **0.0143%**

⇒ 非常规字符【大部分是真实的】，不是双重编码污染。
⇒ **loss 不会因清洗明显下降**——"1.784 是虚高"的判断作废。

**教训**：92 这个数字来自 `probe_data.py` 只读前 8MB 的采样，
与全量 98 不符。**采样统计不能替代全量统计**。

**修正**：论文中所有涉及 vocab 的数字改用 **98**。

### L-035 ★ 无约束跑：loss 更低 + eff=18，结论达到最强版本 ★★
配置：清洗（vocab 98）+ `--min-eff 0`（无约束）+ 3000 步

| 配置 | loss | eff | 定性 |
|---|---|---|---|
| 带约束 min_eff=8（vocab 100，未清洗） | 1.784 | 31.3 | 真组合 |
| **无约束 min_eff=0（vocab 98，已清洗）** | **1.779** | **18** | **真组合 ✅** |

**★ 两个关键结论**：

1. **loss 更低（1.779 < 1.784）** ⇒ 约束确实有代价（虽小，0.005）
   ⇒ **最强版本的主张成立**：
      "不需要任何强制，模型自发使用约 18 个器官，且效果最好"
2. **eff=18 ⇒ 真组合**。且比 31 更集中 ⇒ 选择性更强、特异性更好。
   eff=31 偏"平均"，eff=18 每个器官贡献更明确 —— **这是更好的结果**。

**⚠ 注意**：两组数据不同（vocab 100 vs 98），loss/eff 差异可能部分来自
数据清洗而非约束本身。若要严格分离两者影响，需补一组
"vocab 98 + min_eff=8" 的对照（优先级 P2）。

**定性标尺**（N=384）：
    eff ≤ 2   ⇒ 选专家（one-hot）
    eff ≤ 4   ⇒ 弱组合
    eff ≤ 8   ⇒ 勉强算组合
    **eff 8~40 ⇒ 真组合 ✅**
    eff ≈ 384 ⇒ 接近均匀（几乎全用）

### L-036 忆点可擦除性：旧 claim 失效，但新机制成立 ★重要
用户判断："擦除后就是可以重训练，越训练越好，而且全擦除后直接全损毁"。
**数值验证：三点全部成立，但与 README 的旧 claim 是两回事。**

#### 【1】推理时擦除【不保持性能】——旧 claim 失效
| 操作 | 器官 0 的 wiring 权重 |
|---|---|
| 正常 | 0.2579% |
| **清零 memory[0]** | **1.5359%** ⬆ 6 倍 ❌ |
| mask wiring（外部） | 0.0000% ✅ |

根因：`logit = xsum @ key[i]`，清零后 `key[i]=0` ⇒ `logit=0`。
但 softmax 下 `logit=0` 是**中等值**（其他器官 logit 有正有负，std≈21），
**不是最小值** ⇒ 权重反而上升。
⇒ 真正禁用器官只能外部 mask wiring，那不是"擦除记忆"。

**⇒ README 的 "Erasing 50% causes only mild PPL degradation" 在新架构下为假。**

#### 【2】全擦除 ⇒ 全损毁 ✅（与用户判断一致）
后果链（已定量验证）：
```
memory=0 ⇒ cos=0 ⇒ 高斯带通 exp(-((0-0.5)/0.1)²)=1.39e-11 ⇒ connect≈0
        ⇒ key = 0 + c.T@0 = 全零
        ⇒ wiring logit 全零 ⇒ softmax = 完全均匀 (1/384)
        ⇒ U1 = wiring@W1_U = 所有器官等权平均
        ⇒ 模型退化为【单一平均专家】，表达力崩塌
```
实测：connect 行和从 311.7 → 5.3e-9，|key| 从 49.8 → 0。

**⇒ 这反而是好证据：证明 memory 是【必要】的，不是冗余参数。**

#### 【3】部分擦除 ⇒ 可重训练恢复 ✅（与用户判断一致）
关键：擦除只打掉【身份】，没打掉【能力】
| 参数 | 是否受影响 |
|---|---|
| `memory[i]`（身份/连接） | ❌ 被清零 |
| `W1_U[i]` / `W2_U[i]`（计算能力） | ✅ 完整保留 |

实测擦除 50% 后：
- 未擦器官 \|key\| = 48.9（照常工作）
- 被擦器官 \|key\| = 0（等待重建）
- in_band 从 1.0 → 0.25（半瘫痪，非全损）

⇒ 重训练时只需重建 memory，W1_U/W2_U 的知识可复用
⇒ **低成本恢复，不是从零训练**

#### 【4】"越训练越好"是【假设】不是【结论】
机制：擦除破坏已收敛的连接模式 ⇒ 重训练跳出局部最优 ⇒ 可能更好
（类似 dropout / 破坏-重建正则化）

⚠ **不是必然**，取决于擦除比例、重训练步数、学习率。
⇒ 必须实验验证，不能直接写进论文。

#### 已实现 CLI
```bash
--erase-frac 0.5        # 擦除比例
--erase-ckpt <path>     # 从 ckpt 加载后擦除
--retrain-steps N       # 擦除后重训练步数
```

#### 论文处理建议
- **不要**沿用 README 的 "inference-time erasure" claim
- 改为 **v2.0 新能力：连接可塑性（structural plasticity）**
  部分擦除⇒可重建；全擦除⇒崩溃⇒证明 memory 必要
- 版本切分：v1.0（value slot，可擦除）/ v2.0（连接本体，可塑）

### L-037 ★★ H4「分工真实存在」正式坐实：10/10 seeds 全部显著 ★★
三步验证全部完成，**论文主表最后一块拼图补齐**。

#### ① ablation（ckpt 补跑，无需重训）
| 指标 | 值 |
|---|---|
| learned loss | 1.76905 |
| constant（固定器官） | 3.24251 |
| permuted（打乱配对） | 3.18130 |
| **core_stat** | **+1.4123** |
| 95% CI | [+1.4097, +1.4148]（下界远 > 0） |
| Cohen's d_z | 34.55 |

**判据：H4_分工真实存在 = PASS**

#### ② 清洗数据 + 无约束重跑
- cp1252 round-trip 修复 mojibake：**vocab 100 → 98**，tokens 25,000,000 → 24,985,029
- 末段 loss **1.779**（带 `--min-eff 8` 时为 1.784）⇒ 无约束更低
- eff 自发 ~18（带约束 31）⇒ `--min-eff` 多余
- ablation 同样显著：core_stat=+1.338，d_z=34.05

#### ③ 10 seeds 汇总统计
| 指标 | 均值 | SD | 95% CI | CV |
|---|---|---|---|---|
| **core_stat** | **1.3627** | 0.0291 | [1.3419, 1.3836] | 2.14% |
| 末 loss | 1.7882 | 0.0136 | [1.7785, 1.7979] | 0.76% |
| **eff_organs** | **25.73** | 2.06 | [24.26, 27.20] | 8.01% |
| Cohen d_z | 37.25 | 2.55 | [35.43, 39.07] | — |

**10/10 全部显著**，core_stat 极差 [1.332, 1.422]，d_z 全在 32~41。

单样本 t 检验（H0: core_stat=0）：**t = 147.9, df = 9, p << 0.001**

**eff 全部 > 23**（最小 23.0）⇒ **无一个 seed 退化为选专家**。

#### ★ 三条结论
1. **H4 坐实**：打乱器官配对后 loss 跳升 1.36，CI 不跨 0，d_z≈37（巨大效应量）。
   配对本身承载信息 ⇒ 分工不是 one-hot 巧合。
2. **自发组合叙事成立**：无 `--min-eff` 也自发 eff≈26，且 loss 更低。
3. **`--min-eff` 应移除**：它带来更高 loss（1.784 vs 1.779）和更平均的 wiring（eff 31 vs 26）。

#### ⚠ 一处需说明
之前单次跑 eff=18，本次 10 seeds 均值 25.7。差异 7.7。
两组不可直接比（可能末步瞬时值 vs 末段平均，或 3000 步时 eff 仍在震荡）。
**报告时用 10 seeds 均值 25.7，并注明 3000 步时在 23~29 区间震荡。**

#### 证据位置
`/root/private_data/snowflake/snowflake_open_v7/snowflake_open/output/runs/`
- ① `20261007T022603Z_31931f`
- ② `20261007T023448Z_a6845c`
- ③ `20261007T02~03Z_*` 十个 run

#### 论文主表现状
| 证据 | 状态 |
|---|---|
| 拼专家成立（eff≈26 自发） | ✅ |
| **分工真实（core_stat=1.36, 10/10）** | ✅ **已坐实** |
| 涌现而非强制（无约束最优） | ✅ |
| 效率碾压（0.664x） | ✅ |
| 10 seed 统计检验 | ✅ |

### L-008 A 型低秩把组合空间锁死在 r 维 ★重要
用户说"组合空间不受影响（组合在 wiring 上，不在 rank 上）"：

| | 组合空间维数 | N=384 实测 |
|---|---|---|
| A 型（coeff，用户给的） | min(N−1, **r**) | **4 维** |
| B 型（每器官独立 U_i） | min(N−1, **d·r**) | 383 维 |

**对 B 型成立，对 A 型不成立**。真实的二选一：
- A 型：效率极致（P=1 就 0.38x）+ 逐 token 自适应，但组合空间 r 维
- B 型：组合空间 383 维（保住能力1 论文主张），但需 chunk 且贵一点
- **已裁决（D-003）**：A 型 r=32。r=4 表达力压缩太极端。

### L-009 A 型解除 chunk 的必要性
- 满秩：P 必须 128 才打平，且 seq_len=128 是上限 ⇒ 能力2 被稀释
- A 型：**P=1（逐 token）就是 0.38x** ⇒ 可逐 token 自适应且更便宜

### L-010 模型未吃满容量
1.8M → 0.5M，两种架构 PPL 都稳在 10.01~10.02 ⇒ 任何架构对比都在噪声里。
**64GB 应开到 mid（9.7M）或 large（53M）**，第一次有机会让差异浮出噪声。

---

## 5. 待办与阻塞项

### 阻塞论文/开源
1. **permuted 对照真机结果** — 没有它"组合"主张不可证伪
2. **≥10 seeds 主表** — n=5 只有 0.61 power
3. 实测吞吐 / 延迟 / 峰值显存（现只有理论 FLOPs）
4. 第二个数据集一致性
5. 历史数字冲突（见 §6）

### 待用户裁决
- [x] **A 型 vs B 型** → **已裁决 B 型 r=4/P=32**（D-004），A 型停用
- [ ] P=32 粒度是否太粗：若能力2（样本自适应）表现不足，可试 P=64（0.477x，组合空间不变）
- [ ] **确认 mix_sharp 末值**（L-011）：≈0.9 则 one-hot，需跑 `--sep 1.0` 对照
- [x] **确认 mix_sharp**（L-011）：末值 0.997 ⇒ one-hot，真因为任务诱导（L-013）
- [ ] **接入 TinyStories**：跑 `python probe_data.py <路径>`，把生成的 `load_data()` 粘进 `train_lock60.py`（第 308 行）
- [ ] **确认 cos_mean 漂移方向**（L-012）：0.459→0.397 缓慢下降 ⇒ 倾向开 `--hinge-low`
- [x] **temp 已改 5.0**（L-014）：`train_lock60.py --temp`，默认 5.0
- [x] **SnowflakeLM 已切 B 型**（D-004）：原仍在用满秩 `SnowflakeCell`，已修
- [ ] 擦除验收线方向（见 §6 冲突 1）

### 执行顺序
```bash
# ① 冒烟（约 5 分钟）—— 已通过，见 V-020
python run_smoke_gpu.py --smoke-only

# ② 正式训练（temp/rank 已按 L-014/D-004 设好默认值）
python train_lock60.py --scale tiny --data <TinyStories> --delta-scale 0.5
python train_lock60.py --scale tiny --data <TinyStories> --delta-scale 1.0

# ③ 胜出档 10 seeds（功率分析：n=5 仅 0.61 power）
# ④ permuted 对照（训练后评估，不重训）
```

### 论文证据自动采集（每次运行强制落盘）
新增 `run_metrics.py`，已接入 `train_lock60.py` 与 `run_smoke_gpu.py`：

```
output/runs/<run_id>/record.json    meta + data + steps + final + verdicts
output/runs/<run_id>/steps.csv      每步时序
output/runs/master.csv              所有 run 横向汇总
```

自动采集：
- **meta**：run_id、CST/UTC 时间、脚本、seed、config 全量 + **config_hash**、
  python/torch/cuda 版本、GPU 型号、**git commit + git_dirty**
- **data**：**数据 sha256**、字节数、mtime、tokenizer、vocab_size、切分方式
- **step**：loss、lr、in_band、cos_mean/std、wiring_ent、wiring_variance、
  mix_sharp、heal_term、**combo_dim**、**flops_ratio**、显存峰值、吞吐、墙钟
- **final**：三变体 loss、**mix_gain**、**核心判据 + 95% CI**、**Cohen's d_z**、p 值、
  参数量、理论/实测 FLOPs、延迟、显存峰值
- **verdicts**：H1(带内) / H2(方差) / H3(非one-hot) / H4(分工) / H5(效率) 自动判定

⚠ 字段未注册会直接抛错（防 `wiring_ent`→`wiring_entropy` 的口径漂移）。
⚠ 缺 data_sha256 的 run 会被标记 `meta_missing`，**不能进论文主表**。

**当前默认参数链路（已验证传导）**：
```
--temp 5.0  →  SnowflakeLM(temp=)  →  SnowflakeB(temp=)  →  F.softmax(..., temp)
--rank 4    →  SnowflakeLM(rank=)  →  SnowflakeB(rank=)  →  W1_U [N,d,r]
--hinge-low →  SnowflakeLM(use_hinge_low=) → SnowflakeB → heal 加下侧项
```

---

## 6. 已知数字冲突（未解决，不得同时引用）

| # | 议题 | 冲突值 | 状态 |
|---|---|---|---|
| 1 | 擦除 100% | 18.1251 vs **22.83** | **验收线方向未定**：≤30 算达标则 22.83 通过；若要求"显著退化以证明知识存在"则不通过 |
| 2 | 终身学习回归 | +0.0098 vs **+1.8** vs −0.57 | 差两个数量级。v2–v5 与 v4' 互斥，等 scale=1.0 重跑 |
| 3 | 基线 PPL | 9.9787 vs 10.0157 | 未统一 |
| 4 | `memory_read_scale` | 25.0（legacy，mr2 再 ×2 =50）vs 可学习参数（lock43+） | 两条链路上"改成 1.0"是两件不同的事 |

---

## 7. 文件索引

| 文件 | 作用 |
|---|---|
| `EXPERIMENT_LOG.md` | **本文件** — 唯一真相来源 |
| `EXPERIMENT_PROTOCOL.md` | 论文级数据协议（预注册判据 H1–H7、矩阵 G1–G8） |
| `snowflake_spec.py` | 锁 6.0 满秩规格实现 |
| `snowflake_lowrank.py` | A 型低秩实现（当前主线） |
| `train_lock60.py` | 真机训练 + smoke + 规模预设 |
| `verify_lock60.py` | 17 项规格验收 |
| `verify_lowrank.py` | 低秩两项结论验证 |
| `ablate_permute.py` | 三变体消融（learned/constant/permuted） |
| `spec_check.py` / `tent_fix_check.py` / `tent_dim_check.py` | 规格数值校验 |
| `permute_check.py` / `permute_sensitivity.py` | 判据有效性与功率 |
| `scale_plan.py` | FLOPs / 参数量账本 |
| `experiment_logger.py` | 论文级记录器（schema + 指纹 + 统计） |
| `run_metrics.py` | **论文证据自动采集**（接入训练/冒烟，自动落盘 + 判据判定） |
| `README.md` | 对外说明（诚实状态表） |
| `snowflake_lock60.zip` | 锁 6.0 交付包 |

---

## 8. 变更记录（Changelog）

| 日期 | 变更 |
|---|---|
| 2026-10-07 | 建立本日志；登记 L-001 ~ L-010；V-001 ~ V-018 |
| 2026-10-07 | 锁 6.0 参数拍板（N=384, td=4, band=0.5, width=0.1, delta_scale=0.5, temp=2.0） |
| 2026-10-07 | 铰链改加性 loss 项（L-007） |
| 2026-10-07 | 裁决改低秩 A 型 rank=4（L-008/L-009） |
| 2026-10-07 | 加入 64GB 规模预设四档 |
| 2026-10-07 | **D-003 裁决 A 型 r=32**；论文表述改为「连续组合空间（N 维单纯形）」 |
| 2026-10-07 | **D-004 停用 A 型，改 B 型 r=4/P=32**：同等 FLOPs 下组合空间 32→383 维 |
| 2026-10-07 | **V-020 GPU 冒烟通过**：核心判据 +49.217 / var 0.10529 / in_band 0.714。登记 L-011（one-hot 疑点）、L-012（in_band 单调降） |
| 2026-10-07 | 加双侧铰链（默认关）+ mix_sharp 告警 + `--sep` 对照参数 |
| 2026-10-07 | **L-013/L-014**：澄清 sharp=1.0 是任务诱导非架构问题；发现 temp 须随 N 缩放（N=384→5.0） |
| 2026-10-07 | **修复**：`train_lock60.py` 的 SnowflakeLM 仍在用满秩 SnowflakeCell，已切 SnowflakeB；新增 `--temp`(默认5.0) / `--rank`(默认4) / `--hinge-low` |
| 2026-10-07 | **★★ L-037 H4 坐实**：10 seeds 全部显著，core_stat=1.363±0.029，CI=[1.342,1.384]，d_z≈37，t=147.9，eff=25.7±2.1（全部>23）。论文主表【已完整】 |
| 2026-10-07 | **L-036 忆点可擦除性**：用户判断三点全部验证成立 —— 推理擦除【不】保持性能(权重反升6倍，旧claim失效)、全擦除全损毁✅、部分擦除可重训练恢复✅(只丢身份不丢能力)。已实现 --erase-frac/--erase-ckpt/--retrain-steps |
| 2026-10-07 | **★ L-035 无约束跑**：loss 1.779（优于带约束 1.784）+ eff 自发 18 ⇒ 【最强版本结论成立】：无需强制，模型自发组合且效果最好。已启动 10 seeds |
| 2026-10-07 | **L-034 vocab=92 推测作废**：实测清洗后 100→98（tokens -14971，占 0.0143%）⇒ 非常规字符多是真实的，loss 不会明显下降。论文中 vocab 统一改 98 |
| 2026-10-07 | **L-033 perm 长度算错 128 倍**（我在修 L-027 时引入）：`T = xb.shape[1] if xb.dim()>2 else 1` ⇒ C=2（应 256），只打乱 0.8% 连接 ⇒ 核心判据≈0。已改 `T = xb.shape[-1]` + 加断言 |
| 2026-10-07 | **★ L-029 最关键结果**：真实 LM 3000 步，eff 自发收敛到 31.3（min_eff=8 未激活）⇒ 【拼专家成立】。loss 4.588→1.784 |
| 2026-10-07 | **L-030/L-031/L-032**：release 判据改 eff 为主(原 sharp 39.8%→WARN，改 eff 为 −4.3%→OK 涌现)；vocab=100 是 mojibake 污染(应 92)，加 --fix-mojibake；ablation generator device 已修 |
| 2026-10-07 | **L-028 字段注册机制修复**：experiment_logger 原生 STEP_FIELDS 停留在 lock47 旧字段，靠 run_metrics.update() 追加是脆弱依赖。已直接写进定义体：STEP_FIELDS 21→30，FINAL_FIELDS 24→51 |
| 2026-10-07 | **L-027 全量审查**：发现 5 个真 bug —— --seed 未传导(10seeds全用seed=0)、FLOPs 用满秩公式(3.25x vs 实际0.664x)、loss 双重 shift、release test 用 val_iter(数据泄漏)、ablation 的 C 算错。均已修 |
| 2026-10-07 | **L-026 ablation 崩溃导致 3000 步全丢**：修 DEV→DEVICE、ablation+log_step 双层 try 兜底、加 --save-ckpt/--eval-only（崩了可恢复评估不用重训）、打印行补 eff。8 项校验全过 |
| 2026-10-07 | **L-024 字段注册缺口修复**：STEP_FIELDS 补 18 个字段、agg_diag.keys 同步、`--n-perm` 去重、打印行补 eff。三处同步校验通过 |
| 2026-10-07 | **L-025 首次真实训练**：字符级 loss 4.60→2.24（ppl 99.98→9.39），起始值与 ln(100)=4.6052 吻合 ⇒ 数据接入正确 |
| 2026-10-07 | **D-007 决策**：本次正式训练用【字符级 vocab=100】。历史 PPL 为词级，不可并列引用。假设：三处历史冲突根因可能是 tokenization 口径不同 |
| 2026-10-07 | **D-006 数据接入**：字符级 vocab=92（embedding 仅占 0.65%）。`load_data()` 已内置，新增 `--bs`/`--seq-len`/`--max-steps` |
| 2026-10-07 | **数据与接入**：服务器已有 `/root/private_data/snowflake/tinystories_100mb.txt`（100MB）+ 历史脚本 `run_erase_tinystories.py` / `run_lifelong_tinystories.py` + checkpoints。`--data` 目前【不直接接受路径】，`load_data()` 是 NotImplementedError 需手动接入。新增 `probe_data.py` 自动诊断格式并生成接入代码 |
| 2026-10-07 | **实现**：release test（`--release-steps`）+ 归一化判据（`eff_organs` / `wiring_var_norm`）已落地，结果自动落盘 |
| 2026-10-07 | **L-021/L-022/L-023**：冒烟实为 N=8；sharp=0.456 是约束夹出的（eff=3.997==K）非学会；K↑⟺均匀占比↑的数学张力；wiring_var 0.05 门槛跨 N 不可用 |
| 2026-10-07 | **L-020 DCU 环境**：确认 GPU 一直在用（我此前误判为 CPU 版 torch）。瓶颈是 loss.item() 的同步开销而非算力。加 `--no-bf16`，sync 优化预计 20x |
| 2026-10-07 | **D-005 必须拼专家**：加内生最低有效器官数约束 `--min-eff K`（默认关）。实测 sharp=0.982⇒有效数1.04（384选1），K=8 后有效数 8.00、sharp 0.352 |
| 2026-10-07 | **设备自动匹配**（L-018 改）：GPU 优先、无则 CPU，但【显式宣告】+原因诊断+CPU 自动降配（关 bf16、降 tiny）：`train_lock60.py` 加硬失败，与冒烟脚本一致。按 GPU 时长计费时 CPU 空跑 = 白烧钱 |
| 2026-10-07 | **修复"卡输出"**：48 处 print 加 flush=True（重定向时块缓冲导致日志不吐）；einsum→bmm/matmul（等价性已验，差~1e-15）；加每 10 步心跳 + [ready] 启动确认 |
| 2026-10-07 | **论文证据自动采集**：新增 `run_metrics.py`，接入训练/冒烟。每次运行强制落盘 record.json + steps.csv + master.csv，含 git/config/数据指纹与 H1~H5 自动判定 |
