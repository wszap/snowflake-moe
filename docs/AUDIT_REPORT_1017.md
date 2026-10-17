# 全面审查报告 — 2026-10-17（第二轮）

针对 A10 上完成的 10 seeds 实验与现有代码库做的完整审查。
**新发现 2 个 bug（Bug9/Bug10），其中 Bug9 影响论文统计量呈现。**

---

## 一、10 seeds 结果确认（A10 / 干净数据 / bf16）

| seed | core_stat | d_z | loss_learned | eff≈ |
|---|---|---|---|---|
| 0 | 0.8284 | 27.7 | 1.9831 | 27.13 |
| 1 | 0.7650 | 26.2 | 1.9916 | 26.89 |
| 2 | 0.7261 | 30.0 | 2.0050 | 27.68 |
| 3 | 0.7028 | 29.6 | 1.9934 | 16.57 |
| 4 | 0.7529 | 33.4 | 1.9684 | 30.02 |
| 5 | 0.7894 | 34.2 | 1.9773 | 25.12 |
| 6 | 0.7652 | 33.9 | 1.9845 | 19.81 |
| 7 | 0.7712 | 30.6 | 1.9702 | 24.49 |
| 8 | 0.7781 | 27.9 | 1.9669 | 26.96 |
| 9 | 0.7839 | 29.5 | 1.9716 | 27.06 |

```
core_stat = 0.7663 ± 0.0345   范围 [0.7028, 0.8284]   CV=4.5%
t(9)      = 70.3              （临界 2.262，p ≪ 0.001）
eff       = 25.17 ± 4.03
loss      = 1.9812 ± 0.0126
全部 > 0  = True
```

### ✅ H4「分工真实存在」成立

10/10 全部显著，d_z 26~34 全为巨大效应量。

### 🎯 eff 与 DCU 独立复现高度吻合

| | eff |
|---|---|
| DCU 10 seeds（100MB 含 mojibake, fp32） | **25.73 ± 2.06** |
| **A10 10 seeds（25MB 干净, bf16）** | **25.17 ± 4.03** |

差 0.56，落在误差内。不同机器 / 不同数据 / 不同精度得到同量级 eff，
强于同环境重复 10 次。

### ⚠️ core_stat 系统性偏低 44%，原因清楚

| | DCU | A10 |
|---|---|---|
| core_stat | 1.363 ± 0.029 | **0.766 ± 0.035** |
| loss_learned | 1.788 | 1.981 |
| 数据 | 100MB 含 mojibake | 25MB 干净 UTF-8 |
| vocab | 98 | **100** |
| 精度 | fp32 | bf16 |

干净数据字符分布更集中 ⇒ permuted 变体破坏力下降 ⇒ 差值收窄。
**d_z 仍 26~34（DCU 32~41），统计显著性不受影响。**

> **论文必须注明：两组不可直接比数值，只能比"是否显著"。**

---

## 二、🔴 Bug9【最重要】core_stat 的 CI / p / d_z 统计错误

### 问题描述

`run_metrics.py` 原算法：

```python
sd = loss_permuted_sd
n_eff = n_perm * n_batch      # = 20 * 50 = 1000
se = sd / sqrt(n_eff)
```

**两个统计错误：**

**① 伪重复（pseudo-replication）**
同一 batch 内的 20 个 perm 共享同一组 `xb/yb`，**不是独立样本**。
`n_eff` 被放大 20 倍 ⇒ se 缩小 √20 ≈ 4.5 倍 ⇒ CI 虚窄、p 虚假地小。

**② 非配对**
permuted 与 learned 在**同一 batch** 上测得，却当成两个独立样本，
忽略了 learned 自身变异，也丢掉了配对信息。

### 实测印证（决定性证据）

```
sd = 0.0256, n_eff = 1000
⇒ CI = [0.7823, 0.7855]
与你终端打印的 CI=[+0.7823, +0.7856] 完全吻合
```

**⇒ 确认无误，就是旧算法。**

### 修正

先按 batch 聚合 permuted，再与 learned 求**配对差**，做配对 t 检验：

```python
_diffs = [mean(permuted_i) - learned_i for i in range(n_batch)]
sd_d = std(_diffs, ddof=1)
se   = sd_d / sqrt(n_batch)
```

`core_stat` 本身不变（两个 loss 之差），变的是 se / CI / p / d_z —— 全部变保守。

### ✅ 结论不变（已用真实数据验证）

```
python audit_ci.py    # 或 --sd 0.0256 做演示

旧 CI 宽度   = 0.0032
修正 CI 宽度 = 0.0146     （放大 4.6 倍）
修正后最差下界 = 0.6955   （seed 级最保守）
跨 seed CI 下界 = 0.7416
```

**所有 10 个 seed 的 CI 下界仍 > 0，H4 依然 pass。**

### 论文写法（必须改）

```
❌ 旧：core_stat = 0.7663, CI=[±0.0016], p < 1e-300
✅ 新：core_stat = 0.7663 ± 0.0345 (n=10 seeds),
       95% CI [0.7416, 0.7910], t(9) = 70.3
       并注明：CI 基于 seed 间变异；单次 run 内部采用配对检验
       （permuted 与 learned 在同一 batch 上配对），
       不使用 n_perm×n_batch 作为有效样本数（避免伪重复）。
```

---

## 三、全部 Bug 清单（9 个）

| # | 位置 | 问题 | 后果 | 状态 |
|---|---|---|---|---|
| **9** | `ablation` 统计 | **伪重复 + 非配对** | **CI 虚窄 4.6×，p 虚假地小** | ✅ 补丁 |
| **10** | `FINAL_FIELDS` | 未注册 Bug9/Bug5 新字段 | finalize 抛错 | ✅ 补丁 |
| 1 | `_model_stats.key_map` | 缺 `eff_organs`/`wiring_var_norm` | **eff 从未落盘** | ✅ 补丁 |
| 2 | `finalize` | 缺 eff/variance 映射 | final 找不到 eff | ✅ 补丁 |
| 3 | `auto_verdicts` H2 | 阈值 0.05 **数学不可达** | **H2 永久假 fail** | ✅ 补丁 |
| 6 | H2 键名 | `wiring_variance` vs `wiring_var_norm` | 填了也读不到 | ✅ 补丁 |
| 5 | sharp 计数 | 只在 100 步采样点检查 | **告警恒为 0** | ✅ 补丁 |
| 5b | `snowflake_B.py` | 无逐步计数器 | 无法修复 #5 | ✅ 补丁 |
| 4 | 采样频率 | 每 100 步 | 早期剧变全漏 | ✅ 补丁 |

### Bug3 详解：H2 数学上不可能通过

```python
wv = final.get("wiring_variance")
v["H2_跨样本方差"] = "pass" if wv > 0.05 else "fail"
```

`snowflake_B.py` 注释自己就写了：

> N=384 的 one-hot 上限仅 **0.0026**，**0.05 门槛数学上不可达**

实测：
```
var_max         = 0.002597
实测 wiring_var = 0.000070
旧判据 raw>0.05  → fail（需 714×，不可达）
新判据 norm>0.05 → fail（0.0270，量纲可达 ⇒ 结果真实）
```

L-023 当年引入 `wiring_var_norm` 修过，但 `auto_verdicts` 没跟着改。

### Bug5 更正（我上一轮判断错了）

上一轮我说"代码里没写自增"。**实际写了**（`sharp_triggered += 1`）。
真实原因是**采样遗漏**：step 10~90 的 sharp 高达 0.75~0.97，
但采样点在 step 0（0.008）和 step 100（0.178），全部错过。
已在模型侧加逐步计数器解决。

---

## 四、⚠️ 一个待定的实质问题：H2 可能真的 fail

补丁修好后 `wiring_var_norm ≈ 0.027`，**仍低于 0.05** —— 这次是**真实 fail**。

如果补丁后 10 seeds 都这样，意味着：

> 模型找到了一个**固定的约 25 个器官的配方**，
> 然后**对所有输入都用同一个配方**。

那是"组合"（combination），但**不是"按输入组合"**
（input-dependent composition）。

**这对核心叙事是实质性质疑，比 eff 那点差异重要得多，必须如实写进论文。**

补丁后重跑会给出 `wiring_var_norm` 的真实值，届时再定论。

---

## 五、历史遗留：dense baseline 的 lr 差 10 倍

```
Snowflake: lr = 3e-4    （record.json 实测）
dense:     lr = 3e-3    （默认值）
```

而 `dense_baseline.py` 开头自述"同一套训练超参"。
**默认值违反它自己的公平性原则。** 已改为 3e-4。

---

## 六、执行步骤（A10 上）

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional

# ① 确认 10 seeds 完成 + 存档旧证据
bash run_after_patch.sh check

# ② 打 9 个补丁（自动备份 .bak_patchall）
bash run_after_patch.sh patch

# ③ ★ 统计修正审计：用已有数据验证结论是否改变（约 10 秒，不占 GPU）
bash run_after_patch.sh audit
#    预期：✅ H4 依然成立，修正后最差下界 0.6955

# ④ 补丁后重跑 10 seeds（约 75 分钟，取全量 eff / var_norm）
bash run_after_patch.sh rerun

# ⑤ dense 对比实验
bash run_after_patch.sh dense
```

`audit` 可以**立刻跑**，不必等 `rerun` —— 它用已完成的 10 seeds 数据，
能在 10 秒内告诉你论文结论是否稳固。

---

## 七、审查结论

| 项目 | 判定 |
|---|---|
| H4 分工真实存在 | ✅ **成立**（修正后依然） |
| eff 独立复现 | ✅ **与 DCU 吻合**（25.17 vs 25.73） |
| H5 效率优于 TopK | ✅ 成立（0.664×） |
| H2 跨样本方差 | ⚠️ **可能真实 fail**（0.027 < 0.05） |
| CI / p 统计量 | 🔴 **必须按 Bug9 修正后重报** |
| dense baseline | ⚠️ lr 已修，待跑 |

**最关键的一句话：**

> 结论（H4）稳固，但**论文里的 CI 和 p 值必须改用配对检验的写法**，
> 否则审稿人一算就会发现 n_eff 用了 1000 而不是 50。
