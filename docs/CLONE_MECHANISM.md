# 复制-联结机制（Clone & Link）

> 用户 2026-10-17 明确的核心机制。本文记录**实现**与**数学验证**，
> 包括一条推翻前一轮诊断的结论。

---

## 一、机制：严格按描述实现

代码位置：`snowflake_v4.py` 的 `SnowflakeBClone`，启用方式 `--variant clone`

```
① 需求   活跃细胞器从【当前问题】提出 demand
         demand = x @ demand_probe

② 选择   细胞器【自己】从细胞器库里挑想复制的标本
         spec_w = softmax(demand @ key^T / √d × clone_temp)
         key = 忆点 + 忆点互相关联（c.T @ m）

③ 复制   忆点执行 —— 加权复制标本能力 + 可学习变体
         U_clone = Σ_j spec_w[j] · (W1_U[j] × clone_scale[j])

④ 联结   忆点把【提出者】与【复制体】绑成临时团队
         link = sigmoid(link_gate + spec_w @ link_probe)
         U_team = U_self + link · U_clone

⑤ 求解   团队一起解决【同一个】问题
         xA = x @ U_team → hh = xA @ W1_V
```

**与标准 MoE 的本质区别**

| | 标准 MoE | 本机制 |
|---|---|---|
| 谁决定组合 | router（外部指派） | 细胞器自己提出需求 |
| 专家来源 | 固定的 k 个 | 从**库里复制**标本当队友 |
| 谁执行组合 | router 输出权重 | **忆点**执行复制并联结 |
| 专家间关系 | 各干各的，加权求和 | 被忆点联结成**团队** |

---

## 二、数学验证（numpy，不需要 GPU）

脚本：`verify_clone.py`、`verify_rank_bound.py`

### 验证①：clone 是否提升「单次改造能力」—— **没有**

```
复合变换 = U_team @ W1_V，形状 [d, h]

  rank   无clone   +clone(16个不同标本)   上限
     4       4            4               4
     8       8            8               8
    16      16           16              16
    32      32           32              32
    64      32           32              32   ← 卡在 h=32
```

**秩 = `rank`（低秩因子宽度），与 eff 无关，与 clone 也无关。**

原因：`U_team` 是 `[d, r]` 矩阵，秩天然 ≤ r = 4。
两个 `[d,r]` 矩阵相加仍然是 `[d,r]`，秩不可能超过 r。

### ⚠️ 这条推翻了前一轮诊断

前一轮我说「有效秩 ≈ eff × rank ≈ 11，对比 dense 的 128」。
**这个说法是错的。**

正确的秩：

```
拼专家路径的秩 ≤ rank = 4        （低秩因子硬限制）
dense FFN 的秩 ≤ d = 128         （全秩）

⇒ 差距是 4 : 128 = 32 倍，不是 11 : 128
```

**clone 无法改变这个数字。**

### 验证②：什么才能真正提升秩

```
  rank=4   n_cells  复合秩   上限
              1        4      4
              2        8      8
              4       16     16
              8       32     32   ← 卡在 h=32
```

- **加大 `rank`** → 秩线性提升，但合成项 `n·d·r/P` 线性涨
  ```
  r=4  → 0.664x   ✅
  r=8  → 1.078x   ⚠
  r=16 → 1.906x   ❌ 超过 dense
  ```
- **并联更多 `n_cells`** → 秩叠加到 `n_cells × r`，但上限卡在 `h`
- **加大 `h`** → 抬天花板（当前 h=32）

**核心矛盾**：要秩 → 加 rank → FLOPs 涨 → 失去效率优势。

### 验证③：clone 真实带来什么

```
离散配对数：384 × 384 = 147,456   （v3 只有 384）
```

clone 扩大的是**可达变换的集合**（两组单纯形的 Minkowski 和），
即「能表达多少种不同的秩-4 变换」——从 384 种涨到 147,456 种配对。

它提升的是**条件计算的丰富度**，不是**单次变换的秩**。

### 验证④：无泄漏

`spec_w` 来自当前 token 的 `x`，`chunk_size=1` 时 `xsum = x` 本身，
结构上不可能看到未来。**仍需在 A10 实测 `leak_check = 0`。**

---

## 三、一句话总结

> **clone 实现了你要的「细胞器自己组队」，但它不是「单次改造能力弱」的解药。**
>
> 秩被 `rank=4` 锁死，clone 改变不了；
> 真正的旋钮是 `rank`、`n_cells`、`h`，而 `rank` 一涨 FLOPs 就输给 dense。

---

## 四、实验建议

### 必跑：确认 clone 本身有没有用

```bash
python train_v4.py --data data_25mb.txt --L 4 --n-cells 4 \
    --max-steps 3000 --seed 0 --variant clone --tag v4_clone
python train_v4.py --data data_25mb.txt --L 4 --n-cells 4 \
    --max-steps 3000 --seed 0 --variant base  --tag v4_base
```

看三个数：`final_loss`、`clone_eff`、`clone_link`

### 关键：解开秩的枷锁（这才是瓶颈）

```bash
# rank 扫描（最直接的因果检验）
for r in 4 8 16; do
  python train_v4.py --data data_25mb.txt --max-steps 3000 --seed 0 \
      --rank $r --variant base --tag rank_$r
done

# h 扫描（抬天花板）
for hh in 32 64 128; do
  python train_v4.py --data data_25mb.txt --max-steps 3000 --seed 0 \
      --h $hh --variant base --tag h_$hh
done
```

### 组合：clone + 大 rank

```bash
python train_v4.py --data data_25mb.txt --max-steps 3000 --seed 0 \
    --rank 8 --variant clone --tag clone_rank8
```

### 盯紧这两个诊断

- `clone_eff`：复制了几种标本
- **spec 与 wiring 的重合度**（需新增诊断）——若两者选到同一批标本，
  团队退化为「自己干两遍」，clone 完全白费

---

## 五、诚实边界

1. **clone 不提升秩**，只扩大可达变换的多样性。
2. **秩瓶颈是 `rank=4`**，提升它的代价是 FLOPs 超过 dense。
3. clone 的价值（147,456 种配对）**尚未被实验证实能转化为 loss 下降**，
   需要实测。
4. 本文所有数学结论由 numpy 验证，非 GPU 实测。
