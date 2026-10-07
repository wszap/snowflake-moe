# Snowflake MoE v2.0 — Compositional（忆点即连接）

> **忆点定义（v2.0）**：忆点**就是连接本身**。
> 器官身份由 `organelle_memory` 决定 → 高斯带通生成 `connect` → 折叠进路由 `wiring`。
> ⚠️ 与 v1.0 的「独立值槽」是**不同架构下的不同定义**，请勿混用。

---

## 架构：为什么是「拼」不是「选」

```python
U1 = matmul(wiring, W1_U.view(N,-1)).view(C,d,r)   # 先按 wiring 合成参数
xA = bmm(xc, U1)                                    # 再前向一次
```

**非线性在求和之后** —— 这是「拼专家」的数学定义，与 Top-K 的「选专家」本质不同。

---

## 主表：五条假设全 PASS

完整数据见 [RESULTS.md](RESULTS.md)。

| 假设 | 数值 | 统计 | 判定 |
|---|---|---|---|
| H1 架构是拼专家 | 数学事实 | — | ✅ |
| H2 自发组合 | eff = **25.73 ± 2.06** | 10 seeds，全部 > 23 | ✅ |
| H3 涌现而非强制 | loss **1.779**（无约束）vs 1.784 | 无约束更优 | ✅ |
| H4 分工真实存在 | core_stat = **+1.363 ± 0.029** | CI[1.342, 1.384]，d_z≈37 | ✅ |
| H5 效率碾压 | FLOPs **0.664x** | vs Top-K MoE | ✅ |

**H4 单样本 t 检验**：t = 147.9，df = 9，p ≪ 0.001。

### 最关键的一点

`--min-eff 8` 约束在真实 LM 任务上**根本没激活**（eff 25.7 ≫ K=8 ⇒ a=0）。
**模型自发使用约 26 个器官，且不加约束时 loss 更低。**

这比「加了约束才组合」强得多 —— 不需要任何强制机制。

---

## 参数量（逐项核算）

| 组件 | 计算 | 参数量 |
|---|---|---|
| embed | 98×128 | 12,544 |
| in_proj | 128×128 | 16,384 |
| head | 128×98+98 | 12,642 |
| **每个 SnowflakeB** | | **320,128** |
| └ memory | 384×128 | 49,152 |
| └ W1_U | 384×128×4 | **196,608** |
| └ W2_U | 384×32×4 | 49,152 |
| └ 其余 | | 25,216 |
| 细胞层 | 16 × 320,128 | 5,122,048 |
| **合计** | | **5,163,618** |

> ⚠️ 早期文档里的「3.6M」是错的。也**不是** v1.0 的 1,823,077（那是词级 CellMoE）。

---

## 效率

| 规模 | FLOPs 比值 | 含义 |
|---|---|---|
| tiny | **0.664x** | 便宜 33.6% |
| mid | 0.457x | 便宜 54.3% |
| large | 0.385x | 便宜 61.5% |

⚠️ **理论计算，未经 wall-clock 实测**。当前 DCU 算力利用率约 7.4%，
瓶颈在 kernel launch / sync，理论优势未必兑现为实际加速。

---

## Reproduction

```bash
# 摸底
python train_lock60.py --scale tiny \
    --data <TinyStories.txt> \
    --delta-scale 0.5 --no-bf16 --fix-mojibake \
    --max-steps 3000 --seed 0 --save-ckpt ckpt.pt

# ablation（从 ckpt 恢复，不用重训）
python train_lock60.py --scale tiny --no-bf16 --eval-only ckpt.pt

# dense baseline 对照
python dense_baseline.py --data <TinyStories.txt> --scale-sweep
python dense_baseline.py --data <TinyStories.txt> --L 5 --d 288 --sweep
python compare_dense.py --dense output/dense

# 六个陷阱验证
python verify_six_traps.py
```

## 数据清洗

TinyStories 原始数据含 mojibake（UTF-8 双重编码）：

```
… (U+2026) → utf8 e2 80 a6 → latin-1 误读 "â\x80¦" → 再存 utf8 c3a2c280c2a6
```

`--fix-mojibake` 做 cp1252 round-trip 修复：**vocab 100 → 98**。
影响极小（0.06% token），loss 改善 0.005。

---

## 已知局限

1. **FLOPs 0.664x 是理论值**，未实测 wall-clock。
2. **字符级口径**，与 v1.0 词级 PPL **不可比**（loss 1.788 ⇒ ppl ≈ 5.95）。
3. **陷阱 2、4 的幅度仍为单 seed**，机制已验证（`verify_six_traps.py` 可复跑），
   具体数值需补 8~13 seeds。
