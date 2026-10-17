# A10 重建指南

> 服务器已释放，从零重建。约 **1.5 小时** GPU 时间。
> 生成时间：2026-10-11

---

## 零、先记住：不用从头想，结论已经在那了

| 已保留 | 位置 |
|---|---|
| 全部代码 | `v2_compositional/` 21 个文件 |
| 10 seeds 完整数值 | `v2_compositional/RESULTS.md` |
| 37 条实验裁决 | `docs/EXPERIMENT_LOG.md` |
| 六个陷阱 + 验证脚本 | `docs/blog_moe_six_traps.md` + `verify_six_traps.py` |

**重跑是为了：① 恢复原始 steps.csv 曲线 ② 拿 dense baseline 这个从未有过的新证据。**

---

## 一、A10 与 DCU 的关键差异（务必改）

| 项 | DCU（旧） | **A10（新）** |
|---|---|---|
| `--no-bf16` | **必须加**（ROCm bf16 慢 2 倍） | **必须去掉**（Ampere 真实 Tensor Core） |
| 显存 | — | 24GB（tiny 无压力） |
| 散热 | — | 被动散热，长跑留意降频 |

**这是最容易踩的坑**：直接把 DCU 的命令复制过来会白白损失 bf16 加速。

---

## 二、工作量

| 任务 | 估算 | 必要性 |
|---|---|---|
| ① 下载数据 + 环境检查 | 2 min | 必做 |
| ② 泄漏自检 | 几秒 | **必做** |
| ③ 单次摸底 3000 步 | 3 min | 必做 |
| ④ ablation（从 ckpt） | 1 min | 必做 |
| ⑤ 10 seeds | 30 min | ★ 恢复 core_stat |
| ⑥ dense 规模扫描 | 18 min | ★ **新证据** |
| ⑦ dense 10 seeds | 30 min | ★ 配对检验 |
| ⑧ 擦除扫描（可选） | 5 min | P2 |
| **合计** | **~90 min** | |

**必做部分（①~⑤）约 36 分钟。**

---

## 三、一步步跑

### 第 0 步：上传归档包

```bash
# 本地（Windows PowerShell）
scp SNOWFLAKE_ARCHIVE.zip root@<A10>:~/
# 或用 WinSCP 图形化上传

# A10 上
unzip SNOWFLAKE_ARCHIVE.zip
cd SNOWFLAKE_ARCHIVE/v2_compositional
```

### 第 1 步：环境与数据

```bash
export DATA=/root/tinystories_100mb.txt
bash ../scripts/a10_rebuild.sh --setup
```

会自动检查 torch/CUDA/bf16，数据不存在则自动下载并截取 100MB。

### 第 2 步：泄漏自检（**别跳过**）

```bash
bash ../scripts/a10_rebuild.sh --verify
```

必须看到：
```
✅ 无泄漏（正确 causal）
```

**原因**：dense_baseline.py 之前的 causal mask 方向写反了
（`triu` + SDPA 的 True=参与语义 ⇒ 模型只看未来），导致 loss 崩到 **0.019**（标签泄漏）。
已修复为 `is_causal=True`，但**换环境必须重新验证**。

### 第 3 步：单次摸底

```bash
bash ../scripts/a10_rebuild.sh --smoke
```

检查四项：
```
[data] 行 vocab = 98        ← 清洗后
起始 loss ≈ 4.585 (= ln 98)
末段 loss ≈ 1.78
eff ≈ 23~29
```

**若起始 loss 不是 4.585** ⇒ 数据接入有问题，先别往下跑。

### 第 4 步：ablation

```bash
bash ../scripts/a10_rebuild.sh --ablation
```

目标：`core_stat` 显著为正（原值 **+1.41**）。

### 第 5 步：10 seeds

```bash
bash ../scripts/a10_rebuild.sh --seeds
```

目标：`core_stat ≈ 1.363 ± 0.029`，全部 CI 不跨 0。

### 第 6-7 步：dense baseline（**新证据**）

```bash
bash ../scripts/a10_rebuild.sh --dense-sweep
bash ../scripts/a10_rebuild.sh --dense-seeds
```

`compare_dense.py` 会自动判定 README 那句 claim 保留还是改写。

**关键配置（别改）**：
- `L=5, d=288, ff=1152` → 5,024,736 参数（**0.973x** Snowflake 的 5,163,618）
- **不加 pos embedding** —— SnowflakeLM 也没有，加了参数量虚高 9.9%，对比不公平

---

## 四、⚠ 提前预警：dense 可能会赢

这不是打击，是提前让你有心理准备、并准备好正确叙事。

### 为什么

**SnowflakeLM 没有跨 token 注意力：**

```python
xc = self._chunk_view(x)     # [C, P, d]  按 32 切块
xsum = xc.mean(1)            # [C, d]     ← chunk 内平均
wiring = softmax(xsum @ key.T ...)   # 每 chunk 一个 wiring
hh = matmul(xc, W1_base) + bmm(xA, W1_V)   # per-token
```

每 token 独立变换，**唯一上下文是所在 chunk（32 字符）的平均**。

而 dense 修好 causal mask 后有**完整因果注意力，能看前 127 个 token**。

### 预期

| | 上下文 | 预期 loss |
|---|---|---|
| 随机基线 | 无 | 4.585 |
| Snowflake | chunk 平均（32 字符） | **1.788** |
| **dense（修好后）** | **完整因果（127）** | **1.4~1.6** |

### 所以

README 这句**几乎确定不成立**：

> "achieving strong generalization under **a fraction of the parameters of a dense baseline**"

### 正确的叙事（换成这张牌）

**不要打「参数更少还更好」，改打「效率-精度前沿」：**

> Snowflake 用 **0.664x 的 FLOPs** 达到 loss 1.788；
> dense 需要 **X 倍 FLOPs** 才能达到同等 loss。

- 若 X > 1 ⇒ claim 成立且诚实
- 若 X < 1 ⇒ 如实写「Snowflake 以精度换取了 FLOPs 效率」

**并且**：「上下文机制不对等是主因」本身是**有价值的发现**，值得写进 limitations。

### 可选补救

想让精度对比公平，可给 Snowflake 加轻量上下文（chunk 间 GRU 或 sliding-window attention）。但那是改架构，**需要你拍板**。

---

## 五、验证清单

```
□ torch.cuda.is_available() = True
□ torch.cuda.is_bf16_supported() = True
□ 泄漏自检 ✅ 通过
□ [data] vocab = 98
□ 起始 loss ≈ 4.585
□ 末段 loss ≈ 1.78
□ eff ≈ 23~29
□ core_stat ≈ 1.363 ± 0.029（10 seeds）
□ dense 规模扫描完成
□ compare_dense.py 输出 claim 判定
□ 证据已下载到本地（★ 这次别再丢了）
```

---

## 六、★ 这次一定要实时备份

服务器释放一次就丢了全部原始数据。**这次务必**：

```bash
# 每跑完一批就同步一次（不要等最后）
rsync -avz output/ <本地或网盘路径>/
# 或
tar -czf backup_$(date +%m%d_%H%M).tar.gz output/ && scp ...
```

**关键数字也要随时抄进 `RESULTS.md`** —— 这次正是因为提前抄了，10 seeds 的结论才没丢。

---

## 七、如果 A10 也跑不动

退路：直接写论文。

H2/H3/H4 的数值**都在 `RESULTS.md` 里**，不依赖任何服务器。
H5（FLOPs 0.664x）是理论计算，本来就不用跑。

只需在方法部分注明：
> "Per-step diagnostics were logged during training; this paper reports
> aggregate statistics (mean ± SD across 10 seeds)."
