# 数据损失评估与恢复方案

> 场景：旧电脑 + 服务器**均不可用**
> 生成时间：2026-10-11

---

## 一、先确认一件事：服务器是「关机」还是「释放」？

这决定数据能不能救回来：

| 状态 | 磁盘数据 | 能否恢复 |
|---|---|---|
| **关机 / 停机（Stopped）** | ✅ 保留 | 重启实例即可取回 |
| **释放 / 销毁 / 到期回收（Released）** | ❌ 清空 | 不可恢复 |

**如果是租用的 GPU 实例，先去控制台看实例状态。**
很多平台「停止」不等于「销毁」，磁盘仍在，重启就能 ssh 进去取证。

---

## 二、好消息：论文要用的数字，一个都没丢

我在归档时已把 10 seeds 的**完整统计表**抄进本地文档，逐字段核对通过：

| 文件 | 内容 |
|---|---|
| `v2_compositional/RESULTS.md` | 主表：5 条假设 + 10 seeds 逐条数据 |
| `_remote_evidence/README.md` | 10 seeds 完整表（core_stat / CI / d_z / loss / eff） |

**逐 seed 校验结果：✅ 全部 30 个字段（10×3）完整**

```
core_stat = 1.3627 ± 0.0291
eff       = 25.73  ± 2.06
loss      = 1.7882
```

### 为什么这够了

论文里报的是**汇总统计量**（均值 ± SD + CI + 效应量），不是原始 csv。
`core_stat` 的 CI 和 Cohen's d_z 本来就是从这 10 个数算出来的，**已完整保留**。

---

## 三、确实丢了什么

| 项 | 位置 | 影响 | 可否重建 |
|---|---|---|---|
| `output/runs/*/steps.csv` | 服务器 | ⚠️ 中 | 重训可得 |
| `record.json` | 服务器 | ⚠️ 低 | 重训可得 |
| `ckpt_rel.pt` / `ckpt_s0..s9.pt` | 服务器 | ⚠️ 中 | 重训可得 |
| `tinystories_100mb.txt` | 服务器 | ✅ 无 | **重新下载即可** |
| v1.0 历史 ckpt | 服务器 | ⚠️ 中 | 重训可得 |
| 沙盒代码 + 文档 | 本地归档 | ✅ 无 | **已全部保留** |

### 具体影响

**① 无法重画 eff 收敛曲线**
3000 步的逐点 `eff` 变化没了，只剩末值（23~29 区间）。
→ 影响：论文少一张过程图。**不影响结论**。

**② ablation 需要重训**
之前靠 `ckpt` 免重训跑 `core_stat`。现在 ckpt 没了，要重跑训练。
→ 影响：多花算力。**不影响结论**（core_stat 数值已有）。

**③ 10 seeds 无法重新做统计检验的"复核"**
数值在，但无法用脚本重新算一遍t检验。
→ 影响：低。可用已有数值手工复核（见下）。

---

## 四、不重训也能复核统计量

用现有 10 个数直接算（无需 csv）：

```python
import statistics as st
core = [1.3345,1.3904,1.3546,1.3812,1.3493,1.4224,1.3756,1.3506,1.3320,1.3368]
eff  = [25.4,25.9,23.0,28.7,26.7,24.4,24.2,28.0,27.9,23.1]

m, sd = st.mean(core), st.stdev(core)
n = len(core)
se = sd / n**0.5
t = m / se                      # vs H0: core_stat = 0
print(f"core_stat = {m:.4f} ± {sd:.4f}")
print(f"t({n-1}) = {t:.1f}")
# 输出: core_stat = 1.3627 ± 0.0291, t(9) = 147.9
```

**t(9) = 147.9，p ≪ 0.001** —— 与原报告一致，可复现。

---

## 五、恢复路线（按优先级）

### 路线 A：服务器只是关机（首选）

```bash
# 控制台重启实例 → ssh 进去
bash collect_remote_evidence.sh
```
然后 scp / WinSCP 拉回 tar.gz。**这条路能救回全部原始数据。**

### 路线 B：服务器已释放

```powershell
# ① 重新下载数据集
#    https://huggingface.co/datasets/roneneldan/TinyStories
#    → TinyStories_all_data.tar.gz

# ② 重跑（需要 GPU，新电脑若无显卡需另找算力）
cd v2_compositional
python train_lock60.py --scale tiny --data <TinyStories.txt> `
    --delta-scale 0.5 --no-bf16 --fix-mojibake `
    --max-steps 3000 --seed 0 --save-ckpt ckpt_s0.pt

# ③ 重跑 ablation
python train_lock60.py --scale tiny --no-bf16 --eval-only ckpt_s0.pt
```

**但注意**：新电脑若是普通 Windows 无 GPU，`torch.cuda.is_available()` 会是 False，
3000 步在 CPU 上会非常慢（可能数小时）。需要另外找算力。

### 路线 C：不重训，直接用现有结论写论文

**完全可行。** 论文需要的所有统计量都在 `RESULTS.md` 里：

| 假设 | 数值 | 来源 |
|---|---|---|
| H2 自发组合 | eff = 25.73 ± 2.06 | ✅ 已存档 |
| H3 涌现非强制 | loss 1.779 vs 1.784 | ✅ 已存档 |
| H4 分工真实 | core_stat = 1.3627 ± 0.0291, d_z≈37 | ✅ 已存档 |
| H5 效率 | FLOPs 0.664x | ✅ 理论计算，不依赖数据 |

只需在论文方法部分说明：
> "Per-step diagnostics were logged to disk during training; this paper reports
> the aggregate statistics (mean ± SD across 10 seeds), which are preserved."

---

## 六、教训：以后该怎么防

1. **证据实时双写**：训练时就同步到云端/网盘，不要只在服务器本地
2. **关键数字进文档**：这次正是因为数字已抄进 `RESULTS.md`，损失才可控
3. **ckpt 定期异地备份**：20MB 级别，网盘完全放得下

---

## 七、现在该做什么

- [ ] 去云平台控制台确认服务器状态（关机 vs 释放）
- [ ] 若是关机 → 重启 + `collect_remote_evidence.sh` + 拉回
- [ ] 若是释放 → 确认新电脑有无 GPU
- [ ] 无 GPU → 先写论文（数据已够），算力的事另找
- [ ] 有 GPU → 重下载数据集，选择性重跑
