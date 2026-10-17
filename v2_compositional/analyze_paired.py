#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配对差值分析：topk vs soft

为什么用配对？
  不同 seed 之间的 loss 差异（seed 噪声 ~0.021）远大于
  筛选门带来的差异（~0.004）。直接比较会被淹没。
  同一 seed 下 topk 与 soft 相减，seed 随机性被消掉，
  剩下的才是筛选门的真实效应。
"""
import json, glob, os, sys, statistics as st

D = "output/v3_seq512"
files = sorted(glob.glob(os.path.join(D, "*.json")))
res = {}

for f in files:
    try:
        r = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    tag = r.get("tag", "")
    if not tag.startswith("s512_"):
        continue
    parts = tag.split("_")          # s512_topk_s0
    if len(parts) < 3:
        continue
    mode, seed = parts[1], parts[2]
    res.setdefault(seed, {})[mode] = r

if not res:
    print("未找到结果文件，检查 output/v3_seq512/")
    sys.exit(1)

print()
print("=" * 66)
print("  配对差值分析：筛选门 topk(硬筛选) vs soft(全收)")
print("=" * 66)
print(f"{'seed':<8}{'topk val':>12}{'soft val':>12}{'delta':>14}")
print("-" * 66)

deltas = []
for s in sorted(res, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
    t = res[s].get("topk", {}).get("val_loss")
    f = res[s].get("soft", {}).get("val_loss")
    if t is None or f is None:
        continue
    d = t - f
    deltas.append(d)
    print(f"{s:<8}{t:>12.4f}{f:>12.4f}{d:>+14.4f}")

print("-" * 66)

if len(deltas) < 2:
    print("有效配对不足 2 组，无法做检验")
    sys.exit(0)

n = len(deltas)
m = st.mean(deltas)
sd = st.stdev(deltas)
se = sd / (n ** 0.5)
tstat = m / se if se > 0 else float("inf")

# 配对 t 检验临界值（双尾 0.05）
CRIT = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 9: 2.262}
crit = CRIT.get(n - 1, 2.0)

print(f"配对差均值 = {m:+.5f}   SD = {sd:.5f}   n = {n}")
print(f"配对 t     = {tstat:+.2f}   (临界 t({n-1}) = {crit})")
print(f"95% CI     = [{m - crit*se:+.5f}, {m + crit*se:+.5f}]")
print()

# delta < 0 表示 topk（筛选）loss 更低 = 更好
significant = abs(tstat) > crit
if significant and m < 0:
    verdict = "topk 显著更好 —— 筛选门有价值"
    mark = "[PASS]"
elif significant and m > 0:
    verdict = "soft 显著更好 —— 筛选门有害，应砍掉"
    mark = "[FAIL]"
else:
    verdict = "无显著差异 —— 筛选门是纯开销"
    mark = "[NULL]"

print("=" * 66)
print(f"  判定 {mark}: {verdict}")
print("=" * 66)
print()
print("补充说明：")
print(f"  - delta = val_loss(topk) - val_loss(soft)")
print(f"  - delta < 0 表示筛选(topk)更好")
print(f"  - seed 间噪声约 0.021；配对后 SD = {sd:.5f}")
print(f"  - 若 |delta| < 噪声量级，即使 t 显著也需谨慎解读实际价值")
print()
