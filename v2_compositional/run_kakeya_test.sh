#!/bin/bash
# ==========================================================================
#  挂谷翻转塞 · K 扫描实验
# ==========================================================================
#  核心问题：被 rank=4 锁死的 delta，用几个翻转轴能拉回多少？
#
#  数学自检已证明（verify_kakeya_math.py）：
#      固定轴  rank 4 → 4      （反射保秩，白搭）
#      检索轴  rank 4 → 128    （K=8 即满秩）
#  但"能表示"不等于"学得会"——所以必须训练验证。
#
#  用法: bash run_kakeya_test.sh [步数]   默认 3000
# ==========================================================================
set -u
cd /mnt/workspace/snowflake-moe/v2_compositional
DATA=data_25mb.txt
STEPS=${1:-3000}
mkdir -p output/v6

echo "════════════════════════════════════════════"
echo "  0. smoke：先验证链路通（2 分钟）"
echo "════════════════════════════════════════════"
python train_v6.py --data $DATA --smoke --L 2 --n-cells 2 \
    --kakeya-k 8 --tag smoke_k8 2>&1 | tail -14

echo ""
echo "════════════════════════════════════════════"
echo "  1. 基线 K=0（等价于 v3，用来对照）"
echo "════════════════════════════════════════════"
python train_v6.py --data $DATA --max-steps $STEPS --seed 0 \
    --kakeya-k 0 --tag kak_k0 2>&1 | tail -12

echo ""
echo "════════════════════════════════════════════"
echo "  2. K 扫描"
echo "════════════════════════════════════════════"
for k in 4 8 16 32; do
    echo "─── K=$k ───"
    python train_v6.py --data $DATA --max-steps $STEPS --seed 0 \
        --kakeya-k $k --tag kak_k$k 2>&1 | tail -12
done

echo ""
echo "════════════════════════════════════════════"
echo "  3. 汇总"
echo "════════════════════════════════════════════"
python3 - << 'PYEOF'
import json, glob, os

rows = []
for f in sorted(glob.glob("output/v6/kak_k*.json")):
    rows.append(json.load(open(f)))

if not rows:
    print("没有结果文件")
else:
    base = None
    for r in rows:
        if r.get("kakeya_k") == 0:
            base = r["val_loss"]
    print(f"{'K':>4} {'val_loss':>10} {'ppl':>8} {'α均值':>8} "
          f"{'|α|':>7} {'参数':>11} {'秒':>6} {'vs K=0':>9}")
    print("-" * 72)
    for r in sorted(rows, key=lambda x: x.get("kakeya_k", 0)):
        k = r.get("kakeya_k", 0)
        vl = r["val_loss"]
        delta = f"{(base - vl):+.4f}" if base is not None else "-"
        print(f"{k:>4} {vl:>10.4f} {r['ppl']:>8.3f} "
              f"{r.get('kakeya_alpha_mean', 0):>+8.4f} "
              f"{r.get('kakeya_alpha_absmean', 0):>7.4f} "
              f"{r['params']:>11,} {r['elapsed_sec']:>6.0f} {delta:>9}")

    print()
    best = min(rows, key=lambda x: x["val_loss"])
    print(f"最小 val_loss: K={best.get('kakeya_k')}  "
          f"loss={best['val_loss']:.4f}")
    if base is not None:
        gain = base - best["val_loss"]
        print(f"相对基线 K=0 的提升: {gain:+.4f}  "
              f"({gain / base * 100:+.2f}%)")
        print()
        print("判据（seed 噪声约 ±0.021）：")
        if gain > 0.021:
            print("  ✅ 超出噪声 → 翻转塞有效，继续加 K")
        elif gain > 0.010:
            print("  ⚠  模糊区 → 需补 seed 确认")
        else:
            print("  ❌ 噪声内 → 秩不是瓶颈，得重新找原因")
PYEOF

echo ""
echo "══════ 完成 ══════"
