#!/bin/bash
# ============================================================
# 挂谷翻转塞 · 换挂载位置（沿序列段）—— 对照实验
#
# 三组，各 3000 步，共约 20 分钟：
#   A 基线          K=0（不启用，等价于 v3）        ~4.5 min
#   B 零初始化启用  K=8, 段数 8                     ~5 min
#   C 强行打开诊断  K=8, θ=0.5, 门=0.1              ~5 min
#
# 判定：
#   B 相对 A 提升 > 0.021（seed 噪声）⇒ 有效
#   C 用来确认「模块是不是活的」：
#       C 的起始 loss ≠ A ⇒ 模块确实在起作用
#       C 训练完 θ 被压回 0 ⇒ 模型主动关掉它 ⇒ 挂载点仍不对
# ============================================================
set -u
cd "$(dirname "$0")"

DATA=data_25mb.txt
STEPS=${1:-3000}
mkdir -p output/v7

COMMON="--data $DATA --L 4 --n-cells 4 --max-steps $STEPS --seed 0"

echo "════════════════════════════════════════════"
echo "  A. 基线 K=0（不启用翻转塞）"
echo "════════════════════════════════════════════"
python train_v7.py $COMMON --kakeya-k 0 --tag seg_k0 2>&1 | tail -12

echo ""
echo "════════════════════════════════════════════"
echo "  B. 零初始化启用  K=8, n_seg=8"
echo "════════════════════════════════════════════"
python train_v7.py $COMMON --kakeya-k 8 --n-seg 8 --tag seg_k8_s8 2>&1 | tail -12

echo ""
echo "════════════════════════════════════════════"
echo "  C. 强行打开诊断  θ=0.5  门=0.1"
echo "════════════════════════════════════════════"
python train_v7.py $COMMON --kakeya-k 8 --n-seg 8 \
    --theta-init 0.5 --gate-init 0.1 --tag seg_k8_open 2>&1 | tail -12

echo ""
echo "════════ 完成 ════════"
echo ""
echo "汇总："
python3 -c "
import json,glob
print(f\"{'tag':14} {'val_loss':>9} {'theta_abs':>10} {'inj':>8} {'out':>8}\")
print('-'*54)
for f in sorted(glob.glob('output/v7/*.json')):
    r=json.load(open(f))
    print(f\"{r['tag']:14} {r.get('val_loss',0):>9.4f} {r.get('kakeya_theta_absmean',0):>10.4f} {r.get('inj_gate',0):>8.4f} {r.get('out_gate',0):>8.4f}\")
"
