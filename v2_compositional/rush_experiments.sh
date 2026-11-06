#!/bin/bash
# v3 急救实验：优先 H2 统计证据，再跑筛选门消融
set -u
cd /mnt/workspace/snowflake-moe/v2_compositional
DATA=data_25mb.txt
mkdir -p output/v3

echo "════════════════════════════════════════"
echo "  轨道A：H2 统计证据（3 seeds，优先）"
echo "════════════════════════════════════════"
for s in 0 1 2; do
    echo "─── H2 seed $s ───"
    timeout 900 python train_v3.py --data $DATA \
        --L 4 --n-cells 4 --max-steps 1500 --seed $s \
        --select-mode topk --tag h2_s$s 2>&1 | tail -6
done

echo ""
echo "════════════════════════════════════════"
echo "  轨道B：筛选门消融（topk/soft/none）"
echo "════════════════════════════════════════"
for m in topk soft none; do
    echo "─── select_mode=$m ───"
    timeout 900 python train_v3.py --data $DATA \
        --L 4 --n-cells 4 --max-steps 1500 --seed 0 \
        --select-mode $m --tag sel_$m 2>&1 | tail -6
done

echo ""
echo "════════ 全部完成 ════════"
