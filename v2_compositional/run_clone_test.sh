#!/bin/bash
# clone 机制验证实验
set -u
cd /mnt/workspace/snowflake-moe/v2_compositional
DATA=data_25mb.txt
mkdir -p output/v4

echo "══════ A. clone 本身有没有用 ══════"
python train_v4.py --data $DATA --L 4 --n-cells 4 --max-steps 3000 \
    --seed 0 --variant base  --tag clone_base  2>&1 | tail -8
python train_v4.py --data $DATA --L 4 --n-cells 4 --max-steps 3000 \
    --seed 0 --variant clone --tag clone_on    2>&1 | tail -8

echo ""
echo "══════ B. 秩的瓶颈（rank 扫描）══════"
for r in 4 8 16; do
    echo "─── rank=$r ───"
    python train_v4.py --data $DATA --L 4 --n-cells 4 --max-steps 3000 \
        --seed 0 --rank $r --variant base --tag rank_$r 2>&1 | tail -6
done

echo ""
echo "══════ C. 组合：clone + 大 rank ══════"
python train_v4.py --data $DATA --L 4 --n-cells 4 --max-steps 3000 \
    --seed 0 --rank 8 --variant clone --tag clone_rank8 2>&1 | tail -8

echo ""
echo "══════ 完成 ══════"
