#!/bin/bash
# v4 架构变体 —— 需要 snowflake_v4.py / train_v4.py
#
# 全部保留「拼专家」核心（忆点 + 384细胞器 + 无 router 自发分工 + 核架），
# 只在周边加改进：
#   base   : 基线（应 ≈ v3 的 1.7066）
#   shared : 共享全秩专家 —— 补有效秩
#   pair   : 二阶协作 (w@P)⊙(w@Q) —— 器官两两协作，O(n·d)
#   gate   : 细胞门控 —— 4 个 cell 由平均改可学习加权
#   all    : 三者全开
set -u
cd /mnt/workspace/snowflake-moe/v2_compositional
DATA=data_25mb.txt
mkdir -p output/v4

run () {
    local name=$1
    local variant=$2
    shift 2
    echo ""
    echo "═══════ variant=$name ═══════"
    timeout 2400 python train_v4.py \
        --data "$DATA" --L 4 --n-cells 4 --max-steps 3000 --seed 0 \
        --select-mode soft --variant "$variant" --tag "v4_$name" "$@" 2>&1 \
        | tee -a variants_full.log | tail -7
}

echo "════════ v4 变体实验开始 ════════"

run base   base
run shared shared
run pair   pair
run gate   gate
run all    all

echo ""
echo "════════ 完成 ════════"
