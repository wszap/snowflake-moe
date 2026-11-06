#!/bin/bash
set -u
cd /mnt/workspace/snowflake-moe/v2_compositional
DATA=data_25mb.txt
mkdir -p output/v3

# 探测 train_v3.py 支持哪些参数
HELP=$(python train_v3.py --help 2>&1)
HAS_SEQ=$(echo "$HELP" | grep -c -- "--seq-len")
HAS_BS=$(echo "$HELP"  | grep -c -- "--bs")
HAS_K=$(echo "$HELP"   | grep -c -- "--topk\|--k")
echo "参数支持: seq_len=$HAS_SEQ bs=$HAS_BS topk=$HAS_K"

run_one () {
    local mode=$1 bs=$2 extra=$3
    local args="--data $DATA --L 4 --n-cells 4 --max-steps 3000 --seed 0"
    args="$args --select-mode $mode --tag seq512_$mode"
    [ "$HAS_SEQ" -gt 0 ] && args="$args --seq-len 512"
    [ "$HAS_BS"  -gt 0 ] && args="$args --bs $bs"
    [ -n "$extra" ] && args="$args $extra"

    echo "─── $mode (bs=$bs) $extra ───"
    timeout 2400 python train_v3.py $args 2>&1 | tee -a seq_ab_full.log | tail -8
    if grep -q "out of memory" seq_ab_full.log; then
        echo "OOM! 降 bs 重试"
        return 1
    fi
    return 0
}

echo "══════ 长上下文筛选门对照 (seq=512, 3000步) ══════"
run_one topk 32 "" || run_one topk 16 "" || run_one topk 8 ""
run_one soft 32 "" || run_one soft 16 "" || run_one soft 8 ""

echo ""
echo "══════ 完成 ══════"
