#!/bin/bash
# 免费超参扫描 —— 不需要任何新代码，直接用现有 train_v3.py
#
# 目的：验证「有效秩瓶颈」假设
#   细胞器有效秩 ≈ rank × eff
#   当前 rank=4, eff≈2.8  → 有效秩 ≈ 11
#   dense FFN 全秩         → 有效秩 = d
#
# 因此按优先级扫：rank（最便宜的有效秩提升）→ temp（提高 eff）→ d（加宽残差流）
set -u
cd /mnt/workspace/snowflake-moe/v2_compositional
DATA=data_25mb.txt
mkdir -p output/v3

run () {
    local name=$1
    shift
    echo ""
    echo "═══════ $name ═══════"
    timeout 2400 python train_v3.py \
        --data "$DATA" --L 4 --n-cells 4 --max-steps 3000 --seed 0 \
        --select-mode soft --tag "sw_$name" "$@" 2>&1 \
        | tee -a sweep_free_full.log | tail -7
}

echo "════════ 免费超参扫描开始 ════════"

# ---- ① rank：直接提高单器官秩，最便宜的有效秩提升 ----
run rank16   --rank 16
run rank32   --rank 32
run rank64   --rank 64

# ---- ② temp：降低温度 → 激活更多器官 → 提高 eff ----
run temp20   --temp 2.0
run temp10   --temp 1.0

# ---- ③ 组合：rank + temp 一起 ----
run r32t20   --rank 32 --temp 2.0

# ---- ④ d：加宽残差流 ----
run d192     --d 192
run d256     --d 256

echo ""
echo "════════ 扫描完成 ════════"
