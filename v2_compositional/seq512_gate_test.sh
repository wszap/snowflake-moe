#!/bin/bash
# ==========================================================
#  筛选门生死判定：seq=512 配对实验
#
#  设计要点：
#   1. 配对设计 —— 同 seed 下 topk 与 soft 相减，消除 seed 间噪声
#   2. seq=512  —— 拉长上下文，给筛选门发挥空间
#   3. 自动探测参数名（--seq-len / --seq，--bs / --batch-size）
#   4. OOM 自动降 batch
# ==========================================================
set -u
cd /mnt/workspace/snowflake-moe/v2_compositional
DATA=data_25mb.txt
mkdir -p output/v3_seq512

# ---- 参数名自动探测 ----
HELP=$(python train_v3.py --help 2>&1)
if echo "$HELP" | grep -q -- "--seq-len"; then SEQARG="--seq-len"; else SEQARG="--seq"; fi
if echo "$HELP" | grep -q -- "--bs";      then BSARG="--bs";      else BSARG="--batch-size"; fi
echo "[参数] SEQARG=$SEQARG  BSARG=$BSARG"

STEPS=2000
BATCH=32

# ---- 阶段0：显存探测 ----
echo ""
echo "=== 阶段0: 显存探测 (seq=512, 100步) ==="
timeout 600 python train_v3.py --data $DATA $SEQARG 512 $BSARG $BATCH \
    --L 4 --n-cells 4 --max-steps 100 --seed 0 \
    --select-mode soft --tag probe 2>&1 | tail -6
if [ $? -ne 0 ]; then
    echo "[!] 探测失败，batch 降到 16 再试"
    BATCH=16
fi
echo "[配置] seq=512  bs=$BATCH  steps=$STEPS"

# ---- 阶段1：配对实验 ----
echo ""
echo "=== 阶段1: 配对实验 (3 seeds x topk/soft) ==="
for s in 0 1 2; do
    for m in topk soft; do
        echo "--- seed $s / $m ---"
        timeout 1800 python train_v3.py --data $DATA $SEQARG 512 $BSARG $BATCH \
            --L 4 --n-cells 4 --max-steps $STEPS --seed $s \
            --select-mode $m --tag s512_${m}_s${s} 2>&1 | tail -4
    done
done

echo ""
echo "=== 全部完成，开始分析 ==="
