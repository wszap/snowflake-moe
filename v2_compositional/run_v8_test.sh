#!/usr/bin/env bash
# v8 真·拼专家：扫描「拼几个」和「每个多宽」
# 用法： bash run_v8_test.sh [steps]
set -u
STEPS=${1:-3000}
cd "$(dirname "$0")"

DATA=data_25mb.txt
[ -f "$DATA" ] || DATA=../data_25mb.txt

echo "================================================================"
echo "  0. smoke：先验证链路通（1 分钟）"
echo "================================================================"
python3 train_v8.py --data "$DATA" --max-steps 200 --tag smoke_k8 \
    --k 8 --h-cell 32 --d 128 --L 4 --n 384 2>&1 | tail -20

if [ $? -ne 0 ]; then
  echo "smoke 失败，别往下跑了。把报错发我。"
  exit 1
fi

echo
echo "================================================================"
echo "  1. 扫描拼装粒度 k（固定 h_cell=32）"
echo "================================================================"
for K in 4 8 16; do
  echo "--- k=$K （拼后等效宽度 = ${K}×32+32）---"
  python3 train_v8.py --data "$DATA" --max-steps "$STEPS" --seed 0 \
      --k "$K" --h-cell 32 --d 128 --L 4 --n 384 2>&1 | tail -14
done

echo
echo "================================================================"
echo "  2. 扫描每忆点宽度 h_cell（固定 k=8）"
echo "================================================================"
for HC in 16 64; do
  echo "--- h_cell=$HC （拼后等效宽度 = 8×${HC}+32）---"
  python3 train_v8.py --data "$DATA" --max-steps "$STEPS" --seed 0 \
      --k 8 --h-cell "$HC" --d 128 --L 4 --n 384 2>&1 | tail -14
done

echo
echo "================================================================"
echo "  全部完成。判据："
echo "    val_loss 相对 v3 baseline(1.7288) 提升 > 0.021 才算有效"
echo "    gate(末) 若仍 ≈ 0 → 模型没启用拼装，说明挂载/初始化还有问题"
echo "================================================================"
