#!/bin/bash
# paperplaneMoe / step25 —— 真·拼专家 A/B（CRN 配对）
# 在 v2_compositional/ 下运行；必须显式 --data（RESULTS.md 的空数据坑）
set -uo pipefail

D=data_25mb.txt
BASE="--data $D --d 256 --h 128 --L 4 --n-cells 4 --variant shared \
--rank 4 --temp 5.0 --bs 64 --seq-len 128 --lr 3e-4 --max-steps 3000 \
--split-k 8 --split-at 1500 --mode split --join-mode share"

# ★ 先只跑 seed0 三档看效应量与单步耗时；确认可行再 SEEDS="0 1 2 3 4 5 6 7"
SEEDS="${SEEDS:-0}"
# ★ chunk=1 与 B 档逐位同口径但慢；若 smoke 显示 >200ms/步，改 CHUNK=32
CHUNK="${CHUNK:-1}"

run() {
  TAG="$1"; shift
  echo "########## $TAG ##########"
  python3 train_compose.py $BASE "$@" --compose-chunk "$CHUNK" --tag "$TAG" > "$TAG.log" 2>&1
  echo "  exit=$?"
  grep -E "\[saved\]|val =|params |clip:|chunk=" "$TAG.log" | tail -5
}

for S in $SEEDS; do
  # A 档：历史 B 型 + 全局 clip（应复现 b5_E1_split 的 1.1926 量级）
  run "st25_s${S}_B" \
      --seed "$S" --cell-type B --clip-mode global

  # C 档：v10 + 参数量对齐档 + 全局 clip（分离「架构」与「裁剪」）
  run "st25_s${S}_V10w48" \
      --seed "$S" --cell-type v10 --clip-mode global \
      --h-cell 3 --compose-k 16

  # D 档：v10 + 分组裁剪（主处理档，参数量与 B 档对齐）
  run "st25_s${S}_V10iso" \
      --seed "$S" --cell-type v10 --clip-mode grouped \
      --h-cell 3 --compose-k 16
done

echo "COMPOSE_AB_DONE"
