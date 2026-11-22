#!/bin/bash
# paperplane-Moe V1.001 · 生物学代码落地后的第一组验收（在 v2_compositional/ 下跑）
set -uo pipefail
cd "$(dirname "$0")" || exit 1

echo "########## [1/4] 真机自检（9 条，必须全 PASS）##########"
python3 verify_bio.py 2>&1 | tee verify_bio.log
echo "  exit=$?"

echo "########## [2/4] B 档 + 观测层 20 步 smoke ##########"
python3 train_bio.py --data data_25mb.txt --d 256 --h 128 --L 4 \
  --n-cells 4 --variant shared --rank 4 --temp 5.0 --bs 8 --seq-len 128 \
  --lr 3e-4 --max-steps 20 --seed 0 --split-k 8 --split-at 10 \
  --cell-type B --tag bio_smoke_B > bio_smoke_B.log 2>&1
echo "  exit=$?"; grep -E "\[bio\]|\[saved\]|events=" bio_smoke_B.log | tail -4

echo "########## [3/4] 残差版真拼 20 步 smoke ##########"
python3 train_bio.py --data data_25mb.txt --d 256 --h 128 --L 4 \
  --n-cells 4 --variant shared --rank 4 --temp 5.0 --bs 8 --seq-len 128 \
  --lr 3e-4 --max-steps 20 --seed 0 --split-k 8 --split-at 10 \
  --cell-type residual --compose-k 16 --h-cell 3 --compose-chunk 1 \
  --clip-mode global --tag bio_smoke_residual > bio_smoke_residual.log 2>&1
echo "  exit=$?"; grep -E "\[bio\]|\[saved\]|events=|params " bio_smoke_residual.log | tail -5

echo "########## [4/4] 正式两档（确认 smoke 全 exit=0 再跑，约 10 分钟）##########"
echo "  python3 train_bio.py --data data_25mb.txt --d 256 --h 128 --L 4 \\"
echo "    --n-cells 4 --variant shared --rank 4 --temp 5.0 --bs 64 --seq-len 128 \\"
echo "    --lr 3e-4 --max-steps 3000 --seed 0 --split-k 8 --split-at 1500 \\"
echo "    --cell-type B --tag bio_s0_B --zero-check"
echo "  python3 train_bio.py --data data_25mb.txt --d 256 --h 128 --L 4 \\"
echo "    --n-cells 4 --variant shared --rank 4 --temp 5.0 --bs 64 --seq-len 128 \\"
echo "    --lr 3e-4 --max-steps 3000 --seed 0 --split-k 8 --split-at 1500 \\"
echo "    --cell-type residual --compose-k 16 --h-cell 3 --compose-chunk 1 \\"
echo "    --clip-mode global --tag bio_s0_residual"
echo "  判据：python3 step27d_bio_summary.py 2>&1 | tee step27_summary.log"
echo "BIO_SMOKE_DONE"
