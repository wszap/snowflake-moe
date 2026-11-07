#!/bin/bash
set -uo pipefail
D=data_25mb.txt
BASE="--data $D --d 256 --h 128 --L 4 --n-cells 4 --variant shared --rank 4 --temp 5.0 --bs 64 --seq-len 128 --lr 3e-4 --max-steps 3000 --split-k 8 --split-at 1500 --mode split"
for S in 1 2; do
  echo "########## seed=$S  D0 dummy ##########"
  FLIP_MODE=dummy python3 step22_dummy_ab.py $BASE --seed $S --tag flip_s${S}_D0 > flip_s${S}_D0.log 2>&1
  echo "  exit=$?"
done
echo "D0_SEEDS_DONE"
