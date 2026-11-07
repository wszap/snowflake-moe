#!/bin/bash
D=data_25mb.txt
BASE="--data $D --d 256 --h 128 --L 4 --n-cells 4 --variant shared --rank 4 --temp 5.0 --bs 64 --seq-len 128 --lr 3e-4 --max-steps 3000 --split-k 8 --split-at 1500"
for S in 1 2; do
  for spec in "split share ms_s${S}_E1" "split compete ms_s${S}_E3" "nosplit share ms_s${S}_E1p"; do
    set -- $spec
    echo "########## seed=$S  mode=$1 join=$2 tag=$3 ##########"
    python3 train_split.py $BASE --seed $S --mode "$1" --join-mode "$2" --tag "$3" \
      > "$3.log" 2>&1
    echo "  exit=$?  val=$(grep -oP 'val = \K[0-9.]+' "$3.log" | tail -1)"
  done
done
echo "MULTISEED_DONE"
