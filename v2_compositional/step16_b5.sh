#!/bin/bash
D=data_25mb.txt
BASE="--data $D --d 256 --h 128 --L 4 --n-cells 4 --variant shared --rank 4 --temp 5.0 --bs 64 --seq-len 128 --lr 3e-4 --max-steps 3000 --seed 0 --split-k 8 --split-at 1500"
run() {
  echo "########## $1 ##########"
  python3 train_split.py $BASE --mode "$2" --tag "$3" > "$3.log" 2>&1
  echo "  exit=$?  ($3.log)"
  tail -14 "$3.log"
}
run "E1 主动分裂 384->392"  split       b5_E1_split
run "E1' 不分裂对照"       nosplit     b5_E1p_nosplit
run "E2 从头 392"          fromscratch b5_E2_fromscratch
echo "B5_DONE"
