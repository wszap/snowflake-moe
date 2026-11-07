#!/bin/bash
D=data_25mb.txt
BASE="--data $D --d 256 --h 128 --L 4 --n-cells 4 --variant shared --rank 4 --temp 5.0 --bs 64 --seq-len 128 --lr 3e-4 --max-steps 3000 --seed 0 --split-k 8 --split-at 1500"
echo "########## E1' 补跑（392 不分裂，冻结 c 与 E1 对称）##########"
python3 train_split.py $BASE --mode nosplit --tag b5_E1p_nosplit > b5_E1p_nosplit.log 2>&1
echo "  exit=$?"
tail -14 b5_E1p_nosplit.log
echo "E1P_DONE"
