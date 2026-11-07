#!/bin/bash
D=data_25mb.txt
BASE="--data $D --d 256 --h 128 --L 4 --n-cells 4 --variant shared --rank 4 --temp 5.0 --bs 64 --seq-len 128 --lr 3e-4 --max-steps 3000 --seed 0 --split-k 8 --split-at 1500"
echo "########## E3 384→392 但新细胞进正常竞争（E1 的干净对照）##########"
python3 train_split.py $BASE --mode split --join-mode compete --tag b5_E3_compete > b5_E3_compete.log 2>&1
echo "  exit=$?"; tail -14 b5_E3_compete.log
echo "E3_DONE"
