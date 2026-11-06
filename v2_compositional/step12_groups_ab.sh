#!/bin/bash
D=data_25mb.txt
BASE="--data $D --L 4 --n-cells 4 --d 256 --h 128 --variant shared --max-steps 3000 --seed 0"
echo "########## groups=0 基线复现（必须=1.2041）##########"
python3 train_v4.py $BASE --groups 0 --tag abl_G0_baseline 2>&1 | tail -8
echo "########## groups=8 软群乘性先验 ##########"
python3 train_v4.py $BASE --groups 8 --tag abl_G8_soft 2>&1 | tail -8
echo "GROUPS_DONE"
