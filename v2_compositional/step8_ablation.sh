#!/bin/bash
D=data_25mb.txt
BASE="--data $D --L 4 --n-cells 4 --d 256 --h 128 --variant shared --max-steps 3000 --seed 0"
echo "########## A 动态拼（基线复现，应=1.2041）##########"
python3 train_v4.py $BASE --force-mix none    --tag abl_A_compose 2>&1 | tail -12
echo "########## B 静态全库平均拼（384个全参与，仅关动态）##########"
python3 train_v4.py $BASE --force-mix uniform --tag abl_B_static   2>&1 | tail -12
echo "########## C 单专家选（反面）##########"
python3 train_v4.py $BASE --force-mix onehot  --tag abl_C_select   2>&1 | tail -12
echo "ABL_DONE"
