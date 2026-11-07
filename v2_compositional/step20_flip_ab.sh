#!/bin/bash
D=data_25mb.txt
BASE="--data $D --d 256 --h 128 --L 4 --n-cells 4 --variant shared --rank 4 --temp 5.0 --bs 64 --seq-len 128 --lr 3e-4 --max-steps 3000 --seed 0 --split-k 8 --split-at 1500 --mode split"
echo "########## F0 翻盖 OFF ##########"
FLIP_MODE=off       python3 step20_flip_ab.py $BASE --tag flip_F0_off > flip_F0_off.log 2>&1
echo "  exit=$?"; tail -8 flip_F0_off.log
echo "########## F1 翻盖 DECOUPLED ##########"
FLIP_MODE=decoupled python3 step20_flip_ab.py $BASE --tag flip_F1_decoupled > flip_F1_decoupled.log 2>&1
echo "  exit=$?"; tail -8 flip_F1_decoupled.log
echo "FLIP_AB_DONE"
