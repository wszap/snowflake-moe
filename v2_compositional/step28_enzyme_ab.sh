#!/bin/bash
set -uo pipefail
D=data_25mb.txt
BASE="--data $D --d 256 --h 128 --L 4 --n-cells 4 --variant shared --rank 4 --temp 5.0 --bs 64 --seq-len 128 --lr 3e-4 --max-steps 3000 --seed 0 --split-k 8 --split-at 1500 --mode split"
echo "########## A  B 对照 + 零污染硬前置 ##########"
python3 train_bio.py $BASE --cell-type B --tag bio_s0_B --zero-check > bio_s0_B.log 2>&1
echo "  exit=$?"
echo "########## B  residual 纯残差（酶 OFF）##########"
python3 train_bio.py $BASE --cell-type residual --compose-k 16 --h-cell 3 --compose-chunk 1 --clip-mode global --tag bio_s0_residual > bio_s0_residual.log 2>&1
echo "  exit=$?"
echo "########## C  residual + 酶 ON（单变量）##########"
python3 train_bio.py $BASE --cell-type residual --compose-k 16 --h-cell 3 --compose-chunk 1 --clip-mode global --bio-on --tag bio_v1003_enz > bio_v1003_enz.log 2>&1
echo "  exit=$?"
echo "ENZYME_AB_DONE"
