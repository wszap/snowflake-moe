#!/bin/bash
D=data_25mb.txt
S=3000
echo "########## A. dense 参数对齐 ~17M (d=596) ##########"
python3 dense_baseline.py --data $D --d 596 --L 4 --max-steps $S --seed 0 --outdir output/dense --fix-mojibake 2>&1 | tail -4
echo "########## B. v4 d=256 h=256 + shared ##########"
python3 train_v4.py --data $D --L 4 --n-cells 4 --d 256 --h 256 --variant shared --max-steps $S --seed 0 --tag d256h256_sh_s0 2>&1 | tail -12
echo "ALL DONE"
