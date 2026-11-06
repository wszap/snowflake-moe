#!/bin/bash
D=data_25mb.txt
S=3000
echo "########## 1. v3 d=256 h=64 ##########"
python3 train_v3.py --data $D --L 4 --n-cells 4 --d 256 --h 64 \
  --max-steps $S --seed 0 --tag d256h64_s0 2>&1 | tail -12

echo "########## 2. v3 d=256 h=128 ##########"
python3 train_v3.py --data $D --L 4 --n-cells 4 --d 256 --h 128 \
  --max-steps $S --seed 0 --tag d256h128_s0 2>&1 | tail -12

echo "########## 3. v4 d=256 h=128 + shared ##########"
python3 train_v4.py --data $D --L 4 --n-cells 4 --d 256 --h 128 \
  --variant shared --max-steps $S --seed 0 --tag d256h128_sh_s0 2>&1 | tail -12

echo "########## 4. dense 参数对齐（目标 ~12M）##########"
python3 dense_baseline.py --data $D --d 496 --L 4 --max-steps $S --seed 0 \
  --outdir output/dense --fix-mojibake 2>&1 | tail -6

echo "ALL DONE"
