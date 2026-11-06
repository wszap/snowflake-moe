#!/bin/bash
D=data_25mb.txt
echo "########## you 12000 steps ##########"
python3 train_v4.py --data $D --L 4 --n-cells 4 --d 256 --h 128 \
  --variant shared --max-steps 12000 --seed 0 --tag sh_12000 2>&1 | tail -12
echo "########## dense 12000 steps ##########"
python3 dense_baseline.py --data $D --d 600 --L 4 --max-steps 12000 \
  --seed 0 --outdir output/dense --fix-mojibake 2>&1 | tail -4
echo "ALL DONE"
