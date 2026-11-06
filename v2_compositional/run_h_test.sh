#!/bin/bash
DATA=${1:-data_25mb.txt}
STEPS=${2:-3000}
SEED=${3:-0}
for H in 32 64 128 256; do
  echo "=========== h = $H ==========="
  python3 train_v3.py --data "$DATA" --L 4 --n-cells 4 --d 128 \
      --h $H --max-steps $STEPS --seed $SEED --tag "h${H}_s${SEED}" 2>&1 | tail -12
done
echo "ALL DONE"
