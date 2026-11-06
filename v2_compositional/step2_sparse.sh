#!/bin/bash
D=data_25mb.txt
for NC in 4 16 64 256; do
  echo "########## n_cells=$NC  scaffold-top-k=4 ##########"
  python3 train_v4.py --data $D --L 4 --n-cells $NC --d 256 --h 128 \
    --variant shared --scaffold-top-k 4 \
    --max-steps 3000 --seed 0 --tag "nc${NC}_k4_s0" 2>&1 | tail -12
done
echo "ALL DONE"
