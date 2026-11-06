#!/bin/bash
D=data_25mb.txt
for NC in 64 256; do
  echo "########## iso n_cells=$NC ##########"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 train_v4.py --data $D --L 4 --n-cells $NC --d 256 --h 128 \
    --variant shared --max-steps 3000 --seed 0 \
    --tag "nc${NC}_k4_s0" 2>&1 | tail -12
done
echo "ISO_DONE"
