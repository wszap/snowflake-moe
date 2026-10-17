#!/bin/bash
# 翻转式下水道塞 (Flip-Trap) A/B 测试
#   一维 支点→轴 u   二维 密封圈→软门槛   三维 厚度→兜住   四维 翻转角→跨位置传递
# 用法: bash run_flipper_test.sh [步数]   默认 3000
set -u
cd /mnt/workspace/snowflake-moe/v2_compositional
DATA=data_25mb.txt
STEPS=${1:-3000}
mkdir -p output/v5

echo "════════════════════════════════════════════"
echo "  A: baseline（无翻转塞）"
echo "════════════════════════════════════════════"
timeout 2400 python train_v4.py --data $DATA --variant base \
    --L 4 --n-cells 4 --max-steps $STEPS --seed 0 \
    --tag flip_base 2>&1 | tee -a flipper_full.log | tail -8

echo ""
echo "════════════════════════════════════════════"
echo "  B: flipper（翻转塞开启）"
echo "════════════════════════════════════════════"
timeout 2400 python train_v4.py --data $DATA --variant flipper \
    --L 4 --n-cells 4 --max-steps $STEPS --seed 0 \
    --tag flip_on 2>&1 | tee -a flipper_full.log | tail -8

echo ""
echo "════════════════════════════════════════════"
echo "  C: flipper + clone（翻转塞 + 复制组队）"
echo "════════════════════════════════════════════"
timeout 2400 python train_v4.py --data $DATA --variant flipper,clone \
    --L 4 --n-cells 4 --max-steps $STEPS --seed 0 \
    --tag flip_clone 2>&1 | tee -a flipper_full.log | tail -8

echo ""
echo "════════════════════════════════════════════"
echo "  汇总"
echo "════════════════════════════════════════════"
python3 -c "
import json, glob, os
print(f\"{'tag':14} {'loss':>8} {'val':>8} {'ppl':>7} {'eff':>6} {'var_norm':>9}\")
print('-'*58)
for p in sorted(glob.glob('output/v5/*.json')) + sorted(glob.glob('output/v3/*.json')):
    try:
        r = json.load(open(p))
    except Exception:
        continue
    if not any(k in os.path.basename(p) for k in ('flip_base','flip_on','flip_clone')):
        continue
    print(f\"{r.get('tag',''):14} {r.get('final_loss',0):>8.4f} {r.get('val_loss',0):>8.4f} {r.get('ppl',0):>7.2f} {r.get('eff_organs_last',0):>6.2f} {r.get('var_norm_last',0):>9.4f}\")
"

echo ""
echo "══════ 完成 ══════"
echo "提示：看 flipper 是否真在工作 → grep flipper flipper_full.log"
