#!/bin/bash
# step29b / V1.004 实验：酶 = per-cell 参数更新速率常数
# 修正：SMOKE / SHORT 关的 --zero-check 去掉（warmup 随 max_steps 变，20/300 步
#       与 3000 步归档不可逐位比）；改由「同配置未补丁基线」做 U3 实证。
#       FULL 关保留 --zero-check（301 点、warmup 一致，才是真硬前置）。
set -uo pipefail

cd "$(dirname "$0")" || exit 1

D=data_25mb.txt
if [ ! -f "$D" ]; then echo "缺 $D → 停"; exit 1; fi

FULL=${FULL:-0}
SHORT=${SHORT:-0}
SEED=${SEED:-0}
SKIP_BASELINE=${STEP29B_SKIP_BASELINE:-0}

if   [ "$FULL"  = "1" ]; then STEPS=3000; SUFFIX="";        SPLIT=1500
elif [ "$SHORT" = "1" ]; then STEPS=300;  SUFFIX="_s300";    SPLIT=150
else                          STEPS=20;   SUFFIX="_s20";     SPLIT=1500
fi
LR=3e-4
BASE="--data $D --d 256 --h 128 --L 4 --n-cells 4 --variant shared --rank 4 \
--temp 5.0 --bs 64 --seq-len 128 --lr $LR --max-steps $STEPS --seed $SEED \
--split-k 8 --split-at $SPLIT --mode split"
RES="--cell-type residual --compose-k 16 --h-cell 3 --compose-chunk 1 --clip-mode global"

mkdir -p output/v4
TAGS=""
BAD=0

# ---------------------------------------------------------------- nopatch 基线
# 用安装前的 train_bio.py.bak_step29 跑同 max_steps 的 B 档；
# 它 import 的 compose.py 是补丁版（多两行 detach 统计）⇒ 同时实证那两行不改数值。
REF_TAG="st29_ref${SUFFIX}_nopatch"
if [ "$FULL" != "1" ]; then
  if [ "$SKIP_BASELINE" = "1" ]; then
    echo "########## nopatch 基线：SKIP（沙盒校验用）##########"
  elif [ -f "train_bio.py.bak_step29" ]; then
    echo "########## nopatch 基线  →  $REF_TAG ##########"
    cp -f train_bio.py.bak_step29 train_bio_nopatch.py
    python3 train_bio_nopatch.py $BASE --cell-type B --tag "$REF_TAG" \
      > "${REF_TAG}.log" 2>&1
    echo "  exit=$?  log=${REF_TAG}.log"
    [ -f "output/v4/${REF_TAG}.json" ] && echo "  json=OK" \
      || { echo "  json=MISSING"; BAD=$((BAD+1)); }
  else
    echo "  ⚠ 缺 train_bio.py.bak_step29 → 无同配置基线，U3 会走归档 SKIP"
  fi
fi

run_one () {
  local NAME="$1"; shift
  local TAG="st29_${NAME}${SUFFIX}"
  TAGS="$TAGS $TAG"
  echo "########## $NAME  →  $TAG  (steps=$STEPS seed=$SEED) ##########"
  python3 train_bio.py $BASE "$@" --tag "$TAG" > "${TAG}.log" 2>&1
  local RC=$?
  echo "  exit=$RC   log=${TAG}.log"
  [ "$RC" -ne 0 ] && BAD=$((BAD+1))
  if [ -f "output/v4/${TAG}.json" ]; then
    echo "  json=OK"
  else
    echo "  json=MISSING"
    BAD=$((BAD+1))
  fi
}

if [ "$FULL" != "1" ] && [ "$SHORT" != "1" ]; then
  # ---- 第一关：4 档（A0_B 不带 --zero-check）----
  run_one A0_B  --cell-type B --enz-rate-mode off
  run_one cat   $RES --enz-rate-mode cat
  run_one const $RES --enz-rate-mode const
  run_one lrx   $RES --enz-rate-mode lrx

elif [ "$SHORT" = "1" ]; then
  # ---- 第二关：7 档 ----
  run_one A0_B      --cell-type B --enz-rate-mode off
  run_one A1_res    $RES --enz-rate-mode off
  run_one amp_old   $RES --bio-on --enz-rate-mode amp
  run_one cat       $RES --enz-rate-mode cat
  run_one cat_het   $RES --enz-rate-mode cat_het
  run_one const     $RES --enz-rate-mode const
  run_one lrx       $RES --enz-rate-mode lrx

else
  # ---- 第三关：8 档 3000 步（--zero-check 只在这一关有意义）----
  run_one A0_B      --cell-type B --enz-rate-mode off --zero-check
  run_one A1_res    $RES --enz-rate-mode off
  run_one amp_old   $RES --bio-on --enz-rate-mode amp
  run_one cat       $RES --enz-rate-mode cat
  run_one cat_het   $RES --enz-rate-mode cat_het
  run_one const     $RES --enz-rate-mode const
  run_one lrx       $RES --enz-rate-mode lrx
  run_one cat_freeze $RES --enz-rate-mode cat --enz-freeze-cell 0
fi

echo "$TAGS" | tr ' ' '\n' | sed '/^$/d' > step29_tags.txt
echo "  tags → step29_tags.txt : $TAGS"

echo "########## 第一关判据（只验静态/公式，不验收益）##########"
SMOKE_RC=0
python3 step29_smoke.py $TAGS 2>&1 | tee step29_smoke${SUFFIX}.log
SMOKE_RC=${PIPESTATUS[0]}
echo "  SMOKE_RC=$SMOKE_RC"

echo "########## 完成标记 ##########"
if [ "$BAD" -eq 0 ] && [ "$SMOKE_RC" -eq 0 ]; then
  echo "STEP29_DONE_OK   bad=$BAD  smoke=$SMOKE_RC  tags=$TAGS"
else
  echo "STEP29_DONE_FAIL bad=$BAD  smoke=$SMOKE_RC  tags=$TAGS"
fi
exit 0
