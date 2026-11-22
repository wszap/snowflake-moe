#!/bin/bash
# =============================================================================
# step30_ab.sh -- V1.005：酶的作用面从 16 个 gate 挪到 16 份主干 lr
# 三关参数化；同名 tag 带后缀互不覆盖；无 set -e ⇒ 末尾假完成检测兜住。
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1

D=data_25mb.txt
if [ ! -f "$D" ]; then echo "缺 $D → 停"; exit 1; fi

FULL=${FULL:-0}; SHORT=${SHORT:-0}; SEED=${SEED:-0}
if   [ "$FULL"  = "1" ]; then STEPS=3000; SUFFIX="";      SPLIT=1500
elif [ "$SHORT" = "1" ]; then STEPS=300;  SUFFIX="_s300"; SPLIT=150
else                          STEPS=20;   SUFFIX="_s20";  SPLIT=1500
fi
LR=3e-4
BASE="--data $D --d 256 --h 128 --L 4 --n-cells 4 --variant shared --rank 4 \
--temp 5.0 --bs 64 --seq-len 128 --lr $LR --max-steps $STEPS --seed $SEED \
--split-k 8 --split-at $SPLIT --mode split"
RES="--cell-type residual --compose-k 16 --h-cell 3 --compose-chunk 1 --clip-mode global"
# k 从 1 起步（β_const=0）⇒ 全自主诱导，无预置分化
ENZ="--enz-beta-const 0.0 --enz-beta-ind 3.0 --enz-tau-e 60 --enz-warmup 1"

mkdir -p output/v4
TAGS=""; BAD=0

run_one () {
  local NAME="$1"; shift
  local TAG="st30_${NAME}${SUFFIX}"
  TAGS="$TAGS $TAG"
  echo "########## $NAME  →  $TAG  (steps=$STEPS seed=$SEED) ##########"
  python3 train_bio.py $BASE "$@" --tag "$TAG" > "${TAG}.log" 2>&1
  local RC=$?
  echo "  exit=$RC   log=${TAG}.log"
  [ "$RC" -ne 0 ] && BAD=$((BAD+1))
  if [ -f "output/v4/${TAG}.json" ]; then echo "  json=OK"; else echo "  json=MISSING"; BAD=$((BAD+1)); fi
}

# ---- 同配置未补丁基线（用安装器留下的备份跑，U3 逐位比对用）----
if [ -f train_bio.py.bak_step30 ]; then
  REF="st30_ref${SUFFIX}_nopatch"
  echo "########## nopatch 基线  →  $REF ##########"
  python3 train_bio.py.bak_step30 $BASE --cell-type B --enz-rate-mode off \
      --tag "$REF" > "${REF}.log" 2>&1
  echo "  exit=$?   log=${REF}.log"
  [ -f "output/v4/${REF}.json" ] && echo "  json=OK" || echo "  json=MISSING（U3 会 SKIP，不判停线）"
  TAGS="$TAGS $REF"
else
  echo "  ⚠ 无 train_bio.py.bak_step30 → U3 将按配置不同 SKIP（不判停线）"
fi

# ---- 档位（★ 单变量：只改 --enz-scope，其余全同）----
# FULL 关才带 --zero-check（warmup 与 3000 步参考一致）
ZC=""; [ "$FULL" = "1" ] && ZC="--zero-check"
run_one A0_B      --cell-type B --enz-rate-mode off $ZC
run_one gate_cat  $RES --enz-rate-mode cat   --enz-scope gate     $ENZ
run_one bb_cat    $RES --enz-rate-mode cat   --enz-scope backbone $ENZ
run_one bb_const  $RES --enz-rate-mode const --enz-scope backbone $ENZ
run_one lrx       $RES --enz-rate-mode lrx   --enz-scope gate     $ENZ

if [ "$FULL" = "1" ]; then
  # 速率 rescue：单 cell k≡1，不删专家、不改前向
  run_one bb_freeze $RES --enz-rate-mode cat --enz-scope backbone $ENZ --enz-freeze-cell 0
fi

echo "$TAGS" | tr ' ' '\n' | sed '/^$/d' > step30_tags.txt
echo "  tags → step30_tags.txt : $TAGS"

echo "########## 第一关判据（G1 零污染 / G2 作用面 / T1-T3 公式）##########"
SMOKE_RC=0
python3 step30_smoke.py $TAGS 2>&1 | tee step30_smoke${SUFFIX}.log
SMOKE_RC=${PIPESTATUS[0]}
echo "  SMOKE_RC=$SMOKE_RC"

echo "########## 完成标记 ##########"
if [ "$BAD" -eq 0 ] && [ "$SMOKE_RC" -eq 0 ]; then
  echo "STEP30_DONE_OK   bad=$BAD  smoke=$SMOKE_RC  tags=$TAGS"
else
  echo "STEP30_DONE_FAIL bad=$BAD  smoke=$SMOKE_RC  tags=$TAGS"
fi
exit 0
