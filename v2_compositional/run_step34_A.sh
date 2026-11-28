#!/usr/bin/env bash
# V1.010 / A 路线：守恒更新作用面、split 重同步和多 seed 试验。
set -euo pipefail
cd "$(dirname "$0")"

BASE_COMMIT="${BIO_BASE_COMMIT:-d843b76}"
REFERENCE=".train_bio_reference.tmp"
trap 'rm -f "$REFERENCE"' EXIT
git show "$BASE_COMMIT:v2_compositional/train_bio.py" > "$REFERENCE"

BASE="--L 4 --bio-gain 0.0 --bs 64 --cell-type residual --clip-mode global --clone-temp 2.0 --compose-chunk 1 --compose-k 16 --compose-mode fixed --compose-renorm 1 --d 256 --data data_25mb.txt --enz-beta-const 0.0 --enz-beta-ind 3.0 --enz-cost-alpha 0.1 --enz-freeze-cell -1 --enz-gamma-km 0.03 --enz-k-max 6.0 --enz-scope backbone --enz-tau-e 60.0 --enz-warmup 1 --groups 8 --h 128 --h-cell 3 --join-mode share --mem-budget 256.0 --mode split --n 384 --n-cells 4 --n-head 4 --rank 4 --scaffold-top-k 32 --select-mode soft --seq-len 128 --shared-mult 4 --split-at 150 --split-k 8 --temp 5.0 --temp-module-every 500 --temp-module-ttl 0 --variant shared --wiring-mod none"
COMMON="--max-steps 300 --lr 0.000300 --enz-rate-mode off"
META="step34_trial_meta.jsonl"
: > "$META"

run_one() {
  local script="$1" arm="$2" tag="$3" seed="$4"
  shift 4
  local t0 t1 rc wall
  t0=$(date +%s%N)
  set +e
  python3 "$script" $BASE $COMMON --seed "$seed" --tag "$tag" "$@"
  rc=$?
  set -e
  t1=$(date +%s%N)
  wall=$(( (t1 - t0) / 1000000 ))
  printf '%s\n' "{\"tag\":\"$tag\",\"seed\":$seed,\"rc\":$rc,\"wall_ms\":$wall,\"arm\":\"$arm\"}" >> "$META"
  if [ "$rc" -ne 0 ]; then
    echo "STOP arm=$arm tag=$tag rc=$rc"
    exit "$rc"
  fi
}

echo "===== PHASE 0: off-path zero-pollution ====="
run_one "$REFERENCE" P0_original st34_p0_original 0
run_one train_bio.py A_off st34_a_off 0 --bio-update off
python3 - <<'PY'
import hashlib, json, os
def gold(tag):
    rows=json.load(open(os.path.join('output/v4', tag+'.json')))['history']
    keys=('step','loss','val_loss','lr','k_mean','gate_comp')
    blob=json.dumps([[row.get(k) for k in keys] for row in rows], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
a,b=gold('st34_p0_original'),gold('st34_a_off')
print('P0=',a,'A_off=',b)
if a != b:
    raise SystemExit('HARD_GATE_FAIL: off path changed')
print('HARD_GATE_PASS: off path is bitwise identical')
PY

echo "===== PHASE 1: three arms x five seeds ====="
for seed in 0 1 2 3 4; do
  run_one train_bio.py B_budget "st34_b_budget_s$seed" "$seed" \
    --bio-update on --bio-c-weight 0
  python3 verify_bio_trace.py "output/v4/st34_b_budget_s${seed}_bio.jsonl" --require-resync

  run_one train_bio.py C_pressure "st34_c_pressure_s$seed" "$seed" \
    --bio-update on --bio-c-weight 1 --bio-require-active
  python3 verify_bio_trace.py "output/v4/st34_c_pressure_s${seed}_bio.jsonl" \
    --require-active --require-resync

  run_one train_bio.py D_frozen "st34_d_frozen_s$seed" "$seed" \
    --bio-update on --bio-c-weight 1 --bio-trust 1
  python3 verify_bio_trace.py "output/v4/st34_d_frozen_s${seed}_bio.jsonl" --require-resync
done

echo "===== PHASE 2: paired statistics and experiment-level hard gates ====="
python3 analyze_step34_A.py \
  --outdir output/v4 \
  --prefix st34 \
  --write output/v4/st34_A_summary.json

echo "STEP34_A_COMPLETE: traces and paired experiment analysis passed."
