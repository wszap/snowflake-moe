#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  补丁后重跑 + 对比实验 完整流程
#
#  ★ 前提：10 seeds 已跑完（seeds.log 出现 ALL_DONE）
#    训练途中改代码会让前后 seed 口径不一致 → 统计检验作废
#
#  用法:
#     bash run_after_patch.sh check     # 确认 10 seeds 已完成
#     bash run_after_patch.sh patch     # 应用 6+2 个 bug 补丁
#     bash run_after_patch.sh verify    # 验证补丁
#     bash run_after_patch.sh rerun     # 补丁后重跑 10 seeds（取全量指标）
#     bash run_after_patch.sh dense     # dense 对比实验
#     bash run_after_patch.sh all       # 全部
# ══════════════════════════════════════════════════════════════════
set -uo pipefail

DIR="${DIR:-/mnt/workspace/snowflake-moe/v2_compositional}"
DATA="${DATA:-data_25mb.txt}"
SNOW_LOSS="${SNOW_LOSS:-1.916}"      # A10 实测（bf16, 25MB）
SNOW_PARAMS="${SNOW_PARAMS:-5164132}"

C_GREEN='\033[0;32m'; C_RED='\033[0;31m'; C_YELLOW='\033[1;33m'; C_NC='\033[0m'
log()  { echo -e "${C_GREEN}[OK]${C_NC} $*"; }
warn() { echo -e "${C_YELLOW}[WARN]${C_NC} $*"; }
err()  { echo -e "${C_RED}[ERR]${C_NC} $*"; }

check() {
    echo "═══ 检查 10 seeds 是否完成 ═══"
    cd "$DIR" || exit 1
    if grep -q "ALL_DONE" seeds.log 2>/dev/null; then
        log "10 seeds 已完成"
        grep -c "═══ SEED" seeds.log | xargs -I{} echo "  完成 seed 数: {}"
    else
        err "10 seeds 未完成"
        echo "  当前进度: $(grep -c '═══ SEED' seeds.log 2>/dev/null || echo 0) / 10"
        echo
        echo "  ★ 未完成前【不要】跑 patch —— 会破坏口径一致性"
        exit 1
    fi

    # 先把旧结果存档（补丁后会被覆盖）
    if [[ -d output/runs ]]; then
        n=$(ls output/runs | wc -l)
        tar -czf "runs_before_patch_$(date +%m%d_%H%M).tar.gz" output/runs/ 2>/dev/null \
            && log "旧证据已存档（$n 个 run）"
    fi
}

patch() {
    cd "$DIR" || exit 1
    echo "═══ 应用补丁 ═══"
    python3 patch_all.py --check
    echo
    python3 patch_all.py --apply
    echo
    python3 patch_all.py --verify
}

rerun() {
    cd "$DIR" || exit 1
    echo "═══ 补丁后重跑 10 seeds（全量指标）═══"
    echo "  新增采集: eff_organs / wiring_var_norm / 逐步 sharp 计数 / 早期加密采样"
    echo
    nohup bash -c '
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "═══ SEED $s (patched) ═══"
  python train_lock60.py --scale tiny \
      --data '"$DATA"' --delta-scale 0.5 \
      --max-steps 3000 --seed $s --save-ckpt ckpt_p$s.pt
done
echo PATCHED_ALL_DONE
' > seeds_patched.log 2>&1 &
    echo $! > seeds_patched.pid
    log "已启动 PID=$(cat seeds_patched.pid)，约 75 分钟"
    echo "  跟踪: tail -f seeds_patched.log"
}

dense() {
    cd "$DIR" || exit 1
    echo "═══ dense 对比实验 ═══"

    # ① 泄漏自检（必做）
    echo
    echo "--- ① 因果掩码泄漏自检 ---"
    python3 dense_baseline.py --leak-check-only || {
        err "自检失败"; return 1; }

    # ② 规模扫描
    echo
    echo "--- ② 规模扫描（找临界参数量）---"
    echo "  Snowflake 基准: ${SNOW_PARAMS} params, loss=${SNOW_LOSS}"
    python3 dense_baseline.py --data "$DATA" \
        --scale-sweep --max-steps 3000 \
        --snow-params "$SNOW_PARAMS" --snow-loss "$SNOW_LOSS" 2>&1 \
        | tee dense_sweep.log

    # ③ 对齐配置 10 seeds
    echo
    echo "--- ③ 对齐配置 10 seeds（L=5,d=288 ≈ 0.973x）---"
    python3 dense_baseline.py --data "$DATA" \
        --L 5 --d 288 --sweep --max-steps 3000 2>&1 | tee dense_seeds.log

    # ④ 配对检验 + claim 判定
    echo
    echo "--- ④ 配对检验 ---"
    python3 compare_dense.py --dense output/dense 2>&1 | tee dense_compare.log

    echo
    echo "  ★ compare_dense.py 会判定 README 那句"
    echo "    'a fraction of the parameters of a dense baseline'"
    echo "    是否成立，并给出替代表述"
}

summary() {
    cd "$DIR" || exit 1
    echo "═══ 结果汇总 ═══"
    python3 - <<'PYEOF'
import json, glob, os, math, statistics as st

def load(pat, n=10):
    ds = sorted(glob.glob(pat), key=os.path.getmtime)[-n:]
    out = []
    for d in ds:
        try:
            out.append(json.load(open(os.path.join(d, "record.json"))))
        except Exception:
            pass
    return out

print("─" * 72)
print("  Snowflake（补丁后）10 seeds")
print("─" * 72)
rs = load("output/runs/*/")
if rs:
    cs = [r["final"].get("core_stat") for r in rs if r["final"].get("core_stat") is not None]
    ef = [r["final"].get("eff_organs_final") for r in rs]
    ef = [e for e in ef if e is not None]
    wn = [r["final"].get("wiring_var_norm") for r in rs]
    wn = [w for w in wn if w is not None]
    sw = [r["final"].get("sharp_warn_triggered", 0) for r in rs]

    if cs:
        print(f"  core_stat   = {st.mean(cs):.4f} ± {st.stdev(cs):.4f}"
              f"   (n={len(cs)})")
        if len(cs) > 1:
            t = st.mean(cs) / (st.stdev(cs) / len(cs) ** 0.5)
            print(f"  t({len(cs)-1})     = {t:.1f}")
    if ef:
        print(f"  eff_organs  = {st.mean(ef):.2f} ± {st.stdev(ef):.2f}")
    if wn:
        print(f"  var_norm    = {st.mean(wn):.4f} ± {st.stdev(wn):.4f}"
              f"   (H2 阈值 0.05)")
    print(f"  sharp 告警  = {sum(sw)} 次（10 seeds 累计）")
else:
    print("  无数据")
PYEOF
}

# ── Bug9 统计修正审计：用【已跑完的】10 seeds 重算 CI，验证结论是否改变 ──
audit() {
    cd "$DIR" || exit 1
    echo "═══ core_stat 统计修正审计（Bug9）═══"
    echo
    if [[ ! -f audit_ci.py ]]; then
        warn "audit_ci.py 不存在，跳过（从归档包 scripts/ 复制）"
        return 1
    fi
    # 优先审计【打补丁前】的存档，若不存在则审计当前 runs
    if ls runs_before_patch_*.tar.gz >/dev/null 2>&1; then
        TMP=$(mktemp -d)
        tar -xzf "$(ls -t runs_before_patch_*.tar.gz | head -1)" -C "$TMP"
        RD=$(ls -td "$TMP"/output/runs/*/ 2>/dev/null | head -10 | tr '\n' ' ')
        [[ -n "$RD" ]] && python3 audit_ci.py --runs $RD
        rm -rf "$TMP"
    else
        python3 audit_ci.py
    fi
}

case "${1:-check}" in
    check)   check ;;
    patch)   patch ;;
    verify)  cd "$DIR" && python3 patch_all.py --verify ;;
    audit)   audit ;;
    rerun)   rerun ;;
    dense)   dense ;;
    summary) summary ;;
    all)     check && patch && audit && rerun ;;
    *) echo "用法: $0 check|patch|verify|audit|rerun|dense|summary|all" ;;
esac
