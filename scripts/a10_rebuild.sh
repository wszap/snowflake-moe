#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  A10 重建脚本 —— 服务器释放后从头重建全部证据
#
#  用法:
#     bash a10_rebuild.sh --setup        # ① 下载数据 + 环境检查
#     bash a10_rebuild.sh --verify       # ② 泄漏自检（必做）
#     bash a10_rebuild.sh --smoke        # ③ 单次摸底 3000 步
#     bash a10_rebuild.sh --seeds        # ⑤ 10 seeds 重跑
#     bash a10_rebuild.sh --dense-sweep  # ⑥ dense 规模扫描
#     bash a10_rebuild.sh --dense-seeds  # ⑦ dense 10 seeds
#     bash a10_rebuild.sh --all          # 全部（约 1.5 小时）
#
#  ★ A10 与 DCU 的关键差异:
#      DCU: 需要 --no-bf16（ROCm bf16 比 fp32 慢 2 倍）
#      A10: 【不要】--no-bf16 —— Ampere 真实 Tensor Core，bf16 是真加速
# ══════════════════════════════════════════════════════════════════
set -uo pipefail

DATA="${DATA:-/root/tinystories_100mb.txt}"
STEPS="${STEPS:-3000}"
OUT="${OUT:-./a10_rebuild_out}"

C_GREEN='\033[0;32m'; C_RED='\033[0;31m'; C_YELLOW='\033[1;33m'; C_NC='\033[0m'
log()  { echo -e "${C_GREEN}[OK]${C_NC} $*"; }
warn() { echo -e "${C_YELLOW}[WARN]${C_NC} $*"; }
err()  { echo -e "${C_RED}[ERR]${C_NC} $*"; }

mkdir -p "$OUT"

# ── ① 环境 + 数据 ──────────────────────────────────────────────
setup() {
    echo "═══ ① 环境检查 ═══"
    python3 -c "
import torch
print('torch     :', torch.__version__)
print('cuda avail:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device    :', torch.cuda.get_device_name(0))
    print('bf16 支持 :', torch.cuda.is_bf16_supported())
" || err "torch 检查失败"

    echo
    echo "═══ 数据 ═══"
    if [[ -f "$DATA" ]]; then
        log "数据已存在: $DATA ($(du -h "$DATA" | cut -f1))"
    else
        warn "数据不存在，开始下载..."
        mkdir -p "$(dirname "$DATA")"
        wget -q --show-progress \
          https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz \
          -O /tmp/ts.tar.gz || err "下载失败"
        tar -xzf /tmp/ts.tar.gz -C /tmp/
        # 找一个大的 txt 拼成 100MB
        cat /tmp/*.txt > "$DATA" 2>/dev/null || \
            cp /tmp/TinyStories_all_data.txt "$DATA" 2>/dev/null || err "找不到解压后的 txt"
        head -c 104857600 "$DATA" > "$DATA.tmp" && mv "$DATA.tmp" "$DATA"
        log "已生成: $DATA ($(du -h "$DATA" | cut -f1))"
    fi
    md5sum "$DATA" | tee "$OUT/data_md5.txt"
}

# ── ② 泄漏自检 ─────────────────────────────────────────────────
verify() {
    echo "═══ ② 因果掩码泄漏自检（必做）═══"
    python3 dense_baseline.py --leak-check-only || err "自检脚本失败"
    echo
    echo "  必须看到: ✅ 无泄漏（正确 causal）"
    echo "  若看到 ❌ 泄漏，【不要】往下跑 dense"
}

# ── ③ 单次摸底 ─────────────────────────────────────────────────
smoke() {
    echo "═══ ③ 单次摸底 ${STEPS} 步 ═══"
    echo "  A10 上【不加】--no-bf16（与 DCU 不同）"
    python3 train_lock60.py --scale tiny \
        --data "$DATA" \
        --delta-scale 0.5 --fix-mojibake \
        --max-steps "$STEPS" --seed 0 \
        --save-ckpt "$OUT/ckpt_smoke.pt" 2>&1 | tee "$OUT/smoke.log"

    echo
    echo "  检查项:"
    echo "    · [data] 行 vocab 应为 98（清洗后）"
    echo "    · 起始 loss ≈ 4.585 (= ln 98)"
    echo "    · 末段 loss ≈ 1.78"
    echo "    · eff ≈ 23~29"
}

# ── ④ ablation（从 ckpt，不用重训）──────────────────────────────
ablation() {
    echo "═══ ④ ablation（从 ckpt 恢复）═══"
    [[ -f "$OUT/ckpt_smoke.pt" ]] || err "ckpt 不存在，先跑 --smoke"
    python3 train_lock60.py --scale tiny --eval-only "$OUT/ckpt_smoke.pt" 2>&1 \
        | tee "$OUT/ablation.log"
    echo
    echo "  目标: core_stat 应为显著正数（原值 +1.41）"
}

# ── ⑤ 10 seeds ─────────────────────────────────────────────────
seeds() {
    echo "═══ ⑤ 10 seeds 重跑 ═══"
    for s in 0 1 2 3 4 5 6 7 8 9; do
        echo "  --- seed $s ---"
        python3 train_lock60.py --scale tiny \
            --data "$DATA" \
            --delta-scale 0.5 --fix-mojibake \
            --max-steps "$STEPS" --seed "$s" \
            --save-ckpt "$OUT/ckpt_s$s.pt" > "$OUT/seed$s.log" 2>&1
        cs=$(grep -o "core_stat[^,]*" "$OUT/seed$s.log" | tail -1)
        log "seed $s 完成  $cs"
    done
    echo
    echo "  汇总: 应得 core_stat ≈ 1.363 ± 0.029"
}

# ── ⑥ dense 规模扫描 ───────────────────────────────────────────
dense_sweep() {
    echo "═══ ⑥ dense 规模扫描（P0 新证据）═══"
    echo "  ⚠ 这是 v1.0 README 唯一没被证据支撑的 claim"
    python3 dense_baseline.py --data "$DATA" \
        --scale-sweep --max-steps "$STEPS" 2>&1 | tee "$OUT/dense_sweep.log"
    echo
    echo "  目标: 找 dense 达到 loss≈1.788 的临界参数量"
}

# ── ⑦ dense 10 seeds ───────────────────────────────────────────
dense_seeds() {
    echo "═══ ⑦ dense 10 seeds（配对检验）═══"
    python3 dense_baseline.py --data "$DATA" \
        --L 5 --d 288 --sweep 2>&1 | tee "$OUT/dense_seeds.log"
    echo
    python3 compare_dense.py --dense output/dense 2>&1 | tee "$OUT/compare.log"
    echo
    echo "  compare_dense.py 会自动判定 README 那句 claim 保留还是改写"
}

# ── 全部 ───────────────────────────────────────────────────────
all() {
    setup
    echo; verify
    echo; smoke
    echo; ablation
    echo; seeds
    echo; dense_sweep
    echo; dense_seeds
    echo
    echo "═══ 全部完成 ═══"
    echo "  产物: $OUT/"
}

case "${1:-}" in
    --setup)       setup ;;
    --verify)      verify ;;
    --smoke)       smoke ;;
    --ablation)    ablation ;;
    --seeds)       seeds ;;
    --dense-sweep) dense_sweep ;;
    --dense-seeds) dense_seeds ;;
    --all)         all ;;
    *) echo "用法: $0 --setup|--verify|--smoke|--ablation|--seeds|--dense-sweep|--dense-seeds|--all"; exit 1 ;;
esac
