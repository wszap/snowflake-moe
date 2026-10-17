#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  snowflake-moe 仓库整改脚本
#  用法:
#     bash restructure_repo.sh --check     # 只做环境检查，不改动
#     bash restructure_repo.sh --phase 1   # 阶段1：文档硬伤修正
#     bash restructure_repo.sh --phase 2   # 阶段2：git mv 归档
#     bash restructure_repo.sh --phase 3   # 阶段3：写入新文档
#     bash restructure_repo.sh --all       # 全部执行
#
#  安全设计:
#    - 执行前自动 git tag 打快照
#    - 每阶段可单独执行、可重入
#    - 阶段2 用 git mv（保留历史）
# ══════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"

C_GREEN='\033[0;32m'; C_YELLOW='\033[1;33m'; C_RED='\033[0;31m'; C_NC='\033[0m'
log()  { echo -e "${C_GREEN}[OK]${C_NC} $*"; }
warn() { echo -e "${C_YELLOW}[WARN]${C_NC} $*"; }
err()  { echo -e "${C_RED}[ERR]${C_NC} $*"; exit 1; }

# ────────────────────────────────────────────────────────────────
snapshot() {
    local tag="pre-restructure-$(date +%Y%m%dT%H%M%S)"
    git tag "$tag" 2>/dev/null && log "快照 tag: $tag（回滚用 git reset --hard $tag）"
}

# ────────────────────────────────────────────────────────────────
check() {
    echo "═══ 环境检查 ═══"
    git rev-parse --git-dir >/dev/null 2>&1 || err "不是 git 仓库"
    log "git 仓库 OK"
    [[ -f README.md ]] && log "README.md 存在" || warn "README.md 缺失"
    [[ -f .gitignore ]] && log ".gitignore 存在" || warn ".gitignore 缺失"

    echo
    echo "═══ 待修正项检查 ═══"
    grep -q "python run_lm.py" README.md 2>/dev/null \
        && warn "① find: 'python run_lm.py' 是坏命令（文件不存在）" \
        || log "① 坏命令已修或不存在"

    grep -q "OOD Generalization" README.md 2>/dev/null \
        && warn "② OOD 表标题需改（ratio<1，OOD 集更简单）" \
        || log "② OOD 标题已改"

    grep -q "Parameter Efficiency" README.md 2>/dev/null \
        && warn "③ Parameter Efficiency 标题需改（是 active/fixed，非 vs dense）" \
        || log "③ 标题已改"

    grep -q "fraction of the parameters of a dense baseline" README.md 2>/dev/null \
        && warn "④ 'a fraction of dense baseline' 无 dense 数据支撑" \
        || log "④ 已处理"

    grep -q "memory-value slot" README.md 2>/dev/null \
        && warn "⑤ 忆点定义需加版本标注（v1.0 值槽 vs v2.0 连接本体）" \
        || log "⑤ 已标注"

    echo
    echo "═══ 建议的目录结构（阶段2 会创建）═══"
    echo "  v1_memory_points/   ← v1.0 归档"
    echo "  v2_compositional/   ← v2.0 新成果"
    echo "  docs/               ← 博客/负结果/实验日志"
}

# ────────────────────────────────────────────────────────────────
phase1() {
    echo "═══ 阶段1：修 v1.0 文档硬伤（纯文本，零风险）═══"
    cp README.md README.md.bak.$(date +%s)
    log "已备份 README.md"

    python3 - <<'PYEOF'
import re, io
s = io.open("README.md", encoding="utf-8").read()
orig = s
changes = []

# ① 坏复现命令 run_lm.py → run_experiments.py
if "python run_lm.py" in s:
    s = s.replace("python run_lm.py # via run_experiments.py",
                  "python run_experiments.py   # LM scaling & ablation sweep\npython train_lm.py           # 基础 LM 训练")
    s = s.replace("python run_lm.py", "python run_experiments.py")
    changes.append("① 坏命令 run_lm.py → run_experiments.py / train_lm.py")

# ② OOD 表标题 + 说明
if "OOD Generalization" in s:
    s = s.replace("### OOD Generalization (Shakespeare → OOD)",
        "### OOD Robustness — Relative Degradation (Shakespeare → OOD)\n\n"
        "> **注意**：两个模型的 ood/id ratio 均 < 1，说明 OOD 评测集本身比 ID 集更简单。\n"
        "> 因此这里衡量的不是「OOD 泛化有多好」，而是**同等分布偏移下的相对退化程度**。\n")
    changes.append("② OOD 标题 + ratio<1 说明")

# ③ Parameter Efficiency → Sparse Activation Ratio
if "Parameter Efficiency" in s:
    s = s.replace("### Parameter Efficiency (Snowflake v3, Shakespeare)",
        "### Sparse Activation Ratio (Snowflake v3, Shakespeare)\n\n"
        "> **注意**：Param Ratio = Active / Fixed（稀疏激活比），**不是** 与 dense baseline 的对比。\n"
        "> 本仓库目前**尚未包含 dense baseline 对照**（v2.0 正在补充，见 `v2_compositional/`）。\n")
    changes.append("③ Parameter Efficiency → Sparse Activation Ratio")

# ④ dense baseline claim 标注
if "a fraction of the parameters of a dense baseline" in s:
    s = s.replace(
        "achieving strong generalization under a fraction of the parameters of a dense baseline.",
        "achieving strong generalization under a fraction of the parameters of a dense baseline.\n"
        "  <sub>⚠️ 该 claim 目前**尚无 dense baseline 对照数据**。v2.0 已实现 dense 对照脚本\n"
        "  （`v2_compositional/dense_baseline.py` + `compare_dense.py`），结果待补。</sub>")
    changes.append("④ dense claim 加待验证标注")

# ⑤ 忆点定义版本标注
if "memory-value slot" in s:
    s = s.replace(
        "**Memory Points (忆点)**: A dedicated memory-value slot mechanism that stores decoupled factual knowledge, separate from routing weights.",
        "**Memory Points (忆点)**: A dedicated memory-value slot mechanism that stores decoupled factual knowledge, separate from routing weights.\n"
        "  <sub>**v1.0 定义**：忆点 = 独立值槽，与路由分离，可单独擦除。<br>\n"
        "  **v2.0 定义**（见 `v2_compositional/`）：忆点 = 连接本体，与路由耦合。<br>\n"
        "  两者是**不同架构**下的不同定义，请勿混用。</sub>")
    changes.append("⑤ 忆点定义加版本标注")

if s != orig:
    io.open("README.md", "w", encoding="utf-8").write(s)
    for c in changes:
        print("  [OK] " + c)
else:
    print("  无改动（可能已处理）")
PYEOF
    log "阶段1 完成"
}

# ────────────────────────────────────────────────────────────────
phase2() {
    echo "═══ 阶段2：归档 v1.0 + 新建 v2.0 目录 ═══"
    mkdir -p v1_memory_points v2_compositional docs

    # v1.0 代码归档（用 git mv 保留历史）
    V1_FILES="snowflake_moe.py snowflake_moe_improved.py \
run_cellmoe_ckpt.py run_cellmoe_mr2.py run_cellmoe_v2.py \
run_erase.py run_erase_mr2.py run_erase_tinystories.py \
run_lifelong.py run_lifelong_tinystories.py \
run_ood.py run_ood_v2.py \
run_stage1.py run_stage2.py run_stage3.py run_stage4.py run_stage5.py \
run_stage5_fast.py run_stage5_fixed.py run_experiments.py \
train_lm.py train_snowflake.py train_hierarchical.py train_mnist.py \
test_learned_council.py test_lm_learned.py \
analyze_full.py analyze_results.py engineering.py \
make_council_plot.py prepare_checkpoint.py"

    moved=0
    for f in $V1_FILES; do
        if [[ -f "$f" ]]; then
            git mv "$f" "v1_memory_points/$f" 2>/dev/null || mv "$f" "v1_memory_points/$f"
            ((moved++))
        fi
    done
    log "v1.0 归档 $moved 个脚本 → v1_memory_points/"

    # report.md / CONFIG_NOTES 归 v1.0
    for f in report.md CONFIG_NOTES; do
        [[ -f "$f" ]] && { git mv "$f" "v1_memory_points/$f" 2>/dev/null || mv "$f" "v1_memory_points/$f"; }
    done

    # early_experiments 加说明
    if [[ -d early_experiments ]]; then
        cat > early_experiments/README.md <<'EOF'
# Early Experiments (历史存档)

> ⚠️ 本目录为**早期探索代码存档**，不保证可运行、不保证与当前版本一致。
> 保留目的：记录研究过程的演进轨迹。
> 如需复现结果，请使用 `v1_memory_points/` 或 `v2_compositional/` 下的脚本。
EOF
        log "early_experiments/README.md 已加说明"
    fi

    log "阶段2 完成"
}

# ────────────────────────────────────────────────────────────────
phase3() {
    echo "═══ 阶段3：写入新文档 ═══"
    # 由 MARVIS 从交付包拷入：
    #   README_main.md      → README.md
    #   README_v1.md        → v1_memory_points/README.md
    #   README_v2.md        → v2_compositional/README.md
    #   RESULTS.md          → v2_compositional/RESULTS.md
    #   博客_MoE的六个陷阱.md → docs/
    #   NEGATIVE_RESULTS.md  → docs/
    #   EXPERIMENT_LOG.md    → docs/
    #   *.py (v2.0 全部)     → v2_compositional/
    echo "  见 MARVIS_TASK.md 的「文件投放清单」"
    log "阶段3 说明已打印"
}

# ────────────────────────────────────────────────────────────────
case "${1:-}" in
    --check)   check ;;
    --phase)   [[ $# -ge 2 ]] || err "缺参数"; snapshot; "phase$2" ;;
    --all)     snapshot; phase1; phase2; phase3 ;;
    *) echo "用法: $0 --check | --phase {1,2,3} | --all"; exit 1 ;;
esac
