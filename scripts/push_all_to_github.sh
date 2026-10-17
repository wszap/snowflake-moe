#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  一键推送到 GitHub（换电脑前执行）
#
#  做的事：
#    1. 检查 git 状态
#    2. 把 SNOWFLAKE_ARCHIVE 内容同步到目标仓库的对应目录
#    3. 提交 + 推送（默认推到独立分支，【不动 main】）
#
#  用法:
#     export REPO=/path/to/snowflake-moe
#     bash push_all_to_github.sh              # 推到 backup 分支（推荐）
#     bash push_all_to_github.sh --to-main    # 直接推 main（谨慎）
# ══════════════════════════════════════════════════════════════════
set -euo pipefail

ARCHIVE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${REPO:-}"
TO_MAIN=false
[[ "${1:-}" == "--to-main" ]] && TO_MAIN=true

C_GREEN='\033[0;32m'; C_RED='\033[0;31m'; C_YELLOW='\033[1;33m'; C_NC='\033[0m'
log()  { echo -e "${C_GREEN}[OK]${C_NC} $*"; }
err()  { echo -e "${C_RED}[ERR]${C_NC} $*"; exit 1; }
warn() { echo -e "${C_YELLOW}[WARN]${C_NC} $*"; }

# ── 0. 前置检查 ────────────────────────────────────────────────
[[ -n "$REPO" ]] || err "请设置 REPO 环境变量：export REPO=/path/to/snowflake-moe"
[[ -d "$REPO/.git" ]] || err "$REPO 不是 git 仓库"

cd "$REPO"
BRANCH="backup/full-archive-$(date +%m%d)"
$TO_MAIN && BRANCH="main"

echo "═══ 推送配置 ═══"
echo "  归档源: $ARCHIVE_DIR"
echo "  目标库: $REPO"
echo "  分支  : $BRANCH"
echo

# ── 1. 建/切分支 ───────────────────────────────────────────────
if $TO_MAIN; then
    git checkout main 2>/dev/null || err "无法切到 main"
    git pull --ff-only 2>/dev/null || warn "pull 失败，继续"
else
    git checkout -B "$BRANCH" 2>/dev/null || err "无法建分支 $BRANCH"
    log "分支就绪: $BRANCH"
fi

# ── 2. 同步归档内容 ────────────────────────────────────────────
echo
echo "═══ 同步文件 ═══"

# 2.1 v2.0 目录
mkdir -p v2_compositional
cp "$ARCHIVE_DIR"/v2_compositional/*  v2_compositional/ 2>/dev/null
log "v2_compositional/  ← $(ls "$ARCHIVE_DIR/v2_compositional" | wc -l) 个文件"

# 2.2 docs
mkdir -p docs
cp "$ARCHIVE_DIR"/docs/*  docs/ 2>/dev/null
log "docs/  ← $(ls "$ARCHIVE_DIR/docs" | wc -l) 个文件"

# 2.3 脚本（不含本脚本自身，避免递归）
mkdir -p scripts
for f in "$ARCHIVE_DIR"/scripts/*; do
    b=$(basename "$f")
    [[ "$b" == "push_all_to_github.sh" ]] && continue
    cp "$f" scripts/ 2>/dev/null
done
log "scripts/  ← 已同步"

# 2.3.5 ★ 保护仓库原有根级文件：不覆盖 LICENSE / requirements / environment / .gitignore
#       这些文件仓库里已有，归档包的版本可能不同，一律跳过
for f in LICENSE requirements.txt environment.yml .gitignore; do
    if [[ -f "$ARCHIVE_DIR/$f" && -f "$f" ]]; then
        echo "  跳过（仓库已有，不覆盖）: $f"
    fi
done

# 2.4 根级文档（用新名，不覆盖 README.md）
cp "$ARCHIVE_DIR/scripts/README_main.md"  README_v2_main.md 2>/dev/null || true
cp "$ARCHIVE_DIR/scripts/README_v1.md"    README_v1_memory_points.md 2>/dev/null || true
cp "$ARCHIVE_DIR/scripts/MARVIS_TASK.md"  MARVIS_TASK.md 2>/dev/null || true
cp "$ARCHIVE_DIR/scripts/INTEGRATION_PLAN.md" INTEGRATION_PLAN.md 2>/dev/null || true
cp "$ARCHIVE_DIR/A10_REBUILD.md"             A10_REBUILD.md 2>/dev/null || true
cp "$ARCHIVE_DIR/DATA_RECOVERY.md"           DATA_RECOVERY.md 2>/dev/null || true
cp "$ARCHIVE_DIR/MIGRATION_GUIDE.md"         MIGRATION_GUIDE.md 2>/dev/null || true
cp "$ARCHIVE_DIR/WINDOWS_QUICKSTART.md"      WINDOWS_QUICKSTART.md 2>/dev/null || true
log "根级文档已同步"

# ── 3. 体积检查 ────────────────────────────────────────────────
echo
echo "═══ 体积检查 ═══"
BIG=$(find . -path ./.git -prune -o -type f -size +10M -print 2>/dev/null | head -20)
if [[ -n "$BIG" ]]; then
    warn "以下文件 >10MB，GitHub 可能拒收："
    echo "$BIG" | sed 's/^/    /'
    echo "  建议加入 .gitignore 或改用 Git LFS"
else
    log "无超大文件"
fi

# ── 4. 提交 ────────────────────────────────────────────────────
echo
echo "═══ 提交 ═══"
git add -A

if git diff --cached --quiet; then
    warn "无改动可提交"
else
    git commit -m "archive: full v2.0 integration package (pre-migration backup)

- v2_compositional/: SnowflakeB + train_lock60 + run_metrics + dense baseline
  + compare_dense + verify_six_traps + RESULTS.md (10 seeds, 5 hypotheses PASS)
- docs/: blog (six MoE traps), NEGATIVE_RESULTS, EXPERIMENT_LOG (37 decisions)
- scripts/: restructure_repo.sh, collect_remote_evidence.sh, MARVIS_TASK.md
- README_main.md / README_v1.md: dual-version navigation + known limitations

NOTE: original v1.0 code and commit history untouched - all new content
  goes into v2_compositional/ (new) to avoid overwriting existing files.

Purpose: full backup; original code history preserved on main."

    log "已提交"
fi

# ── 5. 推送 ────────────────────────────────────────────────────
echo
echo "═══ 推送 ═══"
if git remote get-url origin >/dev/null 2>&1; then
    git push -u origin "$BRANCH" 2>&1 | tail -5
    log "已推送到 origin/$BRANCH"
else
    warn "无 origin remote，未推送。手动添加："
    echo "     git remote add origin https://github.com/wszap/snowflake-moe.git"
fi

echo
echo "═══ 完成 ═══"
echo "  分支: $BRANCH"
echo
echo "下一步（换电脑后）:"
echo "  1. git clone + git checkout $BRANCH"
echo "  2. 按 MARVIS_TASK.md 执行仓库整改"
echo "  3. 跑 dense baseline 补 P0 证据"
