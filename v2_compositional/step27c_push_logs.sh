#!/bin/bash
# paperplane-Moe V1.001 · 补推日志（D 项）+ 全部新代码 + 本地完整快照
# 在 DSW 的 v2_compositional/ 下跑；push 需要你的 SSH 凭据，我（对话沙盒）无法代推。
set -uo pipefail
cd "$(dirname "$0")" || exit 1
ROOT="$(cd .. && pwd)"
echo "  仓库根 = $ROOT"

echo "===== [1] 补 *.log（V1.000 的 .gitignore 把它们全排掉了）====="
git add -f '*.log' 2>/dev/null || true
echo "  已 add -f 的 log 数 = $(git diff --cached --name-only | grep -c '\.log$' || true)"

echo "===== [2] V1.001 全部新文件 ====="
git add -A ../paperplane_moe 2>/dev/null || true
for f in train_bio.py verify_bio.py paperplane_train.py \
         step27a_biology_smoke.sh step27b_rename.sh \
         step27c_push_logs.sh step27d_bio_summary.py \
         ../06_改名执行_V1.001.md ../07_补推日志_V1.001.md \
         ../08_生物学思想到代码映射_V1.001.md \
         ../09_V1.001改动与验收.md ../README.md ../VERSION \
         ../02_实验报告_V1.000.md ../05_实验改进之处与负结果.md \
         ../MANIFEST.txt ../selfcheck_v1001.py \
         ../SELFCHECK_V1.001.log; do
  if [ -e "$f" ]; then
    case "$f" in
      *.log) git add -f "$f" 2>/dev/null || true ;;
      *)     git add "$f" 2>/dev/null || true ;;
    esac
  fi
done

echo "===== [3] staged 清单（必须非空）====="
git diff --cached --stat | tail -30
if git diff --cached --quiet; then
  echo "  ⚠ staged 为空 → 不提交，检查 .gitignore / 文件路径"
  exit 1
fi

echo "===== [4] 提交 + 推送（不要 --force）====="
git commit -m "paperplane-Moe V1.001: biology primitives as executable code, telemetry events (spec 11/13), residual compose cell, unified naming + logs"
git push origin HEAD:main && git push origin HEAD:backup/full-archive-1017
echo "  新 origin/main = $(git rev-parse --short HEAD)"

echo "===== [5] 本地完整快照（含 181M 级数据/日志，JupyterLab 可直接下载）====="
TS=$(date +%Y%m%d_%H%M%S)
SNAP="/mnt/workspace/paperplane-Moe_V1.001_snapshot_${TS}.tar.gz"
tar czf "$SNAP" -C "$ROOT" --exclude='.git' --exclude='__pycache__' \
    --exclude='*.pyc' .
ls -lh "$SNAP"
echo "PUSH_LOGS_DONE"
