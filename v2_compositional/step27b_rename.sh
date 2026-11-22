#!/bin/bash
# paperplane-Moe V1.001 · 改名 B 落地（默认 dry-run，EXEC=1 才执行）
# 目标：新代码全部 paperplane_* 名字；旧 snowflake_* 保留为兼容层，
#       不批量 git mv 历史文件（会断 import 与 20+ 个 json 的路径）。
set -uo pipefail
cd "$(dirname "$0")" || exit 1
ROOT="$(cd .. && pwd)"
echo "  仓库根 = $ROOT"

echo "===== [0] 硬前置：训练必须全停 ====="
if ps aux | grep -q "[t]rain_.*\.py"; then
  echo "  ❌ 仍有训练进程，先停再改名（避免改名/架构两个混淆变量叠加）"
  ps aux | grep "[t]rain_.*\.py" | awk '{print "   ",$2,$11,$12}'
  exit 1
fi
echo "  OK 无训练进程"

echo "===== [1] 新命名包与别名模块 ====="
for f in ../paperplane_moe/__init__.py ../paperplane_moe/biology.py \
         ../paperplane_moe/telemetry.py ../paperplane_moe/compose.py \
         ../paperplane_moe/legacy_shim.py ../paperplane_moe/paperplane_v4.py \
         ../paperplane_moe/paperplane_B.py \
         ../paperplane_moe/paperplane_compose.py; do
  if [ -f "$f" ]; then echo "  OK  $f"; else echo "  缺  $f → 先把交付包 src/paperplane_moe 拷到仓库根"; fi
done

if [ "${EXEC:-0}" != "1" ]; then
  echo "===== [dry-run] 将执行但不执行 ====="
  echo "  git tag V1.000-before-rename"
  echo "  生成 paperplane_train.py（新名训练入口 shim）"
  echo "  git add ../paperplane_moe paperplane_train.py train_bio.py verify_bio.py"
  echo "  确认后重跑：EXEC=1 bash step27b_rename.sh"
  echo "RENAME_DRYRUN_DONE"
  exit 0
fi

echo "===== [2] 执行 ====="
git tag V1.000-before-rename 2>/dev/null || echo "  tag 已存在，跳过"
cat > paperplane_train.py <<'PY'
# -*- coding: utf-8 -*-
"""paperplane-Moe 新名训练入口（旧 train_bio.py 保持可用）。"""
from train_bio import main

if __name__ == "__main__":
    main()
PY
python3 -c "import ast; ast.parse(open('paperplane_train.py').read()); print('  shim 语法 OK')"
git add ../paperplane_moe paperplane_train.py train_bio.py verify_bio.py \
        step27a_biology_smoke.sh step27b_rename.sh step27c_push_logs.sh \
        step27d_bio_summary.py 2>/dev/null || true

echo "===== [3] 改名后必须重跑的零污染自检（在 DSW 执行）====="
echo "  python3 verify_bio.py 2>&1 | tee verify_bio.log"
echo "  python3 train_bio.py --data data_25mb.txt --cell-type B --seed 0 \\"
echo "    --d 256 --h 128 --L 4 --n-cells 4 --variant shared --rank 4 \\"
echo "    --temp 5.0 --bs 64 --seq-len 128 --lr 3e-4 --max-steps 3000 \\"
echo "    --split-k 8 --split-at 1500 --tag bio_s0_B --zero-check"
echo "  期望：对齐点 301、loss max|Δ| = 0.000e+00、Δval = 0；过不了就 git checkout V1.000-before-rename"

echo "===== [4] 远端改名（只能在 GitHub 网页做）====="
echo "  Settings → Rename → 新 URL 告诉我，然后："
echo "  git remote set-url origin git@github.com:wszap/paperplane-Moe.git"
echo "RENAME_DONE"
