#!/bin/bash
# paperplaneMoe / step25 落地：把补丁包文件放进 v2_compositional/ 并 git add
# 用法：在 v2_compositional/ 下  bash install_step25.sh
#       或  DST=/path/to/v2_compositional bash /path/to/bundle/install_step25.sh
set -uo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DST="${DST:-$(pwd)}"
[ -d "$DST" ] || { echo "目标目录不存在: $DST"; exit 1; }

FILES="compose_v10.py train_compose.py step25_compose_ab.sh \
step25_compose_summary.py PAPERPLANEMOE_STEP25.md install_step25.sh"

for f in $FILES; do
  [ -f "$SRC/$f" ] || { echo "补丁包缺文件: $f"; exit 1; }
  if [ "$SRC" != "$DST" ]; then cp "$SRC/$f" "$DST/$f"; fi
done
chmod +x "$DST/step25_compose_ab.sh" "$DST/install_step25.sh"

cd "$DST" || exit 1
echo "===== git 状态 ====="
git status -sb | head -5
echo "===== 语法自检 ====="
python3 -c "
import ast
for f in ['compose_v10.py','train_compose.py','step25_compose_summary.py']:
    ast.parse(open(f, encoding='utf-8').read()); print('  OK', f)
"
bash -n step25_compose_ab.sh && echo "  OK step25_compose_ab.sh"

echo "===== 暂存 ====="
git add compose_v10.py train_compose.py step25_compose_ab.sh \
        step25_compose_summary.py PAPERPLANEMOE_STEP25.md install_step25.sh
git diff --cached --stat | tail -10
echo
echo "下一步："
echo "  1) 跑 20 步 smoke（见 PAPERPLANEMOE_STEP25.md 第 7 节）"
echo "  2) nohup bash step25_compose_ab.sh > step25.log 2>&1 &"
echo "  3) python3 step25_compose_summary.py | tee step25_summary.log"
echo "  4) 确认后 git commit -m 'paperplaneMoe: step25 true-composable expert + grouped clip A/B' && git push"
