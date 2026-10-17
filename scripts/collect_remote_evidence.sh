#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  远端证据取证脚本（在【服务器】上执行，不是本地）
#
#  目的：把服务器上的实验证据、日志、ckpt 打包，供换电脑时带走
#
#  用法:
#     bash collect_remote_evidence.sh
#     → 生成 ~/snowflake_remote_backup_<时间戳>.tar.gz
# ══════════════════════════════════════════════════════════════════
set -uo pipefail

TS=$(date +%Y%m%dT%H%M%S)
OUT="$HOME/snowflake_remote_backup_${TS}"
mkdir -p "$OUT"

echo "═══ 取证开始：$(date) ═══"
echo "输出目录: $OUT"
echo

BASE="/root/private_data/snowflake"

# ── 1. 实验证据（10 seeds 的 record.json + steps.csv）───────────
echo "[1/6] 实验证据 output/runs/"
for d in "${BASE}/snowflake_open_v7/snowflake_open/output/runs" \
         "${BASE}/snowflake_open/output/runs" \
         "${BASE}/output/runs"; do
    if [[ -d "$d" ]]; then
        cp -r "$d" "$OUT/runs" 2>/dev/null
        echo "  ✓ 已从 $d 复制"
        break
    fi
done
[[ -d "$OUT/runs" ]] && echo "  runs 目录: $(find "$OUT/runs" -type f | wc -l) 个文件" \
                     || echo "  ⚠ 未找到 runs 目录"

# ── 2. 训练日志 ────────────────────────────────────────────────
echo "[2/6] 训练日志 *.log"
mkdir -p "$OUT/logs"
find "$BASE" -maxdepth 3 -name "*.log" -exec cp {} "$OUT/logs/" \; 2>/dev/null
echo "  ✓ $(ls "$OUT/logs" 2>/dev/null | wc -l) 个 log"

# ── 3. Checkpoint（体积大，单独处理）──────────────────────────
echo "[3/6] Checkpoints"
mkdir -p "$OUT/ckpt"
find "$BASE" -maxdepth 4 \( -name "*.pt" -o -name "*.pth" \) \
     -size -100M -exec cp {} "$OUT/ckpt/" \; 2>/dev/null
echo "  ✓ $(ls "$OUT/ckpt" 2>/dev/null | wc -l) 个 ckpt（<100MB 的）"
echo "  ⚠ 大于 100MB 的未复制，如需请手动："
find "$BASE" -maxdepth 4 \( -name "*.pt" -o -name "*.pth" \) -size +100M 2>/dev/null \
    | head -10 | sed 's/^/     /'

# ── 4. 数据集指纹（不复制数据本体，只记录元信息）──────────────
echo "[4/6] 数据集指纹"
mkdir -p "$OUT/data_fingerprint"
for f in "${BASE}/tinystories_100mb.txt" "${BASE}"/*.txt; do
    [[ -f "$f" ]] || continue
    {
        echo "=== $f ==="
        echo "size_bytes: $(stat -c%s "$f")"
        echo "md5: $(md5sum "$f" | cut -d' ' -f1)"
        echo "lines: $(wc -l < "$f")"
        echo "chars: $(wc -m < "$f")"
    } >> "$OUT/data_fingerprint/fingerprints.txt"
done
echo "  ✓ 已记录 $(grep -c '^===' "$OUT/data_fingerprint/fingerprints.txt" 2>/dev/null || echo 0) 个数据集"

# ── 5. 环境信息 ────────────────────────────────────────────────
echo "[5/6] 环境快照"
{
    echo "=== 时间 ==="; date
    echo "=== 主机名 ==="; hostname
    echo "=== Python ==="; python3 --version 2>&1
    echo "=== torch ==="
    python3 -c "import torch; print(torch.__version__); \
print('version.cuda:', torch.version.cuda); \
print('version.hip:', getattr(torch.version,'hip',None)); \
print('cuda_available:', torch.cuda.is_available()); \
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>&1
    echo "=== GPU ==="; hy-smi 2>/dev/null | head -20 || nvidia-smi 2>/dev/null | head -20 || echo "无 GPU 工具"
    echo "=== 磁盘 ==="; df -h "$BASE" 2>/dev/null
} > "$OUT/environment.txt"
echo "  ✓ environment.txt"

# ── 6. 代码（服务器上的版本）──────────────────────────────────
echo "[6/6] 服务器代码快照"
mkdir -p "$OUT/code"
for d in "${BASE}/snowflake_open_v7/snowflake_open" "${BASE}/snowflake_open"; do
    if [[ -d "$d" ]]; then
        cp "$d"/*.py "$OUT/code/" 2>/dev/null
        cp "$d"/*.md "$OUT/code/" 2>/dev/null
        echo "  ✓ 已从 $d 复制 $(ls "$OUT/code" | wc -l) 个文件"
        break
    fi
done

# ── 打包 ───────────────────────────────────────────────────────
echo
echo "═══ 打包 ═══"
TARBALL="$HOME/snowflake_remote_backup_${TS}.tar.gz"
tar -czf "$TARBALL" -C "$HOME" "snowflake_remote_backup_${TS}" 2>/dev/null
rm -rf "$OUT"

SIZE=$(du -h "$TARBALL" | cut -f1)
echo "✅ 完成: $TARBALL  ($SIZE)"
echo
echo "下载到本地:"
echo "  scp root@<服务器>:${TARBALL} ~/"
