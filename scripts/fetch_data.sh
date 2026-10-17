#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  数据获取 —— 阿里云 PAI-DSW 专用（HF 直连不通）
#
#  背景：a10_rebuild.sh --setup 里 wget huggingface.co 失败，
#        下到的是错误页/空内容，解压失败，最终生成【空文件】
#        导致 vocab=0、randint(0,-129) 崩溃
#
#  用法:
#     bash fetch_data.sh --diag        # 诊断当前数据文件
#     bash fetch_data.sh --auto        # 自动尝试所有方案
#     bash fetch_data.sh --modelscope  # 方案A：魔搭（阿里云自家，推荐）
#     bash fetch_data.sh --mirror      # 方案B：HF 镜像
#     bash fetch_data.sh --synth       # 方案D：合成数据（仅验证代码）
# ══════════════════════════════════════════════════════════════════
set -uo pipefail

DATA="${DATA:-/root/tinystories_100mb.txt}"
MIN_BYTES=10485760      # 小于 10MB 视为无效

C_GREEN='\033[0;32m'; C_RED='\033[0;31m'; C_YELLOW='\033[1;33m'; C_NC='\033[0m'
log()  { echo -e "${C_GREEN}[OK]${C_NC} $*"; }
warn() { echo -e "${C_YELLOW}[WARN]${C_NC} $*"; }
err()  { echo -e "${C_RED}[ERR]${C_NC} $*"; }

# ────────────────────────────────────────────────────────────────
valid() {
    [[ -f "$DATA" ]] || return 1
    local sz; sz=$(stat -c%s "$DATA" 2>/dev/null || echo 0)
    [[ "$sz" -ge "$MIN_BYTES" ]] || return 1
    return 0
}

# ────────────────────────────────────────────────────────────────
diag() {
    echo "═══ 数据诊断 ═══"
    echo "  目标文件: $DATA"
    if [[ ! -f "$DATA" ]]; then
        err "文件不存在"
        return 1
    fi
    local sz; sz=$(stat -c%s "$DATA")
    echo "  大小     : $sz bytes ($(( sz / 1024 / 1024 )) MB)"
    echo "  行数     : $(wc -l < "$DATA")"
    echo "  字符数   : $(wc -m < "$DATA")"

    if [[ "$sz" -lt "$MIN_BYTES" ]]; then
        err "文件过小（<$(( MIN_BYTES / 1024 / 1024 ))MB），视为无效"
        echo
        echo "  前 100 字节预览:"
        head -c 100 "$DATA" | od -c | head -5
        return 1
    fi

    log "大小合格"
    echo
    echo "  前 200 字符预览:"
    head -c 200 "$DATA"
    echo
    echo
    # 检查是否像正常文本（可打印字符占比）
    local printable
    printable=$(head -c 10000 "$DATA" | tr -cd '[:print:]\n' | wc -c)
    echo "  前 10000 字节中可打印字符: $printable"
    if [[ "$printable" -lt 8000 ]]; then
        err "可打印字符过少，文件可能是二进制垃圾/HTML 错误页"
        return 1
    fi
    log "内容看起来是正常文本"
    return 0
}

# ────────────────────────────────────────────────────────────────
fetch_modelscope() {
    echo "═══ 方案 A：魔搭 ModelScope（阿里云自家，最稳）═══"
    pip install -q modelscope 2>/dev/null || { err "pip install modelscope 失败"; return 1; }

    python3 - <<'PYEOF'
import os, sys
out = os.environ.get("DATA", "/root/tinystories_100mb.txt")
try:
    from modelscope.msdatasets import MsDataset
    ds = MsDataset.load("AI-ModelScope/TinyStories", split="train")
except Exception as e:
    print(f"[ERR] ModelScope 加载失败: {e}")
    print("     数据集 ID 可能不对，可在 https://modelscope.cn 搜索 TinyStories")
    sys.exit(1)

n = 0
with open(out, "w", encoding="utf-8") as f:
    for row in ds:
        t = row.get("text") or row.get("content") or ""
        f.write(t + "\n")
        n += len(t)
        if n >= 104857600:
            break
print(f"[OK] 已写入 {out}: {n} 字符")
PYEOF
    valid && { log "方案 A 成功"; return 0; }
    err "方案 A 生成的文件无效"
    return 1
}

# ────────────────────────────────────────────────────────────────
fetch_mirror() {
    echo "═══ 方案 B：HF 镜像 hf-mirror.com ═══"
    export HF_ENDPOINT=https://hf-mirror.com

    # 先测连通
    if ! curl -sS -m 15 -o /dev/null -w "%{http_code}" \
         https://hf-mirror.com 2>/dev/null | grep -q "200\|301\|302"; then
        err "hf-mirror.com 也连不上"
        return 1
    fi
    log "镜像可达，开始下载"

    rm -f /tmp/ts.tar.gz
    wget -q --show-progress --timeout=30 --tries=3 \
      https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz \
      -O /tmp/ts.tar.gz || { err "下载失败"; return 1; }

    local sz; sz=$(stat -c%s /tmp/ts.tar.gz 2>/dev/null || echo 0)
    echo "  下载大小: $sz bytes"
    if [[ "$sz" -lt 1000000 ]]; then
        err "下载内容过小，可能是错误页"
        head -c 300 /tmp/ts.tar.gz
        return 1
    fi

    rm -rf /tmp/tsx && mkdir -p /tmp/tsx
    tar -xzf /tmp/ts.tar.gz -C /tmp/tsx 2>/dev/null || { err "解压失败"; return 1; }

    local found
    found=$(find /tmp/tsx -name "*.txt" -size +1M | head -5)
    if [[ -z "$found" ]]; then
        err "解压后没找到 txt"
        find /tmp/tsx -type f | head -20
        return 1
    fi

    cat $(find /tmp/tsx -name "*.txt" -size +1M) > "$DATA.raw"
    head -c 104857600 "$DATA.raw" > "$DATA"
    rm -f "$DATA.raw"
    valid && { log "方案 B 成功"; return 0; }
    err "方案 B 生成的文件无效"
    return 1
}

# ────────────────────────────────────────────────────────────────
make_synth() {
    echo "═══ 方案 D：合成数据（仅用于验证代码能跑通）═══"
    warn "⚠ 合成数据的 loss/eff 没有科学意义，【不能】用于论文结论"
    python3 - <<'PYEOF'
import os, random
out = os.environ.get("DATA", "/root/tinystories_100mb.txt")
random.seed(0)
words = ["the","a","cat","dog","ran","jumped","happy","sad","big","small",
         "tree","house","boy","girl","said","went","saw","liked","very","good"]
target = 104857600
buf = []
n = 0
while n < target:
    s = " ".join(random.choice(words) for _ in range(12)) + ". "
    buf.append(s)
    n += len(s)
with open(out, "w", encoding="utf-8") as f:
    f.write("".join(buf)[:target])
print(f"[OK] 合成数据已写入 {out} ({n} 字符)")
PYEOF
    valid && { log "方案 D 成功"; return 0; }
    return 1
}

# ────────────────────────────────────────────────────────────────
auto() {
    diag && { log "数据已有效，无需处理"; return 0; }
    echo
    fetch_modelscope && return 0
    echo
    fetch_mirror && return 0
    echo
    echo "═══ 自动方案全部失败 ═══"
    cat <<'EOF'

  ★ 请改用【网页上传】：

  1. 本地 Windows 浏览器打开
     https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz
     下载（约 200MB）

  2. 在 PAI-DSW 左侧文件栏，点击上传按钮，把 tar.gz 传到 /root/

  3. 在终端执行:
       cd /root && tar -xzf TinyStories_all_data.tar.gz
       cat /root/*.txt > /root/tinystories_all_data.txt
       head -c 104857600 /root/tinystories_all_data.txt > /root/tinystories_100mb.txt
       bash fetch_data.sh --diag

  或者先用方案 D 合成数据验证代码能跑通（不用于论文）:
       bash fetch_data.sh --synth
EOF
    return 1
}

case "${1:-}" in
    --diag)        diag ;;
    --auto)        auto ;;
    --modelscope)  fetch_modelscope ;;
    --mirror)      fetch_mirror ;;
    --synth)       make_synth ;;
    *) echo "用法: $0 --diag|--auto|--modelscope|--mirror|--synth"; exit 1 ;;
esac
