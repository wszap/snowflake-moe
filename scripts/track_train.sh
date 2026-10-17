#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  后台训练 + 实时跟踪（阿里云 PAI-DSW / 任何 Linux）
#
#  用法:
#     bash track_train.sh start            # 后台启动 smoke（3000 步）
#     bash track_train.sh start seeds      # 后台启动 10 seeds
#     bash track_train.sh watch            # 实时跟踪日志（Ctrl+C 退出，不中断训练）
#     bash track_train.sh status           # 看一眼当前进度
#     bash track_train.sh stop             # 停止后台训练
# ══════════════════════════════════════════════════════════════════
set -uo pipefail

DIR="${DIR:-/mnt/workspace/snowflake-moe/v2_compositional}"
DATA="${DATA:-data_25mb.txt}"
LOG="${LOG:-$DIR/train_bg.log}"
PIDFILE="$DIR/train_bg.pid"

C_GREEN='\033[0;32m'; C_RED='\033[0;31m'; C_YELLOW='\033[1;33m'; C_NC='\033[0m'

start() {
    local mode="${1:-smoke}"
    cd "$DIR" || { echo "目录不存在: $DIR"; exit 1; }

    # 数据先校验（防止又出现空数据）
    if [[ ! -f "$DATA" ]]; then
        echo -e "${C_RED}[ERR] 数据文件不存在: $DIR/$DATA${C_NC}"
        echo "  当前目录文件:"; ls -la *.txt 2>/dev/null
        exit 1
    fi
    local sz; sz=$(stat -c%s "$DATA")
    if [[ "$sz" -lt 10000000 ]]; then
        echo -e "${C_RED}[ERR] 数据文件过小: $sz bytes（空数据指纹）${C_NC}"
        exit 1
    fi
    echo -e "${C_GREEN}[OK] 数据校验通过: $DATA ($sz bytes)${C_NC}"

    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo -e "${C_YELLOW}[WARN] 已有训练在运行 (PID $(cat "$PIDFILE"))${C_NC}"
        echo "  先 stop，或用 track_train.sh watch 跟踪现有的"
        exit 1
    fi

    if [[ "$mode" == "seeds" ]]; then
        echo "[启动] 10 seeds 后台训练"
        nohup bash -c '
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "═══════ SEED $s ═══════"
  python train_lock60.py --scale tiny \
      --data '"$DATA"' --delta-scale 0.5 \
      --max-steps 3000 --seed $s \
      --save-ckpt ckpt_s$s.pt
done
echo "ALL_SEEDS_DONE"
' > "$LOG" 2>&1 &
    else
        echo "[启动] smoke 3000 步后台训练"
        nohup python train_lock60.py --scale tiny \
            --data "$DATA" --delta-scale 0.5 \
            --max-steps 3000 --seed 0 \
            --save-ckpt ckpt_smoke.pt > "$LOG" 2>&1 &
    fi

    echo $! > "$PIDFILE"
    echo -e "${C_GREEN}[OK] 已启动 PID=$(cat "$PIDFILE")${C_NC}"
    echo "  日志: $LOG"
    echo "  跟踪: bash track_train.sh watch"
    sleep 3
    echo
    echo "─── 前 20 行 ───"
    head -20 "$LOG"
}

watch() {
    [[ -f "$LOG" ]] || { echo "日志不存在: $LOG"; exit 1; }
    echo "跟踪: $LOG   (Ctrl+C 退出，不会中断训练)"
    echo
    tail -f "$LOG" | while IFS= read -r line; do
        # 高亮关键行
        case "$line" in
            *"loss"*|*"eff"*|*"step"*)
                printf "\033[0;36m%s\033[0m\n" "$line" ;;
            *"core_stat"*|*"d_z"*|*"CI"*)
                printf "\033[0;32m%s\033[0m\n" "$line" ;;
            *"Error"*|*"Traceback"*|*"RuntimeError"*)
                printf "\033[0;31m%s\033[0m\n" "$line" ;;
            *"SEED"*|*"ALL_SEEDS_DONE"*)
                printf "\033[1;33m%s\033[0m\n" "$line" ;;
            *) echo "$line" ;;
        esac
    done
}

status() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo -e "${C_GREEN}运行中${C_NC}  PID=$(cat "$PIDFILE")"
    else
        echo -e "${C_YELLOW}未运行${C_NC}"
        [[ -f "$PIDFILE" ]] && echo "  （上一次 PID $(cat "$PIDFILE") 已结束）"
    fi
    [[ -f "$LOG" ]] || { echo "无日志"; return; }
    echo
    echo "─── 关键指标 ───"
    grep -E "^\[data\]" "$LOG" | tail -1
    grep -iE "params|参数量" "$LOG" | tail -1
    echo
    echo "─── 最新 5 条训练行 ───"
    grep -iE "step|loss" "$LOG" | tail -5
    echo
    echo "─── 错误（若有）───"
    grep -iE "Error|Traceback" "$LOG" | tail -5 || echo "  无"
    echo
    echo "─── 进度 ───"
    grep -c "SEED" "$LOG" 2>/dev/null | xargs -I{} echo "  已完成 seed 段: {}"
    grep -q "ALL_SEEDS_DONE" "$LOG" && echo "  ✅ 全部完成"
}

stop() {
    if [[ -f "$PIDFILE" ]]; then
        local pid; pid=$(cat "$PIDFILE")
        kill "$pid" 2>/dev/null && echo "已停止 PID=$pid"
        rm -f "$PIDFILE"
    else
        echo "无运行中任务"
    fi
}

case "${1:-status}" in
    start)  start "${2:-smoke}" ;;
    watch)  watch ;;
    status) status ;;
    stop)   stop ;;
    *) echo "用法: $0 start [smoke|seeds] | watch | status | stop" ;;
esac
