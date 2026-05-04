
'''
nohup bash scripts/examples/vagen_base/active_spatial/hold_extra_gpus.sh 5 6 > /tmp/hold_extra.log 2>&1 &
'''
#!/bin/bash
# =============================================================================
# hold_extra_gpus.sh — 占用多余显卡，保持集群 GPU 利用率 ≥70%
# =============================================================================
# 用法（从任意目录运行，脚本会自动找到 gpu_holder.py）:
#
#   # 方式1：显式指定要占用的 GPU 列表（推荐）
#   bash hold_extra_gpus.sh 5 6
#   bash hold_extra_gpus.sh 5 6 7
#
#   # 方式2：自动占用「训练+渲染卡之外」的所有 GPU
#   #  训练用 GPU 0-3，渲染用 GPU 4 → 脚本自动占用 5,6
#   TRAIN_GPUS="0,1,2,3" RENDER_GPU=4 bash hold_extra_gpus.sh
#
#   # 后台运行
#   bash hold_extra_gpus.sh 5 6 > /tmp/hold_extra.log 2>&1 &
#
# 停止：
#   pkill -f "hold_extra_gpus\|HOLDER_GPU=[56]"   # 同时杀掉本脚本和子进程
#   # 或者记录本脚本 PID，用 kill <PID> 触发 EXIT trap
#
# 环境变量（可选覆盖）：
#   HOLDER_TARGET    目标利用率 % (默认 75)
#   HOLDER_MEM_FRAC  显存占用比例 (默认 0.75)
#   PYTHON           Python 解释器路径
#   LOG_DIR          日志目录 (默认 /tmp/gpu_holders)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HOLDER_SCRIPT="$SCRIPT_DIR/gpu_holder.py"
PYTHON="${PYTHON:-/scratch/by2593/miniconda3/envs/vagen/bin/python}"
HOLDER_TARGET="${HOLDER_TARGET:-75}"
HOLDER_MEM_FRAC="${HOLDER_MEM_FRAC:-0.75}"
LOG_DIR="${LOG_DIR:-/tmp/gpu_holders}"

# ---------- 确定要占用的 GPU 列表 ----------
GPU_IDS=()

if [[ $# -gt 0 ]]; then
    # 命令行参数：直接指定 GPU ID
    for arg in "$@"; do
        GPU_IDS+=("$arg")
    done
else
    # 自动模式：找出所有 GPU，排除训练卡和渲染卡
    TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
    RENDER_GPU="${RENDER_GPU:-4}"
    USED_GPUS="${TRAIN_GPUS},${RENDER_GPU}"

    TOTAL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    for i in $(seq 0 $((TOTAL_GPUS - 1))); do
        if ! echo "$USED_GPUS" | tr ',' '\n' | grep -qx "$i"; then
            GPU_IDS+=("$i")
        fi
    done
fi

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "没有需要占用的 GPU，退出。"
    exit 0
fi

echo "将占用 GPU: ${GPU_IDS[*]}  (目标利用率=${HOLDER_TARGET}%，显存占用=${HOLDER_MEM_FRAC})"

# ---------- 启动 holder 进程 ----------
mkdir -p "$LOG_DIR"
HOLDER_PIDS=()

cleanup() {
    echo "收到退出信号，正在清理 ${#HOLDER_PIDS[@]} 个 holder 进程..."
    for pid in "${HOLDER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null && echo "  killed PID $pid" || true
    done
}
trap cleanup EXIT INT TERM

for GPU_ID in "${GPU_IDS[@]}"; do
    LOG_FILE="$LOG_DIR/holder_gpu${GPU_ID}.log"
    HOLDER_GPU="$GPU_ID" \
    HOLDER_MEM_FRAC="$HOLDER_MEM_FRAC" \
    HOLDER_TARGET="$HOLDER_TARGET" \
        "$PYTHON" "$HOLDER_SCRIPT" > "$LOG_FILE" 2>&1 &
    PID=$!
    HOLDER_PIDS+=("$PID")
    echo "  GPU $GPU_ID holder started (PID=$PID, log=$LOG_FILE)"
done

# ---------- 等待 holder 启动并检查存活 ----------
sleep 5
echo "--- Health Check ---"
ALL_OK=true
for i in "${!HOLDER_PIDS[@]}"; do
    PID="${HOLDER_PIDS[$i]}"
    GPU="${GPU_IDS[$i]}"
    if kill -0 "$PID" 2>/dev/null; then
        echo "  GPU $GPU (PID $PID): ALIVE"
    else
        echo "  GPU $GPU (PID $PID): DEAD! 查看日志: $LOG_DIR/holder_gpu${GPU}.log"
        ALL_OK=false
    fi
done

if [[ "$ALL_OK" = false ]]; then
    echo "部分 holder 启动失败，检查日志："
    for GPU_ID in "${GPU_IDS[@]}"; do
        echo "=== GPU $GPU_ID ===" && tail -10 "$LOG_DIR/holder_gpu${GPU_ID}.log" 2>/dev/null
    done
    exit 1
fi

echo "--- 所有 holder 运行正常，等待中（Ctrl+C 退出）---"

# 持续监控，每 60s 打印一次利用率；自动重启挂掉的 holder
while true; do
    sleep 60
    echo -n "[$(date '+%H:%M:%S')] GPU util: "
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader \
        | awk -F', ' -v gpus="${GPU_IDS[*]}" '
            BEGIN{n=split(gpus,g,""); for(i=1;i<=n;i++) keep[g[i]]=1}
            keep[$1]{printf "GPU%s=%s  ",$1,$2}
            END{print ""}
        '

    # 重启挂掉的 holder
    for i in "${!HOLDER_PIDS[@]}"; do
        PID="${HOLDER_PIDS[$i]}"
        GPU="${GPU_IDS[$i]}"
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "  [WARN] GPU $GPU holder (PID $PID) died, restarting..."
            LOG_FILE="$LOG_DIR/holder_gpu${GPU}.log"
            HOLDER_GPU="$GPU" HOLDER_MEM_FRAC="$HOLDER_MEM_FRAC" HOLDER_TARGET="$HOLDER_TARGET" \
                "$PYTHON" "$HOLDER_SCRIPT" >> "$LOG_FILE" 2>&1 &
            HOLDER_PIDS[$i]=$!
            echo "  GPU $GPU restarted (new PID=${HOLDER_PIDS[$i]})"
        fi
    done
done
