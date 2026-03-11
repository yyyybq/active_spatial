#!/bin/bash
# ActiveSpatial Ray Server 启动脚本
# 
# 重要说明：
#   此服务器使用 Ray 进行并行管理，但 **不通过 Ray 分配 GPU 资源**
#   GPU 通过 CUDA_VISIBLE_DEVICES 手动指定，不会与训练进程的 Ray 冲突
#   渲染和训练可以共享同一组 GPU（分时复用）
#
# 使用方法:
#   ./start_ray_server.sh [OPTIONS]
#
# 选项:
#   --port PORT          服务器端口 (默认: 5001)
#   --num-gpus N         使用的 GPU 数量 (默认: 4)
#   --renderers-per-gpu  每个 GPU 上的渲染器数量 (默认: 2)
#   --gs-root PATH       Gaussian Splatting 数据目录
#   --gpu-ids IDS        指定 GPU IDs，逗号分隔 (默认: 0,1,2,3...)
#   --ray-port PORT      独立 Ray 集群端口 (默认: 6380，避免与训练 Ray 6379 冲突)
#   --debug              启用调试模式

set -e

# 默认参数
PORT=5001
NUM_GPUS=4
RENDERERS_PER_GPU=2
GS_ROOT=""
GPU_IDS=""
RAY_PORT=6380  # 独立端口，不与训练 Ray (6379) 冲突
DEBUG=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --renderers-per-gpu)
            RENDERERS_PER_GPU="$2"
            shift 2
            ;;
        --gs-root)
            GS_ROOT="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --ray-port)
            RAY_PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$GS_ROOT" ]; then
    echo "Error: --gs-root is required"
    echo "Usage: $0 --gs-root /path/to/gs/data [OPTIONS]"
    exit 1
fi

# 生成 GPU IDs（如果未指定）
if [ -z "$GPU_IDS" ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
fi

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VAGEN_ROOT="$( cd "$SCRIPT_DIR/../../../.." && pwd )"

echo "============================================"
echo "  ActiveSpatial Ray Server"
echo "============================================"
echo "  Port:              $PORT"
echo "  Num GPUs:          $NUM_GPUS"
echo "  GPU IDs:           $GPU_IDS"
echo "  Renderers/GPU:     $RENDERERS_PER_GPU"
echo "  GS Root:           $GS_ROOT"
echo "  VAGEN Root:        $VAGEN_ROOT"
echo "  Ray Port:          $RAY_PORT (isolated cluster)"
echo "============================================"
echo ""
echo "  NOTE: This server uses an ISOLATED Ray cluster"
echo "        It will NOT interfere with training Ray"
echo "        GPUs are shared via CUDA, not Ray scheduling"
echo "============================================"

# 设置环境变量
export PYTHONPATH="${VAGEN_ROOT}:${PYTHONPATH}"

# 重要：不启动独立的 Ray head！
# 而是让 Python 代码内部使用 ray.init(ignore_reinit_error=True, num_gpus=0)
# 这样不会影响训练进程的 Ray 集群

# 如果存在旧的渲染服务 Ray，停止它
echo "Cleaning up old Ray processes for render server..."
pkill -f "ray_server.py" 2>/dev/null || true
sleep 1

# 启动服务器（不预先启动 Ray，让 Python 自己处理）
echo "Starting ActiveSpatial Ray Server..."
echo "NOTE: The server will initialize its own Ray context internally."
echo ""

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
python -m vagen.env.active_spatial.ray_server \
    --host 0.0.0.0 \
    --port $PORT \
    --num-gpus $NUM_GPUS \
    --renderers-per-gpu $RENDERERS_PER_GPU \
    --gs-root "$GS_ROOT" \
    --gpu-ids "$GPU_IDS" \
    $DEBUG
