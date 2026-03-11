#!/bin/bash
#SBATCH --job-name=gs_render_server
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=gs_render_server_%j.out
#SBATCH --error=gs_render_server_%j.err

# ============================================================
# Gaussian Splatting 渲染服务器启动脚本
# ============================================================
#
# 用法:
#   sbatch start_render_server.sbatch
#
# 或直接运行:
#   bash start_render_server.sbatch
#
# ============================================================

# 配置参数
GS_ROOT="${GS_ROOT:-/scratch/data/interior_gs}"  # 3D数据根目录
PORT="${PORT:-8777}"                              # 服务端口
NUM_SHARDS="${NUM_SHARDS:-8}"                     # 进程分片数
MAX_RENDERERS="${MAX_RENDERERS:-8}"               # 每个worker的渲染器缓存数

# 加载环境
source ~/.bashrc
conda activate viewsuite 2>/dev/null || conda activate base

# 设置 CUDA
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 打印配置
echo "=============================================="
echo "Gaussian Splatting Render Server"
echo "=============================================="
echo "GS_ROOT:        ${GS_ROOT}"
echo "PORT:           ${PORT}"
echo "NUM_SHARDS:     ${NUM_SHARDS}"
echo "MAX_RENDERERS:  ${MAX_RENDERERS}"
echo "CUDA_DEVICES:   ${CUDA_VISIBLE_DEVICES}"
echo "HOSTNAME:       $(hostname)"
echo "=============================================="

# 检查数据目录
if [ ! -d "${GS_ROOT}" ]; then
    echo "ERROR: GS_ROOT directory not found: ${GS_ROOT}"
    exit 1
fi

# 检查 PLY 文件
PLY_COUNT=$(find "${GS_ROOT}" -name "*.ply" 2>/dev/null | wc -l)
echo "Found ${PLY_COUNT} PLY files in ${GS_ROOT}"

# 启动服务器
echo "Starting render server on port ${PORT}..."
echo "WebSocket URL: ws://$(hostname):${PORT}/render/interiorgs"
echo ""

python -m view_suite.interiorGS.service.gs_render_service \
    --interiorgs_root="${GS_ROOT}" \
    --num_shards="${NUM_SHARDS}" \
    --max_renderers_per_worker="${MAX_RENDERERS}" \
    --host="0.0.0.0" \
    --port="${PORT}" \
    --log_level="info"
