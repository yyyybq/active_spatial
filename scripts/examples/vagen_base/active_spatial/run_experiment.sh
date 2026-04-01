#!/bin/bash
set -x

# =============================================================================
# 通用 PPO 训练入口 - 通过实验配置文件驱动
# =============================================================================
# 用法:
#   bash run_experiment.sh experiments/v2_entropy_fix.sh
#   bash run_experiment.sh experiments/v3_multi_scene.sh
#
# 实验配置文件只需 override 你要改的参数，其余走 baseline 默认值
# =============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON=/scratch/by2593/miniconda3/envs/vagen/bin/python

# ========================= BASELINE DEFAULTS =========================
# 这些是经过验证的基线值（来自 yr5xf955）
# 实验配置文件通过重新赋值来 override

EXPERIMENT_NAME="unnamed_$(date +%m%d_%H%M)"
ENV_CONFIG="env_config_balanced.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4
USE_GPU_HOLDER=true

# 模型
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

# Actor
ACTOR_LR="1e-6"
ENTROPY_COEFF="0.001"
GRAD_CLIP="1.0"
USE_KL_LOSS="False"
KL_LOSS_COEF="0.001"

# Critic
CRITIC_LR="1e-5"
CRITIC_WARMUP=0

# Rollout
TEMPERATURE="0.7"
TOP_P="0.95"
GPU_MEM_UTIL="0.5"
TP_SIZE=4

# Data
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=8
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=512
MAX_TRAJECTORY_LENGTH=14000

# Rollout Manager
MAX_TURNS=12
WINDOW_SIZE=5
MINI_BATCH_SIZE=8

# Trainer
SAVE_FREQ=50
TEST_FREQ=50
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# Algorithm
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"

# ========================= LOAD EXPERIMENT CONFIG =========================
if [ -z "$1" ]; then
    echo "ERROR: 请指定实验配置文件"
    echo "用法: bash run_experiment.sh experiments/v2_entropy_fix.sh"
    echo ""
    echo "可用实验:"
    ls "$SCRIPT_DIR/experiments/"*.sh 2>/dev/null | while read f; do
        name=$(basename "$f")
        desc=$(head -3 "$f" | grep "^# " | head -1 | sed 's/^# //')
        printf "  %-35s %s\n" "$name" "$desc"
    done
    exit 1
fi

EXPERIMENT_CONFIG="$1"
if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    # 尝试在 experiments/ 子目录找
    EXPERIMENT_CONFIG="$SCRIPT_DIR/experiments/$1"
fi
if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: 找不到实验配置: $1"
    exit 1
fi

echo "Loading experiment config: $EXPERIMENT_CONFIG"
source "$EXPERIMENT_CONFIG"

# ========================= ENVIRONMENT SETUP =========================
# 构建 CUDA_VISIBLE_DEVICES
GPU_LIST=$(seq -s, 0 $((NUM_TRAIN_GPUS - 1)))
if [ "$USE_GPU_HOLDER" = true ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_LIST},${RENDERING_GPU}"
else
    export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
fi

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager
export RAY_DEDUP_LOGS=0
export RAY_enable_metrics_collection=false
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export GS_RENDERER_VERBOSE=0
export ACTIVE_SPATIAL_ENV_VERBOSE=0
export RENDERING_GPU_ID=${RENDERING_GPU}
export PATH="/scratch/by2593/miniconda3/envs/vagen/bin:$PATH"

# ========================= PRINT CONFIG =========================
echo "=============================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Config:     $(basename $EXPERIMENT_CONFIG)"
echo "Model:      $MODEL_PATH"
echo "Env:        $ENV_CONFIG"
echo "Train GPUs: $GPU_LIST ($NUM_TRAIN_GPUS GPUs)"
echo "Render GPU: $RENDERING_GPU"
echo "----------------------------------------------"
echo "Actor LR=$ACTOR_LR  Critic LR=$CRITIC_LR"
echo "Entropy=$ENTROPY_COEFF  Grad Clip=$GRAD_CLIP"
echo "Temp=$TEMPERATURE  Top-p=$TOP_P"
echo "Critic Warmup=$CRITIC_WARMUP"
echo "Save freq=$SAVE_FREQ  Total steps=$TOTAL_STEPS"
echo "ADV=$ADV_ESTIMATOR  HL_gamma=$HIGH_LEVEL_GAMMA"
echo "=============================================="

# ========================= CLEANUP =========================
HOLDER_PIDS=()
cleanup() {
    echo "Cleaning up..."
    for pid in "${HOLDER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    pkill -P $$ -f "gpu_holder.py" 2>/dev/null || true
    exit 0
}
trap cleanup EXIT INT TERM

# ========================= CREATE DATASET =========================
$PYTHON -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/$ENV_CONFIG" \
    --train_path "data/$EXPERIMENT_NAME/train.parquet" \
    --test_path "data/$EXPERIMENT_NAME/test.parquet"

# ========================= GPU HOLDER (optional) =========================
HOLDER_LOG_DIR="logs/${EXPERIMENT_NAME}/gpu_holders"
mkdir -p "$HOLDER_LOG_DIR"

if [ "$USE_GPU_HOLDER" = true ]; then
    # 渲染卡 holder
    HOLDER_GPU=$RENDERING_GPU HOLDER_MEM_FRAC=0.75 HOLDER_TARGET=75 \
        $PYTHON "$SCRIPT_DIR/gpu_holder.py" \
        > "$HOLDER_LOG_DIR/holder_gpu${RENDERING_GPU}.log" 2>&1 &
    HOLDER_PIDS+=($!)
    echo "GPU Holder started for GPU $RENDERING_GPU, PID=$!, log=$HOLDER_LOG_DIR/holder_gpu${RENDERING_GPU}.log"

    # 训练卡 holders
    for GPU_ID in $(seq 0 $((NUM_TRAIN_GPUS - 1))); do
        HOLDER_GPU=$GPU_ID HOLDER_MEM_FRAC=0.0 HOLDER_TARGET=75 \
            $PYTHON "$SCRIPT_DIR/gpu_holder.py" \
            > "$HOLDER_LOG_DIR/holder_gpu${GPU_ID}.log" 2>&1 &
        HOLDER_PIDS+=($!)
        echo "GPU Holder started for GPU $GPU_ID, PID=$!, log=$HOLDER_LOG_DIR/holder_gpu${GPU_ID}.log"
    done

    # 等待 5s 后检查 holder 存活状态
    sleep 5
    echo "--- GPU Holder Health Check ---"
    ALL_HEALTHY=true
    for i in "${!HOLDER_PIDS[@]}"; do
        pid=${HOLDER_PIDS[$i]}
        if kill -0 "$pid" 2>/dev/null; then
            echo "  PID $pid: ALIVE"
        else
            echo "  PID $pid: DEAD! Check log: $HOLDER_LOG_DIR/"
            ALL_HEALTHY=false
        fi
    done
    if [ "$ALL_HEALTHY" = false ]; then
        echo "WARNING: Some GPU holders failed to start. Check logs in $HOLDER_LOG_DIR/"
        echo "Holder logs:"
        for f in "$HOLDER_LOG_DIR"/*.log; do
            echo "=== $(basename $f) ==="
            tail -20 "$f"
        done
    fi
    echo "-------------------------------"
fi

# ========================= PPO TRAINING =========================
$PYTHON -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.high_level_gamma=$HIGH_LEVEL_GAMMA \
    data.train_files=data/$EXPERIMENT_NAME/train.parquet \
    data.val_files=data/$EXPERIMENT_NAME/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.max_trajectory_length=$MAX_TRAJECTORY_LENGTH \
    data.image_key=images \
    data.truncation=left \
    +data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TRAJECTORY_LENGTH \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.limit_mm_per_prompt=15 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    critic.optim.lr=$CRITIC_LR \
    critic.model.use_remove_padding=True \
    critic.model.path=$MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.critic_warmup=$CRITIC_WARMUP \
    trainer.logger=['console','wandb'] \
    trainer.project_name='vagen_active_spatial' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_TRAIN_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_training_steps=$TOTAL_STEPS \
    rollout_manager.max_turns=$MAX_TURNS \
    rollout_manager.window_size=$WINDOW_SIZE \
    rollout_manager.use_multi_turn_reward=True \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.n_trajectory=1 \
    rollout_manager.use_service=False \
    +rollout_manager.mini_batch_size=$MINI_BATCH_SIZE \
    2>&1 | tee "${EXPERIMENT_NAME}.log"
