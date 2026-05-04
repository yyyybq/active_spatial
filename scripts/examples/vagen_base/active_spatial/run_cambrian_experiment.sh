#!/bin/bash
set -x

# =============================================================================
# Cambrian-S 通用 PPO 训练入口 - 通过实验配置文件驱动
# =============================================================================
# 用法 (从项目根目录 VAGEN/ 下运行):
#   cd /scratch/by2593/project/Active_Spatial/VAGEN && \
#   export CAMBRIAN_MODEL_PATH=/scratch/by2593/hf_cache/cambrian-s-7b && \
#   export CAMBRIAN_SRC=/scratch/by2593/project/Active_Spatial/cambrian-s && \
#   nohup bash scripts/examples/vagen_base/active_spatial/run_cambrian_experiment.sh cambrian_v1_baseline.sh > cambrian_v1.log 2>&1 &
#
# 实验配置文件只需 override 你要改的参数，其余走 baseline 默认值
# =============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON=/scratch/by2593/miniconda3/envs/vagen/bin/python

# ========================= BASELINE DEFAULTS =========================
# Cambrian-S 7B 的基线参数 (来自 run_cambrian_7gpu_v2.sh)

EXPERIMENT_NAME="cambrian_unnamed_$(date +%m%d_%H%M)"
ENV_CONFIG="env_config_v2.yaml"
NUM_TRAIN_GPUS=6
RENDERING_GPU=6
USE_GPU_HOLDER=true

# 模型 (从环境变量读取)
CAMBRIAN_MODEL_PATH="${CAMBRIAN_MODEL_PATH:-/path/to/cambrian-s-weights}"
CAMBRIAN_SRC="${CAMBRIAN_SRC:-/scratch/by2593/project/Active_Spatial/cambrian-s}"

# Actor
ACTOR_LR="1e-6"
ENTROPY_COEFF="0.001"
GRAD_CLIP="1.0"
USE_KL_LOSS="False"
KL_LOSS_COEF="0.001"
KL_LOSS_TYPE="mse"
ACTOR_OPTIMIZER_OFFLOAD="False"

# Critic
CRITIC_LR="1e-5"
CRITIC_WARMUP=0
CRITIC_GRAD_CLIP="1.0"
CLIPRANGE_VALUE="0.5"
CRITIC_OPTIMIZER_OFFLOAD="False"

# Rollout (HF, not vLLM — Cambrian-S not supported by vLLM)
TEMPERATURE="0.7"
TOP_P="0.95"
RESPONSE_LENGTH=512

# Data
TRAIN_BATCH_SIZE=30
VAL_BATCH_SIZE=6
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=512
MAX_TRAJECTORY_LENGTH=18000

# Rollout Manager (Cambrian-specific)
MAX_TURNS=12
WINDOW_SIZE=5
MINI_BATCH_SIZE=6
RM_MAX_PROMPT_LENGTH=8192
SI_TOKEN_LEN=729
ANYRES_MAX_SUBIMAGES=9

# Trainer
SAVE_FREQ=50
TEST_FREQ=50
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# Algorithm
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"

# Resume
RESUME_MODE="auto"

# GPU Holder
RENDER_GPU_HOLDER_TARGET=90
TRAIN_GPU_HOLDER_TARGET=75

# ========================= LOAD EXPERIMENT CONFIG =========================
if [ -z "$1" ]; then
    echo "ERROR: 请指定实验配置文件"
    echo "用法: bash run_cambrian_experiment.sh cambrian_experiments/cambrian_v2_stabilize.sh"
    echo ""
    echo "可用实验:"
    ls "$SCRIPT_DIR/cambrian_experiments/"*.sh 2>/dev/null | while read f; do
        name=$(basename "$f")
        desc=$(head -3 "$f" | grep "^# " | head -1 | sed 's/^# //')
        printf "  %-40s %s\n" "$name" "$desc"
    done
    exit 1
fi

EXPERIMENT_CONFIG="$1"
if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    EXPERIMENT_CONFIG="$SCRIPT_DIR/cambrian_experiments/$1"
fi
if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: 找不到实验配置: $1"
    exit 1
fi

echo "Loading experiment config: $EXPERIMENT_CONFIG"
source "$EXPERIMENT_CONFIG"

# ========================= VALIDATE =========================
if [ "$CAMBRIAN_MODEL_PATH" = "/path/to/cambrian-s-weights" ]; then
    echo "ERROR: Set CAMBRIAN_MODEL_PATH environment variable"
    echo "  e.g.: export CAMBRIAN_MODEL_PATH=/scratch/by2593/hf_cache/cambrian-s-7b"
    exit 1
fi

# ========================= ENVIRONMENT SETUP =========================
GPU_LIST=$(seq -s, 0 $((NUM_TRAIN_GPUS - 1)))
export CUDA_VISIBLE_DEVICES="${GPU_LIST},${RENDERING_GPU}"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager
export RAY_DEDUP_LOGS=0
export RAY_enable_metrics_collection=false
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export GS_RENDERER_VERBOSE=0
export ACTIVE_SPATIAL_ENV_VERBOSE=0
export RENDERING_GPU_ID=${RENDERING_GPU}
export PATH="/scratch/by2593/miniconda3/envs/vagen/bin:$PATH"
export CAMBRIAN_SRC="${CAMBRIAN_SRC}"

# ========================= PRINT CONFIG =========================
echo "=============================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Config:     $(basename $EXPERIMENT_CONFIG)"
echo "Model:      Cambrian-S ($CAMBRIAN_MODEL_PATH)"
echo "Env:        $ENV_CONFIG"
echo "Train GPUs: $GPU_LIST ($NUM_TRAIN_GPUS GPUs)"
echo "Render GPU: $RENDERING_GPU"
echo "Rollout:    HF (not vLLM)"
echo "----------------------------------------------"
echo "Actor LR=$ACTOR_LR  Entropy=$ENTROPY_COEFF  Grad Clip=$GRAD_CLIP"
echo "Critic LR=$CRITIC_LR  Warmup=$CRITIC_WARMUP  Grad Clip=$CRITIC_GRAD_CLIP"
echo "Temp=$TEMPERATURE  Top-p=$TOP_P"
echo "KL_loss=$USE_KL_LOSS  KL_coef=$KL_LOSS_COEF  KL_ctrl=$KL_COEF"
echo "Batch=$TRAIN_BATCH_SIZE  Mini=$MINI_BATCH_SIZE  MaxTurns=$MAX_TURNS"
echo "Save freq=$SAVE_FREQ  Total steps=$TOTAL_STEPS"
echo "ADV=$ADV_ESTIMATOR  HL_gamma=$HIGH_LEVEL_GAMMA"
echo "Resume=$RESUME_MODE"
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

if [ "$USE_GPU_HOLDER" = true ] && [ -f "$SCRIPT_DIR/gpu_holder.py" ]; then
    # 渲染卡 holder
    HOLDER_GPU=$RENDERING_GPU HOLDER_MEM_FRAC=0.75 HOLDER_TARGET=$RENDER_GPU_HOLDER_TARGET \
        $PYTHON "$SCRIPT_DIR/gpu_holder.py" \
        > "$HOLDER_LOG_DIR/holder_gpu${RENDERING_GPU}.log" 2>&1 &
    HOLDER_PIDS+=($!)
    echo "GPU Holder: GPU $RENDERING_GPU target=${RENDER_GPU_HOLDER_TARGET}% PID=$!"

    # 训练卡 holders
    for GPU_ID in $(seq 0 $((NUM_TRAIN_GPUS - 1))); do
        HOLDER_GPU=$GPU_ID HOLDER_MEM_FRAC=0.0 HOLDER_TARGET=$TRAIN_GPU_HOLDER_TARGET \
            $PYTHON "$SCRIPT_DIR/gpu_holder.py" \
            > "$HOLDER_LOG_DIR/holder_gpu${GPU_ID}.log" 2>&1 &
        HOLDER_PIDS+=($!)
    done
    echo "GPU Holders started for training GPUs 0-$((NUM_TRAIN_GPUS-1)) target=${TRAIN_GPU_HOLDER_TARGET}%"

    sleep 3
    echo "--- GPU Holder Health Check ---"
    for pid in "${HOLDER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  PID $pid: ALIVE"
        else
            echo "  PID $pid: DEAD! Check $HOLDER_LOG_DIR/"
        fi
    done
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
    actor_rollout_ref.model.path=$CAMBRIAN_MODEL_PATH \
    actor_rollout_ref.model.external_lib=vagen.models.cambrian_register \
    +actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.response_length=$RESPONSE_LENGTH \
    +actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.optim.lr=$CRITIC_LR \
    critic.model.use_remove_padding=False \
    critic.model.path=$CAMBRIAN_MODEL_PATH \
    critic.model.external_lib=vagen.models.cambrian_register \
    +critic.model.trust_remote_code=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=$CRITIC_OPTIMIZER_OFFLOAD \
    critic.cliprange_value=$CLIPRANGE_VALUE \
    critic.grad_clip=$CRITIC_GRAD_CLIP \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.critic_warmup=$CRITIC_WARMUP \
    trainer.logger=['console','wandb'] \
    trainer.project_name='cambrian_active_spatial' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_TRAIN_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_training_steps=$TOTAL_STEPS \
    rollout_manager.max_turns=$MAX_TURNS \
    rollout_manager.window_size=$WINDOW_SIZE \
    +rollout_manager.max_prompt_length=$RM_MAX_PROMPT_LENGTH \
    ++rollout_manager.max_trajectory_length=$MAX_TRAJECTORY_LENGTH \
    rollout_manager.use_multi_turn_reward=True \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    +rollout_manager.rollout_type=cambrian \
    +rollout_manager.si_token_len=$SI_TOKEN_LEN \
    +rollout_manager.mm_use_im_newline_token=True \
    +rollout_manager.image_aspect_ratio=anyres \
    +rollout_manager.anyres_max_subimages=$ANYRES_MAX_SUBIMAGES \
    trainer.resume_mode=$RESUME_MODE \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.n_trajectory=1 \
    rollout_manager.use_service=False \
    +rollout_manager.mini_batch_size=$MINI_BATCH_SIZE \
    2>&1 | tee "${EXPERIMENT_NAME}.log"
