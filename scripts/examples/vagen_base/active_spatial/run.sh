#!/bin/bash
set -x

# Set CUDA devices - 4 GPUs for training (0-3), 1 GPU for rendering (4)
# Training uses GPUs 0-3, rendering uses GPU 4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0
# Disable flash attention to avoid ABI compatibility issues
# export DISABLE_FLASH_ATTN=1  # 逆天,不知道是不是L40的原因,flash attention会crash
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager

# Ray configuration to fix metrics agent issue
# export RAY_DEDUP_LOGS=1 
export RAY_DEDUP_LOGS=0
export RAY_enable_metrics_collection=false

# Ray 2.x: Do NOT override CUDA_VISIBLE_DEVICES for num_gpus=0 actors.
# This allows main_task (num_gpus=0) to see GPUs for rendering,
# while GPU workers still get proper per-rank GPU assignment.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Disable verbose logging for renderers (set to 1 to enable)
export GS_RENDERER_VERBOSE=0
export ACTIVE_SPATIAL_ENV_VERBOSE=0

# Reserve GPU 4 for rendering - don't let Ray use it for training
export RENDERING_GPU_ID=4

# Use vagen environment Python directly
PYTHON=/scratch/by2593/miniconda3/envs/vagen/bin/python

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Experiment name - clear and explicit definition
EXPERIMENT_NAME="active_spatial_ppo"

echo "Experiment name: $EXPERIMENT_NAME"
conda init
conda activate vagen
# Make sure to run the rendering service first:
# Start the ViewSuite rendering server before training
# python -m view_suite.service.server --port 8766

# Create training dataset (skip if already exists)
$PYTHON -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/env_config_balanced.yaml" \
    --train_path "data/$EXPERIMENT_NAME/train.parquet" \
    --test_path "data/$EXPERIMENT_NAME/test.parquet"

# Start GPU warmer on rendering GPU to keep utilization >= 60%
WARMER_GPU=4 WARMER_SIZE=3072 WARMER_SLEEP_MS=5 \
    $PYTHON "$SCRIPT_DIR/gpu_warmer.py" &
GPU_WARMER_PID=$!
echo "GPU warmer started on GPU 4, PID=$GPU_WARMER_PID"
# Ensure warmer is killed when script exits
trap "kill $GPU_WARMER_PID 2>/dev/null" EXIT

# Run PPO training
$PYTHON -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=masked_gae \
    algorithm.high_level_gamma=0.95 \
    data.train_files=data/$EXPERIMENT_NAME/train.parquet \
    data.val_files=data/$EXPERIMENT_NAME/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.max_trajectory_length=14000 \
    data.image_key=images \
    data.truncation=left \
    +data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=14000 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.limit_mm_per_prompt=15 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.temperature=0.7 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='vagen_active_spatial' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=50 \
    trainer.total_training_steps=2000 \
    rollout_manager.max_turns=10 \
    rollout_manager.window_size=5 \
    rollout_manager.use_multi_turn_reward=True \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    trainer.val_before_train=False \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.n_trajectory=1 \
    rollout_manager.use_service=False \
    +rollout_manager.mini_batch_size=8 \
    2>&1 | tee $EXPERIMENT_NAME.log
