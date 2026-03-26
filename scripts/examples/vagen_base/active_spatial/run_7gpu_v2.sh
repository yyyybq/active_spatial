#!/bin/bash
set -x

# =============================================================================
# 7-GPU Training with V2 Dataset (6 task types)
# =============================================================================
# - 6 GPUs (0-5) for training, 1 GPU (6) for rendering
# - V2 dataset: no centering, fov_inclusion, screen_occupancy
# - Batch sizes adjusted to be divisible by 6
# =============================================================================

# Set CUDA devices - 6 GPUs for training (0-5), 1 GPU for rendering (6)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager

# Ray configuration
export RAY_DEDUP_LOGS=0
export RAY_enable_metrics_collection=false
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Disable verbose logging
export GS_RENDERER_VERBOSE=0
export ACTIVE_SPATIAL_ENV_VERBOSE=0

# Reserve GPU 6 for rendering
export RENDERING_GPU_ID=6

# Ensure ninja is in PATH for Ray workers
export PATH="/scratch/by2593/miniconda3/envs/vagen/bin:$PATH"

PYTHON=/scratch/by2593/miniconda3/envs/vagen/bin/python
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_NAME="active_spatial_ppo_7gpu_v2_7B_0321"

# 清理函数
cleanup() {
    echo "Cleaning up..."
    kill $GPU_HOLDER_PID 2>/dev/null || true
    for i in 0 1 2 3 4 5; do
        eval "kill \$GPU_WARMER_PID_${i} 2>/dev/null || true"
    done
    pkill -P $$ -f "gpu_holder.py" 2>/dev/null || true
    exit 0
}
trap cleanup EXIT INT TERM

echo "=============================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Training GPUs: 0,1,2,3,4,5 (6 GPUs)"
echo "Rendering GPU: 6"
echo "Dataset: V2 (6 task types, 278 tasks)"
echo "Strategy:"
echo "  - GPU 6: 动态补齐到 75% (与渲染共存)"
echo "  - GPU 0-5: 训练间隙填充"
echo "=============================================="

# Create training dataset with V2 env config
$PYTHON -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/env_config_v2.yaml" \
    --train_path "data/$EXPERIMENT_NAME/train.parquet" \
    --test_path "data/$EXPERIMENT_NAME/test.parquet"

# GPU 6 (渲染卡): 占空比 + SM满载控制（与渲染共存）
HOLDER_GPU=6 HOLDER_MEM_FRAC=0.75 HOLDER_TARGET=75 \
    $PYTHON "$SCRIPT_DIR/gpu_holder.py" &
GPU_HOLDER_PID=$!
echo "GPU Holder started for GPU 6, target=75%, PID=$GPU_HOLDER_PID"

# GPU 0-5 (训练卡): 训练间隙自动补齐到 75%
for GPU_ID in 0 1 2 3 4 5; do
    HOLDER_GPU=$GPU_ID HOLDER_MEM_FRAC=0.0 HOLDER_TARGET=75 \
        $PYTHON "$SCRIPT_DIR/gpu_holder.py" &
    eval "GPU_WARMER_PID_${GPU_ID}=$!"
    echo "GPU Holder started for GPU $GPU_ID, target=75%, PID=$!"
done

# Run PPO training - 6 GPUs, batch sizes divisible by 6
$PYTHON -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=masked_gae \
    algorithm.high_level_gamma=0.95 \
    data.train_files=data/$EXPERIMENT_NAME/train.parquet \
    data.val_files=data/$EXPERIMENT_NAME/test.parquet \
    data.train_batch_size=30 \
    data.val_batch_size=6 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.max_trajectory_length=14000 \
    data.image_key=images \
    data.truncation=left \
    +data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
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
    critic.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='vagen_active_spatial' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=50 \
    trainer.total_training_steps=2000 \
    rollout_manager.max_turns=12 \
    rollout_manager.window_size=5 \
    rollout_manager.use_multi_turn_reward=True \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    trainer.val_before_train=False \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.n_trajectory=1 \
    rollout_manager.use_service=False \
    +rollout_manager.mini_batch_size=6 \
    2>&1 | tee $EXPERIMENT_NAME.log
