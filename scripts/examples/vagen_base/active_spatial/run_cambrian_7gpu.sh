#!/bin/bash
set -x

# =============================================================================
# Cambrian-S PPO Training with Active Spatial Environment
# =============================================================================
# Uses HF-based rollout (not vLLM) since Cambrian-S is not natively supported by vLLM.
# 
# Key differences from Qwen VL training:
#   - rollout.name=hf (HF generate instead of vLLM)
#   - external_lib=vagen.models.cambrian_register (registers Cambrian with AutoModelForCausalLM)
#   - use_remove_padding=False (Cambrian not in rmpad registry)
#   - rollout_type=cambrian (uses CambrianRolloutManager)
#   - trust_remote_code=True
#   - No tensor_model_parallel_size for HF rollout
#   - micro_batch_size=1 for rollout (Cambrian generation supports bs=1)
# =============================================================================

# ===== Configuration =====
# Set this to your Cambrian-S model checkpoint path
CAMBRIAN_MODEL_PATH="${CAMBRIAN_MODEL_PATH:-/path/to/cambrian-s/checkpoint}"
# Critic model (can use Qwen or another model)
CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"

# GPU setup: 6 GPUs for training (0-5), 1 GPU (6) for rendering
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# HF Rollout doesn't need XFORMERS but set for compatibility
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
EXPERIMENT_NAME="cambrian_s_ppo_active_spatial_7gpu"

# Cleanup function
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
echo "Model: $CAMBRIAN_MODEL_PATH"
echo "Critic: $CRITIC_MODEL_PATH"
echo "Training GPUs: 0,1,2,3,4,5 (6 GPUs)"
echo "Rendering GPU: 6"
echo "Rollout: HF (not vLLM)"
echo "=============================================="

# Create training dataset with V2 env config
$PYTHON -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/env_config_v2.yaml" \
    --train_path "data/$EXPERIMENT_NAME/train.parquet" \
    --test_path "data/$EXPERIMENT_NAME/test.parquet"

# GPU 6 (rendering GPU): holder
HOLDER_GPU=6 HOLDER_MEM_FRAC=0.75 HOLDER_TARGET=75 \
    $PYTHON "$SCRIPT_DIR/gpu_holder.py" &
GPU_HOLDER_PID=$!
echo "GPU Holder started for GPU 6, target=75%, PID=$GPU_HOLDER_PID"

# GPU 0-5 (training GPUs): warmers
for GPU_ID in 0 1 2 3 4 5; do
    HOLDER_GPU=$GPU_ID HOLDER_MEM_FRAC=0.0 HOLDER_TARGET=75 \
        $PYTHON "$SCRIPT_DIR/gpu_holder.py" &
    eval "GPU_WARMER_PID_${GPU_ID}=$!"
    echo "GPU Holder started for GPU $GPU_ID, target=75%, PID=$!"
done

# Run PPO training - Cambrian-S with HF rollout
$PYTHON -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=masked_gae \
    algorithm.high_level_gamma=0.95 \
    data.train_files=data/$EXPERIMENT_NAME/train.parquet \
    data.val_files=data/$EXPERIMENT_NAME/test.parquet \
    data.train_batch_size=24 \
    data.val_batch_size=6 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.max_trajectory_length=14000 \
    data.image_key=images \
    data.truncation=left \
    +data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=$CAMBRIAN_MODEL_PATH \
    actor_rollout_ref.model.external_lib=vagen.models.cambrian_register \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$CRITIC_MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='vagen_active_spatial_cambrian' \
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
    +rollout_manager.rollout_type=cambrian \
    +rollout_manager.si_token_len=729 \
    +rollout_manager.mm_use_im_newline_token=True \
    trainer.val_before_train=False \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.n_trajectory=1 \
    rollout_manager.use_service=False \
    +rollout_manager.mini_batch_size=6 \
    2>&1 | tee $EXPERIMENT_NAME.log
