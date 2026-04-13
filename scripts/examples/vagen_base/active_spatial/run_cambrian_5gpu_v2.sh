#!/bin/bash
set -x

# =============================================================================
# Cambrian-S 5-GPU Training with Active Spatial V2 Dataset
# =============================================================================
# - 4 GPUs (0-3) for training, 1 GPU (4) for rendering
# - Uses HF rollout (not vLLM) since Cambrian-S is not supported by vLLM
# =============================================================================

# Set CUDA devices - 5 GPUs total
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

export PYTHONHASHSEED=0
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ray configuration
export RAY_DEDUP_LOGS=0
export RAY_enable_metrics_collection=false
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Disable verbose logging
export GS_RENDERER_VERBOSE=0
export ACTIVE_SPATIAL_ENV_VERBOSE=0

# Reserve GPU 4 for rendering
export RENDERING_GPU_ID=4

# Ensure ninja is in PATH for Ray workers
export PATH="/scratch/by2593/miniconda3/envs/vagen/bin:$PATH"

# Cambrian-S source tree (needed by cambrian_register.py on every Ray worker)
export CAMBRIAN_SRC="${CAMBRIAN_SRC:-/scratch/by2593/project/Active_Spatial/cambrian-s}"

PYTHON=/scratch/by2593/miniconda3/envs/vagen/bin/python
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_NAME="cambrian_s_active_spatial_ppo_5gpu_v2"

# === CONFIGURE MODEL PATH ===
CAMBRIAN_MODEL_PATH="${CAMBRIAN_MODEL_PATH:-/path/to/cambrian-s-weights}"
if [ "$CAMBRIAN_MODEL_PATH" = "/path/to/cambrian-s-weights" ]; then
    echo "ERROR: Set CAMBRIAN_MODEL_PATH environment variable to your Cambrian-S weights path"
    echo "  e.g.: export CAMBRIAN_MODEL_PATH=/scratch/by2593/hf_cache/cambrian-s-7b"
    exit 1
fi

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    kill $GPU_HOLDER_PID 2>/dev/null || true
    for i in 0 1 2 3; do
        eval "kill \$GPU_WARMER_PID_${i} 2>/dev/null || true"
    done
    pkill -P $$ -f "gpu_holder.py" 2>/dev/null || true
    exit 0
}
trap cleanup EXIT INT TERM

echo "=============================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Model: Cambrian-S ($CAMBRIAN_MODEL_PATH)"
echo "Training GPUs: 0,1,2,3 (4 GPUs)"
echo "Rendering GPU: 4"
echo "Rollout: HF (not vLLM)"
echo "=============================================="

# Create training dataset with V2 env config
$PYTHON -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/env_config_v2.yaml" \
    --train_path "data/$EXPERIMENT_NAME/train.parquet" \
    --test_path "data/$EXPERIMENT_NAME/test.parquet"

# GPU 4 (rendering): occupy up to 90%
if [ -f "$SCRIPT_DIR/gpu_holder.py" ]; then
    HOLDER_GPU=4 HOLDER_MEM_FRAC=0.75 HOLDER_TARGET=90 \
        $PYTHON "$SCRIPT_DIR/gpu_holder.py" &
    GPU_HOLDER_PID=$!
    echo "GPU Holder started for GPU 4, target=90%, PID=$GPU_HOLDER_PID"

    # GPU 0-3 (training): fill to 75% between training steps
    for GPU_ID in 0 1 2 3; do
        HOLDER_GPU=$GPU_ID HOLDER_MEM_FRAC=0.0 HOLDER_TARGET=75 \
            $PYTHON "$SCRIPT_DIR/gpu_holder.py" &
        eval "GPU_WARMER_PID_${GPU_ID}=$!"
    done
fi

# Run PPO training with Cambrian-S (4 training GPUs)
$PYTHON -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=masked_gae \
    algorithm.high_level_gamma=0.95 \
    data.train_files=data/$EXPERIMENT_NAME/train.parquet \
    data.val_files=data/$EXPERIMENT_NAME/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.max_trajectory_length=18000 \
    data.image_key=images \
    data.truncation=left \
    +data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=$CAMBRIAN_MODEL_PATH \
    actor_rollout_ref.model.external_lib=vagen.models.cambrian_register \
    +actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.response_length=512 \
    +actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.model.path=$CAMBRIAN_MODEL_PATH \
    critic.model.external_lib=vagen.models.cambrian_register \
    +critic.model.trust_remote_code=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='cambrian_active_spatial' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.total_training_steps=2000 \
    rollout_manager.max_turns=12 \
    rollout_manager.window_size=5 \
    +rollout_manager.max_prompt_length=8192 \
    ++rollout_manager.max_trajectory_length=18000 \
    rollout_manager.use_multi_turn_reward=True \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    +rollout_manager.rollout_type=cambrian \
    +rollout_manager.si_token_len=729 \
    +rollout_manager.mm_use_im_newline_token=True \
    +rollout_manager.image_aspect_ratio=anyres \
    +rollout_manager.anyres_max_subimages=9 \
    trainer.val_before_train=False \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.n_trajectory=1 \
    rollout_manager.use_service=False \
    +rollout_manager.mini_batch_size=4 \
    2>&1 | tee $EXPERIMENT_NAME.log
