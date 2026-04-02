# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from vagen.trainer.ppo.ray_trainer import RayPPOTrainer
from vagen.utils.compute_score import compute_score

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config, compute_score=compute_score)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1, num_gpus=0)  # num_gpus=0 with RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 allows GPU access for local rendering
def main_task(config, compute_score=None):
    from verl.utils.fs import copy_to_local
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from vagen.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    # use_ref = (config.algorithm.adv_estimator not in  [AdvantageEstimator.GAE, AdvantageEstimator.MULTI_TURN_GAE] and \
    #     config.actor_rollout_ref.ref.get('use_ref', True))
    use_ref = config.actor_rollout_ref.ref.get('use_ref', True)
    print(f"[DEBUG] use_ref={use_ref}")
    if use_ref:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
    else:
        config.actor_rollout_ref.actor.use_kl_loss = False
        print("[WARNING] Ref policy is disabled, use_kl_loss is set to False")

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    if use_ref:
        mapping[Role.RefPolicy] = global_pool_id

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
    else:
        raise NotImplementedError
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)

    # Note that we always use function-based RM for validation
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            processor=processor,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    
    # For Cambrian-S: load image processor from vision tower config
    # This is needed by CambrianRolloutManager on the main process
    rollout_type = config.rollout_manager.get("rollout_type", "qwen")
    if rollout_type == "cambrian":
        from transformers import AutoConfig as _AC
        from omegaconf import OmegaConf, open_dict

        # Load model config to extract vision tower metadata
        model_config = _AC.from_pretrained(local_path, trust_remote_code=True)

        # -----------------------------------------------------------------------
        # Fix 1: tokens_per_image – read ground truth from the checkpoint config.
        # The launch script sets +rollout_manager.si_token_len=729, but if it is
        # left unset we fall back to the value baked into the model config file.
        # We write into rollout_manager so CambrianRolloutManager picks it up.
        # -----------------------------------------------------------------------
        si_token_len_ckpt = getattr(model_config, "si_token_len", 729)
        mm_use_newline_ckpt = getattr(model_config, "mm_use_im_newline_token", True)
        with open_dict(config):
            # Only override if the user has not set an explicit value
            if config.rollout_manager.get("si_token_len", None) is None:
                config.rollout_manager.si_token_len = si_token_len_ckpt
                print(
                    f"[CambrianSetup] si_token_len auto-set from checkpoint: {si_token_len_ckpt}"
                )
            else:
                user_val = config.rollout_manager.si_token_len
                if user_val != si_token_len_ckpt:
                    print(
                        f"[CambrianSetup] WARNING: si_token_len in launch script ({user_val}) "
                        f"differs from checkpoint config ({si_token_len_ckpt}). "
                        "Using checkpoint value to prevent embedding misalignment."
                    )
                    config.rollout_manager.si_token_len = si_token_len_ckpt

            if config.rollout_manager.get("mm_use_im_newline_token", None) is None:
                config.rollout_manager.mm_use_im_newline_token = mm_use_newline_ckpt
            else:
                user_val = config.rollout_manager.mm_use_im_newline_token
                if user_val != mm_use_newline_ckpt:
                    print(
                        f"[CambrianSetup] WARNING: mm_use_im_newline_token mismatch "
                        f"(script={user_val}, ckpt={mm_use_newline_ckpt}). "
                        "Using checkpoint value."
                    )
                    config.rollout_manager.mm_use_im_newline_token = mm_use_newline_ckpt

        _si = config.rollout_manager.si_token_len
        _nl = config.rollout_manager.mm_use_im_newline_token
        _side = int(_si ** 0.5)
        _tpi = _side * (_side + 1) if _nl else _si
        print(
            f"[CambrianSetup] tokens_per_image = {_tpi} "
            f"(si_token_len={_si}, si_side_len={_side}, newlines={_nl})"
        )

        # -----------------------------------------------------------------------
        # Fix 2: <image> token embedding.
        # If the checkpoint was trained with Cambrian-S (it should already have
        # <image> in its vocab), tokenizer.vocab_size == model.config.vocab_size.
        # If using a raw Qwen2 tokenizer that doesn't know <image>, the mismatch
        # is detected here so the user can re-save the checkpoint after resizing.
        # We cannot resize inside the Ray FSDP workers without a full checkpoint
        # save/reload cycle, so we raise early rather than silently corrupt.
        # -----------------------------------------------------------------------
        ckpt_vocab_size = getattr(model_config, "vocab_size", None)
        tok_vocab_size = len(tokenizer)
        if ckpt_vocab_size is not None and ckpt_vocab_size != tok_vocab_size:
            raise RuntimeError(
                f"[CambrianSetup] Vocabulary size mismatch: "
                f"tokenizer has {tok_vocab_size} tokens but checkpoint config says {ckpt_vocab_size}. "
                "This usually means <image> was added to the tokenizer after the checkpoint was saved. "
                "Fix: load the Cambrian-S weights, call model.resize_token_embeddings(len(tokenizer)), "
                "then re-save the checkpoint before launching RL training."
            )
        print(f"[CambrianSetup] vocab size check passed: {tok_vocab_size} tokens.")

        # -----------------------------------------------------------------------
        # Fix 3: Load SigLIP image processor(s) for the rollout manager.
        # -----------------------------------------------------------------------
        vision_tower_names = getattr(model_config, "vision_tower_aux_list", [])
        image_processor_list = []
        if vision_tower_names:
            from transformers import AutoImageProcessor
            for vt_name in vision_tower_names:
                try:
                    ip = AutoImageProcessor.from_pretrained(vt_name)
                    image_processor_list.append(ip)
                    print(f"[CambrianSetup] Loaded image processor from {vt_name}")
                except Exception as e:
                    print(
                        f"[CambrianSetup] WARNING: Failed to load image processor "
                        f"from {vt_name}: {e}"
                    )
        else:
            print("[CambrianSetup] WARNING: No vision_tower_aux_list in model config")
        trainer.image_processor = image_processor_list if image_processor_list else None
        print(
            f"[CambrianSetup] {len(image_processor_list)} image processor(s) set on trainer"
        )
    
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
