"""
Cambrian-S Rollout Manager for VAGEN PPO Training.

This replaces the QwenVLRolloutManager with Cambrian-S-specific multimodal handling:
- Uses Cambrian-S's SigLIP vision encoder + MLP projector for image processing
- Uses IMAGE_TOKEN_INDEX=-200 as image placeholder in input_ids
- Uses standard 1D RoPE position IDs (not Qwen2-VL's 3D mRoPE)
- Uses HF-based generation since vLLM does not natively support Cambrian-S architecture
- Pre-computes inputs_embeds for the FSDP training forward pass
"""

from typing import List, Union, Optional, Dict, Tuple
import copy
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedTokenizer, ProcessorMixin
from dataclasses import dataclass, field
import PIL
import re

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import process_image, collate_fn
import vagen.env
from vagen.env import REGISTERED_ENV


# Cambrian-S constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


class CambrianRolloutManager():
    """
    Rollout manager adapted for Cambrian-S model architecture.
    
    Key differences from QwenVLRolloutManager:
    1. Image processing uses Cambrian-S's SigLIP vision encoder (frozen) + MLP projector
    2. No 3D mRoPE — uses standard sequential position IDs
    3. Image tokens are represented as IMAGE_TOKEN_INDEX=-200 in input_ids
    4. For vLLM generation: Uses HF generate() since vLLM doesn't support custom Cambrian arch
    5. For FSDP forward: passes pixel_values through as multi_modal_inputs for the model to encode
    """

    def __init__(self,
                 actor_rollout_wg,
                 config,
                 tokenizer: PreTrainedTokenizer,
                 processor=None,  # Not used for Cambrian-S (uses its own image_processor)
                 image_processor=None,  # Cambrian-S's SigLIP image processor
                 ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_processor = image_processor  # list of image processors from vision towers
        self.config = config
        self.actor_rollout_wg = actor_rollout_wg
        self.recorder = None
        self.envs = None
        self.env_states = None
        self.batch_idx_to_env_id = None

        # Image token length from cambrian config
        self.si_token_len = getattr(config, 'si_token_len', 729)  # 27x27 for SigLIP
        self.si_side_len = int(self.si_token_len ** 0.5) if self.si_token_len > 0 else None

        # Calculate tokens_per_image including newline tokens
        mm_use_newline = getattr(config, 'mm_use_im_newline_token', True)
        if self.si_side_len is not None:
            if mm_use_newline:
                self.tokens_per_image = self.si_side_len * (self.si_side_len + 1)  # 27*28=756
            else:
                self.tokens_per_image = self.si_token_len  # 729
        else:
            self.tokens_per_image = self.si_token_len
        print(f"[CambrianRolloutManager] tokens_per_image={self.tokens_per_image} "
              f"(si_token_len={self.si_token_len}, si_side_len={self.si_side_len}, "
              f"newlines={'yes' if mm_use_newline else 'no'})")

        # Ensure <image> is a special token so tokenization is predictable
        if DEFAULT_IMAGE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
            print(f"[CambrianRolloutManager] Added '{DEFAULT_IMAGE_TOKEN}' as special token")
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        print(f"[CambrianRolloutManager] image_token_id={self.image_token_id}")

        # Defensive: ensure all config keys are accessed with getattr or .get()
        self.n_gpus_per_node = getattr(config, 'n_gpus_per_node', 1)
        self.max_prompt_length = getattr(config, 'max_prompt_length', 2048)
        self.max_trajectory_length = getattr(config, 'max_trajectory_length', 2048)
        self.truncation = getattr(config, 'truncation', 'left')
        self.window_size = getattr(config, 'window_size', 1)
        self.max_turns = getattr(config, 'max_turns', 1)
        self.use_multi_turn_reward = getattr(config, 'use_multi_turn_reward', False)
        self.use_loss_mask = getattr(config, 'use_loss_mask', False)
        self.use_gae_mask = getattr(config, 'use_gae_mask', False)
        self.special_token_for_loss_mask = getattr(config, 'special_token_for_loss_mask', ['<|box_start|>', '<|box_end|>'])

    @torch.no_grad()
    def _handle_special_tokens(self, llm_raw_response: str, prep_for_loss_mask: bool) -> str:
        """
        1. Filter out special tokens: <image> and special tokens marking environment observation
        2. prep_for_loss_mask: if true, add special tokens to the beginning and end of the response
        """
        llm_raw_response = llm_raw_response.replace('<image>', '')
        if prep_for_loss_mask:
            sptk_b = self.special_token_for_loss_mask[0]
            sptk_e = self.special_token_for_loss_mask[1]
            llm_raw_response = llm_raw_response.replace(sptk_b, '')
            llm_raw_response = llm_raw_response.replace(sptk_e, '')
            llm_raw_response = sptk_b + llm_raw_response + sptk_e
        return llm_raw_response

    @torch.no_grad()
    def _handle_multi_modal_data(
            self,
            prompt_template: str,
            row_dict: Dict,
            image_data: List[PIL.Image.Image],
            do_embedding: bool = True,
    ) -> Tuple[str, Dict, Optional[torch.Tensor], str]:
        """Handle multi-modal data for Cambrian-S.

        For Cambrian-S, the <image> placeholder remains in the text.
        After tokenization, we expand each <image> token to tokens_per_image IMAGE_TOKEN_INDEX.
        
        Both do_embedding=True (training) and do_embedding=False (generation) need preprocessed
        pixel_values since we use HF-based generation (not vLLM).
        """
        assert len(image_data) == prompt_template.count('<image>'), \
            f'Number of images ({len(image_data)}) does not match number of <image> ({prompt_template.count("<image>")})'

        raw_prompt = prompt_template  # Keep <image> as-is

        # Store images for multimodal processing
        row_dict['multi_modal_data'] = {'image': image_data}

        # Always preprocess images (needed for both HF generate and FSDP training)
        if self.image_processor is not None and len(image_data) > 0:
            # Use the first (only) vision tower's image processor for connector_only
            ip = self.image_processor[0]
            processed = ip.preprocess(image_data, return_tensors='pt')
            pixel_values = processed['pixel_values']  # (num_images, C, H, W)
            row_dict['multi_modal_inputs'] = {
                'pixel_values': pixel_values,
            }

        return prompt_template, row_dict, None, raw_prompt

    @torch.no_grad()
    def _compute_loss_mask(self, input_ids, attention_mask):
        """
        Compute loss mask for the input ids and attention mask.
        Tokens wrapped by special tokens (default: <|box_start|> and <|box_end|>) are trained.
        
        Strategy: Replace special tokens with pad tokens and shift sequence left.
        """
        sptk_b = self.tokenizer.convert_tokens_to_ids(self.special_token_for_loss_mask[0])
        sptk_e = self.tokenizer.convert_tokens_to_ids(self.special_token_for_loss_mask[1])
        pad_token_id = self.tokenizer.pad_token_id

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        new_input_ids = input_ids.clone()
        new_attention_mask = attention_mask.clone()
        loss_mask = torch.zeros_like(new_attention_mask)
        new_loss_mask = torch.zeros_like(new_attention_mask)
        end_of_response_position_mask = torch.zeros_like(new_attention_mask)
        new_end_of_response_position_mask = torch.zeros_like(new_attention_mask)

        for b in range(batch_size):
            right_pad_tokens = (new_input_ids[b] == pad_token_id).sum().item()
            
            if right_pad_tokens > 0 and not torch.all(attention_mask[b, -right_pad_tokens:] == 0):
                print("[DEBUG]: right padding tokens must have attention mask of 0")

            sptk_b_indices = (input_ids[b] == sptk_b).nonzero().flatten()
            sptk_e_indices = (input_ids[b] == sptk_e).nonzero().flatten()

            hole_pos = []
            for start_pos, end_pos in zip(sptk_b_indices, sptk_e_indices):
                loss_mask[b][start_pos + 1:end_pos] = 1
                end_of_response_position_mask[b][end_pos - 1] = 1
                hole_pos.append(start_pos.item())
                hole_pos.append(end_pos.item())
            hole_pos.append(seq_len - right_pad_tokens)

            if right_pad_tokens > 0 and not torch.all(new_input_ids[b][seq_len - right_pad_tokens:] == pad_token_id):
                print("[DEBUG]: right padding tokens must be pad token")

            holes_to_fill = 1
            for i in range(0, len(hole_pos) - 1):
                start_pos = hole_pos[i]
                end_pos = hole_pos[i + 1]
                new_loss_mask[b, start_pos + 1 - holes_to_fill:end_pos - holes_to_fill] = loss_mask[b, start_pos + 1:end_pos]
                new_end_of_response_position_mask[b, start_pos + 1 - holes_to_fill:end_pos - holes_to_fill] = end_of_response_position_mask[b, start_pos + 1:end_pos]
                new_input_ids[b, start_pos + 1 - holes_to_fill:end_pos - holes_to_fill] = input_ids[b, start_pos + 1:end_pos]
                new_attention_mask[b, start_pos + 1 - holes_to_fill:end_pos - holes_to_fill] = attention_mask[b, start_pos + 1:end_pos]
                holes_to_fill += 1

            valid_tokens = seq_len - right_pad_tokens - len(hole_pos) + 1
            new_loss_mask[b][valid_tokens:] = 0
            new_input_ids[b][valid_tokens:] = pad_token_id
            new_attention_mask[b][valid_tokens:] = 0

        return new_input_ids, new_attention_mask, new_loss_mask, new_end_of_response_position_mask

    @torch.no_grad()
    def reset(self, env_configs):
        """Reset environments based on provided configurations."""
        env_buckets = defaultdict(set)
        new_envs = {}

        if self.envs is None:
            self.envs = {}

        for env_id, env in self.envs.items():
            env_config_id = env.config.config_id()
            bucket_key = env_config_id
            env_buckets[bucket_key].add(env_id)

        for i, cfg in enumerate(env_configs):
            env_id = i
            env_name = cfg["env_name"]
            env_config = cfg["env_config"]
            seed = cfg["seed"]

            config_instance = REGISTERED_ENV[env_name]["config_cls"](**env_config)
            env_config_id = config_instance.config_id()
            bucket_key = env_config_id

            if bucket_key in env_buckets and env_buckets[bucket_key]:
                old_env_id = env_buckets[bucket_key].pop()
                new_envs[env_id] = {
                    "env_instance": self.envs[old_env_id],
                    "seed": seed,
                }
            else:
                new_envs[env_id] = {
                    "env_cls": REGISTERED_ENV[env_name]["env_cls"],
                    "seed": seed,
                    "config_instance": config_instance,
                }

        for bucket_key, env_ids in env_buckets.items():
            for env_id in env_ids:
                self.envs[env_id].close()
                del self.envs[env_id]

        if self.recorder is not None:
            del self.recorder
        self.recorder = defaultdict(list)
        initial_obs = {}
        initial_info = {}
        for env_id, env_info in new_envs.items():
            if "env_instance" in env_info:
                self.envs[env_id] = env_info["env_instance"]
            else:
                assert "env_cls" in env_info
                self.envs[env_id] = env_info["env_cls"](env_info["config_instance"])
            obs, info = self.envs[env_id].reset(env_info["seed"])
            initial_obs[env_id] = obs
            initial_info[env_id] = info
            self.record(env_id, obs=obs, reward=0, done=False, info=info)

        self.env_states = {
            env_id: {'step': 0, 'done': False, 'metrics': {"turn_metrics": defaultdict(list), "traj_metrics": {}}}
            for env_id in self.envs
        }
        return initial_obs, initial_info

    @torch.no_grad()
    def record(self, env_id, obs, reward, done, info):
        """Record each step's obs, info, done, reward."""
        assert obs is not None
        assert info is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        record_entry = {
            'env_id': env_id,
            'done': done,
            'reward': reward,
            'info': info,
            'obs_str': obs['obs_str'],
        }
        image_placeholder = self.envs[env_id].config.get('image_placeholder', "<image>")
        if 'multi_modal_data' in obs:
            if image_placeholder in obs['multi_modal_data']:
                record_entry['image_data'] = [process_image(image) for image in obs['multi_modal_data'][image_placeholder]]
            for key, value in obs['multi_modal_data'].items():
                if key != image_placeholder and value:
                    record_entry[f'multi_modal_{key}'] = [process_image(image) for image in value]
        self.recorder[env_id].append(record_entry)

    @torch.no_grad()
    def _single_recording_to_prompt(self,
                                    recording: List[Dict],
                                    step: int,
                                    window_size: int = None,
                                    is_final: bool = False,
                                    prep_for_loss_mask: bool = False,
                                    ):
        """
        Given a recording, generate the prompt for Cambrian-S.
        Chat: Sys -> |InitUser| -> |Assistant, User| -> ... -> |Assistant, User Final|
        
        Uses ChatML format (same as Qwen): <|im_start|>role\ncontent<|im_end|>
        """
        assert step >= 0
        start_step = max(0, step - window_size) if window_size is not None else 0
        end_step = step
        assert len(recording) >= end_step + 1
        history = recording[start_step: end_step + 1]
        rewards = []
        chat = []

        env_id = history[0]['env_id']
        chat.append({"role": "system", "content": self.envs[env_id].system_prompt()})

        image_data = []

        # Prior Images Persistence Logic
        full_first_record = recording[0] if recording else None
        prior_image_keys = []
        prior_images_for_injection = []

        if full_first_record:
            prior_image_keys = [k for k in full_first_record.keys() if k.startswith('multi_modal_')]
            for prior_key in prior_image_keys:
                prior_images_for_injection.extend(full_first_record[prior_key])

        window_slid_past_priors = (start_step > 0) and len(prior_images_for_injection) > 0

        for i, record in enumerate(history):
            if i > 0:
                llm_raw_response = record['info']['llm_raw_response']
                filtered_llm_raw_response = self._handle_special_tokens(llm_raw_response, prep_for_loss_mask=prep_for_loss_mask)
                chat.append({"role": "assistant", "content": filtered_llm_raw_response})
                rewards.append(record['reward'])
            if i < len(history) - 1 or not is_final:
                obs_str = record['obs_str']

                if i == 0 and window_slid_past_priors:
                    for img in prior_images_for_injection:
                        image_data.append(img)
                    prior_placeholder = self.envs[env_id].config.get('spatial_prior_placeholder', '<prior_images>')
                    num_prior = len(prior_images_for_injection)
                    prior_text = f"[Spatial Context (from episode start)]:\n"
                    prior_text += " ".join([prior_placeholder] * num_prior)
                    prior_text += f"\n\n"
                    obs_str = prior_text + obs_str

                chat.append({"role": "user", "content": obs_str})

                if 'image_data' in record:
                    for img in record['image_data']:
                        image_data.append(img)

                if i == 0 and not window_slid_past_priors:
                    for prior_key in [k for k in record.keys() if k.startswith('multi_modal_')]:
                        for img in record[prior_key]:
                            image_data.append(img)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=(not is_final), tokenize=False)
        
        if is_final:
            assert prompt_with_chat_template[-1] == '\n', \
                f"The last token should be new line token, got {prompt_with_chat_template[-1]}"
            prompt_with_chat_template = prompt_with_chat_template[:-1]

        # Switch box_end and im_end so that the model can learn to generate <|im_end|>
        prompt_with_chat_template = prompt_with_chat_template.replace(
            f'{self.special_token_for_loss_mask[1]}{self.tokenizer.eos_token}',
            f'{self.tokenizer.eos_token}{self.special_token_for_loss_mask[1]}')

        return {
            "prompt": prompt_with_chat_template,
            "image_data": image_data,
            "rewards": rewards,
        }

    @torch.no_grad()
    def _generate_input_for_rollout(
            self,
            recording: List[Dict],
            step: int,
            window_size: int = None,
    ):
        """
        Generate input for Cambrian-S generation (HF rollout).
        
        Creates proper input_ids with tokens_per_image IMAGE_TOKEN_INDEX per image,
        preprocessed pixel_values, and proper attention_mask/position_ids.
        """
        rst = self._single_recording_to_prompt(recording, step, window_size, is_final=False, prep_for_loss_mask=False)
        prompt_with_chat_template = rst['prompt']
        image_data = rst['image_data']
        has_images = len(image_data) > 0

        row_dict = {}
        if has_images:
            prompt_with_chat_template, row_dict, _, raw_prompt = self._handle_multi_modal_data(
                prompt_with_chat_template, row_dict, image_data, do_embedding=False)
        else:
            raw_prompt = prompt_with_chat_template

        # Tokenize the prompt — <image> is a single special token
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        
        # Expand each <image> token to tokens_per_image IMAGE_TOKEN_INDEX
        new_ids = []
        for tid in raw_prompt_ids:
            if tid == self.image_token_id:
                new_ids.extend([IMAGE_TOKEN_INDEX] * self.tokens_per_image)
            else:
                new_ids.append(tid)
        
        # Truncate to max_prompt_length if needed
        max_prompt_length = getattr(self.config, 'max_prompt_length', 2048)
        if len(new_ids) > max_prompt_length:
            new_ids = new_ids[-max_prompt_length:]  # Left truncation, keep recent context
        
        # Create proper tensors for HF generation
        input_ids = torch.tensor([new_ids], dtype=torch.long)  # (1, seq_len)
        attention_mask = torch.ones_like(input_ids)  # All tokens attended
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0)  # (1, seq_len)
        
        # Note: raw_prompt_ids is not used by CambrianHFRollout, but kept for compatibility with verl's DataProto
        # and for logging/debugging. Safe to remove if not needed.
        # row_dict['raw_prompt_ids'] = new_ids
        row_dict['input_ids'] = input_ids.squeeze(0)  # (seq_len,)
        row_dict['attention_mask'] = attention_mask.squeeze(0)  # (seq_len,)
        row_dict['position_ids'] = position_ids.squeeze(0)  # (seq_len,)

        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    @torch.no_grad()
    def _generate_input_for_uptate(
            self,
            recording: List[Dict],
            step: int,
            window_size: int = None,
    ):
        """
        Generate the final trajectory input for FSDP training update.
        
        For Cambrian-S:
        - Tokenize the full trajectory
        - Expand each <image> token to tokens_per_image IMAGE_TOKEN_INDEX in input_ids
        - Process images through the vision encoder's image preprocessor
        - Store pixel_values in multi_modal_inputs for the adapter's forward pass
        - Use standard 1D position IDs (no 3D mRoPE)
        """
        # Prompt = single pad token (masked out)
        prompt_with_chat_template = self.tokenizer.pad_token

        # Handle response (full trajectory)
        response_rst = self._single_recording_to_prompt(recording, step, window_size,
                                                         is_final=True, prep_for_loss_mask=True)
        response_with_chat_template = response_rst['prompt']
        image_data = response_rst['image_data']
        rewards = response_rst['rewards']

        has_images = len(image_data) > 0
        row_dict = {}
        if has_images:
            response_with_chat_template, row_dict, _, _ = self._handle_multi_modal_data(
                response_with_chat_template, row_dict, image_data, do_embedding=True)

        # Tokenize response — <image> is a single special token
        response_token_ids = self.tokenizer.encode(response_with_chat_template, add_special_tokens=False)
        
        # Expand each <image> token to tokens_per_image IMAGE_TOKEN_INDEX
        if has_images:
            expanded_ids = []
            for tid in response_token_ids:
                if tid == self.image_token_id:
                    expanded_ids.extend([IMAGE_TOKEN_INDEX] * self.tokens_per_image)
                else:
                    expanded_ids.append(tid)
            response_token_ids = expanded_ids

        # Truncate/pad to max_trajectory_length - 1 (leaving 1 for prompt pad token)
        max_response_len = self.max_trajectory_length - 1
        if len(response_token_ids) > max_response_len:
            # Left truncation
            if self.truncation == 'left':
                response_token_ids = response_token_ids[-max_response_len:]
            else:
                response_token_ids = response_token_ids[:max_response_len]
        
        # Convert to tensor and pad
        response_len = len(response_token_ids)
        pad_len = max_response_len - response_len
        
        if pad_len > 0:
            response_token_ids_padded = response_token_ids + [self.tokenizer.pad_token_id] * pad_len
            response_attention = [1] * response_len + [0] * pad_len
        else:
            response_token_ids_padded = response_token_ids
            response_attention = [1] * response_len
        
        input_ids_response = torch.tensor([response_token_ids_padded], dtype=torch.long)
        attention_mask_response = torch.tensor([response_attention], dtype=torch.long)

        # Tokenize prompt (single pad token)
        input_ids_prompt, attention_mask_prompt = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=1,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation)
        attention_mask_prompt = torch.zeros_like(input_ids_prompt)

        # Compute loss mask
        input_ids_response, attention_mask_response, loss_mask_response, end_of_response_position_mask_response = \
            self._compute_loss_mask(input_ids_response, attention_mask_response)

        input_ids_prompt = input_ids_prompt[0]
        attention_mask_prompt = attention_mask_prompt[0]
        loss_mask_prompt = torch.zeros_like(attention_mask_prompt)
        end_of_response_position_mask_prompt = torch.zeros_like(attention_mask_prompt)

        input_ids_response = input_ids_response[0]
        attention_mask_response = attention_mask_response[0]
        loss_mask_response = loss_mask_response[0]
        end_of_response_position_mask_response = end_of_response_position_mask_response[0]

        loss_mask = torch.cat([loss_mask_prompt, loss_mask_response], dim=-1)
        end_of_response_position_mask = torch.cat([end_of_response_position_mask_prompt, end_of_response_position_mask_response], dim=-1)
        input_ids = torch.cat([input_ids_prompt, input_ids_response], dim=-1)
        attention_mask = torch.cat([attention_mask_prompt, attention_mask_response], dim=-1)

        # Compute position IDs — standard 1D sequential (no 3D mRoPE for Cambrian-S)
        position_ids_prompt = compute_position_id_with_mask(attention_mask_prompt)

        response_length = input_ids_response.shape[0]
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids_prompt.device)
        position_ids_response = position_ids_prompt[-1:] + delta_position_id

        # Multi-turn reward assignment
        if self.use_multi_turn_reward:
            reward_positions = torch.nonzero(end_of_response_position_mask).squeeze(-1)
            multi_turn_token_level_rewards = torch.zeros_like(end_of_response_position_mask, dtype=torch.float)

            if len(reward_positions) == 0 and len(rewards) > 0:
                print("[WARNING] Left truncation removed all reward positions")
                rewards = []
            elif len(reward_positions) < len(rewards):
                print(f"[WARNING] Left truncation: {len(rewards)} rewards but only {len(reward_positions)} positions")
                rewards = rewards[-len(reward_positions):]

            assert len(reward_positions) == len(rewards), \
                f"Rewards ({len(rewards)}) != positions ({len(reward_positions)})"
            for idx, reward in enumerate(rewards):
                multi_turn_token_level_rewards[reward_positions[idx]] = reward
            row_dict["multi_turn_token_level_rewards"] = multi_turn_token_level_rewards
            row_dict["end_of_response_position_mask"] = end_of_response_position_mask

        if self.use_loss_mask:
            row_dict['loss_mask'] = loss_mask
        if self.use_gae_mask:
            row_dict['gae_mask'] = loss_mask
        row_dict["end_of_response_position_mask"] = end_of_response_position_mask

        position_ids = torch.cat([position_ids_prompt, position_ids_response], dim=-1)
        row_dict['prompts'] = input_ids_prompt
        row_dict['responses'] = input_ids_response
        row_dict['input_ids'] = input_ids
        row_dict['attention_mask'] = attention_mask
        row_dict['position_ids'] = position_ids
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["step_reward_sum"] = sum(rewards)
        return row_dict

    @torch.no_grad()
    def generate_batch_for_rollout(self, step, window_size):
        """Generate a batch of data for HF generation at the current step.
        
        Unlike Qwen's vLLM path (which uses dummy input_ids and relies on raw_prompt_ids),
        we create proper input_ids with IMAGE_TOKEN_INDEX expansion. Different samples may
        have different lengths, so we left-pad to the max length in the batch.
        """
        batch = []
        self.batch_idx_to_env_id = {}
        batch_idx = 0
        for env_id in self.envs.keys():
            if self.env_states[env_id]['done']:
                continue
            batch.append(self._generate_input_for_rollout(self.recorder[env_id], step, window_size))
            self.batch_idx_to_env_id[batch_idx] = env_id
            batch_idx += 1
        if not batch:
            return None
        if len(batch) % self.n_gpus_per_node != 0:
            while len(batch) % self.n_gpus_per_node != 0:
                batch.append(batch[-1].copy())
        
        # Left-pad input_ids/attention_mask/position_ids to same length before collate
        max_len = max(b['input_ids'].size(0) for b in batch)
        pad_token_id = self.tokenizer.pad_token_id
        for b in batch:
            cur_len = b['input_ids'].size(0)
            if cur_len < max_len:
                pad_len = max_len - cur_len
                b['input_ids'] = torch.cat([
                    torch.full((pad_len,), pad_token_id, dtype=torch.long),
                    b['input_ids']
                ])
                b['attention_mask'] = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    b['attention_mask']
                ])
                b['position_ids'] = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    b['position_ids']
                ])

        # Note: After collate_fn, multi_modal_inputs is a list of dicts (one per sample),
        # where each dict contains pixel_values (tensor). This is expected by dp_actor.py and CambrianHFRollout.
        # Do NOT change this structure unless you update all downstream consumers.
        return collate_fn(batch)

    @torch.no_grad()
    def rollout_loop(self):
        """Step the environment and record the results using Cambrian-S HF generation."""
        for step in range(self.max_turns):
            input_batch_dict = self.generate_batch_for_rollout(step, self.window_size)
            if input_batch_dict is None:
                break
            input_batch = DataProto.from_single_dict(input_batch_dict)
            
            # Build gen_batch with proper keys for CambrianHFRollout
            # NOTE: Cambrian uses HF rollout (not vLLM), so raw_prompt_ids is NOT needed.
            # CambrianHFRollout uses input_ids + multi_modal_inputs directly.
            non_tensor_keys = []
            if 'multi_modal_data' in input_batch.non_tensor_batch.keys():
                non_tensor_keys.append('multi_modal_data')
            if 'multi_modal_inputs' in input_batch.non_tensor_batch.keys():
                non_tensor_keys.append('multi_modal_inputs')
            
            gen_batch = input_batch.pop(
                batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                non_tensor_batch_keys=non_tensor_keys,
            )

            output_batch = self.actor_rollout_wg.generate_sequences(gen_batch)

            responses_str = self.tokenizer.batch_decode(
                output_batch.batch['responses'],
                skip_special_tokens=True
            )

            for batch_idx, env_id in self.batch_idx_to_env_id.items():
                obs, reward, done, info = self.envs[env_id].step(responses_str[batch_idx])
                info['llm_raw_response'] = responses_str[batch_idx]
                self.env_states[env_id]['step'] += 1
                self.env_states[env_id]['done'] = done
                self.env_states[env_id]['metrics']['traj_metrics'] = info['metrics'].get('traj_metrics', {})
                for k, v in info['metrics']['turn_metrics'].items():
                    self.env_states[env_id]['metrics']['turn_metrics'][k].append(v)
                self.record(env_id, obs, reward, done, info)

    @torch.no_grad()
    def generate_batch_for_update(self) -> DataProto:
        """Get the final trajectory of all environments."""
        batch_list = []
        for env_id in self.envs.keys():
            row_dict = self._generate_input_for_uptate(
                recording=self.recorder[env_id],
                step=self.env_states[env_id]['step'],
                window_size=None,
            )
            step_reward_sum = row_dict['step_reward_sum']
            last_reward = self.envs[env_id].compute_reward()
            row_dict['reward_model'] = {"style": "given", "ground_truth": {"reward": last_reward + step_reward_sum}}
            if self.use_multi_turn_reward:
                end_of_response_position_mask = row_dict['end_of_response_position_mask']
                reward_positions = torch.nonzero(end_of_response_position_mask).squeeze(-1)
                last_reward_index = reward_positions[-1]
                row_dict['multi_turn_token_level_rewards'][last_reward_index] += last_reward
            batch_list.append(row_dict)
        batch_dict = collate_fn(batch_list)
        batch = DataProto.from_single_dict(batch_dict)
        return batch

    @torch.no_grad()
    def recording_to_log(self):
        """Get the recording of all environments."""
        env_info = []
        for env_id, record in self.recorder.items():
            config_id = self.envs[env_id].config.config_id()
            step = self.env_states[env_id]['step']
            output_rst = self._single_recording_to_prompt(record, self.env_states[env_id]['step'], window_size=None, is_final=False)
            image = output_rst['image_data']
            done = self.env_states[env_id]['done']
            score = self.envs[env_id].compute_reward() + sum(output_rst['rewards'])

            metrics = {
                "score": score,
                "done": done,
                "step": step,
            }
            turn_metrics = {
                k: sum(v) / step if step != 0 else 0
                for k, v in self.env_states[env_id]['metrics']['turn_metrics'].items()
            }
            traj_metrics = self.env_states[env_id]['metrics']['traj_metrics']
            metrics.update(turn_metrics)
            metrics.update(traj_metrics)
            env_info.append({
                "env_id": env_id,
                "config_id": config_id,
                "output_str": output_rst['prompt'],
                "image_data": image,
                "metrics": metrics,
            })
        return env_info
