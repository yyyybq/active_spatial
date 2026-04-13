"""
Cambrian-S HF Rollout: extends verl's HFRollout to handle multimodal generation.

The standard HFRollout only passes input_ids/attention_mask/position_ids to generate().
For Cambrian-S, we also need to pass pixel_values for image features.

This rollout:
1. Extracts pixel_values from the DataProto's non_tensor_batch['multi_modal_inputs']
2. Passes pixel_values to the model's generate() (via CambrianForCausalLMAdapter)
3. The adapter creates inputs_embeds with image features and delegates to Qwen2 generate
"""

import contextlib
import os
import torch
import torch.distributed
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask
from verl.workers.rollout.base import BaseRollout
from transformers import GenerationConfig

def _debug_log(msg):
    """Write debug info to a file (bypasses Ray stdout capture)."""
    logdir = "/scratch/by2593/project/Active_Spatial/VAGEN/_cambrian_debug"
    os.makedirs(logdir, exist_ok=True)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    path = os.path.join(logdir, f"rank{rank}.log")
    with open(path, "a") as f:
        f.write(msg + "\n")
        f.flush()


class CambrianHFRollout(BaseRollout):
    """HF-based rollout engine for Cambrian-S with multimodal support."""

    def __init__(self, module, config):
        super().__init__()
        self.config = config
        self.module = module

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences with micro-batching support."""
        batch_size = prompts.batch.batch_size[0]
        micro_batch_size = self.config.get('micro_batch_size', 1)  # Default to 1 for Cambrian (bs=1 constraint)
        num_chunks = max(batch_size // micro_batch_size, 1)
        
        # Chunk carefully to preserve non_tensor_batch
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        """Generate a mini-batch with multimodal support."""
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        eos_token_id = prompts.meta_info['eos_token_id']
        pad_token_id = prompts.meta_info['pad_token_id']

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        # Sampling parameters
        do_sample = prompts.meta_info.get('do_sample', self.config.do_sample)
        response_length = prompts.meta_info.get('response_length', self.config.response_length)
        top_p = prompts.meta_info.get('top_p', self.config.get('top_p', 1.0))
        top_k = prompts.meta_info.get('top_k', self.config.get('top_k', 0))

        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)

        temperature = prompts.meta_info.get('temperature', self.config.temperature)
        generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

        # Extract pixel_values from non_tensor_batch if available
        pixel_values = None
        has_multi_modal = 'multi_modal_inputs' in prompts.non_tensor_batch
        if has_multi_modal:
            # multi_modal_inputs is a list of dicts (one per sample in micro-batch)
            mm_inputs_list = prompts.non_tensor_batch['multi_modal_inputs']
            if mm_inputs_list is not None and len(mm_inputs_list) > 0:
                # Concatenate pixel_values across all samples
                pv_list = []
                for mm_dict in mm_inputs_list:
                    if isinstance(mm_dict, dict) and 'pixel_values' in mm_dict:
                        pv = mm_dict['pixel_values']
                        if isinstance(pv, torch.Tensor):
                            pv_list.append(pv)
                if pv_list:
                    pixel_values = torch.cat(pv_list, dim=0).to(idx.device)
        n_img_tokens = (idx == -200).sum().item()
        _debug_log(f"[CambrianHFRollout] has_multi_modal={has_multi_modal}, "
              f"pixel_values={'None' if pixel_values is None else pixel_values.shape}, "
              f"input_ids_shape={idx.shape}, n_img_tokens={n_img_tokens}, "
              f"non_tensor_keys={list(prompts.non_tensor_batch.keys())}")
        if n_img_tokens > 0 and pixel_values is None:
            _debug_log(f"[CambrianHFRollout] WARNING: {n_img_tokens} image tokens but NO pixel_values!")
            if has_multi_modal:
                mm_list = prompts.non_tensor_batch['multi_modal_inputs']
                _debug_log(f"  mm_list type={type(mm_list)}, len={len(mm_list)}")
                for i, mm in enumerate(mm_list[:3]):
                    _debug_log(f"  mm[{i}] type={type(mm)}, keys={mm.keys() if isinstance(mm, dict) else 'N/A'}")

        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)

        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                generate_kwargs = dict(
                    input_ids=idx,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    output_scores=False,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
                
                # Pass pixel_values for Cambrian multimodal generation
                if pixel_values is not None:
                    generate_kwargs['pixel_values'] = pixel_values

                output = self.module.generate(**generate_kwargs)

        seq = output.sequences
        _debug_log(f"[CambrianHFRollout] generation done, seq_shape={seq.shape}, "
                   f"first_new_tokens={seq[0, prompt_length:prompt_length+10].tolist()}")

        # Pad to response_length if generation stopped early
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)

        # Truncate if somehow too long
        if seq.shape[1] > sequence_length:
            seq = seq[:, :sequence_length]

        assert seq.shape[1] == sequence_length, \
            f"Expected seq length {sequence_length}, got {seq.shape[1]}"

        prompt = seq[:, :prompt_length]
        response = seq[:, prompt_length:]

        response_length_actual = response.size(1)
        delta_position_id = torch.arange(1, response_length_actual + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_eos_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                'prompts': prompt,
                'responses': response,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            },
            batch_size=batch_size,
        )

        torch.cuda.empty_cache()
        self.module.train()
        return DataProto(batch=batch)
