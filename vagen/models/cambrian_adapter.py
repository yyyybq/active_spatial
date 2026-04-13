"""
Cambrian-S model adapter for VAGEN FSDP+vLLM training pipeline.

This module provides:
1. CambrianForCausalLMAdapter: Wraps CambrianQwenForCausalLM to be compatible with verl's
   FSDP training pipeline by adapting the multimodal input interface.
2. Helper functions for model loading and registration.

Key adaptation: verl's actor passes `pixel_values` via **multi_modal_inputs, but Cambrian-S
expects `images` parameter in its forward(). This adapter bridges that gap.
It also handles the GPU-side image encoding (vision tower + projector) which was originally 
designed for TPU with XLA scatter kernels, adapted here for standard GPU operation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


# Import constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200


class CambrianForCausalLMAdapter(nn.Module):
    """
    Adapter that wraps CambrianQwenForCausalLM to work with verl's FSDP pipeline.
    
    verl's dp_actor.py calls:
        output = self.actor_module(input_ids=..., attention_mask=..., position_ids=..., **multi_modal_inputs)
    
    Where multi_modal_inputs = {'pixel_values': tensor} for Cambrian-S.
    
    This adapter:
    1. Intercepts pixel_values from kwargs
    2. Processes images through vision tower + projector (on GPU, not XLA)
    3. Replaces IMAGE_TOKEN_INDEX positions in input_ids with projected image features
    4. Passes inputs_embeds to the Qwen2 backbone
    """

    def __init__(self, model):
        """
        Args:
            model: CambrianQwenForCausalLM instance
        """
        super().__init__()
        self.model = model
        self.config = model.config

        # Fix missing _attn_implementation on the inner CambrianQwenModel.
        # Cambrian-S only sets config._attn_implementation on XLA; on GPU with
        # newer transformers (4.49+), Qwen2Model.forward() reads self._attn_implementation
        # which may not be set. Force 'eager' to match the TRANSFORMERS_ATTN_IMPLEMENTATION env.
        attn_impl = getattr(self.config, '_attn_implementation', 'eager')
        inner_model = getattr(self.model, 'model', None)
        if inner_model is not None and not hasattr(inner_model, '_attn_implementation'):
            inner_model._attn_implementation = attn_impl

    def _encode_and_project_images(self, pixel_values):
        """
        Encode images through vision tower and project to LLM embedding space.
        
        Args:
            pixel_values: (N, C, H, W) tensor of preprocessed images
        
        Returns:
            image_features: (N, num_tokens, hidden_size) projected image features
        """
        vision_tower_aux_list = self.model.get_model().get_vision_tower_aux_list()
        
        # Encode through vision tower(s)
        image_aux_features = vision_tower_aux_list[0](pixel_values)  # (N, L, 1152)
        
        # Project through MLP
        image_features = self.model.get_model().mm_projector(image_aux_features)  # (N, L, hidden_size)
        
        return image_features

    def _prepare_inputs_embeds_gpu(self, input_ids, pixel_values, attention_mask=None):
        """
        Prepare inputs_embeds by encoding images and inserting them at IMAGE_TOKEN_INDEX positions.
        GPU-compatible version (no XLA scatter kernels).
        
        Args:
            input_ids: (B, seq_len)  
            pixel_values: (N, C, H, W) - all images concatenated
            attention_mask: (B, seq_len)
        
        Returns:
            inputs_embeds: (B, seq_len, hidden_size)
            modified attention_mask, position_ids
        """
        device = input_ids.device
        dtype = self.model.get_model().embed_tokens.weight.dtype
        
        # Get text embeddings (replace IMAGE_TOKEN_INDEX with 0 for embedding lookup)
        input_ids_for_embed = input_ids.clone()
        input_ids_for_embed[input_ids_for_embed == IMAGE_TOKEN_INDEX] = 0
        inputs_embeds = self.model.get_model().embed_tokens(input_ids_for_embed).to(dtype)
        
        if pixel_values is None or pixel_values.numel() == 0:
            return inputs_embeds
        
        # Encode and project images  
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        image_features = self._encode_and_project_images(pixel_values)  # (N, L, hidden_size)
        
        # Get config values
        si_token_len = getattr(self.config, 'si_token_len', 729)
        si_side_len = int(si_token_len ** 0.5) if si_token_len > 0 else None
        
        if si_side_len is not None:
            feature_side_len = int(image_features.size(1) ** 0.5)
            # Reshape to spatial: (N, C, H, W)
            image_features_spatial = image_features.unflatten(1, (feature_side_len, feature_side_len)).permute(0, 3, 1, 2)
            
            if si_side_len != feature_side_len:
                image_features_spatial = F.interpolate(
                    image_features_spatial.float(),
                    size=(si_side_len, si_side_len),
                    mode="bilinear",
                    align_corners=False
                ).to(dtype)
            
            # Reshape back: (N, H, W, C)
            image_features_spatial = image_features_spatial.permute(0, 2, 3, 1)
            
            # Add newline tokens
            newline = self.model.get_model().image_newline  # (hidden_size,)
            newline_expanded = newline[None, None, None, :].expand(
                image_features_spatial.size(0), image_features_spatial.size(1), 1, -1)
            image_features_with_newline = torch.cat([image_features_spatial, newline_expanded], dim=2)
            
            # Flatten: (N, H*(W+1), C)
            per_image_tokens = image_features_with_newline.flatten(1, 2)
        else:
            per_image_tokens = image_features
        
        # Insert image features at IMAGE_TOKEN_INDEX positions
        # Images are concatenated across batch: pixel_values = (N_total, C, H, W)
        # We need to track a global image index across batch elements
        tokens_per_image = per_image_tokens.size(1)  # 756 with newlines, 729 without
        batch_size = input_ids.size(0)
        global_img_idx = 0
        
        for b in range(batch_size):
            image_positions = (input_ids[b] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            
            if len(image_positions) == 0:
                continue
            
            num_images_for_sample = len(image_positions) // tokens_per_image if tokens_per_image > 0 else 0
            
            for img_i in range(num_images_for_sample):
                start_tok = img_i * tokens_per_image
                end_tok = start_tok + tokens_per_image
                positions = image_positions[start_tok:end_tok]
                if global_img_idx < per_image_tokens.size(0):
                    inputs_embeds[b, positions] = per_image_tokens[global_img_idx].to(inputs_embeds.dtype)
                    global_img_idx += 1
            
            # Handle leftover positions (partial image, shouldn't happen with correct expansion)
            remaining = len(image_positions) - num_images_for_sample * tokens_per_image
            if remaining > 0 and global_img_idx < per_image_tokens.size(0):
                leftover_positions = image_positions[num_images_for_sample * tokens_per_image:]
                inputs_embeds[b, leftover_positions] = per_image_tokens[global_img_idx, :remaining].to(inputs_embeds.dtype)
        
        return inputs_embeds

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        # Cambrian-S specific
        pixel_values=None,
        **kwargs,
    ):
        """
        Forward pass adapted for verl's FSDP pipeline.
        
        Intercepts pixel_values, processes images, and delegates to the Qwen2 backbone.
        """
        # If pixel_values are provided, compute inputs_embeds ourselves
        if pixel_values is not None and inputs_embeds is None and input_ids is not None:
            inputs_embeds = self._prepare_inputs_embeds_gpu(input_ids, pixel_values, attention_mask)
            # Clear input_ids since we're passing inputs_embeds
            input_ids = None
        elif input_ids is not None and inputs_embeds is None:
            # Safety: clamp input_ids to valid range (IMAGE_TOKEN_INDEX=-200 would crash embed_tokens)
            if (input_ids == IMAGE_TOKEN_INDEX).any():
                safe_ids = input_ids.clamp(min=0)
                inputs_embeds = self.model.get_model().embed_tokens(safe_ids)
                input_ids = None
        
        # Call the underlying Qwen2 model directly (bypass Cambrian's prepare_inputs_labels)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(self, input_ids=None, attention_mask=None, pixel_values=None, **kwargs):
        """
        Generate sequences, handling Cambrian-S image encoding on GPU.
        
        Strategy: Monkey-patch self.model.forward to route through our adapter's 
        forward (which handles pixel_values → inputs_embeds), then call 
        self.model.generate() which is CambrianQwenForCausalLM.generate() 
        (a proper GenerationMixin.generate with lm_head).
        
        This ensures:
        1. First step: images are encoded and embedded 
        2. Subsequent steps: normal autoregressive generation with KV-cache
        3. output.sequences includes prompt tokens (standard HF behavior)
        """
        # Store pixel_values for use by the patched forward
        _pixel_values = pixel_values
        _first_call = [True]  # mutable to allow nonlocal-like access in closure
        original_forward = self.model.forward
        adapter = self
        
        def _patched_forward(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            pixel_values=None,
            **kwargs,
        ):
            """Route through adapter's forward, injecting pixel_values on first call."""
            if _first_call[0] and _pixel_values is not None:
                _first_call[0] = False
                pixel_values = _pixel_values
            # Remove any Cambrian-specific args that the adapter doesn't expect
            kwargs.pop('images', None)
            kwargs.pop('image_sizes', None)
            return adapter.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pixel_values=pixel_values,
                **kwargs,
            )
        
        # Monkey-patch and generate
        self.model.forward = _patched_forward
        try:
            # Remove pixel_values from kwargs (already captured above)
            kwargs.pop('pixel_values', None)
            kwargs.pop('images', None)
            kwargs.pop('image_sizes', None)
            
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        finally:
            # Always restore original forward
            self.model.forward = original_forward
        
        return output

    def gradient_checkpointing_enable(self, *args, **kwargs):
        """Delegate gradient checkpointing."""
        return self.model.gradient_checkpointing_enable(*args, **kwargs)

    @property
    def dtype(self):
        return self.model.dtype

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    @property
    def device(self):
        return self.model.device


def load_cambrian_model(model_path, torch_dtype=torch.bfloat16, device_map=None):
    """
    Load Cambrian-S model for VAGEN training.
    
    Returns:
        model: CambrianQwenForCausalLM (or wrapped in adapter)
        tokenizer: AutoTokenizer
        image_processor: list of image processors (one per vision tower)
    """
    import sys
    # Add cambrian-s to Python path so the model classes can be resolved
    cambrian_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  '..', '..', 'cambrian-s')
    cambrian_path = os.path.normpath(cambrian_path)
    if os.path.exists(cambrian_path) and cambrian_path not in sys.path:
        sys.path.insert(0, cambrian_path)
    
    from cambrian.model.builder import load_pretrained_model
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="cambrian-s",
        device_map=device_map or "auto",
        device="cuda",
    )
    
    return model, tokenizer, image_processor


def ensure_cambrian_vision_towers_loaded(model, device='cuda'):
    """
    Ensure that Cambrian-S vision tower(s) are loaded after from_pretrained.
    
    When CambrianQwenForCausalLM is loaded via AutoModelForCausalLM.from_pretrained(),
    the vision towers might have delay_load=True. This function ensures they're loaded.
    
    The vision towers should be FROZEN during RL training (only LLM + projector are trained).
    """
    # Handle adapter wrapper
    inner_model = model
    while hasattr(inner_model, 'model') and not hasattr(inner_model, 'get_vision_tower_aux_list'):
        inner_model = inner_model.model
    
    if not hasattr(inner_model, 'get_vision_tower_aux_list'):
        print("[WARNING] Model does not have get_vision_tower_aux_list method. Skipping vision tower init.")
        return []
    
    # Determine model dtype for vision towers (match the LLM's dtype)
    model_dtype = getattr(inner_model, 'dtype', torch.bfloat16)
    if hasattr(inner_model, 'embed_tokens'):
        model_dtype = inner_model.embed_tokens.weight.dtype
    elif hasattr(inner_model, 'get_model') and hasattr(inner_model.get_model(), 'embed_tokens'):
        model_dtype = inner_model.get_model().embed_tokens.weight.dtype
    
    vision_tower_aux_list = inner_model.get_vision_tower_aux_list()
    if vision_tower_aux_list is None:
        print("[WARNING] vision_tower_aux_list is None")
        return []
    
    image_processor_list = []
    for vt in vision_tower_aux_list:
        if not vt.is_loaded:
            print(f"[CambrianAdapter] Loading vision tower: {vt.vision_tower_name}")
            vt.load_model()
        vt.to(device=device, dtype=model_dtype)
        vt.eval()
        # Freeze vision tower
        for param in vt.parameters():
            param.requires_grad = False
        image_processor_list.append(vt.image_processor)
        print(f"[CambrianAdapter] Vision tower loaded, dtype={model_dtype}, device={device}")
    
    return image_processor_list
