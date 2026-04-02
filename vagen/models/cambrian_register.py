"""
vagen/models/cambrian_register.py

Registration module for Cambrian-S in VAGEN's FSDP/HF training pipeline.

Imported by verl's `external_lib` mechanism **before** any model is loaded:
    actor_rollout_ref.model.external_lib=vagen.models.cambrian_register
    critic.model.external_lib=vagen.models.cambrian_register

What this file does:
  1. Adds the cambrian-s source tree to sys.path (since it isn't pip-installed).
  2. Imports CambrianQwenForCausalLM, which triggers:
         AutoConfig.register("cambrian_qwen", CambrianQwenConfig)
         AutoModelForCausalLM.register(CambrianQwenConfig, CambrianQwenForCausalLM)
  3. Defines CambrianForCausalLMAdapter – a GPU-compatible subclass that:
       a. Overrides forward()   – batched image embedding without XLA scatter kernels.
       b. Overrides generate()  – accepts pixel_values from verl multi_modal_inputs.
       c. Supports NFP head     – pass nfp_pixel_values + nfp_loss_masks in forward().
  4. Re-registers the adapter class so verl loads it instead of the base class.

---
Background: Why -200 is in input_ids
--------------------------------------
CambrianRolloutManager tokenises prompts that contain the literal string "<image>".
The tokeniser gives "<image>" a single vocab ID (self.image_token_id).  The manager
then walks every token in the sequence and replaces each image_token_id with
`tokens_per_image` copies of IMAGE_TOKEN_INDEX = -200 (Cambrian's convention).

Example (tokens_per_image = 756):
  text:       "…<image>…"
  tokeniser:  […, image_token_id, …]
  after expand: […, -200, -200, …, -200 (×756), …]

The adapter's _embed_multimodal_batch() detects the -200 blocks and scatters the
corresponding projected SigLIP features in their place before the forward pass.

---
NFP (Next-Frame Prediction) head
--------------------------------------
NFP is an auxiliary self-supervised objective:
  At each action position t, the model predicts the visual features of obs_{t+1}.

To enable during RL training:
  1. Set `use_nfp: True` in rollout_manager config and
     `nfp_mse_loss_weight` / `nfp_cosine_loss_weight` in model config
     (or pass +rollout_manager.use_nfp=True in the launch script).
  2. CambrianRolloutManager._generate_input_for_uptate() will populate
       row_dict["nfp_pixel_values"]  – (total_action_turns, C, H, W)
       row_dict["nfp_loss_masks"]    – (seq_len,) 1 at action end positions
     These travel through DataProto.non_tensor_batch → dp_actor.py →
     multi_modal_inputs → this adapter's forward() as nfp_pixel_values / nfp_loss_masks.
"""

import os
import sys
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

# ---------------------------------------------------------------------------
# 1.  Make cambrian importable
# ---------------------------------------------------------------------------
CAMBRIAN_SRC = os.environ.get("CAMBRIAN_SRC", "/nas/baiqiao/cambrian-s")
if CAMBRIAN_SRC not in sys.path:
    sys.path.insert(0, CAMBRIAN_SRC)

# This import triggers the two AutoXxx.register() calls at the bottom of
# cambrian_qwen2.py – side-effects are what we need.
from cambrian.model.language_model.cambrian_qwen2 import (  # noqa: E402
    CambrianQwenConfig,
    CambrianQwenForCausalLM,
)

# Cambrian-S uses IMAGE_TOKEN_INDEX = -200 as a sentinel in input_ids.
# The rollout manager pre-expands each <image> token into tokens_per_image
# copies of this value before sending the batch to the model.
IMAGE_TOKEN_INDEX: int = -200


# ---------------------------------------------------------------------------
# 2.  GPU-compatible adapter
# ---------------------------------------------------------------------------
class CambrianForCausalLMAdapter(CambrianQwenForCausalLM):
    """
    Wraps CambrianQwenForCausalLM for standard GPU FSDP + verl HF rollout.

    Key problems solved vs. base class
    ------------------------------------
    * forward() – base class dispatches to prepare_inputs_labels_for_multimodal()
      which is **XLA-only** and raises NotImplementedError on regular CUDA GPUs.
      This adapter bypasses that dispatch entirely.
    * generate() – verl's HF rollout calls model.generate(input_ids=…), never
      passing images.  This override accepts pixel_values as a kwarg (verl passes
      multi_modal_inputs as **kwargs to generate once we patch hf_rollout.py) and
      pre-computes inputs_embeds before handing off to Qwen2ForCausalLM.generate().
    * NFP head – optional; enabled when nfp_pixel_values is present in the batch.
    * connector_only=True assumed (no SVA cross-attention layers needed).
    """

    # ------------------------------------------------------------------
    # Core image-embedding helper
    # ------------------------------------------------------------------
    def _embed_multimodal_batch(
        self,
        input_ids: torch.Tensor,         # (bs, seq_len)  contains IMAGE_TOKEN_INDEX=-200
        pixel_values: torch.Tensor,       # (total_imgs_in_batch, C, H, W)
    ) -> torch.Tensor:                   # returns inputs_embeds (bs, seq_len, hidden_dim)
        """
        GPU-compatible replacement for prepare_inputs_labels_for_multimodal.

        Algorithm
        ---------
        1. Encode ALL images at once through encode_images() (single SigLIP call).
        2. Project through mm_projector.
        3. Append per-row image_newline tokens if mm_use_im_newline_token=True.
           Result: each image → tokens_per_image feature vectors.
        4. Embed text tokens (clamp -200 → 0 so embed_tokens doesn't crash).
        5. For each sample, find contiguous -200 blocks and scatter image features
           in-place.  Block size == tokens_per_image ensures alignment with step 3.

        Assumptions
        -----------
        * connector_only=True  (no SVA cross-attention; connector-only S-stage).
        * image_aspect_ratio compatible with fixed tokens_per_image per image.
          For anyres mode use preprocessing that produces a fixed grid (e.g. pad
          to 384×384 before SigLIP so every image gives exactly si_side_len² tokens).
        * pixel_values rows are in the same order as the -200 blocks in input_ids
          when iterating samples left-to-right, images left-to-right.
        """
        cfg = self.config
        si_token_len: int = cfg.si_token_len                           # 729
        si_side_len: int = int(si_token_len ** 0.5)                    # 27
        mm_use_newline: bool = getattr(cfg, "mm_use_im_newline_token", True)
        tokens_per_image: int = (
            si_side_len * (si_side_len + 1) if mm_use_newline else si_token_len
        )  # 756 with newlines, 729 without

        batch_size = input_ids.shape[0]

        # --- 1. Encode images ------------------------------------------------
        image_aux_features_list = self.encode_images([pixel_values])
        image_features = image_aux_features_list[0]  # (total_imgs, si_token_len, vis_dim)

        # --- 2. Project -------------------------------------------------------
        proj_weight_dtype = self.get_model().mm_projector[0].weight.dtype
        image_features = self.get_model().mm_projector(
            image_features.to(proj_weight_dtype)
        ).to(pixel_values.dtype)  # (total_imgs, si_token_len, hidden_dim)

        # --- 3. Append newline tokens ----------------------------------------
        if mm_use_newline:
            total_imgs, _, hidden_dim = image_features.shape
            image_features = image_features.view(
                total_imgs, si_side_len, si_side_len, hidden_dim
            )
            newline = self.get_model().image_newline.to(image_features.dtype)  # (hidden_dim,)
            newline_exp = newline.view(1, 1, 1, hidden_dim).expand(
                total_imgs, si_side_len, 1, hidden_dim
            )
            image_features = torch.cat([image_features, newline_exp], dim=2)
            # shape: (total_imgs, 27, 28, hidden_dim)
            image_features = image_features.view(total_imgs, -1, hidden_dim)
            # shape: (total_imgs, 756, hidden_dim)

        # tokens_per_image must match the pre-expanded -200 block size in input_ids
        assert image_features.shape[1] == tokens_per_image, (
            f"After newline expansion image_features has {image_features.shape[1]} tokens/image "
            f"but rollout manager pre-expanded to {tokens_per_image} IMAGE_TOKEN_INDEX tokens. "
            "Check si_token_len / mm_use_im_newline_token consistency."
        )

        # --- 4. Text token embeddings (safe clamp avoids OOB on -200) ---------
        safe_ids = input_ids.clamp(min=0)
        token_embeds = self.get_model().embed_tokens(safe_ids)  # (bs, seq_len, hidden_dim)

        # --- 5. Scatter visual features into image positions ------------------
        img_global_idx = 0
        for b in range(batch_size):
            image_positions = (input_ids[b] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            if image_positions.numel() == 0:
                continue
            n_img_tokens = image_positions.numel()
            if n_img_tokens % tokens_per_image != 0:
                raise ValueError(
                    f"Sample {b}: {n_img_tokens} IMAGE_TOKEN_INDEX tokens, "
                    f"not divisible by tokens_per_image={tokens_per_image}. "
                    "The rollout manager and model config must agree on this value."
                )
            n_images = n_img_tokens // tokens_per_image
            for img_i in range(n_images):
                start = image_positions[img_i * tokens_per_image].item()
                token_embeds[b, start : start + tokens_per_image] = (
                    image_features[img_global_idx].to(token_embeds.dtype)
                )
                img_global_idx += 1

        return token_embeds  # (bs, seq_len, hidden_dim)

    # ------------------------------------------------------------------
    # NFP helper
    # ------------------------------------------------------------------
    def _compute_nfp_targets(
        self,
        nfp_pixel_values: torch.Tensor,  # (bs * max_turns, C, H, W) – padded next-frame images
        nfp_loss_masks: torch.Tensor,    # (bs, seq_len) – 1 at action-end positions for NFP
    ) -> torch.Tensor:                   # nfp_tgt_embeds (bs, seq_len, vis_dim)
        """
        Compute sparse NFP target embeddings.

        nfp_pixel_values uses a fixed-length padding scheme: each sample
        contributes exactly `max_turns` images (real images followed by
        zero-pads).  After extract_multi_modal_inputs() concatenates across
        the batch, total = bs * max_turns.

        For each batch sample b:
          - Its images are at nfp_pixel_values[b*nfp_per_sample : (b+1)*nfp_per_sample]
          - Positions where nfp_loss_masks[b] == 1 map (in order) to the
            first len(positions) images of this sample.
          - Zero-padded images are never indexed because they have no
            matching 1 in the loss mask.
        """
        import torch.nn.functional as F

        miv_token_len: int = getattr(self.config, "miv_token_len", 64)
        miv_side_len: int = int(miv_token_len ** 0.5)       # 8

        # Encode next-frame images through SigLIP (no projection for NFP targets)
        feat_list = self.encode_images([nfp_pixel_values])
        raw_feats = feat_list[0]   # (bs * max_turns, si_token_len, vis_dim)
        feature_side_len = int(raw_feats.shape[1] ** 0.5)  # 27

        # Downsample to MIV resolution
        total_acts, _, vis_dim = raw_feats.shape
        if miv_side_len != feature_side_len:
            raw_feats = raw_feats.view(total_acts, feature_side_len, feature_side_len, vis_dim)
            raw_feats = raw_feats.permute(0, 3, 1, 2).float()  # (N, C, H, W)
            raw_feats = F.interpolate(
                raw_feats, size=(miv_side_len, miv_side_len),
                mode="bilinear", align_corners=False,
            ).to(nfp_pixel_values.dtype)
            raw_feats = raw_feats.permute(0, 2, 3, 1).flatten(1, 2)  # (N, 64, vis_dim)

        bs, seq_len = nfp_loss_masks.shape
        nfp_per_sample = nfp_pixel_values.shape[0] // bs  # = max_turns

        nfp_tgt_embeds = torch.zeros(
            bs, seq_len, vis_dim,
            device=nfp_pixel_values.device, dtype=nfp_pixel_values.dtype,
        )

        for b in range(bs):
            sample_feats = raw_feats[b * nfp_per_sample : (b + 1) * nfp_per_sample]
            positions = (nfp_loss_masks[b] == 1).nonzero(as_tuple=True)[0]
            for i, pos in enumerate(positions):
                if i >= sample_feats.shape[0]:
                    break
                # Store MIV features starting at this position
                end = min(pos.item() + miv_token_len, seq_len)
                actual = end - pos.item()
                nfp_tgt_embeds[b, pos : pos + actual] = sample_feats[i, :actual]

        return nfp_tgt_embeds  # (bs, seq_len, vis_dim)

    # ------------------------------------------------------------------
    # forward – GPU-compatible, replaces XLA-only base class dispatch
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # --- multimodal inputs (from verl multi_modal_inputs dict) ---
        pixel_values: Optional[torch.FloatTensor] = None,
        # --- NFP inputs (optional; populated by rollout manager when use_nfp=True) ---
        nfp_pixel_values: Optional[torch.FloatTensor] = None,
        nfp_loss_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass:
          1. If pixel_values provided and input_ids has -200 tokens →
             compute inputs_embeds via _embed_multimodal_batch().
          2. If no images, embed text tokens directly.
          3. Run through CambrianQwenModel (= Qwen2 model; SVA only if connector_only=False).
          4. LM head.
          5. Optionally add NFP loss when nfp_pixel_values present.

        NOTE: connector_only=False (SVA cross-attention between decoder layers) is NOT
        supported in this adapter because the original SVA code uses XLA-specific
        scatter kernels.  All standard Cambrian-S checkpoints use connector_only=True.
        """
        if getattr(self.config, "connector_only", True) is False:
            raise NotImplementedError(
                "CambrianForCausalLMAdapter does not support connector_only=False (SVA). "
                "All Cambrian-S checkpoints trained with cambrians_7b_s*.sh use connector_only=True."
            )

        # ------------------------------------------------------------------
        # 1.  Build inputs_embeds
        # ------------------------------------------------------------------
        if inputs_embeds is None:
            has_image_tokens = (
                input_ids is not None and (input_ids == IMAGE_TOKEN_INDEX).any()
            )
            if pixel_values is not None and has_image_tokens:
                inputs_embeds = self._embed_multimodal_batch(input_ids, pixel_values)
                input_ids = None
            elif input_ids is not None:
                # text-only or no -200 tokens in this micro-batch
                safe_ids = input_ids.clamp(min=0)
                inputs_embeds = self.get_model().embed_tokens(safe_ids)
                input_ids = None

        # ------------------------------------------------------------------
        # 2.  NFP target embeddings (sparse, same seq_len as inputs_embeds)
        # ------------------------------------------------------------------
        nfp_tgt_embeds = None
        if (
            nfp_pixel_values is not None
            and nfp_loss_masks is not None
            and getattr(self.config, "nfp_head", False)
        ):
            nfp_tgt_embeds = self._compute_nfp_targets(nfp_pixel_values, nfp_loss_masks)

        # ------------------------------------------------------------------
        # 3.  Qwen2 model body (bypass Cambrian's custom dispatch entirely)
        # ------------------------------------------------------------------
        output_attentions = (
            output_attentions if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Call CambrianQwenModel.forward() directly (it is Qwen2Model + optional SVA).
        # Since connector_only=True the SVA block is skipped inside the model.
        outputs = self.model(
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

        # ------------------------------------------------------------------
        # 4.  LM head + optional language modelling loss
        # ------------------------------------------------------------------
        logits = self.lm_head(hidden_states).float()

        lm_loss = None
        if labels is not None:
            from torch.nn import CrossEntropyLoss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            lm_loss = CrossEntropyLoss()(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        # ------------------------------------------------------------------
        # 5.  NFP auxiliary loss
        # ------------------------------------------------------------------
        total_loss = lm_loss
        if nfp_tgt_embeds is not None and hasattr(self.model, "nfp_head"):
            nfp_outputs = self.model.nfp_head(hidden_states)  # (bs, seq_len, vis_dim)
            nfp_mse, nfp_cos = self.nfp_loss(nfp_outputs, nfp_tgt_embeds, nfp_loss_masks)
            nfp_mse = nfp_mse * getattr(self.config, "nfp_mse_loss_weight", 1.0)
            nfp_cos = nfp_cos * getattr(self.config, "nfp_cosine_loss_weight", 1.0)
            if total_loss is not None:
                # verl reads loss as a scalar; wrap extra info in a tuple so callers
                # that check isinstance(loss, tuple) can log sub-losses
                total_loss = (total_loss + nfp_mse + nfp_cos, total_loss, nfp_mse, nfp_cos)
            else:
                total_loss = nfp_mse + nfp_cos

        if not return_dict:
            out = (logits,) + outputs[1:]
            return (total_loss,) + out if total_loss is not None else out

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # generate – accepts pixel_values from verl multi_modal_inputs
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        HF-rollout-compatible generate().

        verl's hf_rollout.py (after our patch) calls:
            model.generate(input_ids=…, attention_mask=…, position_ids=…,
                           **multi_modal_inputs)
        where multi_modal_inputs = {"pixel_values": tensor}.

        We pre-compute inputs_embeds here and call Qwen2ForCausalLM.generate()
        with inputs_embeds instead of input_ids so Qwen2's incremental decoding
        works correctly (no -200 tokens leak into the KV-cache path).
        """
        # Drop any NFP kwargs that may accidentally be passed
        kwargs.pop("nfp_pixel_values", None)
        kwargs.pop("nfp_loss_masks", None)
        kwargs.pop("image_sizes", None)

        if pixel_values is not None and input_ids is not None:
            if (input_ids == IMAGE_TOKEN_INDEX).any():
                inputs_embeds = self._embed_multimodal_batch(input_ids, pixel_values)
            else:
                inputs_embeds = self.get_model().embed_tokens(input_ids.clamp(min=0))
        elif input_ids is not None:
            inputs_embeds = self.get_model().embed_tokens(input_ids.clamp(min=0))
        else:
            inputs_embeds = kwargs.pop("inputs_embeds", None)
            if inputs_embeds is None:
                raise ValueError("generate() requires either input_ids or inputs_embeds.")

        # Delegate to Qwen2ForCausalLM.generate() (grandparent).
        # We pass BOTH input_ids and inputs_embeds:
        #  - input_ids is needed so output.sequences has shape
        #    (bs, prompt_length + response_length), which hf_rollout.py expects.
        #  - inputs_embeds overrides the first-step embedding lookup, so the -200
        #    sentinel tokens in input_ids are never embedded (they are already
        #    replaced by visual features in inputs_embeds).
        #  - On decode steps 1+, prepare_inputs_for_generation drops inputs_embeds
        #    and uses only the newly generated token's input_id.
        from transformers import Qwen2ForCausalLM
        return Qwen2ForCausalLM.generate(
            self,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        """
        Called by HF on every decode step after the first.

        On step 0: inputs_embeds are passed in (pre-computed in generate()).
        On step 1+: past_key_values are populated, only the new token matters.
        We drop pixel_values here – they were already consumed in generate().
        """
        kwargs.pop("pixel_values", None)
        kwargs.pop("nfp_pixel_values", None)
        kwargs.pop("nfp_loss_masks", None)
        # Use Qwen2ForCausalLM's prepare_inputs_for_generation (grandparent)
        from transformers import Qwen2ForCausalLM
        return Qwen2ForCausalLM.prepare_inputs_for_generation(
            self, input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# 3.  Re-register the adapter so AutoModelForCausalLM.from_pretrained()
#     returns a CambrianForCausalLMAdapter instance instead of the base class.
# ---------------------------------------------------------------------------
AutoModelForCausalLM.register(CambrianQwenConfig, CambrianForCausalLMAdapter, exist_ok=True)

print(
    "[cambrian_register] CambrianForCausalLMAdapter registered with AutoModelForCausalLM "
    f"(CAMBRIAN_SRC={CAMBRIAN_SRC})"
)
