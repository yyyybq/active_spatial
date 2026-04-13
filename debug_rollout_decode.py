"""
Debug script: Reproduce the EXACT decode pipeline used in training
to see what the model generates and why action_is_valid=0.

Runs WITHOUT FSDP, just HF generate(), same as hf_rollout.py logic.
"""
import os, sys, torch, re
sys.path.insert(0, '/scratch/by2593/project/Active_Spatial/cambrian-s')
os.environ['CAMBRIAN_SRC'] = '/scratch/by2593/project/Active_Spatial/cambrian-s'

DEVICE = "cuda:0"  # With CUDA_VISIBLE_DEVICES=4, this maps to physical GPU 4
MODEL_PATH = "/scratch/by2593/hf_cache/cambrian-s-7b"

# ---- 1. Load model ----
import vagen.models.cambrian_register  # registers adapter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.add_tokens(["<image>"], special_tokens=True)
image_token_id = tokenizer.convert_tokens_to_ids("<image>")
print(f"image_token_id: {image_token_id}")
print(f"pad_token_id: {tokenizer.pad_token_id}, eos_token_id: {tokenizer.eos_token_id}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
)
model = model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}")

# ---- 2. Create test image and preprocess EXACTLY as CambrianRolloutManager does ----
from PIL import Image
import numpy as np
from cambrian.mm_utils import expand2square, select_best_resolution, resize_and_pad_image, divide_to_patches

img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
test_image = Image.fromarray(img_array)

# Get image processor from model
image_processor = model.get_model().vision_tower_aux_list[0].image_processor
target_resolution = image_processor.crop_size['height']
image_mean = getattr(image_processor, 'image_mean', [0.48145466, 0.4578275, 0.40821073])
bg_color = tuple(int(x * 255) for x in image_mean)

# Anyres preprocessing (same as CambrianRolloutManager._preprocess_single_image)
anyres_max_subimages = 9
snapshot = expand2square(test_image, bg_color).resize((target_resolution, target_resolution))
possible_resolutions = [
    (int(w * target_resolution), int(h * target_resolution))
    for w in range(1, anyres_max_subimages + 1)
    for h in range(1, anyres_max_subimages + 1)
    if w * h <= anyres_max_subimages
]
best_resolution = select_best_resolution(test_image.size, possible_resolutions)
padded_img = resize_and_pad_image(test_image, best_resolution, bg_color)
patches = divide_to_patches(padded_img, target_resolution)
subimages = [snapshot] + patches
pvs = [image_processor.preprocess(si, return_tensors='pt')['pixel_values'][0] for si in subimages]
pixel_values = torch.stack(pvs).to(device=DEVICE, dtype=torch.bfloat16)
num_subimages = pixel_values.shape[0]
print(f"pixel_values shape: {pixel_values.shape} (num_subimages={num_subimages})")

# ---- 3. Build prompt EXACTLY as CambrianRolloutManager does ----
# Using env's actual system prompt
from vagen.env.active_spatial.prompt import system_prompt as build_system_prompt
sys_prompt = build_system_prompt(step_translation=0.2, step_rotation_deg=10.0)
print(f"System prompt length: {len(sys_prompt)} chars")

user_msg = """[Initial Observation]:
<image>
Current camera pose: <1.20, 1.50, 0.80, 45.0\u00b0, 0.0\u00b0, 0.0\u00b0>
Task: Move the camera to the front view of the chair, about 2.00 meters away.

Navigate to reach the specified view of the target object. Use the available actions to position and orient the camera correctly.

Respond in the following format:
<think>Your reasoning process</think>
<action>action1|action2|...|</action>

You can take up to 5 actions per step.

Example: <think>I can see I'm looking at a room with furniture. The target is to reach the front view of the chair. I should move forward and adjust my angle.</think><action>move_forward|move_forward|turn_right|</action>"""

messages = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": user_msg},
]

# Apply chat template (same as CambrianRolloutManager._single_recording_to_prompt)
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\n=== Chat template applied ===")
print(f"Prompt text (last 300 chars): ...{text[-300:]}")

# ---- 4. Tokenize and expand <image> tokens ----
raw_prompt_ids = tokenizer.encode(text, add_special_tokens=False)
print(f"Raw prompt tokens: {len(raw_prompt_ids)}")

# Expand <image> tokens to IMAGE_TOKEN_INDEX=-200 (one per sub-image * tokens_per_image)
IMAGE_TOKEN_INDEX = -200
si_token_len = 729
si_side_len = 27
tokens_per_image = si_side_len * (si_side_len + 1)  # 756 with newlines
print(f"tokens_per_image: {tokens_per_image}")

new_ids = []
img_idx = 0
for tid in raw_prompt_ids:
    if tid == image_token_id:
        new_ids.extend([IMAGE_TOKEN_INDEX] * (num_subimages * tokens_per_image))
        img_idx += 1
    else:
        new_ids.append(tid)

print(f"Expanded prompt tokens: {len(new_ids)} (added {len(new_ids) - len(raw_prompt_ids)} image tokens for {num_subimages} sub-images)")

# ---- 5. Create input tensors (same as CambrianRolloutManager._generate_input_for_rollout) ----
input_ids = torch.tensor([new_ids], dtype=torch.long).to(DEVICE)
attention_mask = torch.ones_like(input_ids)
position_ids = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0).to(DEVICE)

prompt_length = input_ids.size(1)
print(f"prompt_length: {prompt_length}")

# ---- 6. Generate (same as hf_rollout.py._generate_minibatch) ----
response_length = 512
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id

generation_config = GenerationConfig(temperature=0.7, top_p=0.95, top_k=0)

print("\n=== Generating... ===")
with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            max_new_tokens=response_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            generation_config=generation_config,
            output_scores=False,
            return_dict_in_generate=True,
            use_cache=True,
            pixel_values=pixel_values,
        )

seq = output.sequences
print(f"Generated sequence shape: {seq.shape}")

# ---- 7. Extract response (same as hf_rollout.py) ----
sequence_length = prompt_length + response_length
delta_length = sequence_length - seq.shape[1]
if delta_length > 0:
    delta_tokens = torch.full((1, delta_length), pad_token_id, device=seq.device, dtype=seq.dtype)
    seq = torch.cat((seq, delta_tokens), dim=1)

response = seq[:, prompt_length:]
print(f"Response shape: {response.shape}")

# ---- 8. Decode (same as CambrianRolloutManager.rollout_loop) ----
responses_str = tokenizer.batch_decode(response, skip_special_tokens=True)
decoded_text = responses_str[0]

print(f"\n=== DECODED RESPONSE (skip_special_tokens=True) ===")
print(repr(decoded_text[:1000]))

# Also decode without skipping special tokens
responses_str_full = tokenizer.batch_decode(response, skip_special_tokens=False)
print(f"\n=== DECODED RESPONSE (skip_special_tokens=False) ===")
print(repr(responses_str_full[0][:1000]))

# ---- 9. Test parsing (same as env.step) ----
from vagen.env.active_spatial.utils import parse_free_think, parse_actions, check_actions

print(f"\n=== PARSING ===")
ft = parse_free_think(decoded_text)
print(f"parse_free_think ok: {ft['ok']}")
print(f"think: {repr(ft['think'][:200])}")
print(f"actions_blob: {repr(ft['actions_blob'][:200])}")

if ft['ok']:
    ok, parsed_actions = parse_actions(ft['actions_blob'])
    print(f"parse_actions ok: {ok}")
    if ok:
        print(f"Parsed actions: {[a.name for a in parsed_actions]}")
        print(f"check_actions: {check_actions(parsed_actions)}")
    else:
        print("Actions parsing failed!")
else:
    print("Format NOT matched! This is why action_is_valid=0")
    # Try to find what the model actually outputs
    print(f"\nFirst 500 chars of response: {decoded_text[:500]}")
    # Check if there's partial match
    m = re.search(r"<think>", decoded_text, re.IGNORECASE)
    if m:
        print(f"Found <think> at position {m.start()}")
    m = re.search(r"</think>", decoded_text, re.IGNORECASE)
    if m:
        print(f"Found </think> at position {m.start()}")
    m = re.search(r"<action>", decoded_text, re.IGNORECASE)
    if m:
        print(f"Found <action> at position {m.start()}")
    m = re.search(r"</action>", decoded_text, re.IGNORECASE)
    if m:
        print(f"Found </action> at position {m.start()}")
