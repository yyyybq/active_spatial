#!/bin/bash
set -e

echo "========================================="
echo " Qwen3-VL Environment Setup (vagen3)"
echo "========================================="

# Check Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$python_version" != "3.10" ]]; then
    echo "⚠️  Warning: Expected Python 3.10, got Python $python_version"
fi

# --- Step 1: PyTorch 2.9.0 + CUDA 12.9 ---
echo ""
echo "[1/6] Installing PyTorch 2.9.0 + CUDA 12.9 ..."
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu129

# --- Step 2: vLLM 0.12.0 (Qwen3-VL support) ---
echo ""
echo "[2/6] Installing vLLM 0.12.0 ..."
pip install vllm==0.12.0

# --- Step 3: verl (--no-deps to avoid version conflicts) ---
echo ""
echo "[3/6] Installing verl (editable, --no-deps) ..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAGEN_ROOT="$(dirname "$SCRIPT_DIR")"
VERL_DIR="$(dirname "$VAGEN_ROOT")/verl"

if [ -d "$VERL_DIR" ]; then
    pip install -e "$VERL_DIR" --no-deps
else
    echo "⚠️  verl directory not found at $VERL_DIR"
    echo "    Please clone verl first: git clone https://github.com/JamesKrW/verl.git $(dirname "$VAGEN_ROOT")/verl"
    exit 1
fi

# --- Step 4: Project dependencies ---
echo ""
echo "[4/6] Installing project dependencies ..."
pip install qwen-vl-utils mathruler matplotlib flask
pip install gymnasium "gymnasium[toy-text]" gym gym-sokoban together
pip install omegaconf hydra-core pandas
pip install tensordict peft pyarrow pybind11 pylatexenc wandb codetiming torchdata

# --- Step 5: Install vagen (--no-deps) ---
echo ""
echo "[5/6] Installing vagen (editable, --no-deps) ..."
pip install -e "$VAGEN_ROOT" --no-deps

# --- Step 6: flash-attn notice ---
echo ""
echo "[6/6] flash-attn check ..."
echo "⚠️  flash-attn is NOT installed (no prebuilt wheel for PyTorch 2.9 + Python 3.10)."
echo "    verl includes a built-in fallback (flash_attn_fallback), so training works without it."

# --- Verification ---
echo ""
echo "========================================="
echo " Verifying installation ..."
echo "========================================="
python -c "
import torch
print(f'  torch:          {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count:      {torch.cuda.device_count()}')
import transformers
print(f'  transformers:   {transformers.__version__}')
import vllm
print(f'  vllm:           {vllm.__version__}')
import verl
print(f'  verl:           OK')
import vagen
print(f'  vagen:          OK')
from transformers import Qwen3VLForConditionalGeneration
print(f'  Qwen3VL class:  OK')
print()
print('✅ All imports passed! vagen3 environment is ready for Qwen3-VL experiments.')
"
