#!/bin/bash
set -e

echo "Starting installation process..."

# echo "Initializing git submodules..."
# git submodule update --init

# echo "Installing verl package..."
# cd verl
# pip install -e .
# cd ../

echo "Installing vagen dependencies..."
pip install 'qwen-vl-utils'
pip install 'mathruler'
pip install 'matplotlib'
pip install 'flask'


echo "Installing flash-attn from pre-built wheel..."
# Download and install pre-compiled wheel for PyTorch 2.6.0 + CUDA 12.4
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
pip install "$FLASH_ATTN_WHEEL" || {
    echo "⚠️  Failed to install flash-attn from wheel, trying with --no-build-isolation..."
    pip install flash-attn==2.7.4.post1 --no-build-isolation || {
        echo "⚠️  Flash-attn installation failed. The code will fall back to standard attention."
    }
}

echo "Installing vagen package..."
pip install -e .

echo "Installing Sokoban dependencies"
pip install 'gym'
pip install 'gym-sokoban'

echo "Installing Frozenlake dependencies"
pip install 'gymnasium'
pip install "gymnasium[toy-text]"

pip install together # together ai api for process reward
echo "Installation complete, to install dependencies for other environments, refer to env/readme"
