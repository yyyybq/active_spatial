<h1 align="center">Active Spatial Intelligence</h1>
<h3 align="center"><b>Training VLM Agents for 3D Spatial Navigation with Potential Field Rewards</b></h3>

<p align="center">
  <b>Baiqiao Yin</b>, <b>Jiatong Ji</b>
</p>

<p align="center">
  <a href="https://github.com/yyyybq/active_spatial"><img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub"></a>
</p>

This repository implements **Active Spatial Intelligence** - a visual navigation environment for training Vision-Language Model (VLM) agents to navigate 3D indoor scenes and reach target camera poses. Built on top of the [VAGEN](https://github.com/RAGEN-AI/VAGEN) framework.

<div align="center">
  <img src="https://github.com/user-attachments/assets/b364e6c9-4c2c-46d0-afca-ee42f271c59c" width="200"/>
  <img src="https://github.com/user-attachments/assets/65662eb0-9440-4555-9436-8b9272791ac4" width="200"/>
</div>

## Key Features

- **9 Task Types**: Comprehensive spatial reasoning tasks including positioning, orientation control, and multi-object constraints
- **Spatial Potential Field Rewards**: Dense reward signals based on geometric scoring functions
- **3D Gaussian Splatting Rendering**: Real-time photorealistic rendering using gsplat
- **Collision & Visibility Detection**: Wall-aware collision and occlusion checking
- **Multi-Turn RL Training**: Leverages VAGEN's PPO implementation for effective agent training

## Task Types

| Task Type | Description | Target Region |
|-----------|-------------|---------------|
| `absolute_positioning` | Move to fixed distance from object | circle |
| `delta_control` | Reach target with minimal steps | circle |
| `screen_occupancy` | Object fills specific screen percentage | annular_ring |
| `equidistance` | Equal distance from multiple objects | apollonius_circle |
| `projective_relations` | Align objects in specific projection | line |
| `occlusion_alignment` | Create specific occlusion pattern | line |
| `fov_inclusion` | Include multiple objects in view | polygon |
| `size_distance_invariance` | Maintain relative visual sizes | annular_ring |
| `centering` | Center object in camera view | point/circle |

## Installation

We provide two environment setups: **vagen** for Qwen2.5-VL experiments and **vagen3** for Qwen3-VL experiments.

### Option A: Qwen2.5-VL Environment (vagen)

```bash
# Create conda environment
conda create -n vagen python=3.10 -y
conda activate vagen

# Install verl (RL framework)
git clone https://github.com/JamesKrW/verl.git
cd verl && pip install -e . && cd ..

# Install Active Spatial
git clone https://github.com/yyyybq/active_spatial.git
cd active_spatial
bash scripts/install.sh

# Install rendering dependencies (ninja, gsplat, plyfile, ply_gaussian_loader)
# NOTE: ply_gaussian_loader is NOT a pip package — it is copied from ViewSuite.
# ViewSuite must be cloned alongside active_spatial.
git clone <ViewSuite_repo_url> ../ViewSuite  # if not already cloned
bash scripts/examples/vagen_base/active_spatial/install_render_deps.sh
```

### Option B: Qwen3-VL Environment (vagen3)

Qwen3-VL requires newer versions of vllm (≥0.11.0), transformers (≥4.57.3), and PyTorch (≥2.9.0). Use the one-click install script:

```bash
# Create conda environment
conda create -n vagen3 python=3.10 -y
conda activate vagen3

# Install Active Spatial with Qwen3-VL support
bash scripts/install_qwen3vl.sh

# Install rendering dependencies (ninja, gsplat, plyfile, ply_gaussian_loader)
# NOTE: ply_gaussian_loader is NOT a pip package — it is copied from ViewSuite.
bash scripts/examples/vagen_base/active_spatial/install_render_deps.sh
```

<details>
<summary>Manual installation steps (if the script fails)</summary>

```bash
conda create -n vagen3 python=3.10 -y
conda activate vagen3

# 1. Install PyTorch 2.9.0 with CUDA 12.9
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu129

# 2. Install vllm 0.12.0 (brings transformers ≥4.57.3 automatically)
pip install vllm==0.12.0

# 3. Install verl (--no-deps to avoid downgrading vllm/transformers)
cd verl && pip install -e . --no-deps && cd ..

# 4. Install project dependencies
pip install qwen-vl-utils mathruler matplotlib flask
pip install gymnasium "gymnasium[toy-text]" gym gym-sokoban together
pip install omegaconf hydra-core pandas
pip install tensordict peft pyarrow pybind11 pylatexenc wandb codetiming torchdata

# 5. Install vagen (--no-deps)
pip install -e . --no-deps

# 6. Install rendering dependencies
# NOTE: ply_gaussian_loader is NOT a pip package.
# Use the install script instead:
bash scripts/examples/vagen_base/active_spatial/install_render_deps.sh
```

> **Note:** flash-attn has no prebuilt wheel for PyTorch 2.9 + Python 3.10. verl includes a built-in fallback (`flash_attn_fallback`) so training works without it.

</details>

## Quick Start

### 1. Prepare Data

```bash
# Generate training data for a scene
python data_gen/active_spatial_pipeline/run_pipeline.py \
    --scene_id YOUR_SCENE_ID \
    --gs_root /path/to/InteriorGS
```

### 2. Run Training

```bash
# Start PPO training
bash scripts/examples/vagen_base/active_spatial/run.sh
```

### Configuration

Edit `scripts/examples/vagen_base/active_spatial/env_config_balanced.yaml`:

```yaml
env1:
    env_name: active_spatial
    env_config:
        render_backend: local  # or "client" for distributed rendering
        gs_root: "/path/to/InteriorGS"
        enable_potential_field: true
        enable_collision_detection: true
        enable_visibility_check: true
```

## Project Structure

```
├── vagen/env/active_spatial/       # Core environment implementation
│   ├── env.py                      # Main environment class
│   ├── spatial_potential_field.py  # Reward computation (9 task types)
│   ├── collision_detector.py       # Wall & object collision
│   ├── visibility_checker.py       # FOV & occlusion checking
│   └── render/                     # 3DGS rendering (local/client)
├── data_gen/active_spatial_pipeline/ # Data generation tools
├── scripts/examples/               # Training scripts
└── docs/                          # Documentation
```

## Documentation

- [Quick Start Guide](./QUICK_START.md) - Detailed setup and training instructions
- [Data Preparation](./DATA_PREPARATION_GUIDE.md) - How to prepare training data
- [Local Rendering Setup](./LOCAL_RENDERING_SETUP.md) - GPU rendering configuration
- [Environment README](./vagen/env/active_spatial/README.md) - Environment details

## Actions

The agent can perform the following actions:
- `move_forward` / `move_backward`: Translate camera
- `turn_left` / `turn_right`: Rotate camera (yaw)
- `look_up` / `look_down`: Rotate camera (pitch)
- `done`: Signal task completion

## Reward System

The environment uses a **Spatial Potential Field** for dense rewards:
- Each task type has a geometric scoring function
- Score = f(position, orientation) ∈ [0, 1]
- Reward = Δscore (positive when improving)
- Additional rewards: format (+0.2), success (+1.0), collision penalty (-0.15)

## Acknowledgement

This project is built upon:
- [VAGEN](https://github.com/RAGEN-AI/VAGEN) - Multi-turn RL framework for VLM agents
- [verl](https://github.com/volcengine/verl) - Volcano Engine RL framework
- [gsplat](https://github.com/nerfstudio-project/gsplat) - 3D Gaussian Splatting rendering

## License

This project follows the [Apache 2.0 License](./LICENSE).

## Citation

If you use this code, please cite:

```bibtex
@misc{yin2026activespatial,
  title={Active Spatial Intelligence: Training VLM Agents for 3D Spatial Navigation},
  author={Baiqiao Yin and Jiatong Ji},
  year={2026},
  url={https://github.com/yyyybq/active_spatial}
}
```
