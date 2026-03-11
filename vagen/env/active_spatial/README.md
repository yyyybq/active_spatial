# ViewSuite Environments Integration for VAGEN

This document describes the integration of two environments from ViewSuite into the VAGEN training framework:

1. **Active Spatial Intelligence** (`active_spatial`) - A navigation task where the agent moves a camera to reach a target pose relative to an object
2. **View Spatial Bench** (`view_spatial`) - A spatial reasoning QA task where the agent answers questions about 3D scenes

## Environment Overview

### Active Spatial Intelligence (`active_spatial`)

This environment involves navigating a camera in a 3D scene to reach a specific target view of an object (e.g., moving to the front/back/left/right view of a chair).

**Actions:**
- `move_forward`: Move the camera forward
- `move_backward`: Move the camera backward
- `turn_left`: Rotate camera left (yaw)
- `turn_right`: Rotate camera right (yaw)
- `look_up`: Tilt camera upward (pitch)
- `look_down`: Tilt camera downward (pitch)
- `done`: Signal task completion

**Rewards:**
- Format reward: +0.2 for correct response format
- Pose reward: Continuous reward based on improvement in position and orientation toward target
- Success reward: +1.0 for reaching the target pose

### View Spatial Bench (`view_spatial`)

This environment tests spatial reasoning by asking questions about 3D scenes. It supports two modes:

#### No-Tool Mode (Single-turn QA)
The agent answers the question directly based on provided images.

**Actions:**
- `answer(X)`: Submit answer where X is A, B, C, or D

#### Tool Mode (Multi-turn with Exploration)
The agent can explore the scene before answering.

**Actions:**
- `move_forward`, `move_backward`, `turn_left`, `turn_right`, `look_up`, `look_down`: Camera navigation
- `query_pose(view_name)`: Get the pose of a named view
- `select_view(view_name)`: Switch to a named view
- `get_view(tx,ty,tz,rx,ry,rz)`: Set camera pose directly (angles in degrees)
- `answer(X)`: Submit final answer

**Rewards:**
- Format reward: +0.2 for correct response format
- Answer reward: +0.8 for correct answer

## Installation

### Prerequisites

1. Install VAGEN dependencies:
```bash
cd /path/to/VAGEN
pip install -e .
```

2. Install ViewSuite (for rendering support):
```bash
cd /path/to/ViewSuite
pip install -e .
```

3. Install additional dependencies:
```bash
pip install scipy pillow numpy
```

## Dataset Format

### Active Spatial Dataset (JSONL)

Each line should contain:
```json
{
    "object_label": "chair",
    "preset": "front",
    "distance": 2.0,
    "init_camera": {
        "intrinsics": [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "extrinsics": [[...], [...], [...], [...]]
    },
    "target_position": [x, y, z],
    "camera_params": {
        "forward": [fx, fy, fz]
    }
}
```

### View Spatial Dataset (JSONL)

Each line should contain:
```json
{
    "question": "What is the relative position of object A to object B?",
    "choices": "A. Left\nB. Right\nC. Above\nD. Below",
    "answer": "A",
    "image_path": ["path/to/image1.png", "path/to/image2.png"],
    "scene_id": "scene001",
    "sample_id": "sample001",
    "question_type": "relative_direction",
    "image_camera_pose": {
        "view_0": {
            "camera_intrinsics": [[...]],
            "camera_extrinsics": [[...]]
        }
    }
}
```

## Usage

### 1. Start the Rendering Service (Required for Tool Mode)

For environments that need rendering, start the ViewSuite rendering server:

```bash
# For Gaussian Splatting rendering
cd /path/to/ViewSuite
python -m view_suite.interiorGS.service.server --port 8766 --gs_root /path/to/gs_assets

# For ScanNet rendering
python -m view_suite.scannet.service.server --port 8766 --scannet_root /path/to/scannet
```

### 2. Start the VAGEN Service

```bash
cd /path/to/VAGEN
python -m vagen.server.server server.port=5000
```

### 3. Configure the Environment

Edit the environment configuration file:

For Active Spatial:
```yaml
# scripts/examples/vagen_base/active_spatial/env_config.yaml
env1:
    env_name: active_spatial
    env_config:
        jsonl_path: "/path/to/your/dataset.jsonl"
        render_backend: client
        client_url: "ws://127.0.0.1:8766/render"
        gs_root: "/path/to/gs_assets"
        step_translation: 0.1
        step_rotation_deg: 5.0
    train_size: 5000
    test_size: 64
```

For View Spatial:
```yaml
# scripts/examples/vagen_base/view_spatial/env_config.yaml
env1:
    env_name: view_spatial
    env_config:
        jsonl_path: "/path/to/your/dataset.jsonl"
        use_tools: false  # Set to true for exploration mode
    train_size: 5000
    test_size: 64
```

### 4. Run Training

```bash
# For Active Spatial
cd /path/to/VAGEN
bash scripts/examples/vagen_base/active_spatial/run.sh

# For View Spatial
bash scripts/examples/vagen_base/view_spatial/run.sh
```

## Configuration Options

### Active Spatial Environment Config

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `jsonl_path` | str | "" | Path to dataset JSONL file |
| `dataset_root` | str | "" | Root path for relative image paths |
| `render_backend` | str | "client" | "local" or "client" |
| `gs_root` | str | "" | Path to Gaussian Splatting assets |
| `client_url` | str | "ws://127.0.0.1:8766/render" | Rendering server URL |
| `step_translation` | float | 0.1 | Translation step size (meters) |
| `step_rotation_deg` | float | 5.0 | Rotation step size (degrees) |
| `format_reward` | float | 0.2 | Reward for correct format |
| `success_reward` | float | 1.0 | Reward for reaching target |
| `max_episode_steps` | int | 50 | Maximum steps per episode |

### View Spatial Environment Config

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `jsonl_path` | str | "" | Path to dataset JSONL file |
| `use_tools` | bool | False | Enable exploration tools |
| `render_backend` | str | "client" | "local" or "client" |
| `step_translation` | float | 0.3 | Translation step size (meters) |
| `step_rotation_deg` | float | 30.0 | Rotation step size (degrees) |
| `format_reward` | float | 0.2 | Reward for correct format |
| `answer_reward` | float | 0.8 | Reward for correct answer |
| `max_episode_steps` | int | 10 | Maximum steps per episode |

## Response Format

Both environments use a free-think format for LLM responses:

```
<think>Your reasoning process here</think>
<action>action1|action2|...|</action>
```

### Examples

Active Spatial:
```
<think>I can see the chair in front of me. I need to move around it to get to the front view. I'll turn right first.</think>
<action>turn_right|move_forward|move_forward|</action>
```

View Spatial (No-Tool):
```
<think>Looking at the image, object A is positioned to the left of object B.</think>
<action>answer(A)|</action>
```

View Spatial (Tool):
```
<think>I need to see the scene from a different angle. Let me select view_1 to get a better perspective.</think>
<action>select_view(view_1)|answer(B)|</action>
```

## File Structure

```
vagen/env/
├── active_spatial/
│   ├── __init__.py
│   ├── env.py                    # Main environment class
│   ├── env_config.py             # Configuration dataclass
│   ├── service.py                # Batch service wrapper
│   ├── service_config.py
│   ├── prompt.py                 # System prompts and templates
│   ├── utils.py                  # Utilities (parsing, pose math, etc.)
│   ├── visibility_checker.py     # Visibility and occlusion checking (with wall support)
│   ├── collision_detector.py     # Collision detection (with wall support)
│   ├── spatial_potential_field.py  # Potential field scoring for 8 task types
│   └── visualize_potential_field_refactored.py  # Heatmap visualization tool
└── view_spatial/
    ├── __init__.py
    ├── env.py
    ├── env_config.py
    ├── service.py
    ├── service_config.py
    ├── prompt.py
    └── utils.py

scripts/examples/vagen_base/
├── active_spatial/
│   ├── env_config.yaml
│   └── run.sh
└── view_spatial/
    ├── env_config.yaml
    └── run.sh
```

## Core Modules

### Wall-Based Visibility and Collision Detection

The environment uses `structure.json` (room profiles) and `labels.json` (objects) to properly handle:

1. **Wall Occlusion** (`visibility_checker.py`): Prevents agent from "seeing through walls"
2. **Wall Collision** (`collision_detector.py`): Prevents agent from "walking through walls"

#### Key Data Sources

| File | Content | Usage |
|------|---------|-------|
| `structure.json` | Room profiles (wall boundaries), doors/windows | Wall segments for occlusion & collision |
| `labels.json` | Furniture bounding boxes | Object occlusion & collision |

#### How Walls Are Extracted

```python
# Room profile edges become wall segments
# Doors are excluded (they're openings)
for room in structure_data['rooms']:
    profile = room['profile']  # List of 2D points
    # Each edge (profile[i] -> profile[i+1]) is a potential wall
    # Skip edges that overlap with doors
```

#### VisibilityChecker

```python
from visibility_checker import VisibilityChecker

vis_checker = VisibilityChecker()
vis_checker.load_scene(scene_path)  # Loads both labels.json AND structure.json

# Check if camera can see target
occlusion = vis_checker.check_occlusion(
    camera_position=np.array([x, y, z]),
    target_center=np.array([tx, ty, tz]),
    target_label="tv"
)
# occlusion = 0.0 (fully visible) to 1.0 (fully blocked)
```

#### CollisionDetector

```python
from collision_detector import CollisionDetector

col_detector = CollisionDetector()
col_detector.load_scene(scene_path)  # Loads both labels.json AND structure.json

# Check if position has collision
result = col_detector.check_collision(
    position=np.array([x, y, z]),
    previous_position=np.array([px, py, pz])  # Optional, for segment check
)
# result.has_collision: bool
# result.collision_type: "object" | "wall" | "floor" | "ceiling" | "none"
```

### Spatial Potential Field

The `SpatialPotentialField` class computes scores for 9 task types:

| Task Type | Description | Region Type |
|-----------|-------------|-------------|
| `absolute_positioning` | Move to fixed distance from object | circle |
| `delta_control` | Reach target with minimal steps | circle |
| `screen_occupancy` | Object fills specific screen percentage | annular_ring |
| `equidistance` | Equal distance from multiple objects | apollonius_circle |
| `projective_relations` | Align objects in specific projection | line |
| `occlusion_alignment` | Create specific occlusion pattern | line |
| `fov_inclusion` | Include multiple objects in view | polygon |
| `size_distance_invariance` | Maintain relative visual sizes | annular_ring |
| `centering` | Center object in camera view | point/circle |

## Visualization Tool

### Potential Field Heatmap Visualization

The `visualize_potential_field_refactored.py` script generates heatmaps showing valid camera positions and their scores for each task type.

**Key Features:**
- **Reuses shared modules**: Uses `VisibilityChecker` and `CollisionDetector` directly (no code duplication)
- **Ensures consistency**: Visualization shows exactly what the environment computes
- **Wall occlusion**: Gray areas = outside target room or blocked by walls
- **Multi-object support**: Shows all target objects for complex tasks

#### Usage

```bash
cd /path/to/VAGEN

# Generate heatmaps for all 8 task types
python vagen/env/active_spatial/visualize_potential_field_refactored.py \
    --scene_id 0267_840790 \
    --gs_root /path/to/InteriorGS \
    --output_dir ./potential_field_visualizations \
    --grid_size 100

# Options:
#   --scene_id    Scene ID (default: 0267_840790)
#   --gs_root     Root directory for scene data (structure.json, labels.json)
#   --data_dir    Directory containing train_data_*.jsonl files
#   --output_dir  Output directory for heatmap images
#   --grid_size   Heatmap resolution (default: 100)
#   --show        Display heatmaps interactively
```

#### Output

For each task type, generates:
- `heatmap_{task_type}_refactored.png` with two subplots:
  1. **Visibility Check**: Gray=outside room, Red=blocked, Green=visible
  2. **Potential Field Score**: Color gradient from red (low) to green (high)

#### Example

```bash
# Generate heatmaps for scene 0267_840790
python vagen/env/active_spatial/visualize_potential_field_refactored.py \
    --scene_id 0267_840790 \
    --gs_root /scratch/by2593/project/Active_Spatial/InteriorGS \
    --output_dir ./potential_field_visualizations_with_render_v3
```

Output files:
```
potential_field_visualizations/
├── heatmap_absolute_positioning_refactored.png
├── heatmap_delta_control_refactored.png
├── heatmap_screen_occupancy_refactored.png
├── heatmap_equidistance_refactored.png
├── heatmap_projective_relations_refactored.png
├── heatmap_occlusion_alignment_refactored.png
├── heatmap_fov_inclusion_refactored.png
└── heatmap_size_distance_invariance_refactored.png
```

### Enhanced Visualization with 3DGS Rendering

The `visualize_with_rendering.py` script extends the basic heatmap visualization with:
1. **3D Gaussian Splatting rendering** for both initial and target camera views
2. **Random sampling** of N examples per task type
3. **Combined visualization** showing rendered views alongside the potential field heatmap

**Key Features:**
- Renders actual camera views using 3D Gaussian Splatting
- Shows initial camera position (looking at target object)
- Shows planned target view (at optimal position with correct orientation)
- Displays potential field heatmap with navigation path

#### Usage

```bash
cd /path/to/VAGEN

# Generate visualizations with rendering for all task types
python vagen/env/active_spatial/visualize_with_rendering.py \
    --scene_id 0267_840790 \
    --gs_root /path/to/InteriorGS \
    --data_dir data_gen/active_spatial_pipeline/output \
    --output_dir vagen/env/active_spatial/potential_field_heatmaps_with_render_v2 \
    --grid_size 100 \
    --num_samples_per_task 5

# Options:
#   --scene_id             Scene ID (default: 0267_840790)
#   --gs_root              Root directory for scene data
#   --data_dir             Directory containing train_data_*.jsonl files
#   --output_dir           Output directory for visualization images
#   --grid_size            Heatmap resolution (default: 100)
#   --num_samples_per_task Number of random samples per task type (default: 10)
#   --render_width         Rendered image width (default: 640)
#   --render_height        Rendered image height (default: 480)
#   --seed                 Random seed for reproducibility (default: 42)
```

#### Output Layout

Each visualization contains:
1. **Top Left**: Initial camera view (rendered)
2. **Top Right**: Planned target view (rendered, with score)
3. **Bottom**: Full potential field heatmap with:
   - Room boundaries and walls
   - Target objects highlighted
   - Initial camera position (red marker)
   - Planned target position (green star)
   - Navigation path

#### Output Directory Structure

```
potential_field_heatmaps_with_render_v2/
├── absolute_positioning/
│   ├── absolute_positioning_00_bookshelf.png
│   ├── absolute_positioning_01_sofa.png
│   └── ...
├── centering/
├── delta_control/
├── equidistance/
├── fov_inclusion/
├── occlusion_alignment/
├── projective_relations/
├── screen_occupancy/
└── size_distance_invariance/
```

## Troubleshooting

### Common Issues

1. **Rendering service not responding**
   - Ensure the ViewSuite rendering server is running
   - Check the `client_url` in your configuration

2. **Dataset not found**
   - Verify `jsonl_path` is correct
   - Check `dataset_root` for relative image paths

3. **Import errors**
   - Ensure ViewSuite is installed: `pip install -e /path/to/ViewSuite`
   - Check scipy is installed: `pip install scipy`

4. **CUDA out of memory**
   - Reduce `train_batch_size` in the run script
   - Enable gradient checkpointing
   - Use parameter offloading

5. **Agent sees through walls / walks through walls**
   - Ensure `structure.json` exists in scene directory
   - Check that `load_scene()` is called before checking visibility/collision
   - Verify wall segments are loaded: should see `Loaded X rooms, Y wall segments`

6. **Visualization shows incorrect occlusion**
   - Use `visualize_potential_field_refactored.py` (reuses shared checkers)
   - Avoid `visualize_potential_field_with_scene.py` (deprecated, has duplicate logic)

## License

This integration follows the licenses of both VAGEN and ViewSuite projects.
