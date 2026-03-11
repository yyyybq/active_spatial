#!/usr/bin/env python3
"""
Enhanced Visualization with 3D Gaussian Splatting Rendering

This script extends visualize_potential_field_refactored.py to:
1. Randomly sample N examples per task type (not just 1)
2. Render both initial and target viewpoints using 3D Gaussian Splatting

Usage:
    python visualize_with_rendering.py \
        --scene_id 0267_840790 \
        --gs_root /scratch/by2593/project/Active_Spatial/InteriorGS \
        --data_dir data_gen/active_spatial_pipeline/output \
        --output_dir vagen/env/active_spatial/potential_field_heatmaps_with_render \
        --grid_size 100 \
        --num_samples_per_task 10
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
from typing import Dict, Any, List, Tuple, Optional
import sys
import random
from collections import defaultdict
from PIL import Image
import asyncio

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import from existing visualization module
from visualize_potential_field_refactored import (
    SceneVisualizer,
    PathPlanner,
    load_tasks_from_jsonl,
    compute_heatmap_using_shared_checkers,
    plot_heatmap,
    _add_task_annotations,
    compute_forward_direction,
)
from spatial_potential_field import create_potential_field
from env_config import ActiveSpatialEnvConfig

# Import renderer
try:
    from render.unified_renderer import UnifiedRenderGS
except ImportError:
    from vagen.env.active_spatial.render.unified_renderer import UnifiedRenderGS


def sample_tasks_by_type(tasks: List[Dict], num_samples: int = 10, seed: int = 42) -> Dict[str, List[Dict]]:
    """
    Group tasks by type and randomly sample N examples per type.
    
    Args:
        tasks: List of all task dictionaries
        num_samples: Number of samples per task type
        seed: Random seed for reproducibility
        
    Returns:
        Dict mapping task_type -> list of sampled tasks
    """
    random.seed(seed)
    
    # Group by task type
    tasks_by_type = defaultdict(list)
    for task in tasks:
        task_type = task.get("task_type", "unknown")
        tasks_by_type[task_type].append(task)
    
    # Sample from each group
    sampled = {}
    for task_type, task_list in tasks_by_type.items():
        if len(task_list) <= num_samples:
            sampled[task_type] = task_list
        else:
            sampled[task_type] = random.sample(task_list, num_samples)
        print(f"  {task_type}: {len(task_list)} total -> {len(sampled[task_type])} sampled")
    
    return sampled


def extrinsics_c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """Convert camera-to-world to world-to-camera extrinsics."""
    return np.linalg.inv(c2w)


def look_at_matrix(camera_position: np.ndarray, look_at_point: np.ndarray,
                   up_vector: np.ndarray = None) -> np.ndarray:
    """
    Generate camera extrinsics (camera-to-world) matrix using look-at formulation.
    
    This follows the same convention as the existing setup_camera in gs_render.py:
    - X axis: -right
    - Y axis: up
    - Z axis: forward (toward target)
    """
    if up_vector is None:
        up_vector = np.array([0, 0, 1], dtype=np.float64)
    
    position = np.array(camera_position, dtype=np.float64)
    target = np.array(look_at_point, dtype=np.float64)
    up = np.array(up_vector, dtype=np.float64)
    
    # Forward direction: from camera to target
    forward = target - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-10:
        forward = np.array([1, 0, 0], dtype=np.float64)
    else:
        forward = forward / forward_norm
    
    # Right vector
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-10:
        # forward is parallel to up, choose alternative up
        alt_up = np.array([0, 1, 0], dtype=np.float64)
        right = np.cross(forward, alt_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-10:
            right = np.array([1, 0, 0], dtype=np.float64)
        else:
            right = right / right_norm
    else:
        right = right / right_norm
    
    # Recalculate up to ensure orthogonality
    up_corrected = np.cross(right, forward)
    up_corrected = up_corrected / np.linalg.norm(up_corrected)
    
    # Build camera-to-world matrix (same convention as gs_render.py setup_camera)
    # X = -right, Y = up, Z = forward
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = -right
    c2w[:3, 1] = up_corrected
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    
    return c2w


async def render_view(
    renderer: UnifiedRenderGS,
    camera_pos: np.ndarray,
    camera_forward: np.ndarray,
    intrinsics: np.ndarray,
    width: int = 640,
    height: int = 480,
) -> Optional[Image.Image]:
    """
    Render a view from a camera position with specified forward direction.
    
    Args:
        renderer: Initialized renderer
        camera_pos: Camera position [x, y, z]
        camera_forward: Camera forward direction [dx, dy, dz] (normalized or unnormalized)
        intrinsics: 3x3 camera intrinsics matrix
        width: Image width
        height: Image height
        
    Returns:
        PIL Image or None if rendering failed
    """
    try:
        # Normalize forward direction
        forward = np.array(camera_forward, dtype=np.float64)
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-10:
            forward = np.array([1, 0, 0], dtype=np.float64)
        else:
            forward = forward / forward_norm
        
        # Compute look-at point from position and forward direction
        look_at_point = np.array(camera_pos) + forward * 2.0  # Look 2m ahead
        
        # Create camera-to-world matrix
        c2w = look_at_matrix(camera_pos, look_at_point)
        # Convert to world-to-camera for renderer
        w2c = extrinsics_c2w_to_w2c(c2w)
        
        # Ensure intrinsics is 3x3
        K = np.array(intrinsics)
        if K.shape == (4, 4):
            K = K[:3, :3]
        elif K.shape != (3, 3):
            # Create default intrinsics
            fx = fy = width / (2 * np.tan(np.radians(60) / 2))
            K = np.array([
                [fx, 0, width/2],
                [0, fy, height/2],
                [0, 0, 1]
            ])
        
        image = await renderer.render_image_from_cam_param(
            camera_intrinsics=K,
            camera_extrinsics=w2c,
            width=width,
            height=height,
        )
        return image
    except Exception as e:
        print(f"    Render error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_async(coro):
    """Helper to run async coroutine in sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


def create_combined_visualization(
    X: np.ndarray,
    Y: np.ndarray,
    valid_mask: np.ndarray,
    final_scores: np.ndarray,
    metadata: Dict[str, Any],
    scene: SceneVisualizer,
    init_image: Optional[Image.Image],
    target_image: Optional[Image.Image],
    task: Dict[str, Any],
    output_path: str,
):
    """
    Create a combined visualization with heatmap and rendered views.
    
    Layout:
    +----------------+----------------+
    |   Init View    |  Target View   |
    +----------------+----------------+
    |         Potential Field          |
    |          Heatmap (wide)          |
    +----------------------------------+
    """
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('score_cmap', colors, N=256)
    
    vis_reasons = metadata.get("visibility_reasons", np.zeros_like(valid_mask, dtype=int))
    target_room_idx = metadata.get("target_room_idx", None)
    
    # Get room bounds for focused view
    room_bounds = None
    if target_room_idx is not None and target_room_idx < len(scene.room_profiles):
        room_profile = scene.room_profiles[target_room_idx]
        room_x_min, room_y_min = room_profile.min(axis=0)
        room_x_max, room_y_max = room_profile.max(axis=0)
        room_padding = 0.5
        room_bounds = (
            room_x_min - room_padding, room_x_max + room_padding,
            room_y_min - room_padding, room_y_max + room_padding
        )
    
    masked_scores = np.where(valid_mask > 0.5, final_scores, np.nan)
    collision_overlay = np.where(vis_reasons == 4, 0.5, np.nan)
    
    task_type = task.get("task_type", "unknown").replace("_", " ").title()
    task_desc = task.get("task_description", "")
    object_label = task.get("object_label", "object")
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2], wspace=0.1, hspace=0.2)
    
    # ==================== Top Left: Initial View ====================
    ax_init = fig.add_subplot(gs[0, 0])
    if init_image is not None:
        ax_init.imshow(init_image)
        ax_init.set_title("Initial Camera View", fontsize=12, fontweight='bold')
    else:
        ax_init.text(0.5, 0.5, "Render Failed", ha='center', va='center', fontsize=14)
        ax_init.set_title("Initial Camera View (Failed)", fontsize=12)
    ax_init.axis('off')
    
    # ==================== Top Right: Target View ====================
    ax_target = fig.add_subplot(gs[0, 1])
    if target_image is not None:
        ax_target.imshow(target_image)
        planned_score = metadata.get("planned_final_score", 0)
        ax_target.set_title(f"Planned Target View (score={planned_score:.2f})", 
                           fontsize=12, fontweight='bold')
    else:
        ax_target.text(0.5, 0.5, "Render Failed", ha='center', va='center', fontsize=14)
        ax_target.set_title("Target View (Failed)", fontsize=12)
    ax_target.axis('off')
    
    # ==================== Bottom: Potential Field Heatmap ====================
    ax_heatmap = fig.add_subplot(gs[1, :])
    
    im = ax_heatmap.pcolormesh(X, Y, masked_scores, cmap=cmap, shading='auto', vmin=0, vmax=1)
    ax_heatmap.pcolormesh(X, Y, collision_overlay, cmap='gray', shading='auto', alpha=0.7, vmin=0, vmax=1)
    
    if room_bounds:
        ax_heatmap.set_xlim(room_bounds[0], room_bounds[1])
        ax_heatmap.set_ylim(room_bounds[2], room_bounds[3])
        scene.plot_on_axes(ax_heatmap, alpha=0.5, show_labels=True,
                          xlim=(room_bounds[0], room_bounds[1]),
                          ylim=(room_bounds[2], room_bounds[3]))
    else:
        scene.plot_on_axes(ax_heatmap, alpha=0.5, show_labels=True)
    
    _add_task_annotations(ax_heatmap, metadata)
    
    # Add path info to title
    planned_path = metadata.get("planned_path", [])
    num_steps = len(planned_path) - 1 if planned_path else 0
    
    ax_heatmap.set_title(
        f"Task: {task_type} | Object: {object_label}\n"
        f"{task_desc}\n"
        f"Planned Path: {num_steps} steps",
        fontsize=11, fontweight='bold'
    )
    ax_heatmap.set_xlabel('X (m)', fontsize=10)
    ax_heatmap.set_ylabel('Y (m)', fontsize=10)
    ax_heatmap.set_aspect('equal', adjustable='box')
    ax_heatmap.grid(True, alpha=0.3, linestyle=':')
    
    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.6, pad=0.02)
    cbar.set_label('Potential Field Score', fontsize=9)
    
    # Add legend
    ax_heatmap.legend(loc='upper right', fontsize=8)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {output_path}")


async def visualize_tasks_with_rendering(
    jsonl_path: str,
    gs_root: str,
    scene_id: str,
    output_dir: str = "./heatmaps_with_render",
    grid_size: int = 100,
    num_samples_per_task: int = 10,
    render_width: int = 640,
    render_height: int = 480,
    config: Optional[ActiveSpatialEnvConfig] = None,
):
    """
    Generate visualizations for sampled tasks with 3D Gaussian rendering.
    
    Args:
        config: Environment config (uses default if None, ensuring consistency with training)
    """
    # Use provided config or create default (same as training)
    if config is None:
        config = ActiveSpatialEnvConfig()
    
    print(f"Loading tasks from {jsonl_path}...")
    tasks = load_tasks_from_jsonl(jsonl_path)
    print(f"Total tasks: {len(tasks)}")
    
    print(f"\nSampling {num_samples_per_task} examples per task type...")
    sampled_tasks = sample_tasks_by_type(tasks, num_samples=num_samples_per_task)
    
    print(f"\nLoading scene with config (floor_height={config.collision_floor_height}, ceiling_height={config.collision_ceiling_height})...")
    scene = SceneVisualizer(gs_root, scene_id, config=config)
    
    print(f"\nInitializing 3D Gaussian Renderer...")
    renderer = UnifiedRenderGS(
        render_backend="local",
        gs_root=gs_root,
        scene_id=scene_id,
    )
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create potential field with SAME CONFIG as training
    field = create_potential_field({
        "position_weight": config.potential_field_position_weight,
        "orientation_weight": config.potential_field_orientation_weight,
        "max_distance": config.max_distance,
        "fov_horizontal": config.fov_horizontal,
        "fov_vertical": config.fov_vertical,
    })
    
    total_count = 0
    for task_type, task_list in sampled_tasks.items():
        print(f"\n{'='*60}")
        print(f"Processing task type: {task_type} ({len(task_list)} samples)")
        print(f"{'='*60}")
        
        task_dir = output_dir_path / task_type
        task_dir.mkdir(exist_ok=True)
        
        for idx, task in enumerate(task_list):
            total_count += 1
            print(f"\n  [{idx+1}/{len(task_list)}] Processing sample {idx}...")
            
            try:
                # Compute heatmap
                X, Y, valid_mask, final_scores, metadata = compute_heatmap_using_shared_checkers(
                    field, scene, task, grid_size=grid_size, padding=0.3,
                )
                
                # Get camera info
                init_camera = task.get("init_camera", {})
                intrinsics = np.array(init_camera.get("intrinsics", [
                    [320, 0, 320], [0, 320, 240], [0, 0, 1]
                ]))
                
                # Get initial camera position and forward direction
                init_pos = metadata.get("init_camera_pos")
                
                # Get object center as fallback for forward direction
                target_region = task.get("target_region", {})
                params = target_region.get("params", {})
                object_center = params.get("object_center", params.get("center", [0, 0, 1.5]))
                if len(object_center) == 2:
                    object_center = [object_center[0], object_center[1], 1.5]
                object_center = np.array(object_center)
                
                # Render initial view (looking at object center)
                init_image = None
                if init_pos is not None:
                    print(f"    Rendering initial view...")
                    # For initial view, compute forward direction toward object
                    init_forward = object_center - init_pos
                    init_image = await render_view(
                        renderer, init_pos, init_forward, intrinsics,
                        render_width, render_height
                    )
                
                # Get planned target position and forward direction
                planned_target_pos = metadata.get("planned_target_pos")
                planned_target_forward = metadata.get("planned_target_forward")
                
                # Render target view using planned forward direction
                target_image = None
                if planned_target_pos is not None:
                    print(f"    Rendering target view...")
                    # Use planned_target_forward if available, otherwise compute from position to object
                    if planned_target_forward is not None:
                        target_forward = np.array(planned_target_forward)
                        print(f"      Using planned_target_forward: {target_forward}")
                    else:
                        target_forward = object_center - planned_target_pos
                        print(f"      Using direction to object center: {target_forward}")
                    
                    target_image = await render_view(
                        renderer, planned_target_pos, target_forward, intrinsics,
                        render_width, render_height
                    )
                
                # Create combined visualization
                object_label = task.get("object_label", "object").replace(" ", "_")[:20]
                output_path = task_dir / f"{task_type}_{idx:02d}_{object_label}.png"
                
                create_combined_visualization(
                    X, Y, valid_mask, final_scores, metadata,
                    scene, init_image, target_image, task, str(output_path)
                )
                
            except Exception as e:
                print(f"    Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
    
    # Cleanup renderer
    await renderer.close()
    
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Total visualizations: {total_count}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Spatial Potential Field with 3D Gaussian Rendering"
    )
    parser.add_argument("--scene_id", type=str, default="0267_840790")
    parser.add_argument("--gs_root", type=str, 
                       default="/scratch/by2593/project/Active_Spatial/InteriorGS")
    parser.add_argument("--data_dir", type=str, 
                       default="data_gen/active_spatial_pipeline/output")
    parser.add_argument("--output_dir", type=str, 
                       default="vagen/env/active_spatial/potential_field_heatmaps_with_render")
    parser.add_argument("--grid_size", type=int, default=100)
    parser.add_argument("--num_samples_per_task", type=int, default=10,
                       help="Number of samples to visualize per task type")
    parser.add_argument("--render_width", type=int, default=640)
    parser.add_argument("--render_height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Build jsonl path
    jsonl_path = Path(args.data_dir) / f"train_data_{args.scene_id}.jsonl"
    
    if not jsonl_path.exists():
        print(f"Error: JSONL file not found at {jsonl_path}")
        return
    
    # Run async visualization
    asyncio.run(visualize_tasks_with_rendering(
        jsonl_path=str(jsonl_path),
        gs_root=args.gs_root,
        scene_id=args.scene_id,
        output_dir=args.output_dir,
        grid_size=args.grid_size,
        num_samples_per_task=args.num_samples_per_task,
        render_width=args.render_width,
        render_height=args.render_height,
    ))


if __name__ == "__main__":
    main()
