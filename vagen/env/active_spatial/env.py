# Active Spatial Intelligence Environment
# This environment wraps ViewSuite's active spatial intelligence environment for VAGEN training.
#
# Rendering Architecture:
# ========================
# ViewSuite uses a client-server architecture for 3D rendering:
#
#   VAGEN (Client)                     Render Server (GPU)
#   ┌─────────────────┐                ┌─────────────────┐
#   │ ActiveSpatialEnv│  WebSocket     │ GS Render Server│
#   │ UnifiedRenderGS │ ─────────────▶ │ GaussianRenderer│
#   │ (client mode)   │ ◀───────────── │ (gsplat+CUDA)   │
#   └─────────────────┘   PNG images   └─────────────────┘
#
# Two rendering modes:
# 1. "client": Connect to remote render server via WebSocket (recommended for training)
# 2. "local": Render locally using GPU (requires gsplat, good for debugging)
# 3. None: Use pre-rendered images from dataset (no real-time rendering)
#
# Reward System:
# ==============
# This environment uses a Spatial Potential Field for dense rewards:
# - Each task type has a geometric scoring function
# - Score = f(position, orientation) ∈ [0, 1]
# - Reward = Δscore (positive when improving, negative when regressing)

import time
import asyncio
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import os

from vagen.env.base.base_env import BaseEnv
from vagen.env.utils.context_utils import convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP

from .env_config import ActiveSpatialEnvConfig
from .prompt import system_prompt, init_observation_template, action_template, format_prompt
from .utils import (
    ViewManipulator,
    count_lines,
    read_jsonl_line_by_index,
    resolve_rel_image,
    safe_open_rgb,
    parse_free_think,
    parse_actions,
    check_actions,
    calculate_pose_score_smooth,
    c2w_extrinsic_to_se3,
    c2w_se3_to_extrinsic,
    format_pose6_deg,
    fallback_K,
    # Added from ActiveVLN alignment
    compute_translation_distance,
    compute_approach_reward,
    compute_progress_reward,
    is_goal_reached,
)
from .spatial_potential_field import SpatialPotentialField, ScoreResult, create_potential_field
from .collision_detector import CollisionDetector, CollisionResult, create_collision_detector
from .visibility_checker import VisibilityChecker, VisibilityResult, create_visibility_checker, compute_visibility_reward


def _run_async(coro):
    """Helper to run async coroutine in sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop is not None and loop.is_running():
        # We're in an async context, use nest_asyncio or run in executor
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class ActiveSpatialEnv(BaseEnv):
    """
    Active Spatial Intelligence Environment for VAGEN.
    
    This environment involves navigating a camera in a 3D scene to reach
    a target pose relative to an object.
    
    Control logging verbosity with environment variable:
        export ACTIVE_SPATIAL_ENV_VERBOSE=1  # Enable verbose logging
    """
    
    # Class variable for logging control
    VERBOSE = os.environ.get('ACTIVE_SPATIAL_ENV_VERBOSE', '0') == '1'
    
    # Available discrete actions
    ACTION_LOOKUP = {
        "move_forward": 1,
        "move_backward": 2,
        "turn_left": 3,
        "turn_right": 4,
        "look_up": 5,
        "look_down": 6,
        "done": 7,
    }
    
    def __init__(self, config: ActiveSpatialEnvConfig):
        """Initialize the Active Spatial environment."""
        super().__init__()
        self.config = config
        
        # Dataset setup
        self.jsonl_path = Path(config.jsonl_path) if config.jsonl_path else None
        self.dataset_root = config.dataset_root if config.dataset_root else None
        
        # Count dataset lines
        if self.jsonl_path and self.jsonl_path.is_file():
            if config.total_lines > 0:
                self.total_lines = config.total_lines
            else:
                self.total_lines = count_lines(self.jsonl_path)
        else:
            self.total_lines = 0
        
        # View engine for camera manipulation
        self.view_engine = ViewManipulator(
            step_translation=config.step_translation,
            step_rotation_deg=config.step_rotation_deg,
            world_up_axis="Z",
        )
        
        # Rendering client (will be initialized when needed)
        self.renderer = None
        self._renderer_initialized = False
        
        # Episode state
        self.current_item: Optional[Dict[str, Any]] = None
        self.episode_done: bool = False
        self.camera_intrinsics: Optional[np.ndarray] = None
        self._current_step = 0
        self._max_episode_steps = config.max_episode_steps
        self._episode_start_time = 0
        self.total_reward = 0
        self.reward = 0
        self.prev_score = 0.0
        self.prev_pos = None
        self.prev_distance = None  # For progress reward
        
        # Spatial Potential Field for task-driven dense rewards
        self.potential_field: Optional[SpatialPotentialField] = None
        if config.enable_potential_field:
            self.potential_field = create_potential_field({
                "position_weight": config.potential_field_position_weight,
                "orientation_weight": config.potential_field_orientation_weight,
                "max_distance": config.max_distance,
            })
        
        # Task information (loaded from dataset item)
        self.current_task: Optional[Dict[str, Any]] = None
        self.prev_potential_score: float = 0.0  # Previous score for delta reward
        
        # Collision Detector for preventing camera from passing through objects
        self.collision_detector: Optional[CollisionDetector] = None
        if config.enable_collision_detection:
            self.collision_detector = create_collision_detector({
                "camera_radius": config.collision_camera_radius,
                "floor_height": config.collision_floor_height,
                "ceiling_height": config.collision_ceiling_height,
                "safety_margin": config.collision_safety_margin,
                "enable_object_collision": True,
                "enable_boundary_collision": True,
            })
        
        # Collision tracking for episode
        self.collision_count: int = 0
        self.last_collision_result: Optional[CollisionResult] = None
        
        # Visibility Checker for ensuring target is actually visible
        self.visibility_checker: Optional[VisibilityChecker] = None
        if config.enable_visibility_check:
            self.visibility_checker = create_visibility_checker({
                "fov_horizontal": config.fov_horizontal,
                "fov_vertical": config.fov_vertical,
                "min_visible_size": config.min_visible_screen_coverage,
            })
        
        # Visibility tracking
        self.prev_visibility: Optional[VisibilityResult] = None
        self.target_object_info: Optional[Dict[str, Any]] = None  # Target object bbox etc.
        
        # Multi-frame spatial prior
        self.spatial_prior_images: List[Image.Image] = []  # Cached prior images for current episode
        self.spatial_prior_poses: List[Dict[str, Any]] = []  # Camera poses for prior images
        
        # Store format prompt function
        self.format_prompt_func = format_prompt.get(config.prompt_format, format_prompt["free_think"])
        
        # Get parse function
        self.parse_func = PARSE_FUNC_MAP.get(config.prompt_format, self._default_parse_func)
    
    def _default_parse_func(self, response: str, **kwargs) -> Dict[str, Any]:
        """Default parse function for action parsing."""
        ft = parse_free_think(response)
        if not ft["ok"]:
            return {
                "format_correct": False,
                "actions": [],
                "think": "",
                "llm_raw_response": response,
            }
        
        sep = kwargs.get("action_sep", "|")
        max_actions = kwargs.get("max_actions", 5)
        
        ok, parsed_actions = parse_actions(ft["actions_blob"], sep=sep)
        if not ok:
            return {
                "format_correct": False,
                "actions": [],
                "think": ft["think"],
                "llm_raw_response": response,
            }
        
        if not check_actions(parsed_actions):
            return {
                "format_correct": False,
                "actions": [],
                "think": ft["think"],
                "llm_raw_response": response,
            }
        
        # Extract action names
        actions = [a.name for a in parsed_actions[:max_actions]]
        
        return {
            "format_correct": True,
            "actions": actions,
            "think": ft["think"],
            "llm_raw_response": response,
        }
    
    async def _init_renderer(self):
        """Initialize the rendering client if not already done."""
        if self._renderer_initialized:
            return
        
        render_backend = self.config.render_backend
        
        if render_backend is None or render_backend == "none":
            # No real-time rendering, use pre-rendered images from dataset
            self.renderer = None
            self._renderer_initialized = True
            if self.VERBOSE:
                print("[ActiveSpatialEnv] Using pre-rendered images from dataset (no real-time rendering)")
            return
        
        if render_backend in ("client", "local"):
            try:
                # Import local render module (adapted from ViewSuite)
                from .render.unified_renderer import UnifiedRenderGS
                
                # Get GPU device
                gpu_device = self.config.get('gpu_device', None)
                
                self.renderer = UnifiedRenderGS(
                    render_backend=render_backend,
                    gs_root=self.config.gs_root,
                    client_url=self.config.client_url,
                    client_origin=self.config.client_origin,
                    scene_id=None,  # Will be set when loading episode
                    gpu_device=gpu_device,  # Pass GPU device
                )
                self._renderer_initialized = True
                if self.VERBOSE:
                    print(f"[ActiveSpatialEnv] Renderer initialized with backend: {render_backend}")
                    
                    if render_backend == "client":
                        print(f"[ActiveSpatialEnv] Client URL: {self.config.client_url}")
                    elif render_backend == "local":
                        print(f"[ActiveSpatialEnv] GS Root: {self.config.gs_root}")
                        if gpu_device is not None:
                            print(f"[ActiveSpatialEnv] GPU Device: cuda:{gpu_device}")
                    
            except ImportError as e:
                print(f"[ActiveSpatialEnv] Warning: Renderer not available: {e}")
                print("[ActiveSpatialEnv] Falling back to pre-rendered images from dataset")
                self.renderer = None
                self._renderer_initialized = True
        else:
            raise ValueError(f"Unknown render_backend: {render_backend}. Use 'client', 'local', or None.")
    
    def reset(self, seed: int = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to a new episode.
        
        Args:
            seed: Random seed for episode selection
            
        Returns:
            Tuple of (observation, info)
        """
        # Initialize renderer on first reset
        if not self._renderer_initialized:
            _run_async(self._init_renderer())
        
        if seed is None:
            seed = 0
        
        idx = seed % max(1, self.total_lines)
        
        # Reset episode state
        self.episode_done = False
        self._current_step = 0
        self._episode_start_time = time.time()
        self.total_reward = 0
        self.reward = 0
        self.prev_score = 0.0
        self.prev_pos = None
        self.prev_distance = None  # Will be set after loading target pose
        self.prev_potential_score = 0.0  # Reset potential field score
        
        # Load episode data
        if self.jsonl_path and self.total_lines > 0:
            self.current_item = read_jsonl_line_by_index(self.jsonl_path, idx)
        else:
            # Mock data for testing
            self.current_item = {
                "scene_id": "scene_test",
                "object_label": "chair",
                "preset": "front",
                "distance": 2.0,
                "init_camera": {
                    "intrinsics": fallback_K().tolist(),
                    "extrinsics": np.eye(4).tolist(),
                },
                "target_position": [0, 0, 2],
                "camera_params": {"forward": [0, 0, 1]},
                # Mock task information for potential field
                "task_type": "absolute_positioning",
                "task_params": {"target_distance": 2.0},
                "target_region": {
                    "type": "circle",
                    "params": {"center": [0, 0], "radius": 2.0, "object_center": [0, 0, 1]},
                    "sample_point": [0, -2, 1.5],
                    "sample_forward": [0, 1, 0],
                },
            }
        
        item = self.current_item
        
        # Set scene for renderer (important for Gaussian Splatting)
        scene_id = item.get("scene_id", None)
        if self.renderer is not None and scene_id:
            self.renderer.set_scene(scene_id)
        
        # Load collision detection data for this scene (cached - only reloads if scene changes)
        if self.collision_detector is not None and scene_id and self.config.gs_root:
            self.collision_detector.load_scene_from_gs_root(self.config.gs_root, scene_id)
        
        # Load visibility checker data for this scene (cached - only reloads if scene changes)
        if self.visibility_checker is not None and scene_id and self.config.gs_root:
            scene_path = Path(self.config.gs_root) / scene_id
            self.visibility_checker.load_scene(scene_path, scene_id=scene_id)
        
        # Reset collision tracking
        self.collision_count = 0
        self.last_collision_result = None
        
        # Reset visibility tracking
        self.prev_visibility = None
        
        # Load target object info for visibility checking
        self.target_object_info = item.get("target_object", None)
        
        # Initialize camera from data
        init_cam = item.get("init_camera", {})
        if init_cam:
            K = np.array(init_cam.get("intrinsics", fallback_K()), dtype=np.float64)
            E = np.array(init_cam.get("extrinsics", np.eye(4)), dtype=np.float64)
            self.camera_intrinsics = K
            self.view_engine.reset(E)
        else:
            self.camera_intrinsics = fallback_K()
            self.view_engine.reset(None)
        
        # Store initial position for scoring
        self.prev_pos = self.view_engine.get_pose()[:3, 3].copy()
        
        # Load task information for potential field scoring
        self.current_task = {
            "task_type": item.get("task_type", "absolute_positioning"),
            "task_params": item.get("task_params", {}),
            "target_region": item.get("target_region", {}),
        }
        
        # Initialize potential field score
        if self.potential_field is not None and self.current_task.get("target_region"):
            curr_pose = self.view_engine.get_pose()
            curr_pos = curr_pose[:3, 3]
            curr_forward = -curr_pose[:3, 2]  # Camera -Z is forward
            
            initial_score = self.potential_field.compute_score(
                camera_position=curr_pos,
                camera_forward=curr_forward,
                task_type=self.current_task["task_type"],
                task_params=self.current_task["task_params"],
                target_region=self.current_task["target_region"],
            )
            self.prev_potential_score = initial_score.total_score
            
            if self.VERBOSE:
                print(f"[ActiveSpatialEnv] Initial potential score: {initial_score.total_score:.4f}")
                print(f"  Position: {initial_score.position_score:.4f}, Orientation: {initial_score.orientation_score:.4f}")
        
        # Initialize prev_distance for legacy progress reward
        target_pose = self._get_target_pose()
        if target_pose is not None:
            current_pose = c2w_extrinsic_to_se3(self.view_engine.get_pose())
            self.prev_distance = compute_translation_distance(current_pose, target_pose)
        else:
            self.prev_distance = None
        
        # Build task prompt from item or task description
        object_label = item.get("object_label", "object")
        preset = item.get("preset", "front")
        distance = item.get("distance", None)
        task_description = item.get("description", None)
        
        if task_description:
            task_prompt = task_description
        elif distance:
            task_prompt = f"Move the camera to the {preset} view of the {object_label}, about {distance:.2f} meters away."
        else:
            task_prompt = f"Move the camera to the {preset} view of the {object_label}."
        
        # Load spatial prior images if enabled
        self.spatial_prior_images = []
        self.spatial_prior_poses = []
        if self.config.enable_spatial_prior:
            self._load_spatial_prior(item, scene_id)
        
        # Build observation (render image)
        obs = self._render(init_obs=True, task_prompt=task_prompt)
        
        info = {
            "scene_id": scene_id,
            "object_label": object_label,
            "preset": preset,
            "distance": distance,
            "jsonl_idx": idx,
            "current_pose": c2w_extrinsic_to_se3(self.view_engine.get_pose()),
            "task_type": self.current_task["task_type"],
            "initial_potential_score": self.prev_potential_score,
        }
        
        return obs, info
    
    def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        Args:
            action_str: Raw text response from LLM
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.episode_done:
            return {"obs_str": "Episode done", "multi_modal_data": {}}, 0.0, True, {"error": "episode_done"}
        
        assert self.current_item is not None, "reset() must be called before step()."
        item = self.current_item
        
        # Parse the LLM response
        rst = self._default_parse_func(
            action_str,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step,
        )
        
        action_list = rst["actions"]
        format_correct = rst["format_correct"]
        
        # Metrics structure compatible with VAGEN BaseEnv expectations
        metrics = {
            "turn_metrics": {
                "action_is_valid": len(action_list) > 0 and format_correct,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            }
        }
        
        self.reward = 0
        done = False
        info = {}
        info.update(rst)
        
        prev_E = self.view_engine.get_pose()
        prev_pos = prev_E[:3, 3].copy()
        
        # Track collisions in this step
        step_collisions = []
        collision_feedback = ""
        
        # Execute valid actions
        if metrics["turn_metrics"]["action_is_valid"]:
            for action in action_list:
                action_lower = action.lower()
                
                if action_lower == "done":
                    # Agent signals completion
                    done = True
                    self.episode_done = True
                    
                    # Calculate final success based on potential field score
                    final_score = self._calculate_current_score()
                    success_threshold = self.config.success_score_threshold if self.config.enable_potential_field else 0.8
                    
                    if final_score >= success_threshold:
                        self.reward += self.config.success_reward
                        metrics["traj_metrics"]["success"] = True
                        
                    # Store final score in info for logging
                    info["final_score"] = final_score
                    info["success_threshold"] = success_threshold
                    break
                
                elif action_lower in self.ACTION_LOOKUP:
                    # Store position before action
                    pre_action_pos = self.view_engine.get_pose()[:3, 3].copy()
                    
                    # Execute action
                    self.view_engine.step(action_lower)
                    self._current_step += 1
                    
                    # Check for collision after action
                    if self.collision_detector is not None and self.config.enable_collision_detection:
                        new_pos = self.view_engine.get_pose()[:3, 3]
                        collision_result = self.collision_detector.check_collision(
                            position=new_pos,
                            previous_position=pre_action_pos
                        )
                        
                        if collision_result.has_collision:
                            self.collision_count += 1
                            self.last_collision_result = collision_result
                            step_collisions.append(collision_result)
                            
                            # Apply collision penalty
                            self.reward += self.config.collision_penalty
                            
                            # Invalidate action (revert to pre-action position)
                            if self.config.collision_invalidate_action:
                                # Revert camera to pre-action position
                                current_E = self.view_engine.get_pose()
                                current_E[:3, 3] = pre_action_pos
                                self.view_engine.reset(current_E)
                                
                                if self.VERBOSE:
                                    print(f"[Collision] Action '{action_lower}' blocked by {collision_result.collision_type}"
                                          f" ({collision_result.collision_object or 'boundary'})")
                    
                    if self._current_step >= self._max_episode_steps:
                        done = True
                        self.episode_done = True
                        # Check success at natural termination
                        final_score = self._calculate_current_score()
                        if final_score >= self.config.success_score_threshold:
                            self.reward += self.config.success_reward
                            metrics["traj_metrics"]["success"] = True
                            info["final_score"] = final_score
                        break
            
            # Add format reward
            self.reward += self.config.format_reward
            info["is_format_rewarded"] = True
        else:
            info["is_format_rewarded"] = False
        
        # Build collision feedback message
        if step_collisions:
            collision_msgs = []
            for cr in step_collisions:
                if cr.collision_type == "object":
                    collision_msgs.append(f"blocked by {cr.collision_object}")
                elif cr.collision_type == "floor":
                    collision_msgs.append("blocked by floor")
                elif cr.collision_type == "ceiling":
                    collision_msgs.append("blocked by ceiling")
                else:
                    collision_msgs.append(f"blocked by {cr.collision_type}")
            collision_feedback = f" [Collision: {', '.join(collision_msgs)}]"
        
        # Calculate pose-based reward if not done by explicit 'done' action
        if not done and metrics["turn_metrics"]["action_is_valid"]:
            pose_reward = self._calculate_pose_reward()
            self.reward += pose_reward
            
            # Auto-terminate when score >= threshold
            if self.config.enable_auto_termination and self.prev_potential_score >= self.config.success_score_threshold:
                done = True
                self.episode_done = True
                self.reward += self.config.success_reward
                metrics["traj_metrics"]["success"] = True
                info["final_score"] = self.prev_potential_score
                info["auto_terminated"] = True
                if self.VERBOSE:
                    print(f"[AutoTerminate] score={self.prev_potential_score:.4f} >= {self.config.success_score_threshold}, granting success_reward={self.config.success_reward}")
        
        # Check if action was effective
        curr_E = self.view_engine.get_pose()
        metrics["turn_metrics"]["action_is_effective"] = not np.allclose(prev_E, curr_E)
        
        # Add collision metrics
        metrics["turn_metrics"]["collision_count"] = len(step_collisions)
        metrics["traj_metrics"]["total_collisions"] = self.collision_count
        
        # Add potential field metrics
        if self.config.enable_potential_field and self.potential_field is not None:
            metrics["turn_metrics"]["potential_score"] = self.prev_potential_score
            if self.current_task:
                metrics["traj_metrics"]["task_type"] = self.current_task.get("task_type", "unknown")
        
        # Update info
        info["metrics"] = metrics
        info["env_step"] = self._current_step
        info["episode_elapsed_seconds"] = time.time() - self._episode_start_time
        info["task_success"] = metrics["traj_metrics"]["success"]
        info["current_pose"] = c2w_extrinsic_to_se3(self.view_engine.get_pose())
        info["current_potential_score"] = self.prev_potential_score
        info["collision_count"] = self.collision_count
        
        # Build env_feedback with collision info
        if metrics["turn_metrics"]["action_is_effective"]:
            env_feedback = "Action executed."
        elif step_collisions:
            env_feedback = f"Action blocked by collision.{collision_feedback}"
        else:
            env_feedback = "Action had no effect."
        info["env_feedback"] = env_feedback
        
        self.total_reward += self.reward
        
        obs = self._render(init_obs=False, env_feedback=env_feedback)
        
        return obs, self.reward, done, info
    
    def _get_target_pose(self) -> Optional[List[float]]:
        """
        Extract target pose from episode data.
        
        Returns:
            6-DoF target pose [tx, ty, tz, rx, ry, rz] or None
        """
        item = self.current_item
        if item is None:
            return None
        
        # Try different possible keys for target pose
        if "target_pose" in item:
            return item["target_pose"]
        
        # Construct from position + orientation
        target_pos = item.get("target_position", None)
        if target_pos is not None:
            # If we have target position, construct pose with default orientation
            rx, ry, rz = 0.0, 0.0, 0.0
            if "target_orientation" in item:
                orient = item["target_orientation"]
                rx, ry, rz = orient[0], orient[1], orient[2]
            return [target_pos[0], target_pos[1], target_pos[2], rx, ry, rz]
        
        return None
    
    def _calculate_current_score(self) -> float:
        """
        Calculate current pose score using potential field or legacy method.
        
        Returns:
            Score in [0, 1] indicating how well the current pose satisfies the task.
        """
        curr_E = self.view_engine.get_pose()
        curr_pos = curr_E[:3, 3]
        curr_forward = -curr_E[:3, 2]  # Camera -Z is forward
        
        # Use potential field if available
        if self.config.enable_potential_field and self.potential_field is not None:
            if self.current_task and self.current_task.get("target_region"):
                score_result = self.potential_field.compute_score(
                    camera_position=curr_pos,
                    camera_forward=curr_forward,
                    task_type=self.current_task["task_type"],
                    task_params=self.current_task["task_params"],
                    target_region=self.current_task["target_region"],
                )
                return score_result.total_score
        
        # Legacy scoring method
        item = self.current_item
        target_pos = np.array(item.get("target_position", [0, 0, 2]), dtype=np.float64)
        target_dir = np.array(item.get("camera_params", {}).get("forward", [0, 0, 1]), dtype=np.float64)
        
        final_score, _, _, _, _ = calculate_pose_score_smooth(
            curr_pos, curr_forward, target_pos, target_dir,
            transition_distance=self.config.transition_distance,
            max_distance=self.config.max_distance,
        )
        
        return final_score
    
    def _calculate_pose_reward(self) -> float:
        """
        Calculate comprehensive pose-based reward.
        
        Reward Components:
        ==================
        1. Spatial Potential Field: Task-specific geometric scoring (main reward)
        2. Visibility Reward: Bonus for keeping target visible (secondary)
        3. Step Penalty: Small penalty per step (efficiency)
        
        The total reward guides the agent to:
        - Reach the correct position/orientation for the task
        - Keep the target object visible (not occluded)
        - Complete the task efficiently (fewer steps)
        """
        curr_E = self.view_engine.get_pose()
        curr_pos = curr_E[:3, 3]
        curr_forward = -curr_E[:3, 2]  # Camera -Z is forward
        curr_up = curr_E[:3, 1]  # Camera Y is up (in OpenCV convention, down)
        
        total_reward = 0.0
        
        # === 1. Spatial Potential Field Reward (Primary) ===
        if self.config.enable_potential_field and self.potential_field is not None:
            if self.current_task and self.current_task.get("target_region"):
                score_result = self.potential_field.compute_score(
                    camera_position=curr_pos,
                    camera_forward=curr_forward,
                    task_type=self.current_task["task_type"],
                    task_params=self.current_task["task_params"],
                    target_region=self.current_task["target_region"],
                )
                
                current_score = score_result.total_score
                
                if self.config.potential_field_progress_mode == "delta":
                    # Reward is the improvement in score (can be negative)
                    delta_score = current_score - self.prev_potential_score
                    potential_reward = delta_score * self.config.potential_field_reward_scale
                else:  # "absolute"
                    # Reward is the current score scaled
                    potential_reward = current_score * self.config.potential_field_reward_scale * 0.1
                
                total_reward += potential_reward
                
                # Update previous score for next step
                self.prev_potential_score = current_score
                
                if self.VERBOSE:
                    print(f"[Potential Field] score={current_score:.4f}, "
                          f"delta={delta_score if self.config.potential_field_progress_mode == 'delta' else 'N/A'}, "
                          f"reward={potential_reward:.4f}")
        
        # === 2. Visibility Reward (Secondary) ===
        if self.config.enable_visibility_check and self.visibility_checker is not None:
            # Get target center from task
            target_center = self._get_target_center()
            
            if target_center is not None:
                # Get target bbox if available
                bbox_min, bbox_max = self._get_target_bbox()
                
                visibility = self.visibility_checker.check_visibility(
                    camera_position=curr_pos,
                    camera_forward=curr_forward,
                    camera_up=curr_up,
                    target_center=target_center,
                    target_bbox_min=bbox_min,
                    target_bbox_max=bbox_max,
                    target_label=self.current_item.get("object_label", ""),
                )
                
                visibility_reward = compute_visibility_reward(
                    visibility,
                    self.prev_visibility,
                    reward_scale=self.config.visibility_reward_scale,
                )
                total_reward += visibility_reward
                
                # Update previous visibility
                self.prev_visibility = visibility
                
                if self.VERBOSE:
                    print(f"[Visibility] in_fov={visibility.in_fov}, "
                          f"occlusion={visibility.occlusion_ratio:.2f}, "
                          f"screen={visibility.screen_coverage:.3f}, "
                          f"reward={visibility_reward:.4f}")
        
        # === 3. Step Penalty (Efficiency) ===
        if self.config.enable_step_penalty:
            total_reward += self.config.step_penalty  # Small negative value
        
        # If potential field is disabled, use legacy reward
        if not self.config.enable_potential_field or self.potential_field is None:
            total_reward += self._calculate_legacy_reward()
        
        return total_reward
    
    def _get_target_center(self) -> Optional[np.ndarray]:
        """Get target center position from task data."""
        # Try from target_region
        if self.current_task and self.current_task.get("target_region"):
            region = self.current_task["target_region"]
            params = region.get("params", {})
            
            # Try object_center first
            if "object_center" in params:
                return np.array(params["object_center"], dtype=np.float64)
            
            # Try sample_point
            if "sample_point" in region:
                return np.array(region["sample_point"], dtype=np.float64)
        
        # Fallback to target_position
        if self.current_item and "target_position" in self.current_item:
            return np.array(self.current_item["target_position"], dtype=np.float64)
        
        return None
    
    def _get_target_bbox(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get target object bounding box."""
        if self.target_object_info is None:
            return None, None
        
        # Handle single object
        if "bbox_min" in self.target_object_info:
            return (
                np.array(self.target_object_info["bbox_min"], dtype=np.float64),
                np.array(self.target_object_info["bbox_max"], dtype=np.float64),
            )
        
        # Handle multiple objects (use primary)
        if "primary" in self.target_object_info and self.target_object_info["primary"]:
            primary = self.target_object_info["primary"]
            if "bbox_min" in primary:
                return (
                    np.array(primary["bbox_min"], dtype=np.float64),
                    np.array(primary["bbox_max"], dtype=np.float64),
                )
        
        return None, None
    
    def _calculate_legacy_reward(self) -> float:
        """Legacy reward calculation for backwards compatibility."""
        item = self.current_item
        
        # Get current pose as 6-DoF
        current_pose = c2w_extrinsic_to_se3(self.view_engine.get_pose())
        
        # Get target pose
        target_pose = self._get_target_pose()
        
        if target_pose is None:
            return self._calculate_pose_reward_legacy()
        
        total_reward = 0.0
        current_distance = compute_translation_distance(current_pose, target_pose)
        
        # Per-step progress reward
        if self.config.enable_progress_reward and self.prev_distance is not None:
            progress_reward = compute_progress_reward(
                current_distance=current_distance,
                previous_distance=self.prev_distance,
                success_distance=self.config.success_distance,
                max_distance=self.config.max_distance,
                reward_scale=self.config.progress_reward_scale,
                reward_mode=self.config.progress_reward_mode,
            )
            total_reward += progress_reward
        
        # Approach reward
        if self.config.enable_approach_reward:
            approach_reward = compute_approach_reward(
                current_pose=current_pose,
                target_pose=target_pose,
                success_distance=self.config.success_distance,
                max_distance=self.config.max_distance,
                reward_base=self.config.pose_reward_base,
                reward_shaping=self.config.approach_reward_shaping,
            )
            total_reward += approach_reward
        
        # Success reward
        if is_goal_reached(current_pose, target_pose, self.config.success_distance):
            total_reward += self.config.success_reward
        
        self.prev_distance = current_distance
        return total_reward
    
    def _calculate_pose_reward_legacy(self) -> float:
        """Legacy pose reward calculation using score improvement."""
        item = self.current_item
        target_pos = np.array(item.get("target_position", [0, 0, 2]), dtype=np.float64)
        target_dir = np.array(item.get("camera_params", {}).get("forward", [0, 0, 1]), dtype=np.float64)
        
        curr_E = self.view_engine.get_pose()
        curr_pos = curr_E[:3, 3]
        curr_forward = -curr_E[:3, 2]
        
        final_score, _, _, _, _ = calculate_pose_score_smooth(
            curr_pos, curr_forward, target_pos, target_dir,
            transition_distance=self.config.transition_distance,
            max_distance=self.config.max_distance,
        )
        
        # Reward is the improvement in score
        delta = final_score - self.prev_score
        self.prev_score = final_score
        
        # Information gain reward for movement
        if self.prev_pos is not None:
            move_dist = np.linalg.norm(curr_pos - self.prev_pos)
            info_reward = 0.05 * min(move_dist / 1.0, 1.0)
        else:
            info_reward = 0.0
        
        self.prev_pos = curr_pos.copy()
        
        return delta + info_reward
    
    def _load_spatial_prior(self, item: Dict[str, Any], scene_id: str):
        """
        Load multi-frame spatial prior images from the dataset item.
        
        The dataset is responsible for providing the prior images - this method
        simply reads whatever is available in the data. Supports:
        - "spatial_prior_images": List of {image_path, view_name, pose} dicts
        - "prior_image_paths": Simple list of image paths
        
        Args:
            item: Current episode data item
            scene_id: Scene identifier (for logging)
        """
        self.spatial_prior_images = []
        self.spatial_prior_poses = []
        
        # Method 1: Structured prior images with metadata
        prior_images_data = item.get("spatial_prior_images", None)
        if prior_images_data and isinstance(prior_images_data, list):
            for prior_item in prior_images_data:
                img_path = prior_item.get("image_path", None)
                if img_path:
                    if self.jsonl_path:
                        full_path = resolve_rel_image(self.jsonl_path, img_path, self.dataset_root)
                    else:
                        full_path = img_path
                    img = safe_open_rgb(full_path)
                    if img is not None:
                        self.spatial_prior_images.append(img)
                        self.spatial_prior_poses.append({
                            "view_name": prior_item.get("view_name", f"view_{len(self.spatial_prior_images)}"),
                            "pose": prior_item.get("pose", None),
                        })
            
            if self.VERBOSE:
                print(f"[ActiveSpatialEnv] Loaded {len(self.spatial_prior_images)} prior images from spatial_prior_images")
            return
        
        # Method 2: Simple list of image paths
        prior_paths = item.get("prior_image_paths", None)
        if prior_paths and isinstance(prior_paths, list):
            for i, img_path in enumerate(prior_paths):
                if self.jsonl_path:
                    full_path = resolve_rel_image(self.jsonl_path, img_path, self.dataset_root)
                else:
                    full_path = img_path
                
                img = safe_open_rgb(full_path)
                if img is not None:
                    self.spatial_prior_images.append(img)
                    self.spatial_prior_poses.append({
                        "view_name": f"view_{i+1}",
                        "pose": None,
                    })
            
            if self.VERBOSE:
                print(f"[ActiveSpatialEnv] Loaded {len(self.spatial_prior_images)} prior images from prior_image_paths")
            return
        
        if self.VERBOSE:
            print("[ActiveSpatialEnv] Spatial prior enabled but no prior images found in dataset item")
    
    def _render_image(self) -> Optional[Image.Image]:
        """
        Render the current view using the 3D renderer.
        
        Returns:
            PIL Image if rendering is available, None otherwise.
        """
        if self.renderer is None:
            # No renderer - try to use pre-rendered image from dataset
            item = self.current_item
            if item is None:
                return None
            
            # Check if item has pre-rendered image path
            img_path = item.get("image_path", None) or item.get("init_image", None)
            if img_path:
                # Resolve relative path
                if self.jsonl_path:
                    full_path = resolve_rel_image(self.jsonl_path, img_path, self.dataset_root)
                    return safe_open_rgb(full_path)
            return None
        
        # Use real-time rendering
        try:
            # Get current camera extrinsics (camera-to-world)
            c2w = self.view_engine.get_pose()
            # Renderer expects world-to-camera (w2c) extrinsics
            w2c = np.linalg.inv(c2w)
            
            # Get camera intrinsics (ensure 3x3)
            K = self.camera_intrinsics
            if K.shape == (4, 4):
                K = K[:3, :3]
            
            # Get image size from config
            width = self.config.image_width
            height = self.config.image_height
            
            # Render (async call)
            image = _run_async(self.renderer.render_image_from_cam_param(
                camera_intrinsics=K,
                camera_extrinsics=w2c,
                width=width,
                height=height,
            ))
            
            return image
            
        except Exception as e:
            print(f"[ActiveSpatialEnv] Rendering error: {e}")
            return None
    
    def _render(self, init_obs: bool = True, task_prompt: str = "", env_feedback: str = "") -> Dict[str, Any]:
        """
        Render the current observation.
        
        Args:
            init_obs: Whether this is the initial observation
            task_prompt: Task description for initial observation
            env_feedback: Environment feedback for action observations
            
        Returns:
            Observation dictionary with text and images
        """
        img_placeholder = self.config.get("image_placeholder", "<image>")
        prior_placeholder = self.config.get("spatial_prior_placeholder", "<prior_images>")
        
        # Get current pose info
        current_pose = c2w_extrinsic_to_se3(self.view_engine.get_pose())
        pose_str = format_pose6_deg(current_pose)
        
        # Render the current view image
        rendered_image = self._render_image()
        
        # Build spatial prior text for initial observation
        spatial_prior_text = ""
        if init_obs and self.config.enable_spatial_prior and self.spatial_prior_images:
            num_prior = len(self.spatial_prior_images)
            
            # Build view labels from poses metadata
            view_labels = []
            for i, pose_info in enumerate(self.spatial_prior_poses):
                view_name = pose_info.get("view_name", f"view_{i+1}")
                view_labels.append(f"[{view_name.replace('_', ' ').title()}]")
            
            # Build prior text with placeholders for each image
            prior_images_text = " ".join([f"{prior_placeholder}" for _ in range(num_prior)])
            view_labels_text = " | ".join(view_labels) if view_labels else ""
            
            spatial_prior_text = f"""Below are {num_prior} viewpoint images showing different perspectives of the scene. Use these to understand the spatial layout before navigating.
{prior_images_text}"""
            if view_labels_text:
                spatial_prior_text += f"\nViews: {view_labels_text}"
        
        # Build observation string
        if init_obs:
            obs_str = init_observation_template(
                observation=f"{img_placeholder}\nCurrent camera pose: {pose_str}",
                task_prompt=task_prompt,
                spatial_prior=spatial_prior_text,
            )
        else:
            obs_str = action_template(
                observation=f"{img_placeholder}\nCurrent camera pose: {pose_str}",
                env_feedback=env_feedback,
            )
        
        # Add format prompt
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=init_obs,
        )
        obs_str += "\n" + format_prompt_text
        
        # Build images list - current view
        images = []
        if rendered_image is not None:
            images.append(rendered_image)
        
        # Build multi-modal data
        multi_modal_data = {
            img_placeholder: images,
        }
        
        # Add spatial prior images for initial observation
        if init_obs and self.config.enable_spatial_prior and self.spatial_prior_images:
            multi_modal_data[prior_placeholder] = self.spatial_prior_images.copy()
        
        obs = {
            "obs_str": obs_str,
            "multi_modal_data": multi_modal_data,
        }
        
        return obs
    
    def _build_observation_from_image(
        self, 
        image: Image.Image, 
        init_obs: bool = True,
        task_prompt: str = "",
        env_feedback: str = "",
    ) -> Dict[str, Any]:
        """
        Build observation from an externally rendered image.
        
        This is used when rendering is handled by an external service
        (e.g., Ray Actor Pool) instead of the internal renderer.
        
        Args:
            image: PIL Image from external renderer
            init_obs: Whether this is the initial observation
            task_prompt: Task description for initial observation
            env_feedback: Environment feedback for action observations
            
        Returns:
            Observation dictionary with text and images
        """
        img_placeholder = self.config.get("image_placeholder", "<image>")
        prior_placeholder = self.config.get("spatial_prior_placeholder", "<prior_images>")
        
        # Get current pose info
        current_pose = c2w_extrinsic_to_se3(self.view_engine.get_pose())
        pose_str = format_pose6_deg(current_pose)
        
        # Build task prompt if not provided
        if not task_prompt and init_obs and self.current_item:
            object_label = self.current_item.get("object_label", "object")
            preset = self.current_item.get("preset", "front")
            distance = self.current_item.get("distance", None)
            
            if distance:
                task_prompt = f"Move the camera to the {preset} view of the {object_label}, about {distance:.2f} meters away."
            else:
                task_prompt = f"Move the camera to the {preset} view of the {object_label}."
        
        # Build spatial prior text for initial observation
        spatial_prior_text = ""
        if init_obs and self.config.enable_spatial_prior and self.spatial_prior_images:
            num_prior = len(self.spatial_prior_images)
            
            # Build view labels from poses metadata
            view_labels = []
            for i, pose_info in enumerate(self.spatial_prior_poses):
                view_name = pose_info.get("view_name", f"view_{i+1}")
                view_labels.append(f"[{view_name.replace('_', ' ').title()}]")
            
            # Build prior text with placeholders for each image
            prior_images_text = " ".join([f"{prior_placeholder}" for _ in range(num_prior)])
            view_labels_text = " | ".join(view_labels) if view_labels else ""
            
            spatial_prior_text = f"""Below are {num_prior} viewpoint images showing different perspectives of the scene. Use these to understand the spatial layout before navigating.
{prior_images_text}"""
            if view_labels_text:
                spatial_prior_text += f"\nViews: {view_labels_text}"
        
        # Build observation string
        if init_obs:
            obs_str = init_observation_template(
                observation=f"{img_placeholder}\nCurrent camera pose: {pose_str}",
                task_prompt=task_prompt,
                spatial_prior=spatial_prior_text,
            )
        else:
            obs_str = action_template(
                observation=f"{img_placeholder}\nCurrent camera pose: {pose_str}",
                env_feedback=env_feedback if env_feedback else "Action executed.",
            )
        
        # Add format prompt
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=init_obs,
        )
        obs_str += "\n" + format_prompt_text
        
        # Build images list - current view
        images = []
        if image is not None:
            images.append(image)
        
        # Build multi-modal data
        multi_modal_data = {
            img_placeholder: images,
        }
        
        # Add spatial prior images for initial observation
        if init_obs and self.config.enable_spatial_prior and self.spatial_prior_images:
            multi_modal_data[prior_placeholder] = self.spatial_prior_images.copy()
        
        obs = {
            "obs_str": obs_str,
            "multi_modal_data": multi_modal_data,
        }
        
        return obs
    
    def system_prompt(self) -> str:
        """Get the system prompt for this environment."""
        return system_prompt(
            format=self.config.prompt_format,
            step_translation=self.config.step_translation,
            step_rotation_deg=self.config.step_rotation_deg,
        )
    
    def compute_reward(self) -> float:
        """Compute final reward for the episode."""
        return 0.0
    
    def close(self):
        """Close the environment and release resources."""
        if self.renderer is not None:
            try:
                # Close the async renderer
                _run_async(self.renderer.close())
            except Exception as e:
                print(f"[ActiveSpatialEnv] Error closing renderer: {e}")
            finally:
                self.renderer = None
                self._renderer_initialized = False
