from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, field, fields
from typing import Optional, List

# Reward type constants (aligned with ActiveVLN)
REWARD_ANSWER = "answer"
REWARD_POSE = "pose"
REWARD_COMBINED = "combined"

@dataclass
class ActiveSpatialEnvConfig(BaseEnvConfig):
    """Configuration class for the Active Spatial Intelligence environment.
    
    This environment involves navigating a camera in a 3D scene to reach a target pose
    relative to an object (e.g., moving to the front/back/left/right view of an object).
    
    Rendering Modes:
    ----------------
    - render_backend="client": Connect to remote GPU render server via WebSocket
      Requires: client_url to be set (e.g., "ws://gpu-node:8777/render/interiorgs")
      
    - render_backend="local": Render locally using GPU
      Requires: gs_root pointing to directory containing {scene_id}.ply files
      
    - render_backend=None or "none": Use pre-rendered images from dataset
      Requires: dataset items to have "image_path" field
      
    Aligned with ActiveVLN's viewsuite_server/env_config.py for compatibility.
    """
    env_name: str = "active_spatial"
    
    # ====== Dataset Configuration ======
    jsonl_path: str = ""  # Path to dataset JSONL file
    dataset_root: str = ""  # Root path for resolving relative image paths
    total_lines: int = -1  # Number of lines in JSONL (-1 = auto-count)
    
    # ====== Rendering Configuration ======
    # Choose one of: "client", "local", or None/empty
    render_backend: Optional[str] = "local"  # None = use pre-rendered images
    gpu_device: Optional[int] = None  # GPU device ID for local rendering (None = auto-detect from environment)
    
    # For render_backend="local" - local Gaussian Splatting rendering
    gs_root: str = ""  # Directory containing {scene_id}.ply files
    
    # For render_backend="client" - remote render server
    client_url: str = "ws://127.0.0.1:8777/render/interiorgs"  # WebSocket URL
    client_origin: Optional[str] = None  # Origin header for WebSocket connection
    
    # Image dimensions for rendering
    image_width: int = 512
    image_height: int = 512
    
    # Legacy compatibility (deprecated, use image_width/image_height)
    render_width: int = 512
    render_height: int = 512
    
    # ====== Camera Control Configuration ======
    # NOTE: ActiveVLN uses 0.3m/30deg as defaults, but we use smaller steps for finer control
    step_translation: float = 0.3  # Step size for translation (meters) - aligned with ActiveVLN
    step_rotation_deg: float = 30.0  # Step size for rotation (degrees) - aligned with ActiveVLN
    
    # ====== Reward Configuration ======
    reward_type: str = REWARD_COMBINED  # "answer", "pose", or "combined"
    format_reward: float = 0.2  # Reward for correct format
    answer_reward: float = 0.8  # Reward for correct answer (for QA tasks)
    success_reward: float = 1.0  # Reward for reaching target pose
    
    # ====== Spatial Potential Field Configuration ======
    # This enables task-driven dense reward based on geometric scoring
    enable_potential_field: bool = True  # Enable task-specific potential field scoring
    potential_field_position_weight: float = 0.7  # Weight for position component
    potential_field_orientation_weight: float = 0.3  # Weight for orientation component
    potential_field_reward_scale: float = 1.0  # Scale factor for potential field reward
    potential_field_progress_mode: str = "delta"  # "delta" (score change) or "absolute" (current score)
    success_score_threshold: float = 0.85  # Score threshold for task success
    
    # ====== Legacy Per-Step Progress Reward Configuration ======
    # (Deprecated: Use potential field instead)
    enable_progress_reward: bool = False  # Whether to enable legacy per-step progress reward
    progress_reward_scale: float = 0.5  # Scale factor for progress reward
    progress_reward_mode: str = "delta"  # "delta", "delta_normalized", "potential", "scaled_delta"
    
    # ====== Legacy Approach Reward Configuration (aligned with ActiveVLN) ======
    # (Deprecated: Use potential field instead)
    enable_approach_reward: bool = False  # Whether to enable legacy pose-based approach reward
    pose_reward_base: float = 1.0  # Base reward for approach task
    success_distance: float = 1.0  # Distance threshold for success (meters)
    max_distance: float = 5.0  # Maximum distance for reward calculation
    transition_distance: float = 2.0  # Distance threshold for weight transition
    approach_reward_shaping: str = "weighted"  # "binary", "linear", "weighted", "exponential"
    
    # ====== Collision Detection Configuration ======
    # Prevents camera from passing through walls and objects
    enable_collision_detection: bool = True  # Enable collision detection
    collision_camera_radius: float = 0.15  # Camera collision sphere radius (meters)
    collision_floor_height: float = 0.3  # Minimum camera height (meters)
    collision_ceiling_height: float = 2.5  # Maximum camera height (meters)
    collision_safety_margin: float = 0.05  # Additional safety margin around objects
    collision_penalty: float = -0.15  # Penalty for collision attempts
    collision_invalidate_action: bool = True  # If True, collision actions don't move camera
    
    # ====== Visibility & Occlusion Configuration ======
    # Ensures the agent can actually SEE the target, not just reach correct position
    enable_visibility_check: bool = True  # Enable visibility-based rewards
    visibility_reward_scale: float = 0.3  # Scale for visibility reward component
    fov_horizontal: float = 60.0  # Horizontal field of view (degrees)
    fov_vertical: float = 60.0  # Vertical field of view (degrees)
    min_visible_screen_coverage: float = 0.02  # Minimum screen coverage to be "visible"
    
    # ====== Action Efficiency Configuration ======
    # Penalizes unnecessary steps to encourage efficient navigation
    enable_step_penalty: bool = True  # Enable per-step penalty
    step_penalty: float = -0.01  # Small penalty per step (encourages efficiency)
    
    # ====== Prompt Configuration ======
    prompt_format: str = "free_think"  # "free_think", "no_think", "approach"
    max_actions_per_step: int = 5  # Maximum actions allowed per step
    action_sep: str = "|"  # Separator for multiple actions
    image_placeholder: str = "<image>"  # Placeholder for image in prompt
    special_token_list: Optional[List[str]] = field(
        default_factory=lambda: ["<think>", "</think>", "<action>", "</action>"]
    )
    
    # ====== Episode Configuration ======
    max_episode_steps: int = 50  # step_budget in ActiveVLN
    turn_budget: int = 20  # Max turns (for multi-turn interaction)
    
    # ====== Multi-Frame Spatial Prior Configuration ======
    # When enabled, the environment will look for pre-rendered spatial prior images
    # in the dataset (fields: "spatial_prior_images" or "prior_image_paths")
    # The number and type of prior images is determined by the dataset, not here.
    enable_spatial_prior: bool = False  # Enable reading spatial prior from dataset
    spatial_prior_placeholder: str = "<prior_images>"  # Placeholder for prior images in prompt
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Sync legacy render dimensions
        if self.image_width == 512 and self.render_width != 512:
            self.image_width = self.render_width
        if self.image_height == 512 and self.render_height != 512:
            self.image_height = self.render_height
        
        # Note: gpu_device will be auto-detected at render time if None
        # Don't set it here to avoid saving CPU-detected value to dataset
    
    def get(self, key: str, default=None):
        """Get configuration value by key."""
        return getattr(self, key, default)
    
    @property
    def cache_key(self) -> str:
        """Cache key for resource pool reuse (aligned with ActiveVLN)."""
        import hashlib
        import json
        cache_fields = ["render_backend", "gs_root", "client_url"]
        cache_dict = {f: getattr(self, f, "") for f in cache_fields}
        return hashlib.sha256(json.dumps(cache_dict, sort_keys=True).encode()).hexdigest()[:16]
    
    def config_id(self) -> str:
        """Generate a unique identifier for this configuration."""
        id_fields = ["render_backend", "max_actions_per_step", "step_translation", "step_rotation_deg"]
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields])
        return f"ActiveSpatialEnvConfig({id_str})"


if __name__ == "__main__":
    config = ActiveSpatialEnvConfig()
    print(config.config_id())
