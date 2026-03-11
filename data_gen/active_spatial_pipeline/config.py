"""
Configuration module for Active Spatial Pipeline
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class ObjectSelectionConfig:
    """Configuration for object selection/filtering."""
    # Semantic blacklist for excluded object categories
    blacklist: set = field(default_factory=lambda: {
        # Structural elements
        "wall", "floor", "ceiling", "room",
        # Carpet variations
        "carpet", "rug",
        # Light fixtures
        "chandelier", "ceiling lamp", "spotlight", "lamp", "light",
        "downlights", "wall lamp", "table lamp", "strip light", "track light",
        "linear lamp", "decorative pendant",
        # Generic / unclear categories
        "other", "curtain", "bread", "cigar", "wine", "fresh food", "pen",
        "medicine bottle", "toiletries", "chocolate", "paper",
        # Small items that appear in large quantities
        "book", "boxed food", "bagged food", "medicine box",
        "vegetable", "fruit", "drinks", "canned food",
    })
    
    # Geometric filter toggles (set to False to disable)
    enable_dim_filter: bool = True  # Enable min/max dimension filtering
    enable_volume_filter: bool = True  # Enable minimum volume filtering
    enable_aspect_ratio_filter: bool = True  # Enable aspect ratio filtering
    
    # Geometric constraints (only applied if corresponding filter is enabled)
    min_dim_component: float = 0.1  # Minimum dimension per axis (m)
    max_dim_component: float = 3.0  # Maximum dimension per axis (m)
    min_volume: float = 0.1  # Minimum volume (m^3)
    min_aspect_ratio: float = 0.1  # Min shortest/longest edge ratio
    min_dist_to_wall: float = 0.0  # Minimum distance to wall (m)
    
    # Pair constraints - relaxed for more data
    min_pair_dist: float = 0.1  # Minimum distance between paired objects
    max_pair_dist: float = 6.0  # Maximum distance between paired objects (relaxed)
    max_pair_dim_ratio: float = 5.0  # Max ratio of longest edges (relaxed)
    max_pair_dim_diff: float = 4.0  # Max absolute difference in longest edge (relaxed)
    require_same_room: bool = True  # Whether to require objects to be in the same room (MUST be True)
    
    # Dynamic thresholds
    dyn_min_mult: float = 0.1  # Multiplier for dynamic min distance (relaxed from 0.2)
    dyn_max_mult: float = 5.0  # Multiplier for dynamic max distance (relaxed from 2.5)
    
    # Pair visibility constraints (ensure both objects can be seen together)
    pair_fov_deg: float = 90.0  # Camera field of view in degrees for pair visibility check (relaxed from 60)
    pair_fov_margin_deg: float = 5.0  # Margin to leave on each side of FOV (relaxed from 10)
    min_viewing_distance_mult: float = 0.8  # Multiplier for minimum viewing distance (relaxed from 1.2)


@dataclass
class CameraSamplingConfig:
    """Configuration for camera pose sampling.
    
    Move Patterns:
        - 'around': Horizontal circle - sample cameras around object on horizontal plane (default)
        - 'rotation': Room rotation - stand at room center, rotate 360 degrees
        - 'linear': Linear trajectory - walk past object (passing motion)
    """
    # Number of camera poses to sample per object/pair
    num_cameras_per_item: int = 5
    
    # Move pattern: 'around', 'rotation', or 'linear'
    # - 'around': Horizontal circle around object (default)
    # - 'rotation': Stand at room center, rotate 360° (room-centric)
    # - 'linear': Walk past object in straight line (passing motion)
    move_pattern: str = 'around'
    
    # Sampling parameters
    per_angle: int = 36  # Number of angles to try per radius
    max_tries: int = 500  # Maximum sampling attempts
    
    # Camera heights (m) - used as fallback only
    # Dynamic height calculation is now preferred based on object heights
    camera_heights: List[float] = field(default_factory=lambda: [1.0, 1.2, 1.35, 1.5, 1.65, 1.8])
    
    # Dynamic camera height parameters
    max_camera_height: float = 1.6  # Maximum camera height above ground (m)
    min_camera_height: float = 0.8  # Minimum camera height above ground (m)
    camera_height_offset: float = 0.1  # Height offset above object top (m)
    
    # Object pair/triple height validation - relaxed for more pairs
    max_height_ratio: float = 2.0  # Maximum ratio of tallest to shortest object height (relaxed from 2.0)
    
    # ========== Enhanced Visibility Thresholds (from ViewSuite) ==========
    # Projected area check
    min_visibility_ratio: float = 0.08  # Minimum visible area ratio (5% of image)
    
    # Occlusion thresholds
    max_occlusion_ratio: float = 0.7  # Maximum allowed occlusion (70%)
    use_2d_occlusion: bool = True  # Use 2D image-space occlusion (more accurate)
    occlusion_depth_mode: str = "min"  # "min" or "mean" for depth ordering
    
    # Corner visibility
    min_visible_corners: int = 1  # Minimum visible bbox corners
    check_corner_occlusion: bool = False  # Check if corners are occluded (slower but more accurate)
    
    # ===== Rotation mode parameters (move_pattern='rotation') =====
    # Stand at room center, rotate 360° to look around
    rotation_interval: float = 5.0      # Degrees between each camera pose (360/5 = 72 images)
    rotation_camera_height: float = 1.5  # Fixed camera height (m)
    
    # ===== Linear mode parameters (move_pattern='linear') =====
    # Walk past an object in a straight line (passing motion)
    # Camera orientation (yaw/pitch) remains FIXED throughout trajectory
    # Only camera POSITION changes along a linear path
    linear_num_steps: int = 5           # Number of poses along trajectory
    linear_move_distance: float = 0.5   # Total movement distance along trajectory (meters)
    
    # Camera intrinsics
    image_width: int = 640
    image_height: int = 480
    focal_length: float = 300.0  # Reduced focal length for wider FoV (~60 deg vertical)


@dataclass
class TaskConfig:
    """Configuration for task generation."""
    # Which tasks to generate (from 9 task types)
    enabled_tasks: List[str] = field(default_factory=lambda: [
        # Metric Distance Tasks
        'absolute_positioning',
        'delta_control', 
        'equidistance',
        # Relative Position Tasks
        'projective_relations',
        'centering',
        'occlusion_alignment',
        # View Perspective Tasks
        'fov_inclusion',
        'size_distance_invariance',
        'screen_occupancy',
    ])
    
    # Task-specific parameters
    # absolute_positioning: distances should not exceed 2m for practical navigation
    absolute_positioning_distances: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2, 1.5])
    delta_control_deltas: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 1.0])
    projective_relations: List[str] = field(default_factory=lambda: ['left', 'right'])
    screen_occupancy_ratios: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.5, 0.7])
    
    # Minimum distance to object for valid camera positions
    # This prevents camera from getting too close to objects
    min_distance_to_object: float = 0.5  # meters - minimum distance to keep from objects
    
    # Camera FoV parameters (increased for wider field of view)
    fov_horizontal: float = 110.0  # degrees (wider horizontal FoV)
    fov_vertical: float = 90.0  # degrees (wider vertical FoV)
    fov_margin: float = 5.0  # margin in degrees for FoV inclusion task
    
    # Occlusion alignment parameters
    occlusion_min_distance: float = 0.5  # minimum distance for occlusion task (meters)
    
    # Agent height
    agent_height: float = 1.5  # meters


@dataclass 
class PipelineConfig:
    """Main configuration for the entire pipeline."""
    # Input/Output paths
    scenes_root: str = ""  # Root directory containing scene folders
    output_dir: str = ""  # Output directory for generated data
    
    # Scene selection
    scene_list: Optional[List[str]] = None  # List of scenes to process (None = all)
    
    # Sub-configs
    object_selection: ObjectSelectionConfig = field(default_factory=ObjectSelectionConfig)
    camera_sampling: CameraSamplingConfig = field(default_factory=CameraSamplingConfig)
    task_config: TaskConfig = field(default_factory=TaskConfig)
    
    # Processing options
    save_intermediate: bool = True  # Save intermediate results
    
    # Rendering options
    render_previews: bool = False  # Whether to render preview images
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary."""
        # Handle nested configs
        obj_sel = ObjectSelectionConfig(**config_dict.pop('object_selection', {}))
        cam_samp = CameraSamplingConfig(**config_dict.pop('camera_sampling', {}))
        task_cfg = TaskConfig(**config_dict.pop('task_config', {}))
        
        return cls(
            object_selection=obj_sel,
            camera_sampling=cam_samp,
            task_config=task_cfg,
            **config_dict


            
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.scenes_root:
            raise ValueError("scenes_root must be specified")
        if not self.output_dir:
            raise ValueError("output_dir must be specified")
        return True
