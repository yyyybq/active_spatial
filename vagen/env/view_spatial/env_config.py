from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, field, fields

@dataclass
class ViewSpatialEnvConfig(BaseEnvConfig):
    """Configuration class for the View Spatial Bench environment.
    
    This environment involves answering spatial reasoning questions about
    3D scenes, with the ability to explore the scene by changing viewpoints.
    """
    env_name: str = "view_spatial"
    
    # Dataset configuration
    jsonl_path: str = ""  # Path to dataset JSONL file
    dataset_root: str = ""  # Root path for resolving relative image paths
    total_lines: int = -1  # Number of lines in JSONL (-1 = auto-count)
    
    # Rendering configuration
    render_backend: str = "client"  # "local" or "client"
    gs_root: str = ""  # Path to Gaussian Splatting assets (for tool version)
    scannet_root: str = ""  # Path to ScanNet data (for ScanNet scenes)
    client_url: str = "ws://127.0.0.1:8766/render"  # WebSocket URL for remote rendering
    client_origin: str = ""  # Origin header for WebSocket connection
    scene_id: str = ""  # Scene identifier
    render_width: int = 300
    render_height: int = 300
    
    # Environment mode
    use_tools: bool = False  # Whether to enable exploration tools
    
    # Camera control configuration (for tool mode)
    step_translation: float = 0.3  # Step size for translation movements (meters)
    step_rotation_deg: float = 30.0  # Step size for rotation movements (degrees)
    
    # Reward configuration
    format_reward: float = 0.2  # Reward for correct format
    answer_reward: float = 0.8  # Reward for correct answer
    
    # Prompt configuration
    prompt_format: str = "free_think"  # "free_think", "no_think", "grounding", etc.
    max_actions_per_step: int = 5  # Maximum actions allowed per step
    action_sep: str = "|"  # Separator for multiple actions
    
    # Max episode steps
    max_episode_steps: int = 10  # Usually single-turn for no-tool mode
    
    def config_id(self) -> str:
        """Generate a unique identifier for this configuration."""
        id_fields = ["use_tools", "render_backend", "max_actions_per_step"]
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields])
        return f"ViewSpatialEnvConfig({id_str})"


if __name__ == "__main__":
    config = ViewSpatialEnvConfig()
    print(config.config_id())
