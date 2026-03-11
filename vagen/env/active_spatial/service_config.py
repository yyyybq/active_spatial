from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass, field

@dataclass
class ActiveSpatialServiceConfig(BaseServiceConfig):
    """Service configuration for the Active Spatial Intelligence environment."""
    
    # Parallelism configuration
    max_workers: int = 4  # Maximum number of parallel workers for batch operations
    
    # Device configuration for parallel environment creation
    devices: list = field(default_factory=lambda: [0])
    
    # Rendering service configuration
    render_backend: str = "client"  # "local" or "client"
    client_url: str = "ws://127.0.0.1:8766/render"
    client_origin: str = ""
    gs_root: str = ""
    
    # Flag for using state-based rewards
    use_state_reward: bool = False
