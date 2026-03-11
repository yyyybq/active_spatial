# Active Spatial Intelligence Service
# HTTP/WebSocket service wrapper for the Active Spatial environment.

from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from vagen.env.base.base_service import BaseService
from vagen.env.active_spatial.env import ActiveSpatialEnv
from vagen.env.active_spatial.env_config import ActiveSpatialEnvConfig
from vagen.server.serial import serialize_observation
from .service_config import ActiveSpatialServiceConfig


class ActiveSpatialService(BaseService):
    """
    Service class for Active Spatial Intelligence environments.
    Implements batch operations with parallel processing for efficiency.
    """
    
    def __init__(self, config: ActiveSpatialServiceConfig):
        """
        Initialize the ActiveSpatialService.
        
        Args:
            config: Service configuration
        """
        self.max_workers = config.max_workers
        self.device_status = {device_id: set() for device_id in config.devices}
        self.environments: Dict[str, ActiveSpatialEnv] = {}
        self.env_configs: Dict[str, ActiveSpatialEnvConfig] = {}
        self.config = config
        print(f"[DEBUG] ActiveSpatialService initialized with config: {self.config}")
    
    def create_environments_batch(self, ids2configs: Dict[str, Any]) -> None:
        """
        Create multiple Active Spatial environments in parallel.
        
        Args:
            ids2configs: Dictionary mapping environment IDs to configurations
        """
        def create_single_env(env_id: str, config: Dict[str, Any]) -> Tuple[str, Any, Optional[str]]:
            env_name = config.get('env_name', 'active_spatial')
            if env_name != 'active_spatial':
                return env_id, None, f"Expected environment type 'active_spatial', got '{env_name}'"
            
            try:
                env_config_dict = config['env_config']
                env_config = ActiveSpatialEnvConfig(**env_config_dict)
                env = ActiveSpatialEnv(env_config)
                return env_id, (env, env_config), None
            except Exception as e:
                return env_id, None, str(e)
        
        # Assign GPUs to environments
        for i, env_id in enumerate(ids2configs.keys()):
            selected_gpu = min(self.device_status, key=lambda x: len(self.device_status[x]))
            ids2configs[env_id]['env_config']['gpu_device'] = selected_gpu
            self.device_status[selected_gpu].add(env_id)
        
        # Create environments in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(create_single_env, env_id, config): env_id
                for env_id, config in ids2configs.items()
            }
            
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error creating environment {env_id}: {error}")
                    continue
                
                env, env_config = result
                self.environments[env_id] = env
                self.env_configs[env_id] = env_config
    
    def reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """
        Reset multiple Active Spatial environments in parallel.
        
        Args:
            ids2seeds: Dictionary mapping environment IDs to seeds
            
        Returns:
            Dictionary mapping environment IDs to (observation, info) tuples
        """
        results = {}
        
        def reset_single_env(env_id: str, seed: int) -> Tuple[str, Any, Optional[str]]:
            try:
                env = self.environments[env_id]
                observation, info = env.reset(seed=seed)
                serialized_observation = serialize_observation(observation)
                return env_id, (serialized_observation, info), None
            except Exception as e:
                return env_id, None, str(e)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(reset_single_env, env_id, seed): env_id
                for env_id, seed in ids2seeds.items()
            }
            
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error resetting environment {env_id}: {error}")
                    results[env_id] = ({}, {"error": error})
                else:
                    results[env_id] = result
        
        return results
    
    def step_batch(self, ids2actions: Dict[str, Any]) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
        """
        Take a step in multiple Active Spatial environments in parallel.
        
        Args:
            ids2actions: Dictionary mapping environment IDs to actions
            
        Returns:
            Dictionary mapping environment IDs to (observation, reward, done, info) tuples
        """
        results = {}
        
        def step_single_env(env_id: str, action: str) -> Tuple[str, Any, Optional[str]]:
            try:
                env = self.environments[env_id]
                observation, reward, done, info = env.step(action)
                serialized_observation = serialize_observation(observation)
                return env_id, (serialized_observation, reward, done, info), None
            except Exception as e:
                print(f"Error stepping environment {env_id}: {e}")
                try:
                    # Try to recover by resetting
                    env = self.environments[env_id]
                    observation, info = env.reset()
                    serialized_observation = serialize_observation(observation)
                    return env_id, (serialized_observation, 0.0, True, {"error": str(e)}), None
                except Exception as e2:
                    # Recreate environment if reset fails
                    try:
                        config = self.env_configs[env_id]
                        env = ActiveSpatialEnv(config)
                        self.environments[env_id] = env
                        observation, info = env.reset()
                        serialized_observation = serialize_observation(observation)
                        return env_id, (serialized_observation, 0.0, True, {"error": str(e)}), None
                    except Exception as e3:
                        return env_id, None, str(e3)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(step_single_env, env_id, action): env_id
                for env_id, action in ids2actions.items()
            }
            
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error stepping environment {env_id}: {error}")
                    results[env_id] = ({}, 0.0, True, {"error": error})
                else:
                    results[env_id] = result
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[str, float]:
        """
        Compute the total reward for multiple environments in parallel.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            Dictionary mapping environment IDs to rewards
        """
        results = {}
        
        def compute_reward_single_env(env_id: str) -> Tuple[str, float, Optional[str]]:
            try:
                env = self.environments[env_id]
                return env_id, env.compute_reward(), None
            except Exception as e:
                return env_id, 0.0, str(e)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(compute_reward_single_env, env_id): env_id
                for env_id in env_ids
            }
            
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, reward, error = future.result()
                if error:
                    print(f"Error computing reward for environment {env_id}: {error}")
                results[env_id] = reward
        
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[str, str]:
        """
        Get system prompts for multiple environments in parallel.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            Dictionary mapping environment IDs to system prompts
        """
        results = {}
        
        def get_system_prompt_single_env(env_id: str) -> Tuple[str, str, Optional[str]]:
            try:
                env = self.environments[env_id]
                return env_id, env.system_prompt(), None
            except Exception as e:
                return env_id, "", str(e)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(get_system_prompt_single_env, env_id): env_id
                for env_id in env_ids
            }
            
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, prompt, error = future.result()
                if error:
                    print(f"Error getting system prompt for environment {env_id}: {error}")
                results[env_id] = prompt
        
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple environments.
        
        Args:
            env_ids: List of environment IDs to close. If None, close all environments.
        """
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        for env_id in env_ids:
            if env_id in self.environments:
                try:
                    self.environments[env_id].close()
                except Exception as e:
                    print(f"Error closing environment {env_id}: {e}")
                del self.environments[env_id]
            
            if env_id in self.env_configs:
                del self.env_configs[env_id]
            
            # Remove from device status
            for device_id in self.device_status:
                self.device_status[device_id].discard(env_id)
