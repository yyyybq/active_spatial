# View Spatial Bench Environment
# This environment wraps ViewSuite's view spatial bench for VAGEN training.

import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from vagen.env.base.base_env import BaseEnv
from vagen.env.utils.context_utils import convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP

from .env_config import ViewSpatialEnvConfig
from .prompt import (
    system_prompt, 
    system_prompt_no_tool, 
    system_prompt_tool,
    init_observation_template, 
    action_template, 
    format_prompt,
    build_reset_prompt,
)
from .utils import (
    ViewManipulator,
    count_lines,
    read_jsonl_line_by_index,
    resolve_rel_image,
    safe_open_rgb,
    parse_free_think,
    parse_actions,
    check_actions_no_tool,
    check_actions_tool,
    answer_match,
    c2w_extrinsic_to_se3,
    c2w_se3_to_extrinsic,
    format_pose6_deg,
    fallback_K,
    parse_get_view_arg_deg,
)


class ViewSpatialEnv(BaseEnv):
    """
    View Spatial Bench Environment for VAGEN.
    
    This environment involves answering spatial reasoning questions
    about 3D scenes, optionally with the ability to explore.
    """
    
    def __init__(self, config: ViewSpatialEnvConfig):
        """Initialize the View Spatial environment."""
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
        
        # Tool mode vs no-tool mode
        self.use_tools = config.use_tools
        
        # View engine for camera manipulation (only for tool mode)
        if self.use_tools:
            self.view_engine = ViewManipulator(
                step_translation=config.step_translation,
                step_rotation_deg=config.step_rotation_deg,
                world_up_axis="Z",
            )
        else:
            self.view_engine = None
        
        # Renderer (will be initialized when needed)
        self.renderer = None
        self._renderer_initialized = False
        
        # Episode state
        self.current_item: Optional[Dict[str, Any]] = None
        self.episode_done: bool = False
        self.images: List[Image.Image] = []
        self.named_views: Dict[str, Dict[str, Any]] = {}
        self.camera_intrinsics: Optional[np.ndarray] = None
        self.active_K_source: Optional[str] = None
        self._current_step = 0
        self._max_episode_steps = config.max_episode_steps
        self._episode_start_time = 0
        self.total_reward = 0
        self.reward = 0
        
        # Store format prompt function
        if self.use_tools:
            self.format_prompt_func = format_prompt.get("tool", format_prompt["free_think"])
        else:
            self.format_prompt_func = format_prompt.get(config.prompt_format, format_prompt["free_think"])
    
    def _parse_action_str(self, action_str: str) -> Dict[str, Any]:
        """Parse the LLM response to extract actions."""
        ft = parse_free_think(action_str)
        if not ft["ok"]:
            return {
                "format_correct": False,
                "actions": [],
                "parsed_actions": [],
                "think": "",
                "raw_response": action_str,
            }
        
        sep = self.config.action_sep
        max_actions = self.config.max_actions_per_step
        
        ok, parsed_actions = parse_actions(ft["actions_blob"], sep=sep)
        if not ok:
            return {
                "format_correct": False,
                "actions": [],
                "parsed_actions": [],
                "think": ft["think"],
                "raw_response": action_str,
            }
        
        # Validate actions based on mode
        if self.use_tools:
            valid = check_actions_tool(parsed_actions)
        else:
            valid = check_actions_no_tool(parsed_actions)
        
        if not valid:
            return {
                "format_correct": False,
                "actions": [],
                "parsed_actions": [],
                "think": ft["think"],
                "raw_response": action_str,
            }
        
        # Extract action names
        actions = [a.name for a in parsed_actions[:max_actions]]
        
        return {
            "format_correct": True,
            "actions": actions,
            "parsed_actions": parsed_actions[:max_actions],
            "think": ft["think"],
            "raw_response": action_str,
        }
    
    def reset(self, seed: int = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to a new episode.
        
        Args:
            seed: Random seed for episode selection
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is None:
            seed = 0
        
        idx = seed % max(1, self.total_lines)
        
        # Reset episode state
        self.episode_done = False
        self._current_step = 0
        self._episode_start_time = time.time()
        self.total_reward = 0
        self.reward = 0
        self.images.clear()
        self.named_views.clear()
        self.active_K_source = None
        
        # Load episode data
        if self.jsonl_path and self.total_lines > 0:
            self.current_item = read_jsonl_line_by_index(self.jsonl_path, idx)
        else:
            # Mock data for testing
            self.current_item = {
                "question": "What is the relative position of object A to object B?",
                "choices": "A. Left\nB. Right\nC. Above\nD. Below",
                "answer": "A",
                "image_path": [],
            }
        
        item = self.current_item
        
        # Load images from data
        for rel in (item.get("image_path") or []):
            img = safe_open_rgb(resolve_rel_image(self.jsonl_path, rel, self.dataset_root))
            if img is not None:
                self.images.append(img)
        
        # For tool mode: build named views and initialize camera
        if self.use_tools:
            image_pose_dict = item.get("image_camera_pose") or {}
            for view_name, pose_pack in image_pose_dict.items():
                E = np.array(pose_pack.get("camera_extrinsics", np.eye(4)), dtype=np.float64)
                K = np.array(pose_pack.get("camera_intrinsics", fallback_K()), dtype=np.float64)
                self.named_views[str(view_name)] = {
                    "pose6_deg": c2w_extrinsic_to_se3(E),
                    "E": E,
                    "K": K
                }
            
            # Initialize camera
            if self.named_views:
                first_name = next(iter(self.named_views.keys()))
                first_view = self.named_views[first_name]
                self.camera_intrinsics = first_view["K"]
                self.active_K_source = first_name
                self.view_engine.reset(first_view["E"])
            else:
                self.camera_intrinsics = fallback_K()
                self.active_K_source = "fallback"
                self.view_engine.reset(None)
        
        # Build question and choices
        question = (item.get("question") or "").strip()
        choices = item.get("choices") or ""
        
        # Build observation
        obs = self._render(init_obs=True, question=question, choices=choices)
        
        info = {
            "scene_id": item.get("scene_id"),
            "sample_id": item.get("sample_id"),
            "question_type": item.get("question_type"),
            "choices": choices,
            "jsonl_idx": idx,
            "named_views": list(self.named_views.keys()) if self.use_tools else None,
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
        rst = self._parse_action_str(action_str)
        format_correct = rst["format_correct"]
        parsed_actions = rst.get("parsed_actions", [])
        
        # Metrics structure compatible with VAGEN BaseEnv expectations
        metrics = {
            "success": False,
            "action_is_valid": format_correct,
            "action_is_effective": False,
        }
        
        self.reward = 0
        done = False
        answer_correct = False
        parsed_answer = None
        reports: List[str] = []
        info = {}
        info.update(rst)
        
        if format_correct:
            # Add format reward
            self.reward += self.config.format_reward
            info["is_format_rewarded"] = True
            
            if self.use_tools:
                # Process tool actions
                done, answer_correct, parsed_answer, reports = self._process_tool_actions(parsed_actions)
            else:
                # No-tool mode: just check answer
                if parsed_actions and parsed_actions[0].name == "answer":
                    parsed_answer = parsed_actions[0].arg
                    gold = item.get("answer", "")
                    answer_correct = answer_match(parsed_answer, gold)
                    done = True
                    self.episode_done = True
            
            # Add answer reward if correct
            if answer_correct:
                self.reward += self.config.answer_reward
                metrics["success"] = True
        else:
            info["is_format_rewarded"] = False
        
        # Update step count
        self._current_step += 1
        if self._current_step >= self._max_episode_steps:
            done = True
            self.episode_done = True
        
        # Update info
        info["metrics"] = metrics
        info["env_step"] = self._current_step
        info["episode_elapsed_seconds"] = time.time() - self._episode_start_time
        info["task_success"] = metrics["success"]
        info["parsed_answer"] = parsed_answer
        info["answer_correct"] = answer_correct
        info["format_reward"] = self.config.format_reward if format_correct else 0.0
        info["answer_reward"] = self.config.answer_reward if answer_correct else 0.0
        info["total_reward"] = self.reward
        info["reports"] = reports
        
        self.total_reward += self.reward
        
        # Build status message
        status_parts = []
        status_parts.append("format: ok" if format_correct else "format: error")
        if parsed_answer:
            status_parts.append("answer: correct" if answer_correct else "answer: wrong")
        status = " | ".join(status_parts)
        
        obs = self._render(init_obs=False, status=status, reports=reports)
        
        return obs, self.reward, done, info
    
    def _process_tool_actions(self, parsed_actions: List) -> Tuple[bool, bool, Optional[str], List[str]]:
        """
        Process tool mode actions.
        
        Returns:
            Tuple of (done, answer_correct, parsed_answer, reports)
        """
        item = self.current_item
        reports = []
        done = False
        answer_correct = False
        parsed_answer = None
        
        for a in parsed_actions:
            name = a.name
            arg = (a.arg or "").strip() if a.arg else ""
            
            if name == "move_forward":
                self.view_engine.step("w")
            elif name == "move_backward":
                self.view_engine.step("s")
            elif name == "turn_left":
                self.view_engine.step("q")
            elif name == "turn_right":
                self.view_engine.step("e")
            elif name == "look_up":
                self.view_engine.step("r")
            elif name == "look_down":
                self.view_engine.step("f")
            elif name == "query_pose":
                if arg in self.named_views:
                    pose6_deg = self.named_views[arg]["pose6_deg"]
                    reports.append(f"[query_pose] {arg} -> {format_pose6_deg(pose6_deg)}")
                else:
                    reports.append(f"[query_pose] view not found: {arg}")
            elif name == "select_view":
                if arg in self.named_views:
                    view = self.named_views[arg]
                    self.camera_intrinsics = view["K"]
                    self.active_K_source = arg
                    self.view_engine.reset(view["E"])
                else:
                    reports.append(f"[select_view] view not found: {arg}")
            elif name == "get_view":
                se3_deg = parse_get_view_arg_deg(arg) if arg else None
                if se3_deg is not None:
                    se3_deg = np.array(se3_deg, dtype=np.float64)
                    E = c2w_se3_to_extrinsic(se3_deg)
                    self.view_engine.reset(E)
                else:
                    reports.append(f"[get_view] invalid pose: {arg}")
            elif name == "answer":
                parsed_answer = arg
                gold = item.get("answer", "")
                answer_correct = answer_match(parsed_answer, gold)
                done = True
                self.episode_done = True
                break
        
        return done, answer_correct, parsed_answer, reports
    
    def _render(self, init_obs: bool = True, question: str = "", choices: str = "", 
                status: str = "", reports: List[str] = None) -> Dict[str, Any]:
        """
        Render the current observation.
        
        Args:
            init_obs: Whether this is the initial observation
            question: Question text (for initial observation)
            choices: Choice text (for initial observation)
            status: Status message (for action observations)
            reports: List of report strings
            
        Returns:
            Observation dictionary
        """
        img_placeholder = self.config.get("image_placeholder", "<image>")
        
        if init_obs:
            # Build image placeholder string
            if self.images:
                image_str = " ".join([img_placeholder] * len(self.images))
            else:
                image_str = img_placeholder
            
            if self.use_tools:
                obs_str = init_observation_template(
                    observation=image_str,
                    question=question,
                    choices=choices,
                    named_views=list(self.named_views.keys()) if self.named_views else None,
                )
            else:
                obs_str = init_observation_template(
                    observation=image_str,
                    question=question,
                    choices=choices,
                )
            
            # Add format prompt
            format_prompt_text = self.format_prompt_func(
                max_actions_per_step=self.config.max_actions_per_step,
                action_sep=self.config.action_sep,
                add_example=True,
            )
            obs_str += "\n" + format_prompt_text
        else:
            # Build status observation
            if self.use_tools and self.view_engine:
                current_pose = c2w_extrinsic_to_se3(self.view_engine.get_pose())
                pose_str = format_pose6_deg(current_pose)
                obs_str = action_template(
                    observation=f"{img_placeholder}\nCurrent camera pose: {pose_str}",
                    status=status,
                    reports=reports or [],
                )
            else:
                obs_str = status
        
        obs = {
            "obs_str": obs_str,
            "multi_modal_data": {
                img_placeholder: self.images.copy(),
            }
        }
        
        return obs
    
    def system_prompt(self) -> str:
        """Get the system prompt for this environment."""
        if self.use_tools:
            return system_prompt_tool(
                step_translation=self.config.step_translation,
                step_rotation_deg=self.config.step_rotation_deg,
            )
        else:
            return system_prompt_no_tool()
    
    def compute_reward(self) -> float:
        """Compute final reward for the episode."""
        return 0.0
    
    def close(self):
        """Close the environment and release resources."""
        if self.renderer is not None:
            pass
