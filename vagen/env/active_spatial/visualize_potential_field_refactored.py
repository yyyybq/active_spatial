#!/usr/bin/env python3
"""
Refactored Potential Field Heatmap Visualization

This version REUSES the existing VisibilityChecker and CollisionDetector
to ensure consistency between visualization and actual environment behavior.

Key changes from visualize_potential_field_with_scene.py:
1. Uses VisibilityChecker.check_occlusion() for wall occlusion (no duplicate logic)
2. Uses CollisionDetector.check_collision() for collision detection (no duplicate logic)
3. Shared wall loading logic (structure.json parsing)
4. NEW: Path planning from init camera to optimal target using potential field

Usage:
    python visualize_potential_field_refactored.py [--scene_id 0267_840790]
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import argparse
from typing import Dict, Any, List, Tuple, Optional
import sys
import heapq
from dataclasses import dataclass, field

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# REUSE existing modules instead of duplicating logic!
from visibility_checker import VisibilityChecker, create_visibility_checker
from collision_detector import CollisionDetector, create_collision_detector
from spatial_potential_field import (
    SpatialPotentialField,
    create_potential_field,
    TaskType,
    normalize,
)
from env_config import ActiveSpatialEnvConfig


# ============================================================================
# Path Planning Module
# ============================================================================

@dataclass
class PlannerState:
    """State for path planning."""
    position: np.ndarray  # [x, y, z]
    yaw: float  # Yaw angle in radians
    
    def __hash__(self):
        # Discretize for hashing
        px = round(self.position[0], 2)
        py = round(self.position[1], 2)
        pz = round(self.position[2], 2)
        yaw_disc = round(self.yaw / (np.pi / 6)) * (np.pi / 6)  # 30 degree buckets
        return hash((px, py, pz, yaw_disc))
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def __lt__(self, other):
        # For heap comparison
        return hash(self) < hash(other)
    
    def get_forward(self) -> np.ndarray:
        """Get forward direction vector from yaw angle."""
        return np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])


@dataclass(order=True)
class PriorityItem:
    """Priority queue item for A* search."""
    priority: float
    state: PlannerState = field(compare=False)
    path: List[Tuple[str, PlannerState]] = field(compare=False)


class PathPlanner:
    """
    Plans optimal path from initial camera pose to best target pose.
    Uses A* search with potential field scores as heuristic.
    """
    
    # Available actions matching the environment
    ACTIONS = {
        "move_forward": (0.3, 0),      # (translation, rotation)
        "move_backward": (-0.3, 0),
        "turn_left": (0, np.radians(30)),
        "turn_right": (0, np.radians(-30)),
    }
    
    def __init__(
        self,
        field: SpatialPotentialField,
        scene_visualizer: 'SceneVisualizer',
        task: Dict[str, Any],
        step_translation: float = 0.3,
        step_rotation_deg: float = 30.0,
    ):
        self.field = field
        self.scene = scene_visualizer
        self.task = task
        self.step_t = step_translation
        self.step_r = np.radians(step_rotation_deg)
        
        # Update actions with actual step sizes
        self.ACTIONS = {
            "move_forward": (self.step_t, 0),
            "move_backward": (-self.step_t, 0),
            "turn_left": (0, self.step_r),
            "turn_right": (0, -self.step_r),
        }
        
        # Extract task info
        self.task_type = task.get("task_type", "default")
        self.target_region = task.get("target_region", {})
        self.task_params = task.get("task_params", {})
        self.target_object = task.get("target_object", {})
        
        # Get object center(s) for visibility checks
        # Handle both single-object and multi-object tasks
        params = self.target_region.get("params", {})
        self.object_center = None
        self.all_object_centers = []  # For multi-object tasks
        
        # First, collect all object centers for multi-object tasks
        object_a_center = params.get("object_a_center", None)
        object_b_center = params.get("object_b_center", None)
        object_c_center = params.get("object_c_center", None)
        
        if object_a_center is not None:
            center_a = np.array(object_a_center)
            if len(center_a) == 2:
                center_a = np.array([center_a[0], center_a[1], 1.5])
            self.all_object_centers.append(center_a)
        
        if object_b_center is not None:
            center_b = np.array(object_b_center)
            if len(center_b) == 2:
                center_b = np.array([center_b[0], center_b[1], 1.5])
            self.all_object_centers.append(center_b)
        
        if object_c_center is not None:
            center_c = np.array(object_c_center)
            if len(center_c) == 2:
                center_c = np.array([center_c[0], center_c[1], 1.5])
            self.all_object_centers.append(center_c)
        
        # Now determine the primary object_center
        if len(self.all_object_centers) >= 2:
            # Multi-object task: use centroid of all objects
            self.object_center = np.mean(self.all_object_centers, axis=0)
        elif len(self.all_object_centers) == 1:
            self.object_center = self.all_object_centers[0]
        elif params.get("object_center"):
            self.object_center = np.array(params["object_center"])
            if len(self.object_center) == 2:
                self.object_center = np.array([self.object_center[0], self.object_center[1], 1.5])
            self.all_object_centers.append(self.object_center)
        elif params.get("center"):
            center = params["center"]
            self.object_center = np.array([center[0], center[1], 1.5]) if len(center) == 2 else np.array(center)
            self.all_object_centers.append(self.object_center)
        elif "center" in self.target_object:
            center = self.target_object["center"]
            self.object_center = np.array(center) if len(center) == 3 else np.array([center[0], center[1], 1.5])
            self.all_object_centers.append(self.object_center)
        
        self.target_label = self.target_object.get("label", "")
    
    def apply_action(self, state: PlannerState, action: str) -> PlannerState:
        """Apply an action to get a new state."""
        trans, rot = self.ACTIONS[action]
        
        new_yaw = state.yaw + rot
        # Normalize yaw to [-pi, pi]
        while new_yaw > np.pi:
            new_yaw -= 2 * np.pi
        while new_yaw < -np.pi:
            new_yaw += 2 * np.pi
        
        # Translation along current forward direction
        forward = np.array([np.cos(state.yaw), np.sin(state.yaw), 0.0])
        new_pos = state.position.copy()
        new_pos[:2] = state.position[:2] + forward[:2] * trans
        
        return PlannerState(position=new_pos, yaw=new_yaw)
    
    def is_valid_state(self, state: PlannerState) -> bool:
        """Check if a state is valid (no collision, in room)."""
        # Check collision
        has_collision, _ = self.scene.check_collision(state.position)
        if has_collision:
            return False
        
        # Check if in a room
        if not self.scene.is_inside_any_room(state.position[:2]):
            return False
        
        return True
    
    def compute_score(self, state: PlannerState, use_actual_yaw: bool = True) -> float:
        """
        Compute potential field score for a state.
        
        Args:
            state: Current planner state (position + yaw)
            use_actual_yaw: If True, use state's actual yaw for scoring.
                           If False, assume optimal orientation toward target (for heatmap).
        
        When use_actual_yaw=True:
            - Uses state.get_forward() as camera direction
            - Path planning will optimize both position AND orientation
            - FoV constraints are properly enforced during navigation
        
        When use_actual_yaw=False:
            - Assumes camera faces the target object center
            - Only evaluates "best possible score at this position"
            - Used for heatmap visualization
        """
        if not self.is_valid_state(state):
            return -1.0
        
        # Get actual forward direction from state
        actual_forward = state.get_forward()
        
        # Check visibility for all target objects
        # Use relaxed visibility check: only check wall occlusion if NOT in same room
        # This allows planner to find paths through the space without being too restrictive
        if self.all_object_centers:
            for obj_center in self.all_object_centers:
                can_see, reason = self.scene.check_visibility(
                    state.position, obj_center, self.target_label,
                    check_wall_occlusion=True,
                    check_object_occlusion=False,  # Don't check object occlusion during planning
                    skip_same_room=True,  # Allow visibility if in same room
                )
                if not can_see:
                    # Return low score instead of 0 to allow some exploration
                    return 0.05
        elif self.object_center is not None:
            can_see, reason = self.scene.check_visibility(
                state.position, self.object_center, self.target_label,
                check_wall_occlusion=True,
                check_object_occlusion=False,
                skip_same_room=True,
            )
            if not can_see:
                return 0.05
        
        try:
            if use_actual_yaw:
                # Use actual camera orientation from state.yaw
                # This enables path planning to optimize orientation!
                forward = actual_forward
            else:
                # Use direction toward object center (for heatmap visualization)
                if self.object_center is not None:
                    forward = self.object_center - state.position
                    forward[2] = 0  # Only horizontal direction
                    forward_norm = np.linalg.norm(forward)
                    if forward_norm > 1e-6:
                        forward = forward / forward_norm
                    else:
                        forward = actual_forward
                else:
                    forward = actual_forward
            
            result = self.field.compute_score(
                camera_position=state.position,
                camera_forward=forward,
                task_type=self.task_type,
                task_params=self.task_params,
                target_region=self.target_region,
            )
            # Use total_score which combines position AND orientation scores
            return result.total_score
        except Exception:
            return 0.0
    
    def plan_path(
        self,
        init_pos: np.ndarray,
        init_forward: np.ndarray,
        max_steps: int = 50,
        score_threshold: float = 0.95,
    ) -> Tuple[List[Tuple[str, PlannerState]], float]:
        """
        Plan optimal path using A* search.
        
        Args:
            init_pos: Initial camera position [x, y, z]
            init_forward: Initial forward direction
            max_steps: Maximum number of steps
            score_threshold: Score threshold for success
            
        Returns:
            (path, final_score): List of (action, state) tuples and final score
        """
        # Compute initial yaw from forward direction
        init_yaw = np.arctan2(init_forward[1], init_forward[0])
        init_state = PlannerState(position=init_pos.copy(), yaw=init_yaw)
        
        # Priority queue: (priority, state, path)
        # Priority = steps - score (minimize steps, maximize score)
        init_score = self.compute_score(init_state)
        
        # Check if already at goal
        if init_score >= score_threshold:
            return [("start", init_state)], init_score
        
        # A* search
        pq = [PriorityItem(
            priority=-init_score,  # Negative because we want high scores
            state=init_state,
            path=[("start", init_state)]
        )]
        
        visited = {init_state}
        best_path = [("start", init_state)]
        best_score = init_score
        
        iterations = 0
        max_iterations = 5000  # Limit total iterations
        
        while pq and len(best_path) <= max_steps and iterations < max_iterations:
            iterations += 1
            item = heapq.heappop(pq)
            current_state = item.state
            current_path = item.path
            
            # Try each action
            for action_name, (trans, rot) in self.ACTIONS.items():
                new_state = self.apply_action(current_state, action_name)
                
                if new_state in visited:
                    continue
                
                if not self.is_valid_state(new_state):
                    continue
                
                visited.add(new_state)
                score = self.compute_score(new_state)
                
                new_path = current_path + [(action_name, new_state)]
                
                # Update best if this is better
                if score > best_score:
                    best_score = score
                    best_path = new_path
                
                # Check if we've reached the goal
                if score >= score_threshold:
                    return new_path, score
                
                # Add to queue if path is not too long
                if len(new_path) < max_steps:
                    # Priority: penalize long paths, reward high scores
                    priority = len(new_path) * 0.1 - score
                    heapq.heappush(pq, PriorityItem(
                        priority=priority,
                        state=new_state,
                        path=new_path
                    ))
        
        return best_path, best_score
    
    def greedy_path(
        self,
        init_pos: np.ndarray,
        init_forward: np.ndarray,
        max_steps: int = 50,
        score_threshold: float = 0.95,
    ) -> Tuple[List[Tuple[str, PlannerState]], float]:
        """
        Plan path using three-phase approach:
        
        Phase 1: Find best movement direction by sampling different yaw angles
        Phase 2: Move toward optimal position (following the gradient of position score)
        Phase 3: Adjust orientation to face target objects
        
        This solves the problem where greedy gets stuck in local minima because
        the initial forward direction may not lead to the optimal position.
        """
        init_yaw = np.arctan2(init_forward[1], init_forward[0])
        current_state = PlannerState(position=init_pos.copy(), yaw=init_yaw)
        
        path = [("start", current_state)]
        
        # Track best state found
        best_state = current_state
        best_score = self.compute_score(current_state)
        best_path = path.copy()
        
        # Phase 1: Find best movement direction
        # Sample multiple directions and find which one leads to highest position score
        best_direction_yaw = init_yaw
        best_direction_score = self.compute_score(current_state, use_actual_yaw=False)
        
        # Try different yaw angles (every 30 degrees = 12 directions)
        for angle_offset in range(0, 360, 30):
            test_yaw = init_yaw + np.radians(angle_offset - 180)  # Test all directions
            # Normalize yaw
            while test_yaw > np.pi:
                test_yaw -= 2 * np.pi
            while test_yaw < -np.pi:
                test_yaw += 2 * np.pi
            
            # Simulate moving forward in this direction
            test_pos = init_pos.copy()
            forward_dir = np.array([np.cos(test_yaw), np.sin(test_yaw), 0])
            test_pos[:2] = test_pos[:2] + forward_dir[:2] * 0.3 * 5  # 5 steps ahead
            
            test_state = PlannerState(position=test_pos, yaw=test_yaw)
            if self.is_valid_state(test_state):
                score = self.compute_score(test_state, use_actual_yaw=False)
                if score > best_direction_score:
                    best_direction_score = score
                    best_direction_yaw = test_yaw
        
        # Turn toward best direction if needed
        turn_steps = 0
        max_turn_steps = 6  # Max 6 turns (180 degrees)
        
        while turn_steps < max_turn_steps:
            # Check if we're facing the right direction
            yaw_diff = best_direction_yaw - current_state.yaw
            # Normalize
            while yaw_diff > np.pi:
                yaw_diff -= 2 * np.pi
            while yaw_diff < -np.pi:
                yaw_diff += 2 * np.pi
            
            if abs(yaw_diff) < np.radians(20):  # Within 20 degrees
                break
            
            # Turn toward best direction
            if yaw_diff > 0:
                action = "turn_left"
            else:
                action = "turn_right"
            
            next_state = self.apply_action(current_state, action)
            if self.is_valid_state(next_state):
                path.append((action, next_state))
                current_state = next_state
                turn_steps += 1
            else:
                break
        
        # Phase 2: Move toward optimal position
        movement_steps = max_steps - turn_steps - 12  # Reserve 12 for orientation
        
        for step in range(max(0, movement_steps)):
            position_score = self.compute_score(current_state, use_actual_yaw=False)
            if position_score >= score_threshold:
                break
            
            # Try move_forward first
            forward_state = self.apply_action(current_state, "move_forward")
            if self.is_valid_state(forward_state):
                forward_score = self.compute_score(forward_state, use_actual_yaw=False)
                if forward_score > position_score:
                    path.append(("move_forward", forward_state))
                    current_state = forward_state
                    continue
            
            # If forward doesn't improve, try slight turns
            best_action = None
            best_next_state = None
            best_next_score = position_score
            
            for action in ["turn_left", "turn_right"]:
                turned_state = self.apply_action(current_state, action)
                if not self.is_valid_state(turned_state):
                    continue
                
                # Check if moving forward after turn improves score
                forward_after_turn = self.apply_action(turned_state, "move_forward")
                if self.is_valid_state(forward_after_turn):
                    score = self.compute_score(forward_after_turn, use_actual_yaw=False)
                    if score > best_next_score:
                        best_next_score = score
                        best_next_state = turned_state
                        best_action = action
            
            if best_action and best_next_state:
                path.append((best_action, best_next_state))
                current_state = best_next_state
            else:
                break  # Can't improve position further
        
        # Track best position achieved
        pos_score = self.compute_score(current_state, use_actual_yaw=False)
        
        # Phase 3: Optimize orientation
        # Turn to face target for optimal actual score
        for _ in range(12):
            current_score = self.compute_score(current_state)
            if current_score >= score_threshold:
                break
            
            if current_score > best_score:
                best_score = current_score
                best_state = current_state
                best_path = path.copy()
            
            best_turn = None
            best_turn_state = None
            best_turn_score = current_score
            
            for action in ["turn_left", "turn_right"]:
                next_state = self.apply_action(current_state, action)
                if self.is_valid_state(next_state):
                    score = self.compute_score(next_state)
                    if score > best_turn_score:
                        best_turn_score = score
                        best_turn_state = next_state
                        best_turn = action
            
            if best_turn and best_turn_state:
                path.append((best_turn, best_turn_state))
                current_state = best_turn_state
            else:
                break
        
        # Final check
        final_score = self.compute_score(current_state)
        if final_score > best_score:
            best_score = final_score
            best_state = current_state
            best_path = path.copy()
        
        return best_path, best_score


class SceneVisualizer:
    """
    Scene visualization helper that REUSES VisibilityChecker and CollisionDetector.
    
    This ensures the visualization shows EXACTLY what the environment sees.
    
    IMPORTANT: Uses the same configuration parameters as ActiveSpatialEnv to ensure
    consistency between visualization and actual training behavior.
    """
    
    def __init__(self, gs_root: str, scene_id: str, config: Optional[ActiveSpatialEnvConfig] = None):
        self.gs_root = Path(gs_root)
        self.scene_id = scene_id
        self.scene_path = self.gs_root / scene_id
        
        # Use provided config or create default (ensures same defaults as training)
        if config is None:
            config = ActiveSpatialEnvConfig()
        self.config = config
        
        # REUSE existing checkers with SAME CONFIG as training environment!
        # This ensures visualization matches actual training behavior.
        self.visibility_checker = create_visibility_checker({
            "fov_horizontal": config.fov_horizontal,
            "fov_vertical": config.fov_vertical,
            "min_visible_size": config.min_visible_screen_coverage,
        })
        self.collision_detector = create_collision_detector({
            "camera_radius": config.collision_camera_radius,
            "floor_height": config.collision_floor_height,
            "ceiling_height": config.collision_ceiling_height,
            "safety_margin": config.collision_safety_margin,
            "enable_object_collision": True,
            "enable_boundary_collision": True,
        })
        
        # Load scene data (walls, objects)
        self.visibility_checker.load_scene(self.scene_path)
        self.collision_detector.load_scene(self.scene_path)
        
        # For plotting: extract data from the shared checkers
        self.room_profiles = self.visibility_checker.room_profiles
        self.wall_segments = self.visibility_checker.wall_segments
        self.objects = self._extract_objects_for_plotting()
    
    def _extract_objects_for_plotting(self) -> List[Dict]:
        """Extract object data for visualization from CollisionDetector."""
        objects = []
        
        # Read labels.json directly for plotting (we need more info than collision boxes)
        labels_path = self.scene_path / 'labels.json'
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                labels_data = json.load(f)
            
            for obj in labels_data:
                label = obj.get('label', '').lower()
                bbox = obj.get('bounding_box', [])
                
                if len(bbox) != 8:
                    continue
                
                # Skip floor/ceiling/lights for plotting
                if label in {'ceiling', 'floor', 'room', 'other', 'light', 'lamp', 
                            'chandelier', 'pendant', 'ceiling lamp', 'downlights',
                            'strip light', 'curtain', 'rug', 'carpet', 'mat'}:
                    continue
                
                points = np.array([[c['x'], c['y'], c['z']] for c in bbox])
                min_pt = points.min(axis=0)
                max_pt = points.max(axis=0)
                center = (min_pt + max_pt) / 2
                
                objects.append({
                    'label': label,
                    'min': min_pt,
                    'max': max_pt,
                    'center': center,
                })
        
        return objects
    
    def is_inside_any_room(self, point_2d: np.ndarray) -> bool:
        """Check if a 2D point is inside any room."""
        for room in self.room_profiles:
            if self._point_in_polygon(point_2d, room):
                return True
        return False
    
    def find_containing_room(self, point_2d: np.ndarray) -> Optional[int]:
        """Find which room contains this point."""
        for i, room in enumerate(self.room_profiles):
            if self._point_in_polygon(point_2d, room):
                return i
        return None
    
    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(polygon)
        if n < 3:
            return False
        
        inside = False
        x, y = point[0], point[1]
        p1x, p1y = polygon[0]
        
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def check_visibility(self, camera_pos: np.ndarray, target_pos: np.ndarray, 
                        target_label: str = "",
                        check_wall_occlusion: bool = True,
                        check_object_occlusion: bool = False,
                        skip_same_room: bool = True) -> Tuple[bool, str]:
        """
        Check visibility using the REAL VisibilityChecker.
        
        Args:
            camera_pos: Camera position in 3D
            target_pos: Target position in 3D
            target_label: Label of target object (to avoid self-occlusion)
            check_wall_occlusion: If True, check wall occlusion
            check_object_occlusion: If True, check object occlusion
            skip_same_room: If True, skip wall check when camera and target are in same room
        
        Returns:
            (can_see, reason): bool and explanation
            reason can be: "outside_room", "blocked_by_wall", "blocked_by_object", "visible"
        """
        camera_2d = camera_pos[:2]
        
        # Check if camera is inside a room
        if not self.is_inside_any_room(camera_2d):
            return False, "outside_room"
        
        # Use the REAL visibility checker for occlusion
        occlusion, occlusion_type = self.visibility_checker.check_occlusion(
            camera_position=camera_pos,
            target_center=target_pos,
            target_label=target_label,
            check_wall_occlusion=check_wall_occlusion,
            check_object_occlusion=check_object_occlusion,
            skip_same_room=skip_same_room,
        )
        
        if occlusion > 0.5:  # More than 50% occluded = blocked
            if occlusion_type == "wall":
                return False, "blocked_by_wall"
            elif occlusion_type == "object":
                return False, "blocked_by_object"
            else:
                return False, "blocked_by_wall"  # default to wall
        
        return True, "visible"
    
    def check_collision(self, camera_pos: np.ndarray, 
                       prev_pos: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        Check collision using the REAL CollisionDetector.
        
        Returns:
            (has_collision, collision_type)
        """
        result = self.collision_detector.check_collision(
            position=camera_pos,
            previous_position=prev_pos,
        )
        return result.has_collision, result.collision_type
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get scene bounds from room profiles."""
        if not self.room_profiles:
            return -5, 5, -5, 5
        
        all_points = np.vstack(self.room_profiles)
        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)
        return x_min, x_max, y_min, y_max
    
    def plot_on_axes(self, ax: plt.Axes, alpha: float = 0.6, show_labels: bool = False,
                     xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None):
        """Plot scene geometry on matplotlib axes."""
        # Plot room profiles (filled)
        for i, room in enumerate(self.room_profiles):
            color = plt.cm.Pastel1(i % 9)
            polygon = Polygon(room, alpha=0.15, facecolor=color, edgecolor='none')
            ax.add_patch(polygon)
        
        # Plot walls
        for wall_start, wall_end in self.wall_segments:
            ax.plot([wall_start[0], wall_end[0]], [wall_start[1], wall_end[1]], 
                   'k-', linewidth=2, alpha=0.8)
        
        # Plot objects
        for obj in self.objects:
            x_min, y_min = obj['min'][0], obj['min'][1]
            width = obj['max'][0] - obj['min'][0]
            height = obj['max'][1] - obj['min'][1]
            center_x, center_y = obj['center'][0], obj['center'][1]
            
            # Skip objects outside view bounds if specified
            if xlim is not None and ylim is not None:
                if center_x < xlim[0] or center_x > xlim[1] or center_y < ylim[0] or center_y > ylim[1]:
                    # Still draw the rectangle but skip label
                    rect = Rectangle((x_min, y_min), width, height,
                                    linewidth=0.8, edgecolor='gray',
                                    facecolor='lightgray', alpha=alpha)
                    ax.add_patch(rect)
                    continue
            
            rect = Rectangle((x_min, y_min), width, height,
                            linewidth=0.8, edgecolor='gray',
                            facecolor='lightgray', alpha=alpha)
            ax.add_patch(rect)
            
            if show_labels:
                # Improved label display with background
                label = obj['label']
                # Truncate long labels
                if len(label) > 12:
                    label = label[:10] + '..'
                ax.text(center_x, center_y, label,
                       fontsize=8, ha='center', va='center',
                       fontweight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                alpha=0.8, edgecolor='none'))


def load_tasks_from_jsonl(jsonl_path: str) -> List[Dict]:
    """Load tasks from JSONL file."""
    tasks = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def get_unique_tasks_by_type(tasks: List[Dict]) -> Dict[str, Dict]:
    """Get one representative task for each task type."""
    unique = {}
    for task in tasks:
        task_type = task.get("task_type", "unknown")
        if task_type not in unique:
            unique[task_type] = task
    return unique


def compute_forward_direction(camera_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Compute normalized forward direction from camera to target."""
    direction = target_pos[:2] - camera_pos[:2]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.array([1.0, 0.0])
    return direction / norm


def compute_heatmap_using_shared_checkers(
    field: SpatialPotentialField,
    scene: SceneVisualizer,
    task: Dict[str, Any],
    grid_size: int = 100,
    padding: float = 0.5,
    check_wall_occlusion: bool = True,
    check_object_occlusion: bool = False,
    skip_same_room: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute potential field scores using SHARED visibility and collision checkers.
    
    This ensures the heatmap shows exactly what the environment would compute!
    
    Args:
        field: SpatialPotentialField instance
        scene: SceneVisualizer instance
        task: Task dictionary
        grid_size: Grid size for heatmap
        padding: Padding around scene bounds
        check_wall_occlusion: If True, check wall occlusion
        check_object_occlusion: If True, check object occlusion
        skip_same_room: If True, skip wall check when camera and target are in same room
    """
    task_type = task.get("task_type", "default")
    target_region = task.get("target_region", {})
    task_params = task.get("task_params", {})
    params = target_region.get("params", {})
    region_type = target_region.get("type", "unknown")
    
    # Extract object positions (same logic as before)
    target_object = task.get("target_object", {})
    object_a_center = params.get("object_a_center", None)
    object_b_center = params.get("object_b_center", None)
    object_c_center = params.get("object_c_center", None)
    
    # Convert to numpy arrays
    if object_a_center is not None:
        object_a_center = np.array(object_a_center)
        if len(object_a_center) == 2:
            object_a_center = np.array([object_a_center[0], object_a_center[1], 1.5])
    if object_b_center is not None:
        object_b_center = np.array(object_b_center)
        if len(object_b_center) == 2:
            object_b_center = np.array([object_b_center[0], object_b_center[1], 1.5])
    if object_c_center is not None:
        object_c_center = np.array(object_c_center)
        if len(object_c_center) == 2:
            object_c_center = np.array([object_c_center[0], object_c_center[1], 1.5])
    
    # Get primary object center
    object_center = None
    if region_type in ["apollonius_circle", "line"] and object_a_center is not None and object_b_center is not None:
        object_center = (object_a_center + object_b_center) / 2
    elif params.get("object_center"):
        object_center = np.array(params["object_center"])
    elif params.get("center") and region_type == "circle":
        object_center = np.array(params["center"])
    elif "center" in target_object:
        object_center = np.array(target_object["center"])
    
    if object_center is None or (len(object_center) >= 2 and np.allclose(object_center[:2], [0, 0])):
        if object_a_center is not None and object_b_center is not None:
            object_center = (object_a_center + object_b_center) / 2
        elif object_a_center is not None:
            object_center = object_a_center.copy()
    
    if object_center is None:
        object_center = np.array([0, 0, 0])
    elif len(object_center) == 2:
        object_center = np.array([object_center[0], object_center[1], 1.5])
    
    # Target objects list for multi-object tasks
    target_objects_list = []
    if "objects" in target_object:
        for obj in target_object["objects"]:
            center = obj.get("center", None)
            if center is not None:
                center = np.array(center)
                if len(center) == 2:
                    center = np.array([center[0], center[1], 1.5])
                target_objects_list.append({
                    "label": obj.get("label", "object"),
                    "center": center
                })
    elif "center" in target_object and "label" in target_object:
        center = np.array(target_object["center"])
        if len(center) == 2:
            center = np.array([center[0], center[1], 1.5])
        target_objects_list.append({
            "label": target_object.get("label", "object"),
            "center": center
        })
    
    # Grid bounds
    x_min, x_max, y_min, y_max = scene.get_bounds()
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    
    camera_height = target_region.get("height", 1.5)
    target_2d = object_center[:2]
    target_label = target_object.get("label", "")
    
    # Find target room
    target_room_idx = scene.find_containing_room(target_2d)
    
    # Initialize arrays
    valid_mask = np.zeros((grid_size, grid_size))
    final_scores = np.zeros((grid_size, grid_size))
    visibility_reasons = np.zeros((grid_size, grid_size), dtype=int)
    collision_mask = np.zeros((grid_size, grid_size), dtype=int)  # NEW: track collision separately
    
    # Visibility reason codes:
    # 0 = outside target room
    # 1 = blocked by object (visibility)
    # 2 = blocked by wall (visibility)
    # 3 = visible and no collision
    # 4 = visible but has collision (NEW)
    
    # Compute scores
    for i in range(grid_size):
        for j in range(grid_size):
            camera_pos = np.array([X[i, j], Y[i, j], camera_height])
            camera_2d = camera_pos[:2]
            
            # Check if in target room
            camera_room_idx = scene.find_containing_room(camera_2d)
            if camera_room_idx is None or camera_room_idx != target_room_idx:
                visibility_reasons[i, j] = 0  # outside target room
                continue
            
            # Use SHARED collision checker FIRST (check if position is inside an object)
            has_collision, collision_type = scene.check_collision(camera_pos)
            if has_collision and collision_type in ["object", "wall"]:
                collision_mask[i, j] = 1  # Mark as collision area
                visibility_reasons[i, j] = 4  # Collision (unreachable)
                continue
            
            # Use SHARED visibility checker - check ALL target objects for multi-object tasks
            all_visible = True
            visibility_reason = "visible"
            
            # Build list of all object centers to check
            all_centers_to_check = []
            if object_a_center is not None:
                all_centers_to_check.append(object_a_center)
            if object_b_center is not None:
                all_centers_to_check.append(object_b_center)
            if object_c_center is not None:
                all_centers_to_check.append(object_c_center)
            
            # If no individual centers, use the primary object_center
            if not all_centers_to_check:
                all_centers_to_check = [object_center]
            
            # Check visibility for all target objects
            for center in all_centers_to_check:
                can_see, reason = scene.check_visibility(
                    camera_pos, center, target_label,
                    check_wall_occlusion=check_wall_occlusion,
                    check_object_occlusion=check_object_occlusion,
                    skip_same_room=skip_same_room,
                )
                if not can_see:
                    all_visible = False
                    visibility_reason = reason
                    break
            
            if not all_visible:
                # visibility_reasons: 1 = blocked by object, 2 = blocked by wall
                if visibility_reason == "blocked_by_object":
                    visibility_reasons[i, j] = 1
                elif visibility_reason == "blocked_by_wall":
                    visibility_reasons[i, j] = 2
                else:
                    visibility_reasons[i, j] = 2  # default to wall
                continue
            
            # Valid position - compute score
            visibility_reasons[i, j] = 3  # visible
            valid_mask[i, j] = 1.0
            
            forward = compute_forward_direction(camera_pos, object_center)
            forward_3d = np.array([forward[0], forward[1], 0])
            
            try:
                result = field.compute_score(
                    camera_position=camera_pos,
                    camera_forward=forward_3d,
                    task_type=task_type,
                    task_params=task_params,
                    target_region=target_region,
                )
                score = result.position_score
            except Exception:
                score = 0.0
            
            final_scores[i, j] = score
    
    # ========== NEW: Extract initial camera pose ==========
    init_camera = task.get("init_camera", {})
    init_camera_pos = None
    init_camera_forward = None
    
    if init_camera and "extrinsics" in init_camera:
        # Extract camera position from extrinsics (4x4 camera-to-world matrix)
        extrinsics = np.array(init_camera["extrinsics"])
        # Position is the translation part (last column, first 3 rows)
        init_camera_pos = extrinsics[:3, 3]
        # Forward direction is the third column (Z-axis of camera)
        init_camera_forward = extrinsics[:3, 2]
    
    # Extract target sample point and forward from target_region
    sample_point = target_region.get("sample_point", None)
    sample_forward = target_region.get("sample_forward", None)
    
    if sample_point is not None:
        sample_point = np.array(sample_point)
    if sample_forward is not None:
        sample_forward = np.array(sample_forward)
    # ========== END: Extract initial camera pose ==========
    
    # ========== NEW: Path planning from init to optimal target ==========
    planned_path = None
    planned_final_score = None
    planned_target_pos = None
    planned_target_forward = None
    
    if init_camera_pos is not None and init_camera_forward is not None:
        # Create path planner
        planner = PathPlanner(
            field=field,
            scene_visualizer=scene,
            task=task,
            step_translation=0.3,
            step_rotation_deg=30.0,
        )
        
        # Plan path using greedy search with orientation refinement
        planned_path, planned_final_score = planner.greedy_path(
            init_pos=init_camera_pos,
            init_forward=init_camera_forward,
            max_steps=50,
            score_threshold=0.95,  # Higher threshold for better results
        )
        
        # If greedy doesn't find good solution, try A* (slower but better)
        if planned_final_score < 0.85:
            planned_path_astar, planned_final_score_astar = planner.plan_path(
                init_pos=init_camera_pos,
                init_forward=init_camera_forward,
                max_steps=50,
                score_threshold=0.95,
            )
            if planned_final_score_astar > planned_final_score:
                planned_path = planned_path_astar
                planned_final_score = planned_final_score_astar
        
        # Extract final position from path
        if planned_path and len(planned_path) > 0:
            _, final_state = planned_path[-1]
            planned_target_pos = final_state.position
            planned_target_forward = final_state.get_forward()
    # ========== END: Path planning ==========
    
    metadata = {
        "object_center": object_center,
        "object_a_center": object_a_center,
        "object_b_center": object_b_center,
        "object_c_center": object_c_center,
        "target_objects_list": target_objects_list,
        "task_type": task_type,
        "task_description": task.get("task_description", ""),
        "region_type": region_type,
        "params": params,
        "visibility_reasons": visibility_reasons,
        "collision_mask": collision_mask,  # NEW: collision mask
        "target_room_idx": target_room_idx,
        "camera_height": camera_height,
        "target_2d": target_2d,
        # Camera pose information
        "init_camera_pos": init_camera_pos,
        "init_camera_forward": init_camera_forward,
        "sample_point": sample_point,  # Original sample point from data (for reference)
        "sample_forward": sample_forward,
        # NEW: Planned path information
        "planned_path": planned_path,
        "planned_final_score": planned_final_score,
        "planned_target_pos": planned_target_pos,
        "planned_target_forward": planned_target_forward,
    }
    
    return X, Y, valid_mask, final_scores, metadata


def plot_heatmap(
    X: np.ndarray,
    Y: np.ndarray,
    valid_mask: np.ndarray,
    final_scores: np.ndarray,
    metadata: Dict[str, Any],
    scene: SceneVisualizer,
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot potential field heatmap with collision visualization and room-focused view."""
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('score_cmap', colors, N=256)
    
    vis_reasons = metadata.get("visibility_reasons", np.zeros_like(valid_mask, dtype=int))
    collision_mask = metadata.get("collision_mask", np.zeros_like(valid_mask, dtype=int))
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
    
    # Prepare visibility display for room-focused views
    # 0 = outside target room (light gray)
    # 1 = blocked by visibility (red)
    # 2 = collision/unreachable (dark gray)
    # 3 = visible and reachable (green)
    vis_display = np.zeros_like(valid_mask)
    vis_display[vis_reasons == 0] = 0  # Outside target room
    vis_display[vis_reasons == 1] = 1  # Blocked by object (visibility)
    vis_display[vis_reasons == 2] = 1  # Blocked by wall (visibility)
    vis_display[vis_reasons == 3] = 3  # Visible and reachable
    vis_display[vis_reasons == 4] = 2  # Collision (unreachable)
    
    validity_cmap = LinearSegmentedColormap.from_list('validity', 
        ['#e0e0e0', '#d73027', '#505050', '#1a9850'], N=4)
    
    masked_scores = np.where(valid_mask > 0.5, final_scores, np.nan)
    collision_overlay = np.where(vis_reasons == 4, 0.5, np.nan)
    
    task_type = metadata["task_type"].replace("_", " ").title()
    region_type = metadata["region_type"]
    
    # Create figure with gridspec for better control
    fig = plt.figure(figsize=(20, 8))
    
    # Use GridSpec: left plot takes 40%, right two plots share 60%
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # ==================== Plot 1: Full scene - Potential Field Score ====================
    im1 = ax1.pcolormesh(X, Y, masked_scores, cmap=cmap, shading='auto', vmin=0, vmax=1)
    # Overlay collision areas with gray
    ax1.pcolormesh(X, Y, collision_overlay, cmap='gray', shading='auto', alpha=0.7, vmin=0, vmax=1)
    
    scene.plot_on_axes(ax1, alpha=0.5, show_labels=False)
    _add_task_annotations(ax1, metadata)
    
    ax1.set_title(f"Full Scene: Potential Field\n{task_type} ({region_type.upper()})",
                 fontsize=11, fontweight='bold')
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.2, linestyle=':')
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7, pad=0.02)
    cbar1.set_label('Score', fontsize=9)
    
    # ==================== Plot 2: Room-focused - Visibility & Collision ====================
    im2 = ax2.pcolormesh(X, Y, vis_display, cmap=validity_cmap, shading='auto', vmin=0, vmax=3)
    
    if room_bounds:
        ax2.set_xlim(room_bounds[0], room_bounds[1])
        ax2.set_ylim(room_bounds[2], room_bounds[3])
        scene.plot_on_axes(ax2, alpha=0.5, show_labels=True,
                          xlim=(room_bounds[0], room_bounds[1]),
                          ylim=(room_bounds[2], room_bounds[3]))
    else:
        scene.plot_on_axes(ax2, alpha=0.5, show_labels=True)
    
    _add_task_annotations(ax2, metadata)
    
    ax2.set_title("Target Room: Visibility & Collision\nGray=Collision, Red=Blocked, Green=Valid",
                 fontsize=11, fontweight='bold')
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3, linestyle='-')
    
    # ==================== Plot 3: Room-focused - Potential Field Score ====================
    im3 = ax3.pcolormesh(X, Y, masked_scores, cmap=cmap, shading='auto', vmin=0, vmax=1)
    ax3.pcolormesh(X, Y, collision_overlay, cmap='gray', shading='auto', alpha=0.7, vmin=0, vmax=1)
    
    if room_bounds:
        ax3.set_xlim(room_bounds[0], room_bounds[1])
        ax3.set_ylim(room_bounds[2], room_bounds[3])
        scene.plot_on_axes(ax3, alpha=0.5, show_labels=True,
                          xlim=(room_bounds[0], room_bounds[1]),
                          ylim=(room_bounds[2], room_bounds[3]))
    else:
        scene.plot_on_axes(ax3, alpha=0.5, show_labels=True)
    
    _add_task_annotations(ax3, metadata)
    
    ax3.set_title(f"Target Room: Potential Field\nGray=Collision (unreachable)",
                 fontsize=11, fontweight='bold')
    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Y (m)', fontsize=10)
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(True, alpha=0.3, linestyle='-')
    
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.7, pad=0.02)
    cbar3.set_label('Score', fontsize=9)
    
    # Use subplots_adjust instead of tight_layout for GridSpec
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")
    
    if show:
        plt.show()
    
    return fig


def _add_task_annotations(ax: plt.Axes, metadata: Dict[str, Any]):
    """Add target object markers and initial camera pose to the plot."""
    # Plot target objects (multi-object support)
    target_objects = metadata.get("target_objects_list", [])
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    if target_objects:
        for idx, obj in enumerate(target_objects):
            center = obj.get("center")
            if center is not None:
                color = colors[idx % len(colors)]
                ax.scatter(center[0], center[1], c=color, s=150, marker='*', 
                          edgecolors='white', linewidths=1.5, zorder=10,
                          label=obj.get("label", f"Object {idx+1}"))
    
    # Also plot object_a, object_b, object_c if present
    for key, color, marker in [
        ("object_a_center", "red", "^"),
        ("object_b_center", "blue", "s"),
        ("object_c_center", "green", "D"),
    ]:
        center = metadata.get(key)
        if center is not None:
            # Check if already plotted as part of target_objects
            already_plotted = False
            for obj in target_objects:
                if np.allclose(obj.get("center", [])[:2], center[:2], atol=0.1):
                    already_plotted = True
                    break
            
            if not already_plotted:
                ax.scatter(center[0], center[1], c=color, s=100, marker=marker,
                          edgecolors='white', linewidths=1, zorder=9)
    
    # ========== NEW: Plot initial camera pose ==========
    init_camera_pos = metadata.get("init_camera_pos")
    init_camera_forward = metadata.get("init_camera_forward")
    
    if init_camera_pos is not None:
        # Plot camera position as a triangle marker
        ax.scatter(init_camera_pos[0], init_camera_pos[1], 
                  c='cyan', s=200, marker='o', 
                  edgecolors='black', linewidths=2, zorder=15,
                  label='Init Camera')
        
        # Draw camera forward direction as an arrow
        if init_camera_forward is not None:
            arrow_length = 0.5  # Arrow length in meters
            forward_2d = np.array(init_camera_forward[:2])
            forward_norm = np.linalg.norm(forward_2d)
            if forward_norm > 1e-6:
                forward_2d = forward_2d / forward_norm * arrow_length
                ax.annotate('', 
                    xy=(init_camera_pos[0] + forward_2d[0], 
                        init_camera_pos[1] + forward_2d[1]),
                    xytext=(init_camera_pos[0], init_camera_pos[1]),
                    arrowprops=dict(arrowstyle='->', color='cyan', lw=2.5),
                    zorder=14)
    
    # ========== NEW: Plot target sample point (from dataset - now shown as reference) ==========
    sample_point = metadata.get("sample_point")
    sample_forward = metadata.get("sample_forward")
    
    if sample_point is not None:
        # Plot target sample position as a small marker (reference only)
        ax.scatter(sample_point[0], sample_point[1], 
                  c='gray', s=80, marker='x', 
                  linewidths=2, zorder=12,
                  label='Dataset Target (ref)')
    
    # ========== NEW: Plot planned path and target ==========
    planned_path = metadata.get("planned_path")
    planned_target_pos = metadata.get("planned_target_pos")
    planned_target_forward = metadata.get("planned_target_forward")
    planned_final_score = metadata.get("planned_final_score")
    
    if planned_path and len(planned_path) > 1:
        # Extract path coordinates
        path_x = [state.position[0] for _, state in planned_path]
        path_y = [state.position[1] for _, state in planned_path]
        
        # Plot path as a line with gradient color
        num_points = len(path_x)
        for i in range(num_points - 1):
            # Color gradient from yellow to green
            t = i / max(1, num_points - 2)
            color = plt.cm.YlGn(0.3 + t * 0.6)
            ax.plot([path_x[i], path_x[i+1]], [path_y[i], path_y[i+1]], 
                   color=color, linewidth=3, alpha=0.8, zorder=13)
        
        # Plot path waypoints
        for i, (action, state) in enumerate(planned_path):
            if i == 0:  # Start point (already plotted as init camera)
                continue
            elif i == len(planned_path) - 1:  # End point (will be plotted separately)
                continue
            else:
                # Intermediate waypoints
                ax.scatter(state.position[0], state.position[1], 
                          c='yellow', s=40, marker='.', 
                          edgecolors='black', linewidths=0.5, zorder=14)
    
    # Plot planned target position (final destination)
    if planned_target_pos is not None:
        score_text = f" (score={planned_final_score:.2f})" if planned_final_score else ""
        num_steps = len(planned_path) - 1 if planned_path else 0
        
        ax.scatter(planned_target_pos[0], planned_target_pos[1], 
                  c='lime', s=250, marker='*', 
                  edgecolors='black', linewidths=2, zorder=16,
                  label=f'Planned Target{score_text}\n({num_steps} steps)')
        
        # Draw planned target forward direction
        if planned_target_forward is not None:
            arrow_length = 0.5
            forward_2d = np.array(planned_target_forward[:2])
            forward_norm = np.linalg.norm(forward_2d)
            if forward_norm > 1e-6:
                forward_2d = forward_2d / forward_norm * arrow_length
                ax.annotate('', 
                    xy=(planned_target_pos[0] + forward_2d[0], 
                        planned_target_pos[1] + forward_2d[1]),
                    xytext=(planned_target_pos[0], planned_target_pos[1]),
                    arrowprops=dict(arrowstyle='->', color='lime', lw=3),
                    zorder=15)
    # ========== END: Planned path visualization ==========
    
    # Legend
    ax.legend(loc='upper right', fontsize=7)


def visualize_all_tasks(
    jsonl_path: str,
    gs_root: str,
    scene_id: str,
    output_dir: str = "./heatmaps_refactored",
    grid_size: int = 100,
    show: bool = False,
    config: Optional[ActiveSpatialEnvConfig] = None,
):
    """Generate heatmaps for all task types using shared checkers.
    
    Args:
        jsonl_path: Path to task JSONL file
        gs_root: Root directory for 3D Gaussian scenes
        scene_id: Scene identifier
        output_dir: Output directory for heatmaps
        grid_size: Grid resolution for heatmaps
        show: Whether to display plots interactively
        config: Environment config (uses default if None, ensuring consistency with training)
    """
    # Use provided config or create default (same as training)
    if config is None:
        config = ActiveSpatialEnvConfig()
    
    print(f"Loading tasks from {jsonl_path}...")
    tasks = load_tasks_from_jsonl(jsonl_path)
    unique_tasks = get_unique_tasks_by_type(tasks)
    print(f"Found {len(unique_tasks)} unique task types")
    
    print(f"\nLoading scene using SHARED VisibilityChecker and CollisionDetector...")
    print(f"  Config: floor_height={config.collision_floor_height}, ceiling_height={config.collision_ceiling_height}")
    print(f"  Config: camera_radius={config.collision_camera_radius}, fov={config.fov_horizontal}°")
    scene = SceneVisualizer(gs_root, scene_id, config=config)
    
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
    
    for task_type, task in unique_tasks.items():
        print(f"\nProcessing: {task_type}")
        
        try:
            X, Y, valid_mask, final_scores, metadata = compute_heatmap_using_shared_checkers(
                field, scene, task, grid_size=grid_size, padding=0.3,
            )
            
            output_path = output_dir_path / f"heatmap_{task_type}_refactored.png"
            
            fig = plot_heatmap(
                X, Y, valid_mask, final_scores, metadata,
                scene, output_path=str(output_path), show=show
            )
            plt.close(fig)
            
        except Exception as e:
            print(f"  Error processing {task_type}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nAll heatmaps saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Spatial Potential Field (Refactored - Uses Shared Checkers)"
    )
    parser.add_argument("--scene_id", type=str, default="0267_840790")
    parser.add_argument("--gs_root", type=str, 
                       default="/scratch/by2593/project/Active_Spatial/InteriorGS")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./potential_field_visualizations")
    parser.add_argument("--grid_size", type=int, default=100)
    parser.add_argument("--show", action="store_true")
    
    args = parser.parse_args()
    
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "active_spatial"
    
    jsonl_path = data_dir / f"train_data_{args.scene_id}.jsonl"
    
    if not jsonl_path.exists():
        print(f"Error: JSONL file not found at {jsonl_path}")
        return
    
    visualize_all_tasks(
        jsonl_path=str(jsonl_path),
        gs_root=args.gs_root,
        scene_id=args.scene_id,
        output_dir=args.output_dir,
        grid_size=args.grid_size,
        show=args.show,
    )


if __name__ == "__main__":
    main()
