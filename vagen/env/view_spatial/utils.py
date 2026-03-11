# View Spatial Bench Environment Utilities
# Contains parsing and utility functions for the View Spatial environment.

import json
import re
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence
from dataclasses import dataclass
from PIL import Image
from scipy.spatial.transform import Rotation as R


# ========== IO Helpers ==========

def count_lines(jsonl_path: Path) -> int:
    """Count the number of lines in a JSONL file."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_jsonl_line_by_index(jsonl_path: Path, idx: int) -> Dict[str, Any]:
    """Read a specific line from a JSONL file by index."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"Index {idx} out of range for {jsonl_path}")


def load_jsonl_items(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load all items from a JSONL file."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_rel_image(jsonl_path: Path, rel: str, dataset_root: str = None) -> Path:
    """Resolve a relative image path."""
    if dataset_root is not None:
        base = Path(dataset_root)
    else:
        base = jsonl_path.parent
    if rel.startswith("./"):
        rel = rel[2:]
    return base / rel


def safe_open_rgb(path: Path) -> Optional[Image.Image]:
    """Safely open an image and convert to RGB."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ========== Action Parsing ==========

@dataclass
class ParsedAction:
    """Represents a parsed action with name and optional argument."""
    name: str
    arg: Optional[str]


def parse_free_think(response: str) -> Dict[str, Any]:
    """
    Parse the free_think format response.
    Expected format: <think>...</think><action>action1|action2|...|</action>
    
    Returns:
        Dict with 'ok', 'think', and 'actions_blob' keys
    """
    m = re.search(r"<think>(.*?)</think>\s*<action>(.*?)</action>", response, re.DOTALL | re.IGNORECASE)
    if not m:
        # Also try without think block (no_think format)
        m = re.search(r"<action>(.*?)</action>", response, re.DOTALL | re.IGNORECASE)
        if m:
            return {"ok": True, "think": "", "actions_blob": m.group(1).strip()}
        return {"ok": False, "think": "", "actions_blob": ""}
    return {"ok": True, "think": m.group(1).strip(), "actions_blob": m.group(2).strip()}


def _normalize_ws(s: str) -> str:
    """Normalize whitespace in a string."""
    return re.sub(r"\s+", " ", s.strip())


def parse_actions(actions_blob: str,
                  sep: str = "|",
                  allow_trailing_empty: bool = True) -> Tuple[bool, List[ParsedAction]]:
    """
    Parse action string like 'action1|action2|...|' into list of ParsedAction.
    
    Returns:
        Tuple of (format_ok, actions_list)
    """
    tokens = [t for t in (x.strip() for x in actions_blob.split(sep)) if t or not allow_trailing_empty]
    actions: List[ParsedAction] = []
    
    for tok in tokens:
        # Match action(arg) format
        m = re.fullmatch(r"([A-Za-z_]\w*)\((.*)\)", tok)
        if m:
            actions.append(ParsedAction(name=m.group(1).lower(), arg=_normalize_ws(m.group(2))))
            continue
        # Match simple action name
        m = re.fullmatch(r"([A-Za-z_]\w*)", tok)
        if m:
            actions.append(ParsedAction(name=m.group(1).lower(), arg=None))
            continue
        # Invalid token
        return False, []
    
    return True, actions


# No-tool mode: only answer action
def check_actions_no_tool(actions: List[ParsedAction]) -> bool:
    """Check that only answer action is used (no-tool mode)."""
    return len(actions) == 1 and actions[0].name == "answer" and actions[0].arg is not None


# Tool mode: allow navigation + tool actions
ALLOWED_TOOL_ACTIONS = {
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "query_pose",
    "select_view",
    "get_view",
    "answer",
}


def check_actions_tool(actions: List[ParsedAction]) -> bool:
    """
    Check that all actions are allowed for tool mode.
    'answer' must be the last action if present.
    """
    for i, a in enumerate(actions):
        if a.name not in ALLOWED_TOOL_ACTIONS:
            return False
        if a.name == "answer" and i != len(actions) - 1:
            return False
    return True


def answer_match(pred: str, gold: str) -> bool:
    """Check if the predicted answer matches the gold answer."""
    pred = pred.strip().lower()
    gold = gold.strip().lower()
    if not pred or not gold:
        return False
    return pred[0] == gold[0]


# ========== Pose Utilities ==========

def deg2rad(x: float) -> float:
    """Convert degrees to radians."""
    return x * math.pi / 180.0


def rad2deg(x: float) -> float:
    """Convert radians to degrees."""
    return x * 180.0 / math.pi


def check_4x4(M: np.ndarray, name: str = "matrix"):
    """Validate a (4,4) homogeneous matrix."""
    if not isinstance(M, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(M)}")
    if M.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4,4), got {M.shape}")


def extrinsic_c2w_to_w2c(camera_to_world: np.ndarray) -> np.ndarray:
    """Convert camera-to-world matrix to world-to-camera matrix."""
    check_4x4(camera_to_world, name="camera_to_world")
    return np.linalg.inv(camera_to_world)


def c2w_extrinsic_to_se3(E: np.ndarray) -> List[float]:
    """Convert 4x4 c2w extrinsic to [tx, ty, tz, rx, ry, rz] in degrees."""
    check_4x4(E, name="extrinsic")
    t = E[:3, 3]
    rot_matrix = E[:3, :3]
    euler_rad = R.from_matrix(rot_matrix).as_euler("xyz", degrees=False)
    euler_deg = [rad2deg(e) for e in euler_rad]
    return [float(t[0]), float(t[1]), float(t[2]),
            float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])]


def c2w_se3_to_extrinsic(pose6_deg: Sequence[float]) -> np.ndarray:
    """Convert [tx, ty, tz, rx, ry, rz] in degrees to 4x4 c2w extrinsic."""
    if len(pose6_deg) != 6:
        raise ValueError("6-DoF requires 6 values.")
    tx, ty, tz, rx_deg, ry_deg, rz_deg = [float(x) for x in pose6_deg]
    E = np.eye(4, dtype=np.float64)
    rx, ry, rz = deg2rad(rx_deg), deg2rad(ry_deg), deg2rad(rz_deg)
    E[:3, :3] = R.from_euler("xyz", [rx, ry, rz], degrees=False).as_matrix()
    E[:3, 3] = [tx, ty, tz]
    return E


def parse_get_view_arg_deg(arg: str) -> Optional[List[float]]:
    """Parse 'tx,ty,tz,rx,ry,rz' (angles in degrees) into list of floats."""
    try:
        parts = [p.strip() for p in arg.split(",")]
        if len(parts) != 6:
            return None
        return [float(x) for x in parts]
    except Exception:
        return None


def format_pose6_deg(p: List[float]) -> str:
    """Format 6-DoF pose for display."""
    tx, ty, tz, rx, ry, rz = p
    return f"[tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}, rx={rx:.2f}°, ry={ry:.2f}°, rz={rz:.2f}°]"


# ========== View Manipulator ==========

class ViewManipulator:
    """
    Camera pose controller for 3D environments.
    Maintains camera-to-world (c2w) transformation.
    """
    
    def __init__(
        self,
        step_translation: float = 0.3,
        step_rotation_deg: float = 30.0,
        world_up_axis: str = "Z",
        image_y_down: bool = True,
    ):
        self.step_t = float(step_translation)
        self.step_r_deg = float(step_rotation_deg)
        self.step_r = np.radians(self.step_r_deg)
        self.up_axis = world_up_axis.upper()
        assert self.up_axis in ("Z", "Y"), "world_up_axis must be 'Z' or 'Y'"
        self.image_y_down = bool(image_y_down)
        
        self.c2w = np.eye(4, dtype=np.float64)
    
    def reset(self, initial_extrinsic_c2w: np.ndarray = None) -> np.ndarray:
        """Reset to identity or provided camera-to-world matrix."""
        if initial_extrinsic_c2w is None:
            self.c2w = np.eye(4, dtype=np.float64)
        else:
            check_4x4(initial_extrinsic_c2w)
            self.c2w = initial_extrinsic_c2w.astype(np.float64)
        return self.get_pose()
    
    def get_pose(self, mode: str = "c2w") -> np.ndarray:
        """Get current camera pose."""
        if mode == "c2w":
            return self.c2w.copy()
        elif mode == "w2c":
            return extrinsic_c2w_to_w2c(self.c2w)
        else:
            raise ValueError("mode must be 'c2w' or 'w2c'")
    
    def _Rc_t_from_c2w(self, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract rotation and translation from c2w matrix."""
        return c2w[:3, :3].copy(), c2w[:3, 3].copy()
    
    def _compose_c2w(self, R_c2w: np.ndarray, C_world: np.ndarray) -> np.ndarray:
        """Compose c2w matrix from rotation and camera center."""
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = C_world
        return c2w
    
    def _translate_camera_center(self, C_world: np.ndarray, R_c2w: np.ndarray, delta: np.ndarray):
        """Translate camera center by delta."""
        self.c2w = self._compose_c2w(R_c2w, C_world + delta)
    
    def step(self, action: str) -> np.ndarray:
        """Execute a discrete movement action."""
        a = action.strip().lower()
        if a == "move_forward" or a == "w":
            self.move_forward(+self.step_t)
        elif a == "move_backward" or a == "s":
            self.move_forward(-self.step_t)
        elif a == "turn_left" or a == "q":
            self.yaw_camera(-self.step_r)
        elif a == "turn_right" or a == "e":
            self.yaw_camera(+self.step_r)
        elif a == "look_up" or a == "r":
            ang = (+self.step_r) if self.image_y_down else (-self.step_r)
            self.pitch_camera(ang)
        elif a == "look_down" or a == "f":
            ang = (-self.step_r) if self.image_y_down else (+self.step_r)
            self.pitch_camera(ang)
        return self.get_pose()
    
    def move_forward(self, distance: float):
        """Move camera forward along its view direction."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        dir_world = R_c2w @ np.array([0.0, 0.0, 1.0])
        self._translate_camera_center(C_world, R_c2w, dir_world * distance)
    
    def yaw_camera(self, angle_rad: float):
        """Rotate camera around its local Y axis."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        R_local = R.from_euler("y", angle_rad, degrees=False).as_matrix()
        R_new = R_c2w @ R_local
        self.c2w = self._compose_c2w(R_new, C_world)
    
    def pitch_camera(self, angle_rad: float):
        """Rotate camera around its local X axis."""
        R_c2w, C_world = self._Rc_t_from_c2w(self.c2w)
        R_local = R.from_euler("x", angle_rad, degrees=False).as_matrix()
        R_new = R_c2w @ R_local
        self.c2w = self._compose_c2w(R_new, C_world)


# ========== Fallback Camera Parameters ==========

def fallback_K() -> np.ndarray:
    """Generate fallback camera intrinsics."""
    fx = fy = 300.0
    cx = cy = 150.0
    K4 = np.array([
        [fx, 0.0, cx, 0.0],
        [0.0, fy, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return K4
