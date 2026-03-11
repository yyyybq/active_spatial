# View Spatial Bench Environment Prompts
# This file defines system prompts and format configurations for the view spatial QA task.

FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><action>answer(X)|</action>",
        "example": """<think>Looking at the image, I can see that object A is to the left of object B based on the relative positions visible from this viewpoint.</think><action>answer(A)|</action>"""
    },
    "no_think": {
        "description": "You should provide only your answer.",
        "format": "<action>answer(X)|</action>",
        "example": """<action>answer(A)|</action>"""
    },
    "grounding": {
        "description": "You should first describe what you observe, then reason about the answer.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><action>answer(X)|</action>",
        "example": """<think><observation>I can see a living room with a sofa on the left side, a coffee table in the center, and a bookshelf on the right. The vase mentioned in the question appears to be on the coffee table.</observation><reasoning>Since the vase is on the coffee table which is in the center of the room, and the question asks about its position relative to the sofa, the answer is that the vase is to the right of the sofa.</reasoning></think><action>answer(B)|</action>"""
    },
    "tool_free_think": {
        "description": "You can explore the scene before answering. First give your thought process, then your actions.",
        "format": "<think>...</think><action>action1|action2|...|answer(X)|</action>",
        "example": """<think>I need to see the scene from a different angle to determine the spatial relationship. Let me turn right to get a better view.</think><action>turn_right|answer(B)|</action>"""
    },
}

# Action descriptions for tool mode
TOOL_ACTION_DESCRIPTIONS = """
Available actions (use | as separator, end with |):
- move_forward: Move the camera forward by a fixed step
- move_backward: Move the camera backward by a fixed step  
- turn_left: Rotate the camera left (yaw) by a fixed angle
- turn_right: Rotate the camera right (yaw) by a fixed angle
- look_up: Tilt the camera upward (pitch) by a fixed angle
- look_down: Tilt the camera downward (pitch) by a fixed angle
- query_pose(view_name): Get the 6-DoF pose of a named view
- select_view(view_name): Switch to a named view and render
- get_view(tx,ty,tz,rx,ry,rz): Set camera pose directly (angles in degrees)
- answer(X): Submit your final answer (X should be A, B, C, or D)
"""

# Action descriptions for no-tool mode
NO_TOOL_ACTION_DESCRIPTIONS = """
Available action:
- answer(X): Submit your answer where X is one of A, B, C, or D
"""


def system_prompt(**kwargs):
    """Generate the system prompt for the View Spatial environment."""
    selected_format = kwargs.get("format", "free_think")
    use_tools = kwargs.get("use_tools", False)
    step_translation = kwargs.get("step_translation", 0.3)
    step_rotation_deg = kwargs.get("step_rotation_deg", 30.0)
    
    if use_tools:
        format_config = FORMAT_CONFIGS.get("tool_free_think", FORMAT_CONFIGS["free_think"])
        action_desc = TOOL_ACTION_DESCRIPTIONS
        step_info = f"\nStep sizes: translation = {step_translation:.2f} meters, rotation = {step_rotation_deg:.1f} degrees.\n"
    else:
        format_config = FORMAT_CONFIGS.get(selected_format, FORMAT_CONFIGS["free_think"])
        action_desc = NO_TOOL_ACTION_DESCRIPTIONS
        step_info = ""
    
    base_prompt = f"""You are a spatial reasoning agent. You are given a question about a 3D scene and one or more images. Your task is to answer the question based on the visual information provided.

{action_desc}
{step_info}
{format_config['description']}

Response format: {format_config['format']}

Example:
{format_config['example']}

Rewards:
- Format correct: +0.2
- Correct answer: +0.8

Hints:
1. Carefully analyze the spatial relationships in the images
2. Consider the viewpoint when reasoning about spatial relations
3. The answer should be one of the given choices (A, B, C, or D)
"""
    return base_prompt


def system_prompt_no_tool() -> str:
    """Generate system prompt for no-tool mode."""
    return (
        "You are a spatial reasoning agent. You are given a question and a set of images. "
        "You need to answer the question based on the images.\n"
        "Your answer should be in the format of: <think>...</think><action>answer(X)|</action>\n"
        "X should be one of the choices A, B, C, or D."
    )


def system_prompt_tool(step_translation: float = 0.3, step_rotation_deg: float = 30.0) -> str:
    """Generate system prompt for tool mode."""
    return (
        "You are a spatial reasoning agent in a multi-view indoor environment. "
        "Before answering the question, you may explore the scene by issuing camera-control actions. "
        "Always respond using the free_think format:\n"
        "<think>...</think><action>action1|action2|...|</action>\n\n"
        "IMPORTANT CONVENTION:\n"
        " - All angles YOU input or see in the observations are in DEGREES.\n"
        " - The environment internally uses radians and extrinsic matrices, but you do not need to convert them.\n\n"
        "Supported actions (arguments are inside parentheses):\n"
        " - move_forward : move forward on the ground plane by a fixed step (meters).\n"
        " - move_backward: move backward on the ground plane by a fixed step (meters).\n"
        " - turn_left    : yaw left by a fixed angle (degrees).\n"
        " - turn_right   : yaw right by a fixed angle (degrees).\n"
        " - look_up      : pitch up by a fixed angle (degrees).\n"
        " - look_down    : pitch down by a fixed angle (degrees).\n"
        " - query_pose(view_name) : return the 6-DoF pose of the named view in DEGREES; does NOT change the camera.\n"
        " - select_view(view_name): reset the camera to the named view and render an image.\n"
        " - get_view(tx,ty,tz,rx,ry,rz): directly set camera w2c pose with Euler XYZ in DEGREES and render.\n"
        " - answer(X) where X in {A,B,C,D}: submit your final answer and terminate the episode.\n\n"
        f"Step sizes: translation={step_translation:.3f} m, yaw/pitch={step_rotation_deg:.1f} degrees.\n"
        "6-DoF pose definition follows world->camera (extrinsics): "
        "[tx, ty, tz, rx, ry, rz] with rx,ry,rz in DEGREES as Euler XYZ.\n"
        "Return only one free_think container per turn."
    )


def init_observation_template(**kwargs):
    """Generate the initial observation prompt template."""
    observation = kwargs.get("observation", "")
    question = kwargs.get("question", "")
    choices = kwargs.get("choices", "")
    named_views = kwargs.get("named_views", None)
    
    lines = []
    if observation:
        lines.append(f"[Images]: {observation}")
    if question:
        lines.append(f"\nQuestion: {question}")
    if choices:
        lines.append(f"\n{choices}")
    if named_views:
        lines.append(f"\nAvailable named views: {', '.join(sorted(named_views))}")
    
    lines.append("\nReturn your answer in the format: <think>...</think><action>answer(X)|</action>")
    
    return "\n".join(lines)


def action_template(**kwargs):
    """Generate the action observation template for subsequent steps."""
    observation = kwargs.get("observation", "")
    status = kwargs.get("status", "")
    reports = kwargs.get("reports", [])
    
    lines = [f"[Observation]: {observation}"]
    if status:
        lines.append(f"Status: {status}")
    if reports:
        lines.append("Reports:")
        lines.extend(reports)
    
    return "\n".join(lines)


def build_reset_prompt(question: str, choices: str = "", view_names: list = None) -> str:
    """Build the reset prompt for tool mode."""
    lines = []
    q = (question or "").strip()
    if q:
        lines.append(q)
    if choices:
        lines.append(choices.strip())
    if view_names:
        lines.extend([
            "",
            "Available named views:",
            ", ".join(sorted(view_names)),
        ])
    lines.extend([
        "",
        "Return in free_think format. You may explore first (e.g., select_view(view_0)),",
        "or answer directly if you are confident:",
        "<think>...</think><action>select_view(view_0)|</action>",
    ])
    return "\n".join(lines).strip()


# Format prompt functions
def format_prompt_free_think(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    
    prompt = f"""Respond in the following format:
<think>Your reasoning process</think>
<action>answer(X){action_sep}</action>

Where X is your answer (A, B, C, or D).
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['free_think']['example']}"
    return prompt


def format_prompt_no_think(**kwargs):
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    
    prompt = f"""Respond with only the action:
<action>answer(X){action_sep}</action>

Where X is your answer (A, B, C, or D).
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['no_think']['example']}"
    return prompt


def format_prompt_grounding(**kwargs):
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    
    prompt = f"""Respond in the following format:
<think><observation>What you see</observation><reasoning>Your reasoning</reasoning></think>
<action>answer(X){action_sep}</action>

Where X is your answer (A, B, C, or D).
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['grounding']['example']}"
    return prompt


def format_prompt_tool(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    
    prompt = f"""Respond in the following format:
<think>Your reasoning process</think>
<action>action1{action_sep}action2{action_sep}...{action_sep}</action>

You can take up to {max_actions} actions per step. End with answer(X) when ready.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['tool_free_think']['example']}"
    return prompt


# Mapping of prompt formats to their corresponding format functions
format_prompt = {
    "free_think": format_prompt_free_think,
    "no_think": format_prompt_no_think,
    "grounding": format_prompt_grounding,
    "tool": format_prompt_tool,
}
