# Active Spatial Intelligence Environment Prompts
# This file defines system prompts and format configurations for the active spatial navigation task.

FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><action>...</action>",
        "example": """<think>I can see I'm looking at a room with furniture. The target is to reach the front view of the chair. I should move forward and adjust my angle.</think><action>move_forward|move_forward|turn_right|</action>"""
    },
    "no_think": {
        "description": "You should provide only your action.",
        "format": "<action>...</action>",
        "example": """<action>move_forward|move_forward|turn_right|</action>"""
    },
    "grounding": {
        "description": "You should first describe what you observe, then reason about the actions needed, and finally provide your action.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><action>...</action>",
        "example": """<think><observation>I'm in a living room. There's a sofa in front of me, a coffee table to my left, and the target chair is visible in the distance to my right. I appear to be viewing from an oblique angle.</observation><reasoning>To reach the front view of the chair, I need to move right and forward, then adjust my orientation to face the chair directly.</reasoning></think><action>move_forward|turn_right|move_forward|</action>"""
    },
    "worldmodeling": {
        "description": "You should first reason about your actions and predict the expected outcome, then provide your action.",
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>",
        "example": """<think><reasoning>The target is to reach the front view of the chair. Currently I can see the chair from the side. I need to move around it to face it from the front.</reasoning><prediction>After moving forward and turning left, I should be positioned in front of the chair with a clear frontal view.</prediction></think><action>move_forward|move_forward|turn_left|</action>"""
    },
    "grounding_worldmodeling": {
        "description": "You should describe your observation, reason about actions, predict the outcome, then provide your action.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><action>...</action>",
        "example": """<think><observation>I'm in a bedroom looking at a desk with a lamp. The target chair is behind me based on the task description.</observation><reasoning>I need to turn around to locate the chair, then navigate to reach its front view.</reasoning><prediction>After turning around, I expect to see the chair and can then approach it directly.</prediction></think><action>turn_right|turn_right|move_forward|</action>"""
    }
}

# Action descriptions for the agent
ACTION_DESCRIPTIONS = """
Available actions (use | as separator, end with |):
- move_forward: Move the camera forward by a fixed step
- move_backward: Move the camera backward by a fixed step  
- turn_left: Rotate the camera left (yaw) by a fixed angle
- turn_right: Rotate the camera right (yaw) by a fixed angle
- look_up: Tilt the camera upward (pitch) by a fixed angle
- look_down: Tilt the camera downward (pitch) by a fixed angle
- done: Signal that you have reached the target pose (terminates episode)
"""

def system_prompt(**kwargs):
    """Generate the system prompt for the Active Spatial environment."""
    selected_format = kwargs.get("format", "free_think")
    step_translation = kwargs.get("step_translation", 0.1)
    step_rotation_deg = kwargs.get("step_rotation_deg", 5.0)
    
    format_config = FORMAT_CONFIGS.get(selected_format, FORMAT_CONFIGS["free_think"])
    
    base_prompt = f"""You are a spatial navigation agent in a 3D indoor environment. Your task is to navigate a camera to reach a specific target view of an object.

{ACTION_DESCRIPTIONS}

Step sizes: translation = {step_translation:.2f} meters, rotation = {step_rotation_deg:.1f} degrees.

{format_config['description']}

Response format: {format_config['format']}

Example:
{format_config['example']}

Rewards:
- Format correct: +0.2
- Progress toward target pose: continuous reward based on distance and orientation improvement
- Reaching target pose: +1.0

Hints:
1. Pay attention to the target object and the requested view (front, back, left, right, etc.)
2. Use multiple actions per step to make efficient progress
3. Consider both position and orientation when navigating
4. Look around if you're unsure of the target location
"""
    return base_prompt


def init_observation_template(**kwargs):
    """Generate the initial observation prompt template."""
    observation = kwargs.get("observation", "")
    task_prompt = kwargs.get("task_prompt", "Navigate to the target view.")
    spatial_prior = kwargs.get("spatial_prior", "")  # Multi-frame spatial prior text
    
    template = ""
    
    # Add spatial prior section if provided
    if spatial_prior:
        template += f"""[Spatial Context]:
{spatial_prior}

"""
    
    template += f"""[Initial Observation]:
{observation}
Task: {task_prompt}

Navigate to reach the specified view of the target object. Use the available actions to position and orient the camera correctly.
"""
    return template


def action_template(**kwargs):
    """Generate the action observation template for subsequent steps."""
    observation = kwargs.get("observation", "")
    env_feedback = kwargs.get("env_feedback", "")
    
    template = f"""[Observation]:
{observation}
"""
    if env_feedback:
        template += f"Environment Feedback: {env_feedback}\n"
    
    return template


# Format prompt functions for different prompt formats
def format_prompt_free_think(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    
    prompt = f"""Respond in the following format:
<think>Your reasoning process</think>
<action>action1{action_sep}action2{action_sep}...{action_sep}</action>

You can take up to {max_actions} actions per step.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['free_think']['example']}"
    return prompt


def format_prompt_no_think(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    
    prompt = f"""Respond with only the action:
<action>action1{action_sep}action2{action_sep}...{action_sep}</action>

You can take up to {max_actions} actions per step.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['no_think']['example']}"
    return prompt


def format_prompt_grounding(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    
    prompt = f"""Respond in the following format:
<think><observation>What you see</observation><reasoning>Your reasoning</reasoning></think>
<action>action1{action_sep}action2{action_sep}...{action_sep}</action>

You can take up to {max_actions} actions per step.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['grounding']['example']}"
    return prompt


def format_prompt_worldmodeling(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    
    prompt = f"""Respond in the following format:
<think><reasoning>Your reasoning</reasoning><prediction>Expected outcome</prediction></think>
<action>action1{action_sep}action2{action_sep}...{action_sep}</action>

You can take up to {max_actions} actions per step.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['worldmodeling']['example']}"
    return prompt


def format_prompt_grounding_worldmodeling(**kwargs):
    max_actions = kwargs.get("max_actions_per_step", 5)
    action_sep = kwargs.get("action_sep", "|")
    add_example = kwargs.get("add_example", True)
    
    prompt = f"""Respond in the following format:
<think><observation>What you see</observation><reasoning>Your reasoning</reasoning><prediction>Expected outcome</prediction></think>
<action>action1{action_sep}action2{action_sep}...{action_sep}</action>

You can take up to {max_actions} actions per step.
"""
    if add_example:
        prompt += f"\nExample: {FORMAT_CONFIGS['grounding_worldmodeling']['example']}"
    return prompt


# Mapping of prompt formats to their corresponding format functions
format_prompt = {
    "free_think": format_prompt_free_think,
    "no_think": format_prompt_no_think,
    "grounding": format_prompt_grounding,
    "worldmodeling": format_prompt_worldmodeling,
    "grounding_worldmodeling": format_prompt_grounding_worldmodeling,
}
