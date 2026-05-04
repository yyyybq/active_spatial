# Experiment + W&B Analysis Report

## Inputs
- experiment_log: /scratch/by2593/project/Active_Spatial/VAGEN/cambrian_v4_7gpu.log
- wandb_root: /scratch/by2593/project/Active_Spatial/VAGEN/wandb
- matched_run_dir: /scratch/by2593/project/Active_Spatial/VAGEN/wandb/run-20260424_031750-lky5g4xb
- matched_run_id: lky5g4xb
- run_match_confidence: high
- metrics_source: /scratch/by2593/project/Active_Spatial/VAGEN/wandb/run-20260424_031750-lky5g4xb/files/output.log

## Metric Coverage
- parsed_steps: 1 -> 49 (count=49)

## RL Metric Trends (Early/Mid/Late)
- train/score: early=-0.3872, mid=-0.4022, late=-0.4001, delta=-0.0129, trend=down
- train/success: early=0.0104, mid=0.0187, late=0.0156, delta=+0.0053, trend=up
- critic/vf_loss: early=1.3817, mid=0.1628, late=0.0736, delta=-1.3080, trend=down
- actor/pg_loss: N/A
- actor/entropy_loss: N/A
- kl (ppo/actor): N/A
- train/total_collisions: early=2.0894, mid=2.3271, late=2.2060, delta=+0.1166, trend=up
- response_length/mean: early=635.1313, mid=637.8312, late=648.0432, delta=+12.9119, trend=up
- reward-related: early=-0.3234, mid=-0.3274, late=-0.3280, delta=-0.0046, trend=down

## Garbled Output Check
- replacement_char_count: 0
- mojibake_marker_count(Ã/Â): 0
- escaped_hex_count(\xNN): 0
- suspicious_preview_count: 0
- left_truncation_warning_count: 714

- conclusion: no obvious text-encoding garble found in log samples.

## response_preview Samples
- head[1]: Based on the previous observations and the environment feedback stating that the action had no effect, it appears that the camera may need to be adjusted further to achieve the desired view of the chair and its surroundi
- head[2]: Based on the feedback that the action had no effect, it appears that the camera may need to be repositioned to properly align with the target view. The current camera pose indicates that the room is still in view, but th
- head[3]: Based on the feedback that the action had no effect, it appears that the camera may need to be repositioned to properly align with the target view. The current camera pose indicates that the room is still in view, but th
- tail[1]: <think>Based on the current camera pose and the environment feedback, it appears that the camera has been successfully moved to a new position. The new pose is closer to the sofa and table, and the angle is slightly diff
- tail[2]: <think>Based on the initial camera pose, I need to move the camera to the right of the plant with flowers, which is specified to be about 4.72 meters away. To achieve this, I should move forward to increase the distance 
- tail[3]: <think>Based on the camera pose, it seems I am now slightly closer to the target view of the TV and sofa while being equidistant from both. The camera is now oriented more towards the center of the room, as indicated by 
