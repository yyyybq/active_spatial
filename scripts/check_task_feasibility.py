"""
Task Feasibility Checker

For each task in the training data, this script:
1. Samples positions within the room
2. Computes the score at each position (with optimal orientation)
3. Reports whether score=1.0 is achievable
4. Flags tasks where max achievable score < threshold

Also checks:
- Whether initial camera is inside the target room
- FoV mismatch between data generation (90°) and environment (60°)
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from shapely.geometry import Point, Polygon

# Add project root to path
project_root = Path("/scratch/by2593/project/Active_Spatial/VAGEN")
sys.path.insert(0, str(project_root))

from vagen.env.active_spatial.spatial_potential_field import SpatialPotentialField, normalize


def load_room_polygons(scene_path: str) -> list:
    """Load room polygons from structure.json."""
    struct_path = Path(scene_path) / "structure.json"
    if not struct_path.exists():
        return []
    with open(struct_path) as f:
        data = json.load(f)
    rooms = data.get("rooms", [])
    polygons = []
    for room in rooms:
        verts = room.get("profile", [])
        if len(verts) >= 3:
            polygons.append(verts)
    return polygons


def point_in_room(x, y, room_poly):
    """Check if (x, y) is inside a room polygon."""
    poly = Polygon(room_poly)
    return poly.contains(Point(x, y))


def find_room_for_point(x, y, room_polys):
    """Find which room index contains the point, or -1."""
    for i, poly in enumerate(room_polys):
        if point_in_room(x, y, poly):
            return i
    return -1


def get_room_bounds(room_poly):
    """Get bounding box of a room polygon.""" 
    coords = np.array(room_poly)
    return coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()


def compute_max_score_in_room(task_data, room_polys, scorer, grid_resolution=0.2):
    """
    Sample positions in the room and compute the maximum achievable score.
    
    For each position, we try the ideal orientation (facing toward relevant objects).
    
    Returns:
        max_score, best_position, best_forward, scores_grid
    """
    task_type = task_data['task_type']
    task_params = task_data.get('task_params', {})
    target_region = task_data['target_region']
    params = target_region.get('params', {})
    
    # Determine target objects for optimal forward direction
    target_centers = []
    if 'object_center' in params:
        target_centers.append(np.array(params['object_center']))
    if 'object_a_center' in params:
        target_centers.append(np.array(params['object_a_center']))
    if 'object_b_center' in params:
        target_centers.append(np.array(params['object_b_center']))
    if 'midpoint_bc' in params:
        target_centers.append(np.array(params['midpoint_bc']))
    
    if not target_centers:
        sample_pt = target_region.get('sample_point')
        if sample_pt:
            target_centers.append(np.array(sample_pt))
    
    # Find which room the target objects are in
    if target_centers:
        ref_center = np.mean(target_centers, axis=0)
        target_room_idx = find_room_for_point(ref_center[0], ref_center[1], room_polys)
    else:
        target_room_idx = 0
    
    if target_room_idx < 0:
        # Objects not in any room - check all rooms
        target_room_idx = 0
    
    # Get room bounds for grid sampling
    if target_room_idx < len(room_polys):
        room_poly = room_polys[target_room_idx]
    else:
        room_poly = room_polys[0] if room_polys else None
    
    if room_poly is None:
        return 0.0, None, None, None
    
    x_min, x_max, y_min, y_max = get_room_bounds(room_poly)
    
    # Sample grid
    xs = np.arange(x_min, x_max + grid_resolution, grid_resolution)
    ys = np.arange(y_min, y_max + grid_resolution, grid_resolution)
    
    max_score = 0.0
    best_pos = None
    best_fwd = None
    agent_height = target_region.get('height', 1.5)
    
    for x in xs:
        for y in ys:
            if not point_in_room(x, y, room_poly):
                continue
            
            pos = np.array([x, y, agent_height])
            
            # Try multiple forward directions (toward each target + centroid)
            forwards_to_try = []
            
            # Toward centroid of targets
            if target_centers:
                centroid = np.mean(target_centers, axis=0)
                to_centroid = centroid - pos
                to_centroid[2] = 0
                norm = np.linalg.norm(to_centroid)
                if norm > 1e-6:
                    forwards_to_try.append(to_centroid / norm)
            
            # Toward each individual target
            for tc in target_centers:
                to_tc = tc - pos
                to_tc[2] = 0
                norm = np.linalg.norm(to_tc)
                if norm > 1e-6:
                    forwards_to_try.append(to_tc / norm)
            
            # Also try 12 evenly spaced directions
            for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                forwards_to_try.append(np.array([np.cos(angle), np.sin(angle), 0]))
            
            for fwd in forwards_to_try:
                result = scorer.compute_score(pos, fwd, task_type, task_params, target_region)
                if result.total_score > max_score:
                    max_score = result.total_score
                    best_pos = pos.copy()
                    best_fwd = fwd.copy()
    
    return max_score, best_pos, best_fwd, None


def check_init_in_room(task_data, room_polys):
    """Check if the initial camera position is inside any room."""
    ext = task_data['init_camera']['extrinsics']
    init_x = ext[0][3]
    init_y = ext[1][3]
    init_z = ext[2][3]
    
    room_idx = find_room_for_point(init_x, init_y, room_polys)
    
    # Also find which room target objects are in
    params = task_data['target_region'].get('params', {})
    target_centers = []
    if 'object_center' in params:
        target_centers.append(np.array(params['object_center']))
    if 'object_a_center' in params:
        target_centers.append(np.array(params['object_a_center']))
    if 'object_b_center' in params:
        target_centers.append(np.array(params['object_b_center']))
    
    target_room_idx = -1
    if target_centers:
        ref = np.mean(target_centers, axis=0)
        target_room_idx = find_room_for_point(ref[0], ref[1], room_polys)
    
    return {
        'init_pos': [init_x, init_y, init_z],
        'init_in_any_room': room_idx >= 0,
        'init_room_idx': room_idx,
        'target_room_idx': target_room_idx,
        'init_in_target_room': room_idx == target_room_idx and room_idx >= 0,
    }


def main():
    scene_id = "0267_840790"
    scene_path = f"/scratch/by2593/project/Active_Spatial/InteriorGS/{scene_id}"
    data_path = f"/scratch/by2593/project/Active_Spatial/VAGEN/data_gen/active_spatial_pipeline/output/train_data_{scene_id}.jsonl"
    
    print("=" * 80)
    print("TASK FEASIBILITY ANALYSIS")
    print("=" * 80)
    
    # Load room polygons
    room_polys = load_room_polygons(scene_path)
    print(f"\nLoaded {len(room_polys)} room polygons")
    for i, poly in enumerate(room_polys):
        coords = np.array(poly)
        print(f"  Room {i}: x=[{coords[:,0].min():.1f}, {coords[:,0].max():.1f}], y=[{coords[:,1].min():.1f}, {coords[:,1].max():.1f}]")
    
    # Create scorer with ENV settings (60° FoV)
    scorer_env = SpatialPotentialField(
        position_weight=0.7,
        orientation_weight=0.3,
        fov_horizontal=60.0,
        fov_vertical=60.0,
    )
    
    # Load all tasks
    tasks = []
    with open(data_path) as f:
        for line in f:
            tasks.append(json.loads(line))
    
    print(f"\nLoaded {len(tasks)} tasks")
    
    # ========================================================================
    # Issue 1: FoV mismatch analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("ISSUE 1: FoV MISMATCH (data_gen=90°, env=60°)")
    print("=" * 80)
    
    fov_mismatch_count = 0
    for t in tasks:
        params = t['target_region'].get('params', {})
        stored_fov = params.get('fov_vertical')
        if stored_fov and stored_fov != 60.0:
            fov_mismatch_count += 1
    
    print(f"\nTasks with fov_vertical != 60°: {fov_mismatch_count} / {len(tasks)}")
    
    # For screen_occupancy: show what the clamped radius means
    print("\nScreen occupancy: required distance at 60° vs 90° FoV")
    print(f"{'ratio':>6} {'obj_h':>6} {'r(90)':>7} {'r(60)':>7} {'clamped_r':>10} {'min_d':>6}")
    seen = set()
    for t in tasks:
        if t['task_type'] != 'screen_occupancy':
            continue
        params = t['target_region'].get('params', {})
        ratio = params.get('occupancy_ratio')
        obj_h = params.get('object_height', 1.0)
        min_d = params.get('min_distance', 0.5)
        key = (ratio, round(obj_h, 3))
        if key in seen:
            continue
        seen.add(key)
        
        # Required distance with 90° FoV (as generated)
        fov_90 = np.radians(90.0)
        ang_90 = ratio * fov_90
        r_90 = (obj_h / 2) / np.tan(ang_90 / 2) if ang_90 > 0.01 else 99
        r_90 = np.clip(r_90, min_d, 20.0)
        
        # Required distance with 60° FoV (actual env)
        fov_60 = np.radians(60.0)
        ang_60 = ratio * fov_60
        r_60 = (obj_h / 2) / np.tan(ang_60 / 2) if ang_60 > 0.01 else 99
        r_60 = np.clip(r_60, min_d, 20.0)
        
        stored_r = params.get('radius', 0)
        
        print(f"{ratio:>6.1%} {obj_h:>6.3f} {r_90:>7.2f} {r_60:>7.2f} {stored_r:>10.2f} {min_d:>6.3f}")
    
    # ========================================================================
    # Issue 2: Init camera outside room
    # ========================================================================
    print("\n" + "=" * 80)
    print("ISSUE 2: INITIAL CAMERA OUTSIDE ROOM")
    print("=" * 80)
    
    outside_count = 0
    wrong_room_count = 0
    room_stats = defaultdict(int)
    outside_tasks = []
    
    for i, t in enumerate(tasks):
        info = check_init_in_room(t, room_polys)
        if not info['init_in_any_room']:
            outside_count += 1
            outside_tasks.append((i, t['task_type'], info['init_pos']))
        elif not info['init_in_target_room']:
            wrong_room_count += 1
        room_stats[info['init_room_idx']] += 1
    
    print(f"\nInit camera NOT in any room: {outside_count} / {len(tasks)}")
    print(f"Init camera in WRONG room (not target room): {wrong_room_count} / {len(tasks)}")
    print(f"\nRoom distribution:")
    for room_idx, count in sorted(room_stats.items()):
        label = f"Room {room_idx}" if room_idx >= 0 else "OUTSIDE"
        print(f"  {label}: {count}")
    
    if outside_tasks:
        print(f"\nFirst 10 tasks with init camera outside room:")
        for idx, tt, pos in outside_tasks[:10]:
            print(f"  Task {idx}: {tt}, init_pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    # ========================================================================
    # Issue 3: Task feasibility (max score < 0.95)
    # ========================================================================
    print("\n" + "=" * 80)
    print("ISSUE 3: TASK FEASIBILITY (checking max achievable score)")
    print("=" * 80)
    print("Sampling positions in room grid (resolution=0.3m)...")
    print("Using ENV FoV settings (60° x 60°)")
    
    # Sample a subset of tasks (one per unique task_type + preset) for speed
    # But also do a broad check
    type_preset_seen = set()
    tasks_to_check = []
    
    for i, t in enumerate(tasks):
        key = (t['task_type'], t.get('preset', ''))
        if key not in type_preset_seen:
            type_preset_seen.add(key)
            tasks_to_check.append((i, t))
    
    # Also randomly sample some additional tasks per type
    rng = np.random.RandomState(42)
    type_counts = defaultdict(list)
    for i, t in enumerate(tasks):
        type_counts[t['task_type']].append(i)
    
    for tt, indices in type_counts.items():
        if len(indices) > 3:
            sample_idx = rng.choice(indices, min(3, len(indices)), replace=False)
            for idx in sample_idx:
                key = (tasks[idx]['task_type'], tasks[idx].get('preset', ''))
                if key not in type_preset_seen:
                    tasks_to_check.append((idx, tasks[idx]))
                    type_preset_seen.add(key)
    
    print(f"\nChecking {len(tasks_to_check)} representative tasks...")
    
    infeasible_tasks = []
    feasibility_by_type = defaultdict(list)
    
    for task_idx, task_data in tasks_to_check:
        max_score, best_pos, best_fwd, _ = compute_max_score_in_room(
            task_data, room_polys, scorer_env, grid_resolution=0.3
        )
        
        tt = task_data['task_type']
        preset = task_data.get('preset', '')
        feasibility_by_type[tt].append(max_score)
        
        if max_score < 0.95:
            infeasible_tasks.append({
                'idx': task_idx,
                'task_type': tt,
                'preset': preset,
                'max_score': max_score,
                'best_pos': best_pos.tolist() if best_pos is not None else None,
                'description': task_data.get('task_description', '')[:80],
            })
        
        status = "OK" if max_score >= 0.95 else "INFEASIBLE"
        print(f"  [{status}] Task {task_idx}: {tt}/{preset} -> max_score={max_score:.4f}")
    
    print(f"\n{'='*60}")
    print("SUMMARY: Max achievable scores by task type")
    print(f"{'='*60}")
    for tt in sorted(feasibility_by_type.keys()):
        scores = feasibility_by_type[tt]
        print(f"  {tt:30s}: min={min(scores):.4f}, max={max(scores):.4f}, mean={np.mean(scores):.4f}")
    
    if infeasible_tasks:
        print(f"\n{'='*60}")
        print(f"INFEASIBLE TASKS (max_score < 0.95): {len(infeasible_tasks)}")
        print(f"{'='*60}")
        for t in infeasible_tasks:
            print(f"  Task {t['idx']}: {t['task_type']}/{t['preset']}")
            print(f"    max_score={t['max_score']:.4f}")
            print(f"    best_pos={t['best_pos']}")
            print(f"    {t['description']}")
    
    # ========================================================================
    # Summary & Recommendations
    # ========================================================================
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
1. FoV MISMATCH:
   - Data generation uses fov_vertical=90° but env uses fov_vertical=60°
   - screen_occupancy tasks store 90° in params; scorer reads it for occupancy calc
   - The SCORING is internally consistent (uses stored 90°), but the ACTUAL RENDERING
     uses 60° FoV - so the visual occupancy won't match the score
   - FIX: Regenerate data with fov_vertical=60° OR update env to use 90°

2. INIT CAMERA OUTSIDE ROOM:
   - Some initial cameras are placed outside the target room
   - The camera_sampler has require_in_target_room validation, but it may not
     always be enabled
   - FIX: Post-filter tasks where init camera is not in target room

3. TASK FEASIBILITY:
   - Some tasks may be geometrically infeasible within room constraints
   - projective_relations: half-plane may not intersect room
   - screen_occupancy: required distance may exceed room dimensions
   - FIX: Add post-generation validation that samples room positions and
     verifies max_score >= 0.95
""")


if __name__ == "__main__":
    main()
