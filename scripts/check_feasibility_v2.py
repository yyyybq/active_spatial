"""
Comprehensive Task Feasibility Analysis - V2
Finer grid (0.15m) + more orientation samples
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOGETHER_NO_BANNER"] = "1"

import json
import numpy as np
import sys
sys.path.insert(0, '/scratch/by2593/project/Active_Spatial/VAGEN')

from vagen.env.active_spatial.spatial_potential_field import SpatialPotentialField, normalize
from shapely.geometry import Point, Polygon
from collections import defaultdict
from pathlib import Path


def load_room_polygons(scene_path):
    with open(Path(scene_path) / "structure.json") as f:
        data = json.load(f)
    return [r['profile'] for r in data.get('rooms', []) if len(r.get('profile', [])) >= 3]


def point_in_room(x, y, room_poly):
    return Polygon(room_poly).contains(Point(x, y))


def find_room_for_point(x, y, room_polys):
    for i, poly in enumerate(room_polys):
        if point_in_room(x, y, poly):
            return i
    return -1


def get_target_centers(target_region):
    params = target_region.get('params', {})
    centers = []
    for key in ['object_center', 'object_a_center', 'object_b_center', 'midpoint_bc']:
        if key in params:
            centers.append(np.array(params[key]))
    if not centers:
        sp = target_region.get('sample_point')
        if sp:
            centers.append(np.array(sp))
    return centers


def compute_max_score(task_data, room_polys, scorer, grid_res=0.15):
    """Find max achievable score in any room, using fine grid + many orientations."""
    task_type = task_data['task_type']
    task_params = task_data.get('task_params', {})
    target_region = task_data['target_region']
    params = target_region.get('params', {})
    
    target_centers = get_target_centers(target_region)
    if target_centers:
        ref = np.mean(target_centers, axis=0)
        target_room = find_room_for_point(ref[0], ref[1], room_polys)
    else:
        target_room = 0
    
    # Search in target room + neighboring rooms
    rooms_to_check = set()
    if target_room >= 0:
        rooms_to_check.add(target_room)
    # Also check all rooms for completeness
    for i in range(len(room_polys)):
        rooms_to_check.add(i)
    
    max_score = 0.0
    best_pos = None
    best_fwd = None
    best_details = {}
    agent_height = target_region.get('height', 1.5)
    
    for room_idx in rooms_to_check:
        poly = room_polys[room_idx]
        coords = np.array(poly)
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        xs = np.arange(x_min, x_max + grid_res, grid_res)
        ys = np.arange(y_min, y_max + grid_res, grid_res)
        
        for x in xs:
            for y in ys:
                if not point_in_room(x, y, poly):
                    continue
                
                pos = np.array([x, y, agent_height])
                
                # Try optimal orientations
                forwards = []
                if target_centers:
                    centroid = np.mean(target_centers, axis=0)
                    d = centroid - pos; d[2] = 0
                    n = np.linalg.norm(d)
                    if n > 1e-6:
                        forwards.append(d / n)
                    for tc in target_centers:
                        d2 = tc - pos; d2[2] = 0
                        n2 = np.linalg.norm(d2)
                        if n2 > 1e-6:
                            forwards.append(d2 / n2)
                
                # + 24 evenly spaced directions
                for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                    forwards.append(np.array([np.cos(angle), np.sin(angle), 0]))
                
                for fwd in forwards:
                    result = scorer.compute_score(pos, fwd, task_type, task_params, target_region)
                    if result.total_score > max_score:
                        max_score = result.total_score
                        best_pos = pos.copy()
                        best_fwd = fwd.copy()
                        best_details = {
                            'pos_score': result.position_score,
                            'ori_score': result.orientation_score,
                            'room': room_idx,
                        }
    
    return max_score, best_pos, best_fwd, best_details


def main():
    scene_id = "0267_840790"
    scene_path = f"/scratch/by2593/project/Active_Spatial/InteriorGS/{scene_id}"
    data_path = f"/scratch/by2593/project/Active_Spatial/VAGEN/data_gen/active_spatial_pipeline/output/train_data_{scene_id}.jsonl"
    
    out = open('/scratch/by2593/project/Active_Spatial/VAGEN/scripts/feasibility_results.txt', 'w')
    def pr(s=""):
        print(s, file=out, flush=True)
    
    room_polys = load_room_polygons(scene_path)
    pr(f"Loaded {len(room_polys)} rooms")
    
    scorer = SpatialPotentialField(
        position_weight=0.7, orientation_weight=0.3,
        fov_horizontal=60.0, fov_vertical=60.0,
    )
    
    tasks = []
    with open(data_path) as f:
        for line in f:
            tasks.append(json.loads(line))
    pr(f"Loaded {len(tasks)} tasks\n")
    
    # ======================================================================
    # 1. Check EVERY task for init-camera-in-room
    # ======================================================================
    pr("=" * 70)
    pr("INIT CAMERA ROOM ANALYSIS")
    pr("=" * 70)
    
    outside_any = 0
    wrong_room = 0
    ok_count = 0
    
    for i, t in enumerate(tasks):
        ext = t['init_camera']['extrinsics']
        ix, iy = ext[0][3], ext[1][3]
        init_room = find_room_for_point(ix, iy, room_polys)
        
        tcs = get_target_centers(t['target_region'])
        if tcs:
            ref = np.mean(tcs, axis=0)
            target_room = find_room_for_point(ref[0], ref[1], room_polys)
        else:
            target_room = -1
        
        if init_room < 0:
            outside_any += 1
            if i < 860:  # print first few
                pr(f"  OUTSIDE: Task {i} ({t['task_type']}) init=({ix:.2f},{iy:.2f}) target_room={target_room}")
        elif init_room != target_room and target_room >= 0:
            wrong_room += 1
        else:
            ok_count += 1
    
    pr(f"\nInit in same room as target: {ok_count}")
    pr(f"Init in WRONG room: {wrong_room}")
    pr(f"Init OUTSIDE all rooms: {outside_any}")
    
    # ======================================================================
    # 2. Feasibility check - sample representative tasks per type+preset
    # ======================================================================
    pr("\n" + "=" * 70)
    pr("TASK FEASIBILITY (grid=0.15m, 24 orientations)")
    pr("=" * 70)
    
    # Group by task_type + preset
    type_preset_map = defaultdict(list)
    for i, t in enumerate(tasks):
        key = (t['task_type'], t.get('preset', ''))
        type_preset_map[key].append(i)
    
    rng = np.random.RandomState(42)
    tasks_to_check = []
    for (tt, preset), indices in sorted(type_preset_map.items()):
        # Check 2 random tasks per type+preset
        sample_n = min(2, len(indices))
        chosen = rng.choice(indices, sample_n, replace=False)
        for idx in chosen:
            tasks_to_check.append(idx)
    
    pr(f"\nChecking {len(tasks_to_check)} representative tasks...\n")
    
    feasibility_by_type = defaultdict(list)
    infeasible = []
    
    for task_idx in sorted(tasks_to_check):
        t = tasks[task_idx]
        tt = t['task_type']
        preset = t.get('preset', '')
        
        max_score, best_pos, best_fwd, details = compute_max_score(t, room_polys, scorer, grid_res=0.15)
        feasibility_by_type[tt].append(max_score)
        
        status = "OK" if max_score >= 0.95 else "FAIL"
        pos_str = f"({best_pos[0]:.2f},{best_pos[1]:.2f})" if best_pos is not None else "None"
        pr(f"  [{status}] Task {task_idx:4d}: {tt:30s}/{preset:15s} max={max_score:.4f} pos_s={details.get('pos_score','?'):.4f} ori_s={details.get('ori_score','?'):.4f} best={pos_str} room={details.get('room','?')}")
        
        if max_score < 0.95:
            infeasible.append({
                'idx': task_idx, 'type': tt, 'preset': preset,
                'max_score': max_score, 'details': details,
                'desc': t.get('task_description', '')[:60],
                'params': {k: v for k, v in t['target_region'].get('params', {}).items() 
                          if k in ['occupancy_ratio', 'fov_vertical', 'radius', 'relation', 'min_distance', 'object_height']},
            })
    
    # Summary
    pr("\n" + "=" * 70)
    pr("SUMMARY BY TASK TYPE")
    pr("=" * 70)
    for tt in sorted(feasibility_by_type.keys()):
        scores = feasibility_by_type[tt]
        n_fail = sum(1 for s in scores if s < 0.95)
        pr(f"  {tt:30s}: min={min(scores):.4f} max={max(scores):.4f} mean={np.mean(scores):.4f} fail={n_fail}/{len(scores)}")
    
    if infeasible:
        pr(f"\nINFEASIBLE TASKS ({len(infeasible)}):")
        pr(f"{'='*70}")
        for t in infeasible:
            pr(f"  Task {t['idx']}: {t['type']}/{t['preset']}")
            pr(f"    max_score={t['max_score']:.4f}, {t['details']}")
            pr(f"    params={t['params']}")
            pr(f"    desc={t['desc']}")
    
    # ======================================================================
    # 3. screen_occupancy detailed analysis
    # ======================================================================
    pr("\n" + "=" * 70)
    pr("SCREEN OCCUPANCY DETAILED ANALYSIS")
    pr("=" * 70)
    
    # Check if 70% occupancy tasks are systematically infeasible
    # The issue: at occupancy_ratio=0.7, fov=90°, required_angular_size=63°
    # So object needs to subtend 63° vertically
    # For obj_height h, distance d: angular_size = 2*arctan(h/(2d))
    # At min_distance, angular_size = 2*arctan(h/(2*min_dist))
    # If this is less than 63°, the task is infeasible even at closest allowed distance
    
    pr("\nChecking if 70% occupancy is achievable at min_distance:")
    seen = set()
    for t in tasks:
        if t['task_type'] != 'screen_occupancy':
            continue
        params = t['target_region'].get('params', {})
        ratio = params.get('occupancy_ratio', 0)
        obj_h = params.get('object_height', 1.0)
        min_d = params.get('min_distance', 0.5)
        stored_r = params.get('radius', 0) 
        fov_v = params.get('fov_vertical', 90.0)
        
        key = (ratio, round(obj_h, 3))
        if key in seen:
            continue
        seen.add(key)
        
        fov_rad = np.radians(fov_v)
        target_ang = ratio * fov_rad
        
        # Max achievable occupancy at min_distance
        max_ang_at_min = 2 * np.arctan(obj_h / (2 * min_d))
        max_occ_at_min = max_ang_at_min / fov_rad
        
        # Is stored radius clamped?
        theoretical_r = (obj_h / 2) / np.tan(target_ang / 2) if target_ang > 0.01 else 99
        clamped = stored_r > theoretical_r * 0.99  # radius >= theoretical means not clamped
        
        status = "OK" if max_occ_at_min >= ratio * 0.9 else "INFEASIBLE"
        pr(f"  ratio={ratio:4.0%} obj_h={obj_h:.3f} min_d={min_d:.3f} stored_r={stored_r:.3f} theoretical_r={theoretical_r:.3f} max_occ@min={max_occ_at_min:.3f} [{status}]")
    
    out.close()
    print("Results written to scripts/feasibility_results.txt")


if __name__ == "__main__":
    main()
