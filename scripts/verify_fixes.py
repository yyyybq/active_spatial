"""
Verify that the pipeline fixes correctly filter infeasible tasks.
Tests:
1. screen_occupancy feasibility check in task_generator
2. target reachability check in pipeline
3. Camera room constraint (can't test directly, but verify logic)
"""
import sys, json
import numpy as np
sys.path.insert(0, '/scratch/by2593/project/Active_Spatial/VAGEN')

from data_gen.active_spatial_pipeline.task_generator import TaskGenerator, RegionType, TargetRegion, TaskResult, BoundingBox3D
from data_gen.active_spatial_pipeline.pipeline import validate_target_reachability
from data_gen.active_spatial_pipeline.config import PipelineConfig

# ============================================================
# Test 1: screen_occupancy feasibility check
# ============================================================
print("=" * 60)
print("TEST 1: screen_occupancy feasibility validation")
print("=" * 60)

config = PipelineConfig()
tg = TaskGenerator(config)

# Create a small object (height=0.6m) with a large min_distance (0.6m due to object size)
# For 70% occupancy at fov=90°, theoretical distance = 0.3/tan(0.315) = 0.92m
# But min_dist = 0.6m ... still ok
# For 70% occupancy of a very small object (height 0.2m):
# theoretical_distance = 0.1/tan(0.315) = 0.31m, min_dist = 0.5m (absolute min)
# max_occ_at_min_dist = 2*arctan(0.2/(2*0.5))/pi*2 = 2*arctan(0.2)/1.5708
small_obj_vertices = np.array([
    [-0.1 + d1*0.1, -0.1 + d2*0.1, 0.0 + d3*0.1]
    for d1 in [-1, 1] for d2 in [-1, 1] for d3 in [-1, 1]
])
small_bbox = BoundingBox3D(vertices=small_obj_vertices, label="tiny_object", ins_id="test1")
print(f"  Object height: {small_bbox.height:.3f}m")

for occ_ratio in [0.2, 0.3, 0.5, 0.7]:
    result = tg.generate_screen_occupancy(small_bbox, occ_ratio, fov_vertical=90.0, agent_height=1.5)
    print(f"  occupancy={occ_ratio}: is_valid={result.is_valid}")

# Larger object (height=1.5m)
large_obj_vertices = np.array([
    [0.0 + d1*0.5, 0.0 + d2*0.5, 0.0 + d3*0.75]
    for d1 in [-1, 1] for d2 in [-1, 1] for d3 in [-1, 1]
])
large_bbox = BoundingBox3D(vertices=large_obj_vertices, label="large_object", ins_id="test2")
print(f"  Object height: {large_bbox.height:.3f}m")

for occ_ratio in [0.2, 0.3, 0.5, 0.7]:
    result = tg.generate_screen_occupancy(large_bbox, occ_ratio, fov_vertical=90.0, agent_height=1.5)
    print(f"  occupancy={occ_ratio}: is_valid={result.is_valid}")

# ============================================================
# Test 2: target reachability validation
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: target reachability validation")
print("=" * 60)

# Simple room polygon: square room from (0,0) to (4,4)
room_polys = [[[0, 0], [4, 0], [4, 4], [0, 4]]]

# Test 2a: Apollonius circle centered at (2,2) with radius 1.0 — inside room
curve_points_inside = []
for i in range(20):
    theta = 2 * np.pi * i / 20
    pt = [2.0 + 1.0 * np.cos(theta), 2.0 + 1.0 * np.sin(theta), 1.5]
    curve_points_inside.append(pt)

fake_task_inside = TaskResult(
    task_type='size_distance_invariance',
    task_params={},
    target_region=TargetRegion(
        region_type=RegionType.CURVE,
        params={"points": curve_points_inside, "center": [2.0, 2.0], "radius": 1.0},
        sample_point=np.array([3.0, 2.0, 1.5]),
        sample_forward=np.array([0, 1, 0]),
        height=1.5
    ),
    preset='equal_size',
    is_valid=True,
    description="test inside",
    object_label="test"
)
print(f"  Circle at (2,2) r=1.0 in room [0,4]x[0,4]: reachable={validate_target_reachability(fake_task_inside, room_polys)}")

# Test 2b: Apollonius circle centered at (10,10) with radius 0.5 — outside room
curve_points_outside = []
for i in range(20):
    theta = 2 * np.pi * i / 20
    pt = [10.0 + 0.5 * np.cos(theta), 10.0 + 0.5 * np.sin(theta), 1.5]
    curve_points_outside.append(pt)

fake_task_outside = TaskResult(
    task_type='size_distance_invariance',
    task_params={},
    target_region=TargetRegion(
        region_type=RegionType.CURVE,
        params={"points": curve_points_outside, "center": [10.0, 10.0], "radius": 0.5},
        sample_point=np.array([10.5, 10.0, 1.5]),
        sample_forward=np.array([0, 1, 0]),
        height=1.5
    ),
    preset='equal_size',
    is_valid=True,
    description="test outside",
    object_label="test"
)
print(f"  Circle at (10,10) r=0.5 outside room: reachable={validate_target_reachability(fake_task_outside, room_polys)}")

# Test 2c: Apollonius circle that barely touches room edge
curve_points_edge = []
for i in range(20):
    theta = 2 * np.pi * i / 20
    pt = [4.5 + 1.0 * np.cos(theta), 2.0 + 1.0 * np.sin(theta), 1.5]
    curve_points_edge.append(pt)

fake_task_edge = TaskResult(
    task_type='size_distance_invariance',
    task_params={},
    target_region=TargetRegion(
        region_type=RegionType.CURVE,
        params={"points": curve_points_edge, "center": [4.5, 2.0], "radius": 1.0},
        sample_point=np.array([3.5, 2.0, 1.5]),
        sample_forward=np.array([0, 1, 0]),
        height=1.5
    ),
    preset='equal_size',
    is_valid=True,
    description="test edge",
    object_label="test"
)
print(f"  Circle at (4.5,2) r=1.0 partially in room: reachable={validate_target_reachability(fake_task_edge, room_polys)}")

# ============================================================
# Test 3: Check existing dataset for infeasible tasks
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Check existing dataset for infeasible tasks")
print("=" * 60)

data_path = '/scratch/by2593/project/Active_Spatial/VAGEN/data_gen/active_spatial_pipeline/output/train_data_0267_840790.jsonl'
scene_path = '/scratch/by2593/project/Active_Spatial/InteriorGS/0267_840790'

# Load room polygons
import json as _json
structure_path = f"{scene_path}/structure.json"
with open(structure_path) as f:
    structure = _json.load(f)
room_polys_real = []
for room in structure.get('rooms', []):
    profile = room.get('profile', [])
    if profile:
        poly = [[p[0], p[1]] for p in profile]
        room_polys_real.append(poly)
print(f"  Loaded {len(room_polys_real)} rooms")

# Load tasks and check
with open(data_path) as f:
    tasks_data = [json.loads(line) for line in f]

infeasible_count = 0
infeasible_by_type = {}
for i, task_data in enumerate(tasks_data):
    task_type = task_data.get('task_type', '')
    target_region = task_data.get('target_region', {})
    region_type_str = target_region.get('type', '')
    params = target_region.get('params', {})
    
    # Use circle-based check for circle tasks too
    center = params.get('object_center', params.get('center', None))
    radius = params.get('radius', params.get('sample_distance', None))

    def point_in_poly(x, y, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for ii in range(n):
            xi, yi = poly[ii][0], poly[ii][1]
            xj, yj = poly[j][0], poly[j][1]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = ii
        return inside
    
    if region_type_str == 'curve':
        curve_points = params.get('points', [])
        reachable = 0
        for pt in curve_points:
            for poly in room_polys_real:
                if point_in_poly(pt[0], pt[1], poly):
                    reachable += 1
                    break
        if reachable < max(1, len(curve_points) * 0.1):
            infeasible_count += 1
            infeasible_by_type[task_type] = infeasible_by_type.get(task_type, 0) + 1
            print(f"  [INFEAS] Task {i}: {task_type} - {reachable}/{len(curve_points)} points reachable, radius={params.get('radius', 'N/A'):.2f}")
    
    elif center is not None and radius is not None:
        center_arr = np.array(center)
        center_2d = center_arr[:2] if len(center_arr) > 2 else center_arr
        n_samples = 36
        reachable = 0
        for k in range(n_samples):
            theta = 2 * np.pi * k / n_samples
            x = center_2d[0] + radius * np.cos(theta)
            y = center_2d[1] + radius * np.sin(theta)
            for poly in room_polys_real:
                if point_in_poly(x, y, poly):
                    reachable += 1
                    break
        if reachable < max(1, n_samples * 0.1):
            infeasible_count += 1
            infeasible_by_type[task_type] = infeasible_by_type.get(task_type, 0) + 1
            print(f"  [INFEAS] Task {i}: {task_type} - {reachable}/{n_samples} arc points reachable")

print(f"\n  Total infeasible (unreachable target region): {infeasible_count}")
if infeasible_by_type:
    for tt, cnt in sorted(infeasible_by_type.items()):
        print(f"    {tt}: {cnt}")

# Also check init cameras
init_outside_count = 0
init_wrong_room_count = 0
for i, task_data in enumerate(tasks_data):
    init_cam = task_data.get('init_camera', {})
    ext = init_cam.get('extrinsics', None)
    if ext is None:
        continue
    ext_arr = np.array(ext)
    if ext_arr.shape == (4, 4):
        cam_pos = ext_arr[:3, 3]  # c2w, position is last column
    else:
        continue
    
    x, y = cam_pos[0], cam_pos[1]
    in_any = False
    for poly in room_polys_real:
        if point_in_poly(x, y, poly):
            in_any = True
            break
    
    if not in_any:
        init_outside_count += 1

print(f"\n  Init cameras outside all rooms: {init_outside_count}/{len(tasks_data)}")
print("\nDone!")
