#!/usr/bin/env python3
"""Check a meta.json camera pose for legality and visible objects.

Usage:
  python tools/check_meta_view.py /path/to/meta.json

The script will:
- load the meta.json (supports meta_positions.camera_pos / camera_pos and poses.view)
- locate the scene folder (tries absolute path, then ./data/<scene_name>)
- run occupancy / AABB checks via ensure_position_legal
- compute screen-space visible objects using the same method as the batch generator

Prints a small JSON summary to stdout.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import List
import numpy as np

# reuse helpers from the data_gen modules
from view_suite.envs.active_spatial_intelligence.data_gen.qa_2d_generator import (
    create_intrinsics,
    ensure_position_legal,
    compute_screen_space_visible,
)
from view_suite.envs.active_spatial_intelligence.utils.occlusion import (
    load_scene_aabbs,
    load_scene_wall_aabbs,
    occluded_area_on_image,
    camtoworld_from_pos_target,
    is_box_occluded_by_any,
)


def find_scene_path(scene_str: str) -> Path | None:
    p = Path(scene_str)
    if p.exists() and p.is_dir():
        return p
    # try ./data/<scene_str>
    cand = Path.cwd() / 'data' / scene_str
    if cand.exists() and cand.is_dir():
        return cand
    # try ../data (if running from tools/)
    cand2 = Path.cwd().parent / 'data' / scene_str
    if cand2.exists() and cand2.is_dir():
        return cand2
    # search shallowly under ./data
    root = Path.cwd() / 'data'
    if root.exists():
        for d in root.iterdir():
            if d.is_dir() and d.name == scene_str:
                return d
    return None


def extract_pose_from_meta(meta: dict):
    # Try a few common locations used by generators
    # 1) meta_positions.camera_pos / camera_target
    mp = meta.get('meta_positions') or meta.get('meta') or {}
    if isinstance(mp, dict):
        cam_pos = mp.get('camera_pos') or mp.get('camera_position') or mp.get('current_pos')
        cam_tgt = mp.get('camera_target') or mp.get('camera_target_position') or mp.get('current_target')
        if cam_pos is not None and cam_tgt is not None:
            return np.array(cam_pos, dtype=float), np.array(cam_tgt, dtype=float)
    # 2) poses.view.position / target
    poses = meta.get('poses') or {}
    if isinstance(poses, dict) and 'view' in poses:
        v = poses['view']
        if isinstance(v, dict) and 'position' in v and 'target' in v:
            return np.array(v['position'], dtype=float), np.array(v['target'], dtype=float)
    # 3) top-level fields
    top_pos = meta.get('camera_pos') or meta.get('position')
    top_tgt = meta.get('camera_target') or meta.get('target')
    if top_pos is not None and top_tgt is not None:
        return np.array(top_pos, dtype=float), np.array(top_tgt, dtype=float)
    return None, None


def main():
    if len(sys.argv) < 2:
        print('Usage: python tools/check_meta_view.py /path/to/meta.json')
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.exists():
        print(json.dumps({'error': 'meta.json not found', 'path': str(path)}))
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scene_field = data.get('scene') or (data.get('meta') or {}).get('scene')
    if not scene_field:
        print(json.dumps({'error': 'scene not found in meta.json'}))
        sys.exit(1)

    scene_path = find_scene_path(scene_field)
    if scene_path is None:
        # also accept full path in meta (sometimes scene contains full path)
        if Path(scene_field).exists():
            scene_path = Path(scene_field)
    if scene_path is None:
        print(json.dumps({'error': 'scene folder not found', 'scene': scene_field}))
        sys.exit(1)

    # extract pose
    pos, tgt = extract_pose_from_meta(data)
    if pos is None or tgt is None:
        print(json.dumps({'error': 'camera pose not found in meta.json'}))
        sys.exit(1)

    # load occupancy bounds from qa_2d_generator helpers if available
    try:
        # we call ensure_position_legal which internally prints reasons; reuse it
        min_h, max_h = None, None
        # call helper
        legal = ensure_position_legal(str(scene_path), pos, min_h=min_h, max_h=max_h)
    except Exception as e:
        legal = False

    # build visible list using screen-space algorithm
    aabbs_objs = load_scene_aabbs(str(scene_path))
    aabbs_all = aabbs_objs + load_scene_wall_aabbs(str(scene_path))
    K = np.array(create_intrinsics()['K'], dtype=float)
    width = create_intrinsics()['width']
    height = create_intrinsics()['height']

    try:
        visible = compute_screen_space_visible(aabbs_objs, pos, tgt, K, width, height, corner_threshold=int(getattr(sys, 'argv', [])[2]) if False else 4, overlap_thresh=0.8)
    except Exception:
        # fallback to simple corner counting
        visible = []
        for b in aabbs_objs:
            corners = count_visible_corners_fallback(pos, tgt, K, b.bmin, b.bmax, width, height)
            if corners >= 4:
                visible.append(b)

    # produce a small summary and per-object occlusion degrees
    visible_ids = {getattr(b, 'id', None) for b in visible}
    per_object = []
    c2w = camtoworld_from_pos_target(pos, tgt)
    wall_aabbs = load_scene_wall_aabbs(str(scene_path))
    for b in aabbs_objs:
        try:
            # occlusion w.r.t. all aabbs (objects + walls)
            res_all = occluded_area_on_image(
                ray_o=pos,
                target_bmin=b.bmin,
                target_bmax=b.bmax,
                aabbs=aabbs_all,
                K=K,
                camtoworld=c2w,
                width=width,
                height=height,
                target_id=b.id,
                depth_mode="min",
            )
            occ_all = float(res_all.get('occlusion_ratio_image', 0.0))
        except Exception:
            occ_all = 0.0
        try:
            # occlusion considering only walls (helps answer "is it the wall that blocks?")
            res_walls = occluded_area_on_image(
                ray_o=pos,
                target_bmin=b.bmin,
                target_bmax=b.bmax,
                aabbs=wall_aabbs,
                K=K,
                camtoworld=c2w,
                width=width,
                height=height,
                target_id=b.id,
                depth_mode="min",
            )
            occ_walls = float(res_walls.get('occlusion_ratio_image', 0.0))
        except Exception:
            occ_walls = 0.0
        per_object.append({
            'id': getattr(b, 'id', None),
            'label': getattr(b, 'label', None),
            'occlusion_ratio': occ_all,
            'occlusion_by_walls': occ_walls,
            'accepted_visible': (getattr(b, 'id', None) in visible_ids),
        })

    labels = [p['label'] for p in per_object if p['accepted_visible']]
    out = {
        'meta_path': str(path),
        'scene_path': str(scene_path),
        'camera_pos': pos.tolist(),
        'camera_target': tgt.tolist(),
        'position_legal': bool(legal),
        'visible_count': len(visible),
        'visible_labels': labels,
        'per_object_occlusion': per_object,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


# small fallback corner counter if compute_screen_space_visible not available
def count_visible_corners_fallback(pos, tgt, K, bmin, bmax, width, height):
    from view_suite.envs.active_spatial_intelligence.utils.occlusion import camtoworld_from_pos_target, world_to_camera, project_point, point_in_image, aabb_corners
    c2w = camtoworld_from_pos_target(pos, tgt)
    view = np.linalg.inv(np.array(c2w, dtype=float))
    corners = aabb_corners(bmin, bmax)
    pcs = np.array([world_to_camera(view, c) for c in corners])
    uvz = [project_point(np.array(K, dtype=float), pc) for pc in pcs]
    cnt = 0
    for u, v, z in uvz:
        if z > 1e-6 and point_in_image(u, v, width, height, border=2):
            cnt += 1
    return cnt


if __name__ == '__main__':
    main()
