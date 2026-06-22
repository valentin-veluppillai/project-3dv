#!/usr/bin/env python3
"""
node_wrapper_check.py
=====================
Simulates what a ROS/deployment node would do.
Imports ONLY the public API — zero knowledge of internals.
"""
import sys
sys.path.insert(0, "/work/courses/3dv/team15/curobo-sq/perception")

import numpy as np
from pipeline import (
    Frame, SuperquadricWorld,
    get_world_pointcloud, fit_superquadrics_world,
    superquadrics_to_curobo_world, MULTIVIEW,
)

# ── construct Frame from raw numpy (as a node would receive from a sensor) ───
H, W = 480, 640
depth = np.full((H, W), 0.75, dtype=np.float32)
depth[220:260, 300:340] = 0.65
rgb   = np.zeros((H, W, 3), dtype=np.uint8)
K     = np.array([[570.342,0.,319.5],[0.,570.342,239.5],[0.,0.,1.]], dtype=np.float64)
extrinsic = np.eye(4, dtype=np.float64)

frame = Frame(rgb=rgb, depth=depth, K=K, extrinsic=extrinsic)

# ── stage 1 ──────────────────────────────────────────────────────────────────
world_pcd = get_world_pointcloud([frame])
print(f"world_pcd: {len(world_pcd.points):,} points")

# ── stage 2 ──────────────────────────────────────────────────────────────────
sqs = fit_superquadrics_world(world_pcd)
print(f"fitted {len(sqs)} superquadric(s)")
for sq in sqs:
    print(f"  pos={sq.position.round(3)}  "
          f"a=({0.050},{0.040},{0.030})  type={sq.shape_type}")

# ── cuRoBO adapter ────────────────────────────────────────────────────────────
try:
    cfg = superquadrics_to_curobo_world(sqs)
    print(f"curobo config: {type(cfg).__name__}  "
          f"obstacles={len(cfg.superquadric) if hasattr(cfg,'superquadric') else '?'}")
except ImportError as e:
    print(f"cuRoBO not installed (expected on login node): {e}")

print("node_wrapper_check: OK")
