"""
run_on_frame.py — run perception + SuperDec on a single OCID-style RGB-D frame
and dump the fitted superquadrics as PLY meshes.

Inputs (defaults point at the trio under project-3dv/data/):
    --rgb     <path>.png   (8-bit RGB, 640x480)
    --depth   <path>.png   (16-bit, mm; OCID/Asus Xtion depth_scale=1000)
    --label   <path>.png   (16-bit instance mask; optional, currently unused)
    --ckpt_dir <path>      SuperDec checkpoint dir (auto-picks latest epoch_*.pt)
    --superdec_dir <path>  SuperDec repo root (default: superdec_concave)
    --out_dir <path>       Output directory

Outputs (in --out_dir):
    scene_fit.ply        — combined mesh of all fitted SQs (per-object colour)
    obj_<NN>.ply         — per-object SQ mesh
    scene_segments.ply   — input segmented point cloud (per-object colour)
    scene_fit.npz        — SQ parameters [sx,sy,sz,e1,e2,tx,ty,tz,rx,ry,rz]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

# OCID / Asus Xtion intrinsics
ASUS_XTION_K = np.array([
    [570.3422241210938, 0.0,               319.5],
    [0.0,               570.3422241210938, 239.5],
    [0.0,               0.0,               1.0  ],
], dtype=np.float64)
DEPTH_SCALE = 1000.0  # uint16 mm → metres


def _euler_to_rot(euler):
    rx, ry, rz = euler
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx


def sq_mesh(prim, n_u: int = 40, n_v: int = 60) -> trimesh.Trimesh:
    """Tessellate a superquadric primitive into a closed-ish triangle mesh.

    Wraps in v (azimuth) and caps at the u poles via a fan.
    """
    sx, sy, sz, e1, e2 = (prim.sx, prim.sy, prim.sz, prim.e1, prim.e2)
    t = np.array([prim.tx, prim.ty, prim.tz], dtype=np.float64)
    R = _euler_to_rot([prim.rx, prim.ry, prim.rz])

    u = np.linspace(-np.pi/2, np.pi/2, n_u)
    v = np.linspace(-np.pi,   np.pi,   n_v, endpoint=False)  # wrap
    uu, vv = np.meshgrid(u, v, indexing='ij')   # (n_u, n_v)

    def sp(x, p):
        return np.sign(x) * (np.abs(x) ** p)

    x = sx * sp(np.cos(uu), e1) * sp(np.cos(vv), e2)
    y = sy * sp(np.cos(uu), e1) * sp(np.sin(vv), e2)
    z = sz * sp(np.sin(uu), e1)
    pts_canon = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    verts = (R @ pts_canon.T).T + t

    faces = []
    for i in range(n_u - 1):
        for j in range(n_v):
            j2 = (j + 1) % n_v
            a = i * n_v + j
            b = i * n_v + j2
            c = (i + 1) * n_v + j
            d = (i + 1) * n_v + j2
            faces.append([a, c, b])
            faces.append([b, c, d])
    faces = np.asarray(faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _color_for(idx: int) -> np.ndarray:
    palette = np.array([
        [228, 26, 28],   [55, 126, 184],  [77, 175, 74],
        [152, 78, 163],  [255, 127, 0],   [255, 255, 51],
        [166, 86, 40],   [247, 129, 191], [153, 153, 153],
        [102, 194, 165], [252, 141, 98],  [141, 160, 203],
        [231, 138, 195], [166, 216, 84],  [255, 217, 47],
    ], dtype=np.uint8)
    return palette[idx % len(palette)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb",   default="data/rgb_result_2018-08-20-14-34-01.png")
    ap.add_argument("--depth", default="data/depth_result_2018-08-20-14-34-01.png")
    ap.add_argument("--label", default="data/label_result_2018-08-20-14-34-01.png")
    ap.add_argument("--ckpt_dir",
        default="/work/courses/3dv/team15/superdec_concave/checkpoints/expocc_tt_chamfer")
    ap.add_argument("--superdec_dir",
        default="/work/courses/3dv/team15/superdec_concave")
    ap.add_argument("--out_dir", default="data/fit_out")
    ap.add_argument("--n_u", type=int, default=40)
    ap.add_argument("--n_v", type=int, default=60)
    ap.add_argument("--max_depth", type=float, default=2.0)
    ap.add_argument("--use_gt_label", action="store_true",
        help="Skip TabletopPerception; segment objects via the GT label PNG.")
    ap.add_argument("--min_obj_pts", type=int, default=500,
        help="Drop label IDs with fewer than this many valid-depth pixels.")
    ap.add_argument("--exclude_ids", default="1",
        help="Comma-separated label IDs to skip (default: '1' — the table).")
    ap.add_argument("--exist_threshold", type=float, default=0.5,
        help="Per-primitive existence threshold (higher = fewer, cleaner SQs).")
    ap.add_argument("--table_frame", action="store_true",
        help="Fit a plane to the table cloud (label id=1) and rotate each "
             "object into table-up frame before SuperDec inference.")
    ap.add_argument("--table_id", type=int, default=1,
        help="Label ID treated as the table for plane fitting (default 1).")
    ap.add_argument("--to_scene_npz", default=None,
        help="Path to a TO-Scene .npz (xyz + instance_label). When set, "
             "skips RGB-D loading and uses the npz's instance labels directly.")
    ap.add_argument("--semantic_min", type=int, default=None,
        help="TO-Scene only: keep only instances whose dominant semantic "
             "label is >= this. In TO_Crowd, sem >= 41 selects the 12 "
             "tabletop-added object classes (vs ScanNet context like floor "
             "/chair/sofa with sem <= 7).")
    args = ap.parse_args()

    # data/, outputs/ were not moved — they still live in project-3dv.
    proj_root = Path("/work/courses/3dv/team15/project-3dv")
    perception_dir = Path(__file__).resolve().parent.parent   # curobo-sq/perception/
    sys.path.insert(0, str(perception_dir))

    os.environ["SUPERDEC_DIR"]      = args.superdec_dir
    os.environ["SUPERDEC_CKPT_DIR"] = args.ckpt_dir
    if args.superdec_dir not in sys.path:
        sys.path.insert(0, args.superdec_dir)

    out_dir    = Path(args.out_dir)
    if not out_dir.is_absolute():    out_dir    = proj_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    to_scene_mode = args.to_scene_npz is not None
    depth = label = None

    if to_scene_mode:
        npz_path = Path(args.to_scene_npz)
        if not npz_path.is_absolute():
            npz_path = proj_root / npz_path
        print(f"[run_on_frame] to_scene_npz : {npz_path}")
        print(f"[run_on_frame] ckpt         : {args.ckpt_dir}")
        print(f"[run_on_frame] out          : {out_dir}")
    else:
        rgb_path   = Path(args.rgb)
        depth_path = Path(args.depth)
        if not rgb_path.is_absolute():   rgb_path   = proj_root / rgb_path
        if not depth_path.is_absolute(): depth_path = proj_root / depth_path
        print(f"[run_on_frame] rgb   : {rgb_path}")
        print(f"[run_on_frame] depth : {depth_path}")
        print(f"[run_on_frame] ckpt  : {args.ckpt_dir}")
        print(f"[run_on_frame] out   : {out_dir}")

        depth_u16 = np.array(Image.open(depth_path), dtype=np.uint16)
        depth = depth_u16.astype(np.float32) / DEPTH_SCALE
        depth[(depth <= 0.1) | (depth > args.max_depth)] = 0.0
        print(f"[run_on_frame] depth shape={depth.shape} valid={int((depth>0).sum())}")

    from pipeline import single_frame_pipeline, pointcloud_from_depth
    from superdec_fitter import SuperdecFitter

    # Build the fitter explicitly so we can (a) point at expocc_tt_chamfer and
    # (b) bypass single_frame_pipeline's hard CUDA check (it accepts a pre-built
    # fitter instance and skips its own resolution path).
    fitter = SuperdecFitter(
        superdec_dir=args.superdec_dir,
        checkpoint_dir=args.ckpt_dir,
        exist_threshold=args.exist_threshold,
    )
    print(f"[run_on_frame] exist_threshold={args.exist_threshold}")

    seg_clouds = []   # populated either by GT-label / TO-Scene / TabletopPerception
    table_normal, table_height = None, 0.0   # may be set in either branch

    if to_scene_mode:
        d = np.load(npz_path, allow_pickle=True)
        xyz   = np.asarray(d["xyz"], dtype=np.float64)
        inst  = np.asarray(d["instance_label"]).astype(np.int64)
        sem   = np.asarray(d["semantic_label"]).astype(np.int64)
        print(f"[run_on_frame] to_scene: {len(xyz)} pts, "
              f"extent={(xyz.max(0)-xyz.min(0)).round(3)}")
        excluded = {int(s) for s in args.exclude_ids.split(",") if s.strip()}
        ids = sorted(int(i) for i in np.unique(inst)
                     if int(i) not in excluded)
        if args.semantic_min is not None:
            kept = []
            for iid in ids:
                m = inst == iid
                sems, cnts = np.unique(sem[m], return_counts=True)
                dom = int(sems[cnts.argmax()])
                if dom >= args.semantic_min:
                    kept.append(iid)
            print(f"[run_on_frame] semantic_min={args.semantic_min}: "
                  f"{len(ids)} → {len(kept)} instances")
            ids = kept
        print(f"[run_on_frame] TO-Scene instance IDs "
              f"(excluding {sorted(excluded)}): {ids}")
        for obj_id in ids:
            mask = inst == obj_id
            n_valid = int(mask.sum())
            if n_valid < args.min_obj_pts:
                print(f"  id={obj_id}: only {n_valid} pts, skipping")
                continue
            seg_clouds.append((obj_id, xyz[mask].astype(np.float32)))
            print(f"  id={obj_id}: {n_valid} pts")

        if args.table_frame:
            import open3d as _o3d
            tmask = inst == args.table_id
            n_tbl = int(tmask.sum())
            if n_tbl < 500:
                print(f"[run_on_frame] table_frame: id={args.table_id} has "
                      f"only {n_tbl} pts — disabling table frame")
            else:
                pcd = _o3d.geometry.PointCloud()
                pcd.points = _o3d.utility.Vector3dVector(xyz[tmask])
                plane, _inl = pcd.segment_plane(
                    distance_threshold=0.005, ransac_n=3, num_iterations=500)
                a, b, c, dd = plane
                n_vec = np.array([a, b, c], dtype=np.float64)
                nrm = float(np.linalg.norm(n_vec)) + 1e-12
                n_vec /= nrm
                table_height = -dd / nrm
                ref = seg_clouds[0][1].mean(axis=0).astype(np.float64)
                if float(ref @ n_vec - table_height) < 0:
                    n_vec = -n_vec
                    table_height = -table_height
                table_normal = n_vec
                print(f"[run_on_frame] table_frame: normal={table_normal.round(4)}, "
                      f"height={table_height:.4f} m  ({n_tbl} table pts)")

    elif args.use_gt_label:
        label_path = Path(args.label)
        if not label_path.is_absolute():
            label_path = proj_root / label_path
        label = np.array(Image.open(label_path), dtype=np.int32)
        assert label.shape == depth.shape, (
            f"label {label.shape} != depth {depth.shape}")
        excluded = {int(s) for s in args.exclude_ids.split(",") if s.strip()}
        ids = sorted(int(i) for i in np.unique(label)
                     if i > 0 and int(i) not in excluded)
        print(f"[run_on_frame] GT label IDs (after excluding {sorted(excluded)}): {ids}")
        fx, fy = ASUS_XTION_K[0,0], ASUS_XTION_K[1,1]
        cx, cy = ASUS_XTION_K[0,2], ASUS_XTION_K[1,2]
        H, W = depth.shape
        v_idx, u_idx = np.indices((H, W))
        for obj_id in ids:
            mask = (label == obj_id) & (depth > 0)
            n_valid = int(mask.sum())
            if n_valid < args.min_obj_pts:
                print(f"  id={obj_id}: only {n_valid} valid pts, skipping")
                continue
            Z = depth[mask]
            X = (u_idx[mask] - cx) * Z / fx
            Y = (v_idx[mask] - cy) * Z / fy
            pts = np.stack([X, Y, Z], axis=1).astype(np.float32)
            seg_clouds.append((obj_id, pts))
            print(f"  id={obj_id}: {len(pts)} pts")

        # Optionally fit a plane to the table cloud → table-up canonicalisation.
        if args.table_frame:
            import open3d as _o3d
            tmask = (label == args.table_id) & (depth > 0)
            n_tbl = int(tmask.sum())
            if n_tbl < 500:
                print(f"[run_on_frame] table_frame: id={args.table_id} has "
                      f"only {n_tbl} pts — disabling table frame")
            else:
                Z = depth[tmask]
                X = (u_idx[tmask] - cx) * Z / fx
                Y = (v_idx[tmask] - cy) * Z / fy
                tbl_pts = np.stack([X, Y, Z], axis=1).astype(np.float64)
                pcd = _o3d.geometry.PointCloud()
                pcd.points = _o3d.utility.Vector3dVector(tbl_pts)
                plane, _inl = pcd.segment_plane(
                    distance_threshold=0.005, ransac_n=3, num_iterations=500)
                a, b, c, d = plane
                n_vec = np.array([a, b, c], dtype=np.float64)
                nrm = float(np.linalg.norm(n_vec)) + 1e-12
                n_vec /= nrm
                table_height = -d / nrm
                # orient: ensure objects sit "above" the plane along the normal
                ref = seg_clouds[0][1].mean(axis=0).astype(np.float64)
                if float(ref @ n_vec - table_height) < 0:
                    n_vec = -n_vec
                    table_height = -table_height
                table_normal = n_vec
                print(f"[run_on_frame] table_frame: normal={table_normal.round(4)}, "
                      f"height={table_height:.4f} m  ({n_tbl} table pts)")

    if seg_clouds:
        # Run SuperDec per object via pipeline helpers (preprocess/postprocess
        # mirror single_frame_pipeline).
        from pipeline import preprocess_pointcloud, postprocess_fits
        import time as _t
        t0 = _t.time()
        # SuperDec was trained on ShapeNet (Y-up). The dataloader rotates any
        # z-up dataset by -90° around X before training; the demo scripts do
        # the same at inference. We mirror that here: z-up world → y-up for
        # SuperDec → invert on the output primitive poses.
        R_z2y = np.array([[1, 0, 0],
                          [0, 0, 1],
                          [0,-1, 0]], dtype=np.float64)
        R_y2z = R_z2y.T

        import scipy.spatial.transform as _sst

        fits = []
        for obj_id, pts in seg_clouds:
            try:
                # Center at object's centroid so SuperdecFitter |coord|<=2
                # contract holds when the object sits far from origin.
                centroid_w = pts.mean(axis=0).astype(np.float64)
                pts_centered = (pts.astype(np.float64) - centroid_w)
                pts_pre, _n, meta = preprocess_pointcloud(
                    pts_centered, for_superdec=True,
                    table_normal=table_normal, table_height=table_height,
                )
                # z-up world → y-up canonical (training convention)
                pts_yup = pts_pre @ R_z2y.T
                multi = fitter.fit_adaptive(pts_yup)
                # Rotate each predicted primitive y-up → z-up (inverse)
                for prim in multi.primitives:
                    t = np.array([prim.tx, prim.ty, prim.tz])
                    t_z = R_y2z @ t
                    prim.tx, prim.ty, prim.tz = float(t_z[0]), float(t_z[1]), float(t_z[2])
                    R_yup = _sst.Rotation.from_euler(
                        "xyz", [prim.rx, prim.ry, prim.rz]).as_matrix()
                    R_zup = R_y2z @ R_yup
                    rx, ry, rz = _sst.Rotation.from_matrix(R_zup).as_euler("xyz")
                    prim.rx, prim.ry, prim.rz = float(rx), float(ry), float(rz)
                multi = postprocess_fits([multi], meta)[0]
                for prim in multi.primitives:
                    prim.tx += float(centroid_w[0])
                    prim.ty += float(centroid_w[1])
                    prim.tz += float(centroid_w[2])
            except Exception as e:
                print(f"  id={obj_id}: fit failed ({e})")
                continue
            fits.append(multi)
        timing = {'fit_total_s': _t.time() - t0,
                  'n_segments': len(seg_clouds)}
    else:
        fits, world_model, timing = single_frame_pipeline(
            depth=depth, K=ASUS_XTION_K, extrinsic=None, fitter=fitter,
        )
    print(f"[run_on_frame] timing: {timing}")
    print(f"[run_on_frame] objects fitted: {len(fits)}")
    if not fits:
        print("[run_on_frame] no objects — exiting")
        return

    import open3d as o3d

    all_verts, all_faces, all_vcols = [], [], []
    sq_records = []
    vert_offset = 0
    for i, multi in enumerate(fits):
        col_u8 = _color_for(i)
        col_f  = col_u8.astype(np.float64) / 255.0
        per_obj_v, per_obj_f, per_obj_c = [], [], []
        per_obj_voff = 0
        for j, prim in enumerate(multi.primitives):
            m = sq_mesh(prim, n_u=args.n_u, n_v=args.n_v)
            v = np.asarray(m.vertices)
            f = np.asarray(m.faces)
            per_obj_v.append(v)
            per_obj_f.append(f + per_obj_voff)
            per_obj_c.append(np.tile(col_f, (len(v), 1)))
            per_obj_voff += len(v)
            sq_records.append({
                "obj_id": i, "prim_id": j,
                "params": np.array([prim.sx, prim.sy, prim.sz, prim.e1, prim.e2,
                                    prim.tx, prim.ty, prim.tz,
                                    prim.rx, prim.ry, prim.rz], dtype=np.float64),
                "shape_type": getattr(prim, "shape_type", "Other"),
                "shape_conf": float(getattr(prim, "shape_conf", 0.0)),
                "chamfer_l2": float(getattr(prim, "chamfer_l2", 0.0)),
            })
        if not per_obj_v:
            continue
        ov = np.concatenate(per_obj_v)
        of = np.concatenate(per_obj_f)
        oc = np.concatenate(per_obj_c)

        # per-object PLY via Open3D (writes vertex colors reliably)
        o3m = o3d.geometry.TriangleMesh()
        o3m.vertices       = o3d.utility.Vector3dVector(ov)
        o3m.triangles      = o3d.utility.Vector3iVector(of)
        o3m.vertex_colors  = o3d.utility.Vector3dVector(oc)
        o3d.io.write_triangle_mesh(str(out_dir / f"obj_{i:02d}.ply"), o3m,
                                   write_ascii=False, write_vertex_colors=True)
        print(f"  obj {i:02d}: {len(multi.primitives)} primitives, "
              f"{len(ov)} verts → obj_{i:02d}.ply")

        all_verts.append(ov)
        all_faces.append(of + vert_offset)
        all_vcols.append(oc)
        vert_offset += len(ov)

    if all_verts:
        scene = o3d.geometry.TriangleMesh()
        scene.vertices      = o3d.utility.Vector3dVector(np.concatenate(all_verts))
        scene.triangles     = o3d.utility.Vector3iVector(np.concatenate(all_faces))
        scene.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(all_vcols))
        o3d.io.write_triangle_mesh(str(out_dir / "scene_fit.ply"), scene,
                                   write_ascii=False, write_vertex_colors=True)
        print(f"[run_on_frame] scene_fit.ply: {len(scene.vertices)} verts, "
              f"{len(scene.triangles)} faces")

    if sq_records:
        np.savez(out_dir / "scene_fit.npz",
                 obj_id   =np.array([r["obj_id"]   for r in sq_records]),
                 prim_id  =np.array([r["prim_id"]  for r in sq_records]),
                 params   =np.stack([r["params"]   for r in sq_records]),
                 shape    =np.array([r["shape_type"] for r in sq_records]),
                 conf     =np.array([r["shape_conf"] for r in sq_records]),
                 chamfer  =np.array([r["chamfer_l2"] for r in sq_records]))
        print(f"[run_on_frame] scene_fit.npz: {len(sq_records)} primitives")

    if seg_clouds:
        pc_pts, pc_col = [], []
        for i, (_oid, pts) in enumerate(seg_clouds):
            col_f = _color_for(i).astype(np.float64) / 255.0
            pc_pts.append(pts)
            pc_col.append(np.tile(col_f, (len(pts), 1)))
        all_pts  = np.concatenate(pc_pts)
        all_cols = np.concatenate(pc_col)
        seg_pcd = o3d.geometry.PointCloud()
        seg_pcd.points = o3d.utility.Vector3dVector(all_pts.astype(np.float64))
        seg_pcd.colors = o3d.utility.Vector3dVector(all_cols)
        o3d.io.write_point_cloud(str(out_dir / "scene_segments.ply"), seg_pcd,
                                 write_ascii=False)
        print(f"[run_on_frame] scene_segments.ply: {len(all_pts)} pts")

    print("[run_on_frame] done")


if __name__ == "__main__":
    main()
