"""
pipeline.py — tabletop RGB-D perception pipeline.

The pipeline follows four explicit stages:

  Stage 1 — RGB-D input
      depth_to_pointcloud(depth, K) converts a depth image + camera intrinsics
      to an (N, 3) point cloud.  If you already have a point cloud, skip this.

  Stage 2 — Table / background removal
      remove_table(pts, ...) runs RANSAC plane fitting to locate the work
      surface, then applies three filters:
        a) depth gate — removes background walls visible in angled views
           (ARID20 "bottom" camera orientation)
        b) height filter — keeps only points physically above the table
        c) XY radius crop — discards points far from the camera origin

  Stage 3 — Instance segmentation
      segment_instances(obj_pts, ...) clusters the foreground cloud with
      DBSCAN then applies up to four splitting passes to separate touching
      or stacked objects:
        Pass 1: horizontal density saddle (XY PCA projection)
        Pass 2: vertical density saddle (Y / height axis)
        Pass 3B: 2-D concavity / narrow neck (XZ projection)
        Pass 3C: height-layer gaps (near-empty Y histogram bands)
      (Pass 3A — normal discontinuity — is disabled: open3d
       estimate_normals segfaults on macOS after DBSCAN.)

  Stage 4 — Shape classification / SQ hint
      Each cluster gets a lightweight geometry-heuristic label
      (Ellipsoid / Cylinder / Cuboid / Other) that biases the subsequent
      Levenberg-Marquardt or SuperDec superquadric fit.

Design notes
  • _clean() forces a numpy copy to break open3d shared-memory views.
  • Stage 2 uses camera-space Z for the depth gate (not world height) so
    that background walls in tilted views are correctly rejected.
  • All stage functions are importable standalone for testing.
"""

import logging
import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ObjectSegment:
    """One segmented object returned by the perception pipeline.

    Consumed downstream by SuperquadricFitter / SuperdecFitter and by
    GraspSelector.  All fields are read-only after construction.
    """
    id:         int
    points:     np.ndarray   # (N, 3) float32, world coordinates
    centroid:   np.ndarray   # (3,) float32
    bbox_min:   np.ndarray   # (3,) float32
    bbox_max:   np.ndarray   # (3,) float32
    n_points:   int   = 0
    shape_type: str   = "Other"   # Ellipsoid | Cylinder | Cuboid | Other
    shape_conf: float = 0.0       # heuristic confidence in [0, 1]

    def __post_init__(self):
        self.n_points = len(self.points)

    @property
    def bbox_extent(self) -> np.ndarray:
        """Bounding-box side lengths in metres."""
        return self.bbox_max - self.bbox_min

    def to_dict(self) -> dict:
        return dict(id=self.id, points=self.points, centroid=self.centroid,
                    bbox_min=self.bbox_min, bbox_max=self.bbox_max,
                    shape_type=self.shape_type, shape_conf=self.shape_conf)


@dataclass
class PerceptionResult:
    """Full output of TabletopPerception.run()."""
    objects:                List[ObjectSegment] = field(default_factory=list)
    table_normal:           np.ndarray = field(default_factory=lambda: np.array([0.,0.,1.]))
    table_height:           float = 0.0
    n_points_input:         int = 0
    n_points_after_denoise: int = 0
    n_points_table:         int = 0

    def summary(self) -> str:
        lines = [
            f"Perception result — {len(self.objects)} object(s) found",
            f"  Input points   : {self.n_points_input:,}",
            f"  After denoise  : {self.n_points_after_denoise:,}",
            f"  Table pts      : {self.n_points_table:,}",
            f"  Table height   : {self.table_height:.3f} m",
            f"  Table normal   : {self.table_normal.round(3)}",
        ]
        for obj in self.objects:
            e = obj.bbox_extent
            lines.append(
                f"  Object {obj.id}: {obj.n_points} pts | "
                f"centroid {obj.centroid.round(3)} | "
                f"extent [{e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}] m | "
                f"shape={obj.shape_type} ({obj.shape_conf:.2f})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _clean(pts: np.ndarray, max_coord: float = 4.0) -> np.ndarray:
    """Remove NaN/inf and out-of-range points; force a numpy copy.

    The copy breaks open3d shared-memory views so that downstream filters
    cannot inadvertently corrupt the open3d point cloud buffer.
    """
    pts = np.array(pts, dtype=np.float64, copy=True)
    good = np.isfinite(pts).all(axis=1) & (np.abs(pts) < max_coord).all(axis=1)
    return pts[good]


def _to_o3d(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd


# ---------------------------------------------------------------------------
# Stage 1 — RGB-D input
# ---------------------------------------------------------------------------

def depth_to_pointcloud(depth: np.ndarray, K: np.ndarray,
                         depth_scale: float = 1000.0,
                         max_depth: float = 4.0) -> np.ndarray:
    """Back-project a depth image to a 3-D point cloud.

    Parameters
    ----------
    depth       : (H, W) uint16 or float depth image
    K           : (3, 3) camera intrinsic matrix (fx, fy, cx, cy)
    depth_scale : divisor that converts raw depth units to metres
                  (1000 for PNG millimetre depths, 1 if already in metres)
    max_depth   : discard points beyond this distance (metres)

    Returns
    -------
    (N, 3) float64 array of XYZ points in camera space (Z forward, Y down)
    """
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    zz = depth.astype(np.float64) / depth_scale
    v, u = np.mgrid[0:H, 0:W]

    x = (u - cx) * zz / fx
    y = (v - cy) * zz / fy
    z = zz

    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = (pts[:, 2] > 0) & (pts[:, 2] < max_depth)
    return pts[valid]


# ---------------------------------------------------------------------------
# Stage 2 — Table / background removal
# ---------------------------------------------------------------------------

def remove_table(
    pts: np.ndarray,
    plane_dist_threshold:   float = 0.012,
    plane_ransac_n:         int   = 3,
    plane_num_iterations:   int   = 1000,
    min_height_above_table: float = 0.005,
    max_height_above_table: float = 0.25,
    depth_margin:           float = 0.25,
    xy_radius:              float = 0.80,
    plane_hint:             Optional[Tuple[np.ndarray, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, int]:
    """Locate the table plane and return only the foreground object points.

    Parameters
    ----------
    pts                    : (N, 3) point cloud (already denoised)
    plane_dist_threshold   : RANSAC inlier distance in metres
    plane_ransac_n         : minimum points per RANSAC hypothesis
    plane_num_iterations   : RANSAC iterations
    min_height_above_table : lower height cutoff above table (metres)
    max_height_above_table : upper height cutoff (metres) — rejects ceiling
    depth_margin           : camera-Z slack beyond table max-Z (metres)
    xy_radius              : XY radius crop centred on camera origin (metres)
    plane_hint             : (normal, height) from a previous frame to skip
                             RANSAC (saves ~200 ms/frame in sequences)

    Returns
    -------
    obj_pts      : (M, 3) foreground points above the table
    table_normal : (3,) unit normal of the fitted plane
    table_height : signed distance of the plane from origin (metres)
    table_pts    : (K, 3) points on the table surface (RANSAC inliers)
    n_table_pts  : number of table inlier points
    """
    if plane_hint is not None:
        table_normal, table_height = plane_hint
        table_normal = np.array(table_normal, dtype=np.float64)
        table_height = float(table_height)
    else:
        pcd = _to_o3d(pts)
        plane_model, _ = pcd.segment_plane(
            distance_threshold=plane_dist_threshold,
            ransac_n=plane_ransac_n,
            num_iterations=plane_num_iterations,
        )
        a, b, c, d = [float(x) for x in plane_model]
        n_raw    = np.array([a, b, c], dtype=np.float64)
        norm_len = float(np.linalg.norm(n_raw))
        if norm_len < 1e-9:
            raise ValueError("RANSAC returned a degenerate plane (zero normal)")

        table_normal = n_raw / norm_len
        table_height = float(-d / norm_len)
        # canonical convention: normal points upward (toward camera)
        if table_height > 0:
            table_normal = -table_normal
            table_height = -table_height

    # recompute inlier mask for this frame's point cloud
    signed_dists = pts @ table_normal - table_height
    table_mask   = np.abs(signed_dists) < plane_dist_threshold * 2
    table_pts    = pts[table_mask]
    n_table_pts  = int(table_mask.sum())

    # --- depth gate: remove background behind table in angled views --------
    table_z_max = float(table_pts[:, 2].max()) if len(table_pts) > 0 else 0.0
    obj_mask    = ~table_mask
    obj_pts     = _clean(pts[obj_mask])

    z_mask  = obj_pts[:, 2] < (table_z_max + depth_margin)
    obj_pts = obj_pts[z_mask]

    # --- height filter: keep points physically above the table -------------
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        heights = obj_pts @ table_normal - table_height
    finite = np.isfinite(heights)
    h_mask  = (finite &
               (heights > min_height_above_table) &
               (heights < max_height_above_table))
    obj_pts = obj_pts[h_mask]

    # --- XY radius crop: discard points far from camera origin -------------
    # Use camera projection origin [0,0] not table centroid — table centroid
    # can be far from objects near scene edges.  OCID cameras point along +Z
    # so XY=[0,0] is always in-view.
    xy_dist = np.linalg.norm(obj_pts[:, :2], axis=1)
    obj_pts = obj_pts[xy_dist < xy_radius]

    return obj_pts, table_normal, table_height, table_pts, n_table_pts


# ---------------------------------------------------------------------------
# Stage 3 helpers — splitting heuristics for touching / stacked objects
# ---------------------------------------------------------------------------

def _split_cluster(pts: np.ndarray, min_pts: int = 30,
                   saddle_depth: float = 0.25, n_bins: int = 40,
                   max_depth: int = 3) -> List[np.ndarray]:
    """Recursively split a cluster at horizontal density saddle points.

    Algorithm:
      1. PCA → project onto the dominant horizontal (XY) axis
      2. Build a 1-D density histogram along that axis
      3. Find local minima where density drops > saddle_depth relative to
         the lower of the two neighbouring peaks
      4. Split at the deepest saddle; recurse on each half

    saddle_depth: 0.25 means a 25 % drop from the local peak triggers a split.
    """
    if len(pts) < min_pts * 2 or max_depth == 0:
        return [pts]

    xy = pts[:, :2].astype(np.float64)
    centered = xy - xy.mean(axis=0)
    cov = (centered.T @ centered) / len(centered)
    eigvals, vecs = np.linalg.eigh(cov)

    if eigvals[-1] < 1e-6:
        return [pts]

    axis = vecs[:, -1]
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        proj = centered @ axis
    if not np.isfinite(proj).all():
        return [pts]

    hist, edges = np.histogram(proj, bins=n_bins)
    hist   = hist.astype(float)
    kernel = np.array([0.25, 0.5, 0.25])
    hist_s = np.convolve(hist, kernel, mode='same')

    best_split_proj = None
    best_saddle_score = 0.0

    for i in range(1, n_bins - 1):
        if hist_s[i] >= hist_s[i-1] or hist_s[i] >= hist_s[i+1]:
            continue
        left_peak  = hist_s[:i].max() if i > 0 else 0.0
        right_peak = hist_s[i+1:].max() if i < n_bins - 1 else 0.0
        lower_peak = min(left_peak, right_peak)
        if lower_peak < 1e-6:
            continue
        drop = 1.0 - hist_s[i] / lower_peak
        if drop > saddle_depth and drop > best_saddle_score:
            best_saddle_score = drop
            best_split_proj   = (edges[i] + edges[i+1]) / 2.0

    if best_split_proj is None:
        return [pts]

    left_pts  = pts[proj <= best_split_proj]
    right_pts = pts[proj >  best_split_proj]

    if len(left_pts) < min_pts or len(right_pts) < min_pts:
        return [pts]

    return (_split_cluster(left_pts,  min_pts, saddle_depth, n_bins, max_depth-1) +
            _split_cluster(right_pts, min_pts, saddle_depth, n_bins, max_depth-1))


def _split_cluster_vertical(pts: np.ndarray, min_pts: int = 30,
                             saddle_depth: float = 0.20,
                             n_bins: int = 30) -> List[np.ndarray]:
    """Split a cluster along the vertical (Y / height-above-table) axis.

    Only triggers when the vertical extent is >= 40 % of the horizontal
    extent, to avoid spurious splits on tall-but-single objects.
    Handles objects stacked on top of each other.
    """
    if len(pts) < min_pts * 2:
        return [pts]

    y_ext  = pts[:, 1].max() - pts[:, 1].min()
    xz_ext = max(pts[:, 0].max() - pts[:, 0].min(),
                 pts[:, 2].max() - pts[:, 2].min())

    if y_ext < 0.40 * xz_ext:
        return [pts]

    hist, edges = np.histogram(pts[:, 1], bins=n_bins)
    hist   = hist.astype(float)
    hist_s = np.convolve(hist, np.array([0.25, 0.5, 0.25]), mode='same')

    best_split, best_score = None, 0.0
    for i in range(1, n_bins - 1):
        if hist_s[i] >= hist_s[i-1] or hist_s[i] >= hist_s[i+1]:
            continue
        left_peak  = hist_s[:i].max() if i > 0 else 0.0
        right_peak = hist_s[i+1:].max() if i < n_bins-1 else 0.0
        lower_peak = min(left_peak, right_peak)
        if lower_peak < 1e-6:
            continue
        drop = 1.0 - hist_s[i] / lower_peak
        if drop > saddle_depth and drop > best_score:
            best_score = drop
            best_split = (edges[i] + edges[i+1]) / 2.0

    if best_split is None:
        return [pts]

    bottom = pts[pts[:, 1] <= best_split]
    top    = pts[pts[:, 1] >  best_split]

    if len(bottom) < min_pts or len(top) < min_pts:
        return [pts]

    return [bottom, top]


def _split_by_concavity(pts: np.ndarray,
                         min_pts: int = 30,
                         concavity_threshold: float = 0.015) -> List[np.ndarray]:
    """Split touching side-by-side objects via a 2-D neck in the XZ projection.

    Side-by-side objects create an hourglass cross-section.  The split fires
    at the narrowest column if it is < 30 % of the overall XZ width.
    """
    if len(pts) < min_pts * 2:
        return [pts]

    try:
        xz = pts[:, [0, 2]].astype(np.float64)
        centered = xz - xz.mean(axis=0)
        cov = (centered.T @ centered) / len(centered)
        eigvals, vecs = np.linalg.eigh(cov)

        if eigvals[-1] < 1e-8:
            return [pts]

        axis = vecs[:, -1]
        perp = vecs[:, 0]
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            proj  = centered @ axis
            width = centered @ perp
        if not (np.isfinite(proj).all() and np.isfinite(width).all()):
            return [pts]

        n_bins = 50
        edges  = np.linspace(proj.min(), proj.max(), n_bins + 1)
        spans, mids = [], []

        for i in range(n_bins):
            mask = (proj >= edges[i]) & (proj < edges[i+1])
            cnt  = mask.sum()
            spans.append(width[mask].max() - width[mask].min() if cnt >= 3 else 0.0)
            mids.append((edges[i] + edges[i+1]) / 2)

        spans = np.array(spans)
        overall_width = spans.max()

        if overall_width < 1e-6:
            return [pts]

        interior  = range(3, n_bins - 3)
        min_idx   = min(interior, key=lambda i: spans[i])
        min_width = spans[min_idx]

        if min_width > overall_width * 0.30:
            return [pts]

        split_pos = mids[min_idx]
        if min(int((proj <= split_pos).sum()),
               int((proj > split_pos).sum())) < len(pts) * 0.20:
            return [pts]

        left  = pts[proj <= split_pos]
        right = pts[proj >  split_pos]

        if len(left) < min_pts or len(right) < min_pts:
            return [pts]

        return [left, right]

    except Exception:
        return [pts]


def _split_by_height_layers(pts: np.ndarray,
                             min_pts: int = 30,
                             gap_threshold: float = 0.008) -> List[np.ndarray]:
    """Recursively split clusters at genuine near-empty height gaps.

    Finds height bands (Y axis) where point density drops to ≤ 0.5 % of
    total and the gap is at least gap_threshold metres wide.  More reliable
    than the saddle splitter for stacked objects.
    """
    if len(pts) < min_pts * 2:
        return [pts]

    y       = pts[:, 1]
    y_range = y.max() - y.min()

    if y_range < gap_threshold * 2:
        return [pts]

    n_bins = max(20, int(y_range / 0.003))
    hist, edges = np.histogram(y, bins=n_bins)

    empty          = hist <= max(1, len(pts) * 0.005)
    best_gap_mid   = None
    best_gap_width = 0.0

    i = 0
    while i < n_bins:
        if empty[i]:
            j = i
            while j < n_bins and empty[j]:
                j += 1
            gap_w = (edges[j] if j < n_bins else edges[-1]) - edges[i]
            if gap_w >= gap_threshold and gap_w > best_gap_width:
                best_gap_width = gap_w
                best_gap_mid   = (edges[i] + (edges[j] if j < n_bins else edges[-1])) / 2
            i = j
        else:
            i += 1

    if best_gap_mid is None:
        return [pts]

    bottom = pts[y <= best_gap_mid]
    top    = pts[y >  best_gap_mid]

    if len(bottom) < min_pts or len(top) < min_pts:
        return [pts]

    return (_split_by_height_layers(bottom, min_pts, gap_threshold) +
            _split_by_height_layers(top,    min_pts, gap_threshold))


# ---------------------------------------------------------------------------
# Stage 3 — Instance segmentation
# ---------------------------------------------------------------------------

def segment_instances(
    obj_pts: np.ndarray,
    obj_col: Optional[np.ndarray] = None,
    cluster_eps:       float = 0.018,
    cluster_min_points: int  = 20,
    split_min_points:  int   = 600,
    saddle_depth:      float = 0.25,
    split_n_bins:      int   = 40,
    concavity_threshold: float = 0.015,
    height_gap_threshold: float = 0.008,
    rgb_weight:        float = 0.0,
) -> List[np.ndarray]:
    """Cluster foreground points into per-object groups.

    Applies DBSCAN followed by four splitting passes to separate objects
    that are touching horizontally, stacked vertically, or joined at a neck.

    Parameters
    ----------
    obj_pts               : (N, 3) foreground points above the table
    obj_col               : (N, 3) float32 RGB colours in [0, 1], optional
    cluster_eps           : DBSCAN neighbourhood radius (metres)
    cluster_min_points    : minimum DBSCAN core-point neighbourhood size
    split_min_points      : cluster size threshold to trigger splitting passes
    saddle_depth          : relative density drop to trigger a saddle split
    split_n_bins          : histogram bins for saddle detection
    concavity_threshold   : width ratio threshold for neck split
    height_gap_threshold  : minimum empty-band height to trigger layer split
    rgb_weight            : colour feature weight for colour-aware DBSCAN
                            (0 = geometry-only; useful for strongly coloured scenes)

    Returns
    -------
    List of (N_i, 3) point arrays, one per detected instance.
    """
    if len(obj_pts) == 0:
        return []

    # DBSCAN clustering
    if obj_col is not None and rgb_weight > 0:
        from sklearn.cluster import DBSCAN as skDBSCAN
        feats  = np.hstack([obj_pts, obj_col * rgb_weight]).astype(np.float32)
        labels = skDBSCAN(eps=cluster_eps, min_samples=cluster_min_points,
                          algorithm='ball_tree', n_jobs=1).fit(feats).labels_
    else:
        pcd_obj = _to_o3d(obj_pts)
        labels  = np.array(pcd_obj.cluster_dbscan(
            eps=cluster_eps,
            min_points=cluster_min_points,
            print_progress=False,
        ))

    # Pass 1 — horizontal saddle split (XY PCA projection)
    raw_clusters: List[np.ndarray] = []
    for lbl in sorted(set(labels) - {-1}):
        cluster = obj_pts[labels == lbl]
        if len(cluster) >= split_min_points:
            raw_clusters.extend(_split_cluster(
                cluster, min_pts=cluster_min_points,
                saddle_depth=saddle_depth, n_bins=split_n_bins))
        else:
            raw_clusters.append(cluster)

    # Pass 2 — vertical saddle split (stacked objects)
    after_vertical: List[np.ndarray] = []
    for cluster in raw_clusters:
        if len(cluster) >= split_min_points:
            after_vertical.extend(_split_cluster_vertical(
                cluster, min_pts=cluster_min_points, saddle_depth=saddle_depth))
        else:
            after_vertical.append(cluster)

    # Pass 3A — normal discontinuity (disabled: segfault on macOS after DBSCAN)
    after_normal = after_vertical

    # Pass 3B — 2-D concavity / narrow neck
    after_concavity: List[np.ndarray] = []
    for cluster in after_normal:
        if len(cluster) >= split_min_points:
            after_concavity.extend(_split_by_concavity(
                cluster, min_pts=cluster_min_points,
                concavity_threshold=concavity_threshold))
        else:
            after_concavity.append(cluster)

    # Pass 3C — height-layer gaps
    after_height: List[np.ndarray] = []
    for cluster in after_concavity:
        after_height.extend(_split_by_height_layers(
            cluster, min_pts=cluster_min_points,
            gap_threshold=height_gap_threshold))

    return after_height


# ---------------------------------------------------------------------------
# Stage 4 — Shape classification / SQ hint
# ---------------------------------------------------------------------------

def classify_shape_hint(cluster: np.ndarray) -> Tuple[str, float]:
    """Extent-ratio heuristic for SQ initialisation hint.

    Does NOT require a trained classifier — works on partial views.
    Returns (shape_type, confidence) where shape_type is used only as an
    LM warm-start; the authoritative shape_type comes from the fitted exponents.
    """
    if len(cluster) < 10:
        return "Other", 0.0
    lo, hi   = cluster.min(0), cluster.max(0)
    exts     = np.sort(hi - lo)              # [min, mid, max]
    r_flat   = exts[0] / (exts[2] + 1e-6)   # low  → flat (bowl/cap)
    r_elon   = exts[1] / (exts[2] + 1e-6)   # low  → elongated (box)
    xy_ext   = hi[:2] - lo[:2]
    r_circ   = min(xy_ext) / (max(xy_ext) + 1e-6)  # high → circular XY

    if r_flat < 0.25:                          # very flat → bowl/disc
        return "Ellipsoid", 0.6
    elif r_flat > 0.45 and r_circ > 0.65:     # squat + circular → can
        return "Cylinder", 0.6
    elif r_elon < 0.45:                        # elongated one axis → box
        return "Cuboid", 0.55
    else:
        return "Other", 0.4


# ---------------------------------------------------------------------------
# Stage 4b — Superquadric fitting (bridge to superquadric.py / superdec_fitter.py)
# ---------------------------------------------------------------------------

def fit_superquadrics(
    segments: list,
    fitter=None,
    l2_threshold: float = 0.008,
    max_primitives: int = 3,
) -> list:
    """Fit superquadric primitives to each segmented object.

    This is Stage 4b of the pipeline — it bridges the instance segmentation
    output (ObjectSegment list) to the SQ fitting layer (SuperquadricFitter
    or SuperdecFitter).

    Parameters
    ----------
    segments : List[ObjectSegment]
        Output of TabletopPerception.run() or segment_instances().
    fitter : SuperquadricFitter or SuperdecFitter, optional
        The fitter instance to use.  If None, creates a default
        SuperquadricFitter (LM optimisation).  Pass a SuperdecFitter instance
        to use the fine-tuned neural fitter instead.
    l2_threshold : float
        Chamfer L2 threshold for adaptive splitting (LM fitter only).
    max_primitives : int
        Maximum primitives per object (LM fitter only).

    Returns
    -------
    List[MultiSQFit] — one MultiSQFit per ObjectSegment, same order.
    The list feeds directly into:
        • fits_to_curobo_obstacles(flat_fits) — collision avoidance
        • Scene.from_fits(flat_fits)           — path planning SDF
        • GraspSelector.grasp_candidates()     — grasp planning

    Example
    -------
    ::
        result  = TabletopPerception().run(pts)
        sq_fits = fit_superquadrics(result.objects)
        flat    = [sq for m in sq_fits for sq in m.primitives]
        scene   = Scene.from_fits(flat)
        sd      = scene.get_signed_distance(query_pts)
    """
    if fitter is None:
        from superquadric import SuperquadricFitter
        fitter = SuperquadricFitter(n_restarts=3, n_lm_rounds=15, subsample=512)

    results = []
    for seg in segments:
        multi = fitter.fit_adaptive(
            seg.points,
            shape_hint=seg.shape_type,
            l2_threshold=l2_threshold,
            max_primitives=max_primitives,
        )
        # propagate shape_type / shape_conf from perception onto each primitive
        for prim in multi.primitives:
            if not prim.shape_type or prim.shape_type == "Other":
                prim.shape_type = seg.shape_type
            prim.shape_conf = seg.shape_conf
        results.append(multi)

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TabletopPerception:
    """Full tabletop RGB-D perception pipeline.

    Orchestrates the four stages (preprocessing → table removal → instance
    segmentation → shape classification) and returns a PerceptionResult
    containing ObjectSegment objects ready for superquadric fitting.

    Parameters
    ----------
    voxel_size              : voxel down-sample resolution (metres)
    nb_neighbors / std_ratio: statistical outlier removal parameters
    plane_dist_threshold    : RANSAC inlier distance for table fitting
    plane_ransac_n          : minimum RANSAC hypothesis size
    plane_num_iterations    : RANSAC iterations
    min_height_above_table  : lower height cutoff above table (metres)
    max_height_above_table  : upper height cutoff (metres)
    depth_margin            : camera-Z slack beyond table back edge (metres).
                              Removes background walls in angled views.
    cluster_eps             : DBSCAN eps (metres)
    cluster_min_points      : DBSCAN min_samples
    min_object_points       : reject segments smaller than this
    max_object_points       : reject segments larger than this
    min_object_extent       : reject segments whose longest bbox side < this
    max_object_extent       : reject segments whose longest bbox side > this
    xy_radius               : discard points > this distance from XY origin
    split_min_points        : trigger splitting passes only above this size
    saddle_depth            : relative saddle depth for horizontal/vertical split
    split_n_bins            : histogram bins for saddle detection
    concavity_threshold     : neck-width ratio for concavity split
    height_gap_threshold    : minimum gap height for layer split (metres)
    geo_min_points          : minimum size to run geometric splits (3B/3C)
    rgb_weight              : colour feature weight for colour-aware DBSCAN
    """

    def __init__(
        self,
        voxel_size:              float = 0.005,
        nb_neighbors:            int   = 20,
        std_ratio:               float = 2.0,
        plane_dist_threshold:    float = 0.012,
        plane_ransac_n:          int   = 3,
        plane_num_iterations:    int   = 1000,
        min_height_above_table:  float = 0.005,
        max_height_above_table:  float = 0.25,
        depth_margin:            float = 0.25,
        cluster_eps:             float = 0.018,
        cluster_min_points:      int   = 20,
        min_object_points:       int   = 50,
        max_object_points:       int   = 5000,
        min_object_extent:       float = 0.02,
        max_object_extent:       float = 0.40,
        xy_radius:               float = 0.80,
        split_min_points:        int   = 600,
        saddle_depth:            float = 0.25,
        split_n_bins:            int   = 40,
        normal_angle_threshold:  float = 60.0,
        concavity_threshold:     float = 0.015,
        height_gap_threshold:    float = 0.008,
        geo_min_points:          int   = 800,
        rgb_weight:              float = 0.0,
    ):
        self.voxel_size             = voxel_size
        self.nb_neighbors           = nb_neighbors
        self.std_ratio              = std_ratio
        self.plane_dist_threshold   = plane_dist_threshold
        self.plane_ransac_n         = plane_ransac_n
        self.plane_num_iterations   = plane_num_iterations
        self.min_height_above_table = min_height_above_table
        self.max_height_above_table = max_height_above_table
        self.depth_margin           = depth_margin
        self.cluster_eps            = cluster_eps
        self.cluster_min_points     = cluster_min_points
        self.min_object_points      = min_object_points
        self.max_object_points      = max_object_points
        self.min_object_extent      = min_object_extent
        self.max_object_extent      = max_object_extent
        self.xy_radius              = xy_radius
        self.split_min_points       = split_min_points
        self.saddle_depth           = saddle_depth
        self.split_n_bins           = split_n_bins
        self.normal_angle_threshold = normal_angle_threshold
        self.concavity_threshold    = concavity_threshold
        self.height_gap_threshold   = height_gap_threshold
        self.geo_min_points         = geo_min_points
        self.rgb_weight             = rgb_weight

    # keep old name as alias so test scripts don't break
    def _classify(self, cluster: np.ndarray) -> Tuple[str, float]:
        return self._shape_hint(cluster)

    def _shape_hint(self, cluster: np.ndarray) -> Tuple[str, float]:
        """Module-level classify_shape_hint(), bound to this instance."""
        return classify_shape_hint(cluster)

    def _valid(self, pts: np.ndarray) -> bool:
        """Return True if a cluster passes size and extent filters."""
        if not (self.min_object_points <= len(pts) <= self.max_object_points):
            return False
        extent = pts.max(axis=0) - pts.min(axis=0)
        return self.min_object_extent <= extent.max() <= self.max_object_extent

    def run(self, points: np.ndarray,
            rgb:        Optional[np.ndarray] = None,
            plane_hint: Optional[tuple]      = None) -> PerceptionResult:
        """Run the full perception pipeline on a point cloud.

        Parameters
        ----------
        points     : (N, 3) XYZ point cloud in metres
        rgb        : (N, 3) float32 / uint8 colours, same row order as points
        plane_hint : (normal, height) from a previous run on the same sequence.
                     If provided, skips RANSAC (saves ~200 ms/frame).

        Returns
        -------
        PerceptionResult with ObjectSegment list and table geometry.
        """
        result = PerceptionResult(n_points_input=len(points))

        # ── preprocess ──────────────────────────────────────────────────────
        pts = _clean(points)
        if len(pts) < 100:
            return result
        result.n_points_input = len(pts)

        good_mask = np.isfinite(points).all(axis=1) & (np.abs(points) < 4.0).all(axis=1)
        if rgb is not None and len(rgb) == len(points):
            col = np.array(rgb, dtype=np.float32)[good_mask]
            if col.max() > 1.5:
                col = col / 255.0
            col = np.clip(col, 0.0, 1.0)
        else:
            col = None

        # voxel down-sample
        pcd = _to_o3d(pts)
        if col is not None:
            pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64))
        pcd = pcd.voxel_down_sample(self.voxel_size)

        # statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(self.nb_neighbors, self.std_ratio)
        pts    = _clean(np.asarray(pcd.points))
        col_ds = np.asarray(pcd.colors).astype(np.float32) if (col is not None and pcd.has_colors()) else None
        if len(pts) < 50:
            return result
        result.n_points_after_denoise = len(pts)

        # ── Stage 2: table / background removal ─────────────────────────────
        try:
            obj_pts, table_normal, table_height, table_pts, n_table_pts = remove_table(
                pts,
                plane_dist_threshold   = self.plane_dist_threshold,
                plane_ransac_n         = self.plane_ransac_n,
                plane_num_iterations   = self.plane_num_iterations,
                min_height_above_table = self.min_height_above_table,
                max_height_above_table = self.max_height_above_table,
                depth_margin           = self.depth_margin,
                xy_radius              = self.xy_radius,
                plane_hint             = plane_hint,
            )
        except Exception as e:
            logging.warning("[pipeline] table removal failed: %s", e)
            return result

        result.table_normal   = table_normal
        result.table_height   = table_height
        result.n_points_table = n_table_pts

        if len(obj_pts) < self.cluster_min_points:
            return result

        # propagate colour mask through depth + height filters if present
        # (only needed for colour-aware DBSCAN — geometry path ignores col_ds)
        obj_col = None
        if col_ds is not None and self.rgb_weight > 0:
            # reconstruct which rows of pts survived into obj_pts;
            # use a KD-tree match since obj_pts went through _clean()
            try:
                import open3d as _o3d
                pcd_all = _to_o3d(pts)
                pcd_obj = _to_o3d(obj_pts)
                tree = _o3d.geometry.KDTreeFlann(pcd_all)
                col_obj = []
                for p in obj_pts:
                    _, idx, _ = tree.search_knn_vector_3d(p, 1)
                    col_obj.append(col_ds[idx[0]])
                obj_col = np.array(col_obj, dtype=np.float32)
            except Exception:
                logging.debug("[pipeline] colour KD-tree lookup failed; running geometry-only DBSCAN")
                obj_col = None

        # ── Stage 3: instance segmentation ──────────────────────────────────
        clusters = segment_instances(
            obj_pts,
            obj_col              = obj_col,
            cluster_eps          = self.cluster_eps,
            cluster_min_points   = self.cluster_min_points,
            split_min_points     = self.split_min_points,
            saddle_depth         = self.saddle_depth,
            split_n_bins         = self.split_n_bins,
            concavity_threshold  = self.concavity_threshold,
            height_gap_threshold = self.height_gap_threshold,
            rgb_weight           = self.rgb_weight,
        )

        # ── Stage 4: shape hint + validity filter ────────────────────────────
        obj_id = 0
        for cluster in clusters:
            if not self._valid(cluster):
                continue
            shape_type, shape_conf = self._shape_hint(cluster)
            result.objects.append(ObjectSegment(
                id=obj_id,
                points=cluster.astype(np.float32),
                centroid=cluster.mean(axis=0),
                bbox_min=cluster.min(axis=0),
                bbox_max=cluster.max(axis=0),
                shape_type=shape_type,
                shape_conf=shape_conf,
            ))
            obj_id += 1

        return result

    def run_from_file(self, path: str) -> PerceptionResult:
        """Convenience wrapper — load a PLY file and run the pipeline."""
        pcd = o3d.io.read_point_cloud(path)
        return self.run(np.asarray(pcd.points))
