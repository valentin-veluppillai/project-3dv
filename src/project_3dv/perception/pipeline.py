"""
pipeline.py — tabletop perception for real RGB-D data (OCID-compatible)

Key design decisions:
  1. _clean() forces a numpy copy to break open3d shared-memory views
  2. After RANSAC, a Z-depth filter removes background (walls, furniture)
     that is geometrically "above" the tilted table plane but spatially
     behind it in camera space. Height filtering alone fails when the
     camera views the table at an angle (ARID20 "bottom" view).
  3. Height filter then isolates only points physically above the table.
  4. Shape classifier (sq_shape_library.py) runs on each segment to
     predict SQ type (Ellipsoid/Cylinder/Cuboid/Other) before fitting.
"""

import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from typing import List, Optional

# ── optional shape classifier (requires sq_shape_classifier.pkl) ─────────────
try:
    from sq_shape_library import classify_shape
    _USE_CLASSIFIER  = True
    _SQ_CLASSIFIER   = "sq_shape_classifier.pkl"
except ImportError:
    _USE_CLASSIFIER  = False
    _SQ_CLASSIFIER   = None


@dataclass
class ObjectSegment:
    id:         int
    points:     np.ndarray
    centroid:   np.ndarray
    bbox_min:   np.ndarray
    bbox_max:   np.ndarray
    n_points:   int   = 0
    shape_type: str   = "Other"   # Ellipsoid | Cylinder | Cuboid | Other
    shape_conf: float = 0.0       # classifier confidence in [0, 1]

    def __post_init__(self):
        self.n_points = len(self.points)

    @property
    def bbox_extent(self):
        return self.bbox_max - self.bbox_min

    def to_dict(self):
        return dict(id=self.id, points=self.points, centroid=self.centroid,
                    bbox_min=self.bbox_min, bbox_max=self.bbox_max,
                    shape_type=self.shape_type, shape_conf=self.shape_conf)


@dataclass
class PerceptionResult:
    objects:                List[ObjectSegment] = field(default_factory=list)
    table_normal:           np.ndarray = field(default_factory=lambda: np.array([0.,0.,1.]))
    table_height:           float = 0.0
    n_points_input:         int = 0
    n_points_after_denoise: int = 0
    n_points_table:         int = 0

    def summary(self):
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


def _clean(pts: np.ndarray, max_coord: float = 4.0) -> np.ndarray:
    """
    Sanitise point cloud — forces a copy so open3d shared-memory
    views can't reintroduce bad values through the same buffer.
    Removes NaN/inf and points with any coordinate > max_coord (4m).
    """
    pts = np.array(pts, dtype=np.float64, copy=True)
    good = np.isfinite(pts).all(axis=1) & (np.abs(pts) < max_coord).all(axis=1)
    return pts[good]


def _split_cluster(pts: np.ndarray, min_pts: int = 30,
                   saddle_depth: float = 0.25, n_bins: int = 40,
                   max_depth: int = 3) -> List[np.ndarray]:
    """
    Recursively split a point cluster at density saddle points.

    Algorithm:
      1. PCA → project onto principal axis in XY (table plane)
      2. Build 1D density histogram along that axis
      3. Find local minima where density drops > saddle_depth fraction
         relative to the lower of the two surrounding peaks
      4. Split at the deepest saddle; recurse on each half

    saddle_depth: 0.25 = need 25% drop from local peak to split
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
    hist = hist.astype(float)
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

    left_mask  = proj <= best_split_proj
    right_mask = ~left_mask
    left_pts   = pts[left_mask]
    right_pts  = pts[right_mask]

    if len(left_pts) < min_pts or len(right_pts) < min_pts:
        return [pts]

    return (_split_cluster(left_pts,  min_pts, saddle_depth, n_bins, max_depth-1) +
            _split_cluster(right_pts, min_pts, saddle_depth, n_bins, max_depth-1))


def _split_cluster_vertical(pts: np.ndarray, min_pts: int = 30,
                             saddle_depth: float = 0.20,
                             n_bins: int = 30) -> List[np.ndarray]:
    """
    Split a cluster vertically (along Y = height above table).
    Handles objects stacked on top of each other.
    Only triggers when vertical extent is >= 40% of horizontal extent.
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


def _split_by_normal_discontinuity(pts: np.ndarray,
                                    min_pts: int = 30,
                                    angle_threshold_deg: float = 60.0,
                                    k_neighbors: int = 16) -> List[np.ndarray]:
    """
    Approach A — Normal discontinuity splitting.
    DISABLED: open3d estimate_normals causes a segfault on macOS when called
    after the main pipeline's open3d DBSCAN. Safe to re-enable on Linux/GPU.
    """
    return [pts]


def _split_by_concavity(pts: np.ndarray,
                         min_pts: int = 30,
                         concavity_threshold: float = 0.015) -> List[np.ndarray]:
    """
    Approach B — 2D concavity (neck) splitting.
    Side-by-side touching objects create an hourglass shape in XZ projection.
    Splits at the narrowest neck if sufficiently narrow relative to overall extent.
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

        split_pos   = mids[min_idx]
        left_count  = (proj <= split_pos).sum()
        right_count = (proj >  split_pos).sum()
        if min(left_count, right_count) < len(pts) * 0.20:
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
    """
    Approach C — Height layer segmentation.
    Finds genuine near-zero density gaps in Y (height above table).
    More reliable than saddle splitter for stacked objects.
    gap_threshold: minimum gap width in metres to trigger a split (8mm).
    """
    if len(pts) < min_pts * 2:
        return [pts]

    y       = pts[:, 1]
    y_range = y.max() - y.min()

    if y_range < gap_threshold * 2:
        return [pts]

    n_bins = max(20, int(y_range / 0.003))
    hist, edges = np.histogram(y, bins=n_bins)

    empty            = hist <= max(1, len(pts) * 0.005)
    best_gap_mid     = None
    best_gap_width   = 0.0

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


def _to_o3d(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd


class TabletopPerception:
    """
    Parameters
    ----------
    depth_margin : float
        After finding the table plane, keep only object candidates
        with camera-space Z < (max Z of table inliers) + depth_margin.
        This removes background walls that appear geometrically "above"
        the tilted table plane. Default 0.25 m works for ARID20 bottom view.
    classifier_path : str
        Path to sq_shape_classifier.pkl produced by sq_shape_library.py.
        If None, falls back to geometry-only shape type assignment.
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
        classifier_path:         str   = "sq_shape_classifier.pkl",
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
        self.classifier_path        = classifier_path

        # load classifier once at init time if available
        self._classifier_ready = False
        if _USE_CLASSIFIER and classifier_path:
            import os
            if os.path.exists(classifier_path):
                self._classifier_ready = True
            else:
                print(f"  [pipeline] shape classifier not found at {classifier_path} "
                      f"— run sq_shape_library.py --mode both first")

    def _shape_hint(self, cluster: np.ndarray):
        """
        Extent-ratio heuristic for SQ initialisation hint.
        Does NOT require a trained classifier — works on partial views.
        Returns (shape_type, confidence) where type is used only as an
        LM warm-start; authoritative shape_type comes from fitted exponents.
        """
        if len(cluster) < 10:
            return "Other", 0.0
        lo, hi   = cluster.min(0), cluster.max(0)
        exts     = np.sort(hi - lo)          # [min, mid, max]
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

    # keep old name as alias so test scripts don't break
    def _classify(self, cluster: np.ndarray):
        return self._shape_hint(cluster)

    def run(self, points: np.ndarray,
            rgb:        Optional[np.ndarray] = None,
            plane_hint: Optional[tuple]      = None) -> PerceptionResult:
        """
        points     : (N, 3) XYZ point cloud
        rgb        : (N, 3) float32/uint8 colours, same row order as points.
        plane_hint : (normal, height) tuple from a previous run on the same
                     sequence. If provided, skips RANSAC (saves ~200ms/frame).
        """
        result = PerceptionResult(n_points_input=len(points))

        # 0 — sanitise
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

        # 1 — downsample
        pcd = _to_o3d(pts)
        if col is not None:
            pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64))
        pcd = pcd.voxel_down_sample(self.voxel_size)

        # 2 — denoise
        pcd, _ = pcd.remove_statistical_outlier(self.nb_neighbors, self.std_ratio)
        pts = _clean(np.asarray(pcd.points))
        col_ds = np.asarray(pcd.colors).astype(np.float32) if (col is not None and pcd.has_colors()) else None
        if len(pts) < 50:
            return result
        result.n_points_after_denoise = len(pts)
        pcd = _to_o3d(pts)

        # 3 — table plane
        # If plane_hint provided, skip RANSAC and use cached normal/height.
        # Inliers are always recomputed from the (cached or fitted) plane
        # so that table_pts is correct for this frame's point cloud.
        if plane_hint is not None:
            n_unit, table_height = plane_hint
            n_unit       = np.array(n_unit, dtype=np.float64)
            table_height = float(table_height)
        else:
            try:
                plane_model, inlier_idx = pcd.segment_plane(
                    distance_threshold=self.plane_dist_threshold,
                    ransac_n=self.plane_ransac_n,
                    num_iterations=self.plane_num_iterations,
                )
            except Exception as e:
                print(f"  [pipeline] RANSAC failed: {e}")
                return result

            a, b, c, d = [float(x) for x in plane_model]
            n_raw    = np.array([a, b, c], dtype=np.float64)
            norm_len = float(np.linalg.norm(n_raw))
            if norm_len < 1e-9:
                return result

            n_unit       = n_raw / norm_len
            table_height = float(-d / norm_len)
            if table_height > 0:
                n_unit       = -n_unit
                table_height = -table_height

        # recompute inliers for this frame from the plane
        signed_dists = pts @ n_unit - table_height
        inlier_idx   = np.where(np.abs(signed_dists) < self.plane_dist_threshold * 2)[0]

        result.table_normal   = n_unit
        result.table_height   = table_height
        result.n_points_table = len(inlier_idx)

        # 4 — depth gate (removes back-wall artifacts in angled views)
        # inlier_idx is always an integer numpy array from np.where()
        table_mask  = np.zeros(len(pts), dtype=bool)
        if len(inlier_idx) > 0:
            table_mask[inlier_idx] = True
        table_pts   = pts[table_mask]
        table_z_max = float(table_pts[:, 2].max()) if len(table_pts) else 0.0
        obj_mask    = ~table_mask
        obj_pts     = _clean(pts[obj_mask])
        obj_col     = col_ds[obj_mask] if col_ds is not None else None

        z_mask  = obj_pts[:, 2] < (table_z_max + self.depth_margin)
        obj_pts = obj_pts[z_mask]
        if obj_col is not None:
            obj_col = obj_col[z_mask]

        if len(obj_pts) == 0:
            return result

        # 5 — height filter above table plane
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            heights = obj_pts @ n_unit - table_height
        finite = np.isfinite(heights)
        h_mask = (finite &
                  (heights > self.min_height_above_table) &
                  (heights < self.max_height_above_table))
        obj_pts = obj_pts[h_mask]
        if obj_col is not None:
            obj_col = obj_col[h_mask]

        if len(obj_pts) < self.cluster_min_points:
            return result

        # 5b — XY radius crop
        # Use camera projection origin [0,0] not table centroid —
        # table centroid can be far from objects near scene edges.
        # OCID cameras point along +Z so XY=[0,0] is always in-view.
        table_centre_xy = np.array([0.0, 0.0])
        xy_dist = np.linalg.norm(obj_pts[:, :2] - table_centre_xy, axis=1)
        r_mask  = xy_dist < self.xy_radius
        obj_pts = obj_pts[r_mask]
        if obj_col is not None:
            obj_col = obj_col[r_mask]

        if len(obj_pts) < self.cluster_min_points:
            return result

        # 6 — cluster
        if obj_col is not None and self.rgb_weight > 0:
            from sklearn.cluster import DBSCAN as skDBSCAN
            feats = np.hstack([obj_pts, obj_col * self.rgb_weight]).astype(np.float32)
            db    = skDBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_points,
                             algorithm='ball_tree', n_jobs=1).fit(feats)
            labels = db.labels_
        else:
            pcd_obj = _to_o3d(obj_pts)
            labels  = np.array(pcd_obj.cluster_dbscan(
                eps=self.cluster_eps,
                min_points=self.cluster_min_points,
                print_progress=False,
            ))

        # 6b — split passes
        # Pass 1: horizontal saddle
        raw_clusters = []
        for lbl in sorted(set(labels) - {-1}):
            cluster = obj_pts[labels == lbl]
            if len(cluster) >= self.split_min_points:
                raw_clusters.extend(_split_cluster(
                    cluster, min_pts=self.cluster_min_points,
                    saddle_depth=self.saddle_depth, n_bins=self.split_n_bins))
            else:
                raw_clusters.append(cluster)

        # Pass 2: vertical saddle
        vertically_split = []
        for cluster in raw_clusters:
            if len(cluster) >= self.split_min_points:
                vertically_split.extend(_split_cluster_vertical(
                    cluster, min_pts=self.cluster_min_points,
                    saddle_depth=self.saddle_depth))
            else:
                vertically_split.append(cluster)

        # Pass 3A: normal discontinuity (disabled on macOS)
        after_normal = vertically_split

        # Pass 3B: concavity
        after_concavity = []
        for cluster in after_normal:
            if len(cluster) >= self.geo_min_points:
                after_concavity.extend(_split_by_concavity(
                    cluster, min_pts=self.cluster_min_points,
                    concavity_threshold=self.concavity_threshold))
            else:
                after_concavity.append(cluster)

        # Pass 3C: height layers
        after_height = []
        for cluster in after_concavity:
            after_height.extend(_split_by_height_layers(
                cluster, min_pts=self.cluster_min_points,
                gap_threshold=self.height_gap_threshold))

        # 7 — extract valid segments, classify shape
        obj_id = 0
        for cluster in after_height:
            if not self._valid(cluster):
                continue
            shape_type, shape_conf = self._classify(cluster)
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
        pcd = o3d.io.read_point_cloud(path)
        return self.run(np.asarray(pcd.points))

    def _valid(self, pts: np.ndarray) -> bool:
        if not (self.min_object_points <= len(pts) <= self.max_object_points):
            return False
        extent = pts.max(axis=0) - pts.min(axis=0)
        return self.min_object_extent <= extent.max() <= self.max_object_extent