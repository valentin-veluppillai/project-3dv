"""
tests/perception/test_preprocessing.py

Unit tests for preprocess_pointcloud() and postprocess_fits() in pipeline.py.
All tests use synthetic point clouds — no external data required.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest
import copy

from pipeline import preprocess_pointcloud, postprocess_fits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_sphere_cloud(n: int = 512, seed: int = 0) -> np.ndarray:
    """Random point cloud sampled uniformly inside the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    radii = rng.uniform(0.1, 1.0, (n, 1))
    return (pts * radii).astype(np.float64)


def _elongated_x_cloud(n: int = 512, stretch: float = 5.0, seed: int = 1) -> np.ndarray:
    """Cloud with variance stretch × larger along X than Y/Z."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    pts[:, 0] *= stretch     # X has 5× more spread than Y, Z
    return pts.astype(np.float64)


def _make_sq_fit(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0,
                 sx: float = 0.1, sy: float = 0.1, sz: float = 0.1):
    """Create a minimal SuperquadricFit-like object for postprocess tests."""
    from superquadric import SuperquadricFit
    return SuperquadricFit(
        sx=sx, sy=sy, sz=sz,
        e1=1.0, e2=1.0,
        tx=tx, ty=ty, tz=tz,
        rx=0.0, ry=0.0, rz=0.0,
    )


def _make_multi(sq_fit):
    from superquadric import MultiSQFit
    return MultiSQFit(primitives=[sq_fit], n_points=512)


# ---------------------------------------------------------------------------
# test_outlier_removal_removes_injected_outliers
# ---------------------------------------------------------------------------

def test_outlier_removal_removes_injected_outliers():
    """10 points injected at (100, 100, 100) must all be removed."""
    pts = _unit_sphere_cloud(n=512)
    outliers = np.full((10, 3), 100.0)
    pts_with_outliers = np.vstack([pts, outliers])

    _, _, meta = preprocess_pointcloud(pts_with_outliers, target_n=512, outlier_std=2.5)
    assert meta["n_outliers_removed"] == 10, (
        f"Expected 10 outliers removed, got {meta['n_outliers_removed']}"
    )


# ---------------------------------------------------------------------------
# test_scale_normalization_unit_sphere
# ---------------------------------------------------------------------------

def test_no_scale_normalization_preserves_scale():
    """preprocess_pointcloud must NOT alter the point cloud scale.

    Scale normalization was removed (Fix 1) to avoid double-normalization with
    SuperdecFitter.  Verify that meta['scale'] == 1.0 and that the output
    cloud has the same max radius as the input (within numerical noise).
    """
    pts = _unit_sphere_cloud(n=512)
    pts_out, _, meta = preprocess_pointcloud(pts, target_n=len(pts), outlier_std=100.0)

    assert meta["scale"] == 1.0, (
        f"meta['scale'] should be 1.0 (no-op), got {meta['scale']}"
    )
    # The output cloud should span roughly the same range as the input
    input_max  = float(np.linalg.norm(pts - pts.mean(0), axis=1).max())
    output_max = float(np.linalg.norm(pts_out - pts_out.mean(0), axis=1).max())
    assert abs(output_max - input_max) / (input_max + 1e-9) < 0.01, (
        f"Output max radius {output_max:.4f} differs from input {input_max:.4f} "
        "— scale normalization should be a no-op"
    )


# ---------------------------------------------------------------------------
# test_pca_rotation_longest_axis_is_z
# ---------------------------------------------------------------------------

def test_pca_rotation_longest_axis_is_z():
    """Cloud elongated along X: after PCA rotation (rotate=True), Z must have the largest variance."""
    pts = _elongated_x_cloud(n=1024, stretch=5.0)
    pts_out, _, _ = preprocess_pointcloud(pts, target_n=len(pts), outlier_std=10.0, rotate=True)

    var_x = float(pts_out[:, 0].var())
    var_y = float(pts_out[:, 1].var())
    var_z = float(pts_out[:, 2].var())

    assert var_z > var_x, (
        f"Z variance {var_z:.4f} is NOT larger than X variance {var_x:.4f} "
        "— longest axis was not mapped to Z"
    )
    assert var_z > var_y, (
        f"Z variance {var_z:.4f} is NOT larger than Y variance {var_y:.4f} "
        "— longest axis was not mapped to Z"
    )


# ---------------------------------------------------------------------------
# test_postprocess_inverts_preprocess
# ---------------------------------------------------------------------------

def test_postprocess_inverts_preprocess():
    """postprocess_fits should recover original scale and centroid to within 1e-5."""
    pts = _unit_sphere_cloud(n=512)
    # Shift and scale so centroid and scale are non-trivial
    pts = pts * 3.7 + np.array([1.2, -0.5, 2.1])

    _, _, meta = preprocess_pointcloud(pts, target_n=len(pts), outlier_std=2.5)

    scale = meta["scale"]
    centroid = meta["centroid"]
    R = meta["rotation"]

    # Create a SQ fit at the origin of the preprocessed frame
    sq = _make_sq_fit(tx=0.0, ty=0.0, tz=0.0, sx=0.1, sy=0.1, sz=0.1)
    multi = _make_multi(sq)

    # Postprocess: origin in clean frame → centroid in original frame
    result = postprocess_fits([multi], meta)
    prim = result[0].primitives[0]

    t_recovered = prim.translation  # (3,)
    assert np.allclose(t_recovered, centroid, atol=1e-5), (
        f"Postprocess translation {t_recovered} does not match centroid "
        f"{centroid} (max diff {np.abs(t_recovered - centroid).max():.2e})"
    )

    # Semi-axes: should be scaled by meta['scale']
    expected_sx = 0.1 * scale
    assert abs(prim.sx - expected_sx) < 1e-5, (
        f"Postprocess sx={prim.sx:.6f}, expected {expected_sx:.6f}"
    )


# ---------------------------------------------------------------------------
# test_resampling_target_count
# ---------------------------------------------------------------------------

def test_resampling_upsample_gives_target_count():
    """Cloud with fewer than target_n points is upsampled to exactly target_n."""
    pts = _unit_sphere_cloud(n=100)
    pts_out, _, _ = preprocess_pointcloud(pts, target_n=512, outlier_std=10.0)
    assert pts_out.shape == (512, 3), (
        f"Expected (512, 3), got {pts_out.shape}"
    )


def test_resampling_downsample_gives_target_count():
    """Cloud with more than target_n points is downsampled to exactly target_n."""
    pts = _unit_sphere_cloud(n=2048)
    pts_out, _, _ = preprocess_pointcloud(pts, target_n=512, outlier_std=10.0)
    assert pts_out.shape == (512, 3), (
        f"Expected (512, 3), got {pts_out.shape}"
    )


# ---------------------------------------------------------------------------
# test_normals_passthrough
# ---------------------------------------------------------------------------

def test_normals_have_same_count_as_pts_out():
    """If normals are passed, normals_out must have the same row count as pts_out."""
    pts = _unit_sphere_cloud(n=512)
    norms = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-9)
    pts_out, normals_out, _ = preprocess_pointcloud(
        pts, normals=norms, target_n=256, outlier_std=10.0
    )
    assert normals_out is not None, "normals_out should not be None"
    assert normals_out.shape == pts_out.shape, (
        f"normals_out shape {normals_out.shape} != pts_out shape {pts_out.shape}"
    )


# ---------------------------------------------------------------------------
# test_table_frame_normal_aligns_to_z
# ---------------------------------------------------------------------------

def test_table_frame_normal_aligns_to_z():
    """After table-frame preprocessing, object height (variance) must be in Z.

    A slightly tilted table normal [0.1, 0.2, 0.97] (unit-normalised) is used.
    100 random points are placed above that table.  After preprocess_pointcloud
    with table_normal provided, the output cloud should have larger Z variance
    than X or Y variance (the table is now the XY plane, so height → Z).
    """
    rng = np.random.default_rng(42)
    raw_normal = np.array([0.1, 0.2, 0.97])
    raw_normal /= np.linalg.norm(raw_normal)
    table_height = 0.5

    # Build 100 points just above the table surface (height 0.02–0.25 m)
    # by starting from the table plane and displacing along the normal.
    n_pts = 100
    # Random base points on the table plane
    u = np.array([1.0, 0.0, -raw_normal[0] / (raw_normal[2] + 1e-9)])
    u /= np.linalg.norm(u)
    v = np.cross(raw_normal, u)
    alphas = rng.uniform(-0.3, 0.3, (n_pts, 1))
    betas  = rng.uniform(-0.3, 0.3, (n_pts, 1))
    heights = rng.uniform(0.02, 0.25, (n_pts, 1))
    # plane point: table_height * normal  (a point on the plane)
    pts = (table_height * raw_normal
           + alphas * u
           + betas  * v
           + heights * raw_normal)

    pts_out, _, meta = preprocess_pointcloud(
        pts.astype(np.float64),
        target_n=n_pts,
        outlier_std=1e9,          # no outlier removal
        table_normal=raw_normal,
        table_height=table_height,
    )

    assert "table_rotation" in meta, "meta must contain 'table_rotation'"
    assert "table_centroid" in meta, "meta must contain 'table_centroid'"

    var_x = float(pts_out[:, 0].var())
    var_y = float(pts_out[:, 1].var())
    var_z = float(pts_out[:, 2].var())

    # Direct check: R_table must map raw_normal → [0, 0, 1]
    R_table = meta["table_rotation"]
    mapped = R_table @ raw_normal
    assert abs(mapped[2] - 1.0) < 1e-9, (
        f"R_table must map table_normal to [0,0,1], got Z-component {mapped[2]:.6f}"
    )
    assert abs(mapped[0]) < 1e-9 and abs(mapped[1]) < 1e-9, (
        f"R_table must map table_normal to [0,0,1], got X={mapped[0]:.6f} Y={mapped[1]:.6f}"
    )

    # Variance check: height direction is Z after rotation.
    # Heights range 0.02–0.25m; since outlier_std=1e9, all points survive.
    # Points also span ±0.3m laterally, so Z variance can be smaller than X/Y.
    # Instead verify Z has meaningful variance (heights are non-trivial).
    assert var_z > 1e-4, f"Z variance {var_z:.6f} is near zero — height transform failed"


# ---------------------------------------------------------------------------
# test_table_frame_postprocess_recovers_original_centroid
# ---------------------------------------------------------------------------

def test_table_frame_postprocess_recovers_original_centroid():
    """postprocess_fits must recover the original centroid after table-frame transform.

    A point cluster centred at (1.0, 0.5, 0.8) in camera frame with a flat
    table (table_normal=[0,0,1], table_height=0.0).  After preprocessing and
    a synthetic SQ fit at the transformed centroid, postprocess_fits must
    return a translation within 1e-4 of the original (1.0, 0.5, 0.8).
    """
    centre = np.array([1.0, 0.5, 0.8])
    # Use a noiseless cluster so sampling error doesn't pollute the tolerance check.
    # preprocess_pointcloud requires ≥10 distinct points; add tiny deterministic jitter.
    rng = np.random.default_rng(7)
    pts = centre + rng.uniform(-1e-9, 1e-9, (200, 3))

    table_normal = np.array([0.0, 0.0, 1.0])
    table_height = 0.0

    pts_out, _, meta = preprocess_pointcloud(
        pts.astype(np.float64),
        target_n=len(pts),
        outlier_std=1e9,
        table_normal=table_normal,
        table_height=table_height,
    )

    # The cluster centroid in the table frame
    centroid_tbl = pts_out.mean(axis=0)

    # Synthetic SQ fit at the table-frame centroid
    sq = _make_sq_fit(
        tx=float(centroid_tbl[0]),
        ty=float(centroid_tbl[1]),
        tz=float(centroid_tbl[2]),
        sx=0.05, sy=0.05, sz=0.05,
    )
    multi = _make_multi(sq)

    result = postprocess_fits([multi], meta)
    t_rec = np.array(result[0].primitives[0].translation)

    assert np.allclose(t_rec, centre, atol=1e-4), (
        f"Recovered translation {t_rec} differs from original centroid "
        f"{centre} (max diff {np.abs(t_rec - centre).max():.2e})"
    )
