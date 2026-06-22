"""
tests/perception/test_sq_improvements.py

Unit tests for:
  - _filter_degenerate_primitives()  (superdec_fitter.py)
  - merge_overlapping_primitives()   (superdec_fitter.py)
All tests use synthetic SuperquadricFit objects — no external data required.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest

from superdec_fitter import (
    _filter_degenerate_primitives,
    merge_overlapping_primitives,
    _aabb_iou,
    _sq_aabb,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prim(sx=0.1, sy=0.1, sz=0.1, tx=0.0, ty=0.0, tz=0.0):
    """Create a minimal SuperquadricFit-like object."""
    from superquadric import SuperquadricFit
    return SuperquadricFit(
        sx=sx, sy=sy, sz=sz,
        e1=1.0, e2=1.0,
        tx=tx, ty=ty, tz=tz,
        rx=0.0, ry=0.0, rz=0.0,
    )


def _make_multi(prims, n_points=512):
    from superquadric import MultiSQFit
    return MultiSQFit(primitives=list(prims), n_points=n_points)


# ---------------------------------------------------------------------------
# _filter_degenerate_primitives
# ---------------------------------------------------------------------------

def test_filter_degenerate_removes_small_prims():
    """Primitives with min(sx,sy,sz) < min_scale must be removed."""
    good = _make_prim(sx=0.1, sy=0.1, sz=0.1)
    bad  = _make_prim(sx=0.001, sy=0.1, sz=0.1)   # sx < 0.005
    result = _filter_degenerate_primitives([good, bad], min_scale=0.005)
    assert len(result) == 1
    assert result[0] is good


def test_filter_degenerate_keeps_all_valid():
    """All primitives at or above min_scale must be kept."""
    prims = [_make_prim(sx=0.01, sy=0.1, sz=0.1),
             _make_prim(sx=0.1,  sy=0.1, sz=0.1)]
    result = _filter_degenerate_primitives(prims, min_scale=0.005)
    assert len(result) == 2


def test_filter_degenerate_all_bad_returns_largest_fallback():
    """When all primitives are degenerate the fallback returns the largest one."""
    small = _make_prim(sx=0.001, sy=0.001, sz=0.001)
    large = _make_prim(sx=0.002, sy=0.002, sz=0.002)
    result = _filter_degenerate_primitives([small, large], min_scale=0.005)
    assert len(result) == 1
    assert result[0] is large


def test_filter_degenerate_does_not_modify_input():
    """Input list must not be mutated."""
    prims = [_make_prim(sx=0.001), _make_prim(sx=0.1)]
    original_len = len(prims)
    _filter_degenerate_primitives(prims, min_scale=0.005)
    assert len(prims) == original_len


# ---------------------------------------------------------------------------
# _aabb_iou
# ---------------------------------------------------------------------------

def test_aabb_iou_identical_boxes():
    """Identical boxes have IoU = 1.0."""
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([1.0, 1.0, 1.0])
    assert abs(_aabb_iou(lo, hi, lo, hi) - 1.0) < 1e-6


def test_aabb_iou_disjoint_boxes():
    """Non-overlapping boxes have IoU = 0.0."""
    lo_a, hi_a = np.array([0., 0., 0.]), np.array([1., 1., 1.])
    lo_b, hi_b = np.array([2., 2., 2.]), np.array([3., 3., 3.])
    assert _aabb_iou(lo_a, hi_a, lo_b, hi_b) == 0.0


def test_aabb_iou_partial_overlap():
    """Half-overlap along one axis gives IoU in (0, 1)."""
    lo_a, hi_a = np.array([0., 0., 0.]), np.array([2., 1., 1.])
    lo_b, hi_b = np.array([1., 0., 0.]), np.array([3., 1., 1.])
    iou = _aabb_iou(lo_a, hi_a, lo_b, hi_b)
    assert 0.0 < iou < 1.0
    # intersection = 1×1×1 = 1, union = 2+2-1 = 3 → iou = 1/3
    assert abs(iou - 1.0 / 3.0) < 1e-6


# ---------------------------------------------------------------------------
# merge_overlapping_primitives
# ---------------------------------------------------------------------------

def test_merge_removes_duplicate_prims():
    """Two nearly-identical primitives at the same location must be merged to 1."""
    p1 = _make_prim(sx=0.1, sy=0.1, sz=0.1, tx=0.0, ty=0.0, tz=0.0)
    p2 = _make_prim(sx=0.09, sy=0.09, sz=0.09, tx=0.0, ty=0.0, tz=0.0)
    multi = _make_multi([p1, p2])
    result = merge_overlapping_primitives([multi], iou_threshold=0.3)
    assert len(result) == 1
    assert len(result[0].primitives) == 1


def test_merge_keeps_non_overlapping_prims():
    """Two well-separated primitives must both be kept."""
    p1 = _make_prim(sx=0.05, sy=0.05, sz=0.05, tx=0.0,  ty=0.0, tz=0.0)
    p2 = _make_prim(sx=0.05, sy=0.05, sz=0.05, tx=1.0,  ty=0.0, tz=0.0)
    multi = _make_multi([p1, p2])
    result = merge_overlapping_primitives([multi], iou_threshold=0.3)
    assert len(result) == 1
    assert len(result[0].primitives) == 2


def test_merge_keeps_larger_on_overlap():
    """When two similarly-sized primitives overlap, the larger one must be kept.

    Both boxes are centred at the origin; big (sx=0.15) and smaller (sx=0.12).
    Their AABBs overlap with IoU ≈ 0.51 (> 0.3 threshold), so the smaller is
    dropped and only the bigger is retained.
    """
    big   = _make_prim(sx=0.15, sy=0.15, sz=0.15, tx=0.0, ty=0.0, tz=0.0)
    small = _make_prim(sx=0.12, sy=0.12, sz=0.12, tx=0.0, ty=0.0, tz=0.0)
    # Pass small first so merge must reorder by volume
    multi = _make_multi([small, big])
    result = merge_overlapping_primitives([multi], iou_threshold=0.3)
    assert len(result[0].primitives) == 1
    kept = result[0].primitives[0]
    assert kept.sx == pytest.approx(0.15)


def test_merge_single_prim_unchanged():
    """A MultiSQFit with a single primitive must be returned unchanged."""
    p = _make_prim(sx=0.1, sy=0.1, sz=0.1)
    multi = _make_multi([p])
    result = merge_overlapping_primitives([multi], iou_threshold=0.3)
    assert len(result) == 1
    assert len(result[0].primitives) == 1


def test_merge_preserves_n_points():
    """n_points field on the MultiSQFit must be preserved after merge."""
    p1 = _make_prim(sx=0.1, ty=0.0)
    p2 = _make_prim(sx=0.1, ty=5.0)
    multi = _make_multi([p1, p2], n_points=1024)
    result = merge_overlapping_primitives([multi], iou_threshold=0.3)
    assert result[0].n_points == 1024


def test_merge_multiple_multisq():
    """merge_overlapping_primitives handles a list of multiple MultiSQFit."""
    m1 = _make_multi([_make_prim(tx=0.0), _make_prim(tx=0.0)])  # overlap
    m2 = _make_multi([_make_prim(tx=0.0), _make_prim(tx=10.0)])  # separated
    result = merge_overlapping_primitives([m1, m2], iou_threshold=0.3)
    assert len(result) == 2
    assert len(result[0].primitives) == 1   # merged
    assert len(result[1].primitives) == 2   # both kept


# ---------------------------------------------------------------------------
# distance-weighted merge
# ---------------------------------------------------------------------------

def test_distance_weighted_merge_conservative_for_far_segments():
    """Far-away segments (low weight) must NOT merge even when IoU > base threshold.

    Two MultiSQFit objects, each with one primitive at the same location
    (IoU ≈ 0.51, well above iou_threshold=0.3).  With distance_weights=[0.2, 0.2]
    the effective threshold = 0.3 / (0.2 * 0.2) = 7.5, which no real IoU can
    exceed, so the primitives must remain in their respective fits.
    """
    # Two identical primitives at origin — IoU = 1.0 > 0.3 base threshold.
    # Use weight 0.2 for both (far from camera) → eff_thr = 0.3/0.04 = 7.5.
    p1 = _make_prim(sx=0.15, sy=0.15, sz=0.15, tx=0.0, ty=0.0, tz=0.0)
    p2 = _make_prim(sx=0.12, sy=0.12, sz=0.12, tx=0.0, ty=0.0, tz=0.0)
    m1 = _make_multi([p1])
    m2 = _make_multi([p2])
    result = merge_overlapping_primitives(
        [m1, m2], iou_threshold=0.3, distance_weights=[0.2, 0.2]
    )
    # Both primitives survive — far segments resist merging.
    total_prims = sum(len(r.primitives) for r in result)
    assert total_prims == 2, (
        f"Expected 2 primitives (far segments resist merge), got {total_prims}"
    )


def test_distance_weighted_merge_aggressive_for_near_segments():
    """Near segments (high weight) merge readily when IoU exceeds effective threshold.

    Two MultiSQFit objects with overlapping primitives (IoU ≈ 0.51).
    With distance_weights=[0.95, 0.95] the effective threshold =
    0.3 / (0.95 * 0.95) ≈ 0.332.  IoU ≈ 0.51 > 0.332, so the smaller
    primitive must be discarded (merged into the larger one).
    """
    big   = _make_prim(sx=0.15, sy=0.15, sz=0.15, tx=0.0, ty=0.0, tz=0.0)
    small = _make_prim(sx=0.12, sy=0.12, sz=0.12, tx=0.0, ty=0.0, tz=0.0)
    m1 = _make_multi([big])
    m2 = _make_multi([small])
    result = merge_overlapping_primitives(
        [m1, m2], iou_threshold=0.3, distance_weights=[0.95, 0.95]
    )
    # Only the larger primitive survives.
    total_prims = sum(len(r.primitives) for r in result)
    assert total_prims == 1, (
        f"Expected 1 primitive (near segments merge aggressively), got {total_prims}"
    )
    # The kept primitive must be the larger one (sx=0.15).
    kept = [p for r in result for p in r.primitives]
    assert kept[0].sx == pytest.approx(0.15), (
        f"Kept primitive sx={kept[0].sx:.3f}, expected 0.15 (the larger one)"
    )
