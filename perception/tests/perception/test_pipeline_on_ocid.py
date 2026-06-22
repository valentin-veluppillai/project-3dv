"""
tests/perception/test_pipeline_on_ocid.py

Integration tests that run the full tabletop perception pipeline on a real
OCID RGB-D frame and verify the output against ground-truth instance masks.

Requirements
------------
* OCID dataset downloaded to data/ocid/
  (run: bash scripts/download_ocid.sh)
* All three tests are marked @pytest.mark.requires_data and are skipped
  automatically when the dataset is absent.

Skip dataset tests only:
    pytest tests/ -m "not requires_data"

Run dataset tests only:
    pytest tests/ -m requires_data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest

from pipeline import remove_table, segment_instances, fit_superquadrics, TabletopPerception
from superquadric import MultiSQFit


pytestmark = pytest.mark.requires_data


# ---------------------------------------------------------------------------
# test_table_removed
# ---------------------------------------------------------------------------

@pytest.mark.requires_data
def test_table_removed(ocid_sample):
    """remove_table() must return fewer points than the raw scene cloud.

    The raw cloud contains the table plane, background, and foreground objects.
    After remove_table() only points physically above the table remain, so the
    output must be strictly smaller than the input.
    """
    pts = ocid_sample.get_points()
    assert pts is not None and len(pts) > 0, "get_points() returned empty cloud"

    obj_pts, *_ = remove_table(pts)

    assert len(obj_pts) < len(pts), (
        f"remove_table() did not reduce point count: "
        f"output {len(obj_pts)} >= input {len(pts)}"
    )
    assert len(obj_pts) > 0, (
        "remove_table() removed all points — check table height / camera pose"
    )


# ---------------------------------------------------------------------------
# test_instances_match_gt
# ---------------------------------------------------------------------------

@pytest.mark.requires_data
def test_instances_match_gt(ocid_sample):
    """Detected instance count must be within ±2 of the GT object count.

    A ±2 tolerance accommodates:
      * One object obscured / too few points to form a cluster.
      * Two touching objects that DBSCAN merges into one.
    Larger discrepancies indicate a pipeline regression.
    """
    pts = ocid_sample.get_points()
    assert pts is not None and len(pts) > 0, "get_points() returned empty cloud"

    obj_pts, *_ = remove_table(pts)

    n_gt = ocid_sample.n_gt_objects()
    assert n_gt > 0, "Label image has no foreground objects — check OCID frame"

    segments = segment_instances(obj_pts)
    n_detected = len(segments)

    assert abs(n_detected - n_gt) <= 2, (
        f"Instance count mismatch: detected {n_detected}, GT {n_gt} "
        f"(difference {abs(n_detected - n_gt)} > tolerance 2)"
    )


# ---------------------------------------------------------------------------
# test_sq_fits_are_valid
# ---------------------------------------------------------------------------

@pytest.mark.requires_data
def test_sq_fits_are_valid(ocid_sample):
    """Every MultiSQFit produced by fit_superquadrics() must satisfy:

      * translation  — all three components are finite (no NaN / inf)
      * rotation     — all 9 components are finite
      * scales       — all three semi-axes > 0
      * exponents    — both e1, e2 in the convex SQ regime (0, 2]
                       (clamping is applied by SuperdecFitter / LM fitter)
    """
    pts = ocid_sample.get_points()
    assert pts is not None and len(pts) > 0, "get_points() returned empty cloud"

    pipe = TabletopPerception(
        voxel_size=0.005,
        cluster_eps=0.030,
        cluster_min_points=30,
        min_object_points=40,
        plane_dist_threshold=0.015,
    )
    result = pipe.run(pts)
    assert result is not None, "TabletopPerception.run() returned None"
    assert len(result.objects) > 0, (
        "Pipeline found no objects — check dataset frame / pipeline parameters"
    )

    sq_fits = fit_superquadrics(result.objects)
    assert len(sq_fits) > 0, "fit_superquadrics() returned no fits"

    for i, multi in enumerate(sq_fits):
        assert isinstance(multi, MultiSQFit), (
            f"fit {i} is not a MultiSQFit: {type(multi)}"
        )
        assert len(multi.primitives) > 0, f"MultiSQFit {i} has no primitives"

        for j, prim in enumerate(multi.primitives):
            label = f"object {i} primitive {j}"

            # Pose — translation
            assert np.all(np.isfinite(prim.translation)), (
                f"{label}: non-finite translation {prim.translation}"
            )
            # Pose — rotation
            assert np.all(np.isfinite(prim.rotation)), (
                f"{label}: non-finite rotation matrix"
            )
            # Scale > 0
            assert np.all(prim.scales > 0), (
                f"{label}: non-positive scale {prim.scales}"
            )
            # Shape exponents in convex SQ regime (0, 2]
            e1, e2 = float(prim.exponents[0]), float(prim.exponents[1])
            assert 0 < e1 <= 2.0, (
                f"{label}: e1={e1:.4f} not in (0, 2]"
            )
            assert 0 < e2 <= 2.0, (
                f"{label}: e2={e2:.4f} not in (0, 2]"
            )
