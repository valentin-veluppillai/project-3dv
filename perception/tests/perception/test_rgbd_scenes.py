"""
tests/perception/test_rgbd_scenes.py

Tests for the RGBDScenesV2 loader.
All tests are marked @pytest.mark.requires_rgbd_data and are skipped
automatically when the data is not present.

Data path expected: <project>/data/rgbd-scenes-v2/pc/
"""

import sys
import os
from pathlib import Path

import numpy as np
import pytest

_PERCEPTION_DIR = Path(__file__).resolve().parents[2]   # curobo-sq/perception/
sys.path.insert(0, str(_PERCEPTION_DIR))

from datasets.rgbd_scenes import RGBDScenesV2

# data/ was not moved — it still lives in project-3dv.
_DATA_ROOT = Path("/work/courses/3dv/team15/project-3dv/data/rgbd-scenes-v2/pc")

pytestmark = pytest.mark.requires_rgbd_data


# ---------------------------------------------------------------------------
# test_loader_returns_correct_shapes
# ---------------------------------------------------------------------------

def test_loader_returns_correct_shapes():
    """Scene 1: pts is (N,3), rgb is (N,3), labels is (N,) with the same N."""
    scene = RGBDScenesV2(data_root=str(_DATA_ROOT), scene_id=1)
    pts, rgb, labels = scene.load()

    assert pts.ndim == 2 and pts.shape[1] == 3, (
        f"pts should be (N, 3), got {pts.shape}"
    )
    assert pts.dtype == np.float32, f"pts dtype should be float32, got {pts.dtype}"

    assert rgb.ndim == 2 and rgb.shape[1] == 3, (
        f"rgb should be (N, 3), got {rgb.shape}"
    )
    assert rgb.dtype == np.uint8, f"rgb dtype should be uint8, got {rgb.dtype}"

    assert labels.ndim == 1, f"labels should be 1-D, got {labels.ndim}-D"
    assert labels.shape[0] == pts.shape[0], (
        f"labels length {labels.shape[0]} != pts length {pts.shape[0]}"
    )


# ---------------------------------------------------------------------------
# test_get_object_clouds_excludes_background
# ---------------------------------------------------------------------------

def test_get_object_clouds_excludes_background():
    """get_object_clouds() must never include label 0 (background) as a key."""
    scene = RGBDScenesV2(data_root=str(_DATA_ROOT), scene_id=1)
    clouds = scene.get_object_clouds()

    assert 0 not in clouds, (
        "label 0 (background) should be excluded from get_object_clouds()"
    )
    # Every returned key must be a positive integer
    for lbl, pts in clouds.items():
        assert lbl > 0, f"Expected label > 0, got {lbl}"
        assert pts.ndim == 2 and pts.shape[1] == 3, (
            f"Cloud for label {lbl} should be (M, 3), got {pts.shape}"
        )


# ---------------------------------------------------------------------------
# test_scene_has_multiple_objects
# ---------------------------------------------------------------------------

def test_scene_has_multiple_objects():
    """Scene 1 must contain at least 2 distinct non-zero instance labels."""
    scene = RGBDScenesV2(data_root=str(_DATA_ROOT), scene_id=1)
    _, _, labels = scene.load()

    unique_obj = np.unique(labels[labels > 0])
    assert len(unique_obj) >= 2, (
        f"Expected at least 2 distinct non-zero labels in scene 1, "
        f"got {len(unique_obj)}: {unique_obj}"
    )
