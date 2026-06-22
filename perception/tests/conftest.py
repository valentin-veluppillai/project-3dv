"""
tests/conftest.py — shared pytest fixtures and dataset availability guards.

Copied from project-3dv/tests/conftest.py. The perception code now lives
in curobo-sq/perception/ (this repo); the OCID/RGB-D Scenes datasets were
not moved and still live under project-3dv/data/.

OCID dataset tests
------------------
Tests marked @pytest.mark.requires_data need the OCID dataset on disk.

  * If data/ocid/ is absent  → those tests are automatically skipped with a
    clear message pointing to the download script.
  * If data/ocid/ is present → the `ocid_sample` fixture loads one ARID20
    frame (RGB-D + ground-truth label) via OCIDLoader.

Download the dataset (from project-3dv):
    bash scripts/download_ocid.sh
"""

import sys
import os
from pathlib import Path
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_TESTS_DIR      = Path(__file__).parent
_PERCEPTION_DIR = _TESTS_DIR.parent     # curobo-sq/perception/

# Add perception/ to sys.path so that ocid_loader's own
# `from pipeline import ...` resolves correctly (same pattern as other tests).
sys.path.insert(0, str(_PERCEPTION_DIR))
sys.path.insert(0, str(_PERCEPTION_DIR / "evaluation"))

# data/ was not moved — it still lives in project-3dv.
_PROJECT_3DV_DATA = Path("/work/courses/3dv/team15/project-3dv/data")
_OCID_DIR = _PROJECT_3DV_DATA / "ocid"
_RGBD_DIR = _PROJECT_3DV_DATA / "rgbd-scenes-v2" / "pc"


def _ocid_available() -> bool:
    """Return True if data/ocid/ exists and contains at least one entry."""
    return _OCID_DIR.is_dir() and any(_OCID_DIR.iterdir())


def _rgbd_available() -> bool:
    """Return True if the RGB-D Scenes v2 data is present (scene 01 as proxy)."""
    return (_RGBD_DIR / "01.ply").exists()


# ---------------------------------------------------------------------------
# Auto-skip requires_data / requires_rgbd_data tests when dataset is absent
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    ocid_ok = _ocid_available()
    rgbd_ok = _rgbd_available()
    skip_ocid = pytest.mark.skip(
        reason="OCID data not found — run scripts/download_ocid.sh (in project-3dv) to enable dataset tests"
    )
    skip_rgbd = pytest.mark.skip(
        reason="RGB-D Scenes v2 data not found at project-3dv/data/rgbd-scenes-v2/pc/"
    )
    for item in items:
        if not ocid_ok and item.get_closest_marker("requires_data"):
            item.add_marker(skip_ocid)
        if not rgbd_ok and item.get_closest_marker("requires_rgbd_data"):
            item.add_marker(skip_rgbd)


# ---------------------------------------------------------------------------
# ocid_sample fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ocid_sample():
    """Load one RGB-D frame + ground-truth label from OCID ARID20.

    Returns an OCIDScene object with:
      .load_rgb()   → (H, W, 3) uint8
      .load_depth() → (H, W) float32, metres
      .load_label() → (H, W) uint16, instance IDs (0 = background)
      .get_points() → (N, 3) float32 point cloud
      .n_gt_objects() → int, number of foreground instances in the label
    """
    if not _ocid_available():
        pytest.skip(
            "OCID data not found — run scripts/download_ocid.sh (in project-3dv) "
            "to enable dataset tests"
        )

    from ocid_loader import OCIDLoader  # noqa: PLC0415

    loader = OCIDLoader(str(_OCID_DIR))
    scene = next(loader.iter_scenes("ARID20", "table", "bottom", max_scenes=1))
    return scene
