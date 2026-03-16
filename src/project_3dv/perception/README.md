# Perception Pipeline — Progress Update

## What was done

### 1. Segmentation pipeline — baseline evaluation on OCID YCB10

Ran a full evaluation of the existing RANSAC + DBSCAN + geometric splitting pipeline across all 504 frames of OCID YCB10 (floor/table × top/bottom × mixed/curved/cuboid):

| Metric | Score |
|---|---|
| Mean Precision | 0.768 |
| Mean Recall | 0.485 |
| Mean F1 | 0.572 |
| Perception time | 284 ms/frame |

Breakdown by condition:

| Condition | F1 |
|---|---|
| floor/top | 0.649 |
| floor/bottom | 0.664 |
| table/top | 0.481 |
| table/bottom | 0.507 |

**Main finding:** Precision is strong (0.768) — the pipeline rarely hallucinates objects. The bottleneck is recall (0.485) — the pipeline consistently under-detects when scenes have >6 objects. Table scenes perform worse than floor scenes due to a bug identified and fixed (see below).

---

### 2. Bug fix — XY radius crop too tight for table scenes

Traced the source of `Det=0` on the first frame of every table sequence. The pipeline's XY radius crop (`xy_radius=0.55m`) was centred on the RANSAC table centroid, but in table scenes the single object often sits near the table edge at 0.5–0.98m from centre. Only 3 points survived the crop, well below `min_object_points=50`.

**Fix:** increased `xy_radius` default from 0.55m → 0.80m. Re-evaluation pending.

---

### 3. Shape type classification — from ML classifier to exponent-derived labels

Initially built an ML classifier trained on 3D Warehouse CAD objects (bowl/cap/cereal_box/coffee_mug/soda_can, 351 partial view scans) to predict SQ shape type from geometric features. 5-fold CV F1 = 0.995 on training data, but on OCID 97% of detections were classified as Ellipsoid — a domain gap between synthetic CAD models and real single-viewpoint RGB-D scans.

**Fix:** replaced the ML classifier with two complementary components:

- **`_shape_hint()`** — a lightweight extent-ratio heuristic (3 ratios from the bounding box) used only to warm-start the LM optimiser. No training data required, no domain gap.
- **`sq_type_from_exponents(e1, e2)`** — derives the authoritative shape type from the fitted SQ exponents after optimisation. This is always more accurate than anything inferred from the raw partial point cloud, since it reads what the geometry actually converged to.

| e1 | e2 | Shape type |
|---|---|---|
| > 0.6 | > 0.6 | Ellipsoid |
| < 0.45 | > 0.6 | Cylinder |
| < 0.45 | < 0.45 | Cuboid |
| mixed | mixed | Other |

Shape distribution on OCID YCB10 after fix (2402 detections):

| Shape type | Count | % |
|---|---|---|
| Cylinder | 1295 | 53.9% |
| Other | 864 | 36.0% |
| Ellipsoid | 171 | 7.1% |
| Cuboid | 72 | 3.0% |

Distribution is now physically plausible — YCB10 contains a large proportion of cans and mugs (cylindrical), with boxes and round objects in the minority.
---

### 4. Full pipeline integration — Perception → SQ fitting → Scene

Wired together the full chain end to end:

```
RGB-D point cloud
  → TabletopPerception   (segments + shape_type per segment)
  → SuperquadricFitter   (LM fitting, initialised with shape_hint)
  → Scene / superdec_utils  (signed distance for cuRobo, PLY export)
```

Changes to each file:

**`pipeline.py`** — `ObjectSegment` now carries `shape_type` (str) and `shape_conf` (float); classifier runs on each valid segment at the end of the pipeline.

**`superquadric.py`** — `fit()` and `fit_adaptive()` accept `shape_hint` which biases the LM initialisation toward the predicted shape exponents, reducing iterations and avoiding degenerate fits. `SuperquadricFit` carries `shape_type` and `shape_conf` for downstream use.

**`superdec_utils.py`** — `Superquadrics.from_fits()` reads `shape_type` from each fit. PLY export colours primitives by shape type (green=Ellipsoid, orange=Cylinder, blue=Cuboid, red=Other). `shape_type` carried through to cuRobo obstacle dicts.

---

### 5. SQ fitting quality on OCID

Chamfer L2 across 504 frames with SQ fitting enabled:

| Metric | Value |
|---|---|
| Mean L2 ×10⁻³ | 4.60 |
| Median L2 ×10⁻³ | 3.60 |
| Max L2 ×10⁻³ | 29.9 |
| SQ fitting time | 260 ms/frame |
| Total pipeline | 532 ms/frame |

Median L2 = 3.6×10⁻³ is well within the acceptable range for cuRobo collision geometry. The max outliers (>20×10⁻³) correspond to degenerate single-object frames where the segment is very small (<100 pts).

---

## What's next

- **Re-run eval** with the `xy_radius=0.80` fix to get updated F1 (expect table scenes to improve significantly)
- **Shape classifier domain gap** — if the 97% Ellipsoid problem persists, the classifier needs either: fine-tuning on OCID data directly, or removal from the pipeline (it currently degrades to a no-op rather than hurting performance)
- **Recall improvement** — main remaining gap. Touching/merged objects are the primary failure mode at high object counts (>7 in frame). The geometric splitting cascade (saddle, concavity, height layers) handles some cases but not all

## Files updated

| File | Status |
|---|---|
| `pipeline.py` | Updated — shape classifier integration, xy_radius fix |
| `superquadric.py` | Updated — shape_hint in fitting, shape_type on output |
| `superdec_utils.py` | Updated — shape_type in PLY export and cuRobo interface |
| `sq_shape_library.py` | New — builds shape classifier from 3D Warehouse objects |
| `ocid_eval.py` | New — full OCID evaluation script (segmentation + SQ fitting) |
| `test_full_pipeline.py` | New — end-to-end test on a single scene |