"""
superdec_fitter.py
==================
Drop-in replacement for SuperquadricFitter that uses the fine-tuned SuperDec
model for SQ decomposition on tabletop objects.

Default checkpoint
------------------
The default is the fine-tuned tabletop v2 model, which should be located at:
    ../checkpoints/superdec_tabletop/superdec_tabletop_finetune_v2/
relative to the superdec_dir.  You can override this via the `checkpoint_dir`
parameter (absolute path or relative to CWD).  The directory must contain:
    config.yaml   — Hydra training config (defines model architecture)
    ckpt.pt       — single checkpoint file, OR
    epoch_NNN.pt  — epoch-based saves (highest epoch is used automatically)

Base checkpoints (not fine-tuned) live at:
    {superdec_dir}/checkpoints/{normalized,shapenet}/ckpt.pt
and can still be used by passing checkpoint_dir explicitly.

Interface contract (compatible with SuperquadricFitter)
-------------------------------------------------------
Input:
    points: (N, 3) float32/float64 numpy array, world coordinates.
    SuperDec internally samples to 4096 points and normalises to unit sphere
    (per superdec/data/dataloader.py::normalize_points).

Output:
    MultiSQFit  — list of up to 16 SuperquadricFit primitives filtered by
                  existence score, with symmetric Chamfer L2 computed via
                  surface sampling.  Feeds directly into:
                    • fits_to_curobo_obstacles()  (collision avoidance)
                    • Scene.get_signed_distance()  (path planning SDF)
                    • GraspSelector.grasp_candidates()  (grasp planning)

Usage:
    from superdec_fitter import SuperdecFitter
    fitter = SuperdecFitter(superdec_dir='/path/to/superdec')
    result = fitter.fit_adaptive(points)   # (N,3) numpy array
"""

import logging
import os
import sys
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional
import scipy.spatial.transform as sst

# ---------------------------------------------------------------------------
# Superquadric exponent constraints
# ---------------------------------------------------------------------------
#: Maximum exponent for the convex SQ regime.  Shapes with e₁ or e₂ > 2 are
#: non-convex and produce unbounded implicit-function gradients, making them
#: unsuitable for gradient-based planners or safety filters.  All fitted
#: primitives are clamped to this value before they leave this module.
SQ_EXPONENT_CONVEX_MAX: float = 2.0

#: Minimum exponent to avoid numerical instability in sq_implicit() and
#: sq_radial_distance(), which compute x^(2/e).  Values near zero cause the
#: exponent 2/e to blow up, producing NaN signed-distance values.
SQ_EXPONENT_MIN: float = 1e-3

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# lazy import of superdec — path injected at construction time
# ---------------------------------------------------------------------------
_superdec_loaded = False

def _load_superdec(superdec_dir: str):
    global _superdec_loaded
    if _superdec_loaded:
        return
    if superdec_dir not in sys.path:
        sys.path.insert(0, superdec_dir)
    _superdec_loaded = True


# ---------------------------------------------------------------------------
# reuse SuperquadricFit / MultiSQFit from superquadric.py if available,
# otherwise define minimal stubs matching the canonical interface exactly
# ---------------------------------------------------------------------------
try:
    from superquadric import SuperquadricFit, MultiSQFit, sq_type_from_exponents
except ImportError:
    @dataclass
    class SuperquadricFit:
        sx: float = 0.05
        sy: float = 0.05
        sz: float = 0.05
        e1: float = 1.0
        e2: float = 1.0
        tx: float = 0.0
        ty: float = 0.0
        tz: float = 0.0
        rx: float = 0.0
        ry: float = 0.0
        rz: float = 0.0
        shape_type: str = "Other"
        shape_conf: float = 1.0
        converged: bool = True
        chamfer_l2: float = 0.0

        def surface_points(self, n_u: int = 50, n_v: int = 50) -> np.ndarray:
            return np.zeros((0, 3), dtype=np.float32)

    @dataclass
    class MultiSQFit:
        primitives: List[SuperquadricFit] = field(default_factory=list)
        n_points: int = 0

    def sq_type_from_exponents(e1: float, e2: float) -> str:
        if e1 > 0.6 and e2 > 0.6:   return "Ellipsoid"
        if e1 < 0.45 and e2 > 0.6:  return "Cylinder"
        if e1 < 0.45 and e2 < 0.45: return "Cuboid"
        return "Other"


def _rotmat_to_euler_xyz(R: np.ndarray) -> tuple:
    """Convert 3×3 rotation matrix to Euler XYZ angles (radians)."""
    r = sst.Rotation.from_matrix(R)
    rx, ry, rz = r.as_euler('xyz')
    return float(rx), float(ry), float(rz)


def _chamfer_l2_from_surface(pts: np.ndarray, fit: 'SuperquadricFit',
                              n_u: int = 50, n_v: int = 50) -> float:
    """Symmetric Chamfer L2 between input points and sampled SQ surface.

    Uses SuperquadricFit.surface_points(n_u, n_v) which is available on
    both the canonical class and the stub above.

    Parameters
    ----------
    n_u, n_v : int
        Surface sampling grid resolution (see surface_points() docstring).
        Default 50×50 = 2 500 surface points.
    """
    try:
        surf = fit.surface_points(n_u=n_u, n_v=n_v)
        if surf is None or len(surf) == 0:
            return 0.0
        pts32 = pts.astype(np.float32)
        surf32 = surf.astype(np.float32)
        # points → surface
        diff_ps = pts32[:, None, :] - surf32[None, :, :]
        d_ps = (diff_ps ** 2).sum(-1).min(-1)
        # surface → points
        diff_sp = surf32[:, None, :] - pts32[None, :, :]
        d_sp = (diff_sp ** 2).sum(-1).min(-1)
        return float(d_ps.mean() + d_sp.mean())
    except Exception:
        return 0.0


def _clamp_exponents(primitives: list) -> list:
    """Clamp e1 and e2 of every SuperquadricFit to the convex regime in-place.

    Ensures SQ_EXPONENT_MIN ≤ e1, e2 ≤ SQ_EXPONENT_CONVEX_MAX for all
    primitives.  Emits a single warning log if any clamping was needed so
    that callers know the raw network output was out of range.

    Parameters
    ----------
    primitives : list[SuperquadricFit]
        Modified in-place.

    Returns
    -------
    The same list (for call-chaining convenience).
    """
    clamped = False
    for fit in primitives:
        e1_orig, e2_orig = fit.e1, fit.e2
        fit.e1 = float(np.clip(fit.e1, SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX))
        fit.e2 = float(np.clip(fit.e2, SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX))
        if fit.e1 != e1_orig or fit.e2 != e2_orig:
            clamped = True
    if clamped:
        _log.warning(
            "SuperdecFitter: one or more primitives had exponents outside "
            "[%.4g, %.4g]; clamped to convex regime.",
            SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX,
        )
    return primitives


def _resolve_checkpoint_dir(superdec_dir: str, checkpoint_dir: Optional[str]) -> str:
    """Return the absolute path to a checkpoint directory.

    Resolution order:
      1. If checkpoint_dir is an absolute path, use it directly.
      2. If checkpoint_dir is a relative path, resolve from CWD.
      3. If checkpoint_dir is None, fall back to the default fine-tuned
         tabletop v2 checkpoint at
         ../checkpoints/superdec_tabletop/superdec_tabletop_finetune_v2
         (relative to superdec_dir).
    """
    if checkpoint_dir is not None:
        p = os.path.abspath(checkpoint_dir)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Checkpoint directory not found: {p}")
        return p

    # default: fine-tuned tabletop v2 checkpoint
    default = os.path.abspath(
        os.path.join(superdec_dir, '..', 'checkpoints',
                     'superdec_tabletop', 'superdec_tabletop_finetune_v2')
    )
    if os.path.isdir(default):
        return default

    # graceful fallback to base 'normalized' checkpoint bundled with superdec
    fallback = os.path.join(superdec_dir, 'checkpoints', 'normalized')
    if os.path.isdir(fallback):
        import warnings
        warnings.warn(
            f"Fine-tuned tabletop checkpoint not found at {default}. "
            f"Falling back to base 'normalized' checkpoint at {fallback}. "
            "For best tabletop performance use the fine-tuned model.",
            stacklevel=3,
        )
        return fallback

    raise FileNotFoundError(
        f"No checkpoint found at {default} or {fallback}. "
        "Pass checkpoint_dir= explicitly to specify the checkpoint path."
    )


def _find_checkpoint_file(checkpoint_dir: str) -> str:
    """Return path to the .pt checkpoint file inside checkpoint_dir.

    Supports both:
      • ckpt.pt          — single canonical checkpoint
      • epoch_NNN.pt     — epoch-based saves; highest epoch is used
    """
    ckpt_single = os.path.join(checkpoint_dir, 'ckpt.pt')
    if os.path.exists(ckpt_single):
        return ckpt_single

    import glob
    epoch_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'epoch_*.pt')))
    if epoch_files:
        # sort by epoch number, return the latest
        def _epoch_num(p):
            try:
                return int(os.path.basename(p).split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                return -1
        return max(epoch_files, key=_epoch_num)

    raise FileNotFoundError(
        f"No checkpoint file (ckpt.pt or epoch_*.pt) found in {checkpoint_dir}"
    )


class SuperdecFitter:
    """SuperDec-based superquadric fitter for tabletop objects.

    Defaults to the fine-tuned tabletop v2 model.  The base (non-fine-tuned)
    checkpoints can be used by passing checkpoint_dir explicitly.

    Parameters
    ----------
    superdec_dir : str
        Path to the cloned superdec repository root.
    checkpoint_dir : str, optional
        Path to the checkpoint directory (absolute, or relative to CWD).
        Must contain config.yaml and either ckpt.pt or epoch_*.pt files.
        Defaults to ../checkpoints/superdec_tabletop/superdec_tabletop_finetune_v2
        relative to superdec_dir.
    exist_threshold : float
        Primitives whose existence score is below this value are discarded.
        Lower → more primitives returned. Default 0.3.
    n_points : int
        Number of points sampled from each segment before inference.
        Must match the training resolution (4096). Do not change.
    device : str, optional
        'cuda' or 'cpu'. Defaults to CUDA if available.
    """

    def __init__(
        self,
        superdec_dir: str,
        checkpoint_dir: Optional[str] = None,
        exist_threshold: float = 0.3,
        n_points: int = 4096,
        device: Optional[str] = None,
    ):
        _load_superdec(superdec_dir)

        from omegaconf import OmegaConf
        from superdec.superdec import SuperDec
        self._normalize_points    = None   # initialised below
        self._denormalize_outdict = None

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device          = device
        self.exist_threshold = exist_threshold
        self.n_points        = n_points
        self.superdec_dir    = superdec_dir

        ckpt_dir = _resolve_checkpoint_dir(superdec_dir, checkpoint_dir)
        ckp_path = _find_checkpoint_file(ckpt_dir)
        cfg_path = os.path.join(ckpt_dir, 'config.yaml')

        if not os.path.exists(ckp_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckp_path}")

        ckp = torch.load(ckp_path, map_location=device, weights_only=False)
        cfg = OmegaConf.load(cfg_path)

        self.model = SuperDec(cfg.superdec).to(device)
        self.model.load_state_dict(ckp['model_state_dict'])
        self.model.eval()
        self.model.lm_optimization = False

        # import data utils
        from superdec.data.dataloader import normalize_points, denormalize_outdict
        self._normalize_points   = normalize_points
        self._denormalize_outdict = denormalize_outdict

        print(f"[SuperdecFitter] loaded checkpoint '{os.path.basename(ckp_path)}' "
              f"from {ckpt_dir} on {device}")

    # ------------------------------------------------------------------
    def fit_adaptive(
        self,
        points: np.ndarray,
        shape_hint=None,        # ignored — kept for interface compatibility
        n_restarts: int = 1,    # ignored
        **kwargs,
    ) -> 'MultiSQFit':
        """
        Run SuperDec inference on a single object point cloud.

        Parameters
        ----------
        points : (N, 3) float32 array — object segment in world coordinates.

        Returns
        -------
        MultiSQFit with up to 16 SuperquadricFit primitives (filtered by exist).
        """
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 10:
            return MultiSQFit(primitives=[], n_points=0)

        # sample to fixed resolution
        n = len(pts)
        idx = np.random.choice(n, self.n_points, replace=n < self.n_points)
        pts_sampled = pts[idx]

        # normalise to unit sphere (SuperDec convention)
        pts_norm, translation, scale = self._normalize_points(pts_sampled)
        # translation: (3,)   scale: scalar
        # wrap in batch dim
        x = torch.from_numpy(pts_norm).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            out = self.model(x)
            raw_out = out

        # denormalise back to world coordinates
        # denormalize_outdict expects translation: (B,3), scale: (B,)
        t_batch = torch.from_numpy(translation).unsqueeze(0).float().to(self.device)
        s_batch = torch.tensor([scale], dtype=torch.float32).to(self.device)
        out = self._denormalize_outdict(out, t_batch, s_batch)

        # extract per-primitive parameters — all (B=1, P=16, ...)
        scales      = out['scale'][0].cpu().numpy()       # (16, 3)
        rotations   = out['rotate'][0].cpu().numpy()      # (16, 3, 3)
        translations= out['trans'][0].cpu().numpy()       # (16, 3)
        exponents   = out['shape'][0].cpu().numpy()       # (16, 2)
        exist_score = out['exist'][0].cpu().numpy()       # (16,)  or sigmoid needed

        # exist may be raw logits — apply sigmoid if range is not [0,1]
        if exist_score.min() < -0.5 or exist_score.max() > 1.5:
            exist_score = 1.0 / (1.0 + np.exp(-exist_score))

        primitives = []
        for p_idx in range(scales.shape[0]):
            if float(exist_score[p_idx]) < self.exist_threshold:
                continue

            e1 = float(np.clip(exponents[p_idx, 0], SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX))
            e2 = float(np.clip(exponents[p_idx, 1], SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX))
            sx, sy, sz = (float(v) for v in np.abs(scales[p_idx]))
            tx, ty, tz = (float(v) for v in translations[p_idx])
            rx, ry, rz = _rotmat_to_euler_xyz(rotations[p_idx])

            fit = SuperquadricFit(
                sx=sx, sy=sy, sz=sz,
                e1=e1, e2=e2,
                tx=tx, ty=ty, tz=tz,
                rx=rx, ry=ry, rz=rz,
                shape_type=sq_type_from_exponents(e1, e2),
                shape_conf=float(exist_score[p_idx]),
                converged=True,
                chamfer_l2=0.0,
            )
            primitives.append(fit)

        _clamp_exponents(primitives)

        # compute per-primitive Chamfer L2 via surface sampling
        if primitives and len(pts) > 0:
            for fit in primitives:
                fit.chamfer_l2 = _chamfer_l2_from_surface(pts, fit)

        return MultiSQFit(primitives=primitives, n_points=len(pts))

    def fit_batch(self, points_list: List[np.ndarray]) -> List['MultiSQFit']:
        """
        Batch inference — all objects in one forward pass.
        Dramatically faster than calling fit_adaptive N times.
        """
        if not points_list:
            return []

        valid_idx = [i for i, pts in enumerate(points_list) if len(pts) >= 10]
        if not valid_idx:
            return [MultiSQFit(primitives=[], n_points=0) for _ in points_list]

        pts_raw_list   = []
        translations   = []
        scales         = []
        pts_norm_batch = []

        for i in valid_idx:
            pts = np.asarray(points_list[i], dtype=np.float64)
            pts_raw_list.append(pts)
            n = len(pts)
            idx = np.random.choice(n, self.n_points, replace=n < self.n_points)
            pts_sampled = pts[idx]
            pts_norm, t, s = self._normalize_points(pts_sampled)
            pts_norm_batch.append(pts_norm)
            translations.append(t)
            scales.append(s)

        x = torch.from_numpy(np.stack(pts_norm_batch, axis=0)).float().to(self.device)

        with torch.no_grad():
            out = self.model(x)
            raw_out = out

        t_batch = torch.from_numpy(np.stack(translations, axis=0)).float().to(self.device)
        s_batch = torch.tensor(scales, dtype=torch.float32).to(self.device)
        out = self._denormalize_outdict(out, t_batch, s_batch)

        scales_out  = out['scale'].cpu().numpy()
        rotations   = out['rotate'].cpu().numpy()
        trans_out   = out['trans'].cpu().numpy()
        exponents   = out['shape'].cpu().numpy()
        exist_score = out['exist'].cpu().numpy()
        assign      = raw_out.get('assign_matrix')

        if exist_score.min() < -0.5 or exist_score.max() > 1.5:
            exist_score = 1.0 / (1.0 + np.exp(-exist_score))

        results = [MultiSQFit(primitives=[], n_points=0) for _ in points_list]

        for b, orig_i in enumerate(valid_idx):
            pts_b = pts_raw_list[b]

            primitives = []
            for p_idx in range(scales_out.shape[1]):
                if float(exist_score[b, p_idx]) < self.exist_threshold:
                    continue

                e1 = float(np.clip(exponents[b, p_idx, 0], SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX))
                e2 = float(np.clip(exponents[b, p_idx, 1], SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX))
                sx, sy, sz = (float(v) for v in np.abs(scales_out[b, p_idx]))
                tx, ty, tz = (float(v) for v in trans_out[b, p_idx])
                rx, ry, rz = _rotmat_to_euler_xyz(rotations[b, p_idx])

                fit = SuperquadricFit(
                    sx=sx, sy=sy, sz=sz,
                    e1=e1, e2=e2,
                    tx=tx, ty=ty, tz=tz,
                    rx=rx, ry=ry, rz=rz,
                    shape_type=sq_type_from_exponents(e1, e2),
                    shape_conf=float(exist_score[b, p_idx]),
                    converged=True,
                    chamfer_l2=0.0,
                )
                primitives.append(fit)

            _clamp_exponents(primitives)

            if primitives and len(pts_b) > 0:
                for fit in primitives:
                    fit.chamfer_l2 = _chamfer_l2_from_surface(pts_b, fit)

            results[orig_i] = MultiSQFit(primitives=primitives, n_points=len(pts_b))

        return results

