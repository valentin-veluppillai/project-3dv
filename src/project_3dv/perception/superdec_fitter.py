"""
superdec_fitter.py
==================
Drop-in replacement for SuperquadricFitter that uses the pretrained
SuperDec model for SQ decomposition.

Shares the same fit_adaptive() interface — ocid_eval.py and pipeline.py
need zero changes beyond swapping the fitter instance.

Usage:
    from superdec_fitter import SuperdecFitter
    fitter = SuperdecFitter(
        superdec_dir='/path/to/superdec',
        checkpoint='normalized',   # or 'shapenet'
        exist_threshold=0.3,       # primitives below this are discarded
        device='cuda',             # or 'cpu'
    )
    result = fitter.fit_adaptive(points)   # (N,3) float32 np.ndarray
"""

import os
import sys
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional
import scipy.spatial.transform as sst

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
# otherwise define minimal stubs
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

    @dataclass
    class MultiSQFit:
        primitives: List[SuperquadricFit] = field(default_factory=list)
        total_chamfer_l2: float = 0.0

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


def _chamfer_l2(pts: np.ndarray, sq_fit: 'SuperquadricFit') -> float:
    """Approximate Chamfer L2 from points to the nearest SQ surface point.
    Uses the SQ implicit function as a proxy — fast but approximate."""
    try:
        from superquadric import sq_surface_points
        surf = sq_surface_points(sq_fit, n=512)
        if surf is None or len(surf) == 0:
            return 0.0
        # one-sided: mean squared distance from pts to nearest surface point
        diff = pts[:, None, :] - surf[None, :, :]
        dists = (diff ** 2).sum(-1).min(-1)
        return float(dists.mean())
    except Exception:
        return 0.0


class SuperdecFitter:
    """
    SuperDec-based superquadric fitter.

    Parameters
    ----------
    superdec_dir : str
        Path to the cloned superdec repository root.
    checkpoint : str
        'normalized' (recommended for generic tabletop objects)
        or 'shapenet'.
    exist_threshold : float
        Primitives whose existence score is below this value are discarded.
        Lower → more primitives returned. Default 0.3.
    n_points : int
        Number of points sampled from each segment before inference.
        Must be 4096 (SuperDec training resolution).
    device : str
        'cuda' or 'cpu'. CPU will work but is ~10× slower.
    """

    def __init__(
        self,
        superdec_dir: str,
        checkpoint: str = 'normalized',
        exist_threshold: float = 0.3,
        n_points: int = 4096,
        device: Optional[str] = None,
    ):
        _load_superdec(superdec_dir)

        from omegaconf import OmegaConf
        from superdec.superdec import SuperDec
        self._normalize_points  = None  # lazy
        self._denormalize_outdict = None

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.exist_threshold = exist_threshold
        self.n_points = n_points
        self.superdec_dir = superdec_dir

        ckp_path = os.path.join(superdec_dir, 'checkpoints', checkpoint, 'ckpt.pt')
        cfg_path = os.path.join(superdec_dir, 'checkpoints', checkpoint, 'config.yaml')
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

        print(f"[SuperdecFitter] loaded checkpoint '{checkpoint}' on {device}")

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
            return MultiSQFit(primitives=[], total_chamfer_l2=0.0)

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

            e1 = float(np.clip(exponents[p_idx, 0], 0.1, 1.9))
            e2 = float(np.clip(exponents[p_idx, 1], 0.1, 1.9))
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

        # compute per-primitive Chamfer L2 using assign_matrix
        # assign_matrix: (N_pts, P) soft assignment weights
        if primitives and len(pts) > 0:
            pts32 = pts.astype(np.float32)
            assign = raw_out.get('assign_matrix')
            if assign is not None:
                assign_np = assign[0].cpu().numpy()  # (N_pts, P_total)
            for i, fit in enumerate(primitives):
                centroid = np.array([fit.tx, fit.ty, fit.tz], dtype=np.float32)
                dists_sq = np.sum((pts32 - centroid) ** 2, axis=1)
                if assign is not None and i < assign_np.shape[1]:
                    weights = assign_np[:, i]  # (n_sub,)
                    n = min(weights.shape[0], dists_sq.shape[0])
                    weights = weights[:n] / (weights[:n].sum() + 1e-8)
                    l2 = float(np.sum(weights * dists_sq[:n]))
                else:
                    l2 = float(np.mean(dists_sq))
                fit.chamfer_l2 = l2

        total_l2 = float(np.mean([p.chamfer_l2 for p in primitives])) if primitives else 0.0
        return MultiSQFit(primitives=primitives)

    def fit_batch(self, points_list: List[np.ndarray]) -> List['MultiSQFit']:
        """
        Batch inference — all objects in one forward pass.
        Dramatically faster than calling fit_adaptive N times.
        """
        if not points_list:
            return []

        valid_idx = [i for i, pts in enumerate(points_list) if len(pts) >= 10]
        if not valid_idx:
            return [MultiSQFit(primitives=[]) for _ in points_list]

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

        results = [MultiSQFit(primitives=[]) for _ in points_list]

        for b, orig_i in enumerate(valid_idx):
            pts32 = pts_raw_list[b].astype(np.float32)
            assign_np = assign[b].cpu().numpy() if assign is not None else None

            primitives = []
            for p_idx in range(scales_out.shape[1]):
                if float(exist_score[b, p_idx]) < self.exist_threshold:
                    continue

                e1 = float(np.clip(exponents[b, p_idx, 0], 0.1, 1.9))
                e2 = float(np.clip(exponents[b, p_idx, 1], 0.1, 1.9))
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

            if primitives:
                for i, fit in enumerate(primitives):
                    centroid = np.array([fit.tx, fit.ty, fit.tz], dtype=np.float32)
                    dists_sq = np.sum((pts32 - centroid) ** 2, axis=1)
                    if assign_np is not None and i < assign_np.shape[1]:
                        weights = assign_np[:, i]
                        n = min(weights.shape[0], dists_sq.shape[0])
                        weights = weights[:n] / (weights[:n].sum() + 1e-8)
                        l2 = float(np.sum(weights * dists_sq[:n]))
                    else:
                        l2 = float(np.mean(dists_sq))
                    fit.chamfer_l2 = l2

            results[orig_i] = MultiSQFit(primitives=primitives)

        return results

