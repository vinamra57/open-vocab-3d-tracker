#!/usr/bin/env python3
"""Step 5 / Track A: classical Kalman + RTS smoother for the 3D box trajectory.

Part of the generalized, in-the-wild, open-vocabulary video object tracking
pipeline. Like the other steps it is fully generic: it works on ANY video for
which Steps 1-4 have been run and contains no hardcoded data paths, environment
paths, or dataset-specific assumptions -- everything is CLI-driven.

What it does
------------
Steps 1-4 produce a *subsampled, jittery* per-frame 3D box trajectory (Step 4 /
FoundationPose, ~5 fps). Step 5 is the temporal smoother that turns it into a
*dense, smooth* per-frame trajectory. This script is **Track A** -- the
classical algorithmic baseline: a forward Kalman filter plus a backward RTS
(Rauch-Tung-Striebel) pass. RTS, not a plain filter, because Step 5 is offline
post-processing -- the whole noisy trajectory is available up front, so future
keyframes should inform every frame.

Tracks B (box-only transformer) and C (WildDet3D image-grounded refiner) are
learned smoothers built by sibling agents; this script's output format is kept
consistent with theirs so all three are directly comparable, and evaluable
against ground-truth 3D box tracks (see ``trajectory.npz``).

Box representation (why it filters well)
----------------------------------------
All smoothing happens in a common WORLD frame (Step-1 extrinsics): a monocular
moving camera makes a static object trace a wild path in camera coordinates, so
filtering in camera frame would fight ego-motion. The 9-DoF box is split into
three dynamically-independent blocks, each chosen so smooth physical motion maps
to a smooth, low-order state trajectory:

  * Center  -- world-frame (x,y,z) with a constant-velocity (optionally
    constant-acceleration) model. Linear-Gaussian; the KF handles it natively.
  * Dims    -- log-dimensions. Log keeps dims positive after smoothing and turns
    the OBB's multiplicative size jitter into additive noise. Random-walk model
    (or held constant per trajectory for a rigid object).
  * Rotation-- an error-state formulation. Quaternions break a linear KF
    (unit-norm manifold + double cover); 6D rotation is over-parameterized.
    Instead a nominal rotation R_nom is carried on the SO(3) manifold and the KF
    runs linearly on a small so(3) tangent-space error phi (a 3-vector), with a
    constant-angular-velocity model. After each outer iteration the smoothed phi
    is composed back into R_nom (iterated error-state RTS).

OBB / rotation canonicalization (a prerequisite, not optional)
--------------------------------------------------------------
Step 4's box comes from ``trimesh.bounds.oriented_bounds`` on a *different*
SAM3D mesh every frame, which assigns the three OBB axes arbitrarily. The pair
(R_world, extents) therefore has a 24-fold ambiguity -- the octahedral group of
signed axis permutations P with det +1 acts jointly:

        (R, d)  ->  (R @ P,  |P|^T @ d)

So before filtering we *canonicalize*: greedily, from the highest-confidence
anchor frame outward, pick the relabeling P whose rotation R@P is geodesically
closest to the running estimate. This makes the rotation track AND the per-axis
dimension columns temporally consistent -- it is simultaneously the axis and the
rotation canonicalization. It also subsumes object symmetry (a symmetric box's
equivalent poses are elements of the same group).

Densification, gaps, outliers
------------------------------
The KF predict step runs on every frame; the update step runs only where Step 4
has a box -> a box for every frame in the trajectory span (the dense output).
Frames with no box (object invisible, or simply not subsampled) are predict-only.
Gross FoundationPose failures are rejected by Mahalanobis-distance gating, per
block. Per-frame measurement noise R is scaled by Step 3/4 confidence
(``sam3_score`` x ``proj_box_mask_iou``) -- high-confidence boxes are trusted
more -- and the center's R is built anisotropically in camera frame (depth
jitters worst) then rotated into world.

Inputs
------
  --step4_dir   Step 4 output dir (output/step4/<video>): the trajectory to
                smooth. ``meta.json`` already carries world-frame boxes,
                per-frame camera_to_world and intrinsics, and confidences.
  --step1_dir   Step 1 output dir (output/step1/<video>), OPTIONAL. Only needed
                for the dense camera-frame mirror on non-measured frames and for
                --save_viz. The core smoother runs from Step 4 alone.

Output  (written to ``<output_dir>/<video_name>/``)
---------------------------------------------------
  meta.json        Run metadata, config, conventions, smoothness metrics, and a
                   dense per-frame record list (``frames``): world-frame center,
                   dims, rotation (matrix + wxyz quat), 8 corners, 4x4 box->world
                   transform, per-frame std (uncertainty), the camera-frame
                   mirror where extrinsics are available, and has_measurement /
                   gated flags.
  trajectory.npz   The same dense trajectory as flat arrays for evaluation:
                   frame_index, center, dims, quat, R, corners_world,
                   has_measurement, *_std, plus the raw measured arrays.
  boxes_world.npy  (N,8,3) float32 dense world-frame corners -- mirrors Step 4's
                   convenience array.
  viz/ + viz.mp4   Smoothed box (green) vs raw Step-4 box (red) overlay, with
                   --save_viz and --step1_dir.

Environment
-----------
Pure numpy for the core (cv2 only for --save_viz, imported lazily). Run with any
Python that has numpy, e.g. the pipeline's foundationpose / radio-vipe env:

    python scripts/step5_kalman_smoother.py --step4_dir output/step4/dog-example \\
        --step1_dir output/step1/dog-example --save_viz
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# 8 OBB corners in box-local (centered, axis-aligned) coords -- same ordering as
# Step 4, so corners_world is directly comparable across steps.
_OBB_SIGNS = np.array([
    [+1, +1, +1], [+1, +1, -1], [+1, -1, -1], [+1, -1, +1],
    [-1, +1, +1], [-1, +1, -1], [-1, -1, -1], [-1, -1, +1],
], dtype=np.float64)
_OBB_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] step5: {msg}", flush=True)


# ===========================================================================
# SO(3) helpers (numpy-only -- no scipy dependency)
# ===========================================================================
def hat(w: np.ndarray) -> np.ndarray:
    """so(3) tangent 3-vector -> 3x3 skew-symmetric matrix."""
    w = np.asarray(w, dtype=np.float64)
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vector -> rotation matrix (Rodrigues)."""
    w = np.asarray(w, dtype=np.float64)
    theta = float(np.linalg.norm(w))
    K = hat(w)
    if theta < 1e-8:
        return np.eye(3) + K + 0.5 * (K @ K)
    K /= theta
    return (np.eye(3) + math.sin(theta) * K
            + (1.0 - math.cos(theta)) * (K @ K))


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map: rotation matrix -> rotation vector."""
    R = np.asarray(R, dtype=np.float64)
    cos = (np.trace(R) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    theta = math.acos(cos)
    if theta < 1e-7:  # near identity
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]])
    if math.pi - theta < 1e-5:  # near pi: skew part vanishes, use (R+I)/2
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        col = A[:, k] / (axis[k] if axis[k] > 1e-9 else 1.0)
        col = col / (np.linalg.norm(col) + 1e-12)
        return theta * col
    w = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]])
    return theta / (2.0 * math.sin(theta)) * w


def geodesic_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angle (radians) of the relative rotation between R1 and R2."""
    return float(np.linalg.norm(so3_log(R1.T @ R2)))


def project_so3(R: np.ndarray) -> np.ndarray:
    """Project an approximate matrix onto SO(3) (nearest proper rotation)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0:
        U[:, -1] *= -1.0
        Rp = U @ Vt
    return Rp


def slerp(Ra: np.ndarray, Rb: np.ndarray, t: float) -> np.ndarray:
    """Geodesic interpolation on SO(3): Ra at t=0, Rb at t=1."""
    return Ra @ so3_exp(t * so3_log(Ra.T @ Rb))


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion, (w, x, y, z) order."""
    m = np.asarray(R, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s,
             (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
             (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
             0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q, dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


# ===========================================================================
# Octahedral group + box (R, d) canonicalization
# ===========================================================================
def octahedral_group() -> np.ndarray:
    """The 24 signed 3x3 permutation matrices with det +1 (proper rotations).

    Relabeling the three OBB axes is exactly right-multiplication by one of
    these: (R, d) -> (R @ P, |P|^T @ d) is the same physical box.
    """
    import itertools
    mats = []
    for perm in itertools.permutations(range(3)):
        base = np.zeros((3, 3))
        for i, j in enumerate(perm):
            base[i, j] = 1.0
        for signs in itertools.product((1.0, -1.0), repeat=3):
            P = base * np.asarray(signs)[:, None]
            if abs(np.linalg.det(P) - 1.0) < 1e-9:
                mats.append(P)
    return np.asarray(mats)  # (24, 3, 3)


_OCTAHEDRAL = octahedral_group()


def canonicalize_trajectory(meas_idx, R_list, d_list, anchor_local):
    """Resolve the per-frame OBB axis/rotation ambiguity.

    Walks outward from the highest-confidence anchor; each box is relabeled by
    the octahedral P that puts its rotation geodesically closest to the
    already-canonicalized neighbour. Returns (R_canon list, d_canon list).
    """
    n = len(R_list)
    R_canon = [None] * n
    d_canon = [None] * n
    R_canon[anchor_local] = project_so3(R_list[anchor_local])
    d_canon[anchor_local] = np.asarray(d_list[anchor_local], dtype=np.float64)

    def relabel(local, ref_R):
        R_raw = project_so3(R_list[local])
        d_raw = np.asarray(d_list[local], dtype=np.float64)
        best_P, best_ang = None, None
        for P in _OCTAHEDRAL:
            ang = geodesic_angle(R_raw @ P, ref_R)
            if best_ang is None or ang < best_ang:
                best_ang, best_P = ang, P
        R_canon[local] = R_raw @ best_P
        d_canon[local] = np.abs(best_P).T @ d_raw

    for i in range(anchor_local + 1, n):       # forward from anchor
        relabel(i, R_canon[i - 1])
    for i in range(anchor_local - 1, -1, -1):  # backward from anchor
        relabel(i, R_canon[i + 1])
    return R_canon, d_canon


# ===========================================================================
# Generic linear Kalman filter + RTS smoother for a kinematic block
# ===========================================================================
def kinematic_rts(n_frames, measurements, order, q_proc, p0_var, gate, dt=1.0):
    """Forward KF + backward RTS for a constant-order kinematic block.

    The block models 3 spatial channels with `order` derivatives each
    (order 0 = random walk, 1 = constant-velocity, 2 = constant-acceleration).
    State is grouped by derivative: [d0(3), d1(3), ...], dim D = 3*(order+1).

    measurements : {frame_k: (z(3,), R(3,3))}  -- order-0 observations + noise.
    Returns (state_smooth (n,D), cov_smooth (n,D,D), gated set).
    """
    m = order
    D = 3 * (m + 1)
    fac = [math.factorial(i) for i in range(2 * m + 2)]

    # 1-D kinematic transition F1 and integrated white-noise covariance Q1,
    # then lift to 3 channels via Kronecker product with I_3.
    F1 = np.zeros((m + 1, m + 1))
    Q1 = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            if j >= i:
                F1[i, j] = dt ** (j - i) / fac[j - i]
            p = 2 * m + 1 - i - j
            Q1[i, j] = q_proc * dt ** p / (p * fac[m - i] * fac[m - j])
    F = np.kron(F1, np.eye(3))
    Q = np.kron(Q1, np.eye(3))
    H = np.zeros((3, D))
    H[:, :3] = np.eye(3)

    # Initial prior: order-0 from the first available measurement, derivatives 0.
    s0 = np.zeros(D)
    first = measurements.get(0)
    if first is None and measurements:
        first = measurements[min(measurements)]
    if first is not None:
        s0[:3] = first[0]
    P0 = np.diag(np.repeat(np.asarray(p0_var, dtype=np.float64), 3))

    s_filt = np.zeros((n_frames, D))
    P_filt = np.zeros((n_frames, D, D))
    s_pred = np.zeros((n_frames, D))
    P_pred = np.zeros((n_frames, D, D))
    gated = set()
    s, P = s0.copy(), P0.copy()

    for k in range(n_frames):
        if k == 0:
            sp, Pp = s0.copy(), P0.copy()
        else:
            sp = F @ s
            Pp = F @ P @ F.T + Q
        s_pred[k], P_pred[k] = sp, Pp
        s, P = sp, Pp
        if k in measurements:
            z, Rm = measurements[k]
            y = z - H @ s
            S = H @ P @ H.T + Rm
            Sinv = np.linalg.inv(S)
            d2 = float(y @ Sinv @ y)
            if gate > 0.0 and d2 > gate:
                gated.add(k)                       # outlier -> predict-only
            else:
                K = P @ H.T @ Sinv
                s = s + K @ y
                IKH = np.eye(D) - K @ H
                P = IKH @ P @ IKH.T + K @ Rm @ K.T  # Joseph form
        s_filt[k], P_filt[k] = s, P

    # Backward RTS pass.
    s_sm = s_filt.copy()
    P_sm = P_filt.copy()
    for k in range(n_frames - 2, -1, -1):
        C = P_filt[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
        s_sm[k] = s_filt[k] + C @ (s_sm[k + 1] - s_pred[k + 1])
        P_sm[k] = P_filt[k] + C @ (P_sm[k + 1] - P_pred[k + 1]) @ C.T
    return s_sm, P_sm, gated


# ===========================================================================
# Input loading
# ===========================================================================
def load_step4(step4_dir: Path):
    """Parse Step 4 meta.json into a per-frame world-frame box trajectory."""
    meta = json.loads((step4_dir / "meta.json").read_text())
    recs = []
    for f in meta.get("frames", []):
        if f.get("status") != "ok":
            continue
        box = f.get("box") or {}
        if box.get("center_world") is None or box.get("R_world") is None:
            continue  # no camera pose for this frame -> cannot lift to world
        recs.append({
            "frame_index": int(f["source_frame_index"]),
            "center": np.asarray(box["center_world"], dtype=np.float64),
            "R": np.asarray(box["R_world"], dtype=np.float64),
            "dims": np.asarray(box["obb_extents"], dtype=np.float64),
            "sam3_score": f.get("sam3_score"),
            "proj_iou": f.get("proj_box_mask_iou"),
            "camera_to_world": (np.asarray(f["camera_to_world"], dtype=np.float64)
                                if f.get("camera_to_world") is not None else None),
            "intrinsics": (np.asarray(f["intrinsics"], dtype=np.float64)
                           if f.get("intrinsics") is not None else None),
            "corners_world": (np.asarray(box["corners_world"], dtype=np.float64)
                              if box.get("corners_world") is not None else None),
        })
    recs.sort(key=lambda r: r["frame_index"])
    if not recs:
        sys.exit("FATAL: Step 4 produced no world-frame boxes to smooth.")
    return meta, recs


def load_step1(step1_dir: Path):
    """Load Step 1 camera params, keyed by original video frame index.

    Returns {orig_frame_index: {'w2c', 'c2w', 'K'}} or None if unavailable.
    """
    if step1_dir is None:
        return None
    meta_p = step1_dir / "meta.json"
    if not meta_p.is_file():
        log(f"WARNING: no Step 1 meta.json at {step1_dir}; "
            f"camera-frame output / viz disabled")
        return None
    meta = json.loads(meta_p.read_text())
    src = meta.get("source_frame_indices")
    extr = np.load(step1_dir / "extrinsics.npy")
    c2w = np.load(step1_dir / "poses_c2w.npy")
    K = np.load(step1_dir / "intrinsics.npy")
    if src is None:
        src = list(range(len(extr)))
    cam = {}
    for j, orig in enumerate(src):
        cam[int(orig)] = {"w2c": extr[j].astype(np.float64),
                          "c2w": c2w[j].astype(np.float64),
                          "K": K[j].astype(np.float64),
                          "step1_index": j}
    return cam


# ===========================================================================
# Measurement-noise model
# ===========================================================================
def confidence_weights(recs):
    """Per-measurement quality in (0,1], normalized by the trajectory median.

    Combines Step 3's sam3_score with Step 4's projected-box/mask IoU; both are
    quality proxies. Returns a weight per record -- larger = more trustworthy.
    """
    raw = []
    for r in recs:
        s = r["sam3_score"] if r["sam3_score"] is not None else 1.0
        iou = r["proj_iou"] if r["proj_iou"] is not None else 1.0
        raw.append(max(1e-3, float(s) * float(iou)))
    raw = np.asarray(raw)
    med = float(np.median(raw)) if len(raw) else 1.0
    return np.clip(raw / max(med, 1e-6), 0.25, 4.0)


# ===========================================================================
# Core: smooth one trajectory
# ===========================================================================
def smooth_trajectory(recs, cfg):
    """Run the full canonicalize -> KF -> RTS smoother. Returns a dense result."""
    frame_idx = [r["frame_index"] for r in recs]
    lo, hi = frame_idx[0], frame_idx[-1]
    n_frames = hi - lo + 1                       # dense output grid (dt = 1)
    local_of = {fi: fi - lo for fi in frame_idx}  # source frame -> output index

    # ---- confidence -> per-measurement weight ----------------------------
    weights = confidence_weights(recs)

    # ---- OBB / rotation canonicalization (anchor = best confidence) ------
    anchor_local = int(np.argmax(weights))
    R_canon, d_canon = canonicalize_trajectory(
        frame_idx, [r["R"] for r in recs], [r["dims"] for r in recs],
        anchor_local)
    log(f"canonicalized {len(recs)} boxes; anchor = frame "
        f"{recs[anchor_local]['frame_index']} "
        f"(weight {weights[anchor_local]:.2f})")

    # ===== Center block: constant-velocity / -acceleration in world =======
    meas_c = {}
    for li, r in enumerate(recs):
        k = local_of[r["frame_index"]]
        # Anisotropic base noise in camera frame (depth jitters worst), rotated
        # into the world frame; then scaled by per-frame confidence.
        base = np.diag([cfg["r_center_xy"] ** 2,
                        cfg["r_center_xy"] ** 2,
                        cfg["r_center_z"] ** 2])
        c2w = r["camera_to_world"]
        if c2w is not None:
            Rc = c2w[:3, :3]
            base = Rc @ base @ Rc.T
        meas_c[k] = (r["center"], base / weights[li])
    c_sm, c_cov, c_gated = kinematic_rts(
        n_frames, meas_c, cfg["center_order"], cfg["q_center"],
        [1.0] * (cfg["center_order"] + 1), cfg["gate"])
    center = c_sm[:, :3]
    center_std = np.sqrt(np.clip(np.diagonal(c_cov, 0, 1, 2)[:, :3], 0, None))

    # ===== Dimension block: log-dims random walk (or held constant) =======
    if cfg["dims_mode"] == "constant":
        logd = np.array([np.log(np.maximum(d, 1e-4)) for d in d_canon])
        wmean = np.average(logd, axis=0, weights=weights)
        dims = np.tile(np.exp(wmean), (n_frames, 1))
        dim_std = np.tile(np.sqrt(np.average((logd - wmean) ** 2, axis=0,
                                             weights=weights)), (n_frames, 1))
        d_gated = set()
    else:
        meas_d = {}
        for li, r in enumerate(recs):
            k = local_of[r["frame_index"]]
            z = np.log(np.maximum(d_canon[li], 1e-4))
            Rm = np.eye(3) * (cfg["r_dim"] ** 2) / weights[li]
            meas_d[k] = (z, Rm)
        d_sm, d_cov, d_gated = kinematic_rts(
            n_frames, meas_d, 0, cfg["q_dim"], [1.0], cfg["gate"])
        dims = np.exp(d_sm[:, :3])
        dim_std = np.sqrt(np.clip(np.diagonal(d_cov, 0, 1, 2)[:, :3], 0, None))

    # ===== Rotation block: iterated error-state RTS on SO(3) ==============
    # Nominal starts constant (= anchor rotation); each outer iteration the KF
    # smooths the so(3) tangent residual and the result is composed back in.
    R_nom = np.array([R_canon[anchor_local].copy() for _ in range(n_frames)])
    r_gated = set()
    rot_cov0 = np.zeros((n_frames, 3))
    for it in range(cfg["rot_outer_iters"]):
        meas_r = {}
        for li, r in enumerate(recs):
            k = local_of[r["frame_index"]]
            z = so3_log(R_nom[k].T @ R_canon[li])
            Rm = np.eye(3) * (cfg["r_rot"] ** 2) / weights[li]
            meas_r[k] = (z, Rm)
        r_sm, r_cov, r_gated = kinematic_rts(
            n_frames, meas_r, cfg["rot_order"], cfg["q_rot"],
            [1.0] * (cfg["rot_order"] + 1), cfg["gate"])
        phi = r_sm[:, :3]
        for k in range(n_frames):
            R_nom[k] = project_so3(R_nom[k] @ so3_exp(phi[k]))
        rot_cov0 = np.diagonal(r_cov, 0, 1, 2)[:, :3]
    rot_std = np.sqrt(np.clip(rot_cov0, 0, None))

    measured = set(local_of.values())
    return {
        "lo": lo, "hi": hi, "n_frames": n_frames,
        "center": center, "dims": dims, "R": R_nom,
        "center_std": center_std, "dim_std": dim_std, "rot_std": rot_std,
        "measured": measured, "local_of": local_of,
        "gated": {"center": c_gated, "dims": d_gated, "rotation": r_gated},
        "anchor_frame": recs[anchor_local]["frame_index"],
        "weights": weights,
        "R_canon": R_canon, "d_canon": d_canon,
        "recs": recs,
    }


# ===========================================================================
# Geometry / output assembly
# ===========================================================================
def box_corners(center, R, dims):
    """8 world-frame corners of an oriented box (Step-4 corner ordering)."""
    local = _OBB_SIGNS * (np.asarray(dims, dtype=np.float64) / 2.0)
    return (R @ local.T).T + np.asarray(center, dtype=np.float64)


def smoothness_metrics(res):
    """Compare jitter of the smoothed track vs the raw Step-4 track.

    Reports residual (how far the smoother moved the measured frames -- the
    denoising amount) and second-difference magnitude (discrete acceleration --
    lower = smoother), both evaluated on the measured frames for a fair compare.
    """
    recs = res["recs"]
    local = sorted(res["local_of"].values())
    raw_c = np.array([r["center"] for r in recs])
    raw_d = np.array([r["dims"] for r in recs])
    raw_R = res["R_canon"]
    sm_c = res["center"][local]
    sm_d = res["dims"][local]
    sm_R = [res["R"][k] for k in local]

    def accel(arr):
        if len(arr) < 3:
            return 0.0
        d2 = arr[2:] - 2.0 * arr[1:-1] + arr[:-2]
        return float(np.mean(np.linalg.norm(d2, axis=1)))

    def rot_step(Rs):
        if len(Rs) < 2:
            return 0.0
        return float(np.mean([geodesic_angle(Rs[i], Rs[i + 1])
                              for i in range(len(Rs) - 1)]))

    def rot_accel(Rs):
        if len(Rs) < 3:
            return 0.0
        steps = [so3_log(Rs[i].T @ Rs[i + 1]) for i in range(len(Rs) - 1)]
        d = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
        return float(np.mean([np.linalg.norm(x) for x in d]))

    return {
        "center_residual_rms_m": float(np.sqrt(np.mean(
            np.sum((sm_c - raw_c) ** 2, axis=1)))),
        "center_accel_raw_m": accel(raw_c),
        "center_accel_smoothed_m": accel(sm_c),
        "dims_residual_rms_m": float(np.sqrt(np.mean(
            np.sum((sm_d - raw_d) ** 2, axis=1)))),
        "dims_accel_raw_m": accel(raw_d),
        "dims_accel_smoothed_m": accel(sm_d),
        "rotation_residual_mean_deg": float(np.degrees(np.mean(
            [geodesic_angle(sm_R[i], raw_R[i]) for i in range(len(sm_R))]))),
        "rotation_step_raw_deg": float(np.degrees(rot_step(raw_R))),
        "rotation_step_smoothed_deg": float(np.degrees(rot_step(sm_R))),
        "rotation_accel_raw_deg": float(np.degrees(rot_accel(raw_R))),
        "rotation_accel_smoothed_deg": float(np.degrees(rot_accel(sm_R))),
    }


def build_records(res, cam):
    """Assemble the dense per-frame output records (world + camera frame)."""
    frames = []
    raw_by_frame = {r["frame_index"]: r for r in res["recs"]}
    for k in range(res["n_frames"]):
        fi = res["lo"] + k
        center, R, dims = res["center"][k], res["R"][k], res["dims"][k]
        corners = box_corners(center, R, dims)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = center
        rec = {
            "frame_index": fi,
            "has_measurement": k in res["measured"],
            "gated": {b: (k in res["gated"][b]) for b in res["gated"]},
            "center_world": center.tolist(),
            "dims": dims.tolist(),
            "R_world": R.tolist(),
            "quat_world_wxyz": R_to_quat(R).tolist(),
            "corners_world": corners.tolist(),
            "T_box_in_world": T.tolist(),
            "center_std": res["center_std"][k].tolist(),
            "dim_std": res["dim_std"][k].tolist(),
            "rot_std": res["rot_std"][k].tolist(),
        }
        # Camera-frame mirror, where extrinsics are available for this frame.
        cinfo = cam.get(fi) if cam else None
        if cinfo is None and fi in raw_by_frame and \
                raw_by_frame[fi]["camera_to_world"] is not None:
            cinfo = {"w2c": np.linalg.inv(raw_by_frame[fi]["camera_to_world"]),
                     "K": raw_by_frame[fi]["intrinsics"]}
        if cinfo is not None:
            w2c = cinfo["w2c"]
            T_cam = w2c @ T
            corners_h = np.concatenate(
                [corners, np.ones((8, 1))], axis=1)
            rec["T_box_in_cam"] = T_cam.tolist()
            rec["R_cam"] = T_cam[:3, :3].tolist()
            rec["center_cam"] = T_cam[:3, 3].tolist()
            rec["corners_cam"] = (corners_h @ w2c.T)[:, :3].tolist()
            rec["intrinsics"] = (cinfo["K"].tolist()
                                 if cinfo.get("K") is not None else None)
        else:
            for key in ("T_box_in_cam", "R_cam", "center_cam",
                        "corners_cam", "intrinsics"):
                rec[key] = None
        frames.append(rec)
    return frames


def write_outputs(out_dir, video_name, res, frames, cfg, metrics,
                  step4_dir, step1_dir, smoother_sec, output_fps, frame_stride):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = res["n_frames"]

    meta = {
        "step": "step5_kalman_smoother",
        "track": "A (classical Kalman / RTS smoother)",
        "generated_by": "scripts/step5_kalman_smoother.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video_name": video_name,
        "step4_dir": str(step4_dir.resolve()),
        "step1_dir": str(step1_dir.resolve()) if step1_dir else None,
        "frame_index_range": [res["lo"], res["hi"]],
        "num_frames": n,
        "num_measured": len(res["measured"]),
        "anchor_frame": res["anchor_frame"],
        "frame_stride": frame_stride,
        "output_fps": output_fps,
        "smoother_sec": round(smoother_sec, 4),
        "config": cfg,
        "conventions": {
            "world_frame": "Step-1 world frame (approx. frame-0 camera); "
                           "smoothing is done here",
            "camera_frame": "OpenCV axes (x-right, y-down, z-forward), meters",
            "dims": "full oriented-box extents (meters), not half-extents",
            "rotation": "R_world maps box-local axes to world; quat is (w,x,y,z)",
            "corners": "(8,3), ordering matches Step 4's boxes_world.npy",
            "T_box_in_world": "4x4 box-local (centered, axis-aligned) -> world",
            "has_measurement": "True if Step 4 had a box at this frame; "
                               "others are KF predict-only (densified)",
            "gated": "per-block: measurement rejected by Mahalanobis gating",
            "*_std": "per-frame 1-sigma uncertainty from the RTS covariance",
        },
        "smoothness_metrics": metrics,
        "frames": frames,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Flat arrays for evaluation (against GT 3D box tracks, e.g. CA-1M).
    corners = np.array([f["corners_world"] for f in frames], dtype=np.float32)
    np.savez(
        out_dir / "trajectory.npz",
        frame_index=np.array([f["frame_index"] for f in frames], dtype=np.int32),
        center=res["center"].astype(np.float32),
        dims=res["dims"].astype(np.float32),
        quat_wxyz=np.array([f["quat_world_wxyz"] for f in frames],
                           dtype=np.float32),
        R=res["R"].astype(np.float32),
        corners_world=corners,
        has_measurement=np.array([f["has_measurement"] for f in frames],
                                 dtype=bool),
        center_std=res["center_std"].astype(np.float32),
        dim_std=res["dim_std"].astype(np.float32),
        rot_std=res["rot_std"].astype(np.float32),
        measured_frame_index=np.array([r["frame_index"] for r in res["recs"]],
                                      dtype=np.int32),
        measured_center_raw=np.array([r["center"] for r in res["recs"]],
                                     dtype=np.float32),
        measured_dims_canon=np.array(res["d_canon"], dtype=np.float32),
        measured_R_canon=np.array(res["R_canon"], dtype=np.float32),
    )
    np.save(out_dir / "boxes_world.npy", corners)
    log(f"wrote meta.json + trajectory.npz + boxes_world.npy "
        f"({n} frames) to {out_dir}")


# ===========================================================================
# Optional QA visualization
# ===========================================================================
def render_viz(out_dir, step1_dir, frames, cam, res, output_fps):
    """Overlay the smoothed box (green) vs the raw Step-4 box (red).

    Written at the source video's native fps -- the smoother densifies to
    every original frame, so viz.mp4 plays at real-time speed.
    """
    try:
        import cv2
    except ImportError:
        log("WARNING: cv2 unavailable; skipping --save_viz")
        return
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    raw_by_frame = {r["frame_index"]: r for r in res["recs"]}

    def project(pts_cam, K):
        z = np.clip(pts_cam[:, 2], 1e-6, None)
        u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
        return np.stack([u, v], axis=1), pts_cam[:, 2] > 1e-6

    def draw(img, corners_cam, K, color):
        uv, valid = project(corners_cam, K)
        for a, b in _OBB_EDGES:
            if valid[a] and valid[b]:
                cv2.line(img, tuple(np.round(uv[a]).astype(int)),
                         tuple(np.round(uv[b]).astype(int)), color, 2,
                         cv2.LINE_AA)

    paths = []
    for f in frames:
        fi = f["frame_index"]
        cinfo = cam.get(fi) if cam else None
        if cinfo is None or "step1_index" not in cinfo:
            continue
        img_p = step1_dir / "frames" / f"{cinfo['step1_index']:06d}.jpg"
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        K, w2c = cinfo["K"], cinfo["w2c"]
        sm_corners = np.asarray(f["corners_world"])
        sm_cam = (np.concatenate([sm_corners, np.ones((8, 1))], 1) @ w2c.T)[:, :3]
        draw(img, sm_cam, K, (0, 255, 0))                 # smoothed = green
        if fi in raw_by_frame and raw_by_frame[fi]["corners_world"] is not None:
            raw_cam = (np.concatenate(
                [raw_by_frame[fi]["corners_world"], np.ones((8, 1))], 1)
                @ w2c.T)[:, :3]
            draw(img, raw_cam, K, (0, 0, 255))            # raw Step-4 = red
        tag = "measured" if f["has_measurement"] else "predicted"
        cv2.putText(img, f"{fi} {tag}  green=smoothed red=raw", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
        p = viz_dir / f"{fi:06d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)

    if not paths:
        log("WARNING: --save_viz produced no frames (no Step-1 frames matched)")
        return
    h, w = cv2.imread(str(paths[0])).shape[:2]
    vw = cv2.VideoWriter(str(out_dir / "viz.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         max(1.0, output_fps), (w, h))
    for p in paths:
        vw.write(cv2.imread(str(p)))
    vw.release()
    log(f"wrote viz.mp4 ({len(paths)} frames)")


# ===========================================================================
# CLI
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step4_dir", required=True, type=Path,
                        help="Step 4 output dir (output/step4/<video>)")
    parser.add_argument("--step1_dir", type=Path, default=None,
                        help="Step 1 output dir; enables dense camera-frame "
                             "output + --save_viz (core smoother needs only "
                             "Step 4)")
    parser.add_argument("--output_dir", type=Path,
                        default=REPO_ROOT / "output" / "step5_kalman",
                        help="Parent output dir; results -> <output_dir>/<video>")
    # --- motion-model selection (the representations to compare) ----------
    parser.add_argument("--center_order", type=int, default=1, choices=[1, 2],
                        help="Center model: 1=constant-velocity, "
                             "2=constant-acceleration")
    parser.add_argument("--rot_order", type=int, default=1, choices=[1, 2],
                        help="Rotation model: 1=constant-angular-velocity, "
                             "2=constant-angular-acceleration")
    parser.add_argument("--dims_mode", choices=["random_walk", "constant"],
                        default="random_walk",
                        help="Dimensions: per-frame random walk, or one "
                             "constant size for the whole trajectory")
    parser.add_argument("--rot_outer_iters", type=int, default=3,
                        help="Iterated error-state relinearizations for SO(3)")
    # --- process noise (model trust) --------------------------------------
    # Defaults are tuned on the dog sample for clear denoising without
    # over-smoothing; the principled tuning target is CA-1M GT (design doc S7).
    parser.add_argument("--q_center", type=float, default=5e-4,
                        help="Center acceleration process-noise PSD")
    parser.add_argument("--q_dim", type=float, default=1e-3,
                        help="Log-dimension random-walk process noise")
    parser.add_argument("--q_rot", type=float, default=1.5e-4,
                        help="Rotation angular-velocity process-noise PSD")
    # --- measurement noise (per frame, scaled by confidence) --------------
    parser.add_argument("--r_center_xy", type=float, default=0.06,
                        help="Center image-plane measurement std (m)")
    parser.add_argument("--r_center_z", type=float, default=0.15,
                        help="Center depth-direction measurement std (m)")
    parser.add_argument("--r_dim", type=float, default=0.20,
                        help="Log-dimension measurement std")
    parser.add_argument("--r_rot", type=float, default=0.22,
                        help="Rotation measurement std (radians)")
    parser.add_argument("--gate", type=float, default=16.0,
                        help="Mahalanobis^2 outlier-rejection threshold "
                             "(3 dof; 0 disables)")
    parser.add_argument("--save_viz", action="store_true",
                        help="Render a smoothed-vs-raw projected-box video")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing output dir")
    args = parser.parse_args()

    step4_dir = args.step4_dir.resolve()
    if not (step4_dir / "meta.json").is_file():
        sys.exit(f"FATAL: no Step 4 meta.json at {step4_dir}")
    step1_dir = args.step1_dir.resolve() if args.step1_dir else None

    step4_meta, recs = load_step4(step4_dir)
    video_name = step4_meta.get("video_name") or step4_dir.name
    # The smoother densifies to every original video frame, so the output
    # plays at the source video's NATIVE rate -- not Step 3/4's subsampled
    # target_fps. native_fps = target_fps * frame_stride.
    target_fps = float(step4_meta.get("target_fps") or 5.0)
    frame_stride = int(step4_meta.get("frame_stride") or 1)
    output_fps = target_fps * frame_stride
    cam = load_step1(step1_dir) if step1_dir else None

    out_dir = (args.output_dir / video_name).resolve()
    if (out_dir / "meta.json").exists() and not args.overwrite:
        sys.exit(f"{out_dir}/meta.json exists -- pass --overwrite to redo.")

    cfg = {
        "center_order": args.center_order, "rot_order": args.rot_order,
        "dims_mode": args.dims_mode, "rot_outer_iters": args.rot_outer_iters,
        "q_center": args.q_center, "q_dim": args.q_dim, "q_rot": args.q_rot,
        "r_center_xy": args.r_center_xy, "r_center_z": args.r_center_z,
        "r_dim": args.r_dim, "r_rot": args.r_rot, "gate": args.gate,
    }

    log(f"{video_name}: {len(recs)} Step-4 boxes, "
        f"frames {recs[0]['frame_index']}..{recs[-1]['frame_index']}")
    # Time the smoother proper (canonicalize + forward KF + backward RTS),
    # separately from output assembly / disk I/O.
    t_smooth = time.time()
    res = smooth_trajectory(recs, cfg)
    smoother_sec = time.time() - t_smooth
    frames = build_records(res, cam)
    metrics = smoothness_metrics(res)
    write_outputs(out_dir, video_name, res, frames, cfg, metrics,
                  step4_dir, step1_dir, smoother_sec, output_fps, frame_stride)

    g = res["gated"]
    log(f"Kalman/RTS smoother: {smoother_sec * 1e3:.1f} ms "
        f"({len(recs)} boxes -> {res['n_frames']} dense frames @ "
        f"{output_fps:g} fps); gated "
        f"center={len(g['center'])} dims={len(g['dims'])} "
        f"rotation={len(g['rotation'])}")
    log("smoothness (measured frames, raw -> smoothed):")
    log(f"  center accel : {metrics['center_accel_raw_m']:.4f} -> "
        f"{metrics['center_accel_smoothed_m']:.4f} m  "
        f"(residual {metrics['center_residual_rms_m']:.4f} m)")
    log(f"  dims   accel : {metrics['dims_accel_raw_m']:.4f} -> "
        f"{metrics['dims_accel_smoothed_m']:.4f} m  "
        f"(residual {metrics['dims_residual_rms_m']:.4f} m)")
    log(f"  rot    accel : {metrics['rotation_accel_raw_deg']:.3f} -> "
        f"{metrics['rotation_accel_smoothed_deg']:.3f} deg  "
        f"(residual {metrics['rotation_residual_mean_deg']:.3f} deg)")

    if args.save_viz:
        if cam is None:
            log("WARNING: --save_viz needs --step1_dir; skipping")
        else:
            render_viz(out_dir, step1_dir, frames, cam, res, output_fps)
    log("Step 5 (Track A) complete.")


if __name__ == "__main__":
    main()
