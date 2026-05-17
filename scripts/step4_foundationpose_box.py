#!/usr/bin/env python3
"""Step 4: accurate per-frame 3D bounding box via FoundationPose.

Final step of the generalized, in-the-wild, open-vocabulary video object
tracking pipeline. Like the other steps it is fully generic -- it works on ANY
RGB video and contains no hardcoded data paths, environment paths, or
dataset-specific assumptions; everything is CLI-driven.

What it does
------------
Consumes the outputs of all three previous steps -- Step 1 (per-frame metric
depth + camera intrinsics/extrinsics), Step 2 (per-frame 2D object mask), and
Step 3 (one object mesh PER subsampled frame) -- and runs FoundationPose to
register/track the object in each subsampled frame, producing an accurate 3D
bounding box for that frame.

A SAM3D mesh from Step 3 already carries a placement, so its native oriented
box could be used directly; empirically FoundationPose produces a substantially
more accurate box, which is what this step computes.

Per-frame meshes
----------------
Step 3 emits a *different* mesh for every subsampled frame (the target may be
non-rigid). FoundationPose handles this naturally: ``register`` is a fully
per-frame, global pose search -- it takes one mesh + one RGBD frame + one mask
and has zero cross-frame coupling, so a different mesh every frame is fine.
``track_one`` is a small-motion refiner seeded by the previous frame's pose;
it too uses whatever mesh is currently loaded.

Algorithm (single monocular view)
----------------------------------
The meshed frames are split into contiguous segments (a gap in Step 2's mask
track -> a new segment). Within each segment:
  * REGISTER on the highest-confidence frame (max Step 3 ``sam3_score``) -- a
    global FoundationPose pose search.
  * TRACK bidirectionally outward from that anchor with ``track_one``, each
    frame seeded by the neighbouring frame's FoundationPose pose.

Because FoundationPose works in *camera* coordinates and the camera moves
between (subsampled) frames, the previous frame's pose is transported through
the world frame using Step 1's per-frame camera extrinsics before it seeds the
next ``track_one`` -- otherwise the seed is wrong by the inter-frame camera
motion. ``track_one`` is a small-motion refiner and cannot recover from a bad
seed, so a quality gate (projected-box vs Step 2 mask IoU) falls back to a
fresh ``register`` for any frame where tracking drifts.

Output  (written to ``<output_dir>/<video_name>/``)
---------------------------------------------------
    meta.json          Run metadata + conventions + a per-frame record list
                       (``frames``). Each record carries the source / Step 1 /
                       Step 2 frame indices, the method used (register / track
                       / track->reregister), the 6-DoF object pose in camera
                       and world frames, and the 3D box (oriented extents, the
                       box->camera and box->world transforms, and the 8 corner
                       points in both frames).
    boxes_world.npy    (N, 8, 3) float32 -- the 8 box corners in world
                       coordinates per processed frame (NaN where Step 1 had no
                       camera pose). Convenience mirror of meta.json.
    viz/ + viz.mp4     Projected-box overlay per frame + assembled video
                       (only with --save_viz). QA aid.

Conventions
-----------
* Boxes/poses are keyed by ORIGINAL video frame index (``source_frame_index``),
  matching Steps 1-3.
* Camera frame is OpenCV axes (x-right, y-down, z-forward), metric meters --
  consistent with Step 1's depth and Step 3's meshes.
* ``camera_to_world`` (per frame, copied from Step 1 via Step 3) lifts a
  camera-frame quantity to the world frame; it may be null if Step 1 lacked
  poses, in which case only the camera-frame box is emitted.

Environment
-----------
Run with the ``foundationpose`` conda env's Python -- it carries the
torch / nvdiffrast / warp / pytorch3d stack FoundationPose needs:

    /path/to/envs/foundationpose/bin/python scripts/step4_foundationpose_box.py \\
        --step1_dir output/step1/dog-example \\
        --step2_dir output/step2/dog-example \\
        --step3_dir output/step3/dog-example

The script self-configures CUDA_HOME / LD_LIBRARY_PATH / warp+torch caches from
the running interpreter's conda prefix (re-exec'ing once if needed), so no
shell wrapper is required. ``--fp_repo`` (the pinned third_party/FoundationPose
submodule) is put on sys.path; its ``mycpp`` C++ extension must already be
built (third_party/FoundationPose/mycpp/build/) -- see third_party/patches/
README or build with FoundationPose's build_all_conda.sh.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_env() -> None:
    """Derive CUDA/lib env from the running interpreter's conda prefix.

    FoundationPose's GPU deps (nvdiffrast, warp) need CUDA_HOME and the env's
    libs reachable by the dynamic loader. LD_LIBRARY_PATH is only honoured if
    set before the process starts, so we re-exec once after fixing it. The
    conda prefix is *derived* from sys.executable -- no path is hardcoded, so
    this works for whatever foundationpose env the user runs us with.
    """
    if os.environ.get("_STEP4_ENV_READY"):
        return
    env_prefix = Path(sys.executable).resolve().parents[1]
    env = dict(os.environ)
    env["_STEP4_ENV_READY"] = "1"
    changed = False
    if not env.get("CUDA_HOME"):
        env["CUDA_HOME"] = str(env_prefix)
        changed = True
    lib = str(env_prefix / "lib")
    if lib not in env.get("LD_LIBRARY_PATH", "").split(":"):
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = lib + (":" + prev if prev else "")
        changed = True
    binp = str(env_prefix / "bin")
    if binp not in env.get("PATH", "").split(":"):
        env["PATH"] = binp + ":" + env.get("PATH", "")
        changed = True
    env.setdefault("WARP_CACHE_PATH", str(REPO_ROOT / ".cache" / "warp"))
    env.setdefault("TORCH_EXTENSIONS_DIR", str(REPO_ROOT / ".cache" / "torch_extensions"))
    Path(env["WARP_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
    Path(env["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)
    if changed:
        os.execve(sys.executable, [sys.executable] + sys.argv, env)
    else:
        os.environ.update(env)


_bootstrap_env()

# --- heavy imports happen only after the env is in place ---------------------
import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pycocotools.mask as mask_utils  # noqa: E402
import trimesh  # noqa: E402

# FoundationPose symbols -- populated by _import_foundationpose().
torch = None
dr = None
FoundationPose = None
ScorePredictor = None
PoseRefinePredictor = None
set_seed = None

DEFAULT_FP_REPO = REPO_ROOT / "third_party" / "FoundationPose"

# 8 OBB corners in the box-local (centered, axis-aligned) frame. Ordered so the
# 12 edges below connect corners differing in exactly one axis.
_OBB_SIGNS = np.array([
    [+1, +1, +1], [+1, +1, -1], [+1, -1, -1], [+1, -1, +1],
    [-1, +1, +1], [-1, +1, -1], [-1, -1, -1], [-1, -1, +1],
], dtype=np.float64)
_OBB_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# FoundationPose import / wrapper
# ---------------------------------------------------------------------------
def _import_foundationpose(fp_repo: Path, verbose: bool) -> None:
    """Put the FoundationPose submodule on sys.path and import its API."""
    global torch, dr, FoundationPose, ScorePredictor, PoseRefinePredictor, set_seed
    fp_repo = fp_repo.resolve()
    if not (fp_repo / "estimater.py").is_file():
        sys.exit(f"FATAL: FoundationPose repo not found at {fp_repo} "
                 f"(expected estimater.py). Pass --fp_repo or init the submodule.")
    mycpp_build = fp_repo / "mycpp" / "build"
    if not any(mycpp_build.glob("mycpp*.so")):
        sys.exit(f"FATAL: FoundationPose mycpp extension not built at {mycpp_build}. "
                 f"Build it against this conda env (see build_all_conda.sh).")
    sys.path.insert(0, str(mycpp_build))
    sys.path.insert(0, str(fp_repo))
    import torch as _torch
    import nvdiffrast.torch as _dr
    from estimater import (FoundationPose as _FP, ScorePredictor as _SP,
                           PoseRefinePredictor as _PRP, set_seed as _ss)
    torch, dr = _torch, _dr
    FoundationPose, ScorePredictor, PoseRefinePredictor, set_seed = _FP, _SP, _PRP, _ss
    if not verbose:
        # FoundationPose logs verbosely at INFO on every refine/score call.
        logging.getLogger().setLevel(logging.ERROR)


class FPTracker:
    """One persistent FoundationPose instance; swap meshes via reset()."""

    def __init__(self, debug: int, debug_dir: Path):
        debug_dir.mkdir(parents=True, exist_ok=True)
        log("Initialising FoundationPose (ScorePredictor + PoseRefinePredictor)...")
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        placeholder = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        self.est = FoundationPose(
            model_pts=placeholder.vertices,
            model_normals=placeholder.vertex_normals,
            mesh=placeholder,
            scorer=self.scorer,
            refiner=self.refiner,
            glctx=self.glctx,
            debug=debug,
            debug_dir=str(debug_dir),
        )
        log("FoundationPose ready")

    def reset(self, mesh: trimesh.Trimesh) -> None:
        self.est.reset_object(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
        )

    def register(self, K, rgb, depth, mask, iteration: int) -> np.ndarray:
        """Global pose search. Returns T_obj_in_cam wrt the (uncentered) mesh."""
        return self.est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                  iteration=iteration)

    def track(self, K, rgb, depth, iteration: int) -> np.ndarray:
        """Small-motion refine of pose_last. Returns T_obj_in_cam wrt the mesh."""
        return self.est.track_one(rgb=rgb, depth=depth, K=K, iteration=iteration)

    @property
    def pose_last_np(self) -> np.ndarray:
        """The centered-mesh pose FoundationPose carries for tracking."""
        return self.est.pose_last.detach().cpu().numpy().astype(np.float64)

    def seed_pose_last(self, pose_last_np: np.ndarray) -> None:
        self.est.pose_last = torch.as_tensor(
            pose_last_np, device="cuda", dtype=torch.float32)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _homog(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return np.concatenate([pts, np.ones((len(pts), 1))], axis=1)


def project(pts_cam: np.ndarray, K: np.ndarray):
    """Project camera-frame points to pixels. Returns (uv (N,2), valid (N,))."""
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    z = pts_cam[:, 2]
    valid = z > 1e-6
    safe_z = np.where(valid, z, 1.0)
    x = (K[0, 0] * pts_cam[:, 0] + K[0, 2] * z) / safe_z
    y = (K[1, 1] * pts_cam[:, 1] + K[1, 2] * z) / safe_z
    return np.stack([x, y], axis=1), valid


def bbox_iou(a, b) -> float:
    """IoU of two axis-aligned boxes [x0, y0, x1, y1]."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def proj_box_mask_iou(corners_cam: np.ndarray, K: np.ndarray, mask: np.ndarray) -> float:
    """IoU between the projected 3D box's image bbox and the 2D mask's bbox.

    A drifted track places the box in the wrong image region -> low IoU; this
    is the signal that triggers a re-register fallback.
    """
    uv, valid = project(corners_cam, K)
    if valid.sum() < 4:  # most of the box is behind the camera -> bad pose
        return 0.0
    uvv = uv[valid]
    box_bb = [uvv[:, 0].min(), uvv[:, 1].min(), uvv[:, 0].max(), uvv[:, 1].max()]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    mask_bb = [xs.min(), ys.min(), xs.max(), ys.max()]
    return bbox_iou(box_bb, mask_bb)


def box_from_pose(pose_obj_in_cam: np.ndarray, to_origin: np.ndarray,
                  extents: np.ndarray, c2w):
    """Build the 3D box record from an object pose and the mesh OBB.

    `to_origin`/`extents` come from trimesh.bounds.oriented_bounds(mesh):
    `to_origin` maps mesh coords -> OBB-aligned centered coords. The box ->
    camera transform is therefore pose_obj_in_cam @ inv(to_origin).
    """
    pose_obj_in_cam = np.asarray(pose_obj_in_cam, dtype=np.float64)
    inv_to_origin = np.linalg.inv(to_origin)
    T_box_in_cam = pose_obj_in_cam @ inv_to_origin
    corners_local = _OBB_SIGNS * (np.asarray(extents, dtype=np.float64) / 2.0)
    corners_cam = (T_box_in_cam @ _homog(corners_local).T).T[:, :3]
    rec = {
        "obb_extents": [float(x) for x in extents],
        "obb_to_origin": to_origin.tolist(),
        "size": [float(x) for x in extents],
        "T_box_in_cam": T_box_in_cam.tolist(),
        "R_cam": T_box_in_cam[:3, :3].tolist(),
        "center_cam": [float(x) for x in T_box_in_cam[:3, 3]],
        "corners_cam": corners_cam.tolist(),
    }
    if c2w is not None:
        T_box_in_world = c2w @ T_box_in_cam
        corners_world = (c2w @ _homog(corners_cam).T).T[:, :3]
        rec.update({
            "T_box_in_world": T_box_in_world.tolist(),
            "R_world": T_box_in_world[:3, :3].tolist(),
            "center_world": [float(x) for x in T_box_in_world[:3, 3]],
            "corners_world": corners_world.tolist(),
        })
    else:
        rec.update({"T_box_in_world": None, "R_world": None,
                    "center_world": None, "corners_world": None})
    return rec, corners_cam


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def load_rgb(step1_dir: Path, idx: int) -> np.ndarray:
    """Load a Step 1 frame as an RGB uint8 array (FoundationPose expects RGB)."""
    p = step1_dir / "frames" / f"{idx:06d}.jpg"
    bgr = cv2.imread(str(p))
    if bgr is None:
        raise FileNotFoundError(f"Step 1 RGB frame missing: {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth(step1_dir: Path, idx: int) -> np.ndarray:
    """Load Step 1 metric depth (float32 meters, invalid = 0.0)."""
    p = step1_dir / "depth" / f"{idx:06d}.npy"
    if not p.is_file():
        raise FileNotFoundError(f"Step 1 depth missing: {p}")
    return np.load(p).astype(np.float32)


def decode_mask(rle: dict, target_hw) -> np.ndarray:
    """Decode a pycocotools RLE to a uint8 mask, resized to target (H, W)."""
    m = mask_utils.decode(rle).astype(np.uint8)
    if m.shape != tuple(target_hw):
        m = cv2.resize(m, (target_hw[1], target_hw[0]),
                       interpolation=cv2.INTER_NEAREST)
    return m


# ---------------------------------------------------------------------------
# Segmentation of the meshed-frame sequence
# ---------------------------------------------------------------------------
def split_segments(frames: list, frame_stride: int) -> list:
    """Group meshed frames into contiguous runs; a gap in the mask track (a
    jump larger than ~1.5x the subsample stride) starts a new segment."""
    if not frames:
        return []
    gap = max(2, int(round(1.5 * max(1, frame_stride))))
    segments, cur = [], [0]
    for i in range(1, len(frames)):
        step = frames[i]["source_frame_index"] - frames[i - 1]["source_frame_index"]
        if step > gap:
            segments.append(cur)
            cur = [i]
        else:
            cur.append(i)
    segments.append(cur)
    return segments


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------
def process_video(args) -> int:
    step1_dir = args.step1_dir.resolve()
    step2_dir = args.step2_dir.resolve()
    step3_dir = args.step3_dir.resolve()
    for d, name in [(step1_dir, "step1"), (step2_dir, "step2"), (step3_dir, "step3")]:
        if not d.is_dir():
            sys.exit(f"FATAL: --{name}_dir does not exist: {d}")

    step3_meta = json.loads((step3_dir / "meta.json").read_text())
    video_name = step3_meta.get("video_name") or step3_dir.name
    frame_stride = int(step3_meta.get("frame_stride") or 1)
    target_fps = float(step3_meta.get("target_fps") or 5.0)

    out_dir = (args.output_dir / video_name).resolve()
    if (out_dir / "meta.json").exists() and not args.overwrite:
        sys.exit(f"{out_dir}/meta.json already exists -- pass --overwrite to redo.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 2 masks (keyed by frame-index string).
    masks_rle = json.loads((step2_dir / "masks_rle.json").read_text())

    # Meshed frames = Step 3 records with status 'ok', ordered in time.
    meshed = [f for f in step3_meta["frames"] if f.get("status") == "ok"]
    meshed.sort(key=lambda f: f["source_frame_index"])
    if args.max_frames > 0:
        meshed = meshed[:args.max_frames]
    if not meshed:
        sys.exit("FATAL: Step 3 produced no meshes -- nothing for Step 4 to do.")
    segments = split_segments(meshed, frame_stride)
    log(f"{video_name}: {len(meshed)} meshed frames in {len(segments)} segment(s)")

    _import_foundationpose(args.fp_repo, args.fp_verbose)
    set_seed(args.seed)
    tracker = FPTracker(debug=args.debug, debug_dir=out_dir / "_fp_debug")

    results: dict = {}     # source_frame_index -> per-frame output record
    pose_last_store: dict = {}  # source_frame_index -> centered-mesh pose (np 4x4)
    counts = {"register": 0, "track": 0, "track->reregister": 0, "error": 0}
    t_start = time.time()

    def load_inputs(rec: dict):
        """Resolve a meshed frame's mesh (+ its OBB) + RGBD + mask + camera."""
        mesh = trimesh.load(str(step3_dir / rec["mesh_file"]), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 4:
            raise RuntimeError(f"invalid mesh {rec['mesh_file']}")
        # Oriented bounding box of this frame's mesh, computed once.
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        rgb = load_rgb(step1_dir, rec["step1_index"])
        depth = load_depth(step1_dir, rec["step1_index"])
        K = np.asarray(rec["intrinsics"], dtype=np.float64)
        c2w = (np.asarray(rec["camera_to_world"], dtype=np.float64)
               if rec.get("camera_to_world") is not None else None)
        mrec = masks_rle.get(str(rec["step2_index"]))
        mask = None
        if mrec and mrec.get("mask_rle"):
            mask = decode_mask(mrec["mask_rle"], depth.shape)
        return mesh, rgb, depth, K, c2w, mask, to_origin, extents

    def run_register(rec, rgb, depth, K, mask):
        if mask is None or int(mask.sum()) == 0:
            raise RuntimeError("register requires a Step 2 mask but none is available")
        return tracker.register(K, rgb, depth, mask, iteration=args.register_iter)

    def emit(rec, segment_id, role, method, pose, K, c2w, mask, to_origin, extents):
        """Build + store the per-frame output record."""
        src = rec["source_frame_index"]
        box, corners_cam = box_from_pose(pose, to_origin, extents, c2w)
        iou = proj_box_mask_iou(corners_cam, K, mask) if mask is not None else None
        T_world = (c2w @ pose).tolist() if c2w is not None else None
        results[src] = {
            **_frame_id(rec),
            "segment": segment_id,
            "role": role,
            "method": method,
            "status": "ok",
            "sam3_score": rec.get("sam3_score"),
            "proj_box_mask_iou": iou,
            "intrinsics": K.tolist(),
            "camera_to_world": c2w.tolist() if c2w is not None else None,
            "T_obj_in_cam": pose.tolist(),
            "T_obj_in_world": T_world,
            "box": box,
        }
        return iou

    def process_frame(rec, segment_id, role, prev_src):
        """Register or track one meshed frame. prev_src=None -> anchor."""
        src = rec["source_frame_index"]
        try:
            mesh, rgb, depth, K, c2w, mask, to_origin, extents = load_inputs(rec)
        except Exception as e:  # noqa: BLE001
            results[src] = {**_frame_id(rec), "segment": segment_id, "role": role,
                            "method": "error", "status": "error", "error": str(e)}
            counts["error"] += 1
            log(f"  frame {src}: ERROR (inputs) {e}")
            return
        tracker.reset(mesh)

        # Decide whether we can chain a track from the neighbour.
        prev_c2w = None
        can_track = False
        if prev_src is not None and prev_src in pose_last_store:
            prev_rec = next(f for f in meshed if f["source_frame_index"] == prev_src)
            prev_c2w = (np.asarray(prev_rec["camera_to_world"], dtype=np.float64)
                        if prev_rec.get("camera_to_world") is not None else None)
            # Chaining the camera-frame pose needs both camera poses, to
            # transport it through the world frame across the camera's motion.
            can_track = prev_c2w is not None and c2w is not None

        method = None
        pose = None
        try:
            if can_track:
                # Transport the neighbour's centered-mesh pose from its camera
                # frame into this frame's camera frame: cam_prev -> world -> cam.
                T_rel = np.linalg.inv(c2w) @ prev_c2w
                tracker.seed_pose_last(T_rel @ pose_last_store[prev_src])
                pose = np.asarray(tracker.track(K, rgb, depth, iteration=args.track_iter),
                                  dtype=np.float64)
                method = "track"
                # Quality gate -- fall back to a fresh register on drift.
                if not args.no_reregister and mask is not None and int(mask.sum()) > 0:
                    _, corners_cam = box_from_pose(pose, to_origin, extents, c2w)
                    if proj_box_mask_iou(corners_cam, K, mask) < args.reregister_iou:
                        pose = run_register(rec, rgb, depth, K, mask)
                        method = "track->reregister"
            else:
                pose = run_register(rec, rgb, depth, K, mask)
                method = "register"
        except Exception as e:  # noqa: BLE001
            # A track_one failure -> try a clean register before giving up.
            if method == "track":
                try:
                    pose = run_register(rec, rgb, depth, K, mask)
                    method = "track->reregister"
                except Exception as e2:  # noqa: BLE001
                    e = e2
            if pose is None:
                results[src] = {**_frame_id(rec), "segment": segment_id,
                                "role": role, "method": method or "error",
                                "status": "error", "error": str(e)}
                counts["error"] += 1
                log(f"  frame {src}: ERROR {e}")
                return

        pose = np.asarray(pose, dtype=np.float64)
        if not np.all(np.isfinite(pose)):
            results[src] = {**_frame_id(rec), "segment": segment_id, "role": role,
                            "method": method, "status": "error",
                            "error": "non-finite pose"}
            counts["error"] += 1
            log(f"  frame {src}: ERROR non-finite pose")
            return
        pose_last_store[src] = tracker.pose_last_np
        iou = emit(rec, segment_id, role, method, pose, K, c2w, mask, to_origin, extents)
        counts[method if method in counts else "register"] += 1
        iou_s = f"{iou:.2f}" if iou is not None else "n/a"
        log(f"  frame {src}: {method} (proj-IoU {iou_s})")

    # ---- per-segment: register the best frame, track outward both ways ----
    for seg_id, seg in enumerate(segments):
        seg_recs = [meshed[i] for i in seg]
        anchor_i = max(range(len(seg_recs)),
                       key=lambda j: seg_recs[j].get("sam3_score") or 0.0)
        anchor_rec = seg_recs[anchor_i]
        log(f"segment {seg_id}: {len(seg_recs)} frames, "
            f"anchor=frame {anchor_rec['source_frame_index']} "
            f"(sam3_score={anchor_rec.get('sam3_score')})")
        process_frame(anchor_rec, seg_id, "anchor", prev_src=None)
        # Forward: anchor+1 .. end, each chained from the previous frame.
        for j in range(anchor_i + 1, len(seg_recs)):
            process_frame(seg_recs[j], seg_id, "forward",
                          prev_src=seg_recs[j - 1]["source_frame_index"])
        # Backward: anchor-1 .. start, each chained from the following frame.
        for j in range(anchor_i - 1, -1, -1):
            process_frame(seg_recs[j], seg_id, "backward",
                          prev_src=seg_recs[j + 1]["source_frame_index"])
        gc.collect()
        torch.cuda.empty_cache()

    frames_out = [results[f["source_frame_index"]] for f in meshed
                  if f["source_frame_index"] in results]
    elapsed = time.time() - t_start
    write_outputs(args, out_dir, video_name, step1_dir, step2_dir, step3_dir,
                  step3_meta, frame_stride, target_fps, frames_out, counts, elapsed)

    log(f"Done in {elapsed/60:.1f} min: {counts}")
    if args.save_viz:
        render_viz(out_dir, step1_dir, frames_out, target_fps)
    return counts["error"]


def _frame_id(rec: dict) -> dict:
    return {
        "source_frame_index": rec["source_frame_index"],
        "step1_index": rec["step1_index"],
        "step2_index": rec["step2_index"],
        "mesh_file": rec["mesh_file"],
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------
def write_outputs(args, out_dir, video_name, step1_dir, step2_dir, step3_dir,
                  step3_meta, frame_stride, target_fps, frames_out, counts,
                  elapsed) -> None:
    try:
        import subprocess
        fp_rev = subprocess.run(
            ["git", "-C", str(args.fp_repo.resolve()), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        fp_rev = None

    meta = {
        "step": "step4_foundationpose_box",
        "generated_by": "scripts/step4_foundationpose_box.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video_name": video_name,
        "model": "FoundationPose",
        "fp_repo": str(args.fp_repo.resolve()),
        "fp_rev": fp_rev,
        "step1_dir": str(step1_dir),
        "step2_dir": str(step2_dir),
        "step3_dir": str(step3_dir),
        "frame_stride": frame_stride,
        "target_fps": target_fps,
        "register_iter": args.register_iter,
        "track_iter": args.track_iter,
        "reregister_iou": args.reregister_iou,
        "reregister_fallback": not args.no_reregister,
        "seed": args.seed,
        "num_frames": len(frames_out),
        "counts": counts,
        "elapsed_sec": round(elapsed, 1),
        "camera_frame": "OpenCV axes (x-right, y-down, z-forward), metric meters",
        "box": {
            "obb_extents": "oriented bounding box size (meters), from the "
                           "per-frame Step 3 mesh",
            "T_box_in_cam": "4x4 box-local (centered, axis-aligned) -> camera",
            "T_box_in_world": "4x4 box-local -> world (null if no camera pose)",
            "corners_cam/corners_world": "(8,3) corner points; edge order is "
                                         "implicit in the corner ordering",
            "T_obj_in_cam": "FoundationPose 6-DoF object pose, camera frame",
        },
        "frames": frames_out,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Convenience array: (N, 8, 3) world-frame corners (NaN where unavailable).
    boxes_world = np.full((len(frames_out), 8, 3), np.nan, dtype=np.float32)
    for i, fr in enumerate(frames_out):
        cw = fr.get("box", {}).get("corners_world") if fr.get("status") == "ok" else None
        if cw is not None:
            boxes_world[i] = np.asarray(cw, dtype=np.float32)
    np.save(out_dir / "boxes_world.npy", boxes_world)
    log(f"Wrote {out_dir}/meta.json + boxes_world.npy ({len(frames_out)} frames)")


def render_viz(out_dir: Path, step1_dir: Path, frames_out: list,
               target_fps: float) -> None:
    """Draw the projected 3D box on each frame and assemble a QA video."""
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    paths = []
    for fr in frames_out:
        if fr.get("status") != "ok":
            continue
        src = fr["source_frame_index"]
        bgr = cv2.imread(str(step1_dir / "frames" / f"{fr['step1_index']:06d}.jpg"))
        if bgr is None:
            continue
        K = np.asarray(fr["intrinsics"], dtype=np.float64)
        corners_cam = np.asarray(fr["box"]["corners_cam"], dtype=np.float64)
        uv, valid = project(corners_cam, K)
        for a, b in _OBB_EDGES:
            if valid[a] and valid[b]:
                pa = tuple(np.round(uv[a]).astype(int))
                pb = tuple(np.round(uv[b]).astype(int))
                cv2.line(bgr, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)
        label = f"{src} {fr['method']}"
        cv2.putText(bgr, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2, cv2.LINE_AA)
        p = viz_dir / f"{src:06d}.jpg"
        cv2.imwrite(str(p), bgr)
        paths.append(p)
    if not paths:
        return
    first = cv2.imread(str(paths[0]))
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(str(out_dir / "viz.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         max(1.0, target_fps), (w, h))
    for p in paths:
        vw.write(cv2.imread(str(p)))
    vw.release()
    log(f"Wrote {out_dir}/viz.mp4 ({len(paths)} frames)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step1_dir", required=True, type=Path,
                        help="Step 1 output dir (output/step1/<video>)")
    parser.add_argument("--step2_dir", required=True, type=Path,
                        help="Step 2 output dir (output/step2/<video>)")
    parser.add_argument("--step3_dir", required=True, type=Path,
                        help="Step 3 output dir (output/step3/<video>)")
    parser.add_argument("--output_dir", type=Path,
                        default=REPO_ROOT / "output" / "step4",
                        help="Parent output dir; results go to <output_dir>/<video>")
    parser.add_argument("--fp_repo", type=Path, default=DEFAULT_FP_REPO,
                        help="FoundationPose submodule checkout")
    parser.add_argument("--register_iter", type=int, default=5,
                        help="FoundationPose register refinement iterations")
    parser.add_argument("--track_iter", type=int, default=2,
                        help="FoundationPose track_one refinement iterations")
    parser.add_argument("--reregister_iou", type=float, default=0.20,
                        help="If a tracked frame's projected-box vs mask IoU is "
                             "below this, fall back to a fresh register")
    parser.add_argument("--no_reregister", action="store_true",
                        help="Disable the track-drift re-register fallback")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Process only the first N meshed frames (smoke test)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", type=int, default=0,
                        help="FoundationPose debug level (>=2 dumps renders)")
    parser.add_argument("--fp_verbose", action="store_true",
                        help="Keep FoundationPose's verbose INFO logging")
    parser.add_argument("--save_viz", action="store_true",
                        help="Render a projected-box overlay video for QA")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing output dir")
    args = parser.parse_args()

    try:
        n_err = process_video(args)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
