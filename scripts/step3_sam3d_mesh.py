#!/usr/bin/env python3
"""Step 3: per-frame 3D object mesh generation via SAM 3D Objects.

Part of the generalized, in-the-wild, open-vocabulary video object tracking
pipeline. This step is fully generic: it works on ANY RGB video and has no
hardcoded data paths, environment paths, or dataset-specific assumptions --
everything is CLI-driven.

What it does
------------
Consumes the outputs of Step 1 (per-frame metric depth + camera intrinsics)
and Step 2 (per-frame 2D object-mask tracklet) and, for a subsampled set of
frames, runs SAM 3D Objects on (RGB + object mask + metric pointmap + GT
intrinsics) to reconstruct an object mesh for that frame.

Unlike the DROID reference pipeline -- which reconstructed a single mesh per
video because every target was a rigid object manipulated by a robot arm --
this step produces one mesh PER SAMPLED FRAME. The tracked object may be
non-rigid (people, animals, ...), so each frame gets its own mesh; the
downstream Step 4 then fits a tight 3D box to each frame's mesh.

Frame sampling
--------------
Generating a mesh for every frame is wasteful (SAM 3D Objects is ~10-30 s per
frame). Instead we subsample to a target rate (``--fps``, default 5). The
stride over Step 2's frame sequence is ``round(step2_fps / target_fps)``. A
frame is processed only if Step 2 has a (non-null) mask for it AND Step 1 has
depth for the same source video frame; frames missing either are skipped --
this is normal, the target object need not be visible in every frame.

Environment
-----------
Run with the ``sam3d-objects`` conda env's Python -- it carries the torch /
pytorch3d / kaolin / hydra / trimesh stack that SAM 3D Objects needs:

    /path/to/envs/sam3d-objects/bin/python scripts/step3_sam3d_mesh.py \\
        --step1_dir output/step1/dog-example \\
        --step2_dir output/step2/dog-example

``--sam3d_repo`` (the pinned ``third_party/sam-3d-objects`` submodule) is
prepended to ``sys.path`` so its ``sam3d_objects`` package is authoritative.
The submodule needs a small GT-intrinsics patch (third_party/patches/
sam3d-objects-gt-intrinsics.patch); this script auto-applies it if missing.

Output  (written to ``<output_dir>/<video_name>/``)
---------------------------------------------------
    meshes/{src:06d}.glb       CANONICAL artifact, consumed by Step 4. One
                               mesh per processed frame, keyed by the ORIGINAL
                               video frame index. Placed in the camera frame,
                               metric meters, OpenCV axes (x-right, y-down,
                               z-forward) -- consistent with Step 1's depth.
    meshes_raw/{src:06d}.glb   The untransformed SAM 3D Objects mesh
                               (object-local frame, y-up, ~unit AABB), unless
                               --no_raw_mesh. Pair with meta.json's per-frame
                               `sam3d_pose` to re-place it any other way.
    meta.json                  Metadata + conventions + a per-frame record
                               list (`frames`). Each record carries the
                               source / Step 1 / Step 2 frame indices, the
                               SAM 3D pose (translation / rotation / scale),
                               the intrinsics and camera->world pose used, and
                               mesh stats (vertex/face counts, degenerate-mesh
                               flag).
    debug/{src:06d}.jpg        Masked-input preview per frame (only --save_debug).

Conventions
-----------
* Meshes are keyed by ORIGINAL video frame index (not a resequenced 0..N-1),
  so they cross-reference Step 1 / Step 2 (whose meta.json `source_frame_indices`
  also index original video frames) directly.
* The camera-frame mesh sits where the object actually is relative to the
  camera, at metric scale. To lift it to world coordinates apply that frame's
  `camera_to_world` (recorded per frame in meta.json, copied from Step 1).
* A frame with no Step 2 mask, or no Step 1 depth, is simply skipped -- the
  mesh sequence need not cover the whole video.
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_utils

# `Inference` (notebook/inference.py:5) sets CUDA_HOME from CONDA_PREFIX at
# import time. When invoked via a direct interpreter path (not `conda
# activate`) CONDA_PREFIX may be unset -- fall back to this interpreter's
# prefix, which for an env's python is the env directory.
os.environ.setdefault("CONDA_PREFIX", sys.prefix)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM3D_REPO = REPO_ROOT / "third_party" / "sam-3d-objects"
DEFAULT_SAM3D_PATCH = REPO_ROOT / "third_party" / "patches" / "sam3d-objects-gt-intrinsics.patch"
# Marker string the GT-intrinsics patch inserts (used to detect it is applied).
PATCH_MARKER = "GT-intrinsics patch"
CUBE_SPREAD_THRESHOLD = 0.05  # mesh AABB extents-spread below this == degenerate


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [step3] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Coordinate / mesh helpers (ported from the droid pipeline's
# step3b1_singleframe_box.py -- generalised to a single monocular camera)
# ---------------------------------------------------------------------------

def depth_to_pointmap_cv(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Metric depth (H,W, meters) + intrinsics K (3,3) -> (H,W,3) pointmap in
    the OpenCV camera frame (x-right, y-down, z-forward). Invalid depth (<=0)
    maps to a zero point."""
    H, W = depth_m.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = depth_m.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pm = np.stack([x, y, z], axis=-1)
    pm[depth_m <= 0] = 0.0
    return pm


def pointmap_cv_to_pt3d(pm_cv: np.ndarray) -> np.ndarray:
    """OpenCV camera frame (x-right, y-down, z-forward) -> PyTorch3D camera
    frame (x-left, y-up, z-forward), which is what SAM 3D Objects expects."""
    out = pm_cv.copy().astype(np.float32)
    out[..., 0] *= -1.0
    out[..., 1] *= -1.0
    return out


def mesh_extents_spread(verts: np.ndarray) -> float | None:
    """(max - min) / max of the per-axis AABB extents of a vertex array.

    SAM 3D Objects' degenerate "cube fallback" failure produces a near-unit
    cube whose three extents are ~equal -> spread ~0. Healthy outputs are
    anisotropic (spread > 0.05 typically). Returns None when there are too few
    vertices to judge."""
    if verts is None or len(verts) < 4:
        return None
    ext = verts.max(axis=0) - verts.min(axis=0)
    if float(ext.max()) < 1e-6:
        return None
    return float((ext.max() - ext.min()) / ext.max())


def place_mesh_in_camera_frame(verts_yup: np.ndarray, translation, rotation,
                               scale, device: str) -> np.ndarray:
    """Transform SAM 3D Objects mesh vertices (object-local, y-up) into the
    camera frame (OpenCV axes, metric meters).

    SAM 3D's ``to_glb`` rotates vertices z-up -> y-up before packaging, but the
    returned (translation, rotation, scale) pose is defined against the
    PRE-rotation z-up mesh. So we undo that rotation, apply the PyTorch3D pose,
    then flip X/Y to convert PyTorch3D camera axes -> OpenCV camera axes. This
    is exactly the droid pipeline's ``sam3d_output_to_corners_cv`` transform,
    applied to every vertex instead of only the 8 AABB corners."""
    import torch
    from pytorch3d.transforms import Transform3d, quaternion_to_matrix

    # Undo to_glb's z-up -> y-up rotation: it did `verts @ M` with
    # M = [[1,0,0],[0,0,-1],[0,1,0]]; M is orthogonal so M^-1 = M^T.
    m_inv = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0]], dtype=np.float32)
    verts_zup = verts_yup.astype(np.float32) @ m_inv

    t = translation.float().to(device)
    q = rotation.float().to(device)
    s = scale.float().to(device)
    R = quaternion_to_matrix(q)
    tfm = Transform3d(device=device).scale(s).rotate(R).translate(t)

    v = torch.from_numpy(verts_zup).float().to(device).unsqueeze(0)
    v_pt3d = tfm.transform_points(v)[0].detach().cpu().numpy()
    v_cv = v_pt3d.copy()
    v_cv[:, 0] *= -1.0  # PyTorch3D X -> OpenCV X
    v_cv[:, 1] *= -1.0  # PyTorch3D Y -> OpenCV Y
    return v_cv.astype(np.float32)


def to_single_mesh(glb):
    """SAM 3D Objects' ``to_glb`` returns a trimesh.Trimesh; be defensive in
    case a Scene comes back instead."""
    if hasattr(glb, "vertices") and hasattr(glb, "faces"):
        return glb
    if hasattr(glb, "dump"):  # trimesh.Scene
        return glb.dump(concatenate=True)
    raise RuntimeError(f"unexpected SAM 3D Objects mesh type: {type(glb)}")


def tensor_to_list(x) -> list:
    """Flatten a torch tensor / array to a plain Python list for JSON."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=np.float64).reshape(-1).tolist()


# ---------------------------------------------------------------------------
# Step 1 / Step 2 input loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def decode_mask_rle(rle: dict) -> np.ndarray:
    """Decode a Step 2 pycocotools RLE dict to a bool mask. Step 2 stores
    `counts` as a utf-8 string; pycocotools needs bytes."""
    r = dict(rle)
    if isinstance(r.get("counts"), str):
        r["counts"] = r["counts"].encode("utf-8")
    return mask_utils.decode(r).astype(bool)


def load_step1(step1_dir: Path) -> dict:
    """Load Step 1 metadata + camera arrays. Per-frame depth is loaded lazily."""
    meta = load_json(step1_dir / "meta.json")
    intr = np.load(step1_dir / "intrinsics.npy")
    poses_c2w = None
    p = step1_dir / "poses_c2w.npy"
    if p.exists():
        poses_c2w = np.load(p)
    src = meta.get("source_frame_indices")
    if not src:
        raise SystemExit(f"[step3] Step 1 meta.json has no source_frame_indices: {step1_dir}")
    return {
        "meta": meta,
        "dir": step1_dir,
        "intrinsics": intr,
        "poses_c2w": poses_c2w,
        "src_indices": list(src),
        # original video frame index -> Step 1 output index
        "index_by_src": {int(s): i for i, s in enumerate(src)},
    }


def load_step2(step2_dir: Path) -> dict:
    """Load Step 2 metadata + the per-frame mask RLE dict."""
    meta = load_json(step2_dir / "meta.json")
    masks = load_json(step2_dir / "masks_rle.json")
    src = meta.get("source_frame_indices")
    if not src:
        raise SystemExit(f"[step3] Step 2 meta.json has no source_frame_indices: {step2_dir}")
    return {"meta": meta, "dir": step2_dir, "masks": masks, "src_indices": list(src)}


def step1_K(step1: dict, j1: int) -> np.ndarray:
    """Intrinsics (3,3) for Step 1 output index j1 (handles per-frame or shared)."""
    intr = step1["intrinsics"]
    K = intr if intr.ndim == 2 else intr[j1]
    return np.asarray(K, dtype=np.float64)


# ---------------------------------------------------------------------------
# SAM 3D Objects setup
# ---------------------------------------------------------------------------

def ensure_sam3d_patched(sam3d_repo: Path, patch_path: Path) -> None:
    """Make sure the GT-intrinsics patch is applied to the SAM 3D Objects
    submodule. Idempotent: a no-op if already patched.

    Upstream SAM 3D Objects infers camera intrinsics from the pointmap when an
    external pointmap is supplied. The patch lets the caller instead pass
    GROUND-TRUTH intrinsics (Step 1's RADIO-ViPE camera params) and use them
    directly; it falls back to upstream inference when none are given."""
    target = sam3d_repo / "sam3d_objects" / "pipeline" / "inference_pipeline_pointmap.py"
    if not target.exists():
        raise SystemExit(
            f"[step3] SAM 3D Objects source not found under --sam3d_repo:\n"
            f"    {sam3d_repo}\n"
            f"Initialise the submodule:  git submodule update --init {sam3d_repo}")
    if PATCH_MARKER in target.read_text():
        log(f"SAM 3D Objects GT-intrinsics patch already applied ({sam3d_repo.name})")
        return
    if not patch_path.exists():
        raise SystemExit(f"[step3] GT-intrinsics patch file is missing: {patch_path}")
    log(f"applying GT-intrinsics patch to {sam3d_repo.name} ...")
    r = subprocess.run(["git", "apply", str(patch_path)], cwd=str(sam3d_repo),
                        capture_output=True, text=True)
    if r.returncode != 0 or PATCH_MARKER not in target.read_text():
        raise SystemExit(
            f"[step3] Could not apply the GT-intrinsics patch automatically:\n"
            f"    {r.stderr.strip()}\n"
            f"Apply it manually:  git -C {sam3d_repo} apply {patch_path}")
    log("GT-intrinsics patch applied")


def prewarm_checkpoints(ckpt_dir: Path) -> None:
    """Stream the SAM 3D Objects checkpoints through the page cache.

    A cold mmap of these ~12 GB of weights from network storage can stall the
    model load for tens of minutes (the loading process sits in disk-sleep).
    Reading the bytes once first pulls them into the OS page cache, after which
    SAM 3D's mmap reads are served from RAM. Cheap no-op if already cached."""
    files = sorted(p for ext in ("*.ckpt", "*.pt", "*.safetensors")
                   for p in ckpt_dir.glob(ext) if p.stat().st_size > 0)
    if not files:
        log(f"pre-warm: no checkpoint files under {ckpt_dir}")
        return
    total_gb = sum(p.stat().st_size for p in files) / 1e9
    log(f"pre-warming {len(files)} checkpoint file(s) (~{total_gb:.1f} GB) ...")
    t0 = time.time()
    for p in files:
        with open(p, "rb") as fh:
            while fh.read(32 * 1024 * 1024):
                pass
    log(f"pre-warm done ({time.time() - t0:.0f}s)")


def load_sam3d_model(sam3d_repo: Path, config_file: Path):
    """Load the SAM 3D Objects ``Inference`` pipeline.

    Layout post-optimisation is left at the upstream default (off): the
    ``Inference`` wrapper calls the pipeline with ``with_layout_postprocess
    =False``. SAM 3D's predicted layout already places the mesh metrically via
    the GT-intrinsics pointmap; enabling post-optimisation is a possible future
    quality knob (it would need ``Inference.__call__`` to pass the flag)."""
    notebook = sam3d_repo / "notebook"
    for p in (str(sam3d_repo), str(notebook)):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from inference import Inference
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"[step3] Could not import SAM 3D Objects ({type(e).__name__}: {e}).\n"
            f"Run this script with the `sam3d-objects` conda env's Python "
            f"(it provides torch / pytorch3d / kaolin / hydra / trimesh).")
    if not config_file.exists():
        raise SystemExit(
            f"[step3] SAM 3D Objects config not found: {config_file}\n"
            f"Place the checkpoints (incl. pipeline.yaml) there, or point "
            f"--sam3d_checkpoint_dir at an existing checkpoint directory.")
    log(f"loading SAM 3D Objects pipeline from {config_file} ...")
    t0 = time.time()
    model = Inference(config_file=str(config_file), compile=False)
    log(f"SAM 3D Objects ready ({time.time() - t0:.0f}s)")
    return model


def run_sam3d(model, rgb: np.ndarray, mask: np.ndarray, pm_pt3d: np.ndarray,
              K: np.ndarray, device: str, seed: int):
    """Single SAM 3D Objects forward pass for one frame. Returns the output
    dict (keys: glb, translation, rotation, scale) and the elapsed seconds."""
    import torch
    pm_t = torch.from_numpy(pm_pt3d).to(device)
    K_t = torch.from_numpy(K.astype(np.float32)).to(device)
    t0 = time.time()
    out = model(image=rgb, mask=mask.astype(np.uint8), pointmap=pm_t,
                intrinsics=K_t, seed=seed)
    return out, time.time() - t0


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

def prepare_frame(step1: dict, step2: dict, j2: int, max_depth: float):
    """Assemble the SAM 3D Objects inputs for one sampled Step 2 frame.

    Returns a dict on success, or a string reason ('no_mask' / 'no_step1_frame'
    / ...) when the frame should be skipped."""
    src = int(step2["src_indices"][j2])

    entry = step2["masks"].get(str(j2))
    if not entry or not entry.get("mask_rle"):
        return "no_mask"

    j1 = step1["index_by_src"].get(src)
    if j1 is None:
        # Positional fallback: Step 1 and Step 2 may label their per-frame
        # source differently while ordering the same physical frames the
        # same way -- in CA-1M production, Step 1 stores integer frame
        # indices `[0, 1, ..., N-1]` while Step 2 stores ns timestamps
        # `[4144648974458, ...]`. Same frames, same order, different labels.
        # When the two lists are the same length we trust position over
        # label lookup. (Dev paths with matching labels hit the fast
        # `index_by_src` path above and never reach this branch.)
        if len(step1["src_indices"]) == len(step2["src_indices"]):
            j1 = j2
        else:
            return "no_step1_frame"

    depth_path = step1["dir"] / "depth" / f"{j1:06d}.npy"
    rgb_path = step1["dir"] / "frames" / f"{j1:06d}.jpg"
    if not depth_path.exists() or not rgb_path.exists():
        return "no_step1_frame"
    depth = np.load(depth_path).astype(np.float32)
    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        return "rgb_read_fail"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    Hd, Wd = depth.shape
    K = step1_K(step1, j1)

    # Resolve everything onto the depth grid: the pointmap is built from depth,
    # so RGB / mask / K must all describe that same pixel grid.
    if rgb.shape[:2] != (Hd, Wd):
        sx, sy = Wd / rgb.shape[1], Hd / rgb.shape[0]
        K = K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        rgb = cv2.resize(rgb, (Wd, Hd), interpolation=cv2.INTER_AREA)

    mask = decode_mask_rle(entry["mask_rle"])
    if mask.shape != (Hd, Wd):
        mask = cv2.resize(mask.astype(np.uint8), (Wd, Hd),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    if not mask.any():
        return "empty_mask"

    pm_cv = depth_to_pointmap_cv(depth, K)
    if max_depth > 0:
        pm_cv[depth > max_depth] = 0.0

    poses_c2w = step1.get("poses_c2w")
    c2w = poses_c2w[j1].tolist() if poses_c2w is not None else None

    return {
        "src": src, "j1": j1, "j2": j2,
        "rgb": rgb, "mask": mask, "pm_cv": pm_cv, "K": K,
        "camera_to_world": c2w,
        "mask_valid_px": int(mask.sum()),
        "depth_valid_frac": float((depth > 0).mean()),
        "sam3_score": entry.get("sam3_score"),
        "mask_source": entry.get("source"),
    }


def finalize_frame(prep: dict, out: dict, elapsed: float, device: str,
                   meshes_dir: Path, raw_dir: Path | None) -> dict:
    """Place + export the mesh for one frame and build its metadata record.

    Returns a record dict; on a degenerate / missing mesh the record carries
    `status` != 'ok' and no mesh file."""
    src = prep["src"]
    rec = {
        "source_frame_index": src,
        "step1_index": prep["j1"],
        "step2_index": prep["j2"],
        "mesh_file": None,
        "raw_mesh_file": None,
        "sam3d_elapsed_sec": round(elapsed, 2),
        "sam3_score": prep["sam3_score"],
        "mask_source": prep["mask_source"],
        "mask_valid_px": prep["mask_valid_px"],
        "depth_valid_frac": round(prep["depth_valid_frac"], 4),
        "intrinsics": prep["K"].tolist(),
        "camera_to_world": prep["camera_to_world"],
        "status": "ok",
    }

    glb = out.get("glb") if isinstance(out, dict) else None
    if glb is None:
        rec["status"] = "no_mesh"
        return rec
    mesh = to_single_mesh(glb)
    verts_yup = np.asarray(mesh.vertices, dtype=np.float32)
    if len(verts_yup) < 4:
        rec["status"] = "no_mesh"
        return rec

    spread = mesh_extents_spread(verts_yup)
    rec["num_vertices"] = int(len(verts_yup))
    rec["num_faces"] = int(len(mesh.faces))
    rec["mesh_extents_spread"] = spread
    rec["near_cube"] = bool(spread is not None and spread < CUBE_SPREAD_THRESHOLD)
    rec["sam3d_pose"] = {
        "translation": tensor_to_list(out["translation"]),
        "rotation_wxyz": tensor_to_list(out["rotation"]),
        "scale": tensor_to_list(out["scale"]),
        "note": "PyTorch3D similarity transform mapping the z-up object-local "
                "mesh into the PyTorch3D camera frame.",
    }

    # Raw (object-local, y-up) mesh.
    if raw_dir is not None:
        raw_path = raw_dir / f"{src:06d}.glb"
        try:
            mesh.export(str(raw_path))
            rec["raw_mesh_file"] = f"meshes_raw/{raw_path.name}"
        except Exception as e:  # noqa: BLE001
            log(f"  frame {src}: raw mesh export failed: {e}")

    # Camera-frame (metric, OpenCV axes) mesh -- the canonical artifact.
    verts_cv = place_mesh_in_camera_frame(
        verts_yup, out["translation"], out["rotation"], out["scale"], device)
    placed = mesh.copy()
    placed.vertices = verts_cv
    mesh_path = meshes_dir / f"{src:06d}.glb"
    placed.export(str(mesh_path))
    rec["mesh_file"] = f"meshes/{mesh_path.name}"
    rec["mesh_aabb_min"] = verts_cv.min(axis=0).tolist()
    rec["mesh_aabb_max"] = verts_cv.max(axis=0).tolist()
    if rec["near_cube"]:
        log(f"  frame {src}: WARNING near-degenerate mesh "
            f"(extents-spread={spread:.3f}) -- flagged near_cube")
    return rec


def write_debug(prep: dict, debug_dir: Path) -> None:
    """Save a masked-input preview so the depth/mask alignment can be eyeballed."""
    rgb = prep["rgb"]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = prep["mask"]
    dimmed = (bgr.astype(np.float32) * 0.30).astype(np.uint8)
    dimmed[mask] = bgr[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dimmed, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(str(debug_dir / f"{prep['src']:06d}.jpg"), dimmed,
                [cv2.IMWRITE_JPEG_QUALITY, 90])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: per-frame 3D object mesh generation via SAM 3D "
                    "Objects, from Step 1 (depth + camera) and Step 2 (mask).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--step1_dir", required=True, type=Path,
                        help="Step 1 output dir for the video "
                             "(contains depth/, frames/, intrinsics.npy, meta.json).")
    parser.add_argument("--step2_dir", required=True, type=Path,
                        help="Step 2 output dir for the video "
                             "(contains masks_rle.json, meta.json).")
    parser.add_argument("--output_dir", type=Path,
                        default=REPO_ROOT / "output" / "step3",
                        help="Base output dir; results go to <output_dir>/<video_name>/.")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Target mesh sampling rate (meshes per second of video).")
    parser.add_argument("--input_fps", type=float, default=None,
                        help="Frame rate of the Step 2 frame sequence, used to "
                             "compute the sampling stride. Default: read from "
                             "Step 2 meta.json.")
    parser.add_argument("--sam3d_repo", type=Path, default=DEFAULT_SAM3D_REPO,
                        help="Path to the SAM 3D Objects repository (submodule).")
    parser.add_argument("--sam3d_checkpoint_dir", type=Path, default=None,
                        help="Dir with the SAM 3D Objects checkpoints + pipeline.yaml. "
                             "Default: <sam3d_repo>/checkpoints/hf.")
    parser.add_argument("--sam3d_patch", type=Path, default=DEFAULT_SAM3D_PATCH,
                        help="GT-intrinsics patch auto-applied to the submodule.")
    parser.add_argument("--cache_dir", type=Path, default=REPO_ROOT / ".cache",
                        help="Cache dir for any auxiliary model downloads "
                             "(HF_HOME / TORCH_HOME); gitignored.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device for SAM 3D Objects.")
    parser.add_argument("--seed", type=int, default=42,
                        help="SAM 3D Objects diffusion seed.")
    parser.add_argument("--max_depth", type=float, default=0.0,
                        help="Drop pointmap pixels beyond this depth in meters "
                             "(0 = disabled; the object mask already bounds the input).")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Cap the number of frames processed (0 = all). For smoke tests.")
    parser.add_argument("--no_raw_mesh", action="store_true",
                        help="Do not also save the untransformed SAM 3D Objects mesh.")
    parser.add_argument("--no_prewarm", action="store_true",
                        help="Skip streaming the checkpoints through the page cache.")
    parser.add_argument("--save_debug", action="store_true",
                        help="Write debug/ masked-input previews per processed frame.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute frames whose mesh file already exists.")
    args = parser.parse_args()

    step1_dir = args.step1_dir.resolve()
    step2_dir = args.step2_dir.resolve()
    if not (step1_dir / "meta.json").exists():
        parser.error(f"--step1_dir has no meta.json: {step1_dir}")
    if not (step2_dir / "meta.json").exists():
        parser.error(f"--step2_dir has no meta.json: {step2_dir}")
    if args.fps <= 0:
        parser.error("--fps must be > 0")

    # --- load Step 1 / Step 2 ----------------------------------------------
    step1 = load_step1(step1_dir)
    step2 = load_step2(step2_dir)
    video_name = step1["meta"].get("video_name") or step1_dir.name
    if step2["meta"].get("video_name") not in (None, video_name):
        log(f"WARNING: Step 1 video '{video_name}' != Step 2 video "
            f"'{step2['meta'].get('video_name')}' -- proceeding anyway")

    out_dir = (args.output_dir / video_name).resolve()
    meshes_dir = out_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = None if args.no_raw_mesh else (out_dir / "meshes_raw")
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = (out_dir / "debug") if args.save_debug else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # --- frame sampling -----------------------------------------------------
    n2 = len(step2["src_indices"])
    input_fps = args.input_fps or step2["meta"].get("fps")
    if not input_fps or input_fps <= 0:
        log("Step 2 meta.json has no usable fps; assuming stride 1 (every frame). "
            "Pass --input_fps to control sampling.")
        input_fps = args.fps
    stride = max(1, round(input_fps / args.fps))
    sampled = list(range(0, n2, stride))
    if args.max_frames > 0:
        sampled = sampled[:args.max_frames]

    log(f"video        : {video_name}")
    log(f"step1 dir    : {step1_dir}  ({len(step1['src_indices'])} frames)")
    log(f"step2 dir    : {step2_dir}  ({n2} frames)")
    log(f"output dir   : {out_dir}")
    log(f"sampling     : input_fps={input_fps:.3f} target_fps={args.fps} "
        f"-> stride={stride} -> {len(sampled)} candidate frame(s)")

    # --- SAM 3D Objects setup ----------------------------------------------
    sam3d_repo = args.sam3d_repo.resolve()
    ckpt_dir = (args.sam3d_checkpoint_dir or (sam3d_repo / "checkpoints" / "hf")).resolve()
    config_file = ckpt_dir / "pipeline.yaml"
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for var in ("HF_HOME", "TORCH_HOME"):
        os.environ.setdefault(var, str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hub"))
    ensure_sam3d_patched(sam3d_repo, args.sam3d_patch.resolve())
    if not args.no_prewarm:
        prewarm_checkpoints(ckpt_dir)
    model = load_sam3d_model(sam3d_repo, config_file)

    # --- per-frame loop -----------------------------------------------------
    prev_records: dict[int, dict] = {}
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not args.overwrite:
        try:
            for r in load_json(meta_path).get("frames", []):
                prev_records[int(r["source_frame_index"])] = r
        except Exception:  # noqa: BLE001
            prev_records = {}

    records: list[dict] = []
    counts = {"ok": 0, "resumed": 0, "no_mask": 0, "no_step1_frame": 0,
              "empty_mask": 0, "rgb_read_fail": 0, "no_mesh": 0, "error": 0}
    t_start = time.time()

    for n, j2 in enumerate(sampled):
        src = int(step2["src_indices"][j2])
        mesh_exists = (meshes_dir / f"{src:06d}.glb").exists()
        if mesh_exists and not args.overwrite and src in prev_records:
            records.append(prev_records[src])
            counts["resumed"] += 1
            log(f"[{n + 1}/{len(sampled)}] frame {src}: resumed (mesh exists)")
            continue

        prep = prepare_frame(step1, step2, j2, args.max_depth)
        if isinstance(prep, str):
            counts[prep] = counts.get(prep, 0) + 1
            log(f"[{n + 1}/{len(sampled)}] frame {src}: skipped ({prep})")
            continue

        try:
            out, elapsed = run_sam3d(model, prep["rgb"], prep["mask"],
                                     pointmap_cv_to_pt3d(prep["pm_cv"]),
                                     prep["K"], args.device, args.seed)
            rec = finalize_frame(prep, out, elapsed, args.device,
                                 meshes_dir, raw_dir)
            if args.save_debug:
                write_debug(prep, debug_dir)
        except Exception as e:  # noqa: BLE001 -- one bad frame must not abort the run
            import traceback
            traceback.print_exc()
            counts["error"] += 1
            records.append({"source_frame_index": src, "step1_index": prep["j1"],
                            "step2_index": j2, "status": f"error: {e}",
                            "mesh_file": None})
            log(f"[{n + 1}/{len(sampled)}] frame {src}: ERROR {e}")
            continue
        finally:
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
        records.append(rec)
        log(f"[{n + 1}/{len(sampled)}] frame {src}: {rec['status']} "
            f"({elapsed:.1f}s"
            + (f", {rec.get('num_vertices')} verts" if rec.get("num_vertices") else "")
            + (", near_cube" if rec.get("near_cube") else "") + ")")

    # --- write meta.json ----------------------------------------------------
    records.sort(key=lambda r: r["source_frame_index"])
    num_meshes = sum(1 for r in records if r.get("mesh_file"))
    meta = {
        "step": "step3_sam3d_mesh",
        "generated_by": "scripts/step3_sam3d_mesh.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video_name": video_name,
        "model": "SAM-3D-Objects",
        "sam3d_repo": str(sam3d_repo),
        "step1_dir": str(step1_dir),
        "step2_dir": str(step2_dir),
        "target_fps": args.fps,
        "input_fps": input_fps,
        "frame_stride": stride,
        "seed": args.seed,
        "max_depth": args.max_depth,
        "num_candidate_frames": len(sampled),
        "num_meshes": num_meshes,
        "counts": counts,
        "mesh_frame": "camera, OpenCV axes (x-right, y-down, z-forward), metric meters",
        "meshes": {
            "layout": "meshes/{source_frame_index:06d}.glb",
            "frame": "camera frame, metric meters, OpenCV axes; placed where the "
                     "object is relative to the camera. Apply per-frame "
                     "camera_to_world to obtain world coordinates.",
            "raw_layout": "meshes_raw/{source_frame_index:06d}.glb",
            "raw": "untransformed SAM 3D Objects mesh (object-local, y-up, ~unit "
                   "AABB); re-place via the per-frame sam3d_pose.",
            "key": "source_frame_index == original video frame number, matching "
                   "Step 1 / Step 2 meta.json source_frame_indices.",
        },
        "frames": records,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    elapsed_min = (time.time() - t_start) / 60
    log(f"wrote meta.json ({num_meshes} mesh(es) over {len(sampled)} candidate "
        f"frame(s), {elapsed_min:.1f} min)")
    log("counts: " + "  ".join(f"{k}={v}" for k, v in counts.items() if v))
    log("Step 3 complete.")


if __name__ == "__main__":
    main()
