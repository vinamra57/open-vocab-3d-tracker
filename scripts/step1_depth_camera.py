#!/usr/bin/env python3
"""Step 1: per-frame depth + camera parameters from a monocular RGB video.

Part of the generalized, in-the-wild, open-vocabulary video object tracking
pipeline. This step is fully generic: it works on ANY single RGB video and has
no hardcoded data paths, environment paths, or dataset-specific assumptions.
Everything is CLI-driven.

What it does
------------
1. Runs RADIO-ViPE (``third_party/RADIO-ViPE``) on the input video. RADIO-ViPE
   is a monocular SLAM + metric-depth engine that needs no camera intrinsics,
   depth sensor, or pose initialization. It produces per-frame metric depth,
   per-frame camera intrinsics, and per-frame camera poses.
2. Converts RADIO-ViPE's artifacts into the Step 1 output format below, which
   downstream steps (mask tracklet, SAM3D per-frame mesh, box fit, and point
   cloud visualization) consume.

Environment
-----------
Must be run inside the ``radio-vipe`` conda env so ``run.py`` can import vipe:

    conda activate radio-vipe
    python scripts/step1_depth_camera.py --video /path/to/video.mp4

Output format
-------------
For each processed video, written to ``<output_dir>/<video_name>/``:

    depth/000000.npy ...   (H, W) float32, metric depth in METERS.
                           Invalid / unknown pixels are 0.0 (downstream treats
                           ``depth > 0`` as the validity mask).
    frames/000000.jpg ...  RGB frames, indexed 1:1 with depth (and with the
                           intrinsics / extrinsics rows).
    intrinsics.npy         (N, 3, 3) float32 pinhole K per frame:
                           [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], pixel units.
    extrinsics.npy         (N, 4, 4) float32 world-to-camera transform per
                           frame (the canonical extrinsic; OpenCV camera axes:
                           x-right, y-down, z-forward).
    poses_c2w.npy          (N, 4, 4) float32 camera-to-world transform per
                           frame (== inverse of extrinsics; convenience copy).
    meta.json              Metadata + an explicit description of every
                           convention (units, axes, frame index mapping, ...).
    debug/                 Optional sanity-check artifacts (fused point cloud
                           PLY + a few colorized depth previews).
    radio_vipe_raw/         Raw RADIO-ViPE artifacts (kept unless --no_keep_raw).

Conventions (also restated in meta.json)
-----------------------------------------
* Camera model: pinhole. Camera axes: OpenCV (x-right, y-down, z-forward).
* Depth: metric meters, float32, invalid == 0.0.
* World frame: RADIO-ViPE anchors frame 0 near identity, so the world frame is
  (approximately) the first camera's frame.
* A 3D point is unprojected as:
      X = (u - cx) / fx * d ;  Y = (v - cy) / fy * d ;  Z = d
  then mapped to world coordinates with poses_c2w (or inv(extrinsics)).
* Frame indexing: outputs are indexed 0..N-1. meta.json["source_frame_indices"]
  maps each output index back to the original video frame number, so a Step 2
  mask tracklet computed on the same video can be aligned.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[step1] {msg}", flush=True)


# ---------------------------------------------------------------------------
# RADIO-ViPE invocation
# ---------------------------------------------------------------------------

def ensure_radio_vipe_patched(radio_vipe_dir: Path) -> None:
    """Ensure the RADIO-ViPE submodule checkout carries our local robustness patch.

    RADIO-ViPE's ``extract_slam_map`` crashes on videos that produce no RADSeg
    embeddings (e.g. when bundle adjustment diverges). We keep the fix as a
    tracked patch in third_party/patches/; if the submodule checkout is missing
    it (e.g. after a reset), apply it automatically.
    """
    buffer_py = radio_vipe_dir / "vipe" / "slam" / "components" / "buffer.py"
    marker = "if staged_emb is not None else None"
    if buffer_py.exists() and marker in buffer_py.read_text():
        return  # already patched
    patch = REPO_ROOT / "third_party" / "patches" / "radio-vipe-extract_slam_map-none-guard.patch"
    if not patch.exists():
        raise FileNotFoundError(
            f"RADIO-ViPE is missing the extract_slam_map fix and the patch file "
            f"is absent: {patch}"
        )
    log(f"applying RADIO-ViPE patch: {patch.name}")
    result = subprocess.run(
        ["git", "apply", str(patch)], cwd=str(radio_vipe_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to apply {patch.name}:\n{result.stderr}\n"
            f"Apply it manually: cd {radio_vipe_dir} && git apply {patch}"
        )


def run_radio_vipe(
    video_path: Path,
    raw_dir: Path,
    radio_vipe_dir: Path,
    frame_start: int,
    frame_end: int,
    frame_skip: int,
    pipeline: str,
) -> None:
    """Run RADIO-ViPE's run.py on the input video, writing artifacts to raw_dir."""
    run_py = radio_vipe_dir / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"RADIO-ViPE entrypoint not found: {run_py}")
    ensure_radio_vipe_patched(radio_vipe_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    # Hydra overrides. base_path may be a single file or a directory of mp4s;
    # RADIO-ViPE's RawMP4StreamList handles both.
    cmd = [
        sys.executable,
        "run.py",
        f"pipeline={pipeline}",
        "streams=raw_mp4_stream",
        f"streams.base_path={video_path}",
        f"streams.frame_start={frame_start}",
        f"streams.frame_end={frame_end}",
        f"streams.frame_skip={frame_skip}",
        f"pipeline.output.path={raw_dir}",
        "pipeline.output.save_artifacts=true",
        "pipeline.output.save_slam_map=false",
        "pipeline.output.save_viz=false",
        "pipeline.slam.visualize=false",
        f"pipeline.slam.pca_state_path={raw_dir / 'vipe'}",
        "memory_profiler=false",
    ]
    log(f"running RADIO-ViPE: {' '.join(cmd)}")
    log(f"(cwd={radio_vipe_dir})")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(radio_vipe_dir), check=True)
    log(f"RADIO-ViPE finished in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# RADIO-ViPE artifact readers
# ---------------------------------------------------------------------------

def read_depth_zip(zip_path: Path) -> dict[int, np.ndarray]:
    """Read RADIO-ViPE's zipped half-float EXR depth maps.

    Returns {frame_idx: (H, W) float32 metric depth in meters}.
    """
    import Imath  # noqa: F401  (imported for parity with vipe; not strictly needed)
    import OpenEXR

    depths: dict[int, np.ndarray] = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                exr = OpenEXR.InputFile(f)
                header = exr.header()
                dw = header["dataWindow"]
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1
                (channel,) = exr.channels(["Z"])
                depth = np.frombuffer(channel, dtype=np.float16).reshape(height, width)
                depths[frame_idx] = depth.astype(np.float32)
    return depths


def intrinsics_vec_to_K(vec: np.ndarray) -> np.ndarray:
    """Convert RADIO-ViPE pinhole intrinsics [fx, fy, cx, cy] -> 3x3 K matrix."""
    fx, fy, cx, cy = (float(v) for v in vec)
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
    )


def sanitize_depth(depth: np.ndarray) -> np.ndarray:
    """Force invalid depth (NaN/inf/non-positive) to 0.0; return float32 meters."""
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    depth[depth < 0.0] = 0.0
    return depth


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------

def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary little-endian PLY point cloud (xyz float32 + rgb uint8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    points = points.astype("<f4")
    colors = colors.astype(np.uint8)
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
         ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    rows = np.empty(n, dtype=dtype)
    rows["x"], rows["y"], rows["z"] = points[:, 0], points[:, 1], points[:, 2]
    rows["r"], rows["g"], rows["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(rows.tobytes())


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Turbo-colormap a metric depth map for a quick visual sanity check."""
    valid = depth > 0
    vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[valid] = cm[valid]
    return vis


def make_debug_artifacts(
    out_dir: Path,
    depths: list[np.ndarray],
    rgbs: list[np.ndarray],
    intrinsics: np.ndarray,
    poses_c2w: np.ndarray,
    max_frames: int = 8,
    stride: int = 6,
) -> None:
    """Fuse a few frames into a world-frame point cloud + dump depth previews.

    If depth/intrinsics/extrinsics are mutually consistent, the per-frame clouds
    overlap cleanly in world space -- a fast end-to-end correctness check.
    """
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    n = len(depths)
    sel = list(range(0, n, max(1, n // max_frames)))[:max_frames]

    all_pts, all_cols = [], []
    for i in sel:
        depth = depths[i]
        K = intrinsics[i]
        c2w = poses_c2w[i]
        h, w = depth.shape
        vs, us = np.mgrid[0:h:stride, 0:w:stride]
        d = depth[vs, us]
        valid = d > 0
        if not valid.any():
            continue
        u, v, d = us[valid], vs[valid], d[valid]
        x = (u - K[0, 2]) / K[0, 0] * d
        y = (v - K[1, 2]) / K[1, 1] * d
        cam = np.stack([x, y, d], axis=-1)
        world = cam @ c2w[:3, :3].T + c2w[:3, 3]
        all_pts.append(world.astype(np.float32))
        all_cols.append(rgbs[i][vs, us][valid])

        cv2.imwrite(
            str(debug_dir / f"depth_vis_{i:06d}.jpg"),
            cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR),
        )

    if all_pts:
        pts = np.concatenate(all_pts, axis=0)
        cols = np.concatenate(all_cols, axis=0)
        write_ply(debug_dir / "fused_pointcloud.ply", pts, cols)
        log(f"  debug: fused {len(sel)} frames -> {len(pts)} pts "
            f"(debug/fused_pointcloud.ply)")


# ---------------------------------------------------------------------------
# Artifact conversion
# ---------------------------------------------------------------------------

def convert_artifacts(
    name: str,
    raw_dir: Path,
    out_dir: Path,
    source_video: Path,
    frame_start: int,
    frame_skip: int,
    pipeline: str,
    debug_viz: bool,
) -> None:
    """Convert one video's RADIO-ViPE artifacts into the Step 1 output format."""
    pose_npz = raw_dir / "pose" / f"{name}.npz"
    intr_npz = raw_dir / "intrinsics" / f"{name}.npz"
    depth_zip = raw_dir / "depth" / f"{name}.zip"
    rgb_mp4 = raw_dir / "rgb" / f"{name}.mp4"
    for p in (pose_npz, intr_npz, depth_zip, rgb_mp4):
        if not p.exists():
            raise FileNotFoundError(f"expected RADIO-ViPE artifact missing: {p}")

    pose_data = np.load(pose_npz)          # cam->world 4x4 (OpenCV)
    intr_data = np.load(intr_npz)          # [fx, fy, cx, cy] per frame
    pose_by_idx = dict(zip(pose_data["inds"].tolist(), pose_data["data"]))
    intr_by_idx = dict(zip(intr_data["inds"].tolist(), intr_data["data"]))
    depth_by_idx = read_depth_zip(depth_zip)

    # Only keep frames present in all three artifact sets.
    common = sorted(set(pose_by_idx) & set(intr_by_idx) & set(depth_by_idx))
    if not common:
        raise RuntimeError(f"no frames common to pose/intrinsics/depth for '{name}'")
    n = len(common)
    log(f"  '{name}': {n} frames with depth + intrinsics + pose")

    # Read the RGB frames RADIO-ViPE actually processed (already subsampled).
    cap = cv2.VideoCapture(str(rgb_mp4))
    rgb_frames: dict[int, np.ndarray] = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in depth_by_idx:  # superset of `common`
            rgb_frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()

    depth_dir = out_dir / "depth"
    frames_dir = out_dir / "frames"
    for d in (depth_dir, frames_dir):
        d.mkdir(parents=True, exist_ok=True)

    intrinsics = np.zeros((n, 3, 3), dtype=np.float32)
    poses_c2w = np.zeros((n, 4, 4), dtype=np.float32)
    extrinsics = np.zeros((n, 4, 4), dtype=np.float32)
    depths: list[np.ndarray] = []
    rgbs: list[np.ndarray] = []
    depth_stats = {"min": np.inf, "max": -np.inf, "valid_frac": []}

    for j, src_idx in enumerate(common):
        depth = sanitize_depth(depth_by_idx[src_idx])
        np.save(depth_dir / f"{j:06d}.npy", depth)
        depths.append(depth)

        rgb = rgb_frames.get(src_idx)
        if rgb is None:
            raise RuntimeError(f"RGB frame {src_idx} missing from {rgb_mp4}")
        cv2.imwrite(
            str(frames_dir / f"{j:06d}.jpg"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        rgbs.append(rgb)

        intrinsics[j] = intrinsics_vec_to_K(intr_by_idx[src_idx])
        c2w = pose_by_idx[src_idx].astype(np.float32)
        poses_c2w[j] = c2w
        extrinsics[j] = np.linalg.inv(c2w)

        valid = depth > 0
        if valid.any():
            depth_stats["min"] = min(depth_stats["min"], float(depth[valid].min()))
            depth_stats["max"] = max(depth_stats["max"], float(depth[valid].max()))
        depth_stats["valid_frac"].append(float(valid.mean()))

    np.save(out_dir / "intrinsics.npy", intrinsics)
    np.save(out_dir / "extrinsics.npy", extrinsics)
    np.save(out_dir / "poses_c2w.npy", poses_c2w)

    h, w = depths[0].shape
    intr_constant = bool(np.allclose(intrinsics, intrinsics[0]))
    meta = {
        "step": "step1_depth_camera",
        "generated_by": "scripts/step1_depth_camera.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_video": str(Path(source_video).resolve()),
        "video_name": name,
        "model": "RADIO-ViPE",
        "radio_vipe_pipeline": pipeline,
        "num_frames": n,
        "image_width": w,
        "image_height": h,
        # output index j -> original video frame number
        "source_frame_indices": [frame_start + i * frame_skip for i in common],
        "frame_start": frame_start,
        "frame_skip": frame_skip,
        "depth": {
            "layout": "depth/{index:06d}.npy",
            "dtype": "float32",
            "unit": "meters",
            "invalid_value": 0.0,
            "valid_mask_rule": "depth > 0",
            "min_valid": depth_stats["min"] if np.isfinite(depth_stats["min"]) else None,
            "max_valid": depth_stats["max"] if np.isfinite(depth_stats["max"]) else None,
            "mean_valid_fraction": float(np.mean(depth_stats["valid_frac"])),
        },
        "frames": {"layout": "frames/{index:06d}.jpg", "color": "RGB"},
        "intrinsics": {
            "file": "intrinsics.npy",
            "shape": [n, 3, 3],
            "dtype": "float32",
            "model": "pinhole",
            "layout": "K = [[fx,0,cx],[0,fy,cy],[0,0,1]], pixel units",
            "per_frame": True,
            "constant_across_frames": intr_constant,
        },
        "extrinsics": {
            "file": "extrinsics.npy",
            "shape": [n, 4, 4],
            "dtype": "float32",
            "convention": "world_to_camera",
            "camera_axes": "OpenCV (x-right, y-down, z-forward)",
            "world_frame": "approximately the frame-0 camera (RADIO-ViPE anchors "
                           "the first pose near identity)",
        },
        "poses_c2w": {
            "file": "poses_c2w.npy",
            "shape": [n, 4, 4],
            "dtype": "float32",
            "convention": "camera_to_world",
            "note": "inverse of extrinsics; convenience copy",
        },
        "unprojection": (
            "X=(u-cx)/fx*d; Y=(v-cy)/fy*d; Z=d (camera frame), "
            "then world = poses_c2w @ [X,Y,Z,1]"
        ),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log(f"  '{name}': wrote depth/intrinsics/extrinsics/frames + meta.json "
        f"to {out_dir}")
    log(f"  '{name}': depth range "
        f"{meta['depth']['min_valid']:.3f}-{meta['depth']['max_valid']:.3f} m, "
        f"intrinsics constant={intr_constant}")

    if debug_viz:
        make_debug_artifacts(out_dir, depths, rgbs, intrinsics, poses_c2w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: per-frame depth + camera parameters from a "
                    "monocular RGB video (RADIO-ViPE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video", required=True, type=Path,
        help="Input RGB video (.mp4), or a directory of .mp4 files.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=REPO_ROOT / "output" / "step1",
        help="Base output directory; results go to <output_dir>/<video_name>/.",
    )
    parser.add_argument(
        "--radio_vipe_dir", type=Path,
        default=REPO_ROOT / "third_party" / "RADIO-ViPE",
        help="Path to the RADIO-ViPE repository.",
    )
    parser.add_argument(
        "--cache_dir", type=Path, default=REPO_ROOT / ".cache",
        help="Directory for downloaded models (exported as TORCH_HOME). "
             "RADIO-ViPE auto-downloads the RADIO model here on first run; "
             "keep it gitignored and on persistent storage.",
    )
    parser.add_argument("--frame_start", type=int, default=0,
                        help="First video frame to process.")
    parser.add_argument("--frame_end", type=int, default=-1,
                        help="Last video frame (exclusive); -1 means end of video.")
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="Process every Nth frame (1 = every frame).")
    parser.add_argument("--pipeline", type=str, default="default",
                        help="RADIO-ViPE pipeline config (e.g. default, no_vda).")
    parser.add_argument("--no_debug_viz", action="store_true",
                        help="Skip debug point cloud / depth previews.")
    parser.add_argument("--no_keep_raw", action="store_true",
                        help="Delete raw RADIO-ViPE artifacts after conversion.")
    parser.add_argument("--reuse_raw", action="store_true",
                        help="Reuse existing raw RADIO-ViPE artifacts; skip the "
                             "(slow) RADIO-ViPE run. Useful for re-conversion.")
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.exists():
        parser.error(f"--video does not exist: {video}")
    radio_vipe_dir = args.radio_vipe_dir.resolve()
    if not radio_vipe_dir.exists():
        parser.error(f"--radio_vipe_dir does not exist: {radio_vipe_dir}")

    # RADIO-ViPE's RADSeg encoder auto-downloads the NVlabs/RADIO model via
    # torch.hub. Route that (and any other torch.hub fetches) to a persistent,
    # gitignored cache instead of the ephemeral ~/.cache.
    cache_dir = args.cache_dir.resolve()
    (cache_dir / "torch").mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    log(f"weight cache: {cache_dir} (TORCH_HOME)")

    # Make the RADIO-ViPE JIT CUDA-extension build deterministic: if CUDA_HOME
    # is unset but the active conda env ships nvcc, point CUDA_HOME at it.
    if not os.environ.get("CUDA_HOME"):
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix and (Path(conda_prefix) / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = conda_prefix
            log(f"CUDA_HOME unset; defaulting to CONDA_PREFIX={conda_prefix}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "radio_vipe_raw"

    log(f"input video : {video}")
    log(f"output dir  : {output_dir}")
    log(f"raw artifacts: {raw_dir}")

    if args.reuse_raw and (raw_dir / "pose").exists():
        log("reusing existing raw RADIO-ViPE artifacts (--reuse_raw)")
    else:
        run_radio_vipe(
            video, raw_dir, radio_vipe_dir,
            args.frame_start, args.frame_end, args.frame_skip, args.pipeline,
        )

    # Convert exactly the video(s) for this invocation (RADIO-ViPE names each
    # artifact by the video stem). --video may be a single file or a directory;
    # this avoids re-converting stale artifacts left in raw_dir by earlier runs.
    if video.is_dir():
        names = sorted(p.stem for p in video.glob("*.mp4"))
    else:
        names = [video.stem]
    if not names:
        raise RuntimeError(f"no .mp4 video(s) found at {video}")
    pose_dir = raw_dir / "pose"
    missing = [n for n in names if not (pose_dir / f"{n}.npz").exists()]
    if missing:
        raise RuntimeError(
            f"RADIO-ViPE produced no pose artifact for: {missing} (looked in {pose_dir})"
        )
    log(f"converting {len(names)} video(s): {names}")

    for name in names:
        convert_artifacts(
            name=name,
            raw_dir=raw_dir,
            out_dir=output_dir / name,
            source_video=video,
            frame_start=args.frame_start,
            frame_skip=args.frame_skip,
            pipeline=args.pipeline,
            debug_viz=not args.no_debug_viz,
        )

    if args.no_keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)
        log("removed raw RADIO-ViPE artifacts (--no_keep_raw)")

    log("Step 1 complete.")


if __name__ == "__main__":
    main()
