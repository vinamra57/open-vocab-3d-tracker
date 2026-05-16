#!/usr/bin/env python3
"""Visualise Step 3 meshes as projected 3D bounding boxes on the source frames.

Quick QA utility for Step 3 (``step3_sam3d_mesh.py``). For each frame Step 3
reconstructed, it fits an object-local axis-aligned box to the SAM 3D Objects
mesh -- the tight *oriented* box, built the same way the droid reference
pipeline did (``sam3d_output_to_corners_cv``) -- projects the 8 corners onto the
original RGB frame with that frame's intrinsics, and draws the box wireframe.

Note: fitting the final tight 3D bounding box is Step 4's job. This is only a
visual sanity check that Step 3's per-frame mesh is placed correctly in the
camera frame. The box here is just the mesh's object-local AABB.

Run with the ``sam3d-objects`` conda env (needs torch + pytorch3d + trimesh):

    python scripts/viz_step3_boxes.py --step3_dir output/step3/dog-example

Output: <step3_dir>/viz/{source_frame_index:06d}_box.jpg  (per-frame overlays)
        <step3_dir>/viz/boxes.mp4                          (overlays as a video)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

# 12 edges of an 8-corner box; corner ordering matches mesh_box_corners_cv below
# (and the droid pipeline's step3b1_singleframe_box.py).
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # top face   (y > 0)
    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face (y < 0)
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]


def mesh_box_corners_cv(raw_mesh, sam3d_pose: dict) -> np.ndarray:
    """8 corners of the mesh's object-local AABB, in the OpenCV camera frame.

    Ported from the droid pipeline's ``sam3d_output_to_corners_cv``: undo
    SAM 3D's z-up->y-up packaging rotation, take a robust AABB in the object
    frame, then apply the predicted (scale, rotation, translation) pose and
    flip X/Y (PyTorch3D camera axes -> OpenCV camera axes)."""
    import torch
    from pytorch3d.transforms import Transform3d, quaternion_to_matrix

    verts_yup = np.asarray(raw_mesh.vertices, dtype=np.float32)
    m_inv = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0]], dtype=np.float32)
    verts = verts_yup @ m_inv

    # Robust AABB: drop the outer 2% of mass (mesh noise is common).
    if len(verts) > 10:
        dists = np.linalg.norm(verts - verts.mean(axis=0), axis=1)
        verts = verts[dists <= np.percentile(dists, 98.0)]

    bbox_min, bbox_max = verts.min(axis=0), verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    w, h, l = bbox_max - bbox_min
    x_c = np.array([w/2,  w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2])
    y_c = np.array([h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2])
    z_c = np.array([l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2,  l/2])
    corners = np.vstack([x_c, y_c, z_c]).T + center  # (8, 3) object-local

    t = torch.tensor(sam3d_pose["translation"], dtype=torch.float32).reshape(1, 3)
    q = torch.tensor(sam3d_pose["rotation_wxyz"], dtype=torch.float32).reshape(1, 4)
    s = torch.tensor(sam3d_pose["scale"], dtype=torch.float32).reshape(1, 3)
    R = quaternion_to_matrix(q)
    tfm = Transform3d().scale(s).rotate(R).translate(t)
    corners_t = torch.from_numpy(corners).float().unsqueeze(0)
    corners_cv = tfm.transform_points(corners_t)[0].numpy()
    corners_cv[:, 0] *= -1.0  # PyTorch3D X -> OpenCV X
    corners_cv[:, 1] *= -1.0  # PyTorch3D Y -> OpenCV Y
    return corners_cv


def project_corners(corners_cv: np.ndarray, K: np.ndarray):
    """Project 8 OpenCV-frame corners to 2D pixels; None if any is behind the camera."""
    if np.any(corners_cv[:, 2] <= 1e-4):
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * corners_cv[:, 0] / corners_cv[:, 2] + cx
    v = fy * corners_cv[:, 1] / corners_cv[:, 2] + cy
    return np.stack([u, v], axis=-1)


def draw_box(bgr: np.ndarray, corners_2d: np.ndarray, label: str) -> np.ndarray:
    """Draw the projected 3D-box wireframe (BGR image, in place on a copy)."""
    out = bgr.copy()
    pts = corners_2d.astype(np.int32)
    for i, j in BOX_EDGES:
        cv2.line(out, tuple(pts[i]), tuple(pts[j]), (0, 165, 255), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(out, (int(x), int(y)), 4, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay Step 3 mesh bounding boxes onto the source frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--step3_dir", required=True, type=Path,
                        help="A Step 3 output dir (contains meta.json, meshes_raw/).")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Where to write overlays. Default: <step3_dir>/viz.")
    parser.add_argument("--frames", type=str, default=None,
                        help="Comma-separated source_frame_index values to render "
                             "(default: all frames with a mesh).")
    parser.add_argument("--video_fps", type=float, default=None,
                        help="Frame rate for the assembled overlay video. "
                             "Default: Step 3's target_fps from meta.json.")
    parser.add_argument("--no_video", action="store_true",
                        help="Write only per-frame overlay images, no video.")
    args = parser.parse_args()

    step3_dir = args.step3_dir.resolve()
    meta = json.loads((step3_dir / "meta.json").read_text())
    step1_dir = Path(meta["step1_dir"])
    out_dir = (args.output_dir or (step3_dir / "viz")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = None
    if args.frames:
        wanted = {int(x) for x in args.frames.split(",")}

    overlays: list[np.ndarray] = []
    for rec in meta["frames"]:
        src = rec["source_frame_index"]
        if wanted is not None and src not in wanted:
            continue
        if not rec.get("raw_mesh_file") or not rec.get("sam3d_pose"):
            print(f"frame {src}: skipped (no raw mesh / pose -- re-run Step 3 "
                  f"without --no_raw_mesh)")
            continue
        rgb_path = step1_dir / "frames" / f"{rec['step1_index']:06d}.jpg"
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            print(f"frame {src}: skipped (cannot read {rgb_path})")
            continue

        raw_mesh = trimesh.load(str(step3_dir / rec["raw_mesh_file"]), force="mesh")
        K = np.asarray(rec["intrinsics"], dtype=np.float64)
        corners_cv = mesh_box_corners_cv(raw_mesh, rec["sam3d_pose"])
        corners_2d = project_corners(corners_cv, K)
        if corners_2d is None:
            print(f"frame {src}: skipped (box projects behind the camera)")
            continue

        size = np.asarray(corners_cv).ptp(axis=0)
        label = (f"frame {src}   box ~{size[0]:.2f} x {size[1]:.2f} x "
                 f"{size[2]:.2f} m (camera-frame extent)")
        overlay = draw_box(bgr, corners_2d, label)
        out_path = out_dir / f"{src:06d}_box.jpg"
        cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        overlays.append(overlay)
        print(f"frame {src}: wrote {out_path}")

    print(f"wrote {len(overlays)} overlay image(s) to {out_dir}")

    # Assemble the per-frame overlays into a video. meta['frames'] is ordered
    # by source_frame_index, so the sampled frames play back at Step 3's
    # sampling rate (target_fps) -- i.e. roughly real time.
    if overlays and not args.no_video:
        fps = args.video_fps or meta.get("target_fps") or 5.0
        H, W = overlays[0].shape[:2]
        video_path = out_dir / "boxes.mp4"
        writer = cv2.VideoWriter(str(video_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(fps), (W, H))
        for frame in overlays:
            writer.write(frame)
        writer.release()
        print(f"wrote {video_path}  ({len(overlays)} frames @ {fps:g} fps)")


if __name__ == "__main__":
    main()
