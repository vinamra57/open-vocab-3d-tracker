#!/usr/bin/env python3
"""Step 2: per-frame 2D object-mask tracklet from a monocular RGB video + text label.

Part of the generalized, in-the-wild, open-vocabulary video object tracking
pipeline. This step is fully generic: it works on ANY single RGB video, the only
prompt is a free-text object/category label, and it has no hardcoded data paths,
environment paths, or dataset-specific assumptions. Everything is CLI-driven.

What it does
------------
Given an RGB video and a text label, produce a per-frame 2D segmentation mask
tracklet for that target object, using SAM 3.

Method  (``--method hybrid``, the default)
------------------------------------------
1. Phase 1 -- seed search.  Run SAM 3 *image mode* with a text-only prompt
   independently on every frame, take the top-1 detection per frame, and score
   it (``total = sam3_score * edge_score``).  The highest-scoring frame is the
   ``best_frame``.  No VLM and no box generation -- pure text grounding.
2. Phase 2 -- propagation.  Seed the SAM 3 *video predictor* on ``best_frame``
   with the text label and propagate bidirectionally (forward + backward) to
   obtain a temporally consistent mask tracklet with a stable object identity.

The seed is a *text* prompt on the image-mode-chosen ``best_frame``.  SAM 3
video grounding propagates a text *concept* across the whole clip; a box /
visual prompt, by contrast, is attached only to its single frame and leaves
nothing for propagation to carry.  Anchoring the seed on the highest-confidence
frame avoids the failure mode where SAM 3 cannot ground the object on the seed
frame and the entire tracklet comes out empty.

``--method image`` skips Phase 2 and emits the raw per-frame image-mode masks
(no temporal linking).  Useful as a baseline / for comparison.

Environment
-----------
Run inside an env that has SAM 3 and its dependencies installed (e.g. a ``sam3``
conda env).  ``--sam3_repo`` is prepended to ``sys.path`` so the pinned
``third_party/sam3`` submodule is authoritative; if that is absent the script
falls back to an already-installed ``sam3`` package.

    conda activate sam3
    python scripts/step2_mask_tracking.py --video assets/dog-example.mp4 --text dog

Output format  (written to ``<output_dir>/<video_name>/``)
-----------------------------------------------------------
    frames/000000.jpg ...  RGB frames actually processed (subsampled per the
                           --frame_* args), indexed 0..N-1.
    masks_rle.json         CANONICAL artifact, consumed by Step 3 (SAM3D).
                           A dict keyed by frame-index string ->
                             {mask_rle, sam3_score, score_edge,
                              score_solidity, total, source}
                           mask_rle is pycocotools RLE {size:[H,W],
                           counts:"<utf-8 str>"}, or null when the object is
                           absent on that frame.
    meta.json              Metadata + conventions: text label, method, model,
                           best_frame, top_k_frames, num_frames, coverage,
                           and source_frame_indices (output index -> original
                           video frame number, so a Step 1 result on the same
                           video can be aligned).
    viz/overlay.mp4        Mask-overlay preview for QA (unless --no_viz).

Conventions
-----------
* Masks are emitted at the native video frame resolution (== the RGB frames).
* Frame indexing: outputs are indexed 0..N-1; meta.json["source_frame_indices"]
  maps each output index back to the original video frame number.
* A frame with no object simply has mask_rle == null -- the tracklet need not
  cover the whole video; downstream tracks only the valid-mask frames.
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM3_REPO = REPO_ROOT / "third_party" / "sam3"
DEFAULT_SAM3_CKPT = REPO_ROOT / "third_party" / "checkpoints" / "sam3.pt"
DEFAULT_CONF_THRESHOLD = 0.3
SAM3_HF_REPO = "facebook/sam3"
SAM3_HF_CKPT = "sam3.pt"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [step2] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Mask helpers (borrowed from the droid pipeline's step3a1_wrist_sam3_image.py)
# ---------------------------------------------------------------------------

def encode_mask_rle(mask_bool: np.ndarray) -> dict:
    """Encode a bool mask as a JSON-safe pycocotools RLE."""
    rle = mask_utils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return {"size": rle["size"], "counts": rle["counts"]}


def decode_mask_rle(rle: dict) -> np.ndarray:
    """Decode a JSON RLE dict back to a bool mask."""
    return mask_utils.decode(rle).astype(bool)


def compute_edge_score(mask_bool: np.ndarray) -> float:
    """1 - (outer-contour pixels on image edge) / total contour pixels.

    Penalises masks that are clipped by the frame border (likely partial /
    runaway segmentations).
    """
    mask_u8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    pts = c.reshape(-1, 2)
    if pts.shape[0] == 0:
        return 0.0
    H, W = mask_bool.shape
    on_edge = ((pts[:, 0] == 0) | (pts[:, 0] == W - 1) |
               (pts[:, 1] == 0) | (pts[:, 1] == H - 1)).sum()
    return 1.0 - float(on_edge) / pts.shape[0]


def compute_solidity(mask_bool: np.ndarray) -> float:
    """mask area / convex-hull area for the largest outer contour (diagnostic)."""
    mask_u8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0:
        return 0.0
    return float(cv2.contourArea(c)) / hull_area


def mask_metrics(mask_bool: np.ndarray) -> tuple[float, float]:
    """Return (score_edge, score_solidity) for a mask."""
    return compute_edge_score(mask_bool), compute_solidity(mask_bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two bool masks of the same shape."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def _to_list(x) -> list:
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, frames_dir: Path,
                   frame_start: int, frame_end: int, frame_skip: int) -> dict:
    """Decode the (optionally subsampled) RGB frames to JPEGs under frames_dir.

    Returns {"source_frame_indices": [...], "height": H, "width": W, "fps": f}.
    Output frame j is written as frames_dir/{j:06d}.jpg.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    source_frame_indices: list[int] = []
    src_idx = 0
    out_idx = 0
    H = W = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        in_range = src_idx >= frame_start and (frame_end < 0 or src_idx < frame_end)
        if in_range and (src_idx - frame_start) % frame_skip == 0:
            if H is None:
                H, W = frame.shape[:2]
            cv2.imwrite(str(frames_dir / f"{out_idx:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            source_frame_indices.append(src_idx)
            out_idx += 1
        src_idx += 1
    cap.release()

    if out_idx == 0:
        raise RuntimeError(
            f"no frames extracted from {video_path} "
            f"(frame_start={frame_start}, frame_end={frame_end}, frame_skip={frame_skip})")
    eff_fps = (src_fps / frame_skip) if src_fps > 0 else 10.0
    log(f"extracted {out_idx} frame(s) at {W}x{H} -> {frames_dir}")
    return {"source_frame_indices": source_frame_indices,
            "height": int(H), "width": int(W), "fps": float(eff_fps)}


# ---------------------------------------------------------------------------
# SAM 3 setup
# ---------------------------------------------------------------------------

def add_sam3_to_path(sam3_repo: Path) -> None:
    """Ensure `import sam3` resolves to the pinned submodule when present.

    Handles either layout: the package as ``<repo>/sam3`` or the repo root
    itself being the package.  If no local package is found, do nothing and
    rely on an already-installed ``sam3`` in the environment.
    """
    for pkg in (sam3_repo / "sam3", sam3_repo):
        if (pkg / "__init__.py").exists() and (pkg / "model_builder.py").exists():
            parent = str(pkg.parent.resolve())
            if parent not in sys.path:
                sys.path.insert(0, parent)
            log(f"using SAM 3 package at {pkg}")
            return
    log(f"no SAM 3 package under {sam3_repo}; relying on an installed `sam3`")


def ensure_checkpoint(ckpt_path: Path) -> Path:
    """Return a local SAM 3 checkpoint path.

    If the checkpoint is absent, try to fetch it from the Hugging Face Hub.
    Note that ``facebook/sam3`` is a *gated* repo, so the download only succeeds
    if the caller has been granted access and is authenticated (an HF token via
    ``huggingface-cli login`` or the ``HF_TOKEN`` env var).  Otherwise the
    checkpoint must be placed at ``ckpt_path`` manually.
    """
    if ckpt_path.exists():
        log(f"SAM 3 checkpoint: {ckpt_path}")
        return ckpt_path
    log(f"SAM 3 checkpoint not found at {ckpt_path}; attempting download of "
        f"{SAM3_HF_REPO}:{SAM3_HF_CKPT} from the Hugging Face Hub ...")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        got = Path(hf_hub_download(repo_id=SAM3_HF_REPO, filename=SAM3_HF_CKPT,
                                   local_dir=str(ckpt_path.parent)))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"\n[step2] Could not obtain the SAM 3 checkpoint automatically:\n"
            f"    {type(e).__name__}: {e}\n\n"
            f"`{SAM3_HF_REPO}` is a gated Hugging Face repo. Either:\n"
            f"  (a) request access at https://huggingface.co/{SAM3_HF_REPO}, "
            f"authenticate\n"
            f"      (`huggingface-cli login` or set HF_TOKEN), then re-run; or\n"
            f"  (b) place the checkpoint file directly at\n"
            f"        {ckpt_path}\n"
            f"      (or point --sam3_checkpoint at an existing copy).\n"
        ) from e
    log(f"downloaded SAM 3 checkpoint -> {got}")
    return got


# ---------------------------------------------------------------------------
# Phase 1: per-frame SAM 3 image-mode segmentation (text-only prompt)
# ---------------------------------------------------------------------------

def sam3_image_top1(processor, rgb: np.ndarray, prompt: str):
    """SAM 3 image mode, text-only prompt. Returns (mask_bool, score) or (None, 0)."""
    import torch
    from PIL import Image
    with torch.inference_mode():
        pil = Image.fromarray(rgb)
        state = processor.set_image(pil)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=prompt, state=state)
        scores = state.get("scores")
        masks = state.get("masks")
        if scores is None or masks is None or scores.numel() == 0 or not masks.any():
            return None, 0.0
        best = int(scores.argmax().item())
        mask = masks[best, 0].cpu().numpy().astype(bool)
        if not mask.any():
            return None, 0.0
        return mask, float(scores[best].item())


def run_image_phase(frames_dir: Path, num_frames: int, text: str,
                    ckpt_path: Path, conf_threshold: float):
    """Run text-only SAM 3 image mode on every frame.

    Returns (per_frame, best_frame, best_mask):
      per_frame -- {idx: {mask_rle|None, sam3_score, score_edge,
                          score_solidity, total}}
      best_frame -- index of the highest-`total` frame, or None
      best_mask  -- bool mask on best_frame, or None
    """
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    log("Phase 1: loading SAM 3 image model ...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    autocast.__enter__()
    model = build_sam3_image_model(checkpoint_path=str(ckpt_path), load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=conf_threshold)
    log(f"Phase 1: segmenting {num_frames} frame(s), text prompt = {text!r}")

    per_frame: dict[int, dict] = {}
    best_frame = None
    best_total = -1.0
    best_mask = None
    num_detected = 0
    t0 = time.time()

    for idx in range(num_frames):
        bgr = cv2.imread(str(frames_dir / f"{idx:06d}.jpg"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            mask, score = sam3_image_top1(processor, rgb, text)
        except Exception as e:  # noqa: BLE001 -- one bad frame must not abort the run
            log(f"  frame {idx}: image-mode error: {e}")
            per_frame[idx] = {"mask_rle": None, "sam3_score": None,
                              "score_edge": 0.0, "score_solidity": 0.0,
                              "total": 0.0, "error": str(e)}
            continue
        if mask is None:
            per_frame[idx] = {"mask_rle": None, "sam3_score": 0.0,
                              "score_edge": 0.0, "score_solidity": 0.0,
                              "total": 0.0}
            continue
        s_edge, s_sol = mask_metrics(mask)
        total = score * s_edge
        per_frame[idx] = {"mask_rle": encode_mask_rle(mask), "sam3_score": score,
                          "score_edge": s_edge, "score_solidity": s_sol,
                          "total": total}
        num_detected += 1
        if total > best_total:
            best_total, best_frame, best_mask = total, idx, mask

    elapsed = time.time() - t0
    autocast.__exit__(None, None, None)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    log(f"Phase 1: {num_detected}/{num_frames} frame(s) detected "
        f"({elapsed:.1f}s, {elapsed / max(1, num_frames) * 1000:.0f} ms/frame); "
        f"best_frame={best_frame} (total={best_total:.3f})"
        if best_frame is not None else
        f"Phase 1: 0/{num_frames} frame(s) detected ({elapsed:.1f}s)")
    return per_frame, best_frame, best_mask


# ---------------------------------------------------------------------------
# Phase 2: SAM 3 video-predictor bidirectional propagation
# ---------------------------------------------------------------------------

def run_video_phase(frames_dir: Path, num_frames: int, best_frame: int,
                    best_mask: np.ndarray, text: str, ckpt_path: Path):
    """Seed the SAM 3 video predictor on best_frame and propagate bidirectionally.

    The seed is a *text* prompt. SAM 3 video grounding propagates a text concept
    across the whole clip; a box / visual prompt, by contrast, is attached only
    to its single frame and leaves nothing for propagation to carry (verified:
    box-seeding tracks 0 frames, text-seeding tracks the full clip). Anchoring
    the seed on the image-mode-chosen best_frame keeps grounding reliable.

    Returns (masks, info):
      masks -- {idx: mask_bool} for frames where the tracked object is present.
      info  -- dict with seeding / propagation diagnostics.
    """
    import torch
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    log("Phase 2: loading SAM 3 video predictor ...")
    predictor = Sam3VideoPredictor(checkpoint_path=str(ckpt_path))

    info = {"seed_frame": best_frame, "seed_ok": False,
            "num_seed_objects": 0, "tracked_obj_id": None}
    masks: dict[int, np.ndarray] = {}

    resp = predictor.handle_request(
        dict(type="start_session", resource_path=str(frames_dir)))
    session_id = resp["session_id"]
    try:
        # ---- seed best_frame with the text prompt --------------------------
        log(f"Phase 2: seeding frame {best_frame} with text {text!r}")
        seed_resp = predictor.handle_request(dict(
            type="add_prompt", session_id=session_id, frame_index=best_frame,
            text=text))
        seed_out = seed_resp["outputs"]
        seed_ids = _to_list(seed_out["out_obj_ids"])
        info["num_seed_objects"] = len(seed_ids)
        if not seed_ids:
            log("Phase 2: WARNING -- seed produced no object; aborting propagation")
            return masks, info
        info["seed_ok"] = True

        # If the seed yields multiple instances, keep the one that best matches
        # the Phase-1 image-mode detection on best_frame.
        seed_masks = [_to_numpy(m).astype(bool) for m in seed_out["out_binary_masks"]]
        ious = [mask_iou(m, best_mask) for m in seed_masks]
        pick = int(np.argmax(ious)) if ious else 0
        target_obj_id = seed_ids[pick]
        info["tracked_obj_id"] = target_obj_id
        if len(seed_ids) > 1:
            log(f"Phase 2: seed produced {len(seed_ids)} objects {seed_ids}; "
                f"tracking obj_id={target_obj_id} (IoU={ious[pick]:.3f} with seed mask)")

        # ---- bidirectional propagation -------------------------------------
        log(f"Phase 2: propagating both directions from frame {best_frame} ...")
        t0 = time.time()
        for r in predictor.handle_stream_request(dict(
                type="propagate_in_video", session_id=session_id,
                propagation_direction="both", start_frame_index=best_frame)):
            fi = int(r["frame_index"])
            out = r["outputs"]
            obj_ids = _to_list(out["out_obj_ids"])
            if target_obj_id not in obj_ids:
                continue
            m = _to_numpy(out["out_binary_masks"][obj_ids.index(target_obj_id)])
            m = m.astype(bool)
            if m.any():
                masks[fi] = m
        info["propagation_seconds"] = time.time() - t0
        log(f"Phase 2: tracked object on {len(masks)}/{num_frames} frame(s) "
            f"({info['propagation_seconds']:.1f}s)")
    finally:
        predictor.handle_request(dict(type="close_session", session_id=session_id))
        predictor.shutdown()
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    return masks, info


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def build_per_frame(method: str, num_frames: int, image_per_frame: dict,
                    video_masks: dict) -> dict:
    """Merge Phase 1 / Phase 2 results into the canonical per-frame entries."""
    out: dict[str, dict] = {}
    for idx in range(num_frames):
        img = image_per_frame.get(idx, {})
        img_score = img.get("sam3_score")
        img_total = img.get("total", 0.0)
        if method == "image":
            entry = dict(img)
            entry["source"] = "image" if entry.get("mask_rle") is not None else None
        else:  # hybrid
            vm = video_masks.get(idx)
            if vm is not None:
                s_edge, s_sol = mask_metrics(vm)
                entry = {"mask_rle": encode_mask_rle(vm),
                         "sam3_score": img_score,
                         "score_edge": s_edge, "score_solidity": s_sol,
                         "total": img_total, "source": "propagated"}
            else:
                entry = {"mask_rle": None, "sam3_score": img_score,
                         "score_edge": 0.0, "score_solidity": 0.0,
                         "total": img_total, "source": None}
        out[str(idx)] = entry
    return out


def rank_top_k(image_per_frame: dict, k: int = 10) -> list[dict]:
    """Top-k frames by Phase-1 image-mode `total` (for downstream frame selection)."""
    scored = [
        {"t": idx, "total": e["total"], "sam3_score": e["sam3_score"],
         "score_edge": e["score_edge"], "score_solidity": e["score_solidity"]}
        for idx, e in image_per_frame.items() if e.get("mask_rle") is not None
    ]
    scored.sort(key=lambda d: -d["total"])
    return scored[:k]


def _render_overlay_frame(frame: np.ndarray, entry: dict, idx: int,
                          best_frame) -> np.ndarray:
    """Draw the mask overlay (fill + contour) and a frame label onto a BGR frame."""
    rle = entry.get("mask_rle")
    if rle is not None:
        mask = decode_mask_rle(rle)
        overlay = frame.copy()
        overlay[mask] = (0, 200, 0)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    tag = f"frame {idx}"
    if idx == best_frame:
        tag += "  [SEED]"
    elif rle is None:
        tag += "  [no mask]"
    cv2.putText(frame, tag, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, tag, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def write_viz(frames_dir: Path, per_frame: dict, num_frames: int,
              best_frame, fps: float, viz_path: Path) -> None:
    """Render a mask-overlay preview MP4.

    Encodes H.264 (yuv420p) via ffmpeg when available. OpenCV's built-in
    ``mp4v`` writer emits an MPEG-4 Part 2 stream that many players (browsers,
    QuickTime, IDE preview panes) render with green / corrupt frames, so ffmpeg
    is strongly preferred; the ``mp4v`` writer is only a fallback for when
    ffmpeg is not installed.
    """
    import shutil
    import subprocess

    viz_path.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frames_dir / "000000.jpg"))
    H, W = first.shape[:2]
    fps = max(float(fps), 1.0)

    ffmpeg = shutil.which("ffmpeg")
    proc = writer = None
    if ffmpeg:
        # Pipe raw BGR frames straight into ffmpeg -> H.264 (plays everywhere).
        proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pixel_format", "bgr24",
             "-video_size", f"{W}x{H}", "-framerate", f"{fps:.4f}", "-i", "-",
             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-movflags", "+faststart", str(viz_path)],
            stdin=subprocess.PIPE)
    else:
        writer = cv2.VideoWriter(str(viz_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (W, H))

    for idx in range(num_frames):
        frame = cv2.imread(str(frames_dir / f"{idx:06d}.jpg"))
        frame = _render_overlay_frame(frame, per_frame.get(str(idx), {}),
                                      idx, best_frame)
        if proc is not None:
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        else:
            writer.write(frame)

    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg encoding failed (exit {proc.returncode})")
        log(f"wrote overlay preview (H.264) -> {viz_path}")
    else:
        writer.release()
        log(f"wrote overlay preview (mp4v fallback; install ffmpeg for H.264) "
            f"-> {viz_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: per-frame 2D object-mask tracklet from an RGB "
                    "video + a text label (SAM 3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, type=Path,
                        help="Input RGB video file.")
    parser.add_argument("--text", required=True, type=str,
                        help="Free-text object/category label to track.")
    parser.add_argument("--output_dir", type=Path,
                        default=REPO_ROOT / "output" / "step2",
                        help="Base output dir; results go to <output_dir>/<video_name>/.")
    parser.add_argument("--method", choices=["hybrid", "image"], default="hybrid",
                        help="hybrid = image-mode seed + video-predictor "
                             "propagation; image = per-frame image mode only.")
    parser.add_argument("--sam3_repo", type=Path, default=DEFAULT_SAM3_REPO,
                        help="Path to the SAM 3 repository (submodule).")
    parser.add_argument("--sam3_checkpoint", type=Path, default=DEFAULT_SAM3_CKPT,
                        help="Path to the SAM 3 checkpoint (auto-downloaded if missing).")
    parser.add_argument("--sam3_confidence_threshold", type=float,
                        default=DEFAULT_CONF_THRESHOLD,
                        help="SAM 3 image-mode detection confidence threshold.")
    parser.add_argument("--frame_start", type=int, default=0,
                        help="First video frame to process.")
    parser.add_argument("--frame_end", type=int, default=-1,
                        help="Last video frame (exclusive); -1 means end of video.")
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="Process every Nth frame (1 = every frame).")
    parser.add_argument("--no_viz", action="store_true",
                        help="Skip the mask-overlay preview MP4.")
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.exists():
        parser.error(f"--video does not exist: {video}")
    if args.frame_skip < 1:
        parser.error("--frame_skip must be >= 1")

    out_dir = (args.output_dir / video.stem).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    log(f"input video : {video}")
    log(f"text label  : {args.text!r}")
    log(f"output dir  : {out_dir}")
    log(f"method      : {args.method}")

    # --- SAM 3 setup --------------------------------------------------------
    add_sam3_to_path(args.sam3_repo.resolve())
    ckpt_path = ensure_checkpoint(args.sam3_checkpoint.resolve())

    # --- frame extraction ---------------------------------------------------
    frame_info = extract_frames(video, frames_dir, args.frame_start,
                                args.frame_end, args.frame_skip)
    num_frames = len(frame_info["source_frame_indices"])

    # --- Phase 1: per-frame image-mode segmentation -------------------------
    image_per_frame, best_frame, best_mask = run_image_phase(
        frames_dir, num_frames, args.text, ckpt_path,
        args.sam3_confidence_threshold)

    # --- Phase 2: video-predictor propagation -------------------------------
    video_masks: dict[int, np.ndarray] = {}
    video_info: dict = {}
    method = args.method
    if method == "hybrid":
        if best_frame is None:
            log("hybrid requested but Phase 1 found no object on any frame; "
                "falling back to method=image (empty tracklet).")
            method = "image"
        else:
            try:
                video_masks, video_info = run_video_phase(
                    frames_dir, num_frames, best_frame, best_mask,
                    args.text, ckpt_path)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                log(f"Phase 2 failed ({e}); falling back to method=image.")
                method = "image"
                video_info = {"error": str(e)}
            else:
                if not video_masks:
                    log("Phase 2 produced no masks; falling back to method=image.")
                    method = "image"

    # --- assemble + write output -------------------------------------------
    per_frame = build_per_frame(method, num_frames, image_per_frame, video_masks)
    num_detected = sum(1 for e in per_frame.values() if e.get("mask_rle") is not None)
    image_detected = sum(1 for e in image_per_frame.values()
                         if e.get("mask_rle") is not None)
    top_k = rank_top_k(image_per_frame)

    with open(out_dir / "masks_rle.json", "w") as f:
        json.dump(per_frame, f)

    meta = {
        "step": "step2_mask_tracking",
        "generated_by": "scripts/step2_mask_tracking.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_video": str(video),
        "video_name": video.stem,
        "text": args.text,
        "model": "SAM3",
        "method": method,
        "method_requested": args.method,
        "num_frames": num_frames,
        "num_detected": num_detected,
        "coverage": num_detected / num_frames if num_frames else 0.0,
        "image_mode_num_detected": image_detected,
        "image_mode_coverage": image_detected / num_frames if num_frames else 0.0,
        "best_frame": best_frame,
        "top_k_frames": top_k,
        "video_phase": video_info,
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "frame_skip": args.frame_skip,
        "source_frame_indices": frame_info["source_frame_indices"],
        "image_width": frame_info["width"],
        "image_height": frame_info["height"],
        "fps": frame_info["fps"],
        "masks_rle": {
            "file": "masks_rle.json",
            "layout": "dict keyed by frame-index string -> per-frame entry",
            "entry": "{mask_rle, sam3_score, score_edge, score_solidity, "
                     "total, source}",
            "mask_rle": "pycocotools RLE {size:[H,W], counts:<utf-8 str>}, "
                        "or null when the object is absent on that frame",
            "scores": "sam3_score/score_edge/score_solidity/total are from the "
                      "Phase-1 image-mode pass (total = sam3_score * score_edge)",
            "source": "'propagated' (video predictor), 'image' (image mode), "
                      "or null (no mask)",
        },
        "frames": {"layout": "frames/{index:06d}.jpg", "color": "BGR-encoded JPEG"},
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote masks_rle.json + meta.json  ({num_detected}/{num_frames} "
        f"frame(s) with a mask, coverage={meta['coverage']:.2f})")

    if not args.no_viz:
        try:
            write_viz(frames_dir, per_frame, num_frames, best_frame,
                      frame_info["fps"], out_dir / "viz" / "overlay.mp4")
        except Exception as e:  # noqa: BLE001 -- viz is non-critical
            log(f"viz failed (non-critical): {e}")

    log("Step 2 complete.")


if __name__ == "__main__":
    main()
