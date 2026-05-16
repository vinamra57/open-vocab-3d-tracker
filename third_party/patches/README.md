# Third-party submodule patches

Local patches applied to the `third_party/` submodule **checkouts only**. They
are intentionally *not* pushed to the upstream repositories — they live solely
in our local working tree so the pipeline runs correctly. Each pipeline step
auto-applies the patch it needs at runtime (idempotent), so normally you do not
have to touch these by hand.

Re-apply manually after a submodule is reset / re-checked-out:

```bash
git -C third_party/<submodule> apply <repo-root>/third_party/patches/<name>.patch
```

## radio-vipe-extract_slam_map-none-guard.patch

Applied to the `third_party/RADIO-ViPE` submodule; auto-applied by Step 1
(`scripts/step1_depth_camera.py`, `ensure_radio_vipe_patched()`).

Fixes a crash in `vipe/slam/components/buffer.py::extract_slam_map`. The function
sets `staged_emb = None` when a video produced no RADSeg embeddings during SLAM
(this happens when bundle adjustment diverges — common for short, static, or
synthetic clips), then immediately does `staged_emb[...].permute(...)`, raising
`TypeError: 'NoneType' object is not subscriptable`.

The patch passes `None` through in that case; `SLAMMap.from_masked_dense_disp`
already accepts `embeddings=None`. Required for Step 1 to handle arbitrary
in-the-wild videos rather than only well-behaved ones.

## sam3d-objects-gt-intrinsics.patch

Applied to the `third_party/sam-3d-objects` submodule (pinned at upstream commit
`81a8237`, the latest `main`); auto-applied by Step 3
(`scripts/step3_sam3d_mesh.py`, `ensure_sam3d_patched()`).

Adds an optional `intrinsics` parameter to `Inference.__call__`,
`InferencePipelinePointMap.run()`, and `InferencePipelinePointMap.compute_pointmap()`.
When the caller supplies an external pointmap, upstream SAM 3D Objects *infers*
the camera intrinsics from that pointmap (its external-pointmap branch hardcodes
`intrinsics = None`). The patch instead lets the caller pass **ground-truth
intrinsics**: Step 3 builds the pointmap from Step 1 depth together with Step 1's
RADIO-ViPE camera intrinsics, and using those GT intrinsics directly — rather
than re-inferring approximations — keeps the reconstruction consistent with the
rest of the pipeline. The change is purely additive: with no intrinsics passed
it falls back to upstream's inference behaviour.

Not fixed upstream: even at the pinned latest `main` (`81a8237`),
`compute_pointmap` / `Inference.__call__` have no `intrinsics` parameter, so the
patch is required.
