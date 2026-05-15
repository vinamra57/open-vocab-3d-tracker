# RADIO-ViPE local patches

Patches applied to the `third_party/RADIO-ViPE` submodule **checkout only**.
They are intentionally *not* pushed to the RADIO-ViPE upstream repository — they
live solely in our local working tree so the pipeline runs correctly.

Re-apply after the submodule is reset / re-checked-out:

```bash
cd third_party/RADIO-ViPE
git apply ../patches/<name>.patch
```

## radio-vipe-extract_slam_map-none-guard.patch

Fixes a crash in `vipe/slam/components/buffer.py::extract_slam_map`. The function
sets `staged_emb = None` when a video produced no RADSeg embeddings during SLAM
(this happens when bundle adjustment diverges — common for short, static, or
synthetic clips), then immediately does `staged_emb[...].permute(...)`, raising
`TypeError: 'NoneType' object is not subscriptable`.

The patch passes `None` through in that case; `SLAMMap.from_masked_dense_disp`
already accepts `embeddings=None`. Required for Step 1 to handle arbitrary
in-the-wild videos rather than only well-behaved ones.
