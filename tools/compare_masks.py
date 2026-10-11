"""Compare framesight masks with a reference mask stream.

Usage: compare_masks.py <video> <reference.mask[.zst]> [--model rvm] [--work 640x360] [--frames N]

The reference is the "WxH\\n" + raw 8-bit frames format that `framesight mask`
writes; zstd-compressed references are read through the `zstd` binary.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np

from framesight import video
from framesight.mask import Segmenter


def reference_frames(path: str):
    if path.endswith(".zst"):
        proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
        stream = proc.stdout
    else:
        stream = open(path, "rb")
    w, h = (int(v) for v in stream.readline().decode().strip().split("x"))
    while True:
        buf = stream.read(w * h)
        if len(buf) < w * h:
            return
        yield (w, h), np.frombuffer(buf, np.uint8).reshape(h, w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("reference")
    ap.add_argument("--model", default="rvm")
    ap.add_argument("--work", default="640x360")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ratio", type=float, default=None)
    a = ap.parse_args()
    seg = Segmenter(a.model, a.threads, tuple(int(v) for v in a.work.split("x")), a.ratio)
    ious, maes, edge = [], [], []
    spent = 0.0
    for n, (frame, ((w, h), ref)) in enumerate(zip(video.frames(a.video), reference_frames(a.reference))):
        if a.frames and n >= a.frames:
            break
        t = time.perf_counter()
        m = seg(frame, (w, h)) if frame is not None else np.zeros((h, w), np.float32)
        spent += time.perf_counter() - t
        ours = (m * 255).astype(np.uint8)
        r, o = ref >= 128, ours >= 128
        union = (r | o).sum()
        ious.append((r & o).sum() / union if union else 1.0)
        maes.append(np.abs(ref.astype(np.int16) - ours).mean() / 255)
        # disagreement on the reference's soft band, where edge quality shows
        band = (ref > 16) & (ref < 240)
        edge.append(np.abs(ref[band].astype(np.int16) - ours[band]).mean() / 255 if band.any() else 0.0)
    ious = np.array(ious)
    print(f"{a.model}@{a.work} r{seg.ratio:.2f} frames {len(ious)} IoU mean {ious.mean():.3f} p10 {np.percentile(ious, 10):.3f} "
          f"min {ious.min():.3f}  MAE {np.mean(maes):.4f}  edge-MAE {np.mean(edge):.3f}  {spent / max(1, len(ious)) * 1000:.0f} ms/frame",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
