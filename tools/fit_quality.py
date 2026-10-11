"""Fit the capture-quality weights against reference scores.

Usage: fit_quality.py <train.json> [<holdout.json>] [--root DIR] [--write]

References are in framesight's faces schema, with the reference "quality" per
face; image paths are resolved against --root (default: the current
directory). Faces are matched by IoU, the features from framesight.quality are
regressed onto the reference quality, and the fitted weights are printed (or
written into quality.py with --write).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from framesight import quality
from framesight.faces import FaceAnalyzer


def iou(a, b):
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-12)


def collect(ref_path: Path, root: Path, fa: FaceAnalyzer, keys: list[str]):
    ref = json.load(open(ref_path))
    X, y, missed = [], [], 0
    for path, entry in ref.items():
        img = cv2.imread(str(root / path))
        if img is None or "faces" not in entry:
            continue
        h, w = img.shape[:2]
        faces = fa.detect(img)
        for rf in entry["faces"]:
            r = rf["box"]
            best = max(faces, key=lambda f: iou(r[:4], [f.box[0] / w, f.box[1] / h, f.box[2] / w, f.box[3] / h]),
                       default=None)
            if best is None or iou(r[:4], [best.box[0] / w, best.box[1] / h, best.box[2] / w, best.box[3] / h]) < 0.2:
                missed += 1
                continue
            f = quality.features(img, best)
            X.append([f[k] for k in keys])
            y.append(rf["quality"])
    return np.array(X), np.array(y), missed


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def main() -> int:
    args = sys.argv[1:]
    root = Path(args.pop(args.index("--root") + 1)) if "--root" in args else Path(".")
    files = [Path(a) for a in args if not a.startswith("--")]
    fa = FaceAnalyzer(threads=8)
    keys = [k for k in quality.WEIGHTS if k != "bias"]
    X, y, missed = collect(files[0], root, fa, keys)
    A = np.c_[np.ones(len(X)), X]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = np.clip(A @ coef, 0, 1)
    print(f"train faces {len(y)} (unmatched {missed})  corr {np.corrcoef(pred, y)[0, 1]:.3f}  "
          f"spearman {spearman(pred, y):.3f}  MAE {np.mean(np.abs(pred - y)):.3f}")
    if len(files) > 1:
        Xh, yh, mh = collect(files[1], root, fa, keys)
        ph = np.clip(np.c_[np.ones(len(Xh)), Xh] @ coef, 0, 1)
        print(f"holdout faces {len(yh)} (unmatched {mh})  corr {np.corrcoef(ph, yh)[0, 1]:.3f}  "
              f"spearman {spearman(ph, yh):.3f}  MAE {np.mean(np.abs(ph - yh)):.3f}")
    for k, v in zip(keys, X.T):
        print(f"  {k:8} corr {np.corrcoef(v, y)[0, 1]:+.3f}")
    weights = {"bias": round(float(coef[0]), 4), **{k: round(float(c), 4) for k, c in zip(keys, coef[1:])}}
    print("WEIGHTS =", weights)
    if "--write" in sys.argv:
        p = Path(quality.__file__)
        p.write_text(re.sub(r"WEIGHTS = \{.*?\}", f"WEIGHTS = {weights}", p.read_text(), flags=re.S))
    return 0


if __name__ == "__main__":
    sys.exit(main())
