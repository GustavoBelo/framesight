"""Fit the upper-body rectangle parameters against reference boxes.

Usage: fit_people.py <clips dir> <reference dir>

For every <name>.mp4 in the clips dir, the reference dir holds a
<name>.track.json in framesight's track schema, sampled at 5/s from the
start, whose people boxes are the upper-body rectangles to imitate. Prints
the fitted UPPER_BODY dict and per-clip IoU, so the constants in people.py
can be updated by hand.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys

import cv2
import numpy as np
from scipy.optimize import minimize

from framesight import video
from framesight.faces import FaceAnalyzer
from framesight.people import UPPER_BODY, PersonDetector, attach_upper_bodies

KEYS = list(UPPER_BODY)


def iou(a, b):
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-12)


def collect(ref: str, clips: list[str]):
    cache = os.path.join(ref, "people_features.pkl")  # detection is the slow part; fitting reruns often
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    fa, pd = FaceAnalyzer(threads=8, mesh=False), PersonDetector()
    data = []
    for clip in clips:
        name = os.path.basename(clip)[:-4]
        gold = {s["t"]: s for s in json.load(open(f"{ref}/{name}.track.json"))["samples"]}
        for t, frame in video.sample(clip, 0, 1e9, 5):
            if t not in gold:
                continue
            h, w = frame.shape[:2]
            data.append((name, w, h, fa.detect(frame), pd.detect(frame), [p["box"] for p in gold[t]["people"]]))
        print(name, file=sys.stderr)
    for d in data:  # drop the unpicklable bits
        for f in d[3]:
            f.mesh = None
    pickle.dump(data, open(cache, "wb"))
    return data


def evaluate(data, k):
    per = {}
    for name, w, h, faces, people, gold in data:
        attach_upper_bodies(people, faces, k)
        ours = [(p.upper[0] / w, p.upper[1] / h, p.upper[2] / w, p.upper[3] / h) for p in people]
        used = set()
        for g in gold:
            best, bj = 0.0, None
            for j, o in enumerate(ours):
                if j not in used and (v := iou(g, o)) > best:
                    best, bj = v, j
            if bj is not None:
                used.add(bj)
            per.setdefault(name, []).append(best)
    return per


def main() -> int:
    clips = sorted(glob.glob(f"{sys.argv[1]}/*.mp4"))
    ref = sys.argv[2]
    cv2.setNumThreads(8)
    data = collect(ref, clips)

    def loss(x):
        per = evaluate(data, dict(zip(KEYS, x)))
        return -np.mean([v for vs in per.values() for v in vs])

    x0 = np.array([UPPER_BODY[k] for k in KEYS])
    res = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-4})
    k = dict(zip(KEYS, res.x.round(3).tolist()))
    print("UPPER_BODY =", k)
    for name, vs in evaluate(data, k).items():
        vs = np.array(vs)
        print(f"{name}: n {len(vs)} meanIoU {vs.mean():.3f} IoU>=0.5 {np.mean(vs >= 0.5):.2f} missed {np.mean(vs == 0):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
