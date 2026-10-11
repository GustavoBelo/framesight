"""Regression report: candidate vs reference, both in framesight's schema.

Usage:
  compare.py track <reference.track.json> <candidate.track.json> [...pairs]
  compare.py faces <reference.faces.json> <candidate.faces.json>

Different detectors draw boxes differently, so faces and people are matched
loosely (IoU >= 0.2). The numbers are for spotting regressions between
versions, not for approving a model: when two tools disagree, look at the
frames before deciding which one is wrong.
"""

from __future__ import annotations

import json
import sys

import numpy as np

MATCH = 0.2


def iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-12)


def match(ref, cand):
    """Greedy one-to-one matching by IoU; returns (ref index, cand index, iou)."""
    pairs, ri, ci = [], set(), set()
    for v, i, j in sorted(((iou(r["box"], c["box"]), i, j) for i, r in enumerate(ref) for j, c in enumerate(cand)),
                          reverse=True):
        if v < MATCH or i in ri or j in ci:
            continue
        ri.add(i); ci.add(j); pairs.append((i, j, v))
    return pairs


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")


def main_face(faces, min_score=0.5):
    fs = [f for f in faces if f["score"] >= min_score]
    return max(fs, key=lambda f: f["box"][2] * f["box"][3]) if fs else None


def track(ref_path: str, cand_path: str) -> None:
    ref = {s["t"]: s for s in json.load(open(ref_path))["samples"]}
    cand = {s["t"]: s for s in json.load(open(cand_path))["samples"]}
    common = sorted(set(ref) & set(cand))
    stats = {"faces": [0, 0, 0, []], "people": [0, 0, 0, []]}
    pose = []
    for t in common:
        for kind in stats:
            r, c = ref[t][kind], cand[t][kind]
            m = match(r, c)
            st = stats[kind]
            st[0] += len(r); st[1] += len(c); st[2] += len(m); st[3] += [v for *_, v in m]
        mr, mc = main_face(ref[t]["faces"]), main_face(cand[t]["faces"])
        if mr and mc and iou(mr["box"], mc["box"]) >= MATCH and "pitch" in mc and "pitch" in mr:
            pose.append((mr["pitch"], mc["pitch"], mr["yaw"], mc["yaw"], mr["mouth"], mc["mouth"]))
    print(f"{cand_path}: samples ref {len(ref)} cand {len(cand)} common {len(common)}")
    for kind, (nr, nc, nm, ious) in stats.items():
        print(f"  {kind:6}: recall {nm / max(nr, 1):.3f} ({nm}/{nr})  extra {nc - nm}  mean IoU {np.mean(ious) if ious else 0:.3f}")
    if pose:
        p = np.array(pose)
        print(f"  main face ({len(p)}): pitch MAE {np.mean(np.abs(p[:, 0] - p[:, 1])):.3f} corr {corr(p[:, 0], p[:, 1]):.2f}"
              f" | head-down>=0.3 agree {np.mean((p[:, 0] >= 0.3) == (p[:, 1] >= 0.3)):.2f}"
              f" | yaw MAE {np.mean(np.abs(p[:, 2] - p[:, 3])):.3f} corr {corr(p[:, 2], p[:, 3]):.2f}"
              f" | mouth corr {corr(p[:, 4], p[:, 5]):.2f}")


def faces(ref_path: str, cand_path: str) -> None:
    ref, cand = json.load(open(ref_path)), json.load(open(cand_path))
    nr = nm = 0
    q = []
    errors = 0
    for path, r in ref.items():
        c = cand.get(path, {"error": "missing"})
        if ("error" in r) != ("error" in c):
            errors += 1
        if "error" in r or "error" in c:
            continue
        nr += len(r["faces"])
        for i, j, _ in match(r["faces"], c["faces"]):
            nm += 1
            if "quality" in r["faces"][i] and "quality" in c["faces"][j]:
                q.append((r["faces"][i]["quality"], c["faces"][j]["quality"]))
    q = np.array(q)
    print(f"{cand_path}: images {len(ref)} error mismatches {errors}  faces recall {nm / max(nr, 1):.3f} ({nm}/{nr})"
          + (f"  quality corr {corr(q[:, 0], q[:, 1]):.2f} MAE {np.mean(np.abs(q[:, 0] - q[:, 1])):.3f}" if len(q) else ""))


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[1] not in ("track", "faces") or len(sys.argv) % 2:
        print(__doc__, file=sys.stderr)
        return 2
    fn = track if sys.argv[1] == "track" else faces
    for i in range(2, len(sys.argv), 2):
        fn(sys.argv[i], sys.argv[i + 1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
