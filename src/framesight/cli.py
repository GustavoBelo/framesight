"""framesight: faces, head pose, people and person masks from video, as JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import __version__, models


def _round(v: float, n: int = 4) -> float:
    return round(float(v), n)


def _face_json(f, w: int, h: int, quality: float | None = None) -> dict:
    x, y, bw, bh = f.box
    out = {"box": [_round(x / w), _round(y / h), _round(bw / w), _round(bh / h)], "score": _round(f.score)}
    if quality is not None:
        out["quality"] = _round(quality)
    if f.mesh is not None:
        out.update(pitch=_round(f.pitch), yaw=_round(f.yaw), roll=_round(f.roll), mouth=_round(f.mouth))
    return out


def _person_json(p, w: int, h: int) -> dict:
    x, y, bw, bh = p.upper
    return {"box": [_round(x / w), _round(y / h), _round(bw / w), _round(bh / h)], "score": _round(p.score)}


def cmd_track(a) -> int:
    from . import video
    from .faces import FaceAnalyzer
    from .people import PersonDetector, attach_upper_bodies

    # Small networks scale poorly across cores, so samples are spread over
    # workers that each own their models instead of one model using every core.
    workers = max(1, min(a.workers or a.threads, a.threads))
    per = max(1, a.threads // workers)
    cv2.setNumThreads(per)
    local = threading.local()

    def analyse(item):
        t, frame = item
        if not hasattr(local, "fa"):
            local.fa = FaceAnalyzer(threads=per, score_threshold=a.face_threshold)
            local.pd = PersonDetector(score_threshold=a.person_threshold, threads=per)
        h, w = frame.shape[:2]
        faces, people = local.fa.detect(frame), local.pd.detect(frame)
        attach_upper_bodies(people, faces)
        return w, h, {"t": t, "faces": [_face_json(f, w, h) for f in faces],
                      "people": [_person_json(p, w, h) for p in people]}

    duration = a.duration if a.duration is not None else float("inf")
    frames = video.sample(a.video, a.start, duration, a.rate)
    samples, w, h = [], 0, 0
    with ThreadPoolExecutor(workers) as pool:
        # bounded look-ahead keeps memory flat on long videos
        pending = deque()
        for item in frames:
            pending.append(pool.submit(analyse, item))
            if len(pending) >= 2 * workers:
                w, h, s = pending.popleft().result()
                samples.append(s)
        while pending:
            w, h, s = pending.popleft().result()
            samples.append(s)
    json.dump({"width": w, "height": h, "samples": samples}, sys.stdout, separators=(",", ":"))
    return 0


def _read_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:  # HEIC and friends, when the optional pillow-heif is installed
        from PIL import Image, ImageOps
        import pillow_heif

        pillow_heif.register_heif_opener()
        with Image.open(path) as im:
            return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(im).convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def cmd_faces(a) -> int:
    from .faces import FaceAnalyzer
    from .quality import capture_quality

    fa = FaceAnalyzer(threads=a.threads, score_threshold=a.face_threshold)
    out = {}
    for path in a.images:
        img = _read_image(path)
        if img is None:
            out[path] = {"error": "cannot open"}
            continue
        h, w = img.shape[:2]
        faces = fa.detect(img)
        out[path] = {"width": w, "height": h,
                     "faces": [_face_json(f, w, h, capture_quality(img, f)) for f in faces]}
    json.dump(out, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    return 0


def cmd_mask(a) -> int:
    from . import video
    from .mask import Segmenter

    w, h = (int(v) for v in a.size.lower().split("x"))
    seg = Segmenter(a.model, a.threads, tuple(int(v) for v in a.work.lower().split("x")))
    out = sys.stdout.buffer
    out.write(f"{w}x{h}\n".encode())
    blank = bytes(w * h)
    for frame in video.frames(a.video):
        if frame is None:
            out.write(blank)
            continue
        try:
            m = seg(frame, (w, h))
            out.write(np.clip(m * 255 + 0.5, 0, 255).astype(np.uint8).tobytes())
        except Exception as e:  # one bad frame must not shift every frame after it
            print(f"framesight: mask failed on a frame: {e}", file=sys.stderr)
            out.write(blank)
    out.flush()
    return 0


def cmd_models(a) -> int:
    for name, m in models.REGISTRY.items():
        here = (models.cache_dir() / m.file).exists()
        if a.fetch and not here:
            models.path(name)
            here = True
        print(f"{name:15} {m.license:11} {'cached' if here else 'missing':8} {m.source}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="framesight", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("FRAMESIGHT_THREADS", os.cpu_count() or 1)),
                    help="CPU threads to use (default: all; env FRAMESIGHT_THREADS)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("track", help="sample a video and report faces and upper bodies as JSON")
    t.add_argument("video")
    t.add_argument("--start", type=float, default=0.0, help="seconds")
    t.add_argument("--duration", type=float, default=None, help="seconds (default: to the end)")
    t.add_argument("--rate", type=float, default=5.0, help="samples per second")
    t.add_argument("--face-threshold", type=float, default=0.6)
    t.add_argument("--person-threshold", type=float, default=0.4)
    t.add_argument("--workers", type=int, default=0, help="parallel samples (default: one per thread)")
    t.set_defaults(fn=cmd_track)

    f = sub.add_parser("faces", help="faces in still images, with a capture-quality score")
    f.add_argument("images", nargs="+")
    f.add_argument("--face-threshold", type=float, default=0.6)
    f.set_defaults(fn=cmd_faces)

    m = sub.add_parser("mask", help="person mask for every frame, as raw 8-bit frames on stdout")
    m.add_argument("video")
    m.add_argument("--size", default="640x360", help="output WxH")
    m.add_argument("--model", default="rvm", help="rvm (GPL-3.0 weights) | modnet (Apache-2.0)")
    m.add_argument("--work", default="640x360", help="inference WxH")
    m.set_defaults(fn=cmd_mask)

    s = sub.add_parser("models", help="list models, their licenses and cache state")
    s.add_argument("--fetch", action="store_true", help="download every missing model")
    s.set_defaults(fn=cmd_models)

    a = ap.parse_args(argv)
    try:  # OpenCV's own warnings would end up mixed into callers' stderr logs
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    a.threads = max(1, a.threads)
    cv2.setNumThreads(a.threads)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
