"""Frame access with the sampling rule the tracking contract depends on."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


class VideoError(RuntimeError):
    pass


def _open(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError(f"cannot open video: {path}")
    return cap


def sample(path: str, start: float, duration: float, rate: float) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t, frame) with t relative to `start`.

    Not a fixed grid: the first frame at or after `start` is taken, then the
    first frame at or after previous + 1/rate. This is what downstream tools
    were tuned against, so keep it even though a grid would look tidier.
    """
    cap = _open(path)
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        end = start + duration
        nxt = start
        while cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if t + 1e-6 < start:
                continue
            if t >= end - 1e-6:
                break
            if t + 1e-6 < nxt:
                continue
            nxt = t + 1 / rate
            ok, frame = cap.retrieve()
            if ok:
                yield round(t - start, 3), frame
    finally:
        cap.release()


def frames(path: str) -> Iterator[np.ndarray | None]:
    """Every frame in decode order; None for a frame that fails to decode."""
    cap = _open(path)
    try:
        while cap.grab():
            ok, frame = cap.retrieve()
            yield frame if ok else None
    finally:
        cap.release()
