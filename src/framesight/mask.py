"""Per-frame person masks (soft, 0-255)."""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from . import models

MODELS = ("rvm", "modnet")


def _session(path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class Segmenter:
    """Stateful: feed frames in order. RVM carries recurrent state between frames,
    which is what keeps its edges from flickering on video."""

    def __init__(self, model: str = "rvm", threads: int = 4, work: tuple[int, int] = (640, 360),
                 ratio: float | None = None):
        if model not in MODELS:
            raise ValueError(f"unknown mask model {model!r}, choose from {MODELS}")
        self.model = model
        self.work = work
        # RVM refines at full `work` size but segments at work * ratio; ~256 px on the long side is its default.
        self.ratio = ratio if ratio is not None else min(1.0, 256 / max(work))
        self.sess = _session(models.path(model), threads)
        self.reset()

    def reset(self) -> None:
        self.rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4

    def __call__(self, frame: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
        """Soft mask in [0, 1], at `size` (w, h) or the frame's resolution."""
        w, h = size or (frame.shape[1], frame.shape[0])
        if self.model == "rvm":
            small = cv2.resize(frame, self.work, interpolation=cv2.INTER_AREA)
            src = (cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255).transpose(2, 0, 1)[None]
            ratio = np.array([self.ratio], np.float32)
            _, pha, *self.rec = self.sess.run(None, {"src": src, "r1i": self.rec[0], "r2i": self.rec[1],
                                                       "r3i": self.rec[2], "r4i": self.rec[3], "downsample_ratio": ratio})
            m = pha[0, 0]
        else:
            ww, hh = (self.work[0] // 32) * 32, (self.work[1] // 32) * 32
            small = cv2.resize(frame, (ww, hh), interpolation=cv2.INTER_AREA)
            x = (cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
            m = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0][0, 0]
        return cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
