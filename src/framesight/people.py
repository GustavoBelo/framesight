"""People (NanoDet-Plus, COCO "person") reduced to upper-body boxes."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from . import models

INPUT = 416
STRIDES = (8, 16, 32)  # the exported model has no stride-64 head
REG_MAX = 7
MEAN = np.array([103.53, 116.28, 123.675], np.float32)
STD = np.array([57.375, 57.12, 58.395], np.float32)

# Upper-body rectangle in units of the face box when a face sits inside the
# person (head top to waist), and in units of the person box width otherwise.
# Fitted against Apple Vision's upper-body human rectangles (tools/fit_people.py).
UPPER_BODY = {"face_top": 0.399, "face_bottom": 2.663, "face_half_w": 1.538, "box_h_per_w": 1.465}


@dataclass
class Person:
    box: tuple[float, float, float, float]  # full detection x, y, w, h in pixels
    score: float
    upper: tuple[float, float, float, float] | None = None  # upper body x, y, w, h in pixels


def _centres() -> list[np.ndarray]:
    out = []
    for s in STRIDES:
        n = INPUT // s
        ys, xs = np.mgrid[0:n, 0:n]
        out.append(np.c_[xs.ravel() * s + 0.5 * (s - 1), ys.ravel() * s + 0.5 * (s - 1)].astype(np.float32))
    return out


class PersonDetector:
    def __init__(self, score_threshold: float = 0.4, nms_threshold: float = 0.6, threads: int = 1):
        # onnxruntime runs this net ~1.8x faster than OpenCV DNN on AVX-only CPUs
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(models.path("nanodet")), so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.centres = _centres()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def detect(self, image: np.ndarray) -> list[Person]:
        h, w = image.shape[:2]
        r = min(INPUT / h, INPUT / w)
        canvas = np.full((INPUT, INPUT, 3), 114.0, np.float32)
        nh, nw = int(h * r), int(w * r)
        canvas[:nh, :nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        blob = ((canvas - MEAN) / STD).transpose(2, 0, 1)[None]
        outs = [o.reshape(-1, o.shape[-1]) for o in self.sess.run(None, {self.input: blob})]
        # Pair the per-stride heads by shape rather than trusting output order.
        cls = sorted((o for o in outs if o.shape[1] == 80), key=lambda o: -o.shape[0])
        reg = sorted((o for o in outs if o.shape[1] == 4 * (REG_MAX + 1)), key=lambda o: -o.shape[0])
        boxes, scores = [], []
        proj = np.arange(REG_MAX + 1, dtype=np.float32)
        for s, c, b, ctr in zip(STRIDES, cls, reg, self.centres):
            keep = c[:, 0] >= self.score_threshold  # class 0 = person
            if not keep.any():
                continue
            e = np.exp(b[keep].reshape(-1, REG_MAX + 1))
            d = ((e / e.sum(1, keepdims=True)) @ proj).reshape(-1, 4) * s
            p = ctr[keep]
            x1y1 = np.clip(p - d[:, :2], 0, INPUT)
            x2y2 = np.clip(p + d[:, 2:], 0, INPUT)
            boxes.append(np.c_[x1y1, x2y2 - x1y1] / r)
            scores.append(c[keep, 0])
        if not boxes:
            return []
        boxes = np.concatenate(boxes)
        scores = np.concatenate(scores)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.score_threshold, self.nms_threshold)
        out = []
        for i in np.ravel(idx):
            x, y, bw, bh = (float(v) for v in boxes[i])
            x2, y2 = min(x + bw, w), min(y + bh, h)
            out.append(Person((x, y, x2 - x, y2 - y), float(scores[i])))
        return out


def attach_upper_bodies(people: list[Person], faces, k: dict[str, float] = UPPER_BODY) -> None:
    """Give each person an upper-body box, anchored on its face when one is found.

    A face is assigned to at most one person: the one whose box contains the
    face centre in its top half, preferring the tightest box.
    """
    taken = set()
    for p in sorted(people, key=lambda p: p.box[2] * p.box[3]):
        x, y, w, h = p.box
        best = None
        for j, f in enumerate(faces):
            if j in taken:
                continue
            fx, fy, fw, fh = f.box
            cx, cy = fx + fw / 2, fy + fh / 2
            if x <= cx <= x + w and y - 0.1 * h <= cy <= y + 0.5 * h and fw < 0.8 * w:
                if best is None or f.box[2] > faces[best].box[2]:
                    best = j
        if best is not None:
            taken.add(best)
            fx, fy, fw, fh = faces[best].box
            cx = fx + fw / 2
            top = fy - k["face_top"] * fh
            bottom = fy + fh + k["face_bottom"] * fh
            half = k["face_half_w"] * fw
            ux1, ux2 = max(x, cx - half), min(x + w, cx + half)
            uy1, uy2 = max(y, top), min(y + h, bottom)
        else:
            ux1, ux2 = x, x + w
            uy1, uy2 = y, min(y + h, y + k["box_h_per_w"] * w)
        p.upper = (ux1, uy1, ux2 - ux1, uy2 - uy1)
