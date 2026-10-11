"""Face boxes (YuNet) refined with a dense mesh (MediaPipe Face Mesh V2)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from . import models

# MediaPipe mesh indices of the inner lip contour; its height/width is the mouth-openness ratio.
INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

MESH_SIZE = 256
ROI_SCALE = 1.5


@dataclass
class Face:
    box: tuple[float, float, float, float]  # x, y, w, h in pixels
    score: float
    kps5: np.ndarray  # YuNet: right eye, left eye, nose, right mouth, left mouth
    mesh: np.ndarray | None = None  # 478 x 3, image pixels (z in pixel units, negative = closer)
    presence: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    mouth: float = -1.0


def _session(path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _canonical() -> np.ndarray:
    pts = []
    with open(models.path("canonical_face")) as f:
        for line in f:
            if line.startswith("v "):
                pts.append([float(v) for v in line.split()[1:4]])
    return np.asarray(pts, np.float64)


def _umeyama_rotation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotation of the similarity transform that best maps src onto dst."""
    s = src - src.mean(0)
    d = dst - dst.mean(0)
    u, _, vt = np.linalg.svd(d.T @ s)
    sign = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1, 1, sign]) @ vt


def head_pose(mesh: np.ndarray, canonical: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in radians; pitch > 0 is head down, yaw > 0 is turned toward image right."""
    obs = mesh[: len(canonical)].astype(np.float64) * [1, -1, -1]  # to x right, y up, z toward camera
    r = _umeyama_rotation(canonical, obs)
    fwd = r @ [0.0, 0.0, 1.0]
    up = r @ [0.0, 1.0, 0.0]
    yaw = float(np.arctan2(fwd[0], fwd[2]))
    pitch = float(np.arctan2(-fwd[1], np.hypot(fwd[0], fwd[2])))
    roll = float(np.arctan2(-up[0], up[1]))
    return pitch, yaw, roll


def mouth_ratio(mesh: np.ndarray) -> float:
    p = mesh[INNER_LIPS, :2]
    w = np.ptp(p[:, 0])
    return float(np.ptp(p[:, 1]) / w) if w > 0 else -1.0


def read_net(path) -> cv2.dnn.Net:
    """OpenCV DNN net on the classic engine when the build offers a choice.

    On OpenCV 5 the new graph engine runs these small convolutional nets about
    1.5x slower on pre-AVX2 CPUs, which is the hardware this tool targets.
    """
    engine = getattr(cv2.dnn, "ENGINE_CLASSIC", None)
    if engine is not None:
        try:
            return cv2.dnn.readNet(str(path), "", "", engine=engine)
        except (TypeError, cv2.error):
            pass
    return cv2.dnn.readNet(str(path))


class YuNet:
    """YuNet decoding as in cv2.FaceDetectorYN, kept here so the net's engine can be chosen."""

    STRIDES = (8, 16, 32)

    def __init__(self, path, score_threshold: float, nms_threshold: float = 0.3, top_k: int = 5000):
        self.net = read_net(path)
        self.names = [f"{k}_{s}" for k in ("cls", "obj", "bbox", "kps") for s in self.STRIDES]
        self.score_threshold, self.nms_threshold, self.top_k = score_threshold, nms_threshold, top_k
        self._grids: dict[tuple[int, int], list[np.ndarray]] = {}

    def _grid(self, pw: int, ph: int) -> list[np.ndarray]:
        if (pw, ph) not in self._grids:
            g = []
            for s in self.STRIDES:
                ys, xs = np.mgrid[0 : ph // s, 0 : pw // s]
                g.append(np.c_[xs.ravel(), ys.ravel()].astype(np.float32))
            self._grids[(pw, ph)] = g
        return self._grids[(pw, ph)]

    def detect(self, image: np.ndarray) -> np.ndarray:
        """N x 15: x, y, w, h, five (x, y) keypoints, score."""
        h, w = image.shape[:2]
        pw, ph = ((w - 1) // 32 + 1) * 32, ((h - 1) // 32 + 1) * 32
        padded = cv2.copyMakeBorder(image, 0, ph - h, 0, pw - w, cv2.BORDER_CONSTANT, value=0)
        self.net.setInput(cv2.dnn.blobFromImage(padded))
        outs = dict(zip(self.names, self.net.forward(self.names)))
        rows = []
        for s, grid in zip(self.STRIDES, self._grid(pw, ph)):
            cls = np.clip(outs[f"cls_{s}"].reshape(-1), 0, 1)
            obj = np.clip(outs[f"obj_{s}"].reshape(-1), 0, 1)
            score = np.sqrt(cls * obj)
            keep = score >= self.score_threshold
            if not keep.any():
                continue
            b = outs[f"bbox_{s}"].reshape(-1, 4)[keep]
            k = outs[f"kps_{s}"].reshape(-1, 10)[keep]
            g = grid[keep]
            c = (g + b[:, :2]) * s
            wh = np.exp(b[:, 2:]) * s
            kp = (k.reshape(-1, 5, 2) + g[:, None, :]) * s
            rows.append(np.c_[c - wh / 2, wh, kp.reshape(-1, 10), score[keep]])
        if not rows:
            return np.empty((0, 15), np.float32)
        d = np.concatenate(rows)
        idx = cv2.dnn.NMSBoxes(d[:, :4].tolist(), d[:, 14].tolist(), self.score_threshold, self.nms_threshold,
                               1.0, self.top_k)
        return d[np.ravel(idx)] if len(idx) else np.empty((0, 15), np.float32)


class FaceAnalyzer:
    def __init__(self, threads: int = 4, score_threshold: float = 0.6, mesh: bool = True):
        self.score_threshold = score_threshold
        self.detector = YuNet(models.path("yunet"), score_threshold)
        self.mesh = _session(models.path("facemesh"), threads) if mesh else None
        self.canonical = _canonical() if mesh else None

    def detect(self, image: np.ndarray) -> list[Face]:
        faces = []
        for r in self.detector.detect(image):
            x, y, bw, bh = (float(v) for v in r[:4])
            faces.append(Face((x, y, bw, bh), float(r[14]), r[4:14].reshape(5, 2).astype(np.float64)))
        if self.mesh is not None and faces:
            self._refine(image, faces)
        return faces

    def _roi(self, face: Face) -> np.ndarray:
        """2x3 affine from the 256x256 mesh crop to image pixels, rotated so the eyes are level."""
        x, y, w, h = face.box
        c = np.array([x + w / 2, y + h / 2])
        re, le = face.kps5[0], face.kps5[1]
        theta = np.arctan2(le[1] - re[1], le[0] - re[0])
        side = max(w, h) * ROI_SCALE
        k = side / MESH_SIZE
        cos, sin = np.cos(theta) * k, np.sin(theta) * k
        a = np.array([[cos, -sin], [sin, cos]])
        t = c - a @ [MESH_SIZE / 2, MESH_SIZE / 2]
        return np.c_[a, t]

    def _refine(self, image: np.ndarray, faces: list[Face]) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rois = [self._roi(f) for f in faces]
        crops = np.stack([
            cv2.warpAffine(rgb, cv2.invertAffineTransform(m), (MESH_SIZE, MESH_SIZE), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
            for m in rois
        ]).astype(np.float32) / 255
        name = self.mesh.get_inputs()[0].name
        outs = []
        for i in range(len(crops)):  # the converted graph has a fixed batch of 1
            outs.append(self.mesh.run(["Identity", "Identity_1"], {name: crops[i : i + 1]}))
        for f, m, out in zip(faces, rois, outs):
            pts = out[0].reshape(-1, 3).astype(np.float64)
            xy = pts[:, :2] @ m[:, :2].T + m[:, 2]
            z = pts[:, 2] * np.linalg.norm(m[:, 0])
            f.mesh = np.c_[xy, z]
            f.presence = float(1 / (1 + np.exp(-float(np.ravel(out[1])[0]))))
            f.pitch, f.yaw, f.roll = head_pose(f.mesh, self.canonical)
            f.mouth = mouth_ratio(f.mesh)
