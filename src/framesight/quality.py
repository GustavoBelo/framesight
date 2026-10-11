"""Face capture quality: how usable a face is as a still.

A weighted blend of measurable cues (resolution, sharpness, frontality,
exposure). The weights were fitted against Apple Vision's face capture quality
on ~650 real photo faces, so scores land on a similar 0-1 scale and rank faces
alike (Pearson ~0.6 on held-out photos); see tools/fit_quality.py. Eye openness
was tried and dropped: it did not track the reference score at all.
"""

from __future__ import annotations

import cv2
import numpy as np

CROP = 112

WEIGHTS = {'bias': -0.4729, 'sharp': 0.0089, 'pixels': 0.1218, 'frontal': 0.0418, 'score': 0.1428, 'exposure': 0.1478, 'contrast': 0.0629}


def features(image: np.ndarray, face) -> dict[str, float]:
    x, y, w, h = face.box
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(image.shape[1], int(x + w)), min(image.shape[0], int(y + h))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        sharp = exposure = contrast = 0.0
    else:
        g = cv2.cvtColor(cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        sharp = float(np.log1p(cv2.Laplacian(g, cv2.CV_64F).var()))
        exposure = float(-abs(g.mean() / 255 - 0.5))
        contrast = float(g.std() / 128)
    has_mesh = face.mesh is not None
    return {
        "sharp": sharp,
        "pixels": float(np.log(max(w, 1.0))),
        "frontal": float(np.cos(face.yaw) * np.cos(face.pitch)) if has_mesh else 0.0,
        "score": float(face.score),
        "exposure": exposure,
        "contrast": contrast,
    }


def capture_quality(image: np.ndarray, face) -> float:
    f = features(image, face)
    z = WEIGHTS["bias"] + sum(WEIGHTS[k] * v for k, v in f.items())
    return float(np.clip(z, 0.0, 1.0))
