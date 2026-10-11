"""Model registry and local cache.

Models are never bundled: each one is fetched from its upstream on first use and
pinned by SHA-256, so a changed or truncated download fails loudly instead of
silently degrading results.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Model:
    file: str
    url: str
    sha256: str
    license: str
    source: str


HF_ZOO = "https://huggingface.co/opencv"

REGISTRY: dict[str, Model] = {
    "yunet": Model(
        "face_detection_yunet_2023mar.onnx",
        f"{HF_ZOO}/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "MIT",
        "opencv_zoo / ShiqiYu/libfacedetection.train",
    ),
    "facemesh": Model(
        "face_landmarks_v2.onnx",
        "https://huggingface.co/naklitechie/face-landmarks-onnx/resolve/main/face_landmarks.onnx",
        "f38c3321ceffbc9e95103480ad38cc3f52e7e1bde2bcee7cd9355d0b9138ac0c",
        "Apache-2.0",
        "Google MediaPipe Face Mesh V2, converted to ONNX",
    ),
    "canonical_face": Model(
        "canonical_face_model.obj",
        "https://raw.githubusercontent.com/google-ai-edge/mediapipe/v0.10.21/mediapipe/modules/face_geometry/data/canonical_face_model.obj",
        "8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618",
        "Apache-2.0",
        "Google MediaPipe face geometry",
    ),
    "nanodet": Model(
        "object_detection_nanodet_2022nov.onnx",
        f"{HF_ZOO}/object_detection_nanodet/resolve/main/object_detection_nanodet_2022nov.onnx",
        "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186",
        "Apache-2.0",
        "opencv_zoo / RangiLyu/nanodet (NanoDet-Plus-m)",
    ),
    "modnet": Model(
        "modnet_photographic.onnx",
        "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx",
        "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9",
        "Apache-2.0",
        "ZHKKKe/MODNet",
    ),
    "rvm": Model(
        "rvm_mobilenetv3_fp32.onnx",
        "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
        "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828",
        "GPL-3.0",
        "PeterL1n/RobustVideoMatting",
    ),
}


def cache_dir() -> Path:
    base = os.environ.get("FRAMESIGHT_CACHE") or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "framesight"
    return Path(base)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def path(name: str) -> Path:
    """Local path of a model, downloading and verifying it on first use."""
    m = REGISTRY[name]
    dest = cache_dir() / m.file
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"framesight: downloading {name} ({m.license}) from {m.url}", file=sys.stderr)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    try:
        urllib.request.urlretrieve(m.url, tmp)
        got = _sha256(Path(tmp))
        if m.sha256 and got != m.sha256:
            raise RuntimeError(f"{name}: SHA-256 mismatch (expected {m.sha256}, got {got})")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest
