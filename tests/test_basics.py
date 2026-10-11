import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from framesight import video
from framesight.faces import _umeyama_rotation, head_pose, mouth_ratio
from framesight.people import Person, _centres, attach_upper_bodies


@pytest.fixture
def clip(tmp_path):
    """3 s of 30 fps video whose frames carry their own index as brightness."""
    path = str(tmp_path / "clip.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(90):
        out.write(np.full((48, 64, 3), i * 2, np.uint8))
    out.release()
    return path


def test_nanodet_grid_layout():
    c = _centres()
    assert [len(g) for g in c] == [52 * 52, 26 * 26, 13 * 13]
    assert np.allclose(c[0][0], [3.5, 3.5])


class _F:
    def __init__(self, box):
        self.box = box


def test_upper_body_anchors_on_face_inside_person():
    p = Person((100, 50, 200, 600), 0.9)
    attach_upper_bodies([p], [_F((170, 80, 60, 60))])
    x, y, w, h = p.upper
    assert y < 80 and y + h < 650 and x >= 100 and x + w <= 300
    assert 150 < y + h < 400  # waist, not feet


def test_upper_body_without_face_is_top_of_person():
    p = Person((100, 50, 200, 600), 0.9)
    attach_upper_bodies([p], [])
    assert p.upper[1] == 50 and p.upper[3] < 600


def test_sampling_takes_first_frame_at_or_after_each_step(clip):
    ts = [t for t, _ in video.sample(clip, 1.0, 1.0, 5)]
    assert ts[0] == 0.0
    assert len(ts) == 5
    assert all(abs(b - a - 0.2) < 0.034 for a, b in zip(ts, ts[1:]))


def test_frames_yields_every_frame(clip):
    assert sum(1 for _ in video.frames(clip)) == 90


def test_umeyama_recovers_rotation():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    ang = 0.4
    r = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])
    assert np.allclose(_umeyama_rotation(src, src @ r.T * 2.5 + 3), r, atol=1e-9)


def _mesh_from(canonical, pitch=0.0, yaw=0.0):
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    obs = canonical @ (ry @ rx).T  # x right, y up, z toward camera
    return obs * [10, -10, -10] + [320, 240, 0]  # back to image pixels


@pytest.fixture
def canonical():
    # A rigid, asymmetric point cloud stands in for the face model.
    return np.random.default_rng(1).normal(size=(468, 3))


def test_head_down_is_positive_pitch(canonical):
    # Rotating the face so its forward vector tips toward -y (down) must read as pitch > 0.
    pitch, yaw, _ = head_pose(_mesh_from(canonical, pitch=0.3), canonical)
    assert pitch == pytest.approx(0.3, abs=1e-6)
    assert yaw == pytest.approx(0.0, abs=1e-6)


def test_yaw_sign(canonical):
    _, yaw, _ = head_pose(_mesh_from(canonical, yaw=0.5), canonical)
    assert yaw == pytest.approx(0.5, abs=1e-6)  # turned toward image right


def test_mouth_ratio_closed_and_open():
    mesh = np.zeros((478, 3))
    from framesight.faces import INNER_LIPS

    for k, i in enumerate(INNER_LIPS):
        a = 2 * np.pi * k / len(INNER_LIPS)
        mesh[i, :2] = [100 + 20 * np.cos(a), 100 + 6 * np.sin(a)]
    assert mouth_ratio(mesh) == pytest.approx(0.3, abs=0.02)


def test_mask_stream_header_and_frame_count(clip):
    pytest.importorskip("onnxruntime")
    try:
        from framesight import models

        models.path("rvm")
    except Exception:
        pytest.skip("model not available offline")
    out = subprocess.run([sys.executable, "-m", "framesight", "--threads", "1", "mask", clip, "--size", "32x24",
                          "--model", "rvm"], capture_output=True, check=True).stdout
    head, body = out.split(b"\n", 1)
    assert head == b"32x24"
    assert len(body) == 90 * 32 * 24


def test_faces_reports_unreadable_image(tmp_path):
    bad = tmp_path / "nope.jpg"
    bad.write_bytes(b"not an image")
    try:
        from framesight import models

        models.path("yunet"); models.path("facemesh"); models.path("canonical_face")
    except Exception:
        pytest.skip("models not available offline")
    out = subprocess.run([sys.executable, "-m", "framesight", "faces", str(bad)], capture_output=True, check=True).stdout
    assert json.loads(out) == {str(bad): {"error": "cannot open"}}
