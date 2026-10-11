# framesight

Faces, head pose, mouth openness, upper bodies and person masks from video, as
JSON, on any x86-64 CPU. No GPU, no AVX2.

framesight started as a Linux replacement for three small macOS helpers built on
Apple's Vision framework, which a video-editing pipeline used to reframe shots
around the speaker, spot when they look down to read, pick cover stills and cut
the speaker out of the background. Nothing on Linux returned all of that as
JSON from one CLI, so this does.

It targets old hardware on purpose: everything was measured on an Intel
i5-2400 (Sandy Bridge, 2011, AVX without AVX2), where many prebuilt ML wheels
die with `Illegal instruction`.

## Install

```sh
pip install framesight            # or: pip install -e . from a clone
pip install "framesight[heif]"    # optional HEIC/HEIF support for `faces`
```

Models are downloaded on first use into `~/.cache/framesight` (override with
`FRAMESIGHT_CACHE`) and checked against pinned SHA-256 hashes. To fetch them
ahead of time, for example on a machine that will run offline:

```sh
framesight models --fetch
```

## Usage

```sh
framesight track talk.mp4 --start 120 --duration 60 --rate 5 > track.json
framesight faces photo1.jpg photo2.jpg > faces.json
framesight mask talk.mp4 --size 640x360 > talk.mask
```

`--threads N` (before the subcommand, or `FRAMESIGHT_THREADS`) caps the cores
used. That helps on machines that run hot under sustained load.

### Conventions

- Boxes are `[x, y, w, h]`, normalised to 0–1, origin at the **top-left**.
- Angles are in radians. `pitch > 0` means the head is tilted down (reading);
  `yaw > 0` means the face is turned toward the right side of the image.
- `mouth` is the height/width ratio of the inner lip contour: about 0.1 when the
  mouth is closed, 0.3 or more when it is open.
- `score` is the detector confidence; `quality` (stills only) is a capture-quality
  score from 0 to 1 (resolution, sharpness, frontality, exposure).

### `track`

Samples the video and reports faces and upper bodies in each sample. Sampling is
not a fixed grid: it takes the first frame at or after `start`, then the first
frame at or after the previous sample + `1/rate`. `t` counts from `start`.

```json
{"width": 1280, "height": 720, "samples": [
  {"t": 0.2,
   "faces":  [{"box": [0.41, 0.31, 0.08, 0.19], "score": 0.94,
               "pitch": 0.10, "yaw": -0.52, "roll": 0.02, "mouth": 0.10}],
   "people": [{"box": [0.34, 0.26, 0.24, 0.69], "score": 0.83}]}]}
```

`people` boxes cover the **upper body** (head to waist), not the full figure.

### `faces`

```json
{"photo1.jpg": {"width": 800, "height": 533,
                "faces": [{"box": [...], "score": 0.9, "quality": 0.31,
                           "pitch": 0.05, "yaw": 0.2, "roll": 0.0, "mouth": 0.12}]},
 "broken.jpg": {"error": "cannot open"}}
```

### `mask`

Binary on stdout: one text line `"WxH\n"`, then one `W·H`-byte 8-bit grey frame
per video frame, in decode order. 255 is person, 0 is background, and values in
between are soft edges. A frame that fails comes out all zeros, so the stream
never drifts out of step with the video.

## Models

| Task | Model | License |
|---|---|---|
| Face boxes + 5 keypoints | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT |
| 478-point face mesh → pose, mouth | [MediaPipe Face Mesh V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf) ([ONNX conversion](https://huggingface.co/naklitechie/face-landmarks-onnx)) | Apache-2.0 |
| Head-pose reference | MediaPipe canonical face model | Apache-2.0 |
| People | [NanoDet-Plus](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet) | Apache-2.0 |
| Person mask (default) | [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting), MobileNetV3 | **GPL-3.0** |
| Person mask (`--model modnet`) | [MODNet](https://github.com/ZHKKKe/MODNet) | Apache-2.0 |

framesight's code is MIT. No model is bundled; each is downloaded from its
upstream on first use. The default mask model's weights are GPL-3.0: if that
does not suit your use, pass `--model modnet`, which is slower and a little less
accurate (see below).

## Accuracy and speed

Checked against Apple Vision on six one-minute clips of live-streamed church
services (720p, preacher close-ups, worship band, audience) and about 330
photos. The footage is not published, because it shows people who did not
agree to it.

- **Faces:** 98–100% of Vision's faces found. framesight also finds small faces
  that Vision misses.
- **Head pose:** on the main speaker, "looking down" (pitch ≥ 0.3) agrees with
  Vision in 94–98% of samples. Where the two disagree, frame-by-frame checks
  mostly showed Vision wrong, reporting a level head while the speaker was
  reading. framesight follows what is in the frame, not Vision's numbers.
- **People:** 94–100% of Vision's upper-body boxes found, mean IoU 0.65–0.81.
- **Capture quality:** Pearson ~0.6 with Vision's face capture quality on
  held-out photos.
- **Masks:** IoU 0.90–0.95 with Vision on single speakers. On a crowded stage,
  Vision also keeps people in the background, which RVM leaves out.

Speed on the i5-2400 (4 threads):

| | per minute of 720p video |
|---|---|
| `track` at 5 samples/s | 25–30 s |
| `mask` (RVM, 640×360) | ~80 s |
| `mask` (MODNet) | ~6 min |
| `faces` | ~75 ms per photo |

## Tools

`tools/` holds the scripts used to check and calibrate framesight against a
reference tool. All of them take references in framesight's own schema.

- `compare.py`: regression report (recall, IoU, pose and mouth agreement,
  quality correlation).
- `compare_masks.py`: mask IoU and edge error against a reference mask stream.
- `fit_people.py`: fits the upper-body rectangle constants in `people.py`.
- `fit_quality.py`: fits the capture-quality weights in `quality.py`.

## Development

```sh
pip install -e ".[dev]"
pytest
```

## Credits

Written by Gustavo Belo with [Claude](https://claude.com/claude-code) (Anthropic)
as a coding partner.

If framesight saves you some time, you can
[buy me a coffee](https://buymeacoffee.com/gustavobelo).

## License

MIT, see [LICENSE](LICENSE). Model licenses are listed above.
