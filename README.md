# WA Face Scale CLI

A tiny command-line tool for resizing and re-compositing a face onto an original image.

This is meant to solve a narrow problem similar to DeepFaceLab's face-scale merge adjustment:

> make the detected face slightly smaller or larger, then blend it back into the target image.

It is **not** a ComfyUI custom node. It is a standalone Python script.

## What it does

The script uses:

* MediaPipe FaceMesh for facial landmarks
* OpenCV for affine warping
* NumPy for image blending

Basic process:

1. Detect face landmarks in the source image.
2. Detect face landmarks in the target image.
3. Use eye distance and nose position to calculate scale, rotation, and placement.
4. Create a soft face-oval mask.
5. Warp the source face onto the target image.
6. Blend the result and save a PNG/JPG output.

## EXAMPLE

![Face resize comparison](examples/comparsison.jpg)

## Files

```text
face_resize.py
requirements.txt
README.md
```

## Requirements

Use Python 3.12.

MediaPipe `0.10.21` is used because it still supports the old `mp.solutions.face_mesh` API.

Important:

Do **not** install MediaPipe with normal dependency resolution. Normal installation may pull large unnecessary packages such as `jax`, `jaxlib`, `opencv-contrib-python`, and `matplotlib`.

Use `--no-deps`.

## Installation

Create a venv:

```bash
python3.12 -m venv .venv
```

Install pip tools:

```bash
./.venv/bin/python -m pip install -U pip setuptools wheel
```

Install dependencies without automatic dependency resolution:

```bash
./.venv/bin/python -m pip install --no-deps --only-binary=:all: --prefer-binary -r requirements.txt
```

Verify:

```bash
./.venv/bin/python - <<'PY'
import mediapipe as mp
import cv2
import numpy as np

print("mediapipe:", mp.__version__)
print("has mp.solutions:", hasattr(mp, "solutions"))
print("face_mesh:", mp.solutions.face_mesh.FaceMesh)
print("opencv:", cv2.__version__)
print("numpy:", np.__version__)
PY
```

Expected:

```text
mediapipe: 0.10.21
has mp.solutions: True
```

## Usage

Example:

```bash
# The same command that produced the example above
./.venv/bin/python face_resize.py \
  --source source_face.png \
  --target target_body.png \
  --scale 0.92 \
  --mask-expand 80 \
  --feather 120 \
  --output output_face_scaled.png \
  --debug-mask output_mask.png
```

Make the face smaller:

```bash
./.venv/bin/python face_resize.py \
  --source source_face.png \
  --target target_body.png \
  --scale 0.92 \
  --output output_smaller.png
```

Make the face larger:

```bash
./.venv/bin/python face_resize.py \
  --source source_face.png \
  --target target_body.png \
  --scale 1.08 \
  --output output_larger.png
```

## Parameters

```text
--source        Source face image.
--target        Target/original image.
--output        Output image path.
--scale         Face scale multiplier. Example: 0.92 smaller, 1.08 larger.
--offset-x      Move pasted face horizontally.
--offset-y      Move pasted face vertically.
--mask-expand   Expand or shrink the face mask. Negative values shrink.
--feather       Blur/soften the mask edge.
--no-rotate     Disable rotation alignment.
--color-match   Apply simple color matching.
--debug-mask    Save the final warped mask for debugging.
```

## Practical notes

Enlarging a face is usually easier because the larger pasted face covers the original face underneath.

Shrinking a face is harder because the old face boundary may still be visible around the smaller overlay. For best shrinking results, use a target image where the original face area has been blurred, cleaned, or inpainted first.

## What this is not

This is not a face swap model.

This is not a face restoration model.

This is not a ComfyUI custom node.

This is a lightweight face-geometry compositing script.

## Why install with `--no-deps`?

`mediapipe==0.10.21` declares many dependencies that are not needed for this small script. A normal install may download huge packages such as `jaxlib`.

Use:

```bash
./.venv/bin/python -m pip install --no-deps -r requirements.txt
```

Do not use:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

## Troubleshooting

If Python says a module is missing, install only that missing module with `--no-deps`.

Example:

```bash
./.venv/bin/python -m pip install --no-deps missing-package-name
```

Do not reinstall `mediapipe` with normal dependency resolution.
