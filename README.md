# WA Face Scale CLI

A tiny command-line tool for resizing a detected face with a continuous
face-local landmark mesh warp.

This is meant to solve a narrow problem similar to DeepFaceLab's face-scale
merge adjustment:

> make the detected face slightly smaller or larger inside the original image.

It is **not** a ComfyUI custom node. It is a standalone Python script.

## What it does

The script uses:

* MediaPipe FaceMesh for facial landmarks
* OpenCV for Delaunay triangulation and affine warping
* NumPy for control-point and image operations

Basic process:

1. Detect face landmarks in the input image.
2. Build a face-local coordinate system from the eye axis and forehead-to-chin direction.
3. Scale the face-oval control points in face-local coordinates.
4. Add fixed outer anchor points around an expanded face-local box.
5. Delaunay-triangulate the destination control points.
6. Apply a continuous piecewise-affine warp inside the ROI.
7. Save the warped image and optional point/mesh debug images.

The deformation is one continuous field. There is no separate face patch paste,
source/target swap, mask expansion, inpainting, or semantic hair detection in
this version.

## Requirements

Use Python 3.12.

MediaPipe `0.10.21` is used because it still supports the old
`mp.solutions.face_mesh` API.

Important:

Do **not** install MediaPipe with normal dependency resolution. Normal
installation may pull large unnecessary packages such as `jax`, `jaxlib`,
`opencv-contrib-python`, and `matplotlib`.

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

Basic face resize:

```bash
./.venv/bin/python face_resize.py \
  --input original.jpg \
  --output resized.jpg \
  --scale 0.92
```

Use different face-local X/Y scales:

```bash
./.venv/bin/python face_resize.py \
  --input original.jpg \
  --output resized_xy.jpg \
  --scale-x 0.94 \
  --scale-y 0.90
```

Write debug point and mesh overlays:

```bash
./.venv/bin/python face_resize.py \
  --input original.jpg \
  --output resized_debug.jpg \
  --scale 0.94 \
  --anchor-scale 1.6 \
  --debug-points debug_points.png \
  --debug-mesh debug_mesh.png
```

Include more forehead in the moving control area:

```bash
./.venv/bin/python face_resize.py \
  --input original.jpg \
  --output resized_forehead.jpg \
  --scale 0.94 \
  --anchor-scale 1.6 \
  --forehead-expand 0.20
```

Try a few shrink strengths:

```bash
./.venv/bin/python face_resize.py --input original.jpg --output resize_0.97.jpg --scale 0.97
./.venv/bin/python face_resize.py --input original.jpg --output resize_0.94.jpg --scale 0.94
./.venv/bin/python face_resize.py --input original.jpg --output resize_0.90.jpg --scale 0.90
```

## Parameters

```text
--input         Input image path.
--output        Output image path.
--scale         Uniform face-local scale. Example: 0.92 smaller, 1.08 larger.
--scale-x       Face-local horizontal scale. Overrides --scale for the eye-to-eye axis.
--scale-y       Face-local vertical scale. Overrides --scale for the forehead-to-chin axis.
--anchor-scale  Expansion factor for the fixed outer anchor box. Default: 1.6.
--forehead-expand
                Add moving forehead controls above the face oval as a fraction of face height. Example: 0.20.
--debug-points  Save a debug image with moving source points in red, moving destination points in green, and fixed outer anchors in blue.
--debug-mesh    Save a debug image with destination Delaunay triangles in gray.
```

`--scale-x` follows the eye-to-eye axis. `--scale-y` follows the perpendicular
axis corrected toward the chin. The outer anchor points stay fixed so the mesh
deformation fades into the surrounding image.

`--forehead-expand` adds a row of moving control points above the top of the
face oval. Those points are scaled with the face, so extra forehead can shrink
or expand with the rest of the face instead of only changing the fixed outer
anchor boundary.

## Practical notes

Small to moderate shrinking usually works best. If `--scale 0.90` looks too
distorted but `--scale 0.94` looks clean, the mesh warp is working and the
remaining issue is deformation strength.

## What this is not

This is not a face swap model.

This is not a face restoration model.

This is not a ComfyUI custom node.

This is a lightweight face-geometry mesh-warp script.

## Why install with `--no-deps`?

`mediapipe==0.10.21` declares many dependencies that are not needed for this
small script. A normal install may download huge packages such as `jaxlib`.

Use:

```bash
./.venv/bin/python -m pip install --no-deps -r requirements.txt
```

Do not use:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

## Troubleshooting

If Python says a module is missing, install only that missing module with
`--no-deps`.

Example:

```bash
./.venv/bin/python -m pip install --no-deps missing-package-name
```

Do not reinstall `mediapipe` with normal dependency resolution.
