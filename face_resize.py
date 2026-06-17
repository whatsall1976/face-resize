#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
NOSE_TIP = 1


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def detect_landmarks_bgr(img_bgr):
    mp_face_mesh = mp.solutions.face_mesh

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(img_rgb)

    if not result.multi_face_landmarks:
        return None

    pts = []
    for lm in result.multi_face_landmarks[0].landmark:
        pts.append([lm.x * w, lm.y * h])

    return np.array(pts, dtype=np.float32)


def eye_center(landmarks, indices):
    return landmarks[indices].mean(axis=0)


def face_angle(landmarks):
    left = eye_center(landmarks, LEFT_EYE)
    right = eye_center(landmarks, RIGHT_EYE)
    v = right - left
    return math.atan2(float(v[1]), float(v[0]))


def interocular_distance(landmarks):
    left = eye_center(landmarks, LEFT_EYE)
    right = eye_center(landmarks, RIGHT_EYE)
    return float(np.linalg.norm(right - left))


def build_affine(src_lm, dst_lm, scale_adjust=1.0, align_rotation=True, offset_x=0, offset_y=0):
    src_anchor = src_lm[NOSE_TIP]
    dst_anchor = dst_lm[NOSE_TIP].copy()
    dst_anchor[0] += offset_x
    dst_anchor[1] += offset_y

    src_dist = interocular_distance(src_lm)
    dst_dist = interocular_distance(dst_lm)

    if src_dist < 1 or dst_dist < 1:
        raise RuntimeError("Bad eye distance; face detection failed or face too small.")

    scale = (dst_dist / src_dist) * scale_adjust

    angle = 0.0
    if align_rotation:
        angle = face_angle(dst_lm) - face_angle(src_lm)

    cos_a = math.cos(angle) * scale
    sin_a = math.sin(angle) * scale

    # x' = A*x + b
    A = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a],
    ], dtype=np.float32)

    b = dst_anchor - A @ src_anchor

    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = A
    M[:, 2] = b

    return M


def create_face_mask(shape, landmarks, expand=0, feather=12):
    h, w = shape[:2]

    oval = landmarks[FACE_OVAL].astype(np.int32)
    hull = cv2.convexHull(oval)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    if expand != 0:
        k = abs(int(expand)) * 2 + 1
        kernel = np.ones((k, k), np.uint8)
        if expand > 0:
            mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            mask = cv2.erode(mask, kernel, iterations=1)

    if feather > 0:
        k = int(feather) * 2 + 1
        if k % 2 == 0:
            k += 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    return mask.astype(np.float32) / 255.0


def optional_color_match(src_warped, target, mask):
    # Simple mean/std match inside the mask. Not perfect, but helps.
    m = mask > 0.2
    if m.sum() < 100:
        return src_warped

    out = src_warped.astype(np.float32)
    tgt = target.astype(np.float32)

    for c in range(3):
        src_vals = out[:, :, c][m]
        tgt_vals = tgt[:, :, c][m]

        src_mean, src_std = src_vals.mean(), src_vals.std() + 1e-6
        tgt_mean, tgt_std = tgt_vals.mean(), tgt_vals.std() + 1e-6

        out[:, :, c] = (out[:, :, c] - src_mean) / src_std * tgt_std + tgt_mean

    return np.clip(out, 0, 255).astype(np.uint8)


def resize_face_image(
    source_path,
    target_path,
    output_path,
    scale,
    offset_x,
    offset_y,
    mask_expand,
    feather,
    align_rotation,
    color_match,
    debug_mask_path=None,
):
    source = read_image(source_path)
    target = read_image(target_path)

    src_lm = detect_landmarks_bgr(source)
    dst_lm = detect_landmarks_bgr(target)

    if src_lm is None:
        raise RuntimeError("No face detected in source image.")
    if dst_lm is None:
        raise RuntimeError("No face detected in target image.")

    M = build_affine(
        src_lm,
        dst_lm,
        scale_adjust=scale,
        align_rotation=align_rotation,
        offset_x=offset_x,
        offset_y=offset_y,
    )

    th, tw = target.shape[:2]

    src_mask = create_face_mask(
        source.shape,
        src_lm,
        expand=mask_expand,
        feather=feather,
    )

    warped_face = cv2.warpAffine(
        source,
        M,
        (tw, th),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    warped_mask = cv2.warpAffine(
        src_mask,
        M,
        (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    warped_mask = np.clip(warped_mask, 0.0, 1.0)

    if color_match:
        warped_face = optional_color_match(warped_face, target, warped_mask)

    alpha = warped_mask[:, :, None]
    result = warped_face.astype(np.float32) * alpha + target.astype(np.float32) * (1.0 - alpha)
    result = np.clip(result, 0, 255).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Cannot write output image: {output_path}")

    if debug_mask_path:
        debug_mask_path = Path(debug_mask_path)
        debug_mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(debug_mask_path), (warped_mask * 255).astype(np.uint8)):
            raise RuntimeError(f"Cannot write debug mask: {debug_mask_path}")


def iter_image_paths(folder, recursive=False):
    folder = Path(folder)
    pattern = "**/*" if recursive else "*"
    for path in sorted(folder.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def output_path_for(input_path, input_folder, output_folder):
    rel_path = input_path.relative_to(input_folder)
    return Path(output_folder) / rel_path


def debug_mask_path_for(input_path, input_folder, debug_mask_folder):
    rel_path = input_path.relative_to(input_folder)
    return Path(debug_mask_folder) / rel_path.with_name(f"{rel_path.stem}_mask.png")


def run_single(args):
    resize_face_image(
        source_path=args.source,
        target_path=args.target,
        output_path=args.output,
        scale=args.scale,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        mask_expand=args.mask_expand,
        feather=args.feather,
        align_rotation=not args.no_rotate,
        color_match=args.color_match,
        debug_mask_path=args.debug_mask,
    )

    print("saved:", Path(args.output))


def run_batch(args):
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    if not input_folder.is_dir():
        raise RuntimeError(f"Input folder does not exist or is not a directory: {input_folder}")

    image_paths = list(iter_image_paths(input_folder, recursive=args.recursive))
    if not image_paths:
        raise RuntimeError(f"No supported images found in folder: {input_folder}")

    saved = 0
    failed = 0

    for index, input_path in enumerate(image_paths, start=1):
        output_path = output_path_for(input_path, input_folder, output_folder)
        debug_mask_path = None
        if args.debug_mask_folder:
            debug_mask_path = debug_mask_path_for(input_path, input_folder, args.debug_mask_folder)

        print(f"[{index}/{len(image_paths)}] processing: {input_path}")

        try:
            resize_face_image(
                source_path=input_path,
                target_path=input_path,
                output_path=output_path,
                scale=args.scale,
                offset_x=args.offset_x,
                offset_y=args.offset_y,
                mask_expand=args.mask_expand,
                feather=args.feather,
                align_rotation=not args.no_rotate,
                color_match=args.color_match,
                debug_mask_path=debug_mask_path,
            )
        except Exception as exc:
            failed += 1
            print(f"failed: {input_path}: {exc}")
            continue

        saved += 1
        print("saved:", output_path)

    if failed:
        raise RuntimeError(f"Batch finished with failures: saved {saved}, failed {failed}")

    print(f"batch complete: saved {saved} image(s) to {output_folder}")


def validate_args(ap, args):
    batch_args = [args.input_folder, args.output_folder]
    batch_mode = any(batch_args)

    if batch_mode:
        if not all(batch_args):
            ap.error("--input-folder and --output-folder must be used together")
        if args.source or args.target or args.output:
            ap.error("Use either --input-folder/--output-folder or --source/--target/--output, not both")
        if args.debug_mask:
            ap.error("--debug-mask is only for single-image mode; use --debug-mask-folder for batch mode")
        return "batch"

    if not args.source or not args.target or not args.output:
        ap.error("single-image mode requires --source, --target, and --output")
    if args.recursive:
        ap.error("--recursive can only be used with --input-folder")
    if args.debug_mask_folder:
        ap.error("--debug-mask-folder can only be used with --input-folder")
    return "single"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="source face image")
    ap.add_argument("--target", help="target/original image")
    ap.add_argument("--output", help="output image path")
    ap.add_argument("--input-folder", help="folder of images to process one by one")
    ap.add_argument("--output-folder", help="folder to save batch results")
    ap.add_argument("--scale", type=float, default=0.92, help="face scale adjust, e.g. 0.92 smaller, 1.08 larger")
    ap.add_argument("--offset-x", type=int, default=0)
    ap.add_argument("--offset-y", type=int, default=0)
    ap.add_argument("--mask-expand", type=int, default=-2)
    ap.add_argument("--feather", type=int, default=12)
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--color-match", action="store_true")
    ap.add_argument("--debug-mask", default=None, help="optional mask output path")
    ap.add_argument("--debug-mask-folder", default=None, help="optional folder for batch debug masks")
    ap.add_argument("--recursive", action="store_true", help="process images in nested folders")
    args = ap.parse_args()

    mode = validate_args(ap, args)
    if mode == "batch":
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
