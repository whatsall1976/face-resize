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
FOREHEAD = 10
CHIN = 152
SIDES = ("left", "right", "top", "bottom")


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


def face_axes(landmarks):
    left = eye_center(landmarks, LEFT_EYE)
    right = eye_center(landmarks, RIGHT_EYE)
    x_axis = right - left
    x_len = float(np.linalg.norm(x_axis))
    if x_len < 1:
        raise RuntimeError("Bad eye axis; face detection failed or face too small.")

    x_axis = x_axis / x_len
    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)

    forehead_to_chin = landmarks[CHIN] - landmarks[FOREHEAD]
    if float(np.dot(y_axis, forehead_to_chin)) < 0:
        y_axis = -y_axis

    return x_axis.astype(np.float32), y_axis.astype(np.float32)


def resolve_side_values(global_value, left=None, right=None, top=None, bottom=None):
    values = {
        "left": global_value if left is None else left,
        "right": global_value if right is None else right,
        "top": global_value if top is None else top,
        "bottom": global_value if bottom is None else bottom,
    }
    return {side: int(values[side]) for side in SIDES}


def normalize_side_values(values, default):
    if values is None:
        return resolve_side_values(default)
    if isinstance(values, dict):
        return {side: int(values[side]) for side in SIDES}
    return resolve_side_values(values)


def side_values_are_symmetric(values):
    return len({int(values[side]) for side in SIDES}) == 1


def translate_mask(mask, vector):
    h, w = mask.shape[:2]
    M = np.array([
        [1.0, 0.0, float(vector[0])],
        [0.0, 1.0, float(vector[1])],
    ], dtype=np.float32)
    return cv2.warpAffine(
        mask,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def expand_mask_symmetric(mask, expand):
    if expand == 0:
        return mask

    k = abs(int(expand)) * 2 + 1
    kernel = np.ones((k, k), np.uint8)
    if expand > 0:
        return cv2.dilate(mask, kernel, iterations=1)
    return cv2.erode(mask, kernel, iterations=1)


def expand_mask_directional(mask, landmarks, expand):
    if side_values_are_symmetric(expand):
        return expand_mask_symmetric(mask, expand["left"])

    x_axis, y_axis = face_axes(landmarks)
    outward = {
        "left": -x_axis,
        "right": x_axis,
        "top": -y_axis,
        "bottom": y_axis,
    }

    for side in SIDES:
        amount = int(expand[side])
        if amount == 0:
            continue

        direction = outward[side]
        if amount > 0:
            base = mask.copy()
            expanded = mask.copy()
            for step in range(1, amount + 1):
                expanded = cv2.max(expanded, translate_mask(base, direction * step))
            mask = expanded
        else:
            for _ in range(abs(amount)):
                mask = cv2.bitwise_and(mask, translate_mask(mask, -direction))

    return mask


def side_feather_widths(shape, landmarks, feather):
    h, w = shape[:2]
    x_axis, y_axis = face_axes(landmarks)
    center = landmarks[FACE_OVAL].mean(axis=0)

    oval = landmarks[FACE_OVAL] - center
    oval_x = oval @ x_axis
    oval_y = oval @ y_axis
    min_x, max_x = float(oval_x.min()), float(oval_x.max())
    min_y, max_y = float(oval_y.min()), float(oval_y.max())

    yy, xx = np.indices((h, w), dtype=np.float32)
    rel_x = xx - center[0]
    rel_y = yy - center[1]
    local_x = rel_x * x_axis[0] + rel_y * x_axis[1]
    local_y = rel_x * y_axis[0] + rel_y * y_axis[1]

    distances = np.stack([
        np.abs(local_x - min_x),
        np.abs(local_x - max_x),
        np.abs(local_y - min_y),
        np.abs(local_y - max_y),
    ])
    side_index = np.argmin(distances, axis=0)
    side_widths = np.array([
        feather["left"],
        feather["right"],
        feather["top"],
        feather["bottom"],
    ], dtype=np.float32)
    return side_widths[side_index]


def curve_distance_feather(mask, landmarks, feather, feather_curve):
    binary = (mask > 127).astype(np.uint8)
    if max(feather.values()) <= 0:
        return binary.astype(np.float32)

    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    signed_distance = inside - outside

    if side_values_are_symmetric(feather):
        width = np.full(mask.shape[:2], feather["left"], dtype=np.float32)
    else:
        width = side_feather_widths(mask.shape, landmarks, feather)

    alpha = binary.astype(np.float32)
    soft = width > 0
    safe_width = np.maximum(width, 1.0)

    if feather_curve == "gaussian":
        curved = 0.5 * (1.0 + np.tanh(1.472 * signed_distance / safe_width))
    else:
        t = np.clip((signed_distance + safe_width) / (2.0 * safe_width), 0.0, 1.0)
        if feather_curve == "smoothstep":
            curved = t * t * (3.0 - 2.0 * t)
        else:
            curved = t

    alpha[soft] = curved[soft]
    return np.clip(alpha, 0.0, 1.0)


def apply_feather(mask, landmarks, feather, feather_curve, feather_gamma):
    if side_values_are_symmetric(feather) and feather_curve == "gaussian":
        amount = int(feather["left"])
        if amount > 0:
            k = amount * 2 + 1
            if k % 2 == 0:
                k += 1
            alpha = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0
        else:
            alpha = mask.astype(np.float32) / 255.0
    else:
        alpha = curve_distance_feather(mask, landmarks, feather, feather_curve)

    if feather_gamma != 1.0:
        alpha = np.power(np.clip(alpha, 0.0, 1.0), feather_gamma)

    return np.clip(alpha, 0.0, 1.0)


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


def create_face_mask(
    shape,
    landmarks,
    expand=None,
    feather=None,
    feather_curve="gaussian",
    feather_gamma=1.0,
):
    feather = normalize_side_values(feather, 12)

    mask = create_face_mask_binary(shape, landmarks, expand=expand)
    return apply_feather(mask, landmarks, feather, feather_curve, feather_gamma)


def create_face_mask_binary(shape, landmarks, expand=None):
    h, w = shape[:2]
    expand = normalize_side_values(expand, 0)

    oval = landmarks[FACE_OVAL].astype(np.int32)
    hull = cv2.convexHull(oval)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    return expand_mask_directional(mask, landmarks, expand)


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
    feather_curve,
    feather_gamma,
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
        feather_curve=feather_curve,
        feather_gamma=feather_gamma,
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
    mask_expand = resolve_side_values(
        args.mask_expand,
        left=args.mask_expand_left,
        right=args.mask_expand_right,
        top=args.mask_expand_top,
        bottom=args.mask_expand_bottom,
    )
    feather = resolve_side_values(
        args.feather,
        left=args.feather_left,
        right=args.feather_right,
        top=args.feather_top,
        bottom=args.feather_bottom,
    )

    resize_face_image(
        source_path=args.source,
        target_path=args.target,
        output_path=args.output,
        scale=args.scale,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        mask_expand=mask_expand,
        feather=feather,
        feather_curve=args.feather_curve,
        feather_gamma=args.feather_gamma,
        align_rotation=not args.no_rotate,
        color_match=args.color_match,
        debug_mask_path=args.debug_mask,
    )

    print("saved:", Path(args.output))


def run_batch(args):
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    mask_expand = resolve_side_values(
        args.mask_expand,
        left=args.mask_expand_left,
        right=args.mask_expand_right,
        top=args.mask_expand_top,
        bottom=args.mask_expand_bottom,
    )
    feather = resolve_side_values(
        args.feather,
        left=args.feather_left,
        right=args.feather_right,
        top=args.feather_top,
        bottom=args.feather_bottom,
    )

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
                mask_expand=mask_expand,
                feather=feather,
                feather_curve=args.feather_curve,
                feather_gamma=args.feather_gamma,
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
    feather_values = [
        args.feather,
        args.feather_left,
        args.feather_right,
        args.feather_top,
        args.feather_bottom,
    ]
    if any(value is not None and value < 0 for value in feather_values):
        ap.error("--feather and directional feather values must be >= 0")
    if args.feather_gamma <= 0:
        ap.error("--feather-gamma must be > 0")

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
    ap.add_argument("--mask-expand-left", type=int)
    ap.add_argument("--mask-expand-right", type=int)
    ap.add_argument("--mask-expand-top", type=int)
    ap.add_argument("--mask-expand-bottom", type=int)
    ap.add_argument("--feather", type=int, default=12)
    ap.add_argument("--feather-left", type=int)
    ap.add_argument("--feather-right", type=int)
    ap.add_argument("--feather-top", type=int)
    ap.add_argument("--feather-bottom", type=int)
    ap.add_argument("--feather-curve", choices=("gaussian", "linear", "smoothstep", "power"), default="gaussian")
    ap.add_argument("--feather-gamma", type=float, default=1.0)
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
