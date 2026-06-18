#!/usr/bin/env python3
import argparse
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
FOREHEAD = 10
CHIN = 152


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def write_image(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"Cannot write image: {path}")


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


def to_local(points, center, x_axis, y_axis):
    rel = points.astype(np.float32) - center.astype(np.float32)
    return np.column_stack((rel @ x_axis, rel @ y_axis)).astype(np.float32)


def from_local(local_points, center, x_axis, y_axis):
    local_points = local_points.astype(np.float32)
    return (
        center.astype(np.float32)
        + local_points[:, 0:1] * x_axis.astype(np.float32)
        + local_points[:, 1:2] * y_axis.astype(np.float32)
    ).astype(np.float32)


def clip_point_to_image(point, width, height):
    return np.array([
        np.clip(point[0], 0.0, float(width - 1)),
        np.clip(point[1], 0.0, float(height - 1)),
    ], dtype=np.float32)


def make_anchor_points(oval_points, center, x_axis, y_axis, width, height, anchor_scale, samples_per_side=9):
    local = to_local(oval_points, center, x_axis, y_axis)
    min_x, min_y = local.min(axis=0)
    max_x, max_y = local.max(axis=0)

    box_center_x = (min_x + max_x) * 0.5
    box_center_y = (min_y + max_y) * 0.5
    half_w = (max_x - min_x) * 0.5
    half_h = (max_y - min_y) * 0.5

    left = box_center_x - half_w * anchor_scale
    right = box_center_x + half_w * anchor_scale
    top = box_center_y - half_h * anchor_scale
    bottom = box_center_y + half_h * anchor_scale

    xs = np.linspace(left, right, samples_per_side, dtype=np.float32)
    ys = np.linspace(top, bottom, samples_per_side, dtype=np.float32)
    local_anchors = []

    for x in xs:
        local_anchors.append([x, top])
        local_anchors.append([x, bottom])
    for y in ys[1:-1]:
        local_anchors.append([left, y])
        local_anchors.append([right, y])

    anchors = from_local(np.array(local_anchors, dtype=np.float32), center, x_axis, y_axis)
    return np.array([clip_point_to_image(p, width, height) for p in anchors], dtype=np.float32)


def make_forehead_points(oval_points, center, x_axis, y_axis, forehead_expand, samples=9):
    if forehead_expand <= 0:
        return np.empty((0, 2), dtype=np.float32)

    local = to_local(oval_points, center, x_axis, y_axis)
    min_x, min_y = local.min(axis=0)
    max_x, max_y = local.max(axis=0)

    box_center_x = (min_x + max_x) * 0.5
    half_w = (max_x - min_x) * 0.5 * 0.85
    face_h = max_y - min_y
    forehead_y = min_y - face_h * forehead_expand

    xs = np.linspace(box_center_x - half_w, box_center_x + half_w, samples, dtype=np.float32)
    src_local = np.column_stack((xs, np.full(samples, forehead_y, dtype=np.float32))).astype(np.float32)
    return from_local(src_local, center, x_axis, y_axis)


def add_unique_control(src_points, dst_points, src, dst, min_distance=0.25):
    if dst_points:
        existing = np.array(dst_points, dtype=np.float32)
        if float(np.min(np.linalg.norm(existing - dst, axis=1))) < min_distance:
            return
    src_points.append(src.astype(np.float32))
    dst_points.append(dst.astype(np.float32))


def build_control_points(landmarks, width, height, scale_x, scale_y, anchor_scale, forehead_expand):
    center = landmarks[FACE_OVAL].mean(axis=0).astype(np.float32)
    x_axis, y_axis = face_axes(landmarks)

    oval_src = landmarks[FACE_OVAL].astype(np.float32)
    oval_local = to_local(oval_src, center, x_axis, y_axis)
    oval_dst_local = oval_local.copy()
    oval_dst_local[:, 0] *= scale_x
    oval_dst_local[:, 1] *= scale_y
    oval_dst = from_local(oval_dst_local, center, x_axis, y_axis)

    forehead_src = make_forehead_points(
        oval_src,
        center,
        x_axis,
        y_axis,
        forehead_expand,
    )
    if len(forehead_src) > 0:
        forehead_src = np.array([clip_point_to_image(p, width, height) for p in forehead_src], dtype=np.float32)
    forehead_dst = np.empty((0, 2), dtype=np.float32)
    if len(forehead_src) > 0:
        forehead_dst_local = to_local(forehead_src, center, x_axis, y_axis)
        forehead_dst_local[:, 0] *= scale_x
        forehead_dst_local[:, 1] *= scale_y
        forehead_dst = from_local(forehead_dst_local, center, x_axis, y_axis)

    anchors = make_anchor_points(
        oval_src,
        center,
        x_axis,
        y_axis,
        width,
        height,
        anchor_scale,
    )

    src_points = []
    dst_points = []
    point_kinds = []

    for src, dst in zip(oval_src, oval_dst):
        before = len(dst_points)
        add_unique_control(src_points, dst_points, src, clip_point_to_image(dst, width, height))
        if len(dst_points) > before:
            point_kinds.append("oval")

    for src, dst in zip(forehead_src, forehead_dst):
        before = len(dst_points)
        add_unique_control(src_points, dst_points, src, clip_point_to_image(dst, width, height))
        if len(dst_points) > before:
            point_kinds.append("forehead")

    for anchor in anchors:
        before = len(dst_points)
        add_unique_control(src_points, dst_points, anchor, anchor)
        if len(dst_points) > before:
            point_kinds.append("anchor")

    return (
        np.array(src_points, dtype=np.float32),
        np.array(dst_points, dtype=np.float32),
        point_kinds,
        center,
        x_axis,
        y_axis,
    )


def point_inside_image(point, width, height):
    return 0.0 <= point[0] < width and 0.0 <= point[1] < height


def nearest_point_index(point, points, max_distance=2.0):
    distances = np.linalg.norm(points - point.astype(np.float32), axis=1)
    index = int(np.argmin(distances))
    if float(distances[index]) > max_distance:
        return None
    return index


def triangle_area(tri):
    a, b, c = tri
    return abs(float(np.cross(b - a, c - a))) * 0.5


def delaunay_triangles(dst_points, width, height):
    subdiv = cv2.Subdiv2D((0, 0, width, height))
    for point in dst_points:
        if point_inside_image(point, width, height):
            subdiv.insert((float(point[0]), float(point[1])))

    raw_triangles = subdiv.getTriangleList()
    triangles = []
    seen = set()

    for raw in raw_triangles:
        tri = raw.reshape(3, 2).astype(np.float32)
        if not all(point_inside_image(p, width, height) for p in tri):
            continue

        indices = []
        for vertex in tri:
            index = nearest_point_index(vertex, dst_points)
            if index is None:
                indices = []
                break
            indices.append(index)

        if len(indices) != 3 or len(set(indices)) != 3:
            continue

        key = tuple(sorted(indices))
        if key in seen:
            continue

        dst_tri = dst_points[list(indices)]
        if triangle_area(dst_tri) < 1.0:
            continue

        seen.add(key)
        triangles.append(tuple(indices))

    return triangles


def roi_from_points(points, width, height, padding=4):
    min_xy = np.floor(points.min(axis=0) - padding).astype(int)
    max_xy = np.ceil(points.max(axis=0) + padding).astype(int)

    x0 = int(np.clip(min_xy[0], 0, width - 1))
    y0 = int(np.clip(min_xy[1], 0, height - 1))
    x1 = int(np.clip(max_xy[0] + 1, x0 + 1, width))
    y1 = int(np.clip(max_xy[1] + 1, y0 + 1, height))
    return x0, y0, x1, y1


def warp_triangle_roi(src_roi, dst_roi, src_tri, dst_tri):
    matrix = cv2.getAffineTransform(src_tri.astype(np.float32), dst_tri.astype(np.float32))
    warped = cv2.warpAffine(
        src_roi,
        matrix,
        (dst_roi.shape[1], dst_roi.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros(dst_roi.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(dst_tri).astype(np.int32), 255, lineType=cv2.LINE_AA)
    dst_roi[mask > 0] = warped[mask > 0]


def warp_mesh(img, src_points, dst_points, triangles):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi_from_points(np.vstack((src_points, dst_points)), w, h)
    offset = np.array([x0, y0], dtype=np.float32)

    src_roi = img[y0:y1, x0:x1]
    dst_roi = src_roi.copy()

    for indices in triangles:
        src_tri = src_points[list(indices)] - offset
        dst_tri = dst_points[list(indices)] - offset
        if triangle_area(src_tri) < 1.0 or triangle_area(dst_tri) < 1.0:
            continue
        warp_triangle_roi(src_roi, dst_roi, src_tri, dst_tri)

    result = img.copy()
    result[y0:y1, x0:x1] = dst_roi
    return result


def draw_debug_points(img, src_points, dst_points, point_kinds):
    debug = img.copy()
    for src, dst, kind in zip(src_points, dst_points, point_kinds):
        if kind == "anchor":
            cv2.circle(debug, tuple(np.round(dst).astype(int)), 3, (255, 0, 0), -1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(debug, tuple(np.round(src).astype(int)), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(debug, tuple(np.round(dst).astype(int)), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    return debug


def draw_debug_mesh(img, src_points, dst_points, point_kinds, triangles):
    debug = img.copy()
    for indices in triangles:
        pts = np.round(dst_points[list(indices)]).astype(np.int32)
        cv2.polylines(debug, [pts], isClosed=True, color=(160, 160, 160), thickness=1, lineType=cv2.LINE_AA)
    return draw_debug_points(debug, src_points, dst_points, point_kinds)


def resize_face_local(input_path, output_path, scale_x, scale_y, anchor_scale, forehead_expand, debug_points=None, debug_mesh=None):
    img = read_image(input_path)
    h, w = img.shape[:2]

    landmarks = detect_landmarks_bgr(img)
    if landmarks is None:
        raise RuntimeError("No face detected.")

    src_points, dst_points, point_kinds, _, _, _ = build_control_points(
        landmarks,
        w,
        h,
        scale_x,
        scale_y,
        anchor_scale,
        forehead_expand,
    )
    if len(dst_points) < 4:
        raise RuntimeError("Not enough control points for mesh warp.")

    triangles = delaunay_triangles(dst_points, w, h)
    if not triangles:
        raise RuntimeError("Delaunay triangulation produced no usable triangles.")

    result = warp_mesh(img, src_points, dst_points, triangles)
    write_image(output_path, result)

    if debug_points:
        write_image(debug_points, draw_debug_points(img, src_points, dst_points, point_kinds))
    if debug_mesh:
        write_image(debug_mesh, draw_debug_mesh(img, src_points, dst_points, point_kinds, triangles))


def iter_image_paths(folder, recursive=False):
    folder = Path(folder)
    pattern = "**/*" if recursive else "*"
    for path in sorted(folder.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def output_path_for(input_path, input_folder, output_folder):
    rel_path = input_path.relative_to(input_folder)
    return Path(output_folder) / rel_path


def validate_common_args(parser, args):
    scale_x = args.scale if args.scale_x is None else args.scale_x
    scale_y = args.scale if args.scale_y is None else args.scale_y

    if scale_x <= 0 or scale_y <= 0:
        parser.error("scale values must be greater than zero")
    if args.anchor_scale <= 1.0:
        parser.error("--anchor-scale must be greater than 1.0")
    if args.forehead_expand < 0:
        parser.error("--forehead-expand must be greater than or equal to 0")

    return scale_x, scale_y


def validate_args(parser, args):
    scale_x, scale_y = validate_common_args(parser, args)

    batch_args = [args.input_folder, args.output_folder]
    batch_mode = any(batch_args)

    if batch_mode:
        if not all(batch_args):
            parser.error("--input-folder and --output-folder must be used together")
        if args.input or args.output:
            parser.error("Use either --input-folder/--output-folder or --input/--output, not both")
        if args.debug_points or args.debug_mesh:
            parser.error("--debug-points and --debug-mesh are only for single-image mode")
        return "batch", scale_x, scale_y

    if not args.input or not args.output:
        parser.error("single-image mode requires --input and --output")
    if args.recursive:
        parser.error("--recursive can only be used with --input-folder")

    return "single", scale_x, scale_y


def run_single(args, scale_x, scale_y):
    resize_face_local(
        input_path=args.input,
        output_path=args.output,
        scale_x=scale_x,
        scale_y=scale_y,
        anchor_scale=args.anchor_scale,
        forehead_expand=args.forehead_expand,
        debug_points=args.debug_points,
        debug_mesh=args.debug_mesh,
    )
    print("saved:", Path(args.output))


def run_batch(args, scale_x, scale_y):
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
        print(f"[{index}/{len(image_paths)}] processing: {input_path}")

        try:
            resize_face_local(
                input_path=input_path,
                output_path=output_path,
                scale_x=scale_x,
                scale_y=scale_y,
                anchor_scale=args.anchor_scale,
                forehead_expand=args.forehead_expand,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resize a detected face with a continuous face-local landmark mesh warp."
    )
    parser.add_argument("--input", help="Input image path.")
    parser.add_argument("--output", help="Output image path.")
    parser.add_argument("--input-folder", help="Folder of images to process one by one.")
    parser.add_argument("--output-folder", help="Folder to save batch results.")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform face-local scale.")
    parser.add_argument("--scale-x", type=float, default=None, help="Face-local horizontal scale.")
    parser.add_argument("--scale-y", type=float, default=None, help="Face-local vertical scale.")
    parser.add_argument("--anchor-scale", type=float, default=1.6, help="Expansion factor for fixed outer anchors.")
    parser.add_argument("--forehead-expand", type=float, default=0.0, help="Add moving forehead controls above the face oval as a fraction of face height.")
    parser.add_argument("--debug-points", default=None, help="Optional image showing source, destination, and anchor points.")
    parser.add_argument("--debug-mesh", default=None, help="Optional image showing destination Delaunay mesh.")
    parser.add_argument("--recursive", action="store_true", help="Process images in nested folders.")
    args = parser.parse_args()
    mode, scale_x, scale_y = validate_args(parser, args)
    return args, mode, scale_x, scale_y


def main():
    args, mode, scale_x, scale_y = parse_args()
    if mode == "batch":
        run_batch(args, scale_x, scale_y)
    else:
        run_single(args, scale_x, scale_y)


if __name__ == "__main__":
    main()
