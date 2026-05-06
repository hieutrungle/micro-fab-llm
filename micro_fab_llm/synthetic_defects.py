"""Generate synthetic wafer defect images with OpenCV Poisson blending.

The generated labels are JSON files with normalized YOLO-style bounding boxes:
``{"defect": "scratch", "bbox_yolo": [x_center, y_center, width, height]}``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


DEFAULT_COUNT = 500
DEFAULT_IMAGE_SIZE = 512
DEFAULT_OUTPUT_DIR = Path("dataset")


@dataclass(frozen=True)
class DefectAsset:
    """Transparent mock defect asset stored as a BGRA image."""

    defect_type: str
    bgra: np.ndarray


@dataclass(frozen=True)
class InjectedDefect:
    """Result of a single defect injection."""

    image: np.ndarray
    defect_type: str
    bbox_yolo: tuple[float, float, float, float]


def generate_clean_wafer(
    size: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create a clean brushed-metal wafer tile as a BGR image."""

    rng = rng or np.random.default_rng()
    height, width = size

    base = np.full((height, width), 136.0, dtype=np.float32)
    fine_noise = rng.normal(loc=0.0, scale=9.0, size=(height, width)).astype(np.float32)
    brushed_noise = cv2.GaussianBlur(fine_noise, ksize=(0, 0), sigmaX=18.0, sigmaY=1.2)

    low_frequency = rng.normal(loc=0.0, scale=5.0, size=(height, width)).astype(np.float32)
    low_frequency = cv2.GaussianBlur(low_frequency, ksize=(0, 0), sigmaX=80.0, sigmaY=80.0)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    radius /= max(center_x, center_y)
    vignette = -10.0 * np.clip(radius, 0.0, 1.0) ** 2

    wafer_gray = np.clip(base + brushed_noise + low_frequency + vignette, 0, 255).astype(np.uint8)
    return cv2.cvtColor(wafer_gray, cv2.COLOR_GRAY2BGR)


def generate_mock_defect_assets(rng: np.random.Generator | None = None) -> list[DefectAsset]:
    """Create transparent BGRA mock assets for scratch, particle, and bridge_short defects."""

    rng = rng or np.random.default_rng()
    return [
        DefectAsset("scratch", _make_scratch_asset(rng)),
        DefectAsset("particle", _make_particle_asset(rng)),
        DefectAsset("bridge_short", _make_bridge_short_asset(rng)),
    ]


def inject_random_defect(
    wafer: np.ndarray,
    assets: Sequence[DefectAsset],
    rng: np.random.Generator | None = None,
) -> InjectedDefect:
    """Inject one randomly selected defect into a wafer image using Poisson blending."""

    if not assets:
        raise ValueError("At least one defect asset is required.")

    rng = rng or np.random.default_rng()
    asset = assets[int(rng.integers(0, len(assets)))]
    return inject_defect(wafer=wafer, asset=_augment_asset(asset, rng), rng=rng)


def inject_defect(
    wafer: np.ndarray,
    asset: DefectAsset,
    rng: np.random.Generator | None = None,
) -> InjectedDefect:
    """Place a defect at a random valid coordinate and return its image plus YOLO bbox."""

    rng = rng or np.random.default_rng()
    image_height, image_width = wafer.shape[:2]
    defect_height, defect_width = asset.bgra.shape[:2]

    if defect_height > image_height or defect_width > image_width:
        raise ValueError("Defect asset must fit inside the wafer image.")

    top_left_x = int(rng.integers(0, image_width - defect_width + 1))
    top_left_y = int(rng.integers(0, image_height - defect_height + 1))
    center = (top_left_x + defect_width // 2, top_left_y + defect_height // 2)

    alpha = asset.bgra[:, :, 3]
    alpha_float = alpha.astype(np.float32) / 255.0
    alpha_float = alpha_float[:, :, np.newaxis]

    defect_bgr = asset.bgra[:, :, :3]
    wafer_patch = wafer[
        top_left_y : top_left_y + defect_height,
        top_left_x : top_left_x + defect_width,
    ].copy()
    source_patch = (defect_bgr.astype(np.float32) * alpha_float) + (
        wafer_patch.astype(np.float32) * (1.0 - alpha_float)
    )
    source_patch = np.clip(source_patch, 0, 255).astype(np.uint8)

    clone_mask = cv2.dilate((alpha > 0).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    blended = cv2.seamlessClone(source_patch, wafer, clone_mask, center, cv2.NORMAL_CLONE)

    asset_bbox = _alpha_bbox(alpha)
    bbox = (
        top_left_x + asset_bbox[0],
        top_left_y + asset_bbox[1],
        top_left_x + asset_bbox[2],
        top_left_y + asset_bbox[3],
    )

    return InjectedDefect(
        image=blended,
        defect_type=asset.defect_type,
        bbox_yolo=_bbox_to_yolo(bbox=bbox, image_width=image_width, image_height=image_height),
    )


def _augment_asset(asset: DefectAsset, rng: np.random.Generator) -> DefectAsset:
    """Randomly scale and rotate a transparent defect asset without cropping it."""

    scale = float(rng.uniform(0.5, 1.5))
    angle_degrees = float(rng.uniform(0.0, 360.0))

    height, width = asset.bgra.shape[:2]
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(asset.bgra, (scaled_width, scaled_height), interpolation=interpolation)

    center = (scaled_width / 2.0, scaled_height / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos_theta = abs(rotation[0, 0])
    sin_theta = abs(rotation[0, 1])
    rotated_width = int(np.ceil((scaled_height * sin_theta) + (scaled_width * cos_theta)))
    rotated_height = int(np.ceil((scaled_height * cos_theta) + (scaled_width * sin_theta)))

    rotation[0, 2] += (rotated_width / 2.0) - center[0]
    rotation[1, 2] += (rotated_height / 2.0) - center[1]

    rotated = cv2.warpAffine(
        scaled,
        rotation,
        (rotated_width, rotated_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return DefectAsset(defect_type=asset.defect_type, bgra=_trim_transparent_border(rotated))


def generate_dataset(
    count: int = DEFAULT_COUNT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int | None = 42,
) -> None:
    """Generate synthetic wafer images and paired JSON labels."""

    if count <= 0:
        raise ValueError("count must be greater than zero.")
    if image_size <= 0:
        raise ValueError("image_size must be greater than zero.")

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    assets = generate_mock_defect_assets(rng)

    for index in range(count):
        wafer = generate_clean_wafer(size=(image_size, image_size), rng=rng)
        injected = inject_random_defect(wafer=wafer, assets=assets, rng=rng)

        stem = f"wafer_{index:04d}"
        image_path = images_dir / f"{stem}.png"
        label_path = labels_dir / f"{stem}.json"

        if not cv2.imwrite(str(image_path), injected.image):
            raise RuntimeError(f"Failed to write image: {image_path}")

        label = {
            "defect": injected.defect_type,
            "bbox_yolo": [round(value, 6) for value in injected.bbox_yolo],
        }
        label_path.write_text(json.dumps(label, indent=2) + "\n", encoding="utf-8")


def _make_scratch_asset(rng: np.random.Generator) -> np.ndarray:
    height = int(rng.integers(40, 72))
    width = int(rng.integers(120, 196))
    alpha = np.zeros((height, width), dtype=np.uint8)
    base_thickness = int(rng.integers(3, 8))
    inflection_points = int(rng.integers(5, 13))
    points = []
    for x_pos in np.linspace(10, width - 10, num=inflection_points):
        y_pos = height // 2 + int(rng.integers(-(height // 4), (height // 4) + 1))
        points.append([int(x_pos), y_pos])

    scratch_points = np.array(points, dtype=np.int32)
    cv2.polylines(
        alpha,
        [scratch_points],
        isClosed=False,
        color=230,
        thickness=base_thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.polylines(
        alpha,
        [scratch_points],
        isClosed=False,
        color=255,
        thickness=max(1, base_thickness // 2),
        lineType=cv2.LINE_AA,
    )
    return _make_bgra_from_alpha(alpha=alpha, base_bgr_color=(28, 28, 28), rng=rng)


def _make_particle_asset(rng: np.random.Generator) -> np.ndarray:
    height, width = 64, 64
    alpha = np.zeros((height, width), dtype=np.uint8)
    center = (
        width // 2 + int(rng.integers(-3, 4)),
        height // 2 + int(rng.integers(-3, 4)),
    )

    cv2.ellipse(
        alpha,
        center,
        axes=(15, 12),
        angle=20,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    for _ in range(5):
        offset_x = int(rng.integers(-13, 14))
        offset_y = int(rng.integers(-11, 12))
        radius = int(rng.integers(4, 8))
        cv2.circle(
            alpha,
            (center[0] + offset_x, center[1] + offset_y),
            radius,
            255,
            -1,
            lineType=cv2.LINE_AA,
        )

    return _make_bgra_from_alpha(alpha=alpha, base_bgr_color=(238, 238, 238), rng=rng)


def _make_bridge_short_asset(rng: np.random.Generator) -> np.ndarray:
    pad_width = int(rng.integers(16, 34))
    pad_height = int(rng.integers(38, 74))
    pad_spacing = int(rng.integers(16, 42))
    line_thickness = int(rng.integers(6, 18))
    margin_x = int(rng.integers(14, 26))
    margin_y = int(rng.integers(12, 24))

    width = (2 * margin_x) + (2 * pad_width) + pad_spacing
    height = (2 * margin_y) + pad_height
    alpha = np.zeros((height, width), dtype=np.uint8)

    left_pad_start = (margin_x, margin_y)
    left_pad_end = (margin_x + pad_width, margin_y + pad_height)
    right_pad_start = (left_pad_end[0] + pad_spacing, margin_y)
    right_pad_end = (right_pad_start[0] + pad_width, margin_y + pad_height)
    bridge_y = int(rng.integers(margin_y + line_thickness, margin_y + pad_height - line_thickness + 1))

    cv2.rectangle(alpha, left_pad_start, left_pad_end, 255, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(alpha, right_pad_start, right_pad_end, 255, -1, lineType=cv2.LINE_AA)
    cv2.line(
        alpha,
        (left_pad_end[0] - 1, bridge_y),
        (right_pad_start[0] + 1, bridge_y),
        255,
        line_thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(
        alpha,
        ((left_pad_end[0] + right_pad_start[0]) // 2, bridge_y),
        max(4, line_thickness),
        255,
        -1,
        lineType=cv2.LINE_AA,
    )

    return _make_bgra_from_alpha(alpha=alpha, base_bgr_color=(45, 45, 45), rng=rng)


def _make_bgra_from_alpha(
    alpha: np.ndarray,
    base_bgr_color: tuple[int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    bgr = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    foreground = alpha > 0
    base_color = np.array(base_bgr_color, dtype=np.float32)
    color_noise = rng.normal(loc=0.0, scale=20.0, size=(*alpha.shape, 3)).astype(np.float32)
    bgr[foreground] = np.clip(base_color + color_noise[foreground], 0, 255).astype(np.uint8)
    return np.dstack([bgr, alpha])


def _trim_transparent_border(bgra: np.ndarray) -> np.ndarray:
    x_min, y_min, x_max, y_max = _alpha_bbox(bgra[:, :, 3])
    return bgra[y_min:y_max, x_min:x_max].copy()


def _alpha_bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Defect alpha channel is empty.")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _bbox_to_yolo(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bbox
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    x_center = x_min + (bbox_width / 2.0)
    y_center = y_min + (bbox_height / 2.0)
    return (
        x_center / image_width,
        y_center / image_height,
        bbox_width / image_width,
        bbox_height / image_height,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic wafer defect images and labels.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of image/label pairs to generate.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Dataset output directory.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Square image size in pixels.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible data generation.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    generate_dataset(count=args.count, output_dir=args.output_dir, image_size=args.image_size, seed=args.seed)
    print(f"Generated {args.count} synthetic wafer samples in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
