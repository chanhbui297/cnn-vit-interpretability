"""Deterministic RGB crop utilities for the adversarial cropping pilot."""

from dataclasses import dataclass
from math import sqrt
from typing import Literal

from PIL import Image


CropPosition = Literal[
    "center", "top_left", "top_right", "bottom_left", "bottom_right"
]

CROP_AREAS = {
    "area_095": 0.95,
    "area_080": 0.80,
    "area_065": 0.65,
    "area_050": 0.50,
}
CROP_POSITIONS: tuple[CropPosition, ...] = (
    "center",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
)


@dataclass(frozen=True)
class CropResult:
    """A cropped RGB image and its reproducible source-image geometry."""

    image: Image.Image
    condition: str
    severity: str
    box: tuple[int, int, int, int]
    original_size: tuple[int, int]
    crop_size: tuple[int, int]
    realized_area_ratio: float


def get_crop_box(
    image_size: tuple[int, int],
    area_ratio: float,
    position: CropPosition,
) -> tuple[int, int, int, int]:
    """Return an exclusive PIL crop box for a retained-area crop."""
    width, height = image_size
    _validate_image_size(width, height)
    _validate_area_ratio(area_ratio)
    _validate_position(position)

    crop_width = max(1, round(width * sqrt(area_ratio)))
    crop_height = max(1, round(height * sqrt(area_ratio)))
    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)

    left, top = _get_crop_origin(
        width, height, crop_width, crop_height, position
    )
    return left, top, left + crop_width, top + crop_height


def crop_image(
    image: Image.Image,
    area_name: str,
    position: CropPosition,
) -> CropResult:
    """Crop an RGB PIL image using one of the fixed pilot conditions."""
    if image.mode != "RGB":
        raise ValueError(f"Expected an RGB image, got mode {image.mode!r}")
    if area_name not in CROP_AREAS:
        raise ValueError(f"Unknown crop area {area_name!r}")

    area_ratio = CROP_AREAS[area_name]
    box = get_crop_box(image.size, area_ratio, position)
    cropped = image.crop(box)
    original_width, original_height = image.size
    crop_width, crop_height = cropped.size
    realized_area = (crop_width * crop_height) / (original_width * original_height)
    condition = f"{area_name}__{position}"
    return CropResult(
        image=cropped,
        condition=condition,
        severity=f"{area_ratio:.2f}",
        box=box,
        original_size=image.size,
        crop_size=cropped.size,
        realized_area_ratio=realized_area,
    )


def iter_crop_conditions(image: Image.Image):
    """Yield all six fixed crop conditions in the protocol order."""
    for area_name in CROP_AREAS:
        for position in CROP_POSITIONS:
            yield crop_image(image, area_name, position)


def _get_crop_origin(
    width: int,
    height: int,
    crop_width: int,
    crop_height: int,
    position: CropPosition,
) -> tuple[int, int]:
    if position == "center":
        return (width - crop_width) // 2, (height - crop_height) // 2
    if position == "top_left":
        return 0, 0
    if position == "top_right":
        return width - crop_width, 0
    if position == "bottom_left":
        return 0, height - crop_height
    return width - crop_width, height - crop_height


def _validate_image_size(width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")


def _validate_area_ratio(area_ratio: float) -> None:
    if not 0 < area_ratio <= 1:
        raise ValueError("area_ratio must be greater than 0 and at most 1")


def _validate_position(position: str) -> None:
    if position not in CROP_POSITIONS:
        raise ValueError(f"Unknown crop position {position!r}")
