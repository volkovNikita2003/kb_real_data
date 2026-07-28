"""Geometry variants of the two camera positions used in experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NATIVE_PIXEL_WIDTH_M = 2.2e-6 / 0.58 / 2


@dataclass(frozen=True)
class CameraSearchGeometry:
    y_lines: range
    x_min: int
    x_max: int | None
    x_lines: range
    y_boundary: int
    reverse_x: bool


def search_geometry(position: str, height: int, width: int) -> CameraSearchGeometry:
    if position == "new":
        return CameraSearchGeometry(
            range(400, min(1501, height)), 0, min(1000, width),
            range(0, min(400, width)), height // 2, False,
        )
    if position == "old":
        return CameraSearchGeometry(
            range(400, min(1501, height)), min(1700, width - 1), None,
            range(min(2100, width), width), height // 2, True,
        )
    raise ValueError(f"Неизвестное положение детекторов: {position!r}")


def camera_shift_m(center: np.ndarray, position: str, height: int, width: int) -> np.ndarray:
    if position == "new":
        origin = np.array([0.0, height / 2.0])
        return (origin - center) * NATIVE_PIXEL_WIDTH_M
    if position == "old":
        origin = np.array([float(width), height / 2.0])
        return (center - origin) * NATIVE_PIXEL_WIDTH_M
    raise ValueError(f"Неизвестное положение детекторов: {position!r}")
