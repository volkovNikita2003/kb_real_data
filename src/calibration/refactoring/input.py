"""Readers and HDR construction for calibration detector data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def camera_exposures(directory: Path) -> list[int]:
    result = sorted(int(path.stem) for path in directory.glob("*.bmp"))
    if not result:
        raise ValueError(f"Не найдены калибровочные BMP-файлы: {directory}")
    return result


def build_camera_hdr(
    directory: Path,
    exposures: list[int],
    *,
    image_shape: tuple[int, int],
    low_threshold: float = 10.0,
    top_threshold: float = 240.0,
    background_level: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the legacy low-to-high ``l2h`` HDR calibration mode."""
    reference_exposure = max(exposures)
    hdr = np.zeros(image_shape, dtype=np.float64)
    valid_any = np.zeros(image_shape, dtype=bool)
    for exposure in sorted(exposures):
        image = np.asarray(Image.open(directory / f"{exposure}.bmp"), dtype=np.float64)
        if image.shape != image_shape:
            raise ValueError(
                f"Размер {directory / f'{exposure}.bmp'} равен {image.shape}, "
                f"ожидался {image_shape}"
            )
        saturated = image > top_threshold
        image -= background_level
        valid = ~saturated & (image >= low_threshold)
        add = valid & ~valid_any
        effective_exposure = np.floor(exposure / 23.4) * 23.4
        if effective_exposure <= 0:
            raise ValueError(f"Экспозиция {exposure} слишком мала для HDR-модели")
        image[image < 0] = 0
        hdr[add] = image[add] * reference_exposure / effective_exposure
        valid_any |= add
    return hdr, valid_any


def read_line_signal(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # The legacy files use a tab delimiter and occasionally decimal commas.
    raw = np.genfromtxt(path, delimiter="\t", skip_header=1, dtype=str)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"Некорректный файл сигнала линейки: {path}")
    try:
        indices = np.array([int(value) for value in raw[:, 0]])
        amplitude = np.array(
            [float(value.replace(",", ".")) for value in raw[:, 1]], dtype=float
        )
    except ValueError as error:
        raise ValueError(f"Некорректные числа в файле линейки {path}: {error}") from error
    reversed_indices = indices.max() - indices
    order = np.argsort(reversed_indices)
    return reversed_indices[order], amplitude[order]
