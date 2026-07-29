"""Pure(ish) calibration orchestration, independent from the command line."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, least_squares
from scipy.special import j1

from experiment import Experiment
from parameters import ExperimentParameters

from .geometry import NATIVE_PIXEL_WIDTH_M, camera_shift_m, search_geometry
from .input import build_camera_hdr, camera_exposures, read_line_signal
from .result import (
    CalibrationResult,
    CameraCalibrationResult,
    LineSensorCalibrationResult,
    RESULT_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class CalibrationArtifacts:
    result: CalibrationResult
    camera_signal: np.ndarray
    line_signal: np.ndarray | None
    quality: dict[str, float | int]


def _jinc(x: np.ndarray) -> np.ndarray:
    values = np.empty_like(np.asarray(x), dtype=float)
    nonzero = x != 0
    values[nonzero] = 2.0 * j1(x[nonzero]) / x[nonzero]
    values[~nonzero] = 1.0
    return values


def _airy(x_um: np.ndarray, amplitude: float, diameter_um: float, gap_um: float,
          wavelength_um: float, focal_length_um: float) -> np.ndarray:
    argument = (
        np.pi * diameter_um / wavelength_um
        * (x_um - gap_um) / focal_length_um
    )
    return amplitude * _jinc(argument) ** 2


def _fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    x, y = points[:, 0], points[:, 1]
    initial_x, initial_y = np.mean(x), np.mean(y)
    initial_r = np.mean(np.hypot(x - initial_x, y - initial_y))
    result = least_squares(
        lambda p: np.hypot(x - p[0], y - p[1]) - p[2],
        [initial_x, initial_y, initial_r],
    )
    if not result.success:
        raise RuntimeError(f"Не удалось аппроксимировать кольцо: {result.message}")
    return tuple(float(item) for item in result.x)


def _find_circle(data: np.ndarray, position: str) -> tuple[np.ndarray, float, np.ndarray]:
    height, width = data.shape
    geometry = search_geometry(position, height, width)
    points: list[tuple[int, int]] = []
    for y in geometry.y_lines:
        row = gaussian_filter(data[y, :], sigma=1)
        part = row[geometry.x_min:geometry.x_max]
        indices = np.flatnonzero(part == part.min())
        points.append((geometry.x_min + int(indices[len(indices) // 2]), y))
    for x in geometry.x_lines:
        column = gaussian_filter(data[:, x], sigma=1)
        upper = column[:geometry.y_boundary]
        lower = column[geometry.y_boundary:]
        first = np.flatnonzero(upper == upper.min())
        second = np.flatnonzero(lower == lower.min())
        points.append((x, int(first[len(first) // 2])))
        points.append((x, geometry.y_boundary + int(second[len(second) // 2])))
    point_array = np.asarray(points, dtype=float)
    xc, yc, radius = _fit_circle(point_array)
    return np.array([xc, yc]), radius, point_array


def _pinhole_diameter(radius_m: float, wavelength_m: float, focal_length_m: float) -> float:
    angle = np.arctan(radius_m / focal_length_m)
    return 1.22 * wavelength_m / np.sin(angle)


def _rmse(observed: np.ndarray, fitted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observed - fitted) ** 2)))


def _save_circle_figure(data: np.ndarray, points: np.ndarray, center: np.ndarray,
                        radius: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(data, cmap="gray", origin="upper")
    ax.scatter(points[:, 0], points[:, 1], s=3, alpha=0.25)
    ax.add_patch(Circle(center, radius, fill=False, color="red"))
    ax.plot(center[0], center[1], "r+")
    ax.set_title(f"Центр колец ({center[0]:.1f}, {center[1]:.1f}) px")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def calibrate(experiment: Experiment, parameters: ExperimentParameters,
              output_directory: Path) -> CalibrationArtifacts:
    """Calibrate configured detectors and write diagnostic artifacts."""
    camera_parameters = parameters.calibration.camera
    camera_geometry = parameters.general.detectors.camera
    if camera_parameters is None or camera_geometry is None:
        raise ValueError("Автоматическая калибровка требует камеру")

    figures = output_directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    exposures = camera_exposures(experiment.calibration_camera_dir)
    shape = (camera_geometry.height_px, camera_geometry.width_px)
    hdr, valid_mask = build_camera_hdr(
        experiment.calibration_camera_dir, exposures, image_shape=shape
    )
    filtered = gaussian_filter(
        hdr, sigma=(camera_parameters.gaussian_sigma_px,) * 2
    )
    position = parameters.general.instrument.detector_position
    center, radius, circle_points = _find_circle(filtered, position)
    _save_circle_figure(filtered, circle_points, center, radius,
                        figures / "camera-fitted-rings.png")

    wavelength_um = parameters.general.instrument.wavelength_um
    focal_length_um = parameters.general.instrument.focal_length_um
    pinhole_from_ring_um = _pinhole_diameter(
        radius * NATIVE_PIXEL_WIDTH_M, wavelength_um * 1e-6,
        focal_length_um * 1e-6,
    ) * 1e6
    shift = camera_shift_m(center, position, *shape)
    x_pixels = np.arange(camera_geometry.width_px, dtype=float)
    direction = -1.0 if position == "old" else 1.0
    x_m = direction * (x_pixels - center[0]) * NATIVE_PIXEL_WIDTH_M
    half_width = int(200e-6 / NATIVE_PIXEL_WIDTH_M / 2)
    y = int(center[1])
    if y - half_width < 0 or y + half_width > shape[0]:
        raise ValueError("Полоса усреднения камеры выходит за границы изображения")
    profile = np.mean(filtered[y-half_width:y+half_width, :], axis=0)
    normalized = profile / max(exposures) / NATIVE_PIXEL_WIDTH_M**2

    fit_x = x_m * 1e6
    fit_y = normalized
    if position == "old":
        fit_x, fit_y = fit_x[:-1], fit_y[:-1]
    model_no_gap = lambda x, amplitude, diameter: _airy(
        x, amplitude, diameter, 0.0, wavelength_um, focal_length_um
    )
    fitted_parameters, _ = curve_fit(
        model_no_gap, fit_x, fit_y,
        p0=[float(fit_y.max()), pinhole_from_ring_um],
    )
    scale = (
        fitted_parameters[1] / camera_parameters.pinhole_diameter_um
        if camera_parameters.correct_pixel_size else 1.0
    )
    corrected_pixel = NATIVE_PIXEL_WIDTH_M * scale
    corrected_x_um = x_m * scale * 1e6
    corrected_shift = shift * scale
    corrected_fit_x = corrected_x_um[:-1] if position == "old" else corrected_x_um
    corrected_fit_y = normalized[:-1] if position == "old" else normalized
    corrected_parameters, _ = curve_fit(
        model_no_gap, corrected_fit_x, corrected_fit_y,
        p0=[float(corrected_fit_y.max()), pinhole_from_ring_um],
    )
    camera_fit = model_no_gap(corrected_x_um, *corrected_parameters)
    camera_signal = np.column_stack((corrected_x_um, normalized, camera_fit))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(corrected_x_um, normalized, label="signal")
    ax.plot(corrected_x_um, camera_fit, label="fit")
    ax.set_xlabel("X, um")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "camera-fit.png")
    plt.close(fig)

    camera_result = CameraCalibrationResult(
        matrix_diagonal_mm=float(corrected_pixel * np.hypot(*shape[::-1]) * 1e3),
        x_shift_m=float(corrected_shift[0]),
        y_shift_m=float(corrected_shift[1]),
        pixel_width_m=float(corrected_pixel),
    )
    quality: dict[str, float | int] = {
        "camera_fit_rmse": _rmse(normalized, camera_fit),
        "camera_valid_pixel_count": int(valid_mask.sum()),
        "camera_total_pixel_count": int(valid_mask.size),
        "camera_exposure_count": len(exposures),
        "camera_pixel_scale": float(scale),
    }

    line_artifact: np.ndarray | None = None
    line_result: LineSensorCalibrationResult | None = None
    line_parameters = parameters.calibration.line_sensor
    line_geometry = parameters.general.detectors.line_sensor
    if line_parameters is not None and line_geometry is not None:
        line_files = list(experiment.calibration_line_dir.glob("*.txt"))
        if len(line_files) != 1:
            raise ValueError("Для калибровки линейки требуется ровно один TXT-файл")
        exposure = int(line_files[0].stem) + line_parameters.time_offset_us
        pixels, amplitude = read_line_signal(line_files[0])
        signal = amplitude - amplitude.min()
        filtered_line = gaussian_filter(signal, sigma=line_parameters.gaussian_sigma_px)
        width_m = line_geometry.pixel_width_m
        height_m = line_geometry.pixel_height_m
        normalized_line = filtered_line / exposure / width_m / height_m
        peak = int(np.argmax(filtered_line))
        start, stop = peak + 150, peak + 250
        if stop > len(filtered_line):
            raise ValueError("Невозможно найти первый минимум сигнала линейки")
        first_minimum = start + int(np.argmin(filtered_line[start:stop]))
        minimum_width = (first_minimum - peak) * width_m
        pinhole_line_um = _pinhole_diameter(
            minimum_width, wavelength_um * 1e-6, focal_length_um * 1e-6
        ) * 1e6
        x_line_um = (pixels - peak) * (line_geometry.pixel_width_m * 1e6)
        model_line = lambda x, amplitude_value, diameter, gap: _airy(
            x, amplitude_value, diameter, gap, wavelength_um, focal_length_um
        )
        line_fit_parameters, _ = curve_fit(
            model_line, x_line_um, normalized_line,
            p0=[float(normalized_line.max()), pinhole_line_um, 0.0],
        )
        line_scale = line_fit_parameters[1] / line_parameters.pinhole_diameter_um
        corrected_line_x = line_scale * (x_line_um - line_fit_parameters[2])
        corrected_line_width = line_scale * width_m
        corrected_line_height = line_scale * height_m
        corrected_line_shift = line_scale * (
            line_parameters.signal_position_m - line_parameters.pinhole_position_m
        )
        corrected_peak = peak + line_fit_parameters[2] / (line_geometry.pixel_width_m * 1e6)
        final_line_parameters, _ = curve_fit(
            model_line, corrected_line_x, normalized_line,
            p0=[float(normalized_line.max()), pinhole_line_um, 0.0],
        )
        fitted_line = model_line(corrected_line_x, *final_line_parameters)
        line_artifact = np.column_stack(
            (corrected_line_x, normalized_line, fitted_line)
        )
        coordinates_m = corrected_line_x * 1e-6 + corrected_line_shift
        angles = np.degrees(np.arctan(coordinates_m / (focal_length_um * 1e-6)))
        line_result = LineSensorCalibrationResult(
            start_angle_deg=float(angles.min()),
            end_angle_deg=float(angles.max()),
            pixel_width_m=float(corrected_line_width),
            pixel_height_m=float(corrected_line_height),
            to_camera_coefficient=float(corrected_parameters[0] / final_line_parameters[0]),
            shift_m=float(corrected_line_shift),
            peak_pixel=float(corrected_peak),
        )
        quality.update({
            "line_fit_rmse": _rmse(normalized_line, fitted_line),
            "line_pixel_scale": float(line_scale),
            "line_peak_pixel_uncorrected": peak,
        })
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(corrected_line_x, normalized_line, label="signal")
        ax.plot(corrected_line_x, fitted_line, label="fit")
        ax.set_xlabel("X, um")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "line-fit.png")
        plt.close(fig)

    return CalibrationArtifacts(
        result=CalibrationResult(RESULT_SCHEMA_VERSION, camera_result, line_result),
        camera_signal=camera_signal,
        line_signal=line_artifact,
        quality=quality,
    )
