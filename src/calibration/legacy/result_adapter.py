"""Publish original calculation variables in the automated result format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calibration.refactoring.result import (
    RESULT_SCHEMA_VERSION,
    CalibrationResult,
    CameraCalibrationResult,
    LineSensorCalibrationResult,
    save_calibration_result,
)


def save_legacy_result(
    directory: Path,
    *,
    diagonal_mm: float,
    width_px: int,
    height_px: int,
    camera_shift_um,
    camera_pixel_width_m: float,
    camera_distance_um,
    camera_signal,
    camera_fit,
    line_values: dict[str, object] | None = None,
) -> None:
    line_result = None
    if line_values is not None:
        line_result = LineSensorCalibrationResult(
            start_angle_deg=float(line_values["start_angle_deg"]),
            end_angle_deg=float(line_values["end_angle_deg"]),
            logarithmic_radius_percent=float(line_values["logarithmic_radius_percent"]),
            pixel_width_m=float(line_values["pixel_width_m"]),
            pixel_height_m=float(line_values["pixel_height_m"]),
            to_camera_coefficient=float(line_values["to_camera_coefficient"]),
            shift_m=float(line_values["shift_m"]),
            peak_pixel=float(line_values["peak_pixel"]),
        )
    result = CalibrationResult(
        RESULT_SCHEMA_VERSION,
        CameraCalibrationResult(
            matrix_diagonal_mm=float(diagonal_mm),
            width_px=int(width_px),
            height_px=int(height_px),
            x_shift_m=float(camera_shift_um[0]) * 1e-6,
            y_shift_m=float(camera_shift_um[1]) * 1e-6,
            pixel_width_m=float(camera_pixel_width_m),
        ),
        line_result,
    )
    save_calibration_result(result, directory / "result.yaml")
    np.savetxt(
        directory / "camera-calibration-signal.txt",
        np.column_stack((camera_distance_um, camera_signal, camera_fit)),
        delimiter="\t",
        header="distance_um\tnormalized_signal\tfitted_signal",
        comments="",
        fmt="%.17g",
    )
    if line_values is not None:
        np.savetxt(
            directory / "line-calibration-signal.txt",
            np.column_stack((
                line_values["distance_um"],
                line_values["signal"],
                line_values["fit"],
            )),
            delimiter="\t",
            header="distance_um\tnormalized_signal\tfitted_signal",
            comments="",
            fmt="%.17g",
        )
