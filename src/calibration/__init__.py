"""Public calibration API.

The current refactoring is retained as an experimental implementation while
the unchanged legacy calculations are adapted to the automated pipeline.
"""

from .refactoring import (
    CalibrationArtifacts,
    CalibrationResult,
    CalibrationResultError,
    CameraCalibrationResult,
    LineSensorCalibrationResult,
    calibrate,
    load_calibration_result,
    save_calibration_result,
)

__all__ = [
    "CalibrationResult",
    "CalibrationResultError",
    "CameraCalibrationResult",
    "LineSensorCalibrationResult",
    "load_calibration_result",
    "save_calibration_result",
    "CalibrationArtifacts",
    "calibrate",
]
