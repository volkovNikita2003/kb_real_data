"""Public calibration API.

The current refactoring is retained as an experimental implementation while
the unchanged legacy calculations are adapted to the automated pipeline.
"""

from .refactoring.result import (
    CalibrationResult,
    CalibrationResultError,
    CameraCalibrationResult,
    LineSensorCalibrationResult,
    load_calibration_result,
    save_calibration_result,
)


def __getattr__(name: str):
    if name in {"CalibrationArtifacts", "calibrate"}:
        from .refactoring.engine import CalibrationArtifacts, calibrate

        return {
            "CalibrationArtifacts": CalibrationArtifacts,
            "calibrate": calibrate,
        }[name]
    raise AttributeError(name)

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
