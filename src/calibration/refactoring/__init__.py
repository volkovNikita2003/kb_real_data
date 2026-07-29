"""Experimental refactoring of the legacy calibration calculations."""

from .result import (
    CalibrationResult,
    CalibrationResultError,
    CameraCalibrationResult,
    LineSensorCalibrationResult,
    load_calibration_result,
    save_calibration_result,
)


def __getattr__(name: str):
    if name in {"CalibrationArtifacts", "calibrate"}:
        from .engine import CalibrationArtifacts, calibrate

        return {
            "CalibrationArtifacts": CalibrationArtifacts,
            "calibrate": calibrate,
        }[name]
    raise AttributeError(name)

__all__ = [
    "CalibrationArtifacts",
    "CalibrationResult",
    "CalibrationResultError",
    "CameraCalibrationResult",
    "LineSensorCalibrationResult",
    "calibrate",
    "load_calibration_result",
    "save_calibration_result",
]
