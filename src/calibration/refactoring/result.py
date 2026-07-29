"""Strict, versioned representation of ``output/calibration/result.yaml``."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from schema_validation import SchemaValidator
from yaml_support import YamlError, dump_yaml, load_yaml


RESULT_SCHEMA_VERSION = 1


class CalibrationResultError(ValueError):
    """The calibration result is malformed or cannot be read/written."""


_VALIDATOR = SchemaValidator(CalibrationResultError)
_fields = _VALIDATOR.fields
_number = _VALIDATOR.number


@dataclass(frozen=True)
class CameraCalibrationResult:
    matrix_diagonal_mm: float
    x_shift_m: float
    y_shift_m: float
    pixel_width_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix_diagonal_mm", _number(self.matrix_diagonal_mm, "camera.matrix_diagonal_mm", positive=True))
        object.__setattr__(self, "x_shift_m", _number(self.x_shift_m, "camera.x_shift_m"))
        object.__setattr__(self, "y_shift_m", _number(self.y_shift_m, "camera.y_shift_m"))
        object.__setattr__(self, "pixel_width_m", _number(self.pixel_width_m, "camera.pixel_width_m", positive=True))


@dataclass(frozen=True)
class LineSensorCalibrationResult:
    start_angle_deg: float
    end_angle_deg: float
    pixel_width_m: float
    pixel_height_m: float
    to_camera_coefficient: float
    shift_m: float
    peak_pixel: float

    def __post_init__(self) -> None:
        for name in ("start_angle_deg", "end_angle_deg", "shift_m"):
            object.__setattr__(self, name, _number(getattr(self, name), f"line_sensor.{name}"))
        for name in ("pixel_width_m", "pixel_height_m", "to_camera_coefficient"):
            object.__setattr__(self, name, _number(getattr(self, name), f"line_sensor.{name}", positive=True))
        object.__setattr__(self, "peak_pixel", _number(self.peak_pixel, "line_sensor.peak_pixel", non_negative=True))
        if self.start_angle_deg >= self.end_angle_deg:
            raise CalibrationResultError("line_sensor.start_angle_deg must be less than line_sensor.end_angle_deg")


@dataclass(frozen=True)
class CalibrationResult(Mapping[str, Any]):
    """Complete calibration result; also acts as a plain mapping for DARL."""

    schema_version: int
    camera: CameraCalibrationResult
    line_sensor: LineSensorCalibrationResult | None = None

    def __post_init__(self) -> None:
        _VALIDATOR.version(
            self.schema_version,
            "schema_version",
            supported=RESULT_SCHEMA_VERSION,
        )
        if not isinstance(self.camera, CameraCalibrationResult):
            raise CalibrationResultError("camera: expected CameraCalibrationResult")
        if self.line_sensor is not None and not isinstance(self.line_sensor, LineSensorCalibrationResult):
            raise CalibrationResultError("line_sensor: expected LineSensorCalibrationResult or None")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "camera": asdict(self.camera),
        }
        if self.line_sensor is not None:
            result["line_sensor"] = asdict(self.line_sensor)
        return result

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return 2 + (self.line_sensor is not None)


_CAMERA_FIELDS = frozenset({
    "matrix_diagonal_mm", "x_shift_m", "y_shift_m", "pixel_width_m",
})
_LINE_FIELDS = frozenset({
    "start_angle_deg", "end_angle_deg", "pixel_width_m", "pixel_height_m",
    "to_camera_coefficient", "shift_m", "peak_pixel",
})


def calibration_result_from_dict(value: Any) -> CalibrationResult:
    root = _fields(
        value,
        "result.yaml",
        required=frozenset({"schema_version", "camera"}),
        optional=frozenset({"line_sensor"}),
    )
    camera = _fields(root["camera"], "result.yaml.camera", required=_CAMERA_FIELDS)
    line = None
    if "line_sensor" in root:
        line_data = _fields(root["line_sensor"], "result.yaml.line_sensor", required=_LINE_FIELDS)
        line = LineSensorCalibrationResult(**line_data)
    return CalibrationResult(
        schema_version=root["schema_version"],
        camera=CameraCalibrationResult(**camera),
        line_sensor=line,
    )


def load_calibration_result(path: str | Path) -> CalibrationResult:
    source = Path(path)
    try:
        value = load_yaml(source)
    except YamlError as error:
        raise CalibrationResultError(str(error)) from error
    return calibration_result_from_dict(value)


def save_calibration_result(result: CalibrationResult, path: str | Path) -> None:
    if not isinstance(result, CalibrationResult):
        raise CalibrationResultError("result: expected CalibrationResult")
    target = Path(path)
    try:
        dump_yaml(target, result.to_dict())
    except YamlError as error:
        raise CalibrationResultError(str(error)) from error
