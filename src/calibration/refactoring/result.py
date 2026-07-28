"""Strict, versioned representation of ``output/calibration/result.yaml``."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError


RESULT_SCHEMA_VERSION = 1


class CalibrationResultError(ValueError):
    """The calibration result is malformed or cannot be read/written."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader which also rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise CalibrationResultError(f"{path}: expected a YAML mapping with string keys")
    return value


def _fields(
    value: Any,
    path: str,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    data = _mapping(value, path)
    unknown = sorted(data.keys() - required - optional)
    missing = sorted(required - data.keys())
    if unknown:
        raise CalibrationResultError(f"{path}: unknown fields: {', '.join(unknown)}")
    if missing:
        raise CalibrationResultError(f"{path}: missing required fields: {', '.join(missing)}")
    return data


def _number(value: Any, path: str, *, positive: bool = False, non_negative: bool = False) -> float:
    if type(value) not in (int, float):
        raise CalibrationResultError(f"{path}: expected a number")
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationResultError(f"{path}: number must be finite")
    if positive and result <= 0:
        raise CalibrationResultError(f"{path}: number must be positive")
    if non_negative and result < 0:
        raise CalibrationResultError(f"{path}: number must be non-negative")
    return result


def _positive_integer(value: Any, path: str) -> int:
    if type(value) is not int:
        raise CalibrationResultError(f"{path}: expected an integer")
    if value <= 0:
        raise CalibrationResultError(f"{path}: integer must be positive")
    return value


@dataclass(frozen=True)
class CameraCalibrationResult:
    matrix_diagonal_mm: float
    width_px: int
    height_px: int
    x_shift_m: float
    y_shift_m: float
    pixel_width_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix_diagonal_mm", _number(self.matrix_diagonal_mm, "camera.matrix_diagonal_mm", positive=True))
        object.__setattr__(self, "width_px", _positive_integer(self.width_px, "camera.width_px"))
        object.__setattr__(self, "height_px", _positive_integer(self.height_px, "camera.height_px"))
        object.__setattr__(self, "x_shift_m", _number(self.x_shift_m, "camera.x_shift_m"))
        object.__setattr__(self, "y_shift_m", _number(self.y_shift_m, "camera.y_shift_m"))
        object.__setattr__(self, "pixel_width_m", _number(self.pixel_width_m, "camera.pixel_width_m", positive=True))


@dataclass(frozen=True)
class LineSensorCalibrationResult:
    start_angle_deg: float
    end_angle_deg: float
    logarithmic_radius_percent: float
    pixel_width_m: float
    pixel_height_m: float
    to_camera_coefficient: float
    shift_m: float
    peak_pixel: float

    def __post_init__(self) -> None:
        for name in ("start_angle_deg", "end_angle_deg", "shift_m"):
            object.__setattr__(self, name, _number(getattr(self, name), f"line_sensor.{name}"))
        for name in (
            "logarithmic_radius_percent",
            "pixel_width_m",
            "pixel_height_m",
            "to_camera_coefficient",
        ):
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
        if type(self.schema_version) is not int:
            raise CalibrationResultError("schema_version: expected an integer")
        if self.schema_version != RESULT_SCHEMA_VERSION:
            raise CalibrationResultError(
                f"schema_version: unsupported version {self.schema_version}; "
                f"supported version is {RESULT_SCHEMA_VERSION}"
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
    "matrix_diagonal_mm", "width_px", "height_px", "x_shift_m", "y_shift_m", "pixel_width_m",
})
_LINE_FIELDS = frozenset({
    "start_angle_deg", "end_angle_deg", "logarithmic_radius_percent", "pixel_width_m",
    "pixel_height_m", "to_camera_coefficient", "shift_m", "peak_pixel",
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
    if not source.is_file():
        raise CalibrationResultError(f"calibration result file not found: {source}")
    try:
        with source.open("r", encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as error:
        raise CalibrationResultError(f"cannot read calibration result {source}: {error}") from error
    if value is None:
        raise CalibrationResultError(f"calibration result file is empty: {source}")
    return calibration_result_from_dict(value)


def save_calibration_result(result: CalibrationResult, path: str | Path) -> None:
    if not isinstance(result, CalibrationResult):
        raise CalibrationResultError("result: expected CalibrationResult")
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(result.to_dict(), stream, allow_unicode=True, sort_keys=False)
    except OSError as error:
        raise CalibrationResultError(f"cannot write calibration result {target}: {error}") from error
