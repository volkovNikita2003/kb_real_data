"""Strict manifest of artifacts produced by one restoration run."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from errors import ExperimentStructureError, RestorationError
from experiment import validate_safe_name
from schema_validation import SchemaValidator
from yaml_support import YamlError, dump_yaml, load_yaml


RESULT_SCHEMA_VERSION = 1
_DETECTOR_ORDER = ("camera", "line_sensor")
_DETECTORS = frozenset(_DETECTOR_ORDER)


class RestoreResultError(RestorationError):
    """A restoration result is incomplete, malformed, or cannot be stored."""


_VALIDATOR = SchemaValidator(RestoreResultError)


@dataclass(frozen=True)
class RestoreSignalFiles:
    camera: str
    line_sensor: str | None = None
    combined: str | None = None


@dataclass(frozen=True)
class RestoreSolutionFiles:
    restoration: str
    gcv_curve: str
    alpha: str


@dataclass(frozen=True)
class RestoreResult:
    schema_version: int
    restore_profile: str
    measurement: str
    detectors: tuple[str, ...]
    used_parameters_file: str
    legacy_parameters_file: str
    signals: RestoreSignalFiles
    solution: RestoreSolutionFiles
    sliced_solution: RestoreSolutionFiles | None
    expected_signal_comparison: str | None
    figures: tuple[str, ...]
    artifacts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        signals: dict[str, str] = {"camera": self.signals.camera}
        if self.signals.line_sensor is not None:
            signals["line_sensor"] = self.signals.line_sensor
        if self.signals.combined is not None:
            signals["combined"] = self.signals.combined

        def solution(value: RestoreSolutionFiles) -> dict[str, str]:
            return {
                "restoration": value.restoration,
                "gcv_curve": value.gcv_curve,
                "alpha": value.alpha,
            }

        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "restore_profile": self.restore_profile,
            "measurement": self.measurement,
            "detectors": list(self.detectors),
            "used_parameters_file": self.used_parameters_file,
            "legacy_parameters_file": self.legacy_parameters_file,
            "signals": signals,
            "solution": solution(self.solution),
        }
        if self.sliced_solution is not None:
            result["sliced_solution"] = solution(self.sliced_solution)
        if self.expected_signal_comparison is not None:
            result["expected_signal_comparison"] = (
                self.expected_signal_comparison
            )
        result["figures"] = list(self.figures)
        result["artifacts"] = list(self.artifacts)
        return result


def _detectors(values: Iterable[str]) -> tuple[str, ...]:
    items = tuple(values)
    if not items:
        raise RestoreResultError(
            "detectors: должен быть указан хотя бы один детектор"
        )
    if len(items) != len(set(items)):
        raise RestoreResultError("detectors: имена не должны повторяться")
    unknown = set(items) - _DETECTORS
    if unknown:
        raise RestoreResultError(
            "detectors: неизвестные детекторы: "
            + ", ".join(sorted(unknown))
        )
    if "camera" not in items:
        raise RestoreResultError(
            "detectors: legacy-восстановление требует камеру"
        )
    return tuple(name for name in _DETECTOR_ORDER if name in items)


def _relative_path(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise RestoreResultError(f"{location}: ожидался непустой путь")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise RestoreResultError(
            f"{location}: ожидался безопасный относительный путь"
        )
    return value


def _safe_name(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise RestoreResultError(
            f"{location}: ожидалась строка"
        )
    try:
        return validate_safe_name(value, kind=location)
    except ExperimentStructureError as error:
        raise RestoreResultError(str(error)) from error


def _require_file(root: Path, relative: str, *, location: str) -> str:
    safe = _relative_path(relative, location)
    path = root / safe
    if not path.is_file():
        raise RestoreResultError(f"{location}: файл не найден: {path}")
    return safe


def _one_matching_file(
    root: Path,
    pattern: str,
    *,
    location: str,
    required: bool,
) -> str | None:
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    expected_count = 1 if required else 0
    if len(paths) != expected_count:
        raise RestoreResultError(
            f"{location}: ожидалось файлов — {expected_count}, "
            f"найдено — {len(paths)}"
        )
    return paths[0].name if paths else None


def _solution_files(
    root: Path,
    *,
    prefix: str,
    location: str,
) -> RestoreSolutionFiles:
    return RestoreSolutionFiles(
        restoration=_require_file(
            root,
            f"{prefix}reference_restoration.txt",
            location=f"{location}.restoration",
        ),
        gcv_curve=_require_file(
            root,
            f"{prefix}reference_gcv_curve.txt",
            location=f"{location}.gcv_curve",
        ),
        alpha=_require_file(
            root,
            f"{prefix}reference_alpha.txt",
            location=f"{location}.alpha",
        ),
    )


def collect_restore_result(
    directory: str | Path,
    *,
    restore_profile: str,
    measurement: str,
    detector_names: Iterable[str],
    class_slice_prefix: str | None,
    expected_signal_comparison: bool,
) -> RestoreResult:
    """Validate generated legacy artifacts and build their strict manifest."""
    root = Path(directory)
    if not root.is_dir():
        raise RestoreResultError(
            f"Директория результата восстановления не найдена: {root}"
        )
    profile_name = _safe_name(restore_profile, "restore_profile")
    measurement_name = _safe_name(measurement, "measurement")
    detectors = _detectors(detector_names)
    used_parameters = _require_file(
        root, "used-parameters.yaml", location="used_parameters_file"
    )
    legacy_parameters = _require_file(
        root, "params.txt", location="legacy_parameters_file"
    )
    camera_signal = _require_file(
        root, "reference_camera_signal.txt", location="signals.camera"
    )
    uses_line = "line_sensor" in detectors
    line_signal = _one_matching_file(
        root,
        "reference_line_signal.txt",
        location="signals.line_sensor",
        required=uses_line,
    )
    combined_signal = _one_matching_file(
        root,
        "reference_combined_signal.txt",
        location="signals.combined",
        required=uses_line,
    )
    signals = RestoreSignalFiles(
        camera=camera_signal,
        line_sensor=line_signal,
        combined=combined_signal,
    )
    solution = _solution_files(root, prefix="", location="solution")
    sliced_solution = None
    if class_slice_prefix is not None:
        if not class_slice_prefix or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in class_slice_prefix
        ):
            raise RestoreResultError(
                "class_slice_prefix: недопустимый префикс имени файла"
            )
        sliced_solution = _solution_files(
            root,
            prefix=class_slice_prefix,
            location="sliced_solution",
        )
    expected_sliced_files = set()
    if sliced_solution is not None:
        expected_sliced_files = {
            sliced_solution.restoration,
            sliced_solution.gcv_curve,
            sliced_solution.alpha,
        }
    actual_sliced_files = {
        path.name
        for path in root.glob("cutted_*reference_*.txt")
        if path.is_file()
    }
    if actual_sliced_files != expected_sliced_files:
        raise RestoreResultError(
            "sliced_solution: файлы срезанного восстановления не соответствуют "
            "параметрам запуска"
        )

    comparison = _one_matching_file(
        root,
        "compare-real-darl-signals-all-*.txt",
        location="expected_signal_comparison",
        required=expected_signal_comparison,
    )
    figures = tuple(sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.png") if path.is_file()
    ))
    if not figures:
        raise RestoreResultError(
            "figures: не найдено ни одного диагностического графика"
        )
    artifacts = tuple(sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "result.yaml"
    ))
    return RestoreResult(
        schema_version=RESULT_SCHEMA_VERSION,
        restore_profile=profile_name,
        measurement=measurement_name,
        detectors=detectors,
        used_parameters_file=used_parameters,
        legacy_parameters_file=legacy_parameters,
        signals=signals,
        solution=solution,
        sliced_solution=sliced_solution,
        expected_signal_comparison=comparison,
        figures=figures,
        artifacts=artifacts,
    )


def save_restore_result(result: RestoreResult, path: str | Path) -> None:
    if not isinstance(result, RestoreResult):
        raise RestoreResultError("result: ожидался объект RestoreResult")
    try:
        dump_yaml(path, result.to_dict())
    except YamlError as error:
        raise RestoreResultError(str(error)) from error


def load_restore_result(path: str | Path) -> RestoreResult:
    """Load and strictly validate one restoration result manifest."""
    try:
        value = load_yaml(path)
    except YamlError as error:
        raise RestoreResultError(str(error)) from error
    fields = _VALIDATOR.fields(
        value,
        "result.yaml",
        required={
            "schema_version", "restore_profile", "measurement", "detectors",
            "used_parameters_file", "legacy_parameters_file", "signals",
            "solution", "figures", "artifacts",
        },
        optional={"sliced_solution", "expected_signal_comparison"},
    )
    _VALIDATOR.version(
        fields["schema_version"], "schema_version", supported=RESULT_SCHEMA_VERSION
    )
    restore_profile = _safe_name(fields["restore_profile"], "restore_profile")
    measurement = _safe_name(fields["measurement"], "measurement")

    def strings(value: Any, location: str) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise RestoreResultError(f"{location}: ожидался список")
        result = tuple(
            _relative_path(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        )
        if len(result) != len(set(result)):
            raise RestoreResultError(
                f"{location}: пути не должны повторяться"
            )
        return result

    raw_detectors = fields["detectors"]
    if not isinstance(raw_detectors, list):
        raise RestoreResultError("detectors: ожидался список")
    detectors = _detectors(raw_detectors)

    signal_fields = _VALIDATOR.fields(
        fields["signals"],
        "signals",
        required={"camera"},
        optional={"line_sensor", "combined"},
    )
    uses_line = "line_sensor" in detectors
    has_line_fields = {
        "line_sensor", "combined"
    }.issubset(signal_fields)
    has_partial_line_fields = bool(
        {"line_sensor", "combined"}.intersection(signal_fields)
    )
    if uses_line != has_line_fields or has_partial_line_fields != has_line_fields:
        raise RestoreResultError(
            "signals: line_sensor и combined должны точно соответствовать "
            "включённой линейке"
        )
    signals = RestoreSignalFiles(
        camera=_relative_path(signal_fields["camera"], "signals.camera"),
        line_sensor=(
            _relative_path(signal_fields["line_sensor"], "signals.line_sensor")
            if uses_line else None
        ),
        combined=(
            _relative_path(signal_fields["combined"], "signals.combined")
            if uses_line else None
        ),
    )

    def parse_solution(value: Any, location: str) -> RestoreSolutionFiles:
        data = _VALIDATOR.fields(
            value,
            location,
            required={"restoration", "gcv_curve", "alpha"},
        )
        return RestoreSolutionFiles(
            _relative_path(data["restoration"], f"{location}.restoration"),
            _relative_path(data["gcv_curve"], f"{location}.gcv_curve"),
            _relative_path(data["alpha"], f"{location}.alpha"),
        )

    solution = parse_solution(fields["solution"], "solution")
    sliced = (
        parse_solution(fields["sliced_solution"], "sliced_solution")
        if "sliced_solution" in fields else None
    )
    comparison = (
        _relative_path(
            fields["expected_signal_comparison"],
            "expected_signal_comparison",
        )
        if "expected_signal_comparison" in fields else None
    )
    figures = strings(fields["figures"], "figures")
    if not figures or any(not path.endswith(".png") for path in figures):
        raise RestoreResultError(
            "figures: ожидался непустой список PNG-файлов"
        )
    artifacts = strings(fields["artifacts"], "artifacts")
    if not artifacts:
        raise RestoreResultError("artifacts: список не должен быть пустым")
    used_parameters = _relative_path(
        fields["used_parameters_file"], "used_parameters_file"
    )
    legacy_parameters = _relative_path(
        fields["legacy_parameters_file"], "legacy_parameters_file"
    )
    essential_paths = {
        used_parameters,
        legacy_parameters,
        signals.camera,
        solution.restoration,
        solution.gcv_curve,
        solution.alpha,
        *figures,
    }
    if signals.line_sensor is not None:
        essential_paths.add(signals.line_sensor)
    if signals.combined is not None:
        essential_paths.add(signals.combined)
    if sliced is not None:
        essential_paths.update((
            sliced.restoration, sliced.gcv_curve, sliced.alpha,
        ))
    if comparison is not None:
        essential_paths.add(comparison)
    missing_artifacts = essential_paths - set(artifacts)
    if missing_artifacts:
        raise RestoreResultError(
            "artifacts: отсутствуют обязательные пути: "
            + ", ".join(sorted(missing_artifacts))
        )

    return RestoreResult(
        schema_version=fields["schema_version"],
        restore_profile=restore_profile,
        measurement=measurement,
        detectors=detectors,
        used_parameters_file=used_parameters,
        legacy_parameters_file=legacy_parameters,
        signals=signals,
        solution=solution,
        sliced_solution=sliced,
        expected_signal_comparison=comparison,
        figures=figures,
        artifacts=artifacts,
    )
