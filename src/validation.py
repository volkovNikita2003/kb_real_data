"""Read-only validation of experiment inputs and detector data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Literal, Sequence

from errors import ExperimentStructureError, ParametersError
from experiment import Experiment, Measurement, RestoreProfile, validate_safe_name
from parameters import ExperimentParameters, RestoreParameters, load_measurement_parameters


EXPOSURE_PATTERN = re.compile(r"^[1-9][0-9]*$", re.ASCII)
ALLOWED_INPUT_PARAMETER_ENTRIES = frozenset(
    {"general.yaml", "calibration.yaml", "darl.yaml", "measurements", "restore_profiles"}
)
ALLOWED_CALIBRATION_DIRECTORIES = frozenset({"cam", "lin"})
ALLOWED_MEASUREMENT_DIRECTORIES = frozenset({"cam", "cam_back", "lin", "lin_back"})
Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    path: Path | None = None
    measurement: str | None = None
    restore_profile: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        return not self.errors


class _ReportBuilder:
    def __init__(self) -> None:
        self._issues: list[ValidationIssue] = []
        self._seen: set[tuple[object, ...]] = set()

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        *,
        path: Path | None = None,
        measurement: str | None = None,
        restore_profile: str | None = None,
    ) -> None:
        issue = ValidationIssue(
            severity,
            code,
            message,
            path,
            measurement,
            restore_profile,
        )
        key = (
            issue.severity,
            issue.code,
            issue.path,
            issue.measurement,
            issue.restore_profile,
        )
        if key not in self._seen:
            self._seen.add(key)
            self._issues.append(issue)

    def build(self) -> ValidationReport:
        return ValidationReport(tuple(self._issues))


def parse_exposure_filename(path: str | Path, expected_suffix: str) -> int:
    """Return a positive integer exposure encoded in a detector filename."""
    source = Path(path)
    if source.suffix != expected_suffix:
        raise ValueError(
            f"ожидался файл <экспозиция>{expected_suffix}, получено {source.name!r}"
        )
    stem = source.name[: -len(expected_suffix)]
    if not EXPOSURE_PATTERN.fullmatch(stem):
        raise ValueError(
            f"экспозиция должна быть положительным целым числом без ведущих нулей: "
            f"{source.name!r}"
        )
    return int(stem)


def _select_names(
    available: Sequence[str],
    selected: Iterable[str] | None,
    *,
    kind: str,
) -> tuple[str, ...]:
    if selected is None:
        return tuple(available)
    result: list[str] = []
    available_set = set(available)
    for name in selected:
        validate_safe_name(name, kind=kind)
        if name not in available_set:
            raise ExperimentStructureError(f"Не найден {kind} {name!r}")
        if name not in result:
            result.append(name)
    return tuple(result)


def _check_allowed_entries(
    directory: Path,
    allowed: frozenset[str],
    report: _ReportBuilder,
    *,
    code: str,
) -> None:
    if not directory.is_dir():
        return
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if entry.name not in allowed:
            report.add(
                "error",
                code,
                f"Обнаружен недопустимый элемент: {entry}",
                path=entry,
            )


def _scan_exposure_directory(
    directory: Path,
    suffix: str,
    report: _ReportBuilder,
    *,
    measurement: str | None = None,
) -> dict[int, Path] | None:
    if not directory.exists() or not directory.is_dir():
        return None
    exposures: dict[int, Path] = {}
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if not entry.is_file():
            report.add(
                "error",
                "invalid_detector_entry",
                f"В каталоге детектора разрешены только файлы: {entry}",
                path=entry,
                measurement=measurement,
            )
            continue
        try:
            exposure = parse_exposure_filename(entry, suffix)
        except ValueError as error:
            report.add(
                "error",
                "invalid_exposure_name",
                str(error),
                path=entry,
                measurement=measurement,
            )
            continue
        exposures[exposure] = entry
    return exposures


def _require_directory(
    directory: Path,
    report: _ReportBuilder,
    *,
    code: str,
    message: str,
    measurement: str | None = None,
    restore_profile: str | None = None,
) -> bool:
    if not directory.exists() or not directory.is_dir():
        report.add(
            "error",
            code,
            message,
            path=directory,
            measurement=measurement,
            restore_profile=restore_profile,
        )
        return False
    return True


def _validate_parameter_layout(
    experiment: Experiment,
    report: _ReportBuilder,
) -> None:
    _check_allowed_entries(
        experiment.input_parameters_dir,
        ALLOWED_INPUT_PARAMETER_ENTRIES,
        report,
        code="unknown_parameter_entry",
    )
    directory = experiment.measurement_parameters_dir
    if not directory.exists():
        return
    if not directory.is_dir():
        report.add(
            "error",
            "invalid_measurement_parameters_directory",
            f"Путь параметров измерений не является директорией: {directory}",
            path=directory,
        )
        return
    measurement_names = set(experiment.measurement_names())
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if not entry.is_file() or entry.suffix != ".yaml":
            report.add(
                "error",
                "invalid_measurement_parameter_file",
                f"В measurements/ разрешены только YAML-файлы: {entry.name}",
                path=entry,
            )
            continue
        try:
            validate_safe_name(entry.stem, kind="параметров измерения")
        except ExperimentStructureError as error:
            report.add("error", "unsafe_measurement_parameter_name", str(error), path=entry)
            continue
        if entry.stem not in measurement_names:
            report.add(
                "error",
                "unknown_measurement_parameters",
                f"Параметры заданы для несуществующего измерения {entry.stem!r}",
                path=entry,
            )
            continue
        try:
            load_measurement_parameters(entry)
        except ParametersError as error:
            report.add(
                "error",
                "invalid_measurement_parameters",
                str(error),
                path=entry,
                measurement=entry.stem,
            )


def _validate_calibration(
    experiment: Experiment,
    parameters: ExperimentParameters,
    report: _ReportBuilder,
) -> None:
    _check_allowed_entries(
        experiment.calibration_dir,
        ALLOWED_CALIBRATION_DIRECTORIES,
        report,
        code="unknown_calibration_entry",
    )
    detectors = parameters.general.detectors.names()
    specs = (
        ("camera", experiment.calibration_camera_dir, ".bmp"),
        ("line_sensor", experiment.calibration_line_dir, ".txt"),
    )
    for name, directory, suffix in specs:
        active = name in detectors
        if active and not _require_directory(
            directory,
            report,
            code="missing_calibration_detector_data",
            message=f"Не найдены калибровочные данные детектора {name}: {directory}",
        ):
            continue
        if not active:
            if directory.exists():
                report.add(
                    "warning",
                    "unused_calibration_detector_data",
                    f"Детектор {name} не используется, но его калибровочные данные существуют",
                    path=directory,
                )
            continue
        exposures = _scan_exposure_directory(directory, suffix, report)
        if exposures is None:
            continue
        if name == "camera" and not exposures:
            report.add(
                "error",
                "empty_camera_calibration",
                "Для калибровки камеры требуется хотя бы одно изображение",
                path=directory,
            )
        if name == "line_sensor" and len(exposures) != 1:
            report.add(
                "error",
                "invalid_line_calibration_file_count",
                "Для калибровки линейки требуется ровно один TXT-файл",
                path=directory,
            )


@dataclass(frozen=True)
class _MeasurementData:
    measurement: Measurement
    camera: dict[int, Path] | None
    camera_background: dict[int, Path] | None
    line: dict[int, Path] | None
    line_background: dict[int, Path] | None


def _validate_measurement_base(
    measurement: Measurement,
    parameters: ExperimentParameters,
    report: _ReportBuilder,
) -> _MeasurementData:
    for entry in sorted(measurement.path.iterdir(), key=lambda item: item.name):
        if entry.name not in ALLOWED_MEASUREMENT_DIRECTORIES:
            report.add(
                "error",
                "unknown_measurement_entry",
                f"В измерении обнаружен недопустимый элемент: {entry.name}",
                path=entry,
                measurement=measurement.name,
            )
        elif not entry.is_dir():
            report.add(
                "error",
                "detector_path_not_directory",
                f"Путь данных детектора должен быть директорией: {entry}",
                path=entry,
                measurement=measurement.name,
            )

    detectors = parameters.general.detectors.names()
    signal_specs = (
        (
            "camera",
            (measurement.camera_dir, measurement.camera_background_dir),
        ),
        (
            "line_sensor",
            (measurement.line_dir, measurement.line_background_dir),
        ),
    )
    for name, directories in signal_specs:
        signal_directory = directories[0]
        if name in detectors:
            _require_directory(
                signal_directory,
                report,
                code="missing_detector_data",
                message=(
                    f"Не найдены данные используемого детектора {name}: "
                    f"{signal_directory}"
                ),
                measurement=measurement.name,
            )
        else:
            for directory in directories:
                if directory.exists():
                    report.add(
                        "warning",
                        "unused_detector_data",
                        f"Детектор {name} не используется, но его данные существуют",
                        path=directory,
                        measurement=measurement.name,
                    )

    camera = _scan_exposure_directory(
        measurement.camera_dir, ".bmp", report, measurement=measurement.name
    )
    camera_background = _scan_exposure_directory(
        measurement.camera_background_dir,
        ".bmp",
        report,
        measurement=measurement.name,
    )
    line = _scan_exposure_directory(
        measurement.line_dir, ".txt", report, measurement=measurement.name
    )
    line_background = _scan_exposure_directory(
        measurement.line_background_dir,
        ".txt",
        report,
        measurement=measurement.name,
    )

    if "camera" in detectors and camera is not None and not camera:
        report.add(
            "error",
            "empty_camera_signal",
            "Каталог сигнала камеры не содержит корректных экспозиций",
            path=measurement.camera_dir,
            measurement=measurement.name,
        )
    if "line_sensor" in detectors and line is not None and len(line) != 1:
        report.add(
            "error",
            "invalid_line_signal_file_count",
            "Каталог сигнала линейки должен содержать ровно один TXT-файл",
            path=measurement.line_dir,
            measurement=measurement.name,
        )
    return _MeasurementData(
        measurement, camera, camera_background, line, line_background
    )


def _validate_camera_pair(
    data: _MeasurementData,
    profile: RestoreProfile,
    restore: RestoreParameters,
    report: _ReportBuilder,
) -> None:
    settings = restore.detectors.camera
    if settings is None:
        return
    if data.camera is None:
        return
    background_dir = data.measurement.camera_background_dir
    if not settings.use_background:
        if background_dir.exists():
            report.add(
                "warning",
                "unused_camera_background",
                "Фон камеры существует, но выключен в профиле восстановления",
                path=background_dir,
                measurement=data.measurement.name,
                restore_profile=profile.name,
            )
        return
    if not _require_directory(
        background_dir,
        report,
        code="missing_camera_background",
        message="Не найдена требуемая директория фона камеры",
        measurement=data.measurement.name,
        restore_profile=profile.name,
    ):
        return
    background = data.camera_background or {}
    missing = sorted(data.camera.keys() - background.keys())
    extra = sorted(background.keys() - data.camera.keys())
    if missing:
        report.add(
            "error",
            "missing_camera_background_exposure",
            f"Нет фона камеры для экспозиций: {', '.join(map(str, missing))}",
            path=background_dir,
            measurement=data.measurement.name,
            restore_profile=profile.name,
        )
    if extra:
        report.add(
            "warning",
            "extra_camera_background_exposure",
            f"Лишние экспозиции фона камеры: {', '.join(map(str, extra))}",
            path=background_dir,
            measurement=data.measurement.name,
            restore_profile=profile.name,
        )


def _validate_line_pair(
    data: _MeasurementData,
    profile: RestoreProfile,
    restore: RestoreParameters,
    report: _ReportBuilder,
) -> None:
    settings = restore.detectors.line_sensor
    if settings is None:
        return
    background_dir = data.measurement.line_background_dir
    if not settings.use_background:
        if background_dir.exists():
            report.add(
                "warning",
                "unused_line_background",
                "Фон линейки существует, но выключен в профиле восстановления",
                path=background_dir,
                measurement=data.measurement.name,
                restore_profile=profile.name,
            )
        return
    if not _require_directory(
        background_dir,
        report,
        code="missing_line_background",
        message="Не найдена требуемая директория фона линейки",
        measurement=data.measurement.name,
        restore_profile=profile.name,
    ):
        return
    background = data.line_background or {}
    if len(background) != 1:
        report.add(
            "error",
            "invalid_line_background_file_count",
            "Каталог фона линейки должен содержать ровно один TXT-файл",
            path=background_dir,
            measurement=data.measurement.name,
            restore_profile=profile.name,
        )
        return
    if data.line is None or len(data.line) != 1:
        return
    signal_exposure = next(iter(data.line))
    background_exposure = next(iter(background))
    if signal_exposure != background_exposure:
        report.add(
            "error",
            "line_exposure_mismatch",
            "Экспозиции сигнала "
            f"({signal_exposure}) и фона ({background_exposure}) "
            "линейки не совпадают",
            path=background_dir,
            measurement=data.measurement.name,
            restore_profile=profile.name,
        )


def validate_experiment(
    experiment: Experiment,
    parameters: ExperimentParameters,
    *,
    measurement_names: Iterable[str] | None = None,
    restore_profile_names: Iterable[str] | None = None,
) -> ValidationReport:
    """Validate all selected experiment inputs without modifying them."""
    report = _ReportBuilder()
    _validate_parameter_layout(experiment, report)
    _validate_calibration(experiment, parameters, report)

    available_measurements = experiment.measurement_names()
    selected_measurements = _select_names(
        available_measurements,
        measurement_names,
        kind="измерения",
    )
    if not available_measurements:
        report.add(
            "error",
            "no_measurements",
            "В эксперименте не найдено ни одного измерения",
            path=experiment.data_dir,
        )
    profiles = experiment.restore_profiles()
    profile_by_name = {profile.name: profile for profile in profiles}
    selected_profiles = _select_names(
        tuple(profile_by_name),
        restore_profile_names,
        kind="профиля восстановления",
    )
    if not profiles:
        report.add(
            "error",
            "no_restore_profiles",
            "Не найдено ни одного профиля восстановления",
            path=experiment.restore_profiles_dir,
        )

    loaded_profiles: dict[str, RestoreParameters] = {}
    for name in selected_profiles:
        profile = profile_by_name[name]
        try:
            loaded_profiles[name] = parameters.load_restore_profile(profile)
        except ParametersError as error:
            report.add(
                "error",
                "invalid_restore_profile",
                str(error),
                path=profile.path,
                restore_profile=profile.name,
            )

    for name in selected_measurements:
        measurement = experiment.measurement(name)
        data = _validate_measurement_base(measurement, parameters, report)
        for profile_name, restore in loaded_profiles.items():
            profile = profile_by_name[profile_name]
            _validate_camera_pair(data, profile, restore, report)
            _validate_line_pair(data, profile, restore, report)
    return report.build()


def validate_calibration_inputs(
    experiment: Experiment,
    parameters: ExperimentParameters,
) -> ValidationReport:
    """Validate only inputs consumed by the independent calibration stage."""
    report = _ReportBuilder()
    _validate_parameter_layout(experiment, report)
    _validate_calibration(experiment, parameters, report)
    return report.build()
