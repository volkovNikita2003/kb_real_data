"""Application-level orchestration shared by all processing CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from calibration import CalibrationResult, load_calibration_result
import calc_darl
from darl import DarlResult, load_darl_result
from errors import ExperimentStructureError, OutputError, PipelineError
from experiment import Experiment, Measurement, RestoreProfile
from parameters import (
    CalibrationStageParameters,
    DarlStageParameters,
    ExperimentParameters,
    GeneralParameters,
    RestoreParameters,
    load_general_parameters,
    load_restore_parameters,
)
import preprocessing
import restore as restore_command
from validation import (
    ValidationReport,
    validate_calibration_inputs,
    validate_darl_inputs,
    validate_restore_inputs,
)


ReportCallback = Callable[[str, ValidationReport], None]


@dataclass(frozen=True)
class StageResult:
    name: str
    output_paths: tuple[Path, ...]
    validation_report: ValidationReport


def _accept_report(
    name: str,
    report: ValidationReport,
    *,
    warnings_as_errors: bool,
    report_callback: ReportCallback | None,
) -> None:
    if report_callback is not None:
        report_callback(name, report)
    if report.errors:
        raise PipelineError(
            f"Этап {name!r} не запущен: проверка обнаружила ошибки"
        )
    if warnings_as_errors and report.warnings:
        raise PipelineError(
            f"Этап {name!r} не запущен: предупреждения считаются ошибками"
        )


def select_names(
    available: Sequence[str],
    requested: Iterable[str] | None,
    *,
    kind: str,
) -> tuple[str, ...]:
    if requested is None:
        result = tuple(available)
    else:
        selected: list[str] = []
        available_set = set(available)
        for name in requested:
            if name not in available_set:
                raise ExperimentStructureError(f"Не найден {kind} {name!r}")
            if name not in selected:
                selected.append(name)
        result = tuple(selected)
    if not result:
        descriptions = {
            "измерение": "ни одного измерения",
            "профиль восстановления": "ни одного профиля восстановления",
        }
        raise ExperimentStructureError(
            "В эксперименте не найдено " + descriptions.get(kind, kind)
        )
    return result


def selected_restore_inputs(
    experiment: Experiment,
    *,
    measurement_names: Iterable[str] | None = None,
    profile_names: Iterable[str] | None = None,
) -> tuple[tuple[Measurement, ...], tuple[RestoreProfile, ...]]:
    measurements = tuple(
        experiment.measurement(name)
        for name in select_names(
            experiment.measurement_names(), measurement_names, kind="измерение"
        )
    )
    profiles = tuple(
        experiment.restore_profile(name)
        for name in select_names(
            experiment.restore_profile_names(), profile_names,
            kind="профиль восстановления",
        )
    )
    return measurements, profiles


def run_calibration_stage(
    experiment: Experiment,
    *,
    force: bool = False,
    warnings_as_errors: bool = False,
    report_callback: ReportCallback | None = None,
) -> StageResult:
    parameters = CalibrationStageParameters.load(experiment)
    report = validate_calibration_inputs(experiment, parameters)
    _accept_report(
        "calibration", report,
        warnings_as_errors=warnings_as_errors,
        report_callback=report_callback,
    )
    preprocessing.run(experiment, parameters, force=force)
    return StageResult(
        "calibration", (experiment.calibration_output_dir,), report
    )


def run_darl_stage(
    experiment: Experiment,
    *,
    force: bool = False,
    warnings_as_errors: bool = False,
    report_callback: ReportCallback | None = None,
    code_git_dir: str | Path | None = None,
) -> StageResult:
    parameters = DarlStageParameters.load(experiment)
    code_git = Path(
        code_git_dir or calc_darl.default_code_git_dir()
    ).expanduser().resolve()
    report = validate_darl_inputs(
        experiment, parameters, code_git_dir=code_git
    )
    _accept_report(
        "darl", report,
        warnings_as_errors=warnings_as_errors,
        report_callback=report_callback,
    )
    calc_darl.run(
        experiment, parameters, force=force, code_git_dir=code_git
    )
    return StageResult("darl", (experiment.darl_output_dir,), report)


def _restore_context(
    experiment: Experiment,
    *,
    measurement_names: Iterable[str] | None,
    profile_names: Iterable[str] | None,
) -> tuple[
    GeneralParameters,
    CalibrationResult,
    DarlResult,
    tuple[Measurement, ...],
    tuple[RestoreProfile, ...],
    dict[str, RestoreParameters],
]:
    measurements, profiles = selected_restore_inputs(
        experiment,
        measurement_names=measurement_names,
        profile_names=profile_names,
    )
    general = load_general_parameters(experiment.general_parameters_file)
    calibration = load_calibration_result(experiment.calibration_result_file)
    darl = load_darl_result(experiment.darl_result_file)
    restore_by_profile = {
        profile.name: load_restore_parameters(profile.path)
        for profile in profiles
    }
    return (
        general, calibration, darl, measurements, profiles,
        restore_by_profile,
    )


def run_restore_stage(
    experiment: Experiment,
    *,
    measurement_names: Iterable[str] | None = None,
    profile_names: Iterable[str] | None = None,
    force: bool = False,
    warnings_as_errors: bool = False,
    report_callback: ReportCallback | None = None,
) -> StageResult:
    (
        general, calibration, darl, measurements, profiles,
        restore_by_profile,
    ) = _restore_context(
        experiment,
        measurement_names=measurement_names,
        profile_names=profile_names,
    )
    pair_reports = tuple(
        validate_restore_inputs(
            experiment,
            measurement=measurement,
            profile=profile,
            general=general,
            restore=restore_by_profile[profile.name],
            calibration=calibration,
            darl=darl,
        )
        for profile in profiles
        for measurement in measurements
    )
    report = ValidationReport(tuple(
        issue for pair_report in pair_reports for issue in pair_report.issues
    ))
    _accept_report(
        "restore", report,
        warnings_as_errors=warnings_as_errors,
        report_callback=report_callback,
    )

    output_paths = tuple(
        restore_command.run_pair(
            experiment,
            measurement=measurement,
            profile=profile,
            general=general,
            restore=restore_by_profile[profile.name],
            calibration=calibration,
            darl=darl,
            force=force,
        )
        for profile in profiles
        for measurement in measurements
    )
    return StageResult("restore", output_paths, report)


def _check_target(target: Path, *, force: bool) -> None:
    if not target.exists() and not target.is_symlink():
        return
    if not force:
        raise OutputError(
            f"Каталог результатов уже существует: {target}. "
            "Для полного повторного расчёта используйте --force."
        )
    if target.is_symlink() or not target.is_dir():
        raise OutputError(
            f"Путь результата должен быть обычной директорией: {target}"
        )


def preflight_all(
    experiment: Experiment,
    *,
    measurement_names: Iterable[str] | None = None,
    profile_names: Iterable[str] | None = None,
    force: bool = False,
) -> None:
    """Reject all predictable full-run conflicts before calibration starts."""
    measurements, profiles = selected_restore_inputs(
        experiment,
        measurement_names=measurement_names,
        profile_names=profile_names,
    )
    parameters = ExperimentParameters.load(experiment)
    for measurement in measurements:
        parameters_file = experiment.measurement_parameters_file(
            measurement.name
        )
        if parameters_file is not None:
            parameters.load_measurement(parameters_file)
    for profile in profiles:
        parameters.load_restore_profile(profile)
    targets = (
        experiment.calibration_output_dir,
        experiment.darl_output_dir,
        *(
            experiment.restore_output_dir(profile.name, measurement.name)
            for profile in profiles
            for measurement in measurements
        ),
    )
    for target in targets:
        _check_target(target, force=force)


def run_all_stages(
    experiment: Experiment,
    *,
    measurement_names: Iterable[str] | None = None,
    profile_names: Iterable[str] | None = None,
    force: bool = False,
    warnings_as_errors: bool = False,
    report_callback: ReportCallback | None = None,
    code_git_dir: str | Path | None = None,
) -> tuple[StageResult, StageResult, StageResult]:
    preflight_all(
        experiment,
        measurement_names=measurement_names,
        profile_names=profile_names,
        force=force,
    )
    calibration = run_calibration_stage(
        experiment, force=force, warnings_as_errors=warnings_as_errors,
        report_callback=report_callback,
    )
    darl = run_darl_stage(
        experiment, force=force, warnings_as_errors=warnings_as_errors,
        report_callback=report_callback, code_git_dir=code_git_dir,
    )
    restoration = run_restore_stage(
        experiment,
        measurement_names=measurement_names,
        profile_names=profile_names,
        force=force,
        warnings_as_errors=warnings_as_errors,
        report_callback=report_callback,
    )
    return calibration, darl, restoration
