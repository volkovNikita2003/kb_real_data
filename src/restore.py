"""Command-line entry point for restoring real particle distributions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Iterable, Sequence


os.environ.setdefault("MPLBACKEND", "Agg")

from calibration import (
    CalibrationResult,
    CalibrationResultError,
    load_calibration_result,
)
from darl import DarlResult, DarlResultError, load_darl_result
from errors import (
    ExperimentStructureError,
    OutputError,
    ParametersError,
    RestorationError,
)
from experiment import Experiment, Measurement, RestoreProfile
from output import prepare_output_directory
from parameters import (
    GeneralParameters,
    RestoreParameters,
    SCHEMA_VERSION,
    load_general_parameters,
    load_restore_parameters,
    to_plain_data,
    write_used_parameters,
)
from restoration.legacy.config import build_legacy_restore_config
from restoration.legacy.restore import run_legacy_restore
from restoration.result import collect_restore_result, save_restore_result
from validate import format_report
from validation import (
    ValidationReport,
    parse_exposure_filename,
    validate_restore_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Восстановить распределения частиц по реальным сигналам.",
    )
    parser.add_argument("experiment", type=Path, help="директория эксперимента")
    parser.add_argument(
        "--measurement", action="append", dest="measurements",
        help="обработать только указанное измерение; можно повторять",
    )
    parser.add_argument(
        "--profile", action="append", dest="profiles",
        help="использовать только указанный restore profile; можно повторять",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="архивировать существующие результаты выбранных пар",
    )
    warning_group = parser.add_mutually_exclusive_group()
    warning_group.add_argument(
        "--warnings-as-errors", action="store_true",
        help="не выполнять восстановление при наличии предупреждений",
    )
    warning_group.add_argument(
        "--no-warnings", action="store_true",
        help="не выводить предупреждения валидатора",
    )
    return parser


def _select(
    available: Sequence[str],
    requested: Iterable[str] | None,
    *,
    kind: str,
) -> tuple[str, ...]:
    if requested is None:
        return tuple(available)
    result: list[str] = []
    available_set = set(available)
    for name in requested:
        if name not in available_set:
            raise ExperimentStructureError(f"Не найден {kind} {name!r}")
        if name not in result:
            result.append(name)
    return tuple(result)


def _exposures(directory: Path, suffix: str) -> tuple[int, ...]:
    return tuple(sorted(
        parse_exposure_filename(path, suffix)
        for path in directory.iterdir()
        if path.is_file()
    ))


def _measurement_inputs(
    experiment: Experiment,
    measurement: Measurement,
    restore: RestoreParameters,
) -> tuple[dict[str, object], tuple[int, ...], int | None]:
    camera_exposures = _exposures(measurement.camera_dir, ".bmp")
    camera = restore.detectors.camera
    assert camera is not None
    result: dict[str, object] = {
        "camera": {
            "signal_directory": str(measurement.camera_dir.relative_to(experiment.path)),
            "background_directory": (
                str(measurement.camera_background_dir.relative_to(experiment.path))
                if camera.use_background else None
            ),
            "exposures_us": list(camera_exposures),
        }
    }
    line_exposure = None
    line = restore.detectors.line_sensor
    if line is not None:
        line_exposures = _exposures(measurement.line_dir, ".txt")
        line_exposure = line_exposures[0]
        result["line_sensor"] = {
            "signal_directory": str(measurement.line_dir.relative_to(experiment.path)),
            "background_directory": (
                str(measurement.line_background_dir.relative_to(experiment.path))
                if line.use_background else None
            ),
            "exposure_us": line_exposure,
        }
    return result, camera_exposures, line_exposure


def _effective_parameters(
    *,
    general: GeneralParameters,
    restore: RestoreParameters,
    profile: RestoreProfile,
    measurement: Measurement,
    measurement_inputs: dict[str, object],
    calibration: CalibrationResult,
    darl: DarlResult,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": {
            "measurement": measurement.name,
            "restore_profile": profile.name,
        },
        "general": to_plain_data(general),
        "restore": to_plain_data(restore),
        "measurement_inputs": to_plain_data(measurement_inputs),
        "calibration_result": calibration.to_dict(),
        "darl_result": darl.to_dict(),
    }


def run_pair(
    experiment: Experiment,
    *,
    measurement: Measurement,
    profile: RestoreProfile,
    general: GeneralParameters,
    restore: RestoreParameters,
    calibration: CalibrationResult,
    darl: DarlResult,
    force: bool = False,
) -> Path:
    """Calculate and transactionally publish one profile/measurement pair."""
    target = experiment.restore_output_dir(profile.name, measurement.name)
    with prepare_output_directory(experiment, target, force=force) as directory:
        measurement_inputs, camera_exposures, line_exposure = _measurement_inputs(
            experiment, measurement, restore
        )
        artifact = build_legacy_restore_config(
            experiment=experiment,
            measurement=measurement,
            general=general,
            restore=restore,
            calibration=calibration,
            darl=darl,
            output_dir=directory,
            camera_exposures_us=camera_exposures,
            line_exposure_us=line_exposure,
        )
        run_legacy_restore(artifact)
        write_used_parameters(
            directory / "used-parameters.yaml",
            _effective_parameters(
                general=general,
                restore=restore,
                profile=profile,
                measurement=measurement,
                measurement_inputs=measurement_inputs,
                calibration=calibration,
                darl=darl,
            ),
        )
        class_slice_prefix = (
            None
            if restore.class_slice.drop_first is None
            and restore.class_slice.drop_last is None
            else f"cutted_{restore.class_slice.drop_first}_"
        )
        expected_signal = any(
            item.name == measurement.name for item in darl.distributions
        )
        result = collect_restore_result(
            directory,
            restore_profile=profile.name,
            measurement=measurement.name,
            detector_names=restore.detectors.names(),
            class_slice_prefix=class_slice_prefix,
            expected_signal_comparison=expected_signal,
        )
        save_restore_result(result, directory / "result.yaml")
    return target


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        experiment = Experiment.open(args.experiment)
        measurement_names = _select(
            experiment.measurement_names(), args.measurements, kind="измерение"
        )
        profile_names = _select(
            experiment.restore_profile_names(), args.profiles,
            kind="профиль восстановления",
        )
        if not measurement_names:
            raise ExperimentStructureError("В эксперименте нет измерений")
        if not profile_names:
            raise ExperimentStructureError("В эксперименте нет restore profiles")

        general = load_general_parameters(experiment.general_parameters_file)
        calibration = load_calibration_result(experiment.calibration_result_file)
        darl = load_darl_result(experiment.darl_result_file)
        profiles = tuple(experiment.restore_profile(name) for name in profile_names)
        measurements = tuple(experiment.measurement(name) for name in measurement_names)
        restore_by_profile = {
            profile.name: load_restore_parameters(profile.path)
            for profile in profiles
        }

        reports = tuple(
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
            issue for pair_report in reports for issue in pair_report.issues
        ))
        print(format_report(report, show_warnings=not args.no_warnings))
        if report.errors or (args.warnings_as_errors and report.warnings):
            return 1

        completed: list[Path] = []
        for profile in profiles:
            for measurement in measurements:
                target = run_pair(
                    experiment,
                    measurement=measurement,
                    profile=profile,
                    general=general,
                    restore=restore_by_profile[profile.name],
                    calibration=calibration,
                    darl=darl,
                    force=args.force,
                )
                completed.append(target)
                print(f"Восстановление сохранено: {target}")
    except ExperimentStructureError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except (
        ParametersError,
        CalibrationResultError,
        DarlResultError,
        RestorationError,
        OutputError,
        OSError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Восстановление завершено: обработано пар — {len(completed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
