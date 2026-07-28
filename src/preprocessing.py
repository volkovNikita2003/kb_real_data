"""Command-line entry point for automatic detector calibration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

from calibration import calibrate, save_calibration_result
from errors import ExperimentStructureError, OutputError, ParametersError
from experiment import Experiment
from output import prepare_output_directory
from parameters import ExperimentParameters, write_used_parameters
from validate import format_report
from validation import validate_calibration_inputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Выполнить автоматическую калибровку детекторов.",
    )
    parser.add_argument("experiment", type=Path, help="директория эксперимента")
    parser.add_argument(
        "--force", action="store_true",
        help="перенести существующий результат в архив и выполнить расчёт заново",
    )
    warning_group = parser.add_mutually_exclusive_group()
    warning_group.add_argument(
        "--warnings-as-errors", action="store_true",
        help="не выполнять калибровку при наличии предупреждений",
    )
    warning_group.add_argument(
        "--no-warnings", action="store_true",
        help="не выводить предупреждения валидатора",
    )
    return parser


def run(experiment: Experiment, parameters: ExperimentParameters, *, force: bool = False) -> None:
    with prepare_output_directory(
        experiment, experiment.calibration_output_dir, force=force
    ) as directory:
        artifacts = calibrate(experiment, parameters, directory)
        write_used_parameters(
            directory / "used-parameters.yaml",
            parameters.effective_calibration(),
        )
        save_calibration_result(artifacts.result, directory / "result.yaml")
        np.savetxt(
            directory / "camera-calibration-signal.txt",
            artifacts.camera_signal,
            delimiter="\t",
            header="distance_um\tnormalized_signal\tfitted_signal",
            comments="",
            fmt="%.17g",
        )
        if artifacts.line_signal is not None:
            np.savetxt(
                directory / "line-calibration-signal.txt",
                artifacts.line_signal,
                delimiter="\t",
                header="distance_um\tnormalized_signal\tfitted_signal",
                comments="",
                fmt="%.17g",
            )
        with (directory / "quality.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(
                {"schema_version": 1, **artifacts.quality}, stream,
                allow_unicode=True, sort_keys=False,
            )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        experiment = Experiment.open(args.experiment)
        parameters = ExperimentParameters.load(experiment)
        report = validate_calibration_inputs(experiment, parameters)
        print(format_report(report, show_warnings=not args.no_warnings))
        if report.errors or (args.warnings_as_errors and report.warnings):
            return 1
        run(experiment, parameters, force=args.force)
    except ExperimentStructureError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except (ParametersError, OutputError, OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Калибровка сохранена: {experiment.calibration_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
