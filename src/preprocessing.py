"""Command-line entry point for automatic detector calibration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from errors import ExperimentStructureError, OutputError, ParametersError
from experiment import Experiment
from output import prepare_output_directory
from parameters import ExperimentParameters, write_used_parameters
from validate import format_report
from validation import validate_calibration_inputs


SRC_DIR = Path(__file__).resolve().parent


def _legacy_environment() -> dict[str, str]:
    """Return an environment in which the child can import packages from src."""
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC_DIR), existing_pythonpath) if part
    )
    environment.setdefault("MPLBACKEND", "Agg")
    return environment


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
        position = parameters.general.instrument.detector_position
        module = (
            "calibration.legacy.preprocessing_new_position"
            if position == "new"
            else "calibration.legacy.preprocessing"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                module,
                str(experiment.path),
                "--output-dir",
                str(directory),
            ],
            env=_legacy_environment(),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Legacy-калибровка завершилась с кодом {completed.returncode}"
            )
        write_used_parameters(
            directory / "used-parameters.yaml",
            parameters.effective_calibration(),
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
