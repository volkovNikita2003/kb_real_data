"""Command-line entry point for the unchanged legacy DARL calculation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from darl.legacy.config import (
    build_legacy_config,
    default_code_git_dir,
)
from darl.result import collect_darl_result, save_darl_result
from errors import DarlError, ExperimentStructureError, OutputError, ParametersError
from experiment import Experiment
from output import prepare_output_directory
from parameters import DarlStageParameters, write_used_parameters
from validate import format_report


SRC_DIR = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Рассчитать аппаратную матрицу и тестовые сигналы DARL.",
    )
    parser.add_argument("experiment", type=Path, help="директория эксперимента")
    parser.add_argument(
        "--force", action="store_true",
        help="перенести существующий результат в архив и выполнить расчёт заново",
    )
    warning_group = parser.add_mutually_exclusive_group()
    warning_group.add_argument(
        "--warnings-as-errors", action="store_true",
        help="не выполнять DARL при наличии предупреждений",
    )
    warning_group.add_argument(
        "--no-warnings", action="store_true",
        help="не выводить предупреждения валидатора",
    )
    return parser


def _legacy_distributions(parameters: DarlStageParameters) -> list[dict[str, object]]:
    type_names = {"gaussian": "gauss"}
    return [
        {
            "type": type_names[item.type],
            "row": {
                "active": 0,
                "mean": item.mean_nm,
                "sigma": item.sigma_nm,
                "count": item.particle_count,
                "comment": item.name,
            },
        }
        for item in parameters.distributions
    ]


def _legacy_environment(
    *,
    code_git_dir: Path,
    config_name: str,
    result_dir: Path,
    parameters: DarlStageParameters,
) -> dict[str, str]:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC_DIR), existing_pythonpath) if part
    )
    environment["MPLBACKEND"] = "Agg"
    environment["REAL_DATA_AUTO_CODE_GIT_DIR"] = str(code_git_dir)
    environment["REAL_DATA_AUTO_DARL_CONFIG_NAME"] = config_name
    environment["REAL_DATA_AUTO_DARL_RESULT_DIR"] = str(result_dir)
    environment["REAL_DATA_AUTO_DARL_DISTRIBUTIONS"] = json.dumps(
        _legacy_distributions(parameters), ensure_ascii=True,
        separators=(",", ":"),
    )
    return environment


def run(
    experiment: Experiment,
    parameters: DarlStageParameters,
    *,
    force: bool = False,
    code_git_dir: str | Path | None = None,
) -> None:
    """Run legacy numerical code and transactionally publish its artifacts."""
    code_git = Path(code_git_dir or default_code_git_dir()).expanduser().resolve()
    with prepare_output_directory(
        experiment, experiment.darl_output_dir, force=force
    ) as directory:
        config = build_legacy_config(
            experiment, parameters, code_git_dir=code_git, overwrite=True
        )
        (directory / "legacy-config.txt").write_text(
            config.text, encoding="utf-8"
        )
        completed = subprocess.run(
            [sys.executable, "-m", "darl.legacy.calc_darl"],
            env=_legacy_environment(
                code_git_dir=code_git,
                config_name=config.config_name,
                result_dir=directory,
                parameters=parameters,
            ),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Legacy-расчёт DARL завершился с кодом {completed.returncode}"
            )
        result = collect_darl_result(
            directory,
            legacy_config_name=config.config_name,
            detector_names=parameters.general.detectors.names(),
            signal_value_type=parameters.darl.signal.value_type,
            distribution_names=tuple(item.name for item in parameters.distributions),
        )
        write_used_parameters(
            directory / "used-parameters.yaml",
            parameters.effective_parameters(),
        )
        save_darl_result(result, directory / "result.yaml")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        from pipeline import run_darl_stage

        experiment = Experiment.open(args.experiment)
        result = run_darl_stage(
            experiment,
            force=args.force,
            warnings_as_errors=args.warnings_as_errors,
            report_callback=lambda _stage, report: print(format_report(
                report, show_warnings=not args.no_warnings
            )),
        )
    except ExperimentStructureError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except (
        ParametersError, OutputError, DarlError,
        OSError, ValueError, RuntimeError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Результаты DARL сохранены: {result.output_paths[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
