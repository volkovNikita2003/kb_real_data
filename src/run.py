"""Unified command-line interface for experiment processing stages."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from errors import ExperimentStructureError
from experiment import Experiment
from pipeline import (
    StageResult,
    run_all_stages,
    run_calibration_stage,
    run_darl_stage,
    run_restore_stage,
)
from validate import format_report
from validation import ValidationReport


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("experiment", type=Path, help="директория эксперимента")
    parser.add_argument(
        "--force", action="store_true",
        help="архивировать существующий результат и выполнить этап заново",
    )
    warnings = parser.add_mutually_exclusive_group()
    warnings.add_argument(
        "--warnings-as-errors", action="store_true",
        help="не начинать вычисление при наличии предупреждений",
    )
    warnings.add_argument(
        "--no-warnings", action="store_true",
        help="не выводить предупреждения валидатора",
    )


def _restore_selectors(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--measurement", action="append", dest="measurements",
        help="обработать только измерение; можно повторять",
    )
    parser.add_argument(
        "--profile", action="append", dest="profiles",
        help="использовать только restore profile; можно повторять",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Запустить этапы обработки реального эксперимента.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    calibration = commands.add_parser(
        "calibration", help="выполнить калибровку детекторов"
    )
    _common_arguments(calibration)

    darl = commands.add_parser(
        "darl", help="рассчитать аппаратную матрицу и модельные сигналы"
    )
    _common_arguments(darl)

    restoration = commands.add_parser(
        "restore", help="выполнить восстановления выбранных пар"
    )
    _common_arguments(restoration)
    _restore_selectors(restoration)

    all_stages = commands.add_parser(
        "all", help="последовательно выполнить весь процесс"
    )
    _common_arguments(all_stages)
    _restore_selectors(all_stages)
    return parser


def _reporter(show_warnings: bool):
    def report(stage: str, validation: ValidationReport) -> None:
        titles = {
            "calibration": "Калибровка",
            "darl": "DARL",
            "restore": "Восстановление",
        }
        print(f"========== {titles[stage]} ==========")
        print(format_report(validation, show_warnings=show_warnings))
    return report


def _print_result(result: StageResult) -> None:
    if result.name == "restore":
        for path in result.output_paths:
            print(f"Восстановление сохранено: {path}")
    elif result.output_paths:
        messages = {
            "calibration": "Калибровка сохранена",
            "darl": "Результаты DARL сохранены",
        }
        print(f"{messages[result.name]}: {result.output_paths[0]}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        experiment = Experiment.open(args.experiment)
        common = {
            "force": args.force,
            "warnings_as_errors": args.warnings_as_errors,
            "report_callback": _reporter(not args.no_warnings),
        }
        if args.command == "calibration":
            results = (run_calibration_stage(experiment, **common),)
        elif args.command == "darl":
            results = (run_darl_stage(experiment, **common),)
        elif args.command == "restore":
            results = (run_restore_stage(
                experiment,
                measurement_names=args.measurements,
                profile_names=args.profiles,
                **common,
            ),)
        else:
            results = run_all_stages(
                experiment,
                measurement_names=args.measurements,
                profile_names=args.profiles,
                **common,
            )
    except ExperimentStructureError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    for result in results:
        _print_result(result)
    if args.command == "all":
        print(
            "Полная обработка завершена: "
            f"восстановлено пар — {len(results[-1].output_paths)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
