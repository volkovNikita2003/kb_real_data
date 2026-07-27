"""Command-line interface for read-only experiment validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from errors import ExperimentStructureError, ParametersError
from experiment import Experiment
from parameters import ExperimentParameters
from validation import ValidationIssue, ValidationReport, validate_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Проверить структуру, параметры и данные эксперимента.",
    )
    parser.add_argument("experiment", type=Path, help="директория эксперимента")
    parser.add_argument(
        "--measurement",
        action="append",
        dest="measurements",
        help="проверить только указанное измерение; можно повторять",
    )
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        help="проверить только указанный профиль; можно повторять",
    )
    warning_group = parser.add_mutually_exclusive_group()
    warning_group.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="возвращать код 1 при наличии предупреждений",
    )
    warning_group.add_argument(
        "--no-warnings",
        action="store_true",
        help="не выводить предупреждения",
    )
    return parser


def _context(issue: ValidationIssue) -> str:
    parts = [part for part in (issue.measurement, issue.restore_profile) if part]
    return f" [{' / '.join(parts)}]" if parts else ""


def format_report(
    report: ValidationReport,
    *,
    show_warnings: bool = True,
) -> str:
    lines: list[str] = []
    issues = report.issues if show_warnings else report.errors
    for issue in issues:
        lines.append(
            f"{issue.severity.upper():7} {_context(issue)} {issue.code}".rstrip()
        )
        lines.append(f"        {issue.message}")
        if issue.path is not None:
            lines.append(f"        Путь: {issue.path}")
    summary = (
        "Проверка завершена: "
        f"ошибок — {len(report.errors)}, "
        f"предупреждений — {len(report.warnings)}"
    )
    summary += " (скрыты)." if not show_warnings else "."
    lines.append(summary)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        experiment = Experiment.open(args.experiment)
    except ExperimentStructureError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    try:
        parameters = ExperimentParameters.load(experiment)
        report = validate_experiment(
            experiment,
            parameters,
            measurement_names=args.measurements,
            restore_profile_names=args.profiles,
        )
    except ParametersError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except ExperimentStructureError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(format_report(report, show_warnings=not args.no_warnings))
    if report.errors or (args.warnings_as_errors and report.warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
