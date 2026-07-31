from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import restore
import pipeline
from errors import ExperimentStructureError, PipelineError
from pipeline import StageResult
from validation import ValidationIssue, ValidationReport


class RestoreCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.experiment = Mock()
        self.experiment.measurement_names.return_value = ("m1", "m2")
        self.experiment.restore_profile_names.return_value = ("p1", "p2")
        self.experiment.general_parameters_file = Path("general.yaml")
        self.experiment.calibration_result_file = Path("calibration/result.yaml")
        self.experiment.darl_result_file = Path("darl/result.yaml")
        self.measurements = {
            name: SimpleNamespace(name=name) for name in ("m1", "m2")
        }
        self.profiles = {
            name: SimpleNamespace(name=name, path=Path(f"{name}.yaml"))
            for name in ("p1", "p2")
        }
        self.experiment.measurement.side_effect = self.measurements.__getitem__
        self.experiment.restore_profile.side_effect = self.profiles.__getitem__
        self.general = Mock()
        self.calibration = Mock()
        self.darl = Mock()
        self.restore_parameters = {name: Mock() for name in ("p1", "p2")}
        self.report = ValidationReport(())

    def _main(self, *arguments: str):
        def stage(*args, **kwargs):
            profiles = kwargs["profile_names"] or ("p1", "p2")
            measurements = kwargs["measurement_names"] or ("m1", "m2")
            for name in profiles:
                if name not in self.profiles:
                    raise ExperimentStructureError(
                        f"Не найден профиль восстановления {name!r}"
                    )
            kwargs["report_callback"]("restore", self.report)
            if kwargs["warnings_as_errors"] and self.report.warnings:
                raise PipelineError("warnings")
            outputs = tuple(
                Path(f"/output/{profile}/{measurement}")
                for profile in profiles for measurement in measurements
            )
            return StageResult("restore", outputs, self.report)

        with (
            patch.object(restore.Experiment, "open", return_value=self.experiment),
            patch.object(pipeline, "run_restore_stage", side_effect=stage) as run_mock,
            redirect_stdout(StringIO()) as output,
        ):
            code = restore.main(["experiment", *arguments])
        return code, output.getvalue(), run_mock

    def test_default_runs_cartesian_product_in_profile_order(self) -> None:
        code, output, run_mock = self._main("--force")
        self.assertEqual(code, 0)
        self.assertEqual(
            len(run_mock.call_args_list), 1,
        )
        self.assertTrue(run_mock.call_args.kwargs["force"])
        self.assertIn("обработано пар — 4", output)

    def test_selectors_limit_both_dimensions(self) -> None:
        code, _, run_mock = self._main(
            "--profile", "p2", "--measurement", "m1"
        )
        self.assertEqual(code, 0)
        run_mock.assert_called_once()
        self.assertEqual(run_mock.call_args.kwargs["profile_names"], ["p2"])
        self.assertEqual(run_mock.call_args.kwargs["measurement_names"], ["m1"])

    def test_warnings_as_errors_stops_all_pairs(self) -> None:
        self.report = ValidationReport((
            ValidationIssue("warning", "test_warning", "Предупреждение"),
        ))
        code, output, run_mock = self._main("--warnings-as-errors")
        self.assertEqual(code, 1)
        self.assertIn("Предупреждение", output)
        run_mock.assert_called_once()

    def test_no_warnings_hides_details_and_runs(self) -> None:
        self.report = ValidationReport((
            ValidationIssue("warning", "test_warning", "Предупреждение"),
        ))
        code, output, run_mock = self._main(
            "--profile", "p1", "--measurement", "m1", "--no-warnings"
        )
        self.assertEqual(code, 0)
        self.assertNotIn("Предупреждение", output)
        self.assertIn("предупреждений — 1 (скрыты)", output)
        run_mock.assert_called_once()

    def test_unknown_selector_is_structure_error(self) -> None:
        with (
            patch.object(restore.Experiment, "open", return_value=self.experiment),
            patch.object(
                pipeline, "run_restore_stage",
                side_effect=ExperimentStructureError(
                    "Не найден профиль восстановления 'missing'"
                ),
            ),
            redirect_stderr(StringIO()) as error,
        ):
            code = restore.main(["experiment", "--profile", "missing"])
        self.assertEqual(code, 2)
        self.assertIn("Не найден профиль восстановления", error.getvalue())

    def test_warning_flags_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            restore.main([
                "experiment", "--warnings-as-errors", "--no-warnings",
            ])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
