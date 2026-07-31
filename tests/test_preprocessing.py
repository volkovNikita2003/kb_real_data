from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import preprocessing
import pipeline
from errors import PipelineError
from pipeline import StageResult
from validation import ValidationIssue, ValidationReport


class PreprocessingWarningCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.experiment = Mock()
        self.experiment.calibration_output_dir = Path("/tmp/calibration")
        self.parameters = Mock()
        self.report = ValidationReport((
            ValidationIssue("warning", "test_warning", "Тестовое предупреждение"),
        ))

    def run_main(self, *arguments: str) -> tuple[int, str, Mock]:
        def stage(*args, **kwargs):
            kwargs["report_callback"]("calibration", self.report)
            if kwargs["warnings_as_errors"] and self.report.warnings:
                raise PipelineError("warnings")
            return StageResult(
                "calibration", (Path("/tmp/calibration"),), self.report
            )
        with (
            patch.object(preprocessing.Experiment, "open", return_value=self.experiment),
            patch.object(pipeline, "run_calibration_stage", side_effect=stage) as run_mock,
            redirect_stdout(StringIO()) as output,
        ):
            code = preprocessing.main(["experiment", *arguments])
        return code, output.getvalue(), run_mock

    def test_warning_does_not_stop_default_run(self) -> None:
        code, output, run_mock = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("Тестовое предупреждение", output)
        run_mock.assert_called_once()

    def test_warnings_as_errors_stops_before_calibration(self) -> None:
        code, output, run_mock = self.run_main("--warnings-as-errors")
        self.assertEqual(code, 1)
        self.assertIn("Тестовое предупреждение", output)
        run_mock.assert_called_once()

    def test_no_warnings_hides_details_but_runs(self) -> None:
        code, output, run_mock = self.run_main("--no-warnings")
        self.assertEqual(code, 0)
        self.assertNotIn("Тестовое предупреждение", output)
        self.assertIn("предупреждений — 1 (скрыты)", output)
        run_mock.assert_called_once()

    def test_warning_flags_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            preprocessing.main([
                "experiment", "--warnings-as-errors", "--no-warnings",
            ])
        self.assertEqual(raised.exception.code, 2)


class LegacyEnvironmentTests(unittest.TestCase):
    def test_src_is_added_to_child_pythonpath(self) -> None:
        with patch.dict(preprocessing.os.environ, {"PYTHONPATH": "existing"}):
            environment = preprocessing._legacy_environment()
        self.assertEqual(
            environment["PYTHONPATH"].split(preprocessing.os.pathsep),
            [str(SRC_DIR), "existing"],
        )
