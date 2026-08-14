from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import calc_darl
import pipeline
from errors import ParametersError, PipelineError
from pipeline import StageResult
from validation import ValidationIssue, ValidationReport


class LegacyDarlAdapterTests(unittest.TestCase):
    def test_distribution_translation_preserves_order_and_values(self) -> None:
        first = SimpleNamespace(
            type="gaussian", name="pinhole_200", mean_nm=200000.0,
            sigma_nm=500.0, particle_count=20000.0,
        )
        second = SimpleNamespace(
            type="gaussian", name="sample", mean_nm=15000.0,
            sigma_nm=3000.0, particle_count=1e6,
        )
        parameters = Mock(distributions=(first, second))
        translated = calc_darl._legacy_distributions(parameters)
        self.assertEqual([item["row"]["comment"] for item in translated], ["pinhole_200", "sample"])
        self.assertEqual(translated[1]["type"], "gauss")
        self.assertEqual(translated[1]["row"]["count"], 1e6)

    def test_environment_contains_complete_run_contract(self) -> None:
        distribution = SimpleNamespace(
            type="gaussian", name="sample", mean_nm=1.0,
            sigma_nm=2.0, particle_count=3.0,
        )
        parameters = Mock(distributions=(distribution,))
        code_git_dir = Path("code_git")
        result_dir = Path("result")
        with patch.dict(calc_darl.os.environ, {"PYTHONPATH": "existing"}, clear=True):
            environment = calc_darl._legacy_environment(
                code_git_dir=code_git_dir, config_name="auto_test",
                result_dir=result_dir, parameters=parameters,
            )
        self.assertEqual(environment["REAL_DATA_AUTO_DARL_CONFIG_NAME"], "auto_test")
        self.assertEqual(
            environment["REAL_DATA_AUTO_DARL_RESULT_DIR"], str(result_dir)
        )
        self.assertEqual(
            environment["PYTHONPATH"].split(calc_darl.os.pathsep),
            [str(SRC_DIR), "existing"],
        )
        self.assertEqual(
            json.loads(environment["REAL_DATA_AUTO_DARL_DISTRIBUTIONS"])[0]["row"]["comment"],
            "sample",
        )


class DarlCliTests(unittest.TestCase):
    def setUp(self) -> None:
        experiment = Mock()
        experiment.darl_output_dir = Path("tmp") / "darl"
        self.experiment = experiment
        self.parameters = Mock()
        self.report = ValidationReport(())

    def _run_main(self, *arguments: str):
        def stage(*args, **kwargs):
            kwargs["report_callback"]("darl", self.report)
            if kwargs["warnings_as_errors"] and self.report.warnings:
                raise PipelineError("warnings")
            return StageResult("darl", (self.experiment.darl_output_dir,), self.report)
        with (
            patch.object(calc_darl.Experiment, "open", return_value=self.experiment),
            patch.object(pipeline, "run_darl_stage", side_effect=stage) as run_mock,
            redirect_stdout(StringIO()) as output,
        ):
            code = calc_darl.main(["experiment", *arguments])
        return code, output.getvalue(), run_mock

    def test_main_runs_loaded_experiment(self) -> None:
        code, output, run_mock = self._run_main("--force")
        self.assertEqual(code, 0)
        self.assertTrue(run_mock.call_args.kwargs["force"])
        self.assertIn("ошибок — 0", output)
        self.assertIn(str(self.experiment.darl_output_dir), output)

    def test_warning_flags_match_other_stage_commands(self) -> None:
        self.report = ValidationReport((
            ValidationIssue(
                "warning", "test_warning", "Тестовое предупреждение"
            ),
        ))
        code, output, run_mock = self._run_main("--warnings-as-errors")
        self.assertEqual(code, 1)
        self.assertIn("Тестовое предупреждение", output)
        run_mock.assert_called_once()

        code, output, run_mock = self._run_main("--no-warnings")
        self.assertEqual(code, 0)
        self.assertNotIn("Тестовое предупреждение", output)
        self.assertIn("предупреждений — 1 (скрыты)", output)
        run_mock.assert_called_once()

    def test_warning_flags_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            calc_darl.main([
                "experiment", "--warnings-as-errors", "--no-warnings",
            ])
        self.assertEqual(raised.exception.code, 2)

    def test_main_reports_parameter_error(self) -> None:
        with (
            patch.object(calc_darl.Experiment, "open", return_value=Mock()),
            patch.object(
                pipeline, "run_darl_stage",
                side_effect=ParametersError("bad parameters"),
            ),
            redirect_stderr(StringIO()) as error,
        ):
            code = calc_darl.main(["experiment"])
        self.assertEqual(code, 1)
        self.assertIn("bad parameters", error.getvalue())
