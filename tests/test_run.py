from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from errors import ExperimentStructureError, PipelineError
from pipeline import StageResult
import run
from validation import ValidationReport


EMPTY_REPORT = ValidationReport(())


class UnifiedRunCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.experiment = Mock()

    def test_calibration_routes_common_flags(self) -> None:
        output_path = Path("output") / "calibration"
        result = StageResult(
            "calibration", (output_path,), EMPTY_REPORT
        )
        with (
            patch.object(run.Experiment, "open", return_value=self.experiment),
            patch.object(
                run, "run_calibration_stage", return_value=result
            ) as stage,
            redirect_stdout(StringIO()) as output,
        ):
            code = run.main(["calibration", "experiment", "--force"])
        self.assertEqual(code, 0)
        self.assertTrue(stage.call_args.kwargs["force"])
        self.assertIn(str(output_path), output.getvalue())

    def test_restore_forwards_repeated_selectors(self) -> None:
        result = StageResult(
            "restore", (Path("/output/p1/m1"),), EMPTY_REPORT
        )
        with (
            patch.object(run.Experiment, "open", return_value=self.experiment),
            patch.object(run, "run_restore_stage", return_value=result) as stage,
            redirect_stdout(StringIO()),
        ):
            code = run.main([
                "restore", "experiment",
                "--profile", "p1", "--profile", "p2",
                "--measurement", "m1", "--measurement", "m2",
            ])
        self.assertEqual(code, 0)
        self.assertEqual(stage.call_args.kwargs["profile_names"], ["p1", "p2"])
        self.assertEqual(
            stage.call_args.kwargs["measurement_names"], ["m1", "m2"]
        )

    def test_all_prints_final_pair_count(self) -> None:
        results = (
            StageResult("calibration", (Path("calibration"),), EMPTY_REPORT),
            StageResult("darl", (Path("darl"),), EMPTY_REPORT),
            StageResult(
                "restore", (Path("p/m1"), Path("p/m2")), EMPTY_REPORT
            ),
        )
        with (
            patch.object(run.Experiment, "open", return_value=self.experiment),
            patch.object(run, "run_all_stages", return_value=results) as stages,
            redirect_stdout(StringIO()) as output,
        ):
            code = run.main([
                "all", "experiment", "--profile", "p", "--force",
            ])
        self.assertEqual(code, 0)
        self.assertEqual(stages.call_args.kwargs["profile_names"], ["p"])
        self.assertIn("восстановлено пар — 2", output.getvalue())

    def test_warning_flags_are_mutually_exclusive_for_every_command(self) -> None:
        for command in ("calibration", "darl", "restore", "all"):
            with self.subTest(command=command), self.assertRaises(SystemExit) as raised:
                run.main([
                    command, "experiment",
                    "--warnings-as-errors", "--no-warnings",
                ])
            self.assertEqual(raised.exception.code, 2)

    def test_restore_selectors_are_rejected_for_unrelated_commands(self) -> None:
        for command in ("calibration", "darl"):
            with self.subTest(command=command), self.assertRaises(SystemExit) as raised:
                run.main([command, "experiment", "--profile", "p"])
            self.assertEqual(raised.exception.code, 2)

    def test_structure_and_processing_errors_have_distinct_codes(self) -> None:
        with (
            patch.object(
                run.Experiment, "open",
                side_effect=ExperimentStructureError("structure"),
            ),
            redirect_stderr(StringIO()) as error,
        ):
            self.assertEqual(run.main(["calibration", "experiment"]), 2)
        self.assertIn("structure", error.getvalue())

        with (
            patch.object(run.Experiment, "open", return_value=self.experiment),
            patch.object(
                run, "run_darl_stage", side_effect=PipelineError("processing")
            ),
            redirect_stderr(StringIO()) as error,
        ):
            self.assertEqual(run.main(["darl", "experiment"]), 1)
        self.assertIn("processing", error.getvalue())


if __name__ == "__main__":
    unittest.main()
