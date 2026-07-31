from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from errors import OutputError, PipelineError
from experiment import Experiment
import pipeline
from pipeline import StageResult
from tests.test_validation import create_experiment
from validation import ValidationIssue, ValidationReport


EMPTY_REPORT = ValidationReport(())


class StageOrchestrationTests(unittest.TestCase):
    def test_calibration_loads_validates_and_runs(self) -> None:
        experiment = Mock()
        experiment.calibration_output_dir = Path("/output/calibration")
        parameters = Mock()
        with (
            patch.object(
                pipeline.CalibrationStageParameters, "load",
                return_value=parameters,
            ) as load,
            patch.object(
                pipeline, "validate_calibration_inputs",
                return_value=EMPTY_REPORT,
            ) as validate,
            patch.object(pipeline.preprocessing, "run") as run,
        ):
            result = pipeline.run_calibration_stage(experiment, force=True)
        load.assert_called_once_with(experiment)
        validate.assert_called_once_with(experiment, parameters)
        run.assert_called_once_with(experiment, parameters, force=True)
        self.assertEqual(result.output_paths, (Path("/output/calibration"),))

    def test_warning_policy_is_applied_before_calculation(self) -> None:
        report = ValidationReport((
            ValidationIssue("warning", "warning", "warning"),
        ))
        with (
            patch.object(
                pipeline.CalibrationStageParameters, "load", return_value=Mock()
            ),
            patch.object(
                pipeline, "validate_calibration_inputs", return_value=report
            ),
            patch.object(pipeline.preprocessing, "run") as run,
        ):
            with self.assertRaises(PipelineError):
                pipeline.run_calibration_stage(
                    Mock(), warnings_as_errors=True
                )
        run.assert_not_called()

    def test_all_stages_run_in_order_and_forward_restore_selection(self) -> None:
        experiment = Mock()
        order: list[str] = []
        calibration = StageResult("calibration", (), EMPTY_REPORT)
        darl = StageResult("darl", (), EMPTY_REPORT)
        restoration = StageResult("restore", (Path("result"),), EMPTY_REPORT)
        with (
            patch.object(
                pipeline, "preflight_all", side_effect=lambda *a, **k: order.append("preflight")
            ) as preflight,
            patch.object(
                pipeline, "run_calibration_stage",
                side_effect=lambda *a, **k: (order.append("calibration"), calibration)[1],
            ),
            patch.object(
                pipeline, "run_darl_stage",
                side_effect=lambda *a, **k: (order.append("darl"), darl)[1],
            ),
            patch.object(
                pipeline, "run_restore_stage",
                side_effect=lambda *a, **k: (order.append("restore"), restoration)[1],
            ) as restore_stage,
        ):
            result = pipeline.run_all_stages(
                experiment,
                measurement_names=["m1"], profile_names=["p1"], force=True,
            )
        self.assertEqual(order, ["preflight", "calibration", "darl", "restore"])
        self.assertEqual(result, (calibration, darl, restoration))
        preflight.assert_called_once()
        self.assertEqual(
            restore_stage.call_args.kwargs["measurement_names"], ["m1"]
        )
        self.assertEqual(restore_stage.call_args.kwargs["profile_names"], ["p1"])

    def test_failure_stops_following_stages_without_rollback(self) -> None:
        with (
            patch.object(pipeline, "preflight_all"),
            patch.object(
                pipeline, "run_calibration_stage",
                return_value=StageResult("calibration", (), EMPTY_REPORT),
            ) as calibration,
            patch.object(
                pipeline, "run_darl_stage", side_effect=RuntimeError("DARL")
            ) as darl,
            patch.object(pipeline, "run_restore_stage") as restoration,
        ):
            with self.assertRaisesRegex(RuntimeError, "DARL"):
                pipeline.run_all_stages(Mock())
        calibration.assert_called_once()
        darl.assert_called_once()
        restoration.assert_not_called()

    def test_restore_validates_every_pair_before_first_calculation(self) -> None:
        experiment = Mock()
        measurements = (
            SimpleNamespace(name="m1"), SimpleNamespace(name="m2")
        )
        profiles = (SimpleNamespace(name="p1"),)
        restore_parameters = {"p1": Mock()}
        reports = (
            EMPTY_REPORT,
            ValidationReport((
                ValidationIssue("error", "invalid", "invalid"),
            )),
        )
        with (
            patch.object(
                pipeline, "_restore_context",
                return_value=(
                    Mock(), Mock(), Mock(), measurements, profiles,
                    restore_parameters,
                ),
            ),
            patch.object(
                pipeline, "validate_restore_inputs", side_effect=reports
            ) as validate,
            patch.object(pipeline.restore_command, "run_pair") as run_pair,
        ):
            with self.assertRaises(PipelineError):
                pipeline.run_restore_stage(experiment)
        self.assertEqual(validate.call_count, 2)
        run_pair.assert_not_called()


class FullRunPreflightTests(unittest.TestCase):
    def test_existing_late_result_is_rejected_before_processing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            target = path / "output/restore/default/sample"
            target.mkdir(parents=True)
            experiment = Experiment.open(path)
            with self.assertRaisesRegex(OutputError, "--force"):
                pipeline.preflight_all(experiment)

    def test_force_accepts_existing_directories_but_rejects_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            target = path / "output/restore/default/sample"
            target.parent.mkdir(parents=True)
            target.write_text("file", encoding="utf-8")
            experiment = Experiment.open(path)
            with self.assertRaisesRegex(OutputError, "обычной директорией"):
                pipeline.preflight_all(experiment, force=True)


if __name__ == "__main__":
    unittest.main()
