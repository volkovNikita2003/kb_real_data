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
        def load_profile(path: Path):
            return self.restore_parameters[path.stem]

        def output_path(*args, **kwargs):
            measurement = kwargs["measurement"]
            profile = kwargs["profile"]
            return Path(f"/output/{profile.name}/{measurement.name}")

        with (
            patch.object(restore.Experiment, "open", return_value=self.experiment),
            patch.object(restore, "load_general_parameters", return_value=self.general),
            patch.object(restore, "load_calibration_result", return_value=self.calibration),
            patch.object(restore, "load_darl_result", return_value=self.darl),
            patch.object(restore, "load_restore_parameters", side_effect=load_profile),
            patch.object(restore, "validate_restore_inputs", return_value=self.report),
            patch.object(restore, "run_pair", side_effect=output_path) as run_mock,
            redirect_stdout(StringIO()) as output,
        ):
            code = restore.main(["experiment", *arguments])
        return code, output.getvalue(), run_mock

    def test_default_runs_cartesian_product_in_profile_order(self) -> None:
        code, output, run_mock = self._main("--force")
        self.assertEqual(code, 0)
        self.assertEqual(
            [(item.kwargs["profile"].name, item.kwargs["measurement"].name)
             for item in run_mock.call_args_list],
            [("p1", "m1"), ("p1", "m2"), ("p2", "m1"), ("p2", "m2")],
        )
        self.assertTrue(all(item.kwargs["force"] for item in run_mock.call_args_list))
        self.assertIn("обработано пар — 4", output)

    def test_selectors_limit_both_dimensions(self) -> None:
        code, _, run_mock = self._main(
            "--profile", "p2", "--measurement", "m1"
        )
        self.assertEqual(code, 0)
        run_mock.assert_called_once()
        self.assertEqual(run_mock.call_args.kwargs["profile"].name, "p2")
        self.assertEqual(run_mock.call_args.kwargs["measurement"].name, "m1")

    def test_warnings_as_errors_stops_all_pairs(self) -> None:
        self.report = ValidationReport((
            ValidationIssue("warning", "test_warning", "Предупреждение"),
        ))
        code, output, run_mock = self._main("--warnings-as-errors")
        self.assertEqual(code, 1)
        self.assertIn("Предупреждение", output)
        run_mock.assert_not_called()

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
