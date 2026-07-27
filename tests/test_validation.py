from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from experiment import Experiment
from parameters import ExperimentParameters
from validate import format_report, main
from validation import parse_exposure_filename, validate_experiment


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def create_experiment(
    root: Path,
    *,
    line_sensor: bool = False,
    use_background: bool = True,
    measurement_names: tuple[str, ...] = ("sample",),
) -> Path:
    path = root / "experiment"
    parameters = path / "input_parameters"
    detectors: dict[str, object] = {"camera": {}}
    calibration: dict[str, object] = {
        "schema_version": 1,
        "mode": "automatic",
        "camera": {},
    }
    darl_detectors: dict[str, object] = {"camera": {}}
    restore_detectors: dict[str, object] = {
        "camera": {"use_background": use_background}
    }
    if line_sensor:
        detectors["line_sensor"] = {}
        calibration["line_sensor"] = {
            "pinhole_position_m": 0.0002,
            "signal_position_m": 0.0203,
        }
        darl_detectors["line_sensor"] = {}
        restore_detectors["line_sensor"] = {
            "use_background": use_background
        }

    dump(
        parameters / "general.yaml",
        {
            "schema_version": 1,
            "detectors": detectors,
            "instrument": {"detector_position": "new"},
        },
    )
    dump(parameters / "calibration.yaml", calibration)
    dump(
        parameters / "darl.yaml",
        {
            "schema_version": 1,
            "detectors": darl_detectors,
            "particles": {
                "refractive_index": 1.77,
                "absorption_coefficient": 0.0,
                "type": "sphere",
            },
            "medium": {
                "inside_cuvette_refractive_index": 1.3333,
                "cuvette_refractive_index": 1.5,
                "outside_cuvette_refractive_index": 1.0,
            },
        },
    )
    dump(
        parameters / "restore_profiles/default.yaml",
        {
            "schema_version": 1,
            "detectors": restore_detectors,
        },
    )

    touch(path / "data/calibration/cam/100.bmp")
    if line_sensor:
        touch(path / "data/calibration/lin/10.txt")
    for name in measurement_names:
        touch(path / f"data/{name}/cam/100.bmp")
        if use_background:
            touch(path / f"data/{name}/cam_back/100.bmp")
        if line_sensor:
            touch(path / f"data/{name}/lin/10.txt")
            if use_background:
                touch(path / f"data/{name}/lin_back/10.txt")
    return path


def validate(path: Path, **kwargs: object):
    experiment = Experiment.open(path)
    parameters = ExperimentParameters.load(experiment)
    return validate_experiment(experiment, parameters, **kwargs)


class ExposureFilenameTests(unittest.TestCase):
    def test_valid_exposure(self) -> None:
        self.assertEqual(parse_exposure_filename("400000.bmp", ".bmp"), 400000)
        self.assertEqual(parse_exposure_filename("400000.txt", ".txt"), 400000)

    def test_invalid_exposures(self) -> None:
        for name in ("0.bmp", "-1.bmp", "01.bmp", "1.5.bmp", "1us.bmp", "1.BMP"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_exposure_filename(name, ".bmp")


class ValidationTests(unittest.TestCase):
    def test_valid_camera_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = validate(create_experiment(Path(directory)))
            self.assertTrue(report.is_valid)
            self.assertEqual(report.issues, ())

    def test_valid_camera_and_line_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = validate(
                create_experiment(Path(directory), line_sensor=True)
            )
            self.assertTrue(report.is_valid)

    def test_missing_camera_calibration_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            (path / "data/calibration/cam").rename(
                path / "data/calibration/moved_cam"
            )

            report = validate(path)

            self.assertIn(
                "missing_calibration_detector_data",
                {i.code for i in report.errors},
            )

    def test_empty_camera_calibration_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            (path / "data/calibration/cam/100.bmp").unlink()

            report = validate(path)

            self.assertIn(
                "empty_camera_calibration",
                {i.code for i in report.errors},
            )

    def test_line_calibration_requires_exactly_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory), line_sensor=True)
            touch(path / "data/calibration/lin/20.txt")

            report = validate(path)

            self.assertIn(
                "invalid_line_calibration_file_count",
                {i.code for i in report.errors},
            )

    def test_missing_used_detector_directory_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            (path / "data/sample/cam").rename(path / "data/sample/moved_cam")

            report = validate(path)

            self.assertIn("missing_detector_data", {i.code for i in report.errors})
            self.assertIn("unknown_measurement_entry", {i.code for i in report.errors})

    def test_unused_detector_data_is_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            touch(path / "data/sample/lin/10.txt")
            touch(path / "data/sample/lin_back/10.txt")

            report = validate(path)

            warnings = [i for i in report.warnings if i.code == "unused_detector_data"]
            self.assertEqual(len(warnings), 2)
            self.assertTrue(report.is_valid)

    def test_invalid_filename_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            touch(path / "data/sample/cam/10.5.bmp")

            report = validate(path)

            self.assertIn("invalid_exposure_name", {i.code for i in report.errors})

    def test_missing_and_extra_camera_background_exposures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            touch(path / "data/sample/cam/200.bmp")
            touch(path / "data/sample/cam_back/300.bmp")

            report = validate(path)

            self.assertIn(
                "missing_camera_background_exposure",
                {i.code for i in report.errors},
            )
            self.assertIn(
                "extra_camera_background_exposure",
                {i.code for i in report.warnings},
            )

    def test_missing_background_directory_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = create_experiment(root)
            (path / "data/sample/cam_back").rename(root / "removed_cam_back")

            report = validate(path)

            self.assertIn(
                "missing_camera_background",
                {i.code for i in report.errors},
            )

    def test_existing_background_is_warning_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory), use_background=False)
            touch(path / "data/sample/cam_back/100.bmp")

            report = validate(path)

            self.assertIn("unused_camera_background", {i.code for i in report.warnings})
            self.assertTrue(report.is_valid)

    def test_line_signal_requires_exactly_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory), line_sensor=True)
            touch(path / "data/sample/lin/20.txt")

            report = validate(path)

            self.assertIn("invalid_line_signal_file_count", {i.code for i in report.errors})

    def test_line_background_exposure_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory), line_sensor=True)
            (path / "data/sample/lin_back/10.txt").rename(
                path / "data/sample/lin_back/20.txt"
            )

            report = validate(path)

            self.assertIn("line_exposure_mismatch", {i.code for i in report.errors})

    def test_line_background_requires_exactly_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory), line_sensor=True)
            touch(path / "data/sample/lin_back/20.txt")

            report = validate(path)

            self.assertIn(
                "invalid_line_background_file_count",
                {i.code for i in report.errors},
            )

    def test_orphan_measurement_parameters_are_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            dump(
                path / "input_parameters/measurements/missing.yaml",
                {
                    "schema_version": 1,
                    "expected_distribution": {
                        "type": "gaussian",
                        "mean_nm": 10,
                        "sigma_nm": 1,
                        "particle_count": 1,
                    },
                },
            )

            report = validate(path)

            self.assertIn("unknown_measurement_parameters", {i.code for i in report.errors})

    def test_invalid_measurement_parameters_are_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            dump(
                path / "input_parameters/measurements/sample.yaml",
                {
                    "schema_version": 1,
                    "expected_distribution": {
                        "type": "gaussian",
                        "mean_nm": "not-a-number",
                        "sigma_nm": 1,
                        "particle_count": 1,
                    },
                },
            )

            report = validate(path)

            self.assertIn(
                "invalid_measurement_parameters",
                {i.code for i in report.errors},
            )

    def test_absent_measurements_are_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory), measurement_names=())

            report = validate(path)

            self.assertIn("no_measurements", {i.code for i in report.errors})

    def test_absent_restore_profiles_are_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            (path / "input_parameters/restore_profiles/default.yaml").unlink()

            report = validate(path)

            self.assertIn("no_restore_profiles", {i.code for i in report.errors})

    def test_selection_limits_measurement_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(
                Path(directory), measurement_names=("good", "bad")
            )
            touch(path / "data/bad/cam/not-an-exposure.bmp")

            report = validate(path, measurement_names=("good",))
            self.assertNotIn("invalid_exposure_name", {i.code for i in report.errors})

            report = validate(path)
            self.assertIn("invalid_exposure_name", {i.code for i in report.errors})

    def test_selection_limits_profile_background_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            dump(
                path / "input_parameters/restore_profiles/no_background.yaml",
                {
                    "schema_version": 1,
                    "detectors": {
                        "camera": {"use_background": False},
                    },
                },
            )

            all_profiles = validate(path)
            default_only = validate(path, restore_profile_names=("default",))

            self.assertIn(
                "unused_camera_background",
                {i.code for i in all_profiles.warnings},
            )
            self.assertNotIn(
                "unused_camera_background",
                {i.code for i in default_only.warnings},
            )


class ValidationCliTests(unittest.TestCase):
    def test_success_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            output = StringIO()
            with redirect_stdout(output):
                code = main([str(path)])

            self.assertEqual(code, 0)
            self.assertIn("ошибок — 0", output.getvalue())

    def test_warnings_as_errors_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory), use_background=False)
            touch(path / "data/sample/cam_back/100.bmp")
            with redirect_stdout(StringIO()):
                code = main([str(path), "--warnings-as-errors"])

            self.assertEqual(code, 1)

    def test_validation_error_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            touch(path / "data/sample/cam/invalid.bmp")
            with redirect_stdout(StringIO()):
                code = main([str(path)])

            self.assertEqual(code, 1)

    def test_no_warnings_hides_warning_details(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory), use_background=False)
            touch(path / "data/sample/cam_back/100.bmp")
            output = StringIO()
            with redirect_stdout(output):
                code = main([str(path), "--no-warnings"])

            self.assertEqual(code, 0)
            self.assertNotIn("unused_camera_background", output.getvalue())
            self.assertIn("предупреждений — 1 (скрыты)", output.getvalue())

    def test_format_report_does_not_modify_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory), use_background=False)
            touch(path / "data/sample/cam_back/100.bmp")
            report = validate(path)

            text = format_report(report, show_warnings=False)

            self.assertEqual(len(report.warnings), 1)
            self.assertNotIn("unused_camera_background", text)

    def test_open_error_exit_code(self) -> None:
        error = StringIO()
        with redirect_stderr(error):
            code = main(["missing-experiment"])

        self.assertEqual(code, 2)
        self.assertIn("ERROR", error.getvalue())

    def test_unknown_selector_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            error = StringIO()
            with redirect_stderr(error):
                code = main([str(path), "--measurement", "missing"])

            self.assertEqual(code, 2)
            self.assertIn("ERROR", error.getvalue())


if __name__ == "__main__":
    unittest.main()
