from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from calibration.refactoring.result import (
    CalibrationResult,
    CalibrationResultError,
    CameraCalibrationResult,
    LineSensorCalibrationResult,
    load_calibration_result,
    save_calibration_result,
)


def camera() -> CameraCalibrationResult:
    return CameraCalibrationResult(6.19, 1.7e-4, 4.9e-5, 1.91e-6)


def line() -> LineSensorCalibrationResult:
    return LineSensorCalibrationResult(0.88, 5.09, 8.1e-6, 2.02e-4, 0.31, 0.02, 1752.0)


class CalibrationResultTests(unittest.TestCase):
    def test_round_trip_with_line_sensor(self) -> None:
        expected = CalibrationResult(1, camera(), line())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/result.yaml"
            save_calibration_result(expected, path)
            self.assertEqual(load_calibration_result(path), expected)
        self.assertEqual(expected["camera"]["matrix_diagonal_mm"], 6.19)

    def test_round_trip_camera_only_omits_line_sensor(self) -> None:
        expected = CalibrationResult(1, camera())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.yaml"
            save_calibration_result(expected, path)
            self.assertNotIn("line_sensor", path.read_text(encoding="utf-8"))
            self.assertEqual(load_calibration_result(path), expected)

    def test_non_finite_values_are_rejected_by_dataclasses(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(CalibrationResultError, "finite"):
                CameraCalibrationResult(6.19, value, 0.0, 1e-6)

    def test_boolean_is_not_a_number(self) -> None:
        with self.assertRaisesRegex(CalibrationResultError, "expected a number"):
            CameraCalibrationResult(True, 0.0, 0.0, 1e-6)

    def test_unknown_and_missing_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.yaml"
            path.write_text("schema_version: 1\ncamera:\n  unknown: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(CalibrationResultError, "unknown fields"):
                load_calibration_result(path)

    def test_duplicate_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.yaml"
            path.write_text("schema_version: 1\nschema_version: 1\ncamera: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(CalibrationResultError, "duplicate key"):
                load_calibration_result(path)

    def test_invalid_geometry_is_rejected(self) -> None:
        with self.assertRaisesRegex(CalibrationResultError, "positive"):
            CameraCalibrationResult(6.19, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(CalibrationResultError, "less than"):
            LineSensorCalibrationResult(5.0, 1.0, 1e-6, 1e-6, 1.0, 0.0, 0.0)

    def test_unsupported_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(CalibrationResultError, "unsupported"):
            CalibrationResult(2, camera())


if __name__ == "__main__":
    unittest.main()
