from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from calibration.refactoring.engine import calibrate
from calibration.refactoring.geometry import NATIVE_PIXEL_WIDTH_M, camera_shift_m
from calibration.refactoring.input import build_camera_hdr
from experiment import Experiment
from parameters import ExperimentParameters


class CalibrationInputTests(unittest.TestCase):
    def test_hdr_uses_first_valid_exposure_and_reference_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.array([[22, 255]], dtype=np.uint8)).save(root / "24.bmp")
            Image.fromarray(np.array([[32, 42]], dtype=np.uint8)).save(root / "48.bmp")
            hdr, valid = build_camera_hdr(root, [24, 48], image_shape=(1, 2))
            # Effective 24 us exposure in the legacy camera model is 23.4 us.
            self.assertAlmostEqual(hdr[0, 0], 10 * 48 / 23.4)
            self.assertAlmostEqual(hdr[0, 1], 30 * 48 / 46.8)
            self.assertTrue(np.all(valid))

    def test_old_and_new_camera_shift_directions(self) -> None:
        center = np.array([100.0, 900.0])
        old = camera_shift_m(center, "old", 1944, 2592)
        new = camera_shift_m(center, "new", 1944, 2592)
        np.testing.assert_allclose(
            old, (center - np.array([2592.0, 972.0])) * NATIVE_PIXEL_WIDTH_M
        )
        np.testing.assert_allclose(
            new, (np.array([0.0, 972.0]) - center) * NATIVE_PIXEL_WIDTH_M
        )


class ReferenceCalibrationTests(unittest.TestCase):
    def test_reference_experiment_matches_legacy_result(self) -> None:
        project = Path(__file__).resolve().parents[1]
        experiment_path = project / "experiments/test_17_07_26_kmk_15"
        reference = (
            project / "ref/test_17_07_26_kmk_15/kmk_15/calibration_auto"
        )
        if not experiment_path.is_dir() or not reference.is_dir():
            self.skipTest("local reference data are not available")
        experiment = Experiment.open(experiment_path)
        parameters = ExperimentParameters.load(experiment)
        with tempfile.TemporaryDirectory() as directory:
            artifacts = calibrate(experiment, parameters, Path(directory))

        expected: dict[str, float] = {}
        for line in (reference / "calibration_reference.txt").read_text().splitlines():
            name, value = line.split("\t")
            expected[name] = float(value)
        result = artifacts.result.to_dict()
        actual = {
            "camera_matrix_diagonal_mm": result["camera"]["matrix_diagonal_mm"],
            "camera_x_shift_m": result["camera"]["x_shift_m"],
            "camera_y_shift_m": result["camera"]["y_shift_m"],
            "camera_pixel_width_m": result["camera"]["pixel_width_m"],
            "line_start_angle_deg": result["line_sensor"]["start_angle_deg"],
            "line_end_angle_deg": result["line_sensor"]["end_angle_deg"],
            "line_pixel_width_m": result["line_sensor"]["pixel_width_m"],
            "line_pixel_height_m": result["line_sensor"]["pixel_height_m"],
            "line_to_camera_coefficient": result["line_sensor"]["to_camera_coefficient"],
            "line_shift_m": result["line_sensor"]["shift_m"],
            "line_peak_pixel": result["line_sensor"]["peak_pixel"],
        }
        for name, expected_value in expected.items():
            if name not in actual:
                continue
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    actual[name], expected_value,
                    delta=max(abs(expected_value) * 3e-7, 1e-14),
                )

        camera_reference = np.loadtxt(
            reference / "camera_calibration_signal.txt", skiprows=1
        )
        line_reference = np.loadtxt(
            reference / "line_calibration_signal.txt", skiprows=1
        )
        np.testing.assert_allclose(
            artifacts.camera_signal[:, 0], camera_reference[:, 0],
            rtol=3e-7, atol=1e-10,
        )
        self.assertLess(
            np.linalg.norm(artifacts.camera_signal[:, 1:] - camera_reference[:, 1:])
            / np.linalg.norm(camera_reference[:, 1:]),
            3e-7,
        )
        assert artifacts.line_signal is not None
        np.testing.assert_allclose(
            artifacts.line_signal[:, 0], line_reference[:, 0],
            rtol=3e-7, atol=3e-6,
        )
        self.assertLess(
            np.linalg.norm(artifacts.line_signal[:, 1:] - line_reference[:, 1:])
            / np.linalg.norm(line_reference[:, 1:]),
            3e-6,
        )
