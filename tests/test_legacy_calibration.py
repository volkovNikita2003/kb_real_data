from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml


class LegacyCalibrationReferenceTests(unittest.TestCase):
    def test_automated_legacy_calibration_matches_reference(self) -> None:
        if importlib.util.find_spec("pandas") is None:
            self.skipTest("pandas is not installed")
        project = Path(__file__).resolve().parents[1]
        experiment = project / "experiments/test_17_07_26_kmk_15"
        reference = project / "ref/test_17_07_26_kmk_15/kmk_15/calibration_auto"
        if not experiment.is_dir() or not reference.is_dir():
            self.skipTest("local reference data are not available")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            environment = os.environ.copy()
            existing_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = os.pathsep.join(
                part for part in (existing_pythonpath, str(project / "src")) if part
            )
            environment.setdefault("MPLBACKEND", "Agg")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "calibration.legacy.preprocessing_new_position",
                    str(experiment),
                    "--output-dir",
                    str(output),
                ],
                cwd=project,
                env=environment,
                stdout=subprocess.DEVNULL,
                check=True,
            )

            actual_result = yaml.safe_load((output / "result.yaml").read_text())
            expected_values: dict[str, float] = {}
            for line in (reference / "calibration_reference.txt").read_text().splitlines():
                name, value = line.split("\t")
                expected_values[name] = float(value)
            actual_values = {
                "camera_matrix_diagonal_mm": actual_result["camera"]["matrix_diagonal_mm"],
                "camera_x_shift_m": actual_result["camera"]["x_shift_m"],
                "camera_y_shift_m": actual_result["camera"]["y_shift_m"],
                "camera_pixel_width_m": actual_result["camera"]["pixel_width_m"],
                "line_start_angle_deg": actual_result["line_sensor"]["start_angle_deg"],
                "line_end_angle_deg": actual_result["line_sensor"]["end_angle_deg"],
                "line_pixel_width_m": actual_result["line_sensor"]["pixel_width_m"],
                "line_pixel_height_m": actual_result["line_sensor"]["pixel_height_m"],
                "line_to_camera_coefficient": actual_result["line_sensor"]["to_camera_coefficient"],
                "line_shift_m": actual_result["line_sensor"]["shift_m"],
                "line_peak_pixel": actual_result["line_sensor"]["peak_pixel"],
            }
            for name, expected in expected_values.items():
                if name not in actual_values:
                    continue
                with self.subTest(name=name):
                    self.assertAlmostEqual(
                        actual_values[name], expected,
                        delta=max(abs(expected) * 1e-6, 1e-14),
                    )

            for actual_name, reference_name in (
                ("camera-calibration-signal.txt", "camera_calibration_signal.txt"),
                ("line-calibration-signal.txt", "line_calibration_signal.txt"),
            ):
                actual = np.loadtxt(output / actual_name, skiprows=1)
                expected = np.loadtxt(reference / reference_name, skiprows=1)
                relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                self.assertLess(relative_error, 2e-6, actual_name)
