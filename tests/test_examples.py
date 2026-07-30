from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from parameters import (
    load_calibration_parameters,
    load_darl_parameters,
    load_general_parameters,
    load_measurement_parameters,
    load_restore_parameters,
)


class ParameterExamplesTests(unittest.TestCase):
    def test_minimal_parameter_set_is_valid(self) -> None:
        root = PROJECT_DIR / "examples/minimal/input_parameters"

        general = load_general_parameters(root / "general.yaml")
        calibration = load_calibration_parameters(root / "calibration.yaml")
        darl = load_darl_parameters(root / "darl.yaml")
        restore = load_restore_parameters(root / "restore_profiles/default.yaml")

        self.assertEqual(general.detectors.names(), {"camera", "line_sensor"})
        self.assertEqual(general.detectors.line_sensor.pixel_count, 3643)
        self.assertEqual(general.detectors.camera.width_px, 2592)
        self.assertTrue(restore.detectors.camera.use_background)
        self.assertFalse((root / "measurements").exists())

    def test_complete_parameter_set_is_valid(self) -> None:
        root = PROJECT_DIR / "examples/complete/input_parameters"

        general = load_general_parameters(root / "general.yaml")
        calibration = load_calibration_parameters(root / "calibration.yaml")
        darl = load_darl_parameters(root / "darl.yaml")
        measurement = load_measurement_parameters(
            root / "measurements/kmk_15.yaml"
        )
        restore = load_restore_parameters(root / "restore_profiles/default.yaml")

        self.assertEqual(general.instrument.focal_length_um, 400000.0)
        self.assertEqual(calibration.camera.gaussian_sigma_px, 20.0)
        self.assertEqual(darl.particle_classes.max_diameter_nm, 3500000.0)
        self.assertEqual(measurement.expected_distribution.mean_nm, 15000.0)
        self.assertEqual(restore.solver.regularization_alpha, "best")
        self.assertFalse(restore.solver.use_w_critical)


if __name__ == "__main__":
    unittest.main()
