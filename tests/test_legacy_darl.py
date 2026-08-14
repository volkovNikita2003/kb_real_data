from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
ACTUAL = PROJECT_DIR / "experiments/test_17_07_26_kmk_15/output/darl"
REFERENCE = (
    PROJECT_DIR
    / "ref/test_17_07_26_kmk_15/kmk_15/darl-kmk-15-signal"
)


class LegacyDarlReferenceTests(unittest.TestCase):
    """Compare an explicitly generated local result with the saved baseline."""

    def test_generated_numerical_artifacts_match_reference(self) -> None:
        if not REFERENCE.is_dir():
            self.skipTest("local reference data are not available")
        if not ACTUAL.is_dir():
            self.skipTest(
                "Сначала выполните: python src/calc_darl.py "
                "experiments/test_17_07_26_kmk_15"
            )

        numerical_text_actual = (
            "b_kmk_15.txt",
            "b_pinhole_200_auto_control.txt",
            "background_signal_laser_0.txt",
            "bins_front_detector.txt",
            "particle_classes_lasser_0.txt",
            "kmk_15/modeled_signal.txt",
            "pinhole_200_auto_control/modeled_signal.txt",
        )
        numerical_text_reference = (
            "b_kmk_15.txt",
            "b_pinhole_200.txt",
            "background_signal_laser_0.txt",
            "bins_front_detector.txt",
            "particle_classes_lasser_0.txt",
            "kmk_15/modeled_signal.txt",
            "pinhole_200/modeled_signal.txt",
        )
        for act, ref in zip(numerical_text_actual, numerical_text_reference):
            with self.subTest(path=ref):
                np.testing.assert_array_equal(
                    np.loadtxt(ACTUAL / act),
                    np.loadtxt(REFERENCE / ref),
                )

        self.assertEqual(
            (ACTUAL / "FrontDetectorLogLine_detector.txt").read_bytes(),
            (REFERENCE / "FrontDetectorLogLine_detector.txt").read_bytes(),
        )
        actual_matrix = next(ACTUAL.glob("matrix-*.npz"))
        reference_matrix = next(REFERENCE.glob("matrix-*.npz"))
        with np.load(actual_matrix) as actual, np.load(reference_matrix) as reference:
            self.assertEqual(actual.files, reference.files)
            for name in actual.files:
                np.testing.assert_array_equal(actual[name], reference[name])
