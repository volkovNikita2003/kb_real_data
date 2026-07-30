from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from calibration.legacy import func as legacy_func
from calibration.legacy.func import ExperimentConfig
from restoration.legacy.config import (
    LegacyRestoreConfigArtifact,
    LegacyRestoreConfigError,
    LegacyRestoreSolverSettings,
)
from restoration.legacy import restore as legacy_restore


class LegacyRestoreRunnerTests(unittest.TestCase):
    def _artifact(self, mode: str) -> LegacyRestoreConfigArtifact:
        return LegacyRestoreConfigArtifact(
            config=ExperimentConfig(),
            solver=LegacyRestoreSolverSettings(
                regularization_type=2,
                regularization_alpha=0.125,
                w_critical=0.002,
            ),
            run_mode=mode,
        )

    def test_selects_combined_runner_and_applies_solver_globals(self) -> None:
        original = (
            legacy_func.REGULARIZATION_TYPE,
            legacy_func.REGULARIZATION_ALPHA,
            legacy_func.W_CRITICAL,
        )
        observed: list[tuple[object, ...]] = []

        def inspect_globals(config: ExperimentConfig) -> None:
            observed.append((
                config,
                legacy_func.REGULARIZATION_TYPE,
                legacy_func.REGULARIZATION_ALPHA,
                legacy_func.W_CRITICAL,
            ))

        artifact = self._artifact("camera_line")
        with (
            patch.object(legacy_restore, "run_cfg", side_effect=inspect_globals) as combined,
            patch.object(legacy_restore, "run_cfg_lin_cut") as camera,
        ):
            legacy_restore.run_legacy_restore(artifact)

        combined.assert_called_once_with(artifact.config)
        camera.assert_not_called()
        self.assertEqual(observed[0], (artifact.config, 2, 0.125, 0.002))
        self.assertEqual(
            (
                legacy_func.REGULARIZATION_TYPE,
                legacy_func.REGULARIZATION_ALPHA,
                legacy_func.W_CRITICAL,
            ),
            original,
        )

    def test_selects_camera_runner(self) -> None:
        artifact = self._artifact("camera")
        with (
            patch.object(legacy_restore, "run_cfg") as combined,
            patch.object(legacy_restore, "run_cfg_lin_cut") as camera,
        ):
            legacy_restore.run_legacy_restore(artifact)
        camera.assert_called_once_with(artifact.config)
        combined.assert_not_called()

    def test_solver_globals_are_restored_after_failure(self) -> None:
        original = (
            legacy_func.REGULARIZATION_TYPE,
            legacy_func.REGULARIZATION_ALPHA,
            legacy_func.W_CRITICAL,
        )
        with patch.object(
            legacy_restore, "run_cfg_lin_cut", side_effect=RuntimeError("failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                legacy_restore.run_legacy_restore(self._artifact("camera"))
        self.assertEqual(
            (
                legacy_func.REGULARIZATION_TYPE,
                legacy_func.REGULARIZATION_ALPHA,
                legacy_func.W_CRITICAL,
            ),
            original,
        )

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(LegacyRestoreConfigError, "Неизвестный режим"):
            legacy_restore.run_legacy_restore(self._artifact("unknown"))
