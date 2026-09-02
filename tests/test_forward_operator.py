from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from experiment import Experiment
from calibration.refactoring.result import CalibrationResult, CameraCalibrationResult
from parameters import (
    RestoreOperatorParameters,
    load_general_parameters,
    load_restore_parameters,
)
from restoration.operator import ForwardOperatorError, resolve_operator
from tests.test_validation import create_experiment
from validation import validate_restore_inputs


def write_manifest(experiment_path: Path) -> Path:
    bundle = experiment_path / "input_artifacts/operators/imported"
    bundle.mkdir(parents=True)
    for name in ("matrix.npz", "classes.txt", "camera-bins.txt"):
        (bundle / name).write_bytes(b"test")
    manifest = bundle / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({
        "schema_version": 1,
        "detectors": ["camera"],
        "signal": {"value_type": "signal"},
        "files": {
            "matrix": "matrix.npz",
            "particle_classes": "classes.txt",
            "detector_bins": {"camera": "camera-bins.txt"},
        },
    }, sort_keys=False), encoding="utf-8")
    return manifest


class ForwardOperatorTests(unittest.TestCase):
    def test_restore_profile_defaults_to_darl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment.open(create_experiment(Path(directory)))
            parameters = load_restore_parameters(
                experiment.restore_profile("default").path
            )
            self.assertEqual(parameters.operator.source, "darl")
            self.assertIsNone(parameters.operator.manifest)

    def test_file_source_requires_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment.open(create_experiment(Path(directory)))
            profile = experiment.restore_profile("default").path
            profile.write_text(
                "schema_version: 1\noperator:\n  source: file\n"
                "detectors:\n  camera: {}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Exception, "manifest"):
                load_restore_parameters(profile)

    def test_manifest_resolves_without_darl_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment.open(create_experiment(Path(directory)))
            manifest = write_manifest(experiment.path)
            operator = resolve_operator(
                experiment,
                RestoreOperatorParameters(
                    "file", manifest.relative_to(experiment.path).as_posix()
                ),
                darl=None,
            )
            self.assertEqual(operator.source, "file")
            self.assertEqual(operator.name, "imported")
            self.assertEqual(operator.detectors, ("camera",))
            self.assertEqual(operator.matrix_file, manifest.parent / "matrix.npz")
            self.assertIn("matrix", operator.to_dict(experiment)["sha256"])

    def test_bundle_artifact_cannot_escape_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment.open(create_experiment(Path(directory)))
            manifest = write_manifest(experiment.path)
            value = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            value["files"]["matrix"] = "../secret.npz"
            manifest.write_text(
                yaml.safe_dump(value, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ForwardOperatorError, "пределы"):
                resolve_operator(
                    experiment,
                    RestoreOperatorParameters(
                        "file", manifest.relative_to(experiment.path).as_posix()
                    ),
                    darl=None,
                )

    def test_external_matrix_dimensions_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment.open(create_experiment(Path(directory)))
            manifest = write_manifest(experiment.path)
            np.savez(manifest.parent / "matrix.npz", matrix=np.ones((2, 1)))
            (manifest.parent / "classes.txt").write_text(
                "0 1 2\n", encoding="utf-8"
            )
            (manifest.parent / "camera-bins.txt").write_text(
                "0 1\n", encoding="utf-8"
            )
            profile = experiment.restore_profile("default")
            restore = load_restore_parameters(profile.path)
            operator = resolve_operator(
                experiment,
                RestoreOperatorParameters(
                    "file", manifest.relative_to(experiment.path).as_posix()
                ),
                darl=None,
            )
            report = validate_restore_inputs(
                experiment,
                measurement=experiment.measurement("sample"),
                profile=profile,
                general=load_general_parameters(experiment.general_parameters_file),
                restore=restore,
                calibration=CalibrationResult(
                    1, CameraCalibrationResult(6.3, 0, 0, 1.9e-6), None
                ),
                operator=operator,
            )
            self.assertIn(
                "invalid_operator_artifacts",
                {issue.code for issue in report.errors},
            )


if __name__ == "__main__":
    unittest.main()
