from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from errors import ExperimentStructureError
from experiment import Experiment, validate_safe_name


def create_experiment(
    root: Path,
    name: str = "test_experiment",
    *,
    with_output: bool = True,
) -> Path:
    experiment = root / name
    (experiment / "data/calibration").mkdir(parents=True)
    (experiment / "input_parameters").mkdir()
    if with_output:
        (experiment / "output").mkdir()
    return experiment


class ExperimentTests(unittest.TestCase):
    def test_open_and_conventional_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            experiment = Experiment.open(path)

            self.assertEqual(experiment.name, "test_experiment")
            self.assertEqual(experiment.data_dir, path / "data")
            self.assertEqual(
                experiment.calibration_camera_dir,
                path / "data/calibration/cam",
            )
            self.assertEqual(
                experiment.general_parameters_file,
                path / "input_parameters/general.yaml",
            )
            self.assertEqual(
                experiment.calibration_output_dir,
                path / "output/calibration",
            )
            self.assertEqual(experiment.archive_dir, path / "output/archive")

    def test_measurements_are_sorted_and_exclude_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            (path / "data/kmk_270").mkdir()
            (path / "data/kmk_15").mkdir()

            experiment = Experiment.open(path)

            self.assertEqual(experiment.measurement_names(), ("kmk_15", "kmk_270"))
            measurement = experiment.measurement("kmk_15")
            self.assertEqual(measurement.camera_dir, path / "data/kmk_15/cam")
            self.assertEqual(
                measurement.line_background_dir,
                path / "data/kmk_15/lin_back",
            )

    def test_restore_profiles_are_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            profiles = path / "input_parameters/restore_profiles"
            profiles.mkdir()
            (profiles / "strong_regularization.yaml").touch()
            (profiles / "default.yaml").touch()

            experiment = Experiment.open(path)

            self.assertEqual(
                experiment.restore_profile_names(),
                ("default", "strong_regularization"),
            )
            self.assertEqual(
                experiment.restore_profile("default").path,
                profiles / "default.yaml",
            )

    def test_absent_restore_profiles_directory_means_no_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment.open(create_experiment(Path(directory)))

            self.assertEqual(experiment.restore_profiles(), ())

    def test_restore_output_path_uses_profile_and_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            (path / "data/kmk_15").mkdir()
            profiles = path / "input_parameters/restore_profiles"
            profiles.mkdir()
            (profiles / "default.yaml").touch()
            experiment = Experiment.open(path)

            self.assertEqual(
                experiment.restore_output_dir("default", "kmk_15"),
                path / "output/restore/default/kmk_15",
            )

    def test_restore_output_path_requires_existing_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            (path / "data/kmk_15").mkdir()
            experiment = Experiment.open(path)

            with self.assertRaisesRegex(
                ExperimentStructureError,
                "Профиль восстановления 'default' не найден",
            ):
                experiment.restore_output_dir("default", "kmk_15")

    def test_restore_output_path_requires_existing_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            profiles = path / "input_parameters/restore_profiles"
            profiles.mkdir()
            (profiles / "default.yaml").touch()
            experiment = Experiment.open(path)
    
            with self.assertRaisesRegex(
                ExperimentStructureError,
                "Измерение 'kmk_15' не найдено",
            ):
                experiment.restore_output_dir("default", "kmk_15")

    def test_output_directory_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory), with_output=False)
            experiment = Experiment.open(path)

            self.assertEqual(experiment.output_dir, path / "output")
            self.assertFalse(experiment.output_dir.exists())

    def test_missing_required_root_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            (path / "input_parameters").rmdir()

            with self.assertRaisesRegex(
                ExperimentStructureError,
                "отсутствуют обязательные директории: input_parameters",
            ):
                Experiment.open(path)

    def test_output_file_instead_of_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory), with_output=False)
            (path / "output").touch()

            with self.assertRaisesRegex(
                ExperimentStructureError,
                "должны быть директориями: output",
            ):
                Experiment.open(path)

    def test_unexpected_root_entry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            (path / "notes.txt").touch()

            with self.assertRaisesRegex(
                ExperimentStructureError,
                "посторонние элементы: notes.txt",
            ):
                Experiment.open(path)

    def test_file_in_data_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            (path / "data/notes.txt").touch()

            with self.assertRaisesRegex(
                ExperimentStructureError,
                "В data/ разрешены только директории",
            ):
                Experiment.open(path)

    def test_unsafe_measurement_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            (path / "data/bad name").mkdir()

            with self.assertRaisesRegex(
                ExperimentStructureError,
                "Некорректное имя измерения",
            ):
                Experiment.open(path)

    def test_non_yaml_restore_profile_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment(Path(directory))
            profiles = path / "input_parameters/restore_profiles"
            profiles.mkdir()
            (profiles / "default.txt").touch()
            experiment = Experiment.open(path)

            with self.assertRaisesRegex(
                ExperimentStructureError,
                "разрешены только YAML-файлы",
            ):
                experiment.restore_profiles()

    def test_missing_measurement_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment.open(create_experiment(Path(directory)))

            with self.assertRaisesRegex(
                ExperimentStructureError,
                "Измерение 'kmk_15' не найдено",
            ):
                experiment.measurement("kmk_15")

    def test_safe_name_rules(self) -> None:
        for valid in ("a", "17_07_26", "camera-only", "Kmk_15"):
            with self.subTest(valid=valid):
                self.assertEqual(
                    validate_safe_name(valid, kind="теста"),
                    valid,
                )

        for invalid in ("", "_hidden", "-hidden", "bad name", "кмк_15", "../x"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ExperimentStructureError):
                    validate_safe_name(invalid, kind="теста")


if __name__ == "__main__":
    unittest.main()
