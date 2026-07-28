from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from errors import OutputError
from experiment import Experiment
from output import OutputDirectory, archive_timestamp, prepare_output_directory


def create_experiment(root: Path) -> Experiment:
    path = root / "test_experiment"
    (path / "data/calibration").mkdir(parents=True)
    (path / "input_parameters").mkdir()
    return Experiment.open(path)


class OutputDirectoryTests(unittest.TestCase):
    def test_success_publishes_complete_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_experiment(Path(directory))
            target = experiment.calibration_output_dir

            with prepare_output_directory(experiment, target) as working:
                self.assertNotEqual(working, target)
                (working / "result.yaml").write_text("ok\n", encoding="utf-8")
                self.assertFalse(target.exists())

            self.assertEqual(
                (target / "result.yaml").read_text(encoding="utf-8"), "ok\n"
            )

    def test_existing_result_is_not_overwritten_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_experiment(Path(directory))
            target = experiment.matrix_output_dir
            target.mkdir(parents=True)
            (target / "old.txt").write_text("old", encoding="utf-8")

            with self.assertRaisesRegex(OutputError, "--force"):
                with prepare_output_directory(experiment, target):
                    pass
            self.assertEqual((target / "old.txt").read_text(), "old")

    def test_force_archives_old_result_and_publishes_new_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_experiment(Path(directory))
            target = experiment.calibration_output_dir
            target.mkdir(parents=True)
            (target / "value.txt").write_text("old", encoding="utf-8")
            moment = datetime(2026, 7, 24, 12, 30, 45, 120450)

            with OutputDirectory(
                experiment, target, force=True, clock=lambda: moment
            ) as working:
                (working / "value.txt").write_text("new", encoding="utf-8")

            archived = (
                experiment.archive_dir
                / "calibration"
                / "2026-07-24T12-30-45.120450"
            )
            self.assertEqual((target / "value.txt").read_text(), "new")
            self.assertEqual((archived / "value.txt").read_text(), "old")

    def test_archive_mirrors_nested_result_structure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_experiment(Path(directory))
            target = experiment.output_dir / "restore/default/kmk_15"
            target.mkdir(parents=True)
            (target / "old.txt").write_text("old", encoding="utf-8")
            moment = datetime(2026, 1, 2, 3, 4, 5, 6)

            with OutputDirectory(
                experiment, target, force=True, clock=lambda: moment
            ) as working:
                (working / "new.txt").write_text("new", encoding="utf-8")

            archived = experiment.archive_dir / (
                "restore/default/kmk_15/2026-01-02T03-04-05.000006"
            )
            self.assertTrue((archived / "old.txt").is_file())

    def test_exception_removes_temporary_and_preserves_old_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_experiment(Path(directory))
            target = experiment.calibration_output_dir
            target.mkdir(parents=True)
            (target / "old.txt").write_text("old", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "calculation failed"):
                with prepare_output_directory(
                    experiment, target, force=True
                ) as working:
                    (working / "partial.txt").write_text("partial")
                    raise RuntimeError("calculation failed")

            self.assertEqual((target / "old.txt").read_text(), "old")
            self.assertFalse(experiment.archive_dir.exists())
            self.assertEqual(
                list(target.parent.glob(f".{target.name}.tmp-*")), []
            )

    def test_rejects_target_outside_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_experiment(Path(directory))
            with self.assertRaisesRegex(OutputError, "только путь внутри output"):
                prepare_output_directory(experiment, experiment.data_dir / "result")

    def test_rejects_archive_as_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_experiment(Path(directory))
            with self.assertRaisesRegex(OutputError, "сам архив"):
                prepare_output_directory(
                    experiment, experiment.archive_dir / "calibration"
                )

    def test_force_rejects_file_as_existing_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_experiment(Path(directory))
            target = experiment.calibration_output_dir
            target.parent.mkdir(parents=True)
            target.write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(OutputError, "обычной директорией"):
                with prepare_output_directory(experiment, target, force=True):
                    pass

    def test_timestamp_has_fixed_microsecond_precision(self) -> None:
        self.assertEqual(
            archive_timestamp(datetime(2026, 1, 2, 3, 4, 5, 6)),
            "2026-01-02T03-04-05.000006",
        )


if __name__ == "__main__":
    unittest.main()
