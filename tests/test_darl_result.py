from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from darl.result import (
    DarlResultError,
    collect_darl_result,
    load_darl_result,
    save_darl_result,
)


class DarlResultTests(unittest.TestCase):
    def _artifacts(self, root: Path) -> None:
        (root / "matrix-case.npz").write_bytes(b"matrix")
        (root / "bins_front_detector.txt").write_text("bins")
        (root / "FrontDetectorLogLine_detector.txt").write_text("line")
        (root / "particle_classes_lasser_0.txt").write_text("classes")
        (root / "background_signal_laser_0.txt").write_text("background")
        for name in ("pinhole_200", "sample"):
            directory = root / name
            directory.mkdir()
            (directory / "modeled_signal.txt").write_text("signal")
            (directory / "plot.png").write_bytes(b"plot")

    def test_collect_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._artifacts(root)
            result = collect_darl_result(
                root,
                legacy_config_name="auto_test",
                detector_names=("camera", "line_sensor"),
                signal_value_type="signal",
                distribution_names=("pinhole_200", "sample"),
            )
            save_darl_result(result, root / "result.yaml")
            self.assertEqual(load_darl_result(root / "result.yaml"), result)
            self.assertEqual(result.matrix_file, "matrix-case.npz")
            self.assertEqual(result.detectors, ("camera", "line_sensor"))
            self.assertEqual(result.signal.value_type, "signal")
            self.assertEqual(
                result.particle_classes_file,
                "particle_classes_lasser_0.txt",
            )
            self.assertNotIn(
                "particle_classes_lasser_0.txt", result.detector_bin_files
            )
            self.assertEqual(
                tuple(item.name for item in result.distributions),
                ("pinhole_200", "sample"),
            )

    def test_missing_mandatory_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._artifacts(root)
            (root / "sample/modeled_signal.txt").unlink()
            with self.assertRaisesRegex(DarlResultError, "sample"):
                collect_darl_result(
                    root,
                    legacy_config_name="auto_test",
                    detector_names=("camera", "line_sensor"),
                    signal_value_type="signal",
                    distribution_names=("pinhole_200", "sample"),
                )

    def test_detector_bin_files_must_match_enabled_detectors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._artifacts(root)
            (root / "FrontDetectorLogLine_detector.txt").unlink()
            with self.assertRaisesRegex(
                DarlResultError, "FrontDetectorLogLine_detector.txt"
            ):
                collect_darl_result(
                    root,
                    legacy_config_name="auto_test",
                    detector_names=("camera", "line_sensor"),
                    signal_value_type="signal",
                    distribution_names=("pinhole_200", "sample"),
                )

            (root / "FrontDetectorLogLine_detector.txt").write_text("line")
            with self.assertRaisesRegex(DarlResultError, "не ожидались"):
                collect_darl_result(
                    root,
                    legacy_config_name="auto_test",
                    detector_names=("camera",),
                    signal_value_type="signal",
                    distribution_names=("pinhole_200", "sample"),
                )

    def test_unknown_result_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.yaml"
            path.write_text(
                "schema_version: 1\nlegacy_config_name: a\n"
                "detectors: [camera]\nsignal: {value_type: signal}\n"
                "matrix_file: m\n"
                "particle_classes_file: classes.txt\n"
                "detector_bin_files: []\nbackground_signal_files: []\n"
                "distributions: {}\nunknown: true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DarlResultError, "unknown"):
                load_darl_result(path)

    def test_loaded_manifest_requires_known_detector_bins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.yaml"
            base = (
                "schema_version: 1\nlegacy_config_name: a\n"
                "detectors: [camera]\nsignal: {value_type: signal}\n"
                "matrix_file: m\n"
                "particle_classes_file: classes.txt\n"
                "background_signal_files: [background.txt]\n"
                "distributions: {}\n"
            )
            path.write_text(
                base + "detector_bin_files: []\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                DarlResultError, "bins_front_detector.txt"
            ):
                load_darl_result(path)

            path.write_text(
                base + "detector_bin_files: [unknown.txt]\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DarlResultError, "unknown.txt"):
                load_darl_result(path)

    def test_signal_contract_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._artifacts(root)
            with self.assertRaisesRegex(DarlResultError, "signal.value_type"):
                collect_darl_result(
                    root,
                    legacy_config_name="auto_test",
                    detector_names=("camera", "line_sensor"),
                    signal_value_type="unknown",
                    distribution_names=("pinhole_200", "sample"),
                )
