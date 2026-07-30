from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import yaml


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from restoration.result import (
    RestoreResultError,
    collect_restore_result,
    load_restore_result,
    save_restore_result,
)


class RestoreResultTests(unittest.TestCase):
    def _artifacts(
        self,
        root: Path,
        *,
        line: bool,
        sliced: bool,
        comparison: bool,
    ) -> None:
        for name in (
            "used-parameters.yaml",
            "params.txt",
            "reference_camera_signal.txt",
            "reference_restoration.txt",
            "reference_gcv_curve.txt",
            "reference_alpha.txt",
        ):
            (root / name).write_text(name, encoding="utf-8")
        if line:
            (root / "reference_line_signal.txt").write_text("line")
            (root / "reference_combined_signal.txt").write_text("combined")
        if sliced:
            for suffix in (
                "reference_restoration.txt",
                "reference_gcv_curve.txt",
                "reference_alpha.txt",
            ):
                (root / f"cutted_10_{suffix}").write_text("sliced")
        if comparison:
            (root / "compare-real-darl-signals-all-ex_time_100.txt").write_text(
                "comparison"
            )
        figures = root / "signal-img"
        figures.mkdir()
        (figures / "signal.png").write_bytes(b"png")

    def test_collect_save_and_load_combined_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._artifacts(root, line=True, sliced=True, comparison=True)

            result = collect_restore_result(
                root,
                restore_profile="default",
                measurement="kmk_15",
                detector_names=("line_sensor", "camera"),
                class_slice_prefix="cutted_10_",
                expected_signal_comparison=True,
            )
            save_restore_result(result, root / "result.yaml")
            loaded = load_restore_result(root / "result.yaml")

            self.assertEqual(loaded, result)
            self.assertEqual(result.detectors, ("camera", "line_sensor"))
            self.assertEqual(
                result.signals.line_sensor, "reference_line_signal.txt"
            )
            self.assertEqual(
                result.sliced_solution.restoration,
                "cutted_10_reference_restoration.txt",
            )
            self.assertEqual(
                result.expected_signal_comparison,
                "compare-real-darl-signals-all-ex_time_100.txt",
            )
            self.assertIn("signal-img/signal.png", result.figures)
            self.assertNotIn("result.yaml", result.artifacts)

    def test_camera_only_result_rejects_line_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._artifacts(root, line=False, sliced=False, comparison=False)
            result = collect_restore_result(
                root,
                restore_profile="camera",
                measurement="sample",
                detector_names=("camera",),
                class_slice_prefix=None,
                expected_signal_comparison=False,
            )
            self.assertIsNone(result.signals.line_sensor)
            self.assertIsNone(result.signals.combined)
            self.assertIsNone(result.sliced_solution)

            (root / "reference_line_signal.txt").write_text("unexpected")
            with self.assertRaisesRegex(
                RestoreResultError, "signals.line_sensor"
            ):
                collect_restore_result(
                    root,
                    restore_profile="camera",
                    measurement="sample",
                    detector_names=("camera",),
                    class_slice_prefix=None,
                    expected_signal_comparison=False,
                )

    def test_missing_mandatory_solution_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._artifacts(root, line=False, sliced=False, comparison=False)
            (root / "reference_alpha.txt").unlink()
            with self.assertRaisesRegex(RestoreResultError, "solution.alpha"):
                collect_restore_result(
                    root,
                    restore_profile="default",
                    measurement="sample",
                    detector_names=("camera",),
                    class_slice_prefix=None,
                    expected_signal_comparison=False,
                )

    def test_unrequested_sliced_solution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._artifacts(root, line=False, sliced=True, comparison=False)
            with self.assertRaisesRegex(RestoreResultError, "sliced_solution"):
                collect_restore_result(
                    root,
                    restore_profile="default",
                    measurement="sample",
                    detector_names=("camera",),
                    class_slice_prefix=None,
                    expected_signal_comparison=False,
                )

    def test_loader_rejects_unsafe_paths_and_detector_signal_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._artifacts(root, line=False, sliced=False, comparison=False)
            result = collect_restore_result(
                root,
                restore_profile="default",
                measurement="sample",
                detector_names=("camera",),
                class_slice_prefix=None,
                expected_signal_comparison=False,
            )
            path = root / "result.yaml"
            value = result.to_dict()
            value["solution"]["restoration"] = "../outside.txt"
            path.write_text(yaml.safe_dump(value), encoding="utf-8")
            with self.assertRaisesRegex(RestoreResultError, "безопасный"):
                load_restore_result(path)

            value = result.to_dict()
            value["detectors"] = ["camera", "line_sensor"]
            path.write_text(yaml.safe_dump(value), encoding="utf-8")
            with self.assertRaisesRegex(RestoreResultError, "signals"):
                load_restore_result(path)

    def test_unknown_and_duplicate_yaml_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._artifacts(root, line=False, sliced=False, comparison=False)
            result = collect_restore_result(
                root,
                restore_profile="default",
                measurement="sample",
                detector_names=("camera",),
                class_slice_prefix=None,
                expected_signal_comparison=False,
            )
            path = root / "result.yaml"
            value = result.to_dict()
            value["unknown"] = True
            path.write_text(yaml.safe_dump(value), encoding="utf-8")
            with self.assertRaisesRegex(RestoreResultError, "unknown"):
                load_restore_result(path)

            document = yaml.safe_dump(result.to_dict())
            path.write_text(document + "schema_version: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(RestoreResultError, "duplicate"):
                load_restore_result(path)
