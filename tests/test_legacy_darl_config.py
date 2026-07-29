from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
CODE_GIT_DIR = PROJECT_DIR.parent / "code_git"
sys.path.insert(0, str(SRC_DIR))

from calibration.refactoring.result import (
    CalibrationResult,
    CameraCalibrationResult,
    LineSensorCalibrationResult,
)
from darl.legacy.config import (
    LegacyDarlConfigError,
    build_legacy_config,
    legacy_config_name,
    render_legacy_config,
    write_legacy_config,
)
from experiment import Experiment
from parameters import (
    DarlStageParameters,
    load_darl_parameters,
    load_general_parameters,
)


def reference_stage_parameters() -> DarlStageParameters:
    inputs = PROJECT_DIR / "experiments/test_17_07_26_kmk_15/input_parameters"
    general = load_general_parameters(inputs / "general.yaml")
    darl = load_darl_parameters(inputs / "darl.yaml")
    calibration = CalibrationResult(
        1,
        CameraCalibrationResult(
            6.192558300072236,
            0.00017178838009236094,
            4.903750079380921e-05,
            1.911283425948221e-06,
        ),
        LineSensorCalibrationResult(
            0.8821141906204397,
            5.093431705868666,
            8.098304067336103e-06,
            0.0002024576016834026,
            0.31081315123650427,
            0.02034698896918196,
            1751.9944216945232,
        ),
    )
    result = DarlStageParameters(general, darl, calibration, ())
    result.validate_consistency()
    return result


def parse_with_code_git(text: str) -> dict[str, object]:
    script = (
        "import json, sys\n"
        "from logic.config import parse_config_text\n"
        "print(json.dumps(parse_config_text(sys.stdin.read()), "
        "ensure_ascii=False, sort_keys=True))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=CODE_GIT_DIR,
        input=text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(completed.stdout)


def create_experiment(root: Path, name: str = "test_case") -> Experiment:
    path = root / name
    (path / "data/calibration").mkdir(parents=True)
    (path / "input_parameters").mkdir()
    return Experiment.open(path)


class LegacyDarlConfigTests(unittest.TestCase):
    def test_reference_parameters_render_same_values_as_reference_config(self) -> None:
        if not CODE_GIT_DIR.is_dir():
            self.skipTest("code_git checkout is not available")
        reference_path = (
            CODE_GIT_DIR
            / "data/configs/conf_real_data_test_17_07_26-kmk-15-cam-lin-signal.txt"
        )
        if not reference_path.is_file():
            self.skipTest("reference DARL config is not available")

        generated = parse_with_code_git(
            render_legacy_config(reference_stage_parameters())
        )
        reference = parse_with_code_git(reference_path.read_text(encoding="utf-8"))

        self.assertEqual(generated, reference)

    def test_render_preserves_critical_unit_conversions(self) -> None:
        text = render_legacy_config(reference_stage_parameters())

        self.assertIn("длина_волны_(нм): 633", text)
        self.assertIn(
            "отступ_от_оптической_оси_(мкм): 171.78838009236094",
            text,
        )
        self.assertIn(
            "отступ_от_оптической_оси_Y(мкм): 49.037500793809215",
            text,
        )
        self.assertIn(
            "Ширина_пикселя_(мкм): 202.4576016834026",
            text,
        )
        self.assertIn("Размен_классов_в_Ми(нм): 300", text)

    def test_write_publishes_exact_text_and_controls_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code_git = Path(directory) / "code_git"
            (code_git / "data/configs").mkdir(parents=True)

            artifact = write_legacy_config(
                code_git,
                "auto_test",
                "Лазер:\nэтап: 1\n",
            )

            self.assertEqual(artifact.config_name, "auto_test")
            self.assertEqual(
                artifact.path,
                code_git / "data/configs/auto_test.txt",
            )
            self.assertEqual(artifact.path.read_text(encoding="utf-8"), artifact.text)
            with self.assertRaisesRegex(LegacyDarlConfigError, "уже существует"):
                write_legacy_config(
                    code_git,
                    "auto_test",
                    "new",
                    overwrite=False,
                )
            write_legacy_config(code_git, "auto_test", "new", overwrite=True)
            self.assertEqual(artifact.path.read_text(encoding="utf-8"), "new")

    def test_rejects_missing_layout_unsafe_name_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(LegacyDarlConfigError, "не найдена"):
                write_legacy_config(root / "missing", "auto_test", "value")

            code_git = root / "code_git"
            configs = code_git / "data/configs"
            configs.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "Некорректное имя"):
                write_legacy_config(code_git, "../unsafe", "value")

            target = configs / "auto_link.txt"
            target.symlink_to(configs / "actual.txt")
            with self.assertRaisesRegex(LegacyDarlConfigError, "ссылкой"):
                write_legacy_config(code_git, "auto_link", "value")

    def test_build_uses_deterministic_experiment_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = create_experiment(root, "sample-15")
            code_git = root / "code_git"
            (code_git / "data/configs").mkdir(parents=True)

            artifact = build_legacy_config(
                experiment,
                reference_stage_parameters(),
                code_git_dir=code_git,
            )

            self.assertEqual(legacy_config_name(experiment), "auto_sample-15")
            self.assertEqual(artifact.config_name, "auto_sample-15")
            self.assertTrue(artifact.path.is_file())


if __name__ == "__main__":
    unittest.main()
