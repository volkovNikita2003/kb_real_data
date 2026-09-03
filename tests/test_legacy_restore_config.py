from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from calibration.refactoring.result import (
    CalibrationResult,
    CameraCalibrationResult,
    LineSensorCalibrationResult,
)
from darl.result import (
    DarlDistributionResult,
    DarlResult,
    DarlSignalContract,
)
from experiment import Experiment
from parameters import load_general_parameters, load_restore_parameters
from restoration.legacy.config import (
    LegacyRestoreConfigError,
    build_legacy_restore_config,
)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


class LegacyRestoreConfigTests(unittest.TestCase):
    def _inputs(self, root: Path, *, with_line: bool = True):
        experiment_path = root / "experiment"
        (experiment_path / "data/calibration").mkdir(parents=True)
        measurement_path = experiment_path / "data/kmk_15"
        for name in ("cam", "cam_back", "lin", "lin_back"):
            (measurement_path / name).mkdir(parents=True)

        detectors: dict[str, object] = {
            "camera": {"width_px": 2592, "height_px": 1944},
        }
        restore_detectors: dict[str, object] = {
            "camera": {
                "use_background": True,
                "hdr": {
                    "mode": "l2h",
                    "difference_mode": "per_exposure",
                    "background_level": 12,
                    "low_threshold": 10,
                    "top_threshold": 240,
                    "filtered": True,
                    "gaussian_sigma": 5,
                },
            },
        }
        darl_detectors = ["camera"]
        detector_bins = ["bins_front_detector.txt"]
        if with_line:
            detectors["line_sensor"] = {"pixel_count": 3643}
            restore_detectors["line_sensor"] = {
                "use_background": True,
                "signal_mode": 2,
                "time_offset_us": 2,
            }
            darl_detectors.append("line_sensor")
            detector_bins.append("FrontDetectorLogLine_detector.txt")

        parameters = experiment_path / "input_parameters"
        dump(
            parameters / "general.yaml",
            {
                "schema_version": 1,
                "detectors": detectors,
                "instrument": {
                    "detector_position": "new",
                    "wavelength_um": 0.633,
                    "focal_length_um": 400000,
                },
            },
        )
        dump(
            parameters / "restore_profiles/default.yaml",
            {
                "schema_version": 1,
                "detectors": restore_detectors,
                "solver": {
                    "regularization_order": 2,
                    "regularization_alpha": 0.25,
                    "use_w_critical": True,
                    "w_critical": 0.002,
                    "use_chahine": False,
                    "use_concentration_correction": True,
                },
                "class_slice": {"drop_first": 10, "drop_last": 100},
            },
        )
        experiment = Experiment.open(experiment_path)
        general = load_general_parameters(experiment.general_parameters_file)
        restore = load_restore_parameters(
            experiment.restore_profiles_dir / "default.yaml"
        )
        line_result = None
        if with_line:
            line_result = LineSensorCalibrationResult(
                start_angle_deg=1.0,
                end_angle_deg=10.0,
                pixel_width_m=8.098304067336103e-6,
                pixel_height_m=0.0002024576016834026,
                to_camera_coefficient=0.31081315123650427,
                shift_m=0.02034698896918196,
                peak_pixel=1751.9944216945232,
            )
        calibration = CalibrationResult(
            schema_version=1,
            camera=CameraCalibrationResult(
                matrix_diagonal_mm=6.350737019518796,
                x_shift_m=0.00017178838009236094,
                y_shift_m=4.903750079380921e-05,
                pixel_width_m=1.911283425948221e-6,
            ),
            line_sensor=line_result,
        )
        distributions = (
            DarlDistributionResult(
                name="kmk_15",
                modeled_signal="kmk_15/modeled_signal.txt",
                artifacts=("kmk_15/modeled_signal.txt",),
            ),
        )
        darl = DarlResult(
            schema_version=1,
            legacy_config_name="auto_experiment",
            detectors=tuple(darl_detectors),
            signal=DarlSignalContract("signal"),
            matrix_file="matrix-case.npz",
            particle_classes_file="particle_classes_lasser_0.txt",
            detector_bin_files=tuple(detector_bins),
            background_signal_files=("background_signal_laser_0.txt",),
            distributions=distributions,
        )
        output = experiment.restore_output_root / "default/.kmk_15.tmp-test"
        return experiment, general, restore, calibration, darl, output

    def test_camera_and_line_mapping_matches_legacy_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._inputs(Path(directory))
            experiment, general, restore, calibration, darl, output = inputs

            artifact = build_legacy_restore_config(
                experiment=experiment,
                measurement=experiment.measurement("kmk_15"),
                general=general,
                restore=restore,
                calibration=calibration,
                darl=darl,
                output_dir=output,
                camera_exposures_us=(1600000, 24, 400000),
                line_exposure_us=400,
            )
            cfg = artifact.config

            self.assertEqual(artifact.run_mode, "camera_line")
            self.assertEqual(cfg.exposure_time_arr, [24, 400000, 1600000])
            self.assertEqual(cfg.dir_signal_rel, Path("data/kmk_15/cam"))
            self.assertEqual(cfg.dir_back_rel, Path("data/kmk_15/cam_back"))
            self.assertEqual(cfg.matrix_name, Path("output/darl/matrix-case.npz"))
            self.assertEqual(
                cfg.path_signal_darl_rel,
                Path("output/darl/kmk_15/modeled_signal.txt"),
            )
            self.assertEqual(cfg.detector_configuration_type, 1)
            self.assertEqual(cfg.cam_hdr_diff_mode, "per_exposure")
            self.assertTrue(cfg.cam_hdr_filtered)
            self.assertEqual(cfg.cam_pixel_width_m, 1.911283425948221e-6)
            self.assertEqual(cfg.filename_lin_template, "{}.txt")
            self.assertEqual(cfg.exposure_time_us_lin_arr, [400])
            self.assertEqual(cfg.width_pix_x_m, 8.098304067336103e-6)
            self.assertEqual(cfg.width_pix_y_m, 0.0002024576016834026)
            self.assertEqual(cfg.coef_lin_to_cam, 0.31081315123650427)
            self.assertEqual(cfg.lin_time_add, 2.0)
            self.assertEqual(cfg.cut_classes, 10)
            self.assertEqual(cfg.cut_classes_top, 100)
            self.assertFalse(cfg.use_chahine)
            self.assertTrue(cfg.use_w_critical)
            self.assertFalse(cfg.forward_modeling_enabled)
            self.assertIsNone(cfg.signal_vector_file_rel)
            self.assertEqual(artifact.solver.regularization_type, 2)
            self.assertEqual(artifact.solver.regularization_alpha, 0.25)
            self.assertEqual(artifact.solver.w_critical, 0.002)

    def test_camera_only_selects_row_cut_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._inputs(Path(directory), with_line=False)
            experiment, general, restore, calibration, darl, output = inputs
            darl = replace(darl, distributions=())

            artifact = build_legacy_restore_config(
                experiment=experiment,
                measurement=experiment.measurement("kmk_15"),
                general=general,
                restore=restore,
                calibration=calibration,
                darl=darl,
                output_dir=output,
                camera_exposures_us=(100,),
                line_exposure_us=None,
            )

            self.assertEqual(artifact.run_mode, "camera")
            self.assertIsNone(artifact.config.bins_lin_name)
            self.assertIsNone(artifact.config.path_signal_darl_rel)

    def test_rejects_darl_path_outside_result_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._inputs(Path(directory), with_line=False)
            experiment, general, restore, calibration, darl, output = inputs
            darl = replace(darl, matrix_file="../../data/secret.npz")

            with self.assertRaisesRegex(
                LegacyRestoreConfigError, "выходит за пределы output/darl"
            ):
                build_legacy_restore_config(
                    experiment=experiment,
                    measurement=experiment.measurement("kmk_15"),
                    general=general,
                    restore=restore,
                    calibration=calibration,
                    darl=darl,
                    output_dir=output,
                    camera_exposures_us=(100,),
                    line_exposure_us=None,
                )
