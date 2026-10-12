from dataclasses import replace
from pathlib import Path
import sys
import subprocess
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from errors import ParametersError
from experiment import Experiment
from parameters import load_general_parameters, load_restore_parameters, SideDetectorParameters, SideDetectorEnabled
from side import load_side_signal
from darl.legacy.config import render_legacy_config
from darl.result import collect_darl_result, save_darl_result, load_darl_result
from restoration.operator import from_darl
from tests import test_darl_result, test_restore_result
from tests.test_legacy_darl_config import reference_stage_parameters


class SideInputTests(unittest.TestCase):
    def test_single_column_and_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b.txt"
            path.write_text("1.25\n-2.5\n")
            np.testing.assert_array_equal(load_side_signal(path, 2), [1.25, -2.5])
            with self.assertRaisesRegex(ValueError, "числу углов"):
                load_side_signal(path, 3)
            for value in ("1 2\n", "# comment\n1\n", "1\n\n2\n"):
                path.write_text(value)
                with self.assertRaises(ValueError):
                    load_side_signal(path, 2)
            path.write_text("3.5\n")
            self.assertEqual(load_side_signal(path, 1).shape, (1,))

    def test_side_contract_includes_unselected_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_dir = root / "restore_profiles"
            profile_dir.mkdir()
            general = {"schema_version": 1, "instrument": {"detector_position": "new"}, "detectors": {"camera": {}, "line_sensor": {}, "side": {"pixel_width_m": .004, "angle_deg": [135., 110.]}}}
            path = root / "general.yaml"
            path.write_text(yaml.safe_dump(general))
            darl = root / "darl.yaml"
            darl.write_text(yaml.safe_dump({"detectors": {"camera": {}, "line_sensor": {}, "side": {}}}))
            profile = profile_dir / "default.yaml"
            profile.write_text(yaml.safe_dump({"schema_version": 1, "detectors": {"camera": {}, "line_sensor": {}, "side": {}}}))
            self.assertEqual(load_general_parameters(path).detectors.side.angle_deg, (135., 110.))
            self.assertEqual(load_restore_parameters(profile).detectors.names(), {"camera", "line_sensor", "side"})
            (profile_dir / "unused.yaml").write_text("detectors:\n  camera: {}\n")
            with self.assertRaisesRegex(ParametersError, "unused.yaml"):
                load_general_parameters(path)
            (profile_dir / "unused.yaml").unlink()
            del general["detectors"]["side"]
            path.write_text(yaml.safe_dump(general))
            with self.assertRaisesRegex(ParametersError, "side"):
                load_general_parameters(path)

    def test_render_units_and_angle_order(self):
        params = reference_stage_parameters()
        params = replace(params, general=replace(params.general, detectors=replace(params.general.detectors, side=SideDetectorParameters(.004, (135., 110.5)))), darl=replace(params.darl, detectors=replace(params.darl.detectors, side=SideDetectorEnabled())))
        params.validate_consistency()
        text = render_legacy_config(params)
        self.assertIn("линейный_размер_(мм): 4\n", text)
        self.assertIn("Положение_детекторов_(углы_в_градусах): 135, 110.5\n", text)
        self.assertIn("расстояние_до_кюветы_(см): 0\n", text)

    def test_darl_result_side_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_darl_result.DarlResultTests()._artifacts(root)
            (root / "side_detector_angles.txt").write_text("135\n110\n")
            result = collect_darl_result(root, legacy_config_name="auto_test", detector_names=("camera", "line_sensor", "side"), signal_value_type="signal", distribution_names=("pinhole_200", "sample"))
            save_darl_result(result, root / "result.yaml")
            self.assertEqual(load_darl_result(root / "result.yaml"), result)
            experiment = Experiment(root)
            self.assertEqual(from_darl(experiment, result).detector_bin_files["side"], root / "output/darl/side_detector_angles.txt")

    def test_code_git_accepts_zero_distance_and_single_detector(self):
        code_git = Path(__file__).resolve().parents[2] / "code_git"
        if not (code_git / "config/params.py").is_file():
            self.skipTest("local code_git is not available")
        params = reference_stage_parameters()
        for angles in ((135., 110.5), (135.,)):
            with self.subTest(angles=angles), tempfile.TemporaryDirectory() as directory:
                current = replace(params, general=replace(params.general, detectors=replace(params.general.detectors, side=SideDetectorParameters(.004, angles))), darl=replace(params.darl, detectors=replace(params.darl.detectors, side=SideDetectorEnabled())))
                config_dir = Path(directory)
                (config_dir / "side.txt").write_text(render_legacy_config(current))
                script = """import sys, json
from pathlib import Path
from logic.config import load_config_to_state
from logic.state import export_state_dict
from config.params import ParamsClass
import config.params as params_module
params_module.DATA_CASES_DIR = Path(sys.argv[1]) / "cases"
load_config_to_state(Path(sys.argv[1]), 'side')
p = ParamsClass(export_state_dict())
p.update_laser(p.LASER_NUM_ARR[0])
print(json.dumps({'angles': p.SIDE_DETECTOR_ANGLES_DEG.tolist(), 'size_cm': p.SIDE_DETECTOR_WIDTH_CM, 'file': p.FILENAME_SIDE_DETECTOR_ANGELS_COMSOL, 'enabled': p.SIDE_BACK_IS_ADDED}))
"""
                completed = subprocess.run([sys.executable, "-c", script, str(config_dir)], cwd=code_git, capture_output=True, text=True, check=False)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                data = json.loads(completed.stdout.splitlines()[-1])
                self.assertEqual(data["angles"], list(angles))
                self.assertAlmostEqual(data["size_cm"], .4)
                self.assertTrue(data["enabled"])
                self.assertTrue(data["file"].endswith("side_detectors_angles-distance_0cm.txt"))

    def test_restore_result_side_roundtrip(self):
        from restoration.result import collect_restore_result, save_restore_result, load_restore_result
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_restore_result.RestoreResultTests()._artifacts(root, line=True, sliced=False, comparison=False)
            (root / "reference_side_signal.txt").write_text("9\n8\n")
            result = collect_restore_result(root, restore_profile="default", measurement="sample", detector_names=("camera", "line_sensor", "side"), class_slice_prefix=None, expected_signal_comparison=False)
            save_restore_result(result, root / "result.yaml")
            self.assertEqual(load_restore_result(root / "result.yaml"), result)
            self.assertEqual(result.signals.side, "reference_side_signal.txt")

    def test_comsol_copied_before_modeling(self):
        import calc_darl
        params = reference_stage_parameters()
        params = replace(params, general=replace(params.general, detectors=replace(params.general.detectors, side=SideDetectorParameters(.004, (135., 110.5)))), darl=replace(params.darl, detectors=replace(params.darl.detectors, side=SideDetectorEnabled())))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            (root / "data/calibration").mkdir(parents=True)
            (root / "input_parameters").mkdir()
            experiment = Experiment.open(root)
            experiment.side_comsol_file.write_text("COMSOL table\n")
            code_git = Path(directory) / "code_git"
            (code_git / "data/configs").mkdir(parents=True)
            target = code_git / "side_detectors_angles-distance_0cm.txt"
            target.write_text("old table")

            def model(*args, **kwargs):
                self.assertEqual(target.read_bytes(), experiment.side_comsol_file.read_bytes())
                output = Path(kwargs["env"]["REAL_DATA_AUTO_DARL_RESULT_DIR"])
                for filename in ("matrix-case.npz", "particle_classes_lasser_0.txt", "bins_front_detector.txt", "FrontDetectorLogLine_detector.txt", "background_signal_laser_0.txt"):
                    (output / filename).write_text("artifact")
                return SimpleNamespace(returncode=0)

            with patch.object(calc_darl.subprocess, "run", side_effect=model):
                calc_darl.run(experiment, params, code_git_dir=code_git)
            result = load_darl_result(experiment.darl_result_file)
            self.assertEqual(result.detectors, ("camera", "line_sensor", "side"))
            self.assertEqual((experiment.darl_output_dir / "side_detector_angles.txt").read_text(), "135\n110.5\n")

    def test_imported_operator_counts_side_rows(self):
        from tests.test_validation import create_experiment
        from restoration.operator import resolve_operator
        from validation import validate_restore_inputs
        with tempfile.TemporaryDirectory() as directory:
            root = create_experiment(Path(directory), line_sensor=True)
            inputs = root / "input_parameters"
            for path in (inputs / "general.yaml", inputs / "darl.yaml", inputs / "restore_profiles/default.yaml"):
                data = yaml.safe_load(path.read_text())
                data["detectors"]["side"] = {"pixel_width_m": .004, "angle_deg": [135., 110.]} if path.name == "general.yaml" else {}
                if path.parent.name == "restore_profiles":
                    data["operator"] = {"source": "file", "manifest": "input_artifacts/operators/imported/manifest.yaml"}
                path.write_text(yaml.safe_dump(data))
            experiment = Experiment.open(root)
            measurement = experiment.measurement("sample")
            measurement.side_dir.mkdir()
            measurement.side_signal_file.write_text("9\n8\n")
            bundle = experiment.operators_dir / "imported"
            bundle.mkdir(parents=True)
            np.savez(bundle / "matrix.npz", matrix=np.ones((4, 1)))
            (bundle / "classes.txt").write_text("1 100 200\n")
            (bundle / "camera.txt").write_text("0 1\n")
            (bundle / "line.txt").write_text("header\n0 1\n")
            (bundle / "side.txt").write_text("135\n110\n")
            (bundle / "manifest.yaml").write_text(yaml.safe_dump({"schema_version": 1, "detectors": ["camera", "line_sensor", "side"], "signal": {"value_type": "signal"}, "files": {"matrix": "matrix.npz", "particle_classes": "classes.txt", "detector_bins": {"camera": "camera.txt", "line_sensor": "line.txt", "side": "side.txt"}}}))
            general = load_general_parameters(experiment.general_parameters_file)
            profile = experiment.restore_profile("default")
            restore = load_restore_parameters(profile.path)
            operator = resolve_operator(experiment, restore.operator, darl=None)
            calibration = reference_stage_parameters().calibration_result
            def validate():
                return validate_restore_inputs(experiment, measurement=measurement, profile=profile, general=general, restore=restore, calibration=calibration, operator=operator)
            self.assertFalse(validate().errors)
            (bundle / "side.txt").write_text("135\n")
            self.assertIn("invalid_operator_artifacts", [issue.code for issue in validate().errors])
            (bundle / "side.txt").write_text("135\n110\n")
            np.savez(bundle / "matrix.npz", matrix=np.ones((2, 1)))
            self.assertIn("invalid_operator_artifacts", [issue.code for issue in validate().errors])

    def test_comsol_is_not_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            (root / "data/calibration").mkdir(parents=True)
            (root / "data/sample").mkdir()
            (root / "input_parameters").mkdir()
            (root / "data/comsol.txt").write_text("table")
            experiment = Experiment.open(root)
            self.assertEqual(experiment.measurement_names(), ("sample",))
            self.assertEqual(experiment.side_comsol_file, root / "data/comsol.txt")


class SideRestoreTests(unittest.TestCase):
    def test_vector_assembly_and_ready_vector(self):
        from restoration.legacy import restore as runner
        import pandas as pd
        for ready in (False, True):
            with self.subTest(ready=ready), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                side = root / "side.txt"
                side.write_text("9\n8\n")
                full = root / "b.txt"
                full.write_text("2\n4\n9\n8\n")
                np.savez(root / "A.npz", matrix=np.ones((4, 1)))
                cfg = SimpleNamespace(exposure_time_arr=[2], dir_save=root, save_params=lambda: None, path_signal_darl_rel=None, file_bins=root / "cam", file_bins_lin=root / "lin", file_matrix=root / "A.npz", file_classes=root / "classes", side_angle_deg=(135., 110.), side_signal_file=side, cut_classes=None, cut_classes_top=None, signal_vector_file=full if ready else None, cam_pixel_width_m=1., detector_configuration_type="new", x_shift_pix=0., y_shift_pix=0., cam_hdr_filtered=False, signal_type="signal", exposure_time_us_lin_arr=[3], dir_signal_lin=root, filename_lin_template="{}.txt", dir_back_lin=None, num_pix_lin=1, shift_lin_m=0., pix_max_ampl=0., width_pix_x_m=1., width_pix_y_m=1., lin_signal_mode=2, lin_time_add=0., coef_lin_to_cam=2., use_w_critical=False, use_chahine=False, use_conc_corr=False, mode_chahine_q0="default")
                if ready:
                    side.unlink()  # Must never read or append the separate side vector.
                with patch.object(runner, "pd") as mock_pd, patch.object(runner, "plt", MagicMock()), patch.object(runner, "read_classes", return_value=np.array([1.])), patch.object(runner, "get_signal_cam_hdr", return_value=(np.ones((1, 1)), np.ones((1, 1)))), patch.object(runner, "print_cam_pix_valid_per_bin"), patch.object(runner, "get_laser_center_pos_pix"), patch.object(runner, "plot_cam_signal_valid"), patch.object(runner, "get_signal_cam", return_value=np.array([4.])), patch.object(runner, "get_signal_lin", return_value=(np.array([6.]), None)), patch.object(runner, "inverse_solver_type_1", side_effect=RuntimeError("stop after assembly")) as solver:
                    mock_pd.read_csv.return_value = pd.DataFrame({"value": [1.]})
                    with self.assertRaisesRegex(RuntimeError, "stop after assembly"):
                        runner.run_cfg(cfg)
                    np.testing.assert_array_equal(solver.call_args.args[1], [2., 4., 9., 8.])
                    np.testing.assert_array_equal(
                        np.loadtxt(root / "reference_side_signal.txt", skiprows=1),
                        [[9., 9.], [8., 8.]],
                    )
                    self.assertEqual(
                        (root / "reference_side_signal.txt").read_text().splitlines()[0],
                        "raw\tnormalized",
                    )
                    runner.plt.savefig.assert_any_call(
                        root / "signal-side-ex_time_2-ex_time_lin_3.png"
                    )
                    self.assertEqual(np.loadtxt(root / "reference_line_signal.txt", skiprows=1).shape, (2,))

    def test_camera_only_runner_rejects_side(self):
        from restoration.legacy.restore import run_cfg_lin_cut
        with self.assertRaisesRegex(ValueError, "недоступна"):
            run_cfg_lin_cut(SimpleNamespace(side_angle_deg=(135.,)))
