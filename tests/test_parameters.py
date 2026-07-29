from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import yaml


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from errors import ParametersError
from experiment import Experiment
from parameters import (
    CalibrationStageParameters,
    DarlStageParameters,
    ExperimentParameters,
    load_calibration_parameters,
    load_darl_parameters,
    load_general_parameters,
    load_measurement_parameters,
    write_used_parameters,
)
from calibration.refactoring.result import (
    CalibrationResult,
    CalibrationResultError,
    CameraCalibrationResult,
    LineSensorCalibrationResult,
    save_calibration_result,
)


CAMERA_DARL = {
    "schema_version": 1,
    "detectors": {"camera": {}},
    "particles": {
        "refractive_index": 1.77,
        "absorption_coefficient": 0.0,
        "type": "sphere",
    },
    "medium": {
        "inside_cuvette_refractive_index": 1.3333,
        "cuvette_refractive_index": 1.5,
        "outside_cuvette_refractive_index": 1.0,
    },
}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def create_camera_experiment(root: Path) -> Experiment:
    path = root / "experiment"
    (path / "data/calibration").mkdir(parents=True)
    parameters = path / "input_parameters"
    parameters.mkdir()
    dump(
        parameters / "general.yaml",
        {
            "schema_version": 1,
            "detectors": {"camera": {}},
            "instrument": {"detector_position": "old"},
        },
    )
    dump(
        parameters / "calibration.yaml",
        {"schema_version": 1, "mode": "automatic", "camera": {}},
    )
    dump(parameters / "darl.yaml", CAMERA_DARL)
    (path / "data/kmk_15").mkdir()
    return Experiment.open(path)


def save_camera_calibration_result(
    experiment: Experiment,
    *,
    with_line_sensor: bool = False,
) -> None:
    line = None
    if with_line_sensor:
        line = LineSensorCalibrationResult(
            0.88,
            5.09,
            8.1e-6,
            2.02e-4,
            0.31,
            0.02,
            1752.0,
        )
    save_calibration_result(
        CalibrationResult(
            1,
            CameraCalibrationResult(6.19, 1.7e-4, 4.9e-5, 1.91e-6),
            line,
        ),
        experiment.calibration_result_file,
    )


class GeneralParametersTests(unittest.TestCase):
    def test_defaults_are_applied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "general.yaml"
            dump(
                path,
                {
                    "schema_version": 1,
                    "detectors": {"camera": {}},
                    "instrument": {"detector_position": "old"},
                },
            )

            result = load_general_parameters(path)

            self.assertEqual(result.detectors.names(), frozenset({"camera"}))
            self.assertEqual(result.instrument.wavelength_um, 0.633)
            self.assertEqual(result.instrument.focal_length_um, 400_000.0)

    def test_unknown_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "general.yaml"
            dump(
                path,
                {
                    "schema_version": 1,
                    "detectors": {"camera": {}},
                    "instrument": {"detector_position": "old"},
                    "unknown": 42,
                },
            )
            with self.assertRaisesRegex(ParametersError, "неизвестные поля: unknown"):
                load_general_parameters(path)

    def test_bool_is_not_accepted_as_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "general.yaml"
            dump(
                path,
                {
                    "schema_version": 1,
                    "detectors": {"camera": {}},
                    "instrument": {
                        "detector_position": "old",
                        "wavelength_um": True,
                    },
                },
            )
            with self.assertRaisesRegex(ParametersError, "ожидалось число"):
                load_general_parameters(path)

    def test_non_finite_numbers_are_rejected(self) -> None:
        for yaml_number in (".nan", ".inf", "-.inf"):
            with self.subTest(yaml_number=yaml_number):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "general.yaml"
                    path.write_text(
                        "schema_version: 1\n"
                        "detectors:\n"
                        "  camera: {}\n"
                        "instrument:\n"
                        "  detector_position: old\n"
                        f"  wavelength_um: {yaml_number}\n",
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(
                        ParametersError,
                        "число должно быть конечным",
                    ):
                        load_general_parameters(path)

    def test_duplicate_yaml_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "general.yaml"
            path.write_text(
                "schema_version: 1\n"
                "detectors:\n"
                "  camera: {}\n"
                "instrument:\n"
                "  detector_position: old\n"
                "  wavelength_um: 0.633\n"
                "  wavelength_um: 0.488\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ParametersError,
                "duplicate key 'wavelength_um'",
            ):
                load_general_parameters(path)

    def test_unsupported_schema_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "general.yaml"
            dump(
                path,
                {
                    "schema_version": 2,
                    "detectors": {"camera": {}},
                    "instrument": {"detector_position": "old"},
                },
            )
            with self.assertRaisesRegex(ParametersError, "не поддерживается"):
                load_general_parameters(path)


class DarlParametersTests(unittest.TestCase):
    def test_reference_defaults_are_applied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "darl.yaml"
            dump(path, CAMERA_DARL)

            result = load_darl_parameters(path)

            self.assertEqual(result.particle_classes.min_diameter_nm, 100.0)
            self.assertEqual(result.particle_classes.max_diameter_nm, 3_500_000.0)
            self.assertEqual(result.laser.power_w, 30.0)
            self.assertFalse(result.signal.one_particle)

    def test_required_particle_field_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "darl.yaml"
            value = yaml.safe_load(yaml.safe_dump(CAMERA_DARL))
            del value["particles"]["refractive_index"]
            dump(path, value)

            with self.assertRaisesRegex(
                ParametersError,
                "отсутствуют обязательные поля: refractive_index",
            ):
                load_darl_parameters(path)

    def test_python_expression_is_not_a_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "darl.yaml"
            value = yaml.safe_load(yaml.safe_dump(CAMERA_DARL))
            value["particles"]["refractive_index"] = "1.7 + 0.07"
            dump(path, value)

            with self.assertRaisesRegex(ParametersError, "ожидалось число"):
                load_darl_parameters(path)


class CalibrationParametersTests(unittest.TestCase):
    def test_line_positions_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.yaml"
            dump(
                path,
                {
                    "schema_version": 1,
                    "mode": "automatic",
                    "line_sensor": {},
                },
            )

            with self.assertRaisesRegex(
                ParametersError,
                "pinhole_position_m, signal_position_m",
            ):
                load_calibration_parameters(path)

    def test_line_positions_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.yaml"
            dump(
                path,
                {
                    "schema_version": 1,
                    "mode": "automatic",
                    "line_sensor": {
                        "pinhole_position_m": 0.0001,
                        "signal_position_m": 0.0201,
                    },
                },
            )

            result = load_calibration_parameters(path)

            self.assertEqual(result.line_sensor.pinhole_position_m, 0.0001)
            self.assertEqual(result.line_sensor.signal_position_m, 0.0201)


class EffectiveParametersTests(unittest.TestCase):
    def test_calibration_stage_does_not_read_darl_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_camera_experiment(Path(directory))
            experiment.darl_parameters_file.write_text(
                "this is not valid DARL YAML: [",
                encoding="utf-8",
            )

            parameters = CalibrationStageParameters.load(experiment)

            self.assertEqual(parameters.general.detectors.names(), {"camera"})
            self.assertIsNotNone(parameters.calibration.camera)

    def test_detector_sets_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_camera_experiment(Path(directory))
            darl_path = experiment.darl_parameters_file
            value = yaml.safe_load(darl_path.read_text(encoding="utf-8"))
            value["detectors"]["line_sensor"] = {
                "logarithmic_radius_percent": 7.5
            }
            dump(darl_path, value)

            with self.assertRaisesRegex(ParametersError, "набор секций детекторов"):
                ExperimentParameters.load(experiment)

    def test_matrix_and_quality_control_have_separate_effective_parameters(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_camera_experiment(Path(directory))
            parameters = ExperimentParameters.load(experiment)

            matrix = parameters.effective_matrix(
                {"camera": {"shift_x_um": 1.0}}
            )
            quality = parameters.effective_quality_control({"matrix_id": "abc"})

            self.assertEqual(
                matrix["darl"]["particle_classes"]["min_diameter_nm"],
                100.0,
            )
            self.assertNotIn("quality_control", matrix)
            self.assertNotIn("line_sensor", matrix["general"]["detectors"])
            self.assertEqual(quality["quality_control"]["restoration_type"], 1)
            self.assertEqual(quality["quality_control"]["class_frequency"], 10)
            self.assertNotIn("line_sensor", matrix["darl"]["detectors"])

    def test_effective_restore_keeps_meaningful_null_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_camera_experiment(Path(directory))
            profile_path = experiment.restore_profiles_dir / "default.yaml"
            dump(
                profile_path,
                {
                    "schema_version": 1,
                    "detectors": {"camera": {}},
                    "class_slice": {"drop_last": None},
                },
            )
            profile = experiment.restore_profile("default")

            effective = ExperimentParameters.load(experiment).effective_restore(
                profile,
                measurement_name="kmk_15",
                measurement_inputs={},
                calibration_result={},
                matrix_result={},
            )

            self.assertNotIn("line_sensor", effective["restore"]["detectors"])
            self.assertIsNone(effective["restore"]["class_slice"]["drop_last"])


class DarlStageParametersTests(unittest.TestCase):
    def test_loads_only_darl_stage_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_camera_experiment(Path(directory))
            save_camera_calibration_result(experiment)
            experiment.calibration_parameters_file.write_text(
                "invalid calibration YAML: [",
                encoding="utf-8",
            )
            dump(
                experiment.restore_profiles_dir / "broken.yaml",
                {"not": "a restore profile"},
            )

            parameters = DarlStageParameters.load(experiment)

            self.assertEqual(parameters.general.detectors.names(), {"camera"})
            self.assertEqual(parameters.darl.detectors.names(), {"camera"})
            self.assertEqual(
                tuple(item.name for item in parameters.distributions),
                ("pinhole_200",),
            )

    def test_requires_calibration_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_camera_experiment(Path(directory))

            with self.assertRaisesRegex(CalibrationResultError, "не найден"):
                DarlStageParameters.load(experiment)

    def test_loads_expected_distributions_in_measurement_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_camera_experiment(Path(directory))
            (experiment.data_dir / "aaa").mkdir()
            experiment = Experiment.open(experiment.path)
            save_camera_calibration_result(experiment)
            for name, mean in (("kmk_15", 15_000), ("aaa", 10_000)):
                dump(
                    experiment.measurement_parameters_dir / f"{name}.yaml",
                    {
                        "schema_version": 1,
                        "expected_distribution": {
                            "type": "gaussian",
                            "mean_nm": mean,
                            "sigma_nm": 1000,
                            "particle_count": 20_000,
                        },
                    },
                )

            parameters = DarlStageParameters.load(experiment)

            self.assertEqual(
                tuple(item.name for item in parameters.distributions),
                ("pinhole_200", "aaa", "kmk_15"),
            )
            self.assertEqual(
                parameters.distributions[1].parameters_file,
                "input_parameters/measurements/aaa.yaml",
            )

    def test_detector_sets_must_match_calibration_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_camera_experiment(Path(directory))
            general = yaml.safe_load(
                experiment.general_parameters_file.read_text(encoding="utf-8")
            )
            general["detectors"]["line_sensor"] = {}
            dump(experiment.general_parameters_file, general)
            darl = yaml.safe_load(
                experiment.darl_parameters_file.read_text(encoding="utf-8")
            )
            darl["detectors"]["line_sensor"] = {}
            dump(experiment.darl_parameters_file, darl)
            save_camera_calibration_result(experiment)

            with self.assertRaisesRegex(
                ParametersError,
                "набор результатов детекторов",
            ):
                DarlStageParameters.load(experiment)

    def test_detector_sets_must_match_darl_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_camera_experiment(Path(directory))
            darl = yaml.safe_load(
                experiment.darl_parameters_file.read_text(encoding="utf-8")
            )
            darl["detectors"]["line_sensor"] = {}
            dump(experiment.darl_parameters_file, darl)
            save_camera_calibration_result(experiment)

            with self.assertRaisesRegex(
                ParametersError,
                "darl.yaml: набор секций детекторов",
            ):
                DarlStageParameters.load(experiment)

    def test_effective_parameters_include_defaults_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_camera_experiment(Path(directory))
            save_camera_calibration_result(experiment)
            parameters = DarlStageParameters.load(experiment)

            effective = parameters.effective_parameters()

            self.assertEqual(effective["schema_version"], 1)
            self.assertEqual(
                effective["quality_control"]["class_frequency"],
                10,
            )
            self.assertEqual(
                effective["distributions"]["pinhole_200"]["source"],
                "built_in_quality_control",
            )
            self.assertNotIn(
                "parameters_file",
                effective["distributions"]["pinhole_200"],
            )


class ParameterFileTests(unittest.TestCase):
    def test_measurement_parameters_are_optional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_camera_experiment(Path(directory))

            self.assertIsNone(experiment.measurement_parameters_file("kmk_15"))

    def test_expected_distribution_is_loaded_for_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = create_camera_experiment(Path(directory))
            path = experiment.measurement_parameters_dir / "kmk_15.yaml"
            dump(
                path,
                {
                    "schema_version": 1,
                    "expected_distribution": {
                        "type": "gaussian",
                        "mean_nm": 15000,
                        "sigma_nm": 3000,
                        "particle_count": 1_000_000,
                    },
                },
            )
            source = experiment.measurement_parameters_file("kmk_15")
            assert source is not None

            measurement = load_measurement_parameters(source)
            effective = ExperimentParameters.load(
                experiment
            ).effective_expected_signal(
                source,
                matrix_result={"matrix_id": "abc"},
            )

            self.assertEqual(
                measurement.expected_distribution.mean_nm,
                15000.0,
            )
            self.assertEqual(
                effective["experiment"]["measurement"],
                "kmk_15",
            )
            self.assertEqual(effective["matrix_result"]["matrix_id"], "abc")

    def test_write_used_parameters_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "output/darl/matrix/used-parameters.yaml"
            )
            write_used_parameters(path, {"schema_version": 1, "value": 42})
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["value"], 42)

            with self.assertRaisesRegex(ParametersError, "уже существует"):
                write_used_parameters(path, {"value": 43})


if __name__ == "__main__":
    unittest.main()
