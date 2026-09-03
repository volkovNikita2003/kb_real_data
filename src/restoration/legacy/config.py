"""Adapt validated project models to the legacy ``ExperimentConfig``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from calibration.legacy.func import ExperimentConfig
from calibration.refactoring.result import CalibrationResult
from darl.result import DarlResult
from errors import RestorationError
from experiment import Experiment, Measurement
from parameters import GeneralParameters, RestoreParameters

from restoration.operator import (
    ForwardOperator,
    ForwardOperatorError,
    from_darl
)

class LegacyRestoreConfigError(RestorationError):
    """The selected inputs cannot be represented by the legacy algorithm."""


@dataclass(frozen=True)
class LegacyRestoreSolverSettings:
    """Legacy solver settings stored as globals in ``func.py``."""

    regularization_type: int
    regularization_alpha: str | float
    w_critical: float


@dataclass(frozen=True)
class LegacyRestoreConfigArtifact:
    """Complete adapter output for one profile/measurement calculation."""

    config: ExperimentConfig
    solver: LegacyRestoreSolverSettings
    run_mode: Literal["camera", "camera_line"]


def _positive_exposures(values: tuple[int, ...], *, location: str) -> list[int]:
    if not values:
        raise LegacyRestoreConfigError(
            f"{location}: должен быть указан хотя бы один файл экспозиции"
        )
    if any(type(value) is not int or value <= 0 for value in values):
        raise LegacyRestoreConfigError(
            f"{location}: экспозиции должны быть положительными целыми числами"
        )
    if len(values) != len(set(values)):
        raise LegacyRestoreConfigError(
            f"{location}: экспозиции не должны повторяться"
        )
    return sorted(values)


def _relative_to_experiment(
    experiment: Experiment,
    path: str | Path,
    *,
    location: str,
) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(experiment.path)
    except ValueError as error:
        raise LegacyRestoreConfigError(
            f"{location}: путь должен находиться внутри эксперимента: {resolved}"
        ) from error


def build_legacy_restore_config(
    *,
    experiment: Experiment,
    measurement: Measurement,
    general: GeneralParameters,
    restore: RestoreParameters,
    calibration: CalibrationResult,
    operator: ForwardOperator | None = None,
    darl: DarlResult | None = None,
    output_dir: str | Path,
    camera_exposures_us: tuple[int, ...],
    line_exposure_us: int | None,
) -> LegacyRestoreConfigArtifact:
    """Build the exact legacy configuration for one restoration run."""
    if operator is None:
        if darl is None:
            raise LegacyRestoreConfigError("Не задан аппаратный оператор")
        try:
            operator = from_darl(experiment, darl)
        except ForwardOperatorError as error:
            message = str(error).replace(
                "разрешённого каталога", "output/darl"
            )
            raise LegacyRestoreConfigError(message) from error
    if measurement.path.resolve() != (experiment.data_dir / measurement.name).resolve():
        raise LegacyRestoreConfigError(
            "measurement: измерение не принадлежит переданному эксперименту"
        )

    restore_detectors = restore.detectors.names()
    if "camera" not in restore_detectors:
        raise LegacyRestoreConfigError(
            "Legacy-восстановление пока поддерживает только профили с камерой"
        )
    general_detectors = general.detectors.names()
    missing_general = restore_detectors - general_detectors
    if missing_general:
        raise LegacyRestoreConfigError(
            "Профиль использует отсутствующие детекторы general.yaml: "
            + ", ".join(sorted(missing_general))
        )
    missing_operator = restore_detectors - set(operator.detectors)
    if missing_operator:
        raise LegacyRestoreConfigError(
            "Профиль использует детекторы, отсутствующие в операторе: "
            + ", ".join(sorted(missing_operator))
        )

    camera_geometry = general.detectors.camera
    camera_restore = restore.detectors.camera
    if camera_geometry is None or camera_restore is None:
        raise LegacyRestoreConfigError("Отсутствуют параметры камеры")
    camera_exposures = _positive_exposures(
        camera_exposures_us, location="camera_exposures_us"
    )

    output = Path(output_dir).expanduser().resolve()
    restore_root = experiment.restore_output_root.resolve()
    try:
        output.relative_to(restore_root)
    except ValueError as error:
        raise LegacyRestoreConfigError(
            "output_dir: каталог должен находиться внутри output/restore"
        ) from error
    output_relative = _relative_to_experiment(
        experiment, output, location="output_dir"
    )

    matrix_name = _relative_to_experiment(
        experiment, operator.matrix_file, location="operator.matrix_file"
    )
    classes_name = _relative_to_experiment(
        experiment, operator.particle_classes_file,
        location="operator.particle_classes_file",
    )
    bins_name = _relative_to_experiment(
        experiment, operator.detector_bin_files["camera"],
        location="operator.detector_bin_files.camera",
    )
    expected_path = operator.expected_signal(measurement.name)
    expected_signal = (
        _relative_to_experiment(
            experiment, expected_path,
            location=f"operator.expected_signals.{measurement.name}",
        )
        if expected_path is not None else None
    )

    position = {"old": 0, "new": 1}.get(
        general.instrument.detector_position
    )
    if position is None:
        raise LegacyRestoreConfigError(
            "general.instrument.detector_position: ожидалось old или new"
        )

    cfg = ExperimentConfig(
        dir_case=experiment.path,
        dir_save_rel=output_relative,
        labm_um=general.instrument.wavelength_um,
        F_lens_um=general.instrument.focal_length_um,
        detector_configuration_type=position,
        dir_signal_rel=_relative_to_experiment(
            experiment, measurement.camera_dir, location="measurement.camera"
        ),
        dir_back_rel=(
            _relative_to_experiment(
                experiment,
                measurement.camera_background_dir,
                location="measurement.camera_background",
            )
            if camera_restore.use_background else None
        ),
        signal_vector_file_rel=(
            _relative_to_experiment(
                experiment,
                measurement.signal_vector_file,
                location="measurement.signal_vector_file",
            )
            if measurement.signal_vector_file.is_file() else None
        ),
        exposure_time_arr=camera_exposures,
        W=camera_geometry.width_px,
        H=camera_geometry.height_px,
        cam_pixel_width_m=calibration.camera.pixel_width_m,
        x_shift_m=calibration.camera.x_shift_m,
        y_shift_m=calibration.camera.y_shift_m,
        cam_hdr_files_mode="bmp",
        cam_hdr_mode=camera_restore.hdr.mode,
        cam_hdr_diff_mode=camera_restore.hdr.difference_mode,
        cam_hdr_back_level=camera_restore.hdr.background_level,
        cam_hdr_top_thr=camera_restore.hdr.top_threshold,
        cam_hdr_low_thr=camera_restore.hdr.low_threshold,
        cam_hdr_filtered=camera_restore.hdr.filtered,
        cam_hdr_gauss_sigma=camera_restore.hdr.gaussian_sigma,
        cam_hdr_exposure_coefs=None,
        darl_config_name=operator.name,
        path_signal_darl_rel=expected_signal,
        matrix_name=matrix_name,
        bins_name=bins_name,
        classes_name=classes_name,
        signal_type=operator.signal_value_type,
        use_w_critical=restore.solver.use_w_critical,
        use_chahine=restore.solver.use_chahine,
        use_conc_corr=restore.solver.use_concentration_correction,
        forward_modeling_enabled=restore.forward_modeling.enabled,
        cut_classes=restore.class_slice.drop_first,
        cut_classes_top=restore.class_slice.drop_last,
    )

    run_mode: Literal["camera", "camera_line"] = "camera"
    line_restore = restore.detectors.line_sensor
    if line_restore is not None:
        line_geometry = general.detectors.line_sensor
        line_calibration = calibration.line_sensor
        if line_geometry is None or line_calibration is None:
            raise LegacyRestoreConfigError(
                "Для линейки отсутствует геометрия или результат калибровки"
            )
        if type(line_exposure_us) is not int or line_exposure_us <= 0:
            raise LegacyRestoreConfigError(
                "line_exposure_us: ожидалась одна положительная целая экспозиция"
            )
        cfg.dir_signal_lin_rel = _relative_to_experiment(
            experiment, measurement.line_dir, location="measurement.line_sensor"
        )
        cfg.dir_back_lin_rel = (
            _relative_to_experiment(
                experiment,
                measurement.line_background_dir,
                location="measurement.line_sensor_background",
            )
            if line_restore.use_background else None
        )
        cfg.filename_lin_template = "{}.txt"
        cfg.filename_lin_back_template = "{}.txt"
        cfg.exposure_time_us_lin_arr = [line_exposure_us]
        cfg.coef_lin_to_cam = line_calibration.to_camera_coefficient
        cfg.shift_lin_m = line_calibration.shift_m
        cfg.pix_max_ampl = line_calibration.peak_pixel
        cfg.num_pix_lin = line_geometry.pixel_count
        cfg.width_pix_x_m = line_calibration.pixel_width_m
        cfg.width_pix_y_m = line_calibration.pixel_height_m
        cfg.lin_time_add = line_restore.time_offset_us
        cfg.lin_signal_mode = line_restore.signal_mode
        cfg.bins_lin_name = _relative_to_experiment(
            experiment, operator.detector_bin_files["line_sensor"],
            location="operator.detector_bin_files.line_sensor",
        )
        run_mode = "camera_line"
    elif line_exposure_us is not None:
        raise LegacyRestoreConfigError(
            "line_exposure_us: экспозиция передана для выключенной линейки"
        )
    else:
        cfg.bins_lin_name = None

    return LegacyRestoreConfigArtifact(
        config=cfg,
        solver=LegacyRestoreSolverSettings(
            regularization_type=restore.solver.regularization_order,
            regularization_alpha=restore.solver.regularization_alpha,
            w_critical=restore.solver.w_critical,
        ),
        run_mode=run_mode,
    )
