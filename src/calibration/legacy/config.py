"""Translate the automated experiment model to the original config object."""

from __future__ import annotations

from pathlib import Path

from experiment import Experiment
from parameters import CalibrationStageParameters
from validation import parse_exposure_filename

from .func import ExperimentConfig


def load_legacy_config(
    experiment_path: str | Path,
    output_directory: str | Path,
) -> tuple[ExperimentConfig, Experiment, CalibrationStageParameters]:
    experiment = Experiment.open(experiment_path)
    parameters = CalibrationStageParameters.load(experiment)
    camera = parameters.calibration.camera
    camera_geometry = parameters.general.detectors.camera
    if camera is None or camera_geometry is None:
        raise ValueError("Legacy-калибровка требует камеру")

    cfg = ExperimentConfig()
    cfg.dir_case = experiment.path
    cfg.dir_save_preprocessing_rel = Path(output_directory).resolve()
    cfg.dir_signal_rel = experiment.calibration_camera_dir
    cfg.W = camera_geometry.width_px
    cfg.H = camera_geometry.height_px
    cfg.labm_um = parameters.general.instrument.wavelength_um
    cfg.F_lens_um = parameters.general.instrument.focal_length_um
    cfg.calib_d_pinhole_um = camera.pinhole_diameter_um
    cfg.calib_cam_gaussian_sigma = camera.gaussian_sigma_px
    cfg.calib_cam_corrected = camera.correct_pixel_size

    cfg.cam_pixel_width_m = camera_geometry.pixel_width_m
    cfg.cam_hdr_diff_mode = camera.hdr.difference_mode
    cfg.cam_hdr_mode = camera.hdr.mode
    cfg.cam_hdr_low_thr = camera.hdr.low_threshold
    cfg.cam_hdr_back_level = camera.hdr.background_level
    cfg.cam_hdr_top_thr = camera.hdr.top_threshold

    line = parameters.calibration.line_sensor
    if line is not None:
        files = sorted(experiment.calibration_line_dir.glob("*.txt"))
        if len(files) != 1:
            raise ValueError("Для legacy-калибровки линейки нужен один TXT-файл")
        exposure = parse_exposure_filename(files[0], ".txt")
        cfg.dir_signal_lin_rel = experiment.calibration_line_dir
        cfg.filename_lin_template = files[0].name
        cfg.exposure_time_us_lin_arr = [exposure]
        cfg.calib_lin_position_pinhole_m = line.pinhole_position_m
        cfg.calib_lin_position_signal_m = line.signal_position_m
        cfg.calib_lin_gaussian_sigma = line.gaussian_sigma_px
        cfg.lin_time_add = line.time_offset_us
        line_geometry = parameters.general.detectors.line_sensor
        if line_geometry is None:
            raise ValueError("В general.yaml отсутствуют параметры линейки")
        cfg.num_pix_lin = line_geometry.pixel_count
        cfg.width_pix_x_m = line_geometry.pixel_width_m
        cfg.width_pix_y_m = line_geometry.pixel_height_m

    return cfg, experiment, parameters
