"""Translate the automated experiment model to the original config object."""

from __future__ import annotations

from pathlib import Path

from experiment import Experiment
from parameters import ExperimentParameters
from validation import parse_exposure_filename

from .func import ExperimentConfig


def load_legacy_config(
    experiment_path: str | Path,
    output_directory: str | Path,
) -> tuple[ExperimentConfig, Experiment, ExperimentParameters]:
    experiment = Experiment.open(experiment_path)
    parameters = ExperimentParameters.load(experiment)
    camera = parameters.calibration.camera
    camera_geometry = parameters.darl.detectors.camera
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

    # These values reproduce params-general.txt of the reference calculation.
    # They remain legacy defaults until the shared HDR parameter model is
    # introduced; the numerical code below is intentionally unchanged.
    cfg.cam_hdr_diff_mode = "after_hdr"
    cfg.cam_hdr_mode = "l2h"
    cfg.cam_hdr_low_thr = 10.0
    cfg.cam_hdr_back_level = 12.0
    cfg.cam_hdr_top_thr = 240.0

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
        cfg.num_pix_lin = line.pixel_count
        cfg.width_pix_x_m = line.pixel_width_um * 1e-6
        cfg.width_pix_y_m = 0.0002

    return cfg, experiment, parameters
