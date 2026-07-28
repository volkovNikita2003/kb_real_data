import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.interpolate import interp1d
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter

from func import (
    ExperimentConfig,
    # get_signal_cam_good_pix,
    get_signal_cam_hdr,
    read_coefficients,
    read_classes,
    inverse_solver_type_1,
    get_signal_cam,
    get_signal_lin,
    plot_gcv_curve,
    plot_cam_signal,
    plot_cam_signal_valid,
    plot_b_signal,
    get_avg_img,
    get_bmp_numeric_names,
)


def save_reference_restoration(
    directory,
    prefix,
    sizes,
    restored_distr_reg,
    restored_distr_reg_iter,
    restored_distr_reg_iter_w,
    restored_gost_reg,
    restored_gost_reg_iter,
    restored_gost_reg_iter_w,
    gcv_curve,
    alpha_reg,
):
    """Save every numerical inverse-solver output for regression tests."""
    np.savetxt(
        directory / f"{prefix}reference_restoration.txt",
        np.column_stack((
            sizes,
            restored_distr_reg,
            restored_distr_reg_iter,
            restored_distr_reg_iter_w,
            restored_gost_reg,
            restored_gost_reg_iter,
            restored_gost_reg_iter_w,
        )),
        delimiter="\t",
        header=(
            "sizes\tdistribution_regularized\tdistribution_iterative\t"
            "distribution_weight_cut\tgost_regularized\tgost_iterative\t"
            "gost_weight_cut"
        ),
        comments="",
        fmt="%.17g",
    )
    np.savetxt(
        directory / f"{prefix}reference_gcv_curve.txt",
        gcv_curve,
        delimiter="\t",
        header="alpha\tgcv",
        comments="",
        fmt="%.17g",
    )
    np.savetxt(
        directory / f"{prefix}reference_alpha.txt",
        np.atleast_1d(alpha_reg),
        fmt="%.17g",
    )



def make_cfg_arr():
    cfg_arr = []


    particles_cases_arr = [
        # ("kmk_15", "pinhole_200", "b_pinhole_200", "kmk-15", None),
        ("kmk_15", "kmk_15", "b_kmk_15", "kmk-15", 400),
    ]
    signal_type_arr = [
        ("signal", "signal", "signal", 10),
    ]
    exposures_arr = [
        # ([200, 1000, 5000, 25000, 125000],
        # "cam_exp_down_200us"),
        ([24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24000, 48000, 96000, 200000, 400000, 800000, 1600000],
        "cam_exp_down_24us"),
    ]
    calibration_params = {
        "kmk_15": {
            "x_shift_m": 0.00017178838009236094,
            "y_shift_m": 4.903750079380921e-05,
            "cam_pixel_width_m": 1.911283425948221e-06,
            "width_pix_x_m": 8.098304067336103e-06,
            "width_pix_y_m": 0.0002024576016834026,
            "coef_lin_to_cam": 0.31081315123650427,
            "shift_lin_m": 0.02034698896918196,
            "pix_max_ampl": 1751.9944216945232,
        },
    }
    for particles_cases in particles_cases_arr:
        for signal_type in signal_type_arr:
            for exposures in exposures_arr:
                darl_dir_name = f"darl-{particles_cases[3]}-{signal_type[2]}"
                cfg = ExperimentConfig(
                    dir_case=Path(f"test_17_07_26_kmk_15/{particles_cases[0]}/"),
                    # dir_signal_rel=f"data/{particles_cases[0]}/",
                    
                    detector_configuration_type=1,

                    # dir_signal_rel=f"юстировка",
                    dir_signal_rel=f"сигнал",
                    dir_back_rel=f"фон",
                    # dir_save_rel=f"restore_cam_l2h_10_bl12_per_exposure/20_07_26_{particles_cases[1]}_{signal_type[1]}_{exposures[1]}/",
                    dir_save_rel=f"restore_l2h_10_bl12/28_07_26_{particles_cases[1]}_{signal_type[1]}_{exposures[1]}/",
                    
                    path_signal_darl_rel = f"{darl_dir_name}/{particles_cases[2]}.txt" if particles_cases[2] is not None else None,
                    
                    use_w_critical=False,
                    use_chahine=True,
                    use_conc_corr=True,

                    matrix_name=f"{darl_dir_name}/matrix-case_conf_real_data_test_17_07_26-{particles_cases[3]}-cam-lin-{signal_type[2]}-w_633.0.npz",
                    bins_name=f"{darl_dir_name}/bins_front_detector.txt",
                    classes_name=f"{darl_dir_name}/particle_classes_lasser_0.txt",
                    bins_lin_name=f"{darl_dir_name}/FrontDetectorLogLine_detector.txt",
                    x_shift_m = calibration_params[particles_cases[0]]["x_shift_m"],
                    y_shift_m = calibration_params[particles_cases[0]]["y_shift_m"],
                    cam_pixel_width_m = calibration_params[particles_cases[0]]["cam_pixel_width_m"],

                    # cam_hdr_diff_mode = "per_exposure", # "per_exposure" or "after_hdr"
                    cam_hdr_diff_mode = "after_hdr", # "per_exposure" or "after_hdr"
                    # cam_hdr_mode="l2h_longest",
                    cam_hdr_mode="l2h",
                    cam_hdr_low_thr=10,
                    cam_hdr_back_level=12,
                    cam_hdr_top_thr=255-15,
                    cam_hdr_filtered=False,
                    cam_hdr_gauss_sigma=5,
                    
                    signal_type=signal_type[0],

                    # exposure_time_arr=[25, 50, 100, 200, 400, 2000, 10000, 50000, 250000, 1250000],
                    exposure_time_arr=exposures[0],
                    
                    cut_classes=signal_type[3],
                    cut_classes_top=None,

                    dir_signal_lin_rel=f"сигнал/",
                    dir_back_lin_rel=f"фон/",
                    filename_lin_template = f"lin-{particles_cases[3]}-{{}}us.txt",
                    filename_lin_back_template = f"lin-back-{particles_cases[3]}-{{}}us.txt",
                    width_pix_x_m = calibration_params[particles_cases[0]]["width_pix_x_m"],
                    width_pix_y_m = calibration_params[particles_cases[0]]["width_pix_y_m"],
                    coef_lin_to_cam = calibration_params[particles_cases[0]]["coef_lin_to_cam"],
                    shift_lin_m = calibration_params[particles_cases[0]]["shift_lin_m"],
                    pix_max_ampl = calibration_params[particles_cases[0]]["pix_max_ampl"],
                    exposure_time_us_lin_arr = [particles_cases[4]],
                    lin_signal_mode=2,
                )
                cfg_arr.append((cfg, run_cfg))




    particles_cases_arr = [
        # ("kmk_15", "pinhole_200", "b_pinhole_200", "kmk-15", None),
        ("kmk_15", "kmk_15", "b_kmk_15", "kmk-15", 400),
    ]
    signal_type_arr = [
        ("signal", "signal", "signal", 10),
    ]
    exposures_arr = [
        # ([200, 1000, 5000, 25000, 125000],
        # "cam_exp_down_200us"),
        ([24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24000, 48000, 96000, 200000, 400000, 800000, 1600000],
        "cam_exp_down_24us"),
    ]
    calibration_params = {
        "kmk_15": {
            "x_shift_m": 0.00017178838009236094,
            "y_shift_m": 4.903750079380921e-05,
            "cam_pixel_width_m": 1.911283425948221e-06,
            "width_pix_x_m": 8.098304067336103e-06,
            "width_pix_y_m": 0.0002024576016834026,
            "coef_lin_to_cam": 0.31081315123650427,
            "shift_lin_m": 0.02034698896918196,
            "pix_max_ampl": 1751.9944216945232,
        },
    }
    for particles_cases in particles_cases_arr:
        for signal_type in signal_type_arr:
            for exposures in exposures_arr:
                darl_dir_name = f"darl-{particles_cases[3]}-{signal_type[2]}"
                cfg = ExperimentConfig(
                    dir_case=Path(f"test_17_07_26_kmk_15/{particles_cases[0]}/"),
                    # dir_signal_rel=f"data/{particles_cases[0]}/",
                    
                    detector_configuration_type=1,

                    # dir_signal_rel=f"юстировка",
                    dir_signal_rel=f"сигнал",
                    dir_back_rel=f"фон",
                    dir_save_rel=f"restore_l2h_10_bl12_per_exposure/28_07_26_{particles_cases[1]}_{signal_type[1]}_{exposures[1]}/",
                    # dir_save_rel=f"restore_l2h_10_bl12/28_07_26_{particles_cases[1]}_{signal_type[1]}_{exposures[1]}/",
                    
                    path_signal_darl_rel = f"{darl_dir_name}/{particles_cases[2]}.txt" if particles_cases[2] is not None else None,
                    
                    use_w_critical=False,
                    use_chahine=True,
                    use_conc_corr=True,

                    matrix_name=f"{darl_dir_name}/matrix-case_conf_real_data_test_17_07_26-{particles_cases[3]}-cam-lin-{signal_type[2]}-w_633.0.npz",
                    bins_name=f"{darl_dir_name}/bins_front_detector.txt",
                    classes_name=f"{darl_dir_name}/particle_classes_lasser_0.txt",
                    bins_lin_name=f"{darl_dir_name}/FrontDetectorLogLine_detector.txt",
                    x_shift_m = calibration_params[particles_cases[0]]["x_shift_m"],
                    y_shift_m = calibration_params[particles_cases[0]]["y_shift_m"],
                    cam_pixel_width_m = calibration_params[particles_cases[0]]["cam_pixel_width_m"],

                    cam_hdr_diff_mode = "per_exposure", # "per_exposure" or "after_hdr"
                    # cam_hdr_diff_mode = "after_hdr", # "per_exposure" or "after_hdr"
                    # cam_hdr_mode="l2h_longest",
                    cam_hdr_mode="l2h",
                    cam_hdr_low_thr=10,
                    cam_hdr_back_level=12,
                    cam_hdr_top_thr=255-15,
                    cam_hdr_filtered=False,
                    cam_hdr_gauss_sigma=5,
                    
                    signal_type=signal_type[0],

                    # exposure_time_arr=[25, 50, 100, 200, 400, 2000, 10000, 50000, 250000, 1250000],
                    exposure_time_arr=exposures[0],
                    
                    cut_classes=signal_type[3],
                    cut_classes_top=None,

                    dir_signal_lin_rel=f"сигнал/",
                    dir_back_lin_rel=f"фон/",
                    filename_lin_template = f"lin-{particles_cases[3]}-{{}}us.txt",
                    filename_lin_back_template = f"lin-back-{particles_cases[3]}-{{}}us.txt",
                    width_pix_x_m = calibration_params[particles_cases[0]]["width_pix_x_m"],
                    width_pix_y_m = calibration_params[particles_cases[0]]["width_pix_y_m"],
                    coef_lin_to_cam = calibration_params[particles_cases[0]]["coef_lin_to_cam"],
                    shift_lin_m = calibration_params[particles_cases[0]]["shift_lin_m"],
                    pix_max_ampl = calibration_params[particles_cases[0]]["pix_max_ampl"],
                    exposure_time_us_lin_arr = [particles_cases[4]],
                    lin_signal_mode=2,
                )
                cfg_arr.append((cfg, run_cfg))




    particles_cases_arr = [
        ("kmk_15", "pinhole_200", "b_pinhole_200", "kmk-15", None),
        # ("kmk_15", "kmk_15", "b_kmk_15", "kmk-15", 400),
    ]
    signal_type_arr = [
        ("signal", "signal", "signal", 10),
    ]
    exposures_arr = [
        ([200, 1000, 5000, 25000, 125000],
        "cam_exp_down_200us"),
        # ([24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24000, 48000, 96000, 200000, 400000, 800000, 1600000],
        # "cam_exp_down_24us"),
    ]
    calibration_params = {
        "kmk_15": {
            "x_shift_m": 0.00017178838009236094,
            "y_shift_m": 4.903750079380921e-05,
            "cam_pixel_width_m": 1.911283425948221e-06,
            "width_pix_x_m": 8.098304067336103e-06,
            "width_pix_y_m": 0.0002024576016834026,
            "coef_lin_to_cam": 0.31081315123650427,
            "shift_lin_m": 0.02034698896918196,
            "pix_max_ampl": 1751.9944216945232,
        },
    }
    for particles_cases in particles_cases_arr:
        for signal_type in signal_type_arr:
            for exposures in exposures_arr:
                darl_dir_name = f"darl-{particles_cases[3]}-{signal_type[2]}"
                cfg = ExperimentConfig(
                    dir_case=Path(f"test_17_07_26_kmk_15/{particles_cases[0]}/"),
                    # dir_signal_rel=f"data/{particles_cases[0]}/",
                    
                    detector_configuration_type=1,

                    dir_signal_rel=f"юстировка",
                    # dir_signal_rel=f"сигнал",
                    # dir_back_rel=f"фон",
                    # dir_save_rel=f"restore_cam_l2h_10_bl12_per_exposure/20_07_26_{particles_cases[1]}_{signal_type[1]}_{exposures[1]}/",
                    dir_save_rel=f"restore_cam_l2h_10_bl12/28_07_26_{particles_cases[1]}_{signal_type[1]}_{exposures[1]}/",
                    
                    path_signal_darl_rel = f"{darl_dir_name}/{particles_cases[2]}.txt" if particles_cases[2] is not None else None,
                    
                    use_w_critical=False,
                    use_chahine=True,
                    use_conc_corr=True,

                    matrix_name=f"{darl_dir_name}/matrix-case_conf_real_data_test_17_07_26-{particles_cases[3]}-cam-lin-{signal_type[2]}-w_633.0.npz",
                    bins_name=f"{darl_dir_name}/bins_front_detector.txt",
                    classes_name=f"{darl_dir_name}/particle_classes_lasser_0.txt",
                    # bins_lin_name=f"{darl_dir_name}/FrontDetectorLogLine_detector.txt",
                    x_shift_m = calibration_params[particles_cases[0]]["x_shift_m"],
                    y_shift_m = calibration_params[particles_cases[0]]["y_shift_m"],
                    cam_pixel_width_m = calibration_params[particles_cases[0]]["cam_pixel_width_m"],

                    # cam_hdr_diff_mode = "per_exposure", # "per_exposure" or "after_hdr"
                    cam_hdr_diff_mode = "after_hdr", # "per_exposure" or "after_hdr"
                    # cam_hdr_mode="l2h_longest",
                    cam_hdr_mode="l2h",
                    cam_hdr_low_thr=10,
                    cam_hdr_back_level=12,
                    cam_hdr_top_thr=255-15,
                    cam_hdr_filtered=False,
                    cam_hdr_gauss_sigma=5,
                    
                    signal_type=signal_type[0],

                    # exposure_time_arr=[25, 50, 100, 200, 400, 2000, 10000, 50000, 250000, 1250000],
                    exposure_time_arr=exposures[0],
                    
                    cut_classes=signal_type[3],
                    cut_classes_top=None,

                    # dir_signal_lin_rel=f"сигнал/",
                    # dir_back_lin_rel=f"фон/",
                    # filename_lin_template = f"lin-{particles_cases[3]}-{{}}us.txt",
                    # filename_lin_back_template = f"lin-back-{particles_cases[3]}-{{}}us.txt",
                    # width_pix_x_m = calibration_params[particles_cases[0]]["width_pix_x_m"],
                    # width_pix_y_m = calibration_params[particles_cases[0]]["width_pix_y_m"],
                    # coef_lin_to_cam = calibration_params[particles_cases[0]]["coef_lin_to_cam"],
                    # shift_lin_m = calibration_params[particles_cases[0]]["shift_lin_m"],
                    # pix_max_ampl = calibration_params[particles_cases[0]]["pix_max_ampl"],
                    # exposure_time_us_lin_arr = [particles_cases[4]],
                    # lin_signal_mode=2,
                )
                cfg_arr.append((cfg, run_cfg_lin_cut))



    particles_cases_arr = [
        # ("kmk_15", "pinhole_200", "b_pinhole_200", "kmk-15", None),
        ("kmk_15", "kmk_15", "b_kmk_15", "kmk-15", 400),
    ]
    signal_type_arr = [
        ("signal", "signal", "signal", 10),
    ]
    exposures_arr = [
        # ([200, 1000, 5000, 25000, 125000],
        # "cam_exp_down_200us"),
        ([24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24000, 48000, 96000, 200000, 400000, 800000, 1600000],
        "cam_exp_down_24us"),
    ]
    calibration_params = {
        "kmk_15": {
            "x_shift_m": 0.00017178838009236094,
            "y_shift_m": 4.903750079380921e-05,
            "cam_pixel_width_m": 1.911283425948221e-06,
            "width_pix_x_m": 8.098304067336103e-06,
            "width_pix_y_m": 0.0002024576016834026,
            "coef_lin_to_cam": 0.31081315123650427,
            "shift_lin_m": 0.02034698896918196,
            "pix_max_ampl": 1751.9944216945232,
        },
    }
    for particles_cases in particles_cases_arr:
        for signal_type in signal_type_arr:
            for exposures in exposures_arr:
                darl_dir_name = f"darl-{particles_cases[3]}-{signal_type[2]}"
                cfg = ExperimentConfig(
                    dir_case=Path(f"test_17_07_26_kmk_15/{particles_cases[0]}/"),
                    # dir_signal_rel=f"data/{particles_cases[0]}/",
                    
                    detector_configuration_type=1,

                    # dir_signal_rel=f"юстировка",
                    dir_signal_rel=f"сигнал",
                    dir_back_rel=f"фон",
                    # dir_save_rel=f"restore_cam_l2h_10_bl12_per_exposure/20_07_26_{particles_cases[1]}_{signal_type[1]}_{exposures[1]}/",
                    dir_save_rel=f"restore_cam_l2h_10_bl12/28_07_26_{particles_cases[1]}_{signal_type[1]}_{exposures[1]}/",
                    
                    path_signal_darl_rel = f"{darl_dir_name}/{particles_cases[2]}.txt" if particles_cases[2] is not None else None,
                    
                    use_w_critical=False,
                    use_chahine=True,
                    use_conc_corr=True,

                    matrix_name=f"{darl_dir_name}/matrix-case_conf_real_data_test_17_07_26-{particles_cases[3]}-cam-lin-{signal_type[2]}-w_633.0.npz",
                    bins_name=f"{darl_dir_name}/bins_front_detector.txt",
                    classes_name=f"{darl_dir_name}/particle_classes_lasser_0.txt",
                    # bins_lin_name=f"{darl_dir_name}/FrontDetectorLogLine_detector.txt",
                    x_shift_m = calibration_params[particles_cases[0]]["x_shift_m"],
                    y_shift_m = calibration_params[particles_cases[0]]["y_shift_m"],
                    cam_pixel_width_m = calibration_params[particles_cases[0]]["cam_pixel_width_m"],

                    # cam_hdr_diff_mode = "per_exposure", # "per_exposure" or "after_hdr"
                    cam_hdr_diff_mode = "after_hdr", # "per_exposure" or "after_hdr"
                    # cam_hdr_mode="l2h_longest",
                    cam_hdr_mode="l2h",
                    cam_hdr_low_thr=10,
                    cam_hdr_back_level=12,
                    cam_hdr_top_thr=255-15,
                    cam_hdr_filtered=False,
                    cam_hdr_gauss_sigma=5,
                    
                    signal_type=signal_type[0],

                    # exposure_time_arr=[25, 50, 100, 200, 400, 2000, 10000, 50000, 250000, 1250000],
                    exposure_time_arr=exposures[0],
                    
                    cut_classes=signal_type[3],
                    cut_classes_top=None,

                    # dir_signal_lin_rel=f"сигнал/",
                    # dir_back_lin_rel=f"фон/",
                    # filename_lin_template = f"lin-{particles_cases[3]}-{{}}us.txt",
                    # filename_lin_back_template = f"lin-back-{particles_cases[3]}-{{}}us.txt",
                    # width_pix_x_m = calibration_params[particles_cases[0]]["width_pix_x_m"],
                    # width_pix_y_m = calibration_params[particles_cases[0]]["width_pix_y_m"],
                    # coef_lin_to_cam = calibration_params[particles_cases[0]]["coef_lin_to_cam"],
                    # shift_lin_m = calibration_params[particles_cases[0]]["shift_lin_m"],
                    # pix_max_ampl = calibration_params[particles_cases[0]]["pix_max_ampl"],
                    # exposure_time_us_lin_arr = [particles_cases[4]],
                    # lin_signal_mode=2,
                )
                cfg_arr.append((cfg, run_cfg_lin_cut))



    particles_cases_arr = [
        # ("kmk_15", "pinhole_200", "b_pinhole_200", "kmk-15", None),
        ("kmk_15", "kmk_15", "b_kmk_15", "kmk-15", 400),
    ]
    signal_type_arr = [
        ("signal", "signal", "signal", 10),
    ]
    exposures_arr = [
        # ([200, 1000, 5000, 25000, 125000],
        # "cam_exp_down_200us"),
        ([24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24000, 48000, 96000, 200000, 400000, 800000, 1600000],
        "cam_exp_down_24us"),
    ]
    calibration_params = {
        "kmk_15": {
            "x_shift_m": 0.00017178838009236094,
            "y_shift_m": 4.903750079380921e-05,
            "cam_pixel_width_m": 1.911283425948221e-06,
            "width_pix_x_m": 8.098304067336103e-06,
            "width_pix_y_m": 0.0002024576016834026,
            "coef_lin_to_cam": 0.31081315123650427,
            "shift_lin_m": 0.02034698896918196,
            "pix_max_ampl": 1751.9944216945232,
        },
    }
    for particles_cases in particles_cases_arr:
        for signal_type in signal_type_arr:
            for exposures in exposures_arr:
                darl_dir_name = f"darl-{particles_cases[3]}-{signal_type[2]}"
                cfg = ExperimentConfig(
                    dir_case=Path(f"test_17_07_26_kmk_15/{particles_cases[0]}/"),
                    # dir_signal_rel=f"data/{particles_cases[0]}/",
                    
                    detector_configuration_type=1,

                    # dir_signal_rel=f"юстировка",
                    dir_signal_rel=f"сигнал",
                    dir_back_rel=f"фон",
                    dir_save_rel=f"restore_cam_l2h_10_bl12_per_exposure/28_07_26_{particles_cases[1]}_{signal_type[1]}_{exposures[1]}/",
                    
                    path_signal_darl_rel = f"{darl_dir_name}/{particles_cases[2]}.txt" if particles_cases[2] is not None else None,
                    
                    use_w_critical=False,
                    use_chahine=True,
                    use_conc_corr=True,

                    matrix_name=f"{darl_dir_name}/matrix-case_conf_real_data_test_17_07_26-{particles_cases[3]}-cam-lin-{signal_type[2]}-w_633.0.npz",
                    bins_name=f"{darl_dir_name}/bins_front_detector.txt",
                    classes_name=f"{darl_dir_name}/particle_classes_lasser_0.txt",
                    # bins_lin_name=f"{darl_dir_name}/FrontDetectorLogLine_detector.txt",
                    x_shift_m = calibration_params[particles_cases[0]]["x_shift_m"],
                    y_shift_m = calibration_params[particles_cases[0]]["y_shift_m"],
                    cam_pixel_width_m = calibration_params[particles_cases[0]]["cam_pixel_width_m"],

                    cam_hdr_diff_mode = "per_exposure", # "per_exposure" or "after_hdr"
                    # cam_hdr_diff_mode = "after_hdr", # "per_exposure" or "after_hdr"
                    # cam_hdr_mode="l2h_longest",
                    cam_hdr_mode="l2h",
                    cam_hdr_low_thr=10,
                    cam_hdr_back_level=12,
                    cam_hdr_top_thr=255-15,
                    cam_hdr_filtered=False,
                    cam_hdr_gauss_sigma=5,
                    
                    signal_type=signal_type[0],

                    # exposure_time_arr=[25, 50, 100, 200, 400, 2000, 10000, 50000, 250000, 1250000],
                    exposure_time_arr=exposures[0],
                    
                    cut_classes=signal_type[3],
                    cut_classes_top=None,

                    # dir_signal_lin_rel=f"сигнал/",
                    # dir_back_lin_rel=f"фон/",
                    # filename_lin_template = f"lin-{particles_cases[3]}-{{}}us.txt",
                    # filename_lin_back_template = f"lin-back-{particles_cases[3]}-{{}}us.txt",
                    # width_pix_x_m = calibration_params[particles_cases[0]]["width_pix_x_m"],
                    # width_pix_y_m = calibration_params[particles_cases[0]]["width_pix_y_m"],
                    # coef_lin_to_cam = calibration_params[particles_cases[0]]["coef_lin_to_cam"],
                    # shift_lin_m = calibration_params[particles_cases[0]]["shift_lin_m"],
                    # pix_max_ampl = calibration_params[particles_cases[0]]["pix_max_ampl"],
                    # exposure_time_us_lin_arr = [particles_cases[4]],
                    # lin_signal_mode=2,
                )
                cfg_arr.append((cfg, run_cfg_lin_cut))


    print(f"Всего конфигураций: {len(cfg_arr)}")
    return cfg_arr


def make_cfg_arr_from_params():
    import re


    SOURCE_RESTORE_DIR = Path(__file__).resolve().parent / "10_07_26/restore_water_cam_l2h_0"
    NEW_DATE = "14_07_26_new_nnls"


    def make_new_dir_save_rel(source_dir_save_rel: Path) -> Path:
        """Заменяет дату в имени каталога результата, сохраняя его родителя."""
        new_name, replacements = re.subn(
            r"\d{2}_\d{2}_\d{2}",
            NEW_DATE,
            source_dir_save_rel.name,
            count=1,
        )
        if replacements != 1:
            raise ValueError(
                "В имени каталога результата не найдена дата: "
                f"{source_dir_save_rel.name}"
            )
        return source_dir_save_rel.parent / new_name


    def build_configurations(
        source_restore_dir: Path = SOURCE_RESTORE_DIR,
    ) -> list[ExperimentConfig]:
        """Читает параметры всех непосредственных подкаталогов с ``params.txt``."""
        cfg_arr = []
        params_paths = sorted(
            path
            for path in source_restore_dir.glob("*/params.txt")
            if NEW_DATE not in path.parent.name
        )
        for params_path in params_paths:
            # dir_case будет перезаписан значением из params.txt.
            cfg = ExperimentConfig(dir_case=Path("."))
            cfg.load_params(params_path)
            cfg.dir_save_rel = make_new_dir_save_rel(cfg.dir_save_rel)
            cfg_arr.append(cfg)

        if not cfg_arr:
            raise FileNotFoundError(
                f"Не найдены файлы params.txt в {source_restore_dir}"
            )
        return cfg_arr


    cfg_arr = build_configurations()

    return cfg_arr


def run_cfg(cfg:ExperimentConfig):
    print(f"Запуск конфигурации")
    exposure_time = np.max(cfg.exposure_time_arr)
    cfg.dir_save.mkdir(exist_ok=True, parents=True)
    cfg.save_params()

    if cfg.path_signal_darl_rel is not None:
        darl_signal_b = np.loadtxt(cfg.path_signal_darl)
        darl_signal_b_norm = darl_signal_b / np.max(darl_signal_b)

    bins_cam = pd.read_csv(
        cfg.file_bins,
        sep=r"\s+",
        comment="#",
        names=["r_in_m", "r_out_m"],
    )

    bins_lin = pd.read_csv(
        cfg.file_bins_lin,
        sep="\t"
    )

    A = np.load(cfg.file_matrix)["matrix"].astype(np.float64)
    classes = read_classes(cfg.file_classes)

    cut_classes = cfg.cut_classes
    cut_classes_top = cfg.cut_classes_top
    if not(cut_classes is None and cut_classes_top is None):
        A_cutted = A[:, cut_classes:cut_classes_top]
        classes_cutted = classes[cut_classes:cut_classes_top]


    signal_cam, cam_mask_valid = get_signal_cam_hdr(cfg)

    if cfg.cam_hdr_filtered:
        signal_cam = gaussian_filter(signal_cam, sigma=(cfg.cam_hdr_gauss_sigma, cfg.cam_hdr_gauss_sigma))

    dir_save=cfg.dir_save/"signal-img/"
    dir_save.mkdir(exist_ok=True, parents=True)
    plot_cam_signal_valid(
        signal_cam,
        cam_mask_valid,
        dir_save/f"signal-hdr-valid-ex_time_{exposure_time}.png",
    )

    b_cam = get_signal_cam(
        bins_cam,
        None,
        None,
        cam_pixel_width_m=cfg.cam_pixel_width_m,
        data=signal_cam,
        x_shift_pix=cfg.x_shift_pix,
        y_shift_pix=cfg.y_shift_pix,
        plotted=True,
        path_save_dir=cfg.dir_save/"signal-img/",
        postfix=f"{exposure_time}",
        signal_type=cfg.signal_type,
        detector_configuration_type=cfg.detector_configuration_type,
    )
    b_cam_norm = b_cam/exposure_time

    plt.figure(figsize=(10, 5))
    plt.plot(b_cam, marker="o")
    plt.xlabel("№ бинa")
    plt.ylabel("Суммарный сигнал")
    plt.grid()
    plt.tight_layout()
    plt.savefig(cfg.dir_save/f"signal-cam-ex_time_{exposure_time}.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(b_cam_norm, marker="o")
    plt.xlabel("№ бинa")
    plt.ylabel("Суммарный сигнал")
    plt.grid()
    plt.tight_layout()
    plt.savefig(cfg.dir_save/f"signal-cam-norm-ex_time_{exposure_time}.png")
    plt.close()

    for exposure_time_us_lin in cfg.exposure_time_us_lin_arr:
        print(f"\nexposure_time_us_lin={exposure_time_us_lin}")
        filename_lin = cfg.dir_signal_lin / cfg.filename_lin_template.format(exposure_time_us_lin)
        filename_lin_back = None
        if cfg.dir_back_lin is not None:
            filename_lin_back = cfg.dir_back_lin/cfg.filename_lin_back_template.format(exposure_time_us_lin)
        b_lin, _ = get_signal_lin(
            bins_lin,
            filename_lin,
            filename_lin_back,
            num_pix=cfg.num_pix_lin,
            shift_lin_m=cfg.shift_lin_m,
            pix_max_ampl=cfg.pix_max_ampl,
            width_pix_x_m=cfg.width_pix_x_m,
            width_pix_y_m=cfg.width_pix_y_m,
            signal_type=cfg.signal_type,
            mode=cfg.lin_signal_mode,
            path_save_dir=cfg.dir_save,
        )

        b_lin_norm = b_lin/(exposure_time_us_lin+cfg.lin_time_add)*cfg.coef_lin_to_cam
        b = np.hstack((b_cam_norm, b_lin_norm))

        np.savetxt(
            cfg.dir_save / "reference_camera_signal.txt",
            np.column_stack((b_cam, b_cam_norm)),
            delimiter="\t",
            header="raw\tnormalized",
            comments="",
            fmt="%.17g",
        )
        np.savetxt(
            cfg.dir_save / "reference_line_signal.txt",
            np.column_stack((b_lin, b_lin_norm)),
            delimiter="\t",
            header="raw\tnormalized",
            comments="",
            fmt="%.17g",
        )
        np.savetxt(
            cfg.dir_save / "reference_combined_signal.txt", b, fmt="%.17g"
        )

        plt.figure(figsize=(10, 5))
        plt.plot(b_lin, marker="o")
        plt.xlabel("№ бинa")
        plt.ylabel("Суммарный сигнал")
        plt.grid()
        plt.tight_layout()
        plt.savefig(cfg.dir_save/f"signal-lin-ex_time_lin_{exposure_time_us_lin}.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(b_lin_norm, marker="o")
        plt.xlabel("№ бинa")
        plt.ylabel("Суммарный сигнал")
        plt.grid()
        plt.tight_layout()
        plt.savefig(cfg.dir_save/f"signal-lin-norm-ex_time_lin_{exposure_time_us_lin}.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(b, marker="o")
        plt.xlabel("№ бинa")
        plt.ylabel("Суммарный сигнал")
        plt.grid()
        plt.tight_layout()
        plt.savefig(cfg.dir_save/f"signal-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}.png")
        plt.close()


        noise = np.zeros_like(b)
        b_back = np.zeros_like(b)


        (
            sizes, 
            restored_distr_reg, 
            restored_distr_reg_iter, 
            restored_distr_reg_iter_w, 
            restored_gost_reg, 
            restored_gost_reg_iter, 
            restored_gost_reg_iter_w, 
            gcv_curve,
            alpha_reg
        ) = inverse_solver_type_1(
            A, b, noise, b_back, classes, 
            cfg.use_w_critical,
            cfg.use_chahine,
            cfg.use_conc_corr,
        )
        save_reference_restoration(
            cfg.dir_save,
            "",
            sizes,
            restored_distr_reg,
            restored_distr_reg_iter,
            restored_distr_reg_iter_w,
            restored_gost_reg,
            restored_gost_reg_iter,
            restored_gost_reg_iter_w,
            gcv_curve,
            alpha_reg,
        )
        plot_gcv_curve(gcv_curve, alpha_reg, cfg.dir_save/f"gcv_curve-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}.png")

        plt.figure(figsize=(10, 5))
        plt.plot(sizes, restored_distr_reg, label="reg")
        plt.plot(sizes, restored_distr_reg_iter, label="reg+iter")
        # plt.plot(sizes, restored_distr_reg_iter_w, label="reg+iter+w_cut")
        plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel("Размер частицы (нм)")
        plt.ylabel("Количество частиц, шт")
        plt.title(f"Восстановленное распределение количества частиц камерой")
        plt.legend()
        plt.savefig(cfg.dir_save/f"restored_distr-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}.png")
        plt.close()
        np.savetxt(
            cfg.dir_save/f"restored_distr-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}.txt",
            np.column_stack((sizes, restored_distr_reg, restored_distr_reg_iter)),
            delimiter="\t",
            header="sizes\trestored_distr_reg\trestored_distr_reg_iter",
            comments="",
            fmt="%g"
        )
        

        plt.figure(figsize=(10, 5))
        plt.plot(sizes, restored_distr_reg, label="reg")
        plt.plot(sizes, restored_distr_reg_iter, label="reg+iter")
        # plt.plot(sizes, restored_distr_reg_iter_w, label="reg+iter+w_cut")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Размер частицы (нм)")
        plt.ylabel("Количество частиц, шт")
        plt.title(f"Восстановленное распределение количества частиц камерой")
        plt.legend()
        plt.savefig(cfg.dir_save/f"restored_distr-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}-ylog.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(sizes, restored_gost_reg, label="reg")
        plt.plot(sizes, restored_gost_reg_iter, label="reg+iter")
        # plt.plot(sizes, restored_gost_reg_iter_w, label="reg+iter+w_cut")
        plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel("Размер частицы (нм)")
        plt.ylabel("Доля от ГОСТ концентрации")
        plt.title(f"Восстановленное распределение ГОСТ концентраций камерой")
        plt.legend()
        # y_limits = plt.ylim()
        # plt.savefig(cfg.dir_save/f"restored_gost_reg-ex_time_{exposure_time}.png")
        plt.savefig(cfg.dir_save/f"restored_gost-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(sizes, restored_gost_reg, label="reg")
        plt.plot(sizes, restored_gost_reg_iter, label="reg+iter")
        # plt.plot(sizes, restored_gost_reg_iter_w, label="reg+iter+w_cut")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Размер частицы (нм)")
        plt.ylabel("Доля от ГОСТ концентрации")
        plt.title(f"Восстановленное распределение ГОСТ концентраций камерой")
        plt.legend()
        # y_limits = plt.ylim()
        # plt.savefig(cfg.dir_save/f"restored_gost_reg-ex_time_{exposure_time}.png")
        plt.savefig(cfg.dir_save/f"restored_gost-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}-ylog.png")
        plt.close()


        if not(cut_classes is None and cut_classes_top is None):
            (
                sizes_cutted, 
                restored_distr_reg_cutted, 
                restored_distr_reg_iter_cutted, 
                restored_distr_reg_iter_w_cutted,
                restored_gost_reg_cutted,
                restored_gost_reg_iter_cutted,
                restored_gost_reg_iter_w_cutted,
                gcv_curve_cutted,
                alpha_reg_cutted
            ) = inverse_solver_type_1(
                A_cutted, b, noise, b_back, classes_cutted, 
                cfg.use_w_critical,
                cfg.use_chahine,
                cfg.use_conc_corr,
            )
            save_reference_restoration(
                cfg.dir_save,
                f"cutted_{cut_classes}_",
                sizes_cutted,
                restored_distr_reg_cutted,
                restored_distr_reg_iter_cutted,
                restored_distr_reg_iter_w_cutted,
                restored_gost_reg_cutted,
                restored_gost_reg_iter_cutted,
                restored_gost_reg_iter_w_cutted,
                gcv_curve_cutted,
                alpha_reg_cutted,
            )
            plot_gcv_curve(gcv_curve_cutted, alpha_reg_cutted, cfg.dir_save/f"cutted_{cut_classes}_gcv_curve-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}.png")

            plt.figure(figsize=(10, 5))
            plt.plot(sizes_cutted, restored_distr_reg_cutted, label="reg")
            plt.plot(sizes_cutted, restored_distr_reg_iter_cutted, label="reg+iter")
            # plt.plot(sizes_cutted, restored_distr_reg_iter_w_cutted, label="reg+iter+w_cut")
            plt.xscale("log")
            # plt.yscale("log")
            plt.xlabel("Размер частицы (нм)")
            plt.ylabel("Количество частиц, шт")
            plt.title(f"Восстановленное распределение количества частиц камерой")
            plt.legend()
            plt.savefig(cfg.dir_save/f"cutted_{cut_classes}_restored_distr-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}.png")
            plt.close()
            np.savetxt(
                cfg.dir_save/f"cutted_{cut_classes}_restored_distr-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}.txt",
                np.column_stack((sizes_cutted, restored_distr_reg_cutted, restored_distr_reg_iter_cutted)),
                delimiter="\t",
                header="sizes\trestored_distr_reg\trestored_distr_reg_iter",
                comments="",
                fmt="%g"
            )

            plt.figure(figsize=(10, 5))
            plt.plot(sizes_cutted, restored_distr_reg_cutted, label="reg")
            plt.plot(sizes_cutted, restored_distr_reg_iter_cutted, label="reg+iter")
            # plt.plot(sizes_cutted, restored_distr_reg_iter_w_cutted, label="reg+iter+w_cut")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Размер частицы (нм)")
            plt.ylabel("Количество частиц, шт")
            plt.title(f"Восстановленное распределение количества частиц камерой")
            plt.legend()
            plt.savefig(cfg.dir_save/f"cutted_{cut_classes}_restored_distr-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}-ylog.png")
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(sizes_cutted, restored_gost_reg_cutted, label="reg")
            plt.plot(sizes_cutted, restored_gost_reg_iter_cutted, label="reg+iter")
            # plt.plot(sizes_cutted, restored_gost_reg_iter_w_cutted, label="reg+iter+w_cut")
            plt.xscale("log")
            # plt.yscale("log")
            plt.xlabel("Размер частицы (нм)")
            plt.ylabel("Доля от ГОСТ концентрации")
            plt.title(f"Восстановленное распределение ГОСТ концентраций камерой")
            plt.legend()
            # y_limits = plt.ylim()
            # plt.savefig(cfg.dir_save/f"restored_gost_reg-ex_time_{exposure_time}.png")
            plt.savefig(cfg.dir_save/f"cutted_{cut_classes}_restored_gost-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}.png")
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(sizes_cutted, restored_gost_reg_cutted, label="reg")
            plt.plot(sizes_cutted, restored_gost_reg_iter_cutted, label="reg+iter")
            # plt.plot(sizes_cutted, restored_gost_reg_iter_w_cutted, label="reg+iter+w_cut")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Размер частицы (нм)")
            plt.ylabel("Доля от ГОСТ концентрации")
            plt.title(f"Восстановленное распределение ГОСТ концентраций камерой")
            plt.legend()
            # y_limits = plt.ylim()
            # plt.savefig(cfg.dir_save/f"restored_gost_reg-ex_time_{exposure_time}.png")
            plt.savefig(cfg.dir_save/f"cutted_{cut_classes}_restored_gost-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}-ylog.png")
            plt.close()


        if cfg.path_signal_darl_rel is not None:
            b_real_norm = b / np.max(b)
            plt.figure(figsize=(10, 5))
            plt.plot(b_real_norm, marker="o", markersize=3, label="real norm")
            plt.plot(darl_signal_b_norm, marker="o", markersize=3, label="darl norm")
            plt.xlabel("№ бинa")
            plt.ylabel("Нормированный сигнал")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(cfg.dir_save/f"compare-real-darl-signals-all-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}.png")
            plt.close()
            np.savetxt(
                cfg.dir_save/f"compare-real-darl-signals-all-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}.txt",
                np.column_stack((b_real_norm, darl_signal_b_norm)),
                delimiter="\t",
                header="b_real_norm\tdarl_signal_b_norm",
                comments="",
                fmt="%g"
            )

        # plt.figure(figsize=(10, 5))
        # plt.plot(b_real_norm, marker="o", markersize=3, label="real norm")
        # plt.plot(darl_signal_b_norm, marker="o", markersize=3, label="darl norm")
        # x_lin_start = b_cam.shape[0]
        # x_lin_stop = len(b_real_norm)
        # plt.xlim(x_lin_start-0.5, x_lin_stop)
        # # plt.ylim(0.0, np.max([b_real_norm[x_lin_start:x_lin_stop], darl_signal_b_norm[x_lin_start:x_lin_stop]]))
        # plt.ylim(0.0, np.max([b_real_norm[x_lin_start:x_lin_stop], darl_signal_b_norm[x_lin_start:x_lin_stop]]))
        # plt.xlabel("№ бинa")
        # plt.ylabel("Нормированный сигнал")
        # plt.grid()
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(cfg.dir_save/f"compare-real-darl-signals-lin-ex_time_{exposure_time}-ex_time_lin_{exposure_time_us_lin}.png")
        # plt.close()
    print("Готово")


def run_cfg_lin_cut(cfg):
    exposure_time = np.max(cfg.exposure_time_arr)
    cfg.dir_save.mkdir(exist_ok=True, parents=True)
    cfg.save_params()


    bins_cam = pd.read_csv(
        cfg.file_bins,
        sep=r"\s+",
        comment="#",
        names=["r_in_m", "r_out_m"],
    )

    signal_cam, cam_mask_valid = get_signal_cam_hdr(cfg)

    if cfg.cam_hdr_filtered:
        signal_cam = gaussian_filter(signal_cam, sigma=(cfg.cam_hdr_gauss_sigma, cfg.cam_hdr_gauss_sigma))

    dir_save=cfg.dir_save/"signal-img/"
    dir_save.mkdir(exist_ok=True, parents=True)
    plot_cam_signal_valid(
        signal_cam,
        cam_mask_valid,
        dir_save/f"signal-hdr-valid-ex_time_{exposure_time}.png",
    )

    b_cam = get_signal_cam(
        bins_cam,
        None,
        None,
        data=signal_cam,
        cam_pixel_width_m=cfg.cam_pixel_width_m,
        x_shift_pix=cfg.x_shift_pix,
        y_shift_pix=cfg.y_shift_pix,
        plotted=True,
        path_save_dir=cfg.dir_save/"signal-img/",
        postfix=f"{exposure_time}",
        signal_type=cfg.signal_type,
        detector_configuration_type=cfg.detector_configuration_type,
    )
    b_cam_norm = b_cam/exposure_time

    lin_cut = b_cam_norm.shape[0]

    if cfg.path_signal_darl_rel is not None:
        darl_signal_b = np.loadtxt(cfg.path_signal_darl)[:lin_cut]
        darl_signal_b_norm = darl_signal_b / np.max(darl_signal_b)

    A = np.load(cfg.file_matrix)["matrix"].astype(np.float64)[:lin_cut, :]
    classes = read_classes(cfg.file_classes)

    cut_classes = cfg.cut_classes
    cut_classes_top = cfg.cut_classes_top
    if not(cut_classes is None and cut_classes_top is None):
        A_cutted = A[:, cut_classes:cut_classes_top]
        classes_cutted = classes[cut_classes:cut_classes_top]

    print(f"{A.shape=}, {classes.shape=}, {b_cam_norm.shape=}")

    plt.figure(figsize=(10, 5))
    plt.plot(b_cam, marker="o")
    plt.xlabel("№ бинa")
    plt.ylabel("Суммарный сигнал")
    plt.grid()
    plt.tight_layout()
    plt.savefig(cfg.dir_save/f"signal-cam-ex_time_{exposure_time}.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(b_cam_norm, marker="o")
    plt.xlabel("№ бинa")
    plt.ylabel("Суммарный сигнал")
    plt.grid()
    plt.tight_layout()
    plt.savefig(cfg.dir_save/f"signal-cam-norm-ex_time_{exposure_time}.png")
    plt.close()

    b = b_cam_norm

    np.savetxt(
        cfg.dir_save / "reference_camera_signal.txt",
        np.column_stack((b_cam, b_cam_norm)),
        delimiter="\t",
        header="raw\tnormalized",
        comments="",
        fmt="%.17g",
    )

    noise = np.zeros_like(b)
    b_back = np.zeros_like(b)


    (
        sizes, 
        restored_distr_reg, 
        restored_distr_reg_iter, 
        restored_distr_reg_iter_w, 
        restored_gost_reg, 
        restored_gost_reg_iter, 
        restored_gost_reg_iter_w, 
        gcv_curve,
        alpha_reg
    ) = inverse_solver_type_1(
        A, b, noise, b_back, classes, 
        cfg.use_w_critical,
        cfg.use_chahine,
        cfg.use_conc_corr,
    )
    save_reference_restoration(
        cfg.dir_save,
        "",
        sizes,
        restored_distr_reg,
        restored_distr_reg_iter,
        restored_distr_reg_iter_w,
        restored_gost_reg,
        restored_gost_reg_iter,
        restored_gost_reg_iter_w,
        gcv_curve,
        alpha_reg,
    )
    plot_gcv_curve(gcv_curve, alpha_reg, cfg.dir_save/f"gcv_curve-ex_time_{exposure_time}.png")

    plt.figure(figsize=(10, 5))
    plt.plot(sizes, restored_distr_reg, label="reg")
    plt.plot(sizes, restored_distr_reg_iter, label="reg+iter")
    # plt.plot(sizes, restored_distr_reg_iter_w, label="reg+iter+w_cut")
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("Размер частицы (нм)")
    plt.ylabel("Количество частиц, шт")
    plt.title(f"Восстановленное распределение количества частиц камерой")
    plt.legend()
    plt.savefig(cfg.dir_save/f"restored_distr-ex_time_{exposure_time}.png")
    plt.close()
    np.savetxt(
        cfg.dir_save/f"restored_distr-ex_time_{exposure_time}.txt",
        np.column_stack((sizes, restored_distr_reg, restored_distr_reg_iter)),
        delimiter="\t",
        header="sizes\trestored_distr_reg\trestored_distr_reg_iter",
        comments="",
        fmt="%g"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(sizes, restored_distr_reg, label="reg")
    plt.plot(sizes, restored_distr_reg_iter, label="reg+iter")
    # plt.plot(sizes, restored_distr_reg_iter_w, label="reg+iter+w_cut")
    plt.xscale("log")
    plt.yscale("log")
    bottom, top = plt.ylim()
    if bottom < 1e-16:
        bottom=1e-16
    top = 10 * (max(np.max(restored_distr_reg), np.max(restored_distr_reg_iter)) - bottom)
    plt.ylim((bottom, top))
    plt.xlabel("Размер частицы (нм)")
    plt.ylabel("Количество частиц, шт")
    plt.title(f"Восстановленное распределение количества частиц камерой")
    plt.legend()
    plt.savefig(cfg.dir_save/f"restored_distr-ex_time_{exposure_time}-ylog.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(sizes, restored_gost_reg, label="reg")
    plt.plot(sizes, restored_gost_reg_iter, label="reg+iter")
    # plt.plot(sizes, restored_gost_reg_iter_w, label="reg+iter+w_cut")
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("Размер частицы (нм)")
    plt.ylabel("Доля от ГОСТ концентрации")
    plt.title(f"Восстановленное распределение ГОСТ концентраций камерой")
    plt.legend()
    # y_limits = plt.ylim()
    # plt.savefig(cfg.dir_save/f"restored_gost_reg-ex_time_{exposure_time}.png")
    plt.savefig(cfg.dir_save/f"restored_gost-ex_time_{exposure_time}.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(sizes, restored_gost_reg, label="reg")
    plt.plot(sizes, restored_gost_reg_iter, label="reg+iter")
    # plt.plot(sizes, restored_gost_reg_iter_w, label="reg+iter+w_cut")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Размер частицы (нм)")
    plt.ylabel("Доля от ГОСТ концентрации")
    plt.title(f"Восстановленное распределение ГОСТ концентраций камерой")
    plt.legend()
    # y_limits = plt.ylim()
    # plt.savefig(cfg.dir_save/f"restored_gost_reg-ex_time_{exposure_time}.png")
    plt.savefig(cfg.dir_save/f"restored_gost-ex_time_{exposure_time}-ylog.png")
    plt.close()


    if not(cut_classes is None and cut_classes_top is None):
        (
            sizes_cutted, 
            restored_distr_reg_cutted, 
            restored_distr_reg_iter_cutted, 
            restored_distr_reg_iter_w_cutted,
            restored_gost_reg_cutted,
            restored_gost_reg_iter_cutted,
            restored_gost_reg_iter_w_cutted,
            gcv_curve_cutted,
            alpha_reg_cutted
        ) = inverse_solver_type_1(
            A_cutted, b, noise, b_back, classes_cutted, 
            cfg.use_w_critical,
            cfg.use_chahine,
            cfg.use_conc_corr,
        )
        save_reference_restoration(
            cfg.dir_save,
            f"cutted_{cut_classes}_",
            sizes_cutted,
            restored_distr_reg_cutted,
            restored_distr_reg_iter_cutted,
            restored_distr_reg_iter_w_cutted,
            restored_gost_reg_cutted,
            restored_gost_reg_iter_cutted,
            restored_gost_reg_iter_w_cutted,
            gcv_curve_cutted,
            alpha_reg_cutted,
        )
        plot_gcv_curve(gcv_curve_cutted, alpha_reg_cutted, cfg.dir_save/f"cutted_{cut_classes}_gcv_curve-ex_time_{exposure_time}.png")

        plt.figure(figsize=(10, 5))
        plt.plot(sizes_cutted, restored_distr_reg_cutted, label="reg")
        plt.plot(sizes_cutted, restored_distr_reg_iter_cutted, label="reg+iter")
        # plt.plot(sizes_cutted, restored_distr_reg_iter_w_cutted, label="reg+iter+w_cut")
        plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel("Размер частицы (нм)")
        plt.ylabel("Количество частиц, шт")
        plt.title(f"Восстановленное распределение количества частиц камерой")
        plt.legend()
        plt.savefig(cfg.dir_save/f"cutted_{cut_classes}_restored_distr-ex_time_{exposure_time}.png")
        plt.close()
        np.savetxt(
            cfg.dir_save/f"cutted_{cut_classes}_restored_distr-ex_time_{exposure_time}.txt",
            np.column_stack((sizes_cutted, restored_distr_reg_cutted, restored_distr_reg_iter_cutted)),
            delimiter="\t",
            header="sizes\trestored_distr_reg\trestored_distr_reg_iter",
            comments="",
            fmt="%g"
        )

        plt.figure(figsize=(10, 5))
        plt.plot(sizes_cutted, restored_distr_reg_cutted, label="reg")
        plt.plot(sizes_cutted, restored_distr_reg_iter_cutted, label="reg+iter")
        # plt.plot(sizes_cutted, restored_distr_reg_iter_w_cutted, label="reg+iter+w_cut")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Размер частицы (нм)")
        plt.ylabel("Количество частиц, шт")
        plt.title(f"Восстановленное распределение количества частиц камерой")
        plt.legend()
        plt.savefig(cfg.dir_save/f"cutted_{cut_classes}_restored_distr-ex_time_{exposure_time}-ylog.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(sizes_cutted, restored_gost_reg_cutted, label="reg")
        plt.plot(sizes_cutted, restored_gost_reg_iter_cutted, label="reg+iter")
        # plt.plot(sizes_cutted, restored_gost_reg_iter_w_cutted, label="reg+iter+w_cut")
        plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel("Размер частицы (нм)")
        plt.ylabel("Доля от ГОСТ концентрации")
        plt.title(f"Восстановленное распределение ГОСТ концентраций камерой")
        plt.legend()
        # y_limits = plt.ylim()
        # plt.savefig(cfg.dir_save/f"restored_gost_reg-ex_time_{exposure_time}.png")
        plt.savefig(cfg.dir_save/f"cutted_{cut_classes}_restored_gost-ex_time_{exposure_time}.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(sizes_cutted, restored_gost_reg_cutted, label="reg")
        plt.plot(sizes_cutted, restored_gost_reg_iter_cutted, label="reg+iter")
        # plt.plot(sizes_cutted, restored_gost_reg_iter_w_cutted, label="reg+iter+w_cut")
        plt.xscale("log")
        plt.yscale("log")
        bottom, top = plt.ylim()
        if bottom < 1e-16:
            bottom=1e-16
        top = 10 * (max(np.max(restored_distr_reg), np.max(restored_distr_reg_iter)) - bottom)
        plt.ylim((bottom, top))
        plt.xlabel("Размер частицы (нм)")
        plt.ylabel("Доля от ГОСТ концентрации")
        plt.title(f"Восстановленное распределение ГОСТ концентраций камерой")
        plt.legend()
        # y_limits = plt.ylim()
        # plt.savefig(cfg.dir_save/f"restored_gost_reg-ex_time_{exposure_time}.png")
        plt.savefig(cfg.dir_save/f"cutted_{cut_classes}_restored_gost-ex_time_{exposure_time}-ylog.png")
        plt.close()


    if cfg.path_signal_darl_rel is not None:
        b_real_norm = b / np.max(b)
        plt.figure(figsize=(10, 5))
        plt.plot(b_real_norm, marker="o", markersize=3, label="real norm")
        plt.plot(darl_signal_b_norm, marker="o", markersize=3, label="darl norm")
        plt.xlabel("№ бинa")
        plt.ylabel("Нормированный сигнал")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(cfg.dir_save/f"compare-real-darl-signals-all-ex_time_{exposure_time}.png")
        plt.close()
        np.savetxt(
            cfg.dir_save/f"compare-real-darl-signals-all-ex_time_{exposure_time}.txt",
            np.column_stack((b_real_norm, darl_signal_b_norm)),
            delimiter="\t",
            header="b_real_norm\tdarl_signal_b_norm",
            comments="",
            fmt="%g"
        )

    print("Готово")


if __name__ == "__main__":
    cfg_arr = make_cfg_arr()
    # cfg_arr = make_cfg_arr_from_params()
    for cfg, func in cfg_arr:
        print(cfg.dir_save)
        func(cfg)
        # run_cfg(cfg)
        # run_cfg_lin_cut(cfg)
