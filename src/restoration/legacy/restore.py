import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.interpolate import interp1d
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter

from calibration.legacy import func as legacy_func
from calibration.legacy.func import (
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
from restoration.legacy.config import (
    LegacyRestoreConfigArtifact,
    LegacyRestoreConfigError,
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


def run_legacy_restore(artifact: LegacyRestoreConfigArtifact) -> None:
    """Execute one adapted configuration with unchanged legacy numerics."""
    if not isinstance(artifact, LegacyRestoreConfigArtifact):
        raise LegacyRestoreConfigError(
            "artifact: ожидался LegacyRestoreConfigArtifact"
        )
    runners = {
        "camera": run_cfg_lin_cut,
        "camera_line": run_cfg,
    }
    if artifact.run_mode not in runners:
        raise LegacyRestoreConfigError(
            f"Неизвестный режим legacy-восстановления: {artifact.run_mode!r}"
        )

    previous_solver_settings = (
        legacy_func.REGULARIZATION_TYPE,
        legacy_func.REGULARIZATION_ALPHA,
        legacy_func.W_CRITICAL,
    )
    try:
        legacy_func.REGULARIZATION_TYPE = artifact.solver.regularization_type
        legacy_func.REGULARIZATION_ALPHA = artifact.solver.regularization_alpha
        legacy_func.W_CRITICAL = artifact.solver.w_critical
        runners[artifact.run_mode](artifact.config)
    finally:
        (
            legacy_func.REGULARIZATION_TYPE,
            legacy_func.REGULARIZATION_ALPHA,
            legacy_func.W_CRITICAL,
        ) = previous_solver_settings


if __name__ == "__main__":
    raise SystemExit(
        "Legacy-модуль не запускается напрямую; используйте src/restore.py"
    )
