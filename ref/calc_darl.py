from __future__ import annotations

import sys
import shutil
from pathlib import Path

import matplotlib


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1] / "code_git"
# AUTO_DIR = SCRIPT_DIR / "auto"

# The project is used directly from its source checkout.  This affects only
# module lookup; all project data paths are resolved inside code_git itself.
sys.path.insert(0, str(PROJECT_DIR))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config.params import CONFIGS_DIR
from logic.calc import (
    create_plot_b_state,
    dist_build_indicatrices,
    dist_calc_signal_and_indic,
    dist_calc_signal_back,
    plot_b,
    plot_b_back,
    plot_b_without_back,
    plot_indic,
)
from logic.computations import compute_matrix_wrapper
from logic.config import load_config_to_state
from logic.input_distr import add_dist_row, plot_dist, upd_dist_row
from logic.logger import log
from logic.logic_inverse_model import comparing_conf
from logic.progress_bar_terminal import ProgressHandler
from logic.state import STATE


CONFIG_NAME = "conf_real_data_test_17_07_26-kmk-15-cam-lin-signal"
RESULT_DIR = SCRIPT_DIR / "test_17_07_26_kmk_15/kmk_15/darl-kmk-15-signal"
FIGSIZE_DETECTORS = (10, 4)
DISTRIBUTIONS = [
    ("gauss", {"active": 0, "mean": 200000.0, "sigma": 500.0, "count": 2e4, "comment": "pinhole_200"}, None),

    ("gauss", {"active": 0, "mean": 15000.0, "sigma": 3000.0, "count": 1e6, "comment": "kmk_15"}, None),
    # ("gauss", {"active": 0, "mean": 25000.0, "sigma": 3000.0, "count": 1e6, "comment": "kmk_25"}, None),
    # ("gauss", {"active": 0, "mean": 100000.0, "sigma": 10000.0, "count": 2e4, "comment": "kmk_100"}, None),
    # ("gauss", {"active": 0, "mean": 160000.0, "sigma": 10000.0, "count": 2e4, "comment": "kmk_160"}, None),
    # ("gauss", {"active": 0, "mean": 270000.0, "sigma": 20000.0, "count": 2e4, "comment": "kmk_270"}, None),

    # ("gauss", {"active": 0, "mean": 200000.0, "sigma": 10000.0, "count": 2e4, "comment": "solder_200"}, None),
    # ("gauss", {"active": 0, "mean": 300000.0, "sigma": 10000.0, "count": 1e4, "comment": "solder_300"}, None),
    # ("gauss", {"active": 0, "mean": 350000.0, "sigma": 10000.0, "count": 1e4, "comment": "solder_350"}, None),
    # ("gauss", {"active": 0, "mean": 400000.0, "sigma": 10000.0, "count": 1e4, "comment": "solder_400"}, None),
    # ("gauss", {"active": 0, "mean": 450000.0, "sigma": 10000.0, "count": 1e4, "comment": "solder_450"}, None),
    # ("gauss", {"active": 0, "mean": 500000.0, "sigma": 10000.0, "count": 1e4, "comment": "solder_500"}, None),

    # ("gauss", {"active": 0, "mean": 225000.0, "sigma": 25000.0, "count": 2e4, "comment": "cmc_225"}, None),
    # ("gauss", {"active": 0, "mean": 650000.0, "sigma": 25000.0, "count": 1e3, "comment": "cmc_650"}, None),
]
SIGNAL_FILENAMES = {
    # "pinhole": "b_200_0.5.txt",
    # "kmk-100": "b_100_10.txt",
    # "kmk-160": "b_160_10.txt",
    # "kmk-270": "b_270_20.txt",
    # "solder-300": "b_300_10.txt",
    # "cmc-650": "b_650_25.txt",
}
for dist_type, row, _ in DISTRIBUTIONS:
    particle_name = row["comment"]
    SIGNAL_FILENAMES[particle_name] = f"b_{particle_name}.txt"
# MATRIX_FILENAME = "matrix-case_conf_real_data_10_07_26_glass-cam-calib-signal-w_633.0.npz"


def save_background_plot(results) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_DETECTORS)
    state = create_plot_b_state()
    for index, (local_params, background) in enumerate(results):
        plot_b_back(fig, ax, state, local_params, background)
        np.savetxt(
            RESULT_DIR / f"background_signal_laser_{index}.txt",
            background,
            fmt="%.17g",
        )
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "background_signal.png")
    plt.close(fig)


def calculate_distribution(dist_type: str, row: dict, row_index: int) -> None:
    slug = row["comment"]
    output_dir = RESULT_DIR / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Activate distribution: {slug}")
    upd_dist_row(dist_type, row_index, "active", 1, STATE)
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_dist(fig, ax, STATE)
        fig.tight_layout()
        fig.savefig(output_dir / f"{slug}_source_distribution.png")
        plt.close(fig)

        log(f"Build indicatrices: {slug}")
        dist_build_indicatrices()

        log(f"Calculate signal: {slug}")
        handler = ProgressHandler()
        results = dist_calc_signal_and_indic(progress_callback=handler._progress)
        handler.close()

        if len(results) != 1:
            raise RuntimeError(f"Expected one laser result, got {len(results)}")
        # b is the raw modeled signal: noise and background have not been added.
        _, raw_signal, *_ = results[0]
        np.savetxt(
            RESULT_DIR / SIGNAL_FILENAMES[slug], raw_signal, fmt="%.17g"
        )
        np.savetxt(
            output_dir / "modeled_signal.txt", raw_signal, fmt="%.17g"
        )

        fig_b, ax_b = plt.subplots(figsize=FIGSIZE_DETECTORS)
        fig_without, ax_without = plt.subplots(figsize=FIGSIZE_DETECTORS)
        fig_indic = plt.figure()
        ax_indic = fig_indic.add_subplot(111, projection="polar")
        b_state = create_plot_b_state()
        without_state = create_plot_b_state()

        for local_params, b, noise, background, indic_full, indic in results:
            plot_b(fig_b, ax_b, b_state, local_params, b, noise)
            plot_b_without_back(
                fig_without, ax_without, without_state,
                local_params, b, noise, background,
            )
            color = {633: "r", 488: "b"}.get(local_params.LASER_WAVELENGTH_VACUUM_NM)
            if indic_full is not None:
                plot_indic(fig_indic, ax_indic, indic_full, color, local_params.LASER_ANGEL_DEG, local_params.LASER_NUM, False)
            if indic is not None:
                plot_indic(fig_indic, ax_indic, indic, color, local_params.LASER_ANGEL_DEG, local_params.LASER_NUM, True)

        for fig, filename in (
            (fig_b, f"{slug}_signal_with_background.png"),
            (fig_without, f"{slug}_signal_without_background.png"),
            (fig_indic, f"{slug}_indicatrix.png"),
        ):
            fig.tight_layout()
            fig.savefig(output_dir / filename)
            plt.close(fig)

        log(f"Restore distribution: {slug}")
        restored_paths = comparing_conf(output_dir)
        descriptive_names = (
            f"{slug}_restoration_details.png",
            f"{slug}_restoration_without_background.png",
            f"{slug}_restoration_with_background.png",
        )
        for old_path, new_name in zip(restored_paths, descriptive_names):
            Path(old_path).replace(output_dir / new_name)
        plt.close("all")
    finally:
        upd_dist_row(dist_type, row_index, "active", 0, STATE)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    # AUTO_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Load config: {CONFIGS_DIR / (CONFIG_NAME + '.txt')}")
    load_config_to_state(CONFIGS_DIR, CONFIG_NAME)

    distributions = []
    for dist_type, row, _ in DISTRIBUTIONS:
        row_index, stored_row = add_dist_row(dist_type, STATE, row.copy())
        distributions.append((dist_type, stored_row, row_index))

    log("Calculate hardware matrix")
    # коррекно работает только для одного лазера. 
    # handler = ProgressHandler()
    # compute_matrix_wrapper(progress_cb=handler._progress)
    # handler.close()
    # Для двух и больше лазеров лучше использовать
    compute_matrix_wrapper(log_cb=log)

    from config.params import ParamsClass
    from logic.state import export_state_dict
    local_params = ParamsClass(export_state_dict())
    local_params.update_laser(local_params.LASER_NUM_ARR[0])
    matrix_filename = local_params.filename_matrix_template.format(
        local_params.case_name, local_params.LASER_WAVELENGTH_VACUUM_NM
    )
    matrix_path = local_params.folder_matrices / matrix_filename
    shutil.copy2(matrix_path, RESULT_DIR)
    
    case_folder = local_params.case_folder
    filenames = [
        "bins_front_detector.txt",
        "FrontDetectorLogLine_detector.txt",
        f"particle_classes_lasser_{local_params.LASER_NUM}.txt"
    ]
    for filename in filenames:
        filename_path = case_folder / filename
        if filename_path.is_file():
            shutil.copy2(filename_path, RESULT_DIR)
        else:
            print(f"Файл не найден: {filename_path}")


    log("Calculate background signal")
    handler = ProgressHandler()
    save_background_plot(dist_calc_signal_back(progress_callback=handler._progress))
    handler.close()

    for dist_type, row, row_index in distributions:
        calculate_distribution(dist_type, row, row_index)

    log(f"All plots saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
