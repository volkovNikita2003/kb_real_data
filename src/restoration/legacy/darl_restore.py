from __future__ import annotations

import sys
import shutil
import json
import os
from pathlib import Path

import matplotlib


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(
    os.environ.get(
        "REAL_DATA_AUTO_CODE_GIT_DIR",
        SCRIPT_DIR.parents[3] / "code_git",
    )
).resolve()

# The project is used directly from its source checkout. This affects only
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


figsize_det=(10, 4)
figsize_dist=(20, 4)


def plotter_single(plot_func, path_to_save, figsize=(20, 4), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    plot_func(fig, ax, **kwargs)
    fig.tight_layout()
    fig.savefig(path_to_save)
    plt.close()


def get_signal_from_restore(classes, restored_distr, config_name, dir_save: Path, prefix):
    log(f"Load config: {CONFIGS_DIR / (config_name + '.txt')}")
    load_config_to_state(CONFIGS_DIR, config_name)

    distributions = []
    log("Добавление сигнала в виде прямоугольников")    
    x_dist_mean = np.array([(item[0] + item[1])/2 for item in classes])
    widths = np.array([(item[1] - item[0]) for item in classes])
    for i in range(len(x_dist_mean)):
        d_nm = x_dist_mean[i]
        width = widths[i]
        x_min_nm = max(d_nm - width/2, 10)
        x_max_nm = d_nm + width/2
        count = restored_distr[i] * width
        if count <= 0:
            continue
        dist_type = "rectangle"
        dist_params = {"active": 1, "x_min": x_min_nm, "x_max": x_max_nm, "count": count, "comment": "rectangle"}
        row_index, row = add_dist_row(dist_type, STATE, dist_params)
        log(f"Добавлено распределение: {dist_params}")
        distributions.append((dist_type, row, row_index))
    path_save = dir_save/f"{prefix}rect_distribution.png"
    plotter_single(plot_dist, path_save, state=STATE, figsize=figsize_dist)
    log(f"save distribution to {path_save}")
    
    log("Calc indicatrices for rect distribution")
    dist_build_indicatrices()
    
    log("Calc signal for rect distribution")
    handler = ProgressHandler()
    results = dist_calc_signal_and_indic(progress_callback=handler._progress)
    handler.close()
    
    if len(results) != 1:
        raise RuntimeError(
            f"Ожидался один DARL-сигнал, получено: {len(results)}"
        )
    LocalParams, b, e, b_back, indic_full, indic = results[0]

    for i in range(len(distributions)):
        dist = distributions[i]
        dist_type, row, row_index = dist
        upd_dist_row(dist_type, row_index, "active", 0, STATE)
        distributions[i][1]["active"] = 0

    return b
