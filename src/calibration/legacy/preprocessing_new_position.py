import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from .func import (
    ExperimentConfig,
    get_signal_cam_hdr,
    find_circle,
    calc_matrix_shift_m_new_position,
    calc_pinhole_diam_m,
    calc_pinhole_diam_m_lin,
    jinc,
    get_bmp_numeric_names,
    calculate_fit_metrics,
    print_parameter_errors,
)
from .config import load_legacy_config
from .result_adapter import save_legacy_result


parser = argparse.ArgumentParser(description="Калибровка камеры и линейки")
parser.add_argument(
    "experiment",
    type=Path,
    help="Путь к директории автоматизированного эксперимента",
)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
cfg, experiment, automated_parameters = load_legacy_config(
    args.experiment, args.output_dir
)

PATH_CASE_DIR = cfg.dir_case
PATH_CASE_DIR.mkdir(exist_ok=True, parents=True)
PATH_SAVE_PREPROCESSING_DIR = cfg.dir_save_preprocessing
PATH_SAVE_PREPROCESSING_DIR.mkdir(exist_ok=True, parents=True)

cfg.exposure_time_arr = get_bmp_numeric_names(cfg.dir_signal)
print(f"Используемые экспозции: {cfg.exposure_time_arr  }")


gaussian_sigma_cam_calib = cfg.calib_cam_gaussian_sigma
pix_img_width_m = cfg.cam_pixel_width_m
lamb_um = cfg.labm_um
F_lens_um = cfg.F_lens_um
d_um_ref = cfg.calib_d_pinhole_um
cam_correct = cfg.calib_cam_corrected


is_add_lin = False if cfg.dir_signal_lin is None else True
print(f"{cfg.dir_signal_lin=}, {is_add_lin=}")
if is_add_lin:
    assert len(cfg.exposure_time_us_lin_arr) == 1, "Wrong value" 
    exposure_time_us_lin = cfg.exposure_time_us_lin_arr[0] + cfg.lin_time_add
    filename_lin = cfg.dir_signal_lin/cfg.filename_lin_template
    position_pinhole_m = cfg.calib_lin_position_pinhole_m
    position_signal_m  = cfg.calib_lin_position_signal_m
    width_pix_x_m = cfg.width_pix_x_m
    width_pix_y_m = cfg.width_pix_y_m
    F_lens_m = cfg.F_lens_um/1e6
    num_pix = cfg.num_pix_lin
    gaussian_sigma_lin_calib = cfg.calib_lin_gaussian_sigma

# print(f"{PATH_CASE_DIR=}")
# print(f"{PATH_SAVE_PREPROCESSING_DIR=}")
# print(f"{cfg.dir_case=}")
# print(f"{cfg.dir_signal_rel=}")
# print(f"{cfg.exposure_time_arr=}")
# print(f"{cfg.cam_pixel_width_m=}")
# print(f"{cfg.cam_hdr_diff_mode=}")
# print(f"{cfg.cam_hdr_mode=}")
# print(f"{cfg.cam_hdr_low_thr=}")
# print(f"{cfg.cam_hdr_back_level=}")
# print(f"{cfg.cam_hdr_filtered=}")
# print(f"{cfg.cam_hdr_gauss_sigma=}")
# print(f"{cfg.lin_time_add=}")

# print(f"{gaussian_sigma_cam_calib=}")
# print(f"{pix_img_width_m=}")
# print(f"{lamb_um=}")
# print(f"{F_lens_um=}")
# print(f"{d_um_ref=}")
# print(f"{cam_correct=}")
# print(f"{exposure_time_us_lin=}")
# print(f"{filename_lin=}")
# print(f"{position_pinhole_m=}")
# print(f"{position_signal_m=}")
# print(f"{width_pix_x_m=}")
# print(f"{width_pix_y_m=}")
# print(f"{F_lens_m=}")
# print(f"{num_pix=}")
# print(f"{gaussian_sigma_lin_calib=}")

# exit()





# ---------- Калибровка ----------

# ----- Камера -----
filename_cam = f"{cfg.dir_signal_rel}/..."
exposure_time_us_cam = max(cfg.exposure_time_arr)
data, mask_valid_pix = get_signal_cam_hdr(cfg)
sigma = gaussian_sigma_cam_calib
filtered_data = gaussian_filter(data, sigma=(sigma, sigma))

plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='gray')
plt.colorbar(label='Intensity')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.title(filename_cam)
plt.tight_layout()
plt.savefig(PATH_SAVE_PREPROCESSING_DIR/f"cam_calib_hdr_orig.png")
plt.close()

plt.figure(figsize=(10, 8))
plt.imshow(filtered_data, cmap='gray')
plt.colorbar(label='Intensity')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.title(f"gaussian_filter, sigma={sigma}")
plt.tight_layout()
plt.savefig(PATH_SAVE_PREPROCESSING_DIR/f"cam_calib_hdr_gaussian_filtered.png")
plt.close()


y_lines = range(400, 1501, 1)
x_min_find_min = 0
x_max_find_min = 1000
x_lines = range(0, 400, 1)
y_bound = int(1944//2)
radii = [600, 700, 900, 1250]

center_cam, radius_cam = find_circle(
    filtered_data,
    1,
    plotted=True,
    y_lines=y_lines,
    x_min_find_min=x_min_find_min,
    x_max_find_min=x_max_find_min,
    x_lines=x_lines,
    y_bound=y_bound,
    radii=radii,
    save_dir=PATH_SAVE_PREPROCESSING_DIR
)
cam_shift_um = calc_matrix_shift_m_new_position(center_cam)*1e6
diam_um_pinhole_cam = calc_pinhole_diam_m(radius_cam)*1e6
print(f"center_cam = {center_cam}, radius_cam = {radius_cam}")
print(f"смещение матрицы для DARL: {cam_shift_um} мкм")
print(f"Диаметр пинхола по первому дифракционному минимуму: {diam_um_pinhole_cam} мкм")


x_pixels = np.linspace(0, data.shape[1], data.shape[1], endpoint=False)
x_distance_m_cam = (x_pixels-center_cam[0])*pix_img_width_m
x_distance_um_cam = x_distance_m_cam*1e6

y_line = int(center_cam[1])
line_width = int(200e-6/pix_img_width_m/2)
signal_middle = data[y_line, :]
signal_middle_avg = np.mean(data[y_line-line_width:y_line+line_width, :], axis=0)
signal_middle_f = filtered_data[y_line, :]
signal_middle_avg_f = np.mean(filtered_data[y_line-line_width:y_line+line_width, :], axis=0)
signal_norm_cam = signal_middle_avg_f / exposure_time_us_cam / pix_img_width_m**2
signal_middle_norm_cam = signal_middle / exposure_time_us_cam / pix_img_width_m**2

plt.figure(figsize=(10, 5))
plt.plot(x_distance_um_cam, signal_middle, label="signal_middle")
plt.plot(x_distance_um_cam, signal_middle_avg, label="signal_middle_avg")
plt.plot(x_distance_um_cam, signal_middle_f, label="signal_middle_f")
plt.plot(x_distance_um_cam, signal_middle_avg_f, label="signal_middle_avg_f")
plt.xlabel('X, um')
plt.ylabel('Intensity')
plt.title(f"Сигнал пинхола с камеры по оси X при Y={y_line}, line_width={line_width}")
plt.legend()
plt.tight_layout()
plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"cam_signals_middle.png")
plt.close()


# Модель
def model_cam(x_um, A, d_um, x_gap_um):
    x_j = np.pi * d_um / lamb_um * (x_um-x_gap_um)/F_lens_um
    return A * jinc(x_j)**2
x_gap_um_2 = 0
def model2_cam(x_um, A, d_um):
    x_j = np.pi * d_um / lamb_um * (x_um-x_gap_um_2)/F_lens_um
    return A * jinc(x_j)**2
d_um_3 = diam_um_pinhole_cam
def model3_cam(x_um, A):
    x_j = np.pi * d_um_3 / lamb_um * (x_um-x_gap_um_2)/F_lens_um
    return A * jinc(x_j)**2
def model4_cam(x_um, A, x_gap_um):
    x_j = np.pi * d_um_3 / lamb_um * (x_um-x_gap_um)/F_lens_um
    return A * jinc(x_j)**2


x_fit_orig = x_distance_um_cam
y_fit_orig = signal_norm_cam
# y_fit_orig = signal_middle_avg_f
x_fit = x_fit_orig
y_fit = y_fit_orig


p0 = [
    max(y_fit),
    diam_um_pinhole_cam,
    0.0
]
popt, pcov = curve_fit(model_cam, x_fit, y_fit, p0=p0)
A_cam, d_um, x_gap_um = popt
print("\nModel 1 parameter errors:")
print_parameter_errors(["A", "d_um", "x_gap_um"], popt, pcov,)
cam_fit = model_cam(x_fit_orig, *popt)


p0_2 = [
    max(y_fit),
    diam_um_pinhole_cam,
]
popt2, pcov2 = curve_fit(model2_cam, x_fit, y_fit, p0=p0_2)
A_cam2, d_um2 = popt2
print("\nModel 2 parameter errors:")
print_parameter_errors(["A", "d_um"], popt2, pcov2,)
cam_fit_2 = model2_cam(x_fit_orig, *popt2)


p0_3 = [
    max(y_fit),
]
popt3, pcov3 = curve_fit(model3_cam, x_fit, y_fit, p0=p0_3)
A_cam3 = float(popt3[0])
print("\nModel 3 parameter errors:")
print_parameter_errors(["A"], popt3, pcov3,)
cam_fit_3 = model3_cam(x_fit_orig, *popt3)


# p0_4 = [
#     max(y_fit),
#     0
# ]
# popt4, pcov4 = curve_fit(model4_cam, x_fit, y_fit, p0=p0_4)
# A_cam4, x_gap_um4 = popt4
# print("\nModel 4 parameter errors:")
# print_parameter_errors(["A", "x_gap_um"], popt4, pcov4,)
# cam_fit_4 = model4_cam(x_fit_orig, *popt4)


# Построение
plt.figure(figsize=(10, 5))
plt.plot(x_fit_orig, y_fit_orig, lw=2, label="Fitted data")
plt.plot(x_fit_orig, signal_middle_norm_cam, lw=1, zorder=0, label="signal_middle_norm_cam")
plt.plot(x_fit_orig, cam_fit, 'r', lw=1, label=f'\nFit\nd={d_um:3.1f} um\nx_gap={x_gap_um:4.1f} um\n')
plt.plot(x_fit_orig, cam_fit_2, 'g', lw=1, label=f'Fit2\nd={d_um2:3.1f} um')
plt.plot(x_fit_orig, cam_fit_3, 'b', lw=1, label=f'Fit3')
# plt.plot(x_fit_orig, cam_fit_4, 'black', lw=1, label=f'Fit4\nx_gap={x_gap_um4:4.1f} um')
plt.xlabel('X, um')
plt.ylabel('Signal')
plt.legend()
plt.tight_layout()
plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"cam_calib_fits.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(x_fit_orig, y_fit_orig, lw=3, label="Fitted data")
plt.plot(x_fit_orig, signal_middle_norm_cam, lw=1, zorder=0, label="signal_middle_norm_cam")
plt.plot(x_fit_orig, cam_fit, 'r', lw=1, label=f'\nFit\nd={d_um:3.1f} um\nx_gap={x_gap_um:4.1f} um\n')
plt.plot(x_fit_orig, cam_fit_2, 'g', lw=1, label=f'Fit2\nd={d_um2:3.1f} um')
plt.plot(x_fit_orig, cam_fit_3, 'b', lw=1, label=f'Fit3')
# plt.plot(x_fit_orig, cam_fit_4, 'black', lw=1, label=f'Fit4\nx_gap={x_gap_um4:4.1f} um')
plt.ylim(0, np.max(y_fit_orig)*0.03)
plt.xlabel('X, um')
plt.ylabel('Signal')
plt.legend()
plt.tight_layout()
plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"cam_calib_fits_zoom.png")
plt.close()




# models = {
#     # "Fit: A, d, x_gap": {
#         # "function": model_cam,
#         # "parameters": popt,
#         # "n_params": 3,
#     # },
#     "Fit2: A, d": {
#         "function": model2_cam,
#         "parameters": popt2,
#         "n_params": 2,
#     },
#     # "Fit3: A": {
#     #     "function": model3_cam,
#     #     "parameters": popt3,
#     #     "n_params": 1,
#     # },
#     # "Fit4: A, x_gap": {
#     #     "function": model4_cam,
#     #     "parameters": popt4,
#     #     "n_params": 2,
#     # },
# }

# quality_rows = []

# for model_name, model_info in models.items():
#     function = model_info["function"]
#     parameters = model_info["parameters"]
#     n_params = model_info["n_params"]

#     y_pred = function(x_fit, *parameters)

#     metrics = calculate_fit_metrics(
#         y_true=y_fit,
#         y_pred=y_pred,
#         n_model_params=n_params,
#     )

#     quality_rows.append({
#         "Model": model_name,
#         **metrics,
#     })

# quality_df = pd.DataFrame(quality_rows)
# quality_df = quality_df.set_index("Model")

# # Модели сортируются от лучшего BIC к худшему
# quality_df = quality_df.sort_values("BIC")

# quality_df["Delta_AICc"] = (
#     quality_df["AICc"] - quality_df["AICc"].min()
# )

# quality_df["Delta_BIC"] = (
#     quality_df["BIC"] - quality_df["BIC"].min()
# )


# tail_threshold = 0.03 * np.max(y_fit)
# tail_mask = y_fit <= tail_threshold

# tail_rows = []

# for model_name, model_info in models.items():
#     function = model_info["function"]
#     parameters = model_info["parameters"]

#     y_pred = function(x_fit, *parameters)

#     residuals_tail = (
#         y_fit[tail_mask] - y_pred[tail_mask]
#     )

#     tail_rows.append({
#         "Model": model_name,
#         "Tail_RMSE": np.sqrt(
#             np.mean(residuals_tail**2)
#         ),
#         "Tail_MAE": np.mean(
#             np.abs(residuals_tail)
#         ),
#         "Tail_MaxAbsError": np.max(
#             np.abs(residuals_tail)
#         ),
#     })

# tail_quality_df = pd.DataFrame(tail_rows).set_index("Model")

# quality_df = quality_df.join(tail_quality_df)


# columns_to_print = [
#     "RMSE",
#     "MAE",
#     "NRMSE_peak",
#     "R2",
#     "R2_adjusted",
#     "AICc",
#     "Delta_AICc",
#     "BIC",
#     "Delta_BIC",
#     "MeanResidual",
#     "DurbinWatson",
#     "Tail_RMSE",
#     "Tail_MAE",
#     "Tail_MaxAbsError"
# ]

# print("\nFit quality:")
# print(
#     quality_df[columns_to_print].to_string(
#         float_format=lambda value: f"{value:.6g}"
#     )
# )

# quality_df.to_csv(
#     PATH_SAVE_PREPROCESSING_DIR / "cam_fit_quality.csv"
# )



# plt.figure(figsize=(10, 5))

# for model_name, model_info in models.items():
#     function = model_info["function"]
#     parameters = model_info["parameters"]

#     y_pred = function(x_fit, *parameters)
#     residuals = y_fit - y_pred

#     plt.plot(
#         x_fit,
#         residuals,
#         lw=1,
#         label=model_name,
#     )

# plt.axhline(0, color="black", linestyle="--", linewidth=1)
# plt.xlabel("X, um")
# plt.ylabel("Data - model")
# plt.legend()
# plt.tight_layout()

# plt.savefig(
#     PATH_SAVE_PREPROCESSING_DIR / "cam_fit_residuals.png",
#     dpi=200,
# )

# plt.close()















koef_scale_cam = 1
if cam_correct:
    koef_scale_cam = d_um2 / d_um_ref
print(f"Коэффициент растяжения размеров камеры: {koef_scale_cam}")
x_distance_m_cam_cor = x_distance_m_cam * koef_scale_cam
x_distance_um_cam_cor = x_distance_um_cam * koef_scale_cam
pix_img_width_m_cor = pix_img_width_m * koef_scale_cam
diag_matrix_img_mm = pix_img_width_m_cor * np.sqrt(cfg.W**2 + cfg.H**2) * 1e3

cam_shift_um_cor = koef_scale_cam * cam_shift_um

print(f"Скорректированный размер пикселя изображения: {pix_img_width_m_cor*1e6} мкм")
print(f"Диагональ матрицы: {diag_matrix_img_mm} мм")
print(f"Скорректированное смещение матрицы для DARL: {cam_shift_um_cor} мкм")



x_fit_orig = x_distance_um_cam_cor
y_fit_orig = signal_norm_cam
# y_fit_orig = signal_middle_avg_f
x_fit = x_fit_orig
y_fit = y_fit_orig


p0 = [
    max(y_fit),
    diam_um_pinhole_cam,
    0.0
]
popt, pcov = curve_fit(model_cam, x_fit, y_fit, p0=p0)
A_cam, d_um, x_gap_um = popt
print(f"A = {A_cam:.4f}")
print(f"d_um = {d_um:.4f}")
print(f"x_gap_um = {x_gap_um:.4f}")
cam_fit = model_cam(x_fit_orig, *popt)


p0_2 = [
    max(y_fit),
    diam_um_pinhole_cam,
]
popt2, pcov2 = curve_fit(model2_cam, x_fit, y_fit, p0=p0_2)
A_cam2, d_um2 = popt2
print()
print(f"A2 = {A_cam2:.4f}")
print(f"d_um2 = {d_um2:.4f}")
cam_fit_2 = model2_cam(x_fit_orig, *popt2)

# Построение
plt.figure(figsize=(10, 5))
plt.plot(x_fit_orig, y_fit_orig, lw=2, label="Fitted data")
plt.plot(x_fit_orig, signal_middle_norm_cam, lw=1, zorder=0, label="signal_middle_norm_cam")
plt.plot(x_fit_orig, cam_fit, 'r', lw=1, label=f'\nFit\nd={d_um:3.1f} um\nx_gap={x_gap_um:4.1f} um\n')
plt.plot(x_fit_orig, cam_fit_2, 'g', lw=1, label=f'Fit2\nd={d_um2:3.1f} um')
plt.xlabel('X, um')
plt.ylabel('Signal')
plt.legend()
plt.tight_layout()
plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"cam_calib_fits_correction.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(x_fit_orig, y_fit_orig, lw=3, label="Fitted data")
plt.plot(x_fit_orig, signal_middle_norm_cam, lw=1, zorder=0, label="signal_middle_norm_cam")
plt.plot(x_fit_orig, cam_fit, 'r', lw=1, label=f'\nFit\nd={d_um:3.1f} um\nx_gap={x_gap_um:4.1f} um\n')
plt.plot(x_fit_orig, cam_fit_2, 'g', lw=1, label=f'Fit2\nd={d_um2:3.1f} um')
plt.ylim(0, np.max(y_fit_orig)*0.03)
plt.xlabel('X, um')
plt.ylabel('Signal')
plt.legend()
plt.tight_layout()
plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"cam_calib_fits_zoom_correction.png")
plt.close()


if is_add_lin:
    # ----- Линейка -----
    print(f"\nКалибровка линейки")
    shift_lin_m = position_signal_m - position_pinhole_m

    # TODO: получать сигнал с линейки функцией. 
    # Идейно везде получение сигнала (сырой сигнал + обработка) 
    # должны выполняться одинаково 
    df = pd.read_csv(
        filename_lin,
        sep="\t",
        decimal=",",
    )

    pixels_data = df.iloc[:, 0].astype(int).to_numpy()
    amplitude = df.iloc[:, 1].to_numpy()
    pixels_data = pixels_data.max() - pixels_data

    sort_idx = np.argsort(pixels_data)
    pixels = pixels_data[sort_idx]
    signal_lin = amplitude[sort_idx] - amplitude.min()


    signal_lin_orig = signal_lin.copy()
    signal_lin = gaussian_filter(signal_lin, sigma=gaussian_sigma_lin_calib)


    signal_norm_lin = signal_lin / exposure_time_us_lin / width_pix_x_m / width_pix_y_m
    signal_orig_norm_lin = signal_lin_orig / exposure_time_us_lin / width_pix_x_m / width_pix_y_m
    np.savetxt(PATH_SAVE_PREPROCESSING_DIR/"signal_lin.txt", signal_lin)
    np.savetxt(PATH_SAVE_PREPROCESSING_DIR/"signal_norm_lin.txt", signal_norm_lin)
    print(f"{exposure_time_us_lin=}")
    print(f"{width_pix_x_m=}")
    print(f"{width_pix_y_m=}")


    plt.figure(figsize=(10, 5))
    plt.plot(pixels, signal_lin_orig, linewidth=1, linestyle="-", markersize=1, label="orig")
    # plt.plot(pixels_data, amplitude, linewidth=1, linestyle="-", markersize=1, label="amplitude")
    plt.plot(pixels, signal_lin, linewidth=1, linestyle="-", markersize=1, label=f"gauss_filter, sigma={gaussian_sigma_lin_calib}")
    plt.xlim(1500, 2100)
    plt.xlabel("Номер пикселя")
    plt.ylabel("Амплитуда")
    plt.title("signal_lin")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"lin_calib_signal_pix.png")
    plt.close()


    plt.figure(figsize=(10, 5))
    plt.plot(pixels, signal_norm_lin, linewidth=1, linestyle="-", marker="o", markersize=1)
    plt.xlabel("Номер пикселя")
    plt.ylabel("Амплитуда")
    plt.title("signal_norm_lin")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"lin_calib_signal_norm_pix.png")
    plt.close()


    pix_max_ampl = np.argmax(signal_lin)
    # pix_max_ampl += 20
    print(f"Макс. амплитуда: {np.max(signal_lin)} при пикселе {pix_max_ampl}")

    # TODO: у пользователя должен быть контроль над этим параметром.
    # Либо подумать об автоматическом нахождении положения максимума.
    # Либо задать эти параметры в текстовый конфиг. Если результат плохой,
    # то менять конфиг и перезапускать 
    find_pix_1_diff_min_start = pix_max_ampl + 150
    find_pix_1_diff_min_stop = pix_max_ampl + 250
    pix_1_diff_min = np.argmin(signal_lin[find_pix_1_diff_min_start:find_pix_1_diff_min_stop]) + find_pix_1_diff_min_start
    amp_min = signal_lin[pix_1_diff_min]
    width_1_diff_min_m = (pix_1_diff_min - pix_max_ampl) * width_pix_x_m
    phi_1_diff_min_rad = np.arctan(width_1_diff_min_m / F_lens_m)
    diam_um_pinhole_lin = calc_pinhole_diam_m_lin(width_1_diff_min_m)*1e6
    print(f"Полуширина первого дифракционного минимума: {width_1_diff_min_m*1000:.3f} мм; угол: {np.degrees(phi_1_diff_min_rad):.6f} град; пиксель: {pix_1_diff_min}")
    print(f"Диаметр пинхола по первому дифракционному минимуму: {diam_um_pinhole_lin} мкм")

    coord_left_side_m = shift_lin_m - (pix_max_ampl * width_pix_x_m)
    coord_right_side_m = shift_lin_m + ((num_pix - pix_max_ampl) * width_pix_x_m)
    phi_left_rad = np.arctan(coord_left_side_m / F_lens_m)
    phi_right_rad = np.arctan(coord_right_side_m / F_lens_m)
    print(f"Координата левого края линейки в рабочем положении: {coord_left_side_m*1000:.3f} мм; угол: {np.degrees(phi_left_rad):.6f} град")
    print(f"Координата правого края линейки в рабочем положении: {coord_right_side_m*1000:.3f} мм; угол: {np.degrees(phi_right_rad):.6f} град")


    x_min = max(pix_max_ampl-100, 0)
    x_max = min(pix_max_ampl+100, num_pix)
    plt.figure(figsize=(10, 5))
    plt.plot(pixels[x_min:x_max], signal_lin_orig[x_min:x_max], linewidth=1, linestyle="-", markersize=1, label="orig")
    plt.plot(pixels[x_min:x_max], signal_lin[x_min:x_max], linewidth=1, linestyle="-", marker="o", markersize=1, label=f"gauss_filter, sigma={gaussian_sigma_lin_calib}")
    plt.xlabel("Номер пикселя")
    plt.ylabel("Амплитуда")
    plt.title("signal_lin")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"lin_calib_signal_max.png")
    plt.close()

    x_min = max(pix_1_diff_min-10, 0)
    x_max = min(pix_1_diff_min+180, num_pix)
    plt.figure(figsize=(10, 5))
    plt.plot(pixels[x_min:x_max], signal_lin_orig[x_min:x_max], linewidth=1, linestyle="-", markersize=1, label="orig")
    plt.plot(pixels[x_min:x_max], signal_lin[x_min:x_max], linewidth=1, linestyle="-", marker="o", markersize=1, label=f"gauss_filter, sigma={gaussian_sigma_lin_calib}")
    plt.xlabel("Номер пикселя")
    plt.ylabel("Амплитуда")
    plt.title("signal_lin")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"lin_calib_signal_1dif_max.png")
    plt.close()



    x_distance_m_lin = (pixels - pix_max_ampl) * width_pix_x_m
    x_distance_um_lin = x_distance_m_lin * 1e6

    plt.figure(figsize=(10, 5))
    plt.plot(x_distance_um_lin, signal_lin, linewidth=1, linestyle="-", marker="o", markersize=1)
    plt.xlabel("X, um")
    plt.ylabel("Амплитуда")
    plt.title("signal_lin")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"lin_calib_signal_norm_shift_1.png")
    plt.close()



    # Модель
    def model_lin(x_um, A, d_um, x_gap_um):
        x_j = np.pi * d_um / lamb_um * (x_um-x_gap_um)/F_lens_um
        return A * jinc(x_j)**2


    x_fit_orig = x_distance_um_lin
    # y_fit_orig = signal_norm_cam
    y_fit_orig = signal_norm_lin
    x_fit = x_fit_orig
    y_fit = y_fit_orig


    p0 = [
        max(y_fit),
        diam_um_pinhole_lin,
        0.0
    ]
    print("before fit")
    print(f"{p0[0]=}")
    print(f"{p0[1]=}")
    print(f"{p0[2]=}")
    print(f"{F_lens_um=}, {lamb_um=}")
    popt, pcov = curve_fit(model_lin, x_fit, y_fit, p0=p0)
    A_lin, d_um, x_gap_um = popt
    print("after fit")
    print(f"A = {A_lin:.4f}")
    print(f"d_um = {d_um:.4f}")
    print(f"x_gap_um = {x_gap_um:.4f}")
    lin_fit = model_lin(x_fit_orig, *popt)


    # Построение
    plt.figure(figsize=(10, 5))
    plt.plot(x_fit_orig, y_fit_orig, lw=2, label="Fitted data")
    # plt.plot(x_fit_orig, signal_middle, lw=1, zorder=0, label="signal_middle")
    plt.plot(x_fit_orig, lin_fit, 'r', lw=1, label=f'\nFit\nd={d_um:3.1f} um\nx_gap={x_gap_um:4.1f} um\n')
    plt.xlabel('X, um')
    plt.ylabel('Signal')
    plt.legend()
    plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"lin_calib_fits.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(x_fit_orig, y_fit_orig, lw=2, label="Fitted data")
    # plt.plot(x_fit_orig, signal_middle, lw=1, zorder=0, label="signal_middle")
    plt.plot(x_fit_orig, lin_fit, 'r', lw=1, label=f'\nFit\nd={d_um:3.1f} um\nx_gap={x_gap_um:4.1f} um\n')
    plt.ylim(0, np.max(y_fit_orig)*0.02)
    plt.xlim(0, 5000)
    plt.xlabel('X, um')
    plt.ylabel('Signal')
    plt.legend()
    plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"lin_calib_fits_zoom.png")
    plt.close()


    # TODO: добавить флаг выполнения корреткировки линейки аналогично камере.
    koef_scale_lin = d_um / d_um_ref

    x_distance_um_lin_cor = koef_scale_lin * (x_distance_um_lin - x_gap_um)
    x_distance_m_lin_cor = x_distance_um_lin_cor * 1e-6
    width_pix_x_m_cor = koef_scale_lin * width_pix_x_m
    width_pix_y_m_cor = koef_scale_lin * width_pix_y_m
    print(f"Скорректированный размер пикселя линейки: {width_pix_x_m_cor*1e6} x {width_pix_y_m_cor*1e6} мкм")

    # pix_max_ampl_cor = pix_max_ampl + koef_scale_lin * x_gap_um / (width_pix_x_m_cor*1e6)
    pix_max_ampl_cor = pix_max_ampl + x_gap_um / (width_pix_x_m*1e6)

    shift_lin_m_cor = koef_scale_lin * shift_lin_m
    coord_left_side_m = min(x_distance_m_lin_cor) + shift_lin_m_cor 
    coord_right_side_m = max(x_distance_m_lin_cor) + shift_lin_m_cor 
    phi_left_rad = np.arctan(coord_left_side_m / F_lens_m)
    phi_right_rad = np.arctan(coord_right_side_m / F_lens_m)
    print(f"Координата левого края линейки в рабочем положении: {coord_left_side_m*1000:.3f} мм; угол: {np.degrees(phi_left_rad):.6f} град")
    print(f"Координата правого края линейки в рабочем положении: {coord_right_side_m*1000:.3f} мм; угол: {np.degrees(phi_right_rad):.6f} град")
    pix_max_ampl, pix_max_ampl_cor



    x_fit_orig = x_distance_um_lin_cor
    # y_fit_orig = signal_norm_cam
    y_fit_orig = signal_norm_lin
    x_fit = x_fit_orig
    y_fit = y_fit_orig


    p0 = [
        max(y_fit),
        diam_um_pinhole_lin,
        0.0
    ]
    popt, pcov = curve_fit(model_lin, x_fit, y_fit, p0=p0)
    A_lin, d_um, x_gap_um = popt
    print(f"A = {A_lin:.4f}")
    print(f"d_um = {d_um:.4f}")
    print(f"x_gap_um = {x_gap_um:.4f}")
    lin_fit = model_lin(x_fit_orig, *popt)


    # Построение
    plt.figure(figsize=(10, 5))
    plt.plot(x_fit_orig, y_fit_orig, lw=2, label="Fitted data")
    # plt.plot(x_fit_orig, signal_middle, lw=1, zorder=0, label="signal_middle")
    plt.plot(x_fit_orig, lin_fit, 'r', lw=1, label=f'\nFit\nd={d_um:3.1f} um\nx_gap={x_gap_um:4.1f} um\n')
    plt.xlabel('X, um')
    plt.ylabel('Signal')
    plt.legend()
    plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"lin_calib_fits_correction.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(x_fit_orig, y_fit_orig, lw=2, label="Fitted data")
    # plt.plot(x_fit_orig, signal_middle, lw=1, zorder=0, label="signal_middle")
    plt.plot(x_fit_orig, lin_fit, 'r', lw=1, label=f'\nFit\nd={d_um:3.1f} um\nx_gap={x_gap_um:4.1f} um\n')
    plt.ylim(0, np.max(y_fit_orig)*0.02)
    plt.xlim(0, 5000)
    plt.xlabel('X, um')
    plt.ylabel('Signal')
    plt.legend()
    plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"lin_calib_fits_zoom_correction.png")
    plt.close()

    # Построение
    plt.figure(figsize=(10, 5))
    plt.plot(x_fit_orig, y_fit_orig, lw=2, label="Fitted data")
    # plt.plot(x_fit_orig, signal_middle, lw=1, zorder=0, label="signal_middle")
    plt.plot(x_fit_orig, lin_fit, 'r', lw=1, label=f'\nFit\nd={d_um:3.1f} um\nx_gap={x_gap_um:4.1f} um\n')
    plt.xlim(-6000, 6000)
    plt.ylim(0, np.max(y_fit_orig)*0.02)
    plt.xlabel('X, um')
    plt.ylabel('Signal')
    plt.legend()
    plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"lin_calib_fits_zoom_2_correction.png")
    plt.close()



    # ----- калибровка камеры и линейки -----
    mask = (x_distance_um_cam_cor >= 1500) & (x_distance_um_cam_cor <= 2500)
    indices = np.where(mask)[0]
    idx_1_diff_max_cam = indices[np.argmax(cam_fit_2[mask])]
    x_um_1_diff_max_cam = x_distance_um_cam_cor[idx_1_diff_max_cam]
    signal_1_diff_max_cam = cam_fit_2[idx_1_diff_max_cam]

    mask = (x_distance_um_lin_cor >= 1500) & (x_distance_um_lin_cor <= 2500)
    indices = np.where(mask)[0]
    idx_1_diff_max_lin = indices[np.argmax(lin_fit[mask])]
    x_um_1_diff_max_lin = x_distance_um_lin_cor[idx_1_diff_max_lin]
    signal_1_diff_max_lin = lin_fit[idx_1_diff_max_lin]

    # coef_lin_to_cam = signal_1_diff_max_cam / signal_1_diff_max_lin
    coef_lin_to_cam = A_cam2 / A_lin
    print(f"Коэффициент приведения интенсивности линейки к камере\nс делением на площадь (новый):  {coef_lin_to_cam}")


    plt.figure(figsize=(10, 5))
    plt.plot(x_distance_um_cam_cor, signal_norm_cam, 'blue', zorder=0, label="signal_norm_cam")
    plt.plot(x_distance_um_cam_cor, cam_fit_2, linestyle="--", zorder=2, color='orange', label="fit cam")
    plt.plot(x_distance_um_lin_cor, signal_norm_lin, linewidth=1, linestyle="-", marker="o", color='r', markersize=1, zorder=1, label="signal_norm_lin")
    plt.plot(x_distance_um_lin_cor, lin_fit, linestyle="--", color='green', zorder=3, label="fit lin")
    plt.xlim(0, 5000)
    # plt.ylim(-10, 2e9)
    plt.xlabel("X, um")
    plt.ylabel("Амплитуда")
    plt.title("Зависимость амплитуды от номера пикселя")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"cam_lin_signals_fits.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(x_distance_um_cam_cor, signal_norm_cam, 'blue', zorder=0, label="signal_norm_cam")
    plt.plot(x_distance_um_cam_cor, cam_fit_2, linestyle="--", zorder=2, color='orange', label="fit cam")
    plt.vlines([x_um_1_diff_max_cam], min(cam_fit_2), signal_1_diff_max_cam, linestyle="--", zorder=2, color='orange')
    plt.plot(x_distance_um_lin_cor, signal_norm_lin, linewidth=1, linestyle="-", marker="o", color='red', markersize=1, zorder=1, label="signal_norm_lin")
    plt.plot(x_distance_um_lin_cor, lin_fit, linestyle="--", color='green', zorder=3, label="fit lin")
    plt.vlines([x_um_1_diff_max_lin], min(cam_fit_2), signal_1_diff_max_lin, linestyle="--", zorder=3, color='green')
    plt.xlim(0, 5000)
    plt.ylim(-10, 2e9)
    plt.xlabel("X, um")
    plt.ylabel("Амплитуда")
    plt.title("Зависимость амплитуды от номера пикселя")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"cam_lin_signals_fits_zoom.png")
    plt.close()


    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Левая ось Y (камера)
    ax1.plot(
        x_distance_um_cam_cor,
        signal_middle_norm_cam,
        label="signal_middle_norm_cam",
        color="b"
    )
    # ax1.axvline(
    #     x_um_1_diff_max_cam,
    #     linestyle="--",
    #     alpha=0.7,
    #     color="b"
    # )

    ax1.set_xlabel("X, um")
    ax1.set_ylabel("Camera amplitude", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_xlim(0, 5000)
    ax1.grid(True)
    # ax1.set_ylim(0, 8e10)
    # ax1.set_ylim(0, 1.5e9)
    ax1.set_ylim(0, np.max(signal_middle_norm_cam)*1.05)

    # Правая ось Y (линейка)
    ax2 = ax1.twinx()

    ax2.plot(
        x_distance_um_lin_cor,
        signal_orig_norm_lin,
        linewidth=1,
        linestyle="-",
        marker="o",
        markersize=1,
        label="signal_orig_norm_lin",
        color="r"
    )
    # ax2.axvline(
    #     x_um_1_diff_max_lin,
    #     linestyle=":",
    #     alpha=0.7,
    #     color="r"
    # )

    ax2.set_ylabel("Line sensor amplitude", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    # ax2.set_ylim(0, 8e10)
    # ax2.set_ylim(0, 1.5e9)
    ax2.set_ylim(0, np.max(signal_orig_norm_lin)*1.05)

    # Общая легенда
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Зависимость амплитуды от расстояния")
    plt.tight_layout()
    plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"cam_lin_signals_fits_diff_scale.png")
    plt.close()



    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Левая ось Y (камера)
    ax1.plot(
        x_distance_um_cam_cor,
        signal_norm_cam,
        label="signal_norm_cam",
        color="b"
    )
    # ax1.axvline(
    #     x_um_1_diff_max_cam,
    #     linestyle="--",
    #     alpha=0.7,
    #     color="b"
    # )

    ax1.set_xlabel("X, um")
    ax1.set_ylabel("Camera amplitude", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_xlim(0, 5000)
    ax1.grid(True)
    # ax1.set_ylim(0, 1.5e9)
    ax1.set_ylim(0, np.max(signal_norm_cam)*0.03)

    # Правая ось Y (линейка)
    ax2 = ax1.twinx()

    ax2.plot(
        x_distance_um_lin_cor,
        signal_norm_lin,
        linewidth=1,
        linestyle="-",
        marker="o",
        markersize=1,
        label="signal_norm_lin",
        color="r"
    )
    # ax2.axvline(
    #     x_um_1_diff_max_lin,
    #     linestyle=":",
    #     alpha=0.7,
    #     color="r"
    # )

    ax2.set_ylabel("Line sensor amplitude", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    # ax2.set_ylim(0, 1.5e9)
    ax2.set_ylim(0, np.max(signal_orig_norm_lin)*0.03)

    # Общая легенда
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Зависимость амплитуды от расстояния, 3% от максимума")
    plt.tight_layout()
    plt.savefig(PATH_SAVE_PREPROCESSING_DIR/"cam_lin_signals_fits_diff_scale_zoom.png")
    plt.close()


print("\n\nПараметры для ДАРЛ")
print(
f"Передняя_матрица:\n"
f"включена: да\n"
f"диагональ_передней_матрицы_(мм): {diag_matrix_img_mm}\n"
f"количество_пикселей_длина_(шт): {cfg.W}\n"
f"количество_пикселей_ширина_(шт): {cfg.H}\n"
f"отступ_от_оптической_оси_(мкм): {cam_shift_um_cor[0]}\n"
f"отступ_от_оптической_оси_Y(мкм): {cam_shift_um_cor[1]}\n"
)

if is_add_lin:
    print(
    f"Лог_линейка:\n"
    f"Включена: да\n"
    f"Начальный_угол_(град): {np.degrees(phi_left_rad)}\n"
    f"Конечный_угол_(град): {np.degrees(phi_right_rad)}\n"
    f"Лог_радиус_(%): 7.5\n"
    f"Ширина_пикселя_(мкм): {width_pix_y_m_cor*1e6}\n"
    )

print("\nПараметры для восстановления")
print(
f"    x_shift_m = {cam_shift_um_cor[0]*1e-6},\n"
f"    y_shift_m = {cam_shift_um_cor[1]*1e-6},\n"
f"    cam_pixel_width_m = {pix_img_width_m_cor},\n"
)

if is_add_lin:
    print(
        f"    width_pix_x_m = {width_pix_x_m_cor},\n"
        f"    width_pix_y_m = {width_pix_y_m_cor},\n"
        f"    coef_lin_to_cam = {coef_lin_to_cam},\n"
        f"    shift_lin_m = {shift_lin_m_cor},\n"
        f"    pix_max_ampl = {pix_max_ampl_cor},\n"
    )

print(
f'"x_shift_m": {cam_shift_um_cor[0]*1e-6},\n'
f'"y_shift_m": {cam_shift_um_cor[1]*1e-6},\n'
f'"cam_pixel_width_m": {pix_img_width_m_cor},\n'
)

if is_add_lin:
    print(
        f'"width_pix_x_m": {width_pix_x_m_cor},\n'
        f'"width_pix_y_m": {width_pix_y_m_cor},\n'
        f'"coef_lin_to_cam": {coef_lin_to_cam},\n'
        f'"shift_lin_m": {shift_lin_m_cor},\n'
        f'"pix_max_ampl": {pix_max_ampl_cor},\n'
    )

line_result_values = None
if is_add_lin:
    line_result_values = {
        "start_angle_deg": np.degrees(phi_left_rad),
        "end_angle_deg": np.degrees(phi_right_rad),
        "pixel_width_m": width_pix_x_m_cor,
        "pixel_height_m": width_pix_y_m_cor,
        "to_camera_coefficient": coef_lin_to_cam,
        "shift_m": shift_lin_m_cor,
        "peak_pixel": pix_max_ampl_cor,
        "distance_um": x_distance_um_lin_cor,
        "signal": signal_norm_lin,
        "fit": lin_fit,
    }
save_legacy_result(
    PATH_SAVE_PREPROCESSING_DIR,
    diagonal_mm=diag_matrix_img_mm,
    camera_shift_um=cam_shift_um_cor,
    camera_pixel_width_m=pix_img_width_m_cor,
    camera_distance_um=x_distance_um_cam_cor,
    camera_signal=signal_norm_cam,
    camera_fit=cam_fit_2,
    line_values=line_result_values,
)
