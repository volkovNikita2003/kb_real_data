import inspect
from dataclasses import dataclass, field, fields
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from matplotlib.patches import Circle
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.optimize import nnls
from scipy.special import j1

import ast
import types
from typing import Union, get_args, get_origin, get_type_hints


REGULARIZATION_TYPE = 1
REGULARIZATION_ALPHA = "best"
W_CRITICAL = 1e-3


def jinc(x):
    # Избегаем деления на ноль в точке 0
    x = np.asarray(x)
    jinc_vals = np.empty_like(x, dtype=float)
    nonzero = (x != 0)
    jinc_vals[nonzero] = 2.0 * j1(x[nonzero]) / x[nonzero]
    jinc_vals[~nonzero] = 1.0
    return jinc_vals


@dataclass
class ExperimentConfig:
    # основные параметры
    dir_case: Path = Path("case_dir")
    dir_save_rel: Path = Path("save_dir")
    dir_save_preprocessing_rel: Path = Path("save_dir") / "preprocessing"

    # установка и эксперимент
    labm_um: float = 0.633
    F_lens_um: float = 0.4e6
    # тип конфигурации детекторов
    # 0 -- старая конфигурация (до 17.07.26)
    #      камера и линейка справа от пучка
    # 1 -- новая конфигурация (с 17.07.26)
    #      камера и линейка слева от пучка
    detector_configuration_type: int = 0


    # калибровка
    calib_d_pinhole_um: float = 200
    calib_cam_gaussian_sigma: float = 20
    calib_cam_corrected: bool = False
    calib_lin_position_pinhole_m: float = None
    calib_lin_position_signal_m: float = None
    calib_lin_gaussian_sigma: float = 3

    # камера
    dir_signal_rel: str = None
    dir_back_rel: str | None = None
    exposure_time_arr: list[int] = field(default_factory=list)

    W: int = 2592
    H: int = 1944
    cam_pixel_width_m: float = 2.2e-6 / 0.58 / 2
    x_shift_m: float = 0.0
    y_shift_m: float = 0.0

    cam_hdr_files_mode: str = "bmp"
    # l2h / l2h_longest: какую допустимую выдержку выбирать
    cam_hdr_mode: str = "l2h_longest"
    # after_hdr: HDR(signal) - HDR(background), старый режим
    # per_exposure: HDR(signal[exp] - background[exp])
    cam_hdr_diff_mode: str = "after_hdr"
    cam_hdr_back_level: float = 12
    cam_hdr_top_thr: float = 255-15-12
    cam_hdr_low_thr: float = 5
    cam_hdr_filtered: bool = False
    cam_hdr_gauss_sigma: float = 20
    cam_hdr_exposure_coefs: dict = None


    # линейка
    dir_signal_lin_rel: str = None
    dir_back_lin_rel: str | None = None
    filename_lin_template: str = None
    filename_lin_back_template: str = None
    exposure_time_us_lin_arr: list[int] = field(default_factory=list)

    coef_lin_to_cam: float = None
    shift_lin_m: float = None
    pix_max_ampl: float = None
    num_pix_lin: int = 3643
    width_pix_x_m: float = 8e-6
    width_pix_y_m: float = 200e-6
    lin_time_add: float = 2  # us
    lin_signal_mode: int = 2


    # darl
    darl_config_name: str = None
    path_signal_darl_rel: str = None

    matrix_name: str = "matrix.npz"
    bins_name: str = "bins_matrix.txt"
    classes_name: str = "particle_classes.txt"
    bins_lin_name: str = "bins_lin.txt"

    # тип сигнала
    # если "signal", то как было всегда
    # если "intensity", то делим на площадь бина
    signal_type: str = "signal"


    signal_vector_file_rel: str | None = None


    # восстановление
    use_w_critical: bool = False
    use_chahine: bool = True
    use_conc_corr: bool = True
    forward_modeling_enabled: bool = False

    cut_classes: int = None
    cut_classes_top: int = None


    @property
    def dir_signal(self):
        if self.dir_signal_rel is None:
            return None
        return self.dir_case / self.dir_signal_rel

    @property
    def signal_vector_file(self):
        if self.signal_vector_file_rel is None:
            return None
        return self.dir_case / self.signal_vector_file_rel

    @property
    def dir_back(self):
        if self.dir_back_rel is None:
            return None
        return self.dir_case / self.dir_back_rel
    
    @property
    def dir_signal_lin(self):
        if self.dir_signal_lin_rel is None:
            return None
        return self.dir_case / self.dir_signal_lin_rel

    @property
    def dir_back_lin(self):
        if self.dir_back_lin_rel is None:
            return None
        return self.dir_case / self.dir_back_lin_rel
    
    @property
    def path_signal_darl(self):
        if self.path_signal_darl_rel is None:
            return None
        return self.dir_case / self.path_signal_darl_rel

    @property
    def dir_save(self):
        if self.dir_save_rel is None:
            return None
        return self.dir_case / self.dir_save_rel
    
    @property
    def dir_save_preprocessing(self):
        if self.dir_save_preprocessing_rel is None:
            return None
        return self.dir_case / self.dir_save_preprocessing_rel

    @property
    def file_matrix(self):
        if self.matrix_name is None:
            return None
        return self.dir_case / self.matrix_name

    @property
    def file_bins(self):
        if self.bins_name is None:
            return None
        return self.dir_case / self.bins_name
    
    @property
    def file_bins_lin(self):
        if self.bins_lin_name is None:
            return None
        return self.dir_case / self.bins_lin_name

    @property
    def file_classes(self):
        if self.classes_name is None:
            return None
        return self.dir_case / self.classes_name

    @property
    def x_shift_pix(self):
        return self.x_shift_m / self.cam_pixel_width_m

    @property
    def y_shift_pix(self):
        return self.y_shift_m / self.cam_pixel_width_m
    
    def iter_properties(self):
        for name, descriptor in inspect.getmembers(type(self)):
            if isinstance(descriptor, property):
                yield name, getattr(self, name)

    def save_params(self, filename: str = "params.txt"):
        self.dir_save.mkdir(parents=True, exist_ok=True)
        file_path = self.dir_save / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("[FIELDS]\n")
            for field_info in fields(self):
                value = getattr(self, field_info.name)
                f.write(f"{field_info.name} = {value}\n")

            f.write("\n[PROPERTIES]\n")
            for name, value in self.iter_properties():
                f.write(f"{name} = {value}\n")
    
    @staticmethod
    def _parse_param_value(value_str: str, field_type):
        """
        Преобразует строковое значение из params.txt
        к типу соответствующего поля dataclass.
        """
        value_str = value_str.strip()

        if value_str == "None":
            return None

        # Обрабатываем Optional[T], T | None и другие Union-типы
        origin = get_origin(field_type)
        if origin in (Union, types.UnionType):
            possible_types = [
                arg for arg in get_args(field_type)
                if arg is not type(None)
            ]

            if len(possible_types) == 1:
                field_type = possible_types[0]
                origin = get_origin(field_type)

        if field_type is str:
            return value_str

        if field_type is Path:
            return Path(value_str)

        if field_type is bool:
            normalized = value_str.lower()

            if normalized == "true":
                return True

            if normalized == "false":
                return False

            raise ValueError(
                f"Некорректное логическое значение: {value_str!r}"
            )

        if field_type is int:
            return int(value_str)

        if field_type is float:
            return float(value_str)

        # list[int], list[float], dict и другие контейнеры
        if origin in (list, dict, tuple, set):
            value = ast.literal_eval(value_str)

            if not isinstance(value, origin):
                raise TypeError(
                    f"Ожидался тип {origin.__name__}, "
                    f"получен {type(value).__name__}"
                )

            return value

        # Резервный вариант для литералов Python:
        # числа, списки, словари и т. п.
        try:
            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            return value_str

    def load_params(self, filename: str | Path = "params.txt"):
        """
        Обновляет поля конфигурации значениями из секции [FIELDS].

        Поля, отсутствующие в файле, сохраняют свои текущие значения.
        Секция [PROPERTIES] игнорируется, поскольку свойства вычисляются
        автоматически.

        Parameters
        ----------
        filename:
            Можно передать полный или относительный путь.

        Returns
        -------
        list[str]
            Список обновлённых полей.
        """
        file_path = Path(filename)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Файл конфигурации не найден: {file_path.resolve()}"
            )

        dataclass_fields = {
            field_info.name: field_info
            for field_info in fields(self)
        }

        # get_type_hints корректно разрешает аннотации вроде Path | None
        type_hints = get_type_hints(type(self))

        current_section = None
        updated_fields = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1].strip().upper()
                    continue

                if current_section != "FIELDS":
                    continue

                if "=" not in line:
                    raise ValueError(
                        f"Некорректная строка {line_number} "
                        f"в файле {file_path}: {line!r}"
                    )

                field_name, value_str = line.split("=", maxsplit=1)
                field_name = field_name.strip()
                value_str = value_str.strip()

                # Неизвестные поля пропускаются. Это позволяет читать
                # файлы от более новых версий ExperimentConfig.
                if field_name not in dataclass_fields:
                    continue

                field_type = type_hints.get(
                    field_name,
                    dataclass_fields[field_name].type,
                )

                try:
                    value = self._parse_param_value(
                        value_str,
                        field_type,
                    )
                except (ValueError, TypeError, SyntaxError) as exc:
                    raise ValueError(
                        f"Не удалось прочитать поле {field_name!r} "
                        f"в строке {line_number}: {value_str!r}"
                    ) from exc

                setattr(self, field_name, value)
                updated_fields.append(field_name)

        return updated_fields
    
def read_coefficients(filename: str | Path) -> dict[int, float]:
    """
    Читает файл и создаёт словарь:

        ключ = значение первого столбца;
        значение = второй столбец для максимального ключа /
                второй столбец для текущего ключа.
    """
    data: dict[int, float] = {}

    with open(filename, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            # Пропускаем пустые строки
            if not line:
                continue

            columns = line.split()

            if len(columns) < 2:
                raise ValueError(
                    f"В строке {line_number} недостаточно столбцов: {line!r}"
                )

            key = int(columns[0])
            second_value = float(columns[1])

            if second_value == 0:
                raise ValueError(
                    f"Во втором столбце строки {line_number} находится ноль"
                )

            data[key] = second_value

    if not data:
        raise ValueError("Файл не содержит данных")

    max_key = max(data)
    reference_value = data[max_key]

    return {
        key: reference_value / second_value
        for key, second_value in data.items()
    }




# ----- прероцессинг -----
def get_bmp_numeric_names(directory: str | Path) -> list[int]:
    """
    Находит в директории BMP-файлы, имена которых состоят только из цифр.

    Например:
        10.bmp  -> 10
        025.bmp -> 25
        img.bmp -> пропускается

    Возвращает отсортированный по возрастанию список чисел.
    """
    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Директория не существует: {directory.resolve()}"
        )

    numbers = [
        int(file_path.stem)
        for file_path in directory.iterdir()
        if (
            file_path.is_file()
            and file_path.suffix.lower() == ".bmp"
            and file_path.stem.isdigit()
        )
    ]

    return sorted(numbers)

def calc_pinhole_diam_m_lin(first_dif_min_m, lambd_m = 633e-9, F_lens_m = 0.4):
    theta_first_dif_min_rad = np.arctan(first_dif_min_m/F_lens_m)
    sin_theta = np.sin(theta_first_dif_min_rad)
    d_pinhole_m = 1.22 * lambd_m / sin_theta
    return d_pinhole_m


def find_circle(
        data,
        sigma=10,
        plotted=False,
        y_lines = range(400, 1601, 100),
        x_min_find_min = 1500,
        x_max_find_min = None,
        x_lines = range(2000, 2592, 100),
        y_bound = int(1944//2),
        radii = [700, 900, 1250, 1350],
        save_dir=None
):
    points_dif_min = []

    for y_line in y_lines:
        signal_middle = data[y_line, :]
        filtered_data = gaussian_filter(signal_middle, sigma=sigma)

        pix_min_val = filtered_data[x_min_find_min:x_max_find_min].min()
        idxs = np.where(filtered_data[x_min_find_min:x_max_find_min] == pix_min_val)[0]
        median_idx = idxs[len(idxs) // 2]
        pix_min = x_min_find_min+median_idx
        points_dif_min.append((pix_min, y_line))
    
    for x_line in x_lines:
        signal_middle = data[:, x_line]
        filtered_data = gaussian_filter(signal_middle, sigma=sigma)


        pix_min_val = filtered_data[:y_bound].min()
        idxs = np.where(filtered_data[:y_bound] == pix_min_val)[0]
        median_idx = idxs[len(idxs) // 2]
        pix_min = median_idx
        points_dif_min.append((x_line, pix_min))

        pix_min_val_2 = filtered_data[y_bound:].min()
        idxs = np.where(filtered_data[y_bound:] == pix_min_val_2)[0]
        median_idx = idxs[len(idxs) // 2]
        pix_min_2 = y_bound + median_idx
        points_dif_min.append((x_line, pix_min_2))
    
    points_dif_min = np.array(points_dif_min)
    xc, yc, R = fit_circle(points_dif_min)
    center = np.array([xc, yc])
    
    if plotted:
        plot_fitted_points(data, y_lines, x_lines, points_dif_min, center, R, radii, True, save_dir)
        plot_fitted_points(data, y_lines, x_lines, points_dif_min, center, R, radii, False, save_dir)
        

    return center, R


def fit_circle(points):
    x = points[:,0]
    y = points[:,1]
    xc0 = np.mean(x)
    yc0 = np.mean(y)
    r0 = np.mean(np.sqrt((x-xc0)**2 + (y-yc0)**2))
    def residuals(p):
        xc, yc, r = p
        return np.sqrt((x-xc)**2 + (y-yc)**2) - r
    res = least_squares(
        residuals,
        [xc0, yc0, r0]
    )
    return res.x


def plot_fitted_points(
        data,
        y_lines,
        x_lines,
        points_dif_min,
        center,
        raduis,
        radii = [700, 900, 1250, 1350],
        add_points=True,
        save_dir=None
):
    xc, yc = center
    R = raduis
    height, width = data.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(data, cmap='gray', origin='upper')

    if add_points:
        ax.hlines(y_lines, 0, width - 1, alpha=0.1)
        ax.vlines(x_lines, 0, height - 1, alpha=0.1)
        ax.scatter(points_dif_min[:, 0], points_dif_min[:, 1], alpha=0.2)


    circle = Circle(
        (xc, yc),
        R,
        fill=False,
        color='red',
        linewidth=1,
        alpha=0.5
    )
    ax.add_patch(circle)

    # center_laser = (2650, 972)
    for r in radii:
        circle = Circle(
            (xc, yc),
            r,
            fill=False,
            color='red',
            linewidth=1,
            alpha=0.3
        )
        ax.add_patch(circle)
    ax.plot(xc, yc, 'r+', markersize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Intensity')

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_ylabel('Pixel Y')
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    ax.set_title(f"Изображение колец. Центр ({xc:.1f}, {yc:.1f}) pix")
    fig.tight_layout()
    if save_dir is not None:
        save_dir = Path(save_dir)
        filename = "cam_fitted_rings"
        filename += "_points" if add_points else ""
        filename += ".png"
        plt.savefig(save_dir/filename)
    else:
        plt.show()
    plt.close()

def calc_matrix_shift_m(center):
    pix_img_width_m = 2.2e-6 / 0.58 / 2  # м
    matrix_orig_pix = np.array([2592, 972])
    matrix_shift_pix = center - matrix_orig_pix
    matrix_shift_m = matrix_shift_pix * pix_img_width_m
    return matrix_shift_m

def calc_matrix_shift_m_new_position(center):
    pix_img_width_m = 2.2e-6 / 0.58 / 2  # м
    matrix_orig_pix = np.array([0, 972])
    matrix_shift_pix = matrix_orig_pix - center
    matrix_shift_m = matrix_shift_pix * pix_img_width_m
    return matrix_shift_m

def calc_pinhole_diam_m(radius, lambd_m = 633e-9, F_lens_m = 0.4):
    pix_img_width_m = 2.2e-6 / 0.58 / 2  # м
    first_dif_min_m = radius * pix_img_width_m
    theta_first_dif_min_rad = np.arctan(first_dif_min_m/F_lens_m)
    sin_theta = np.sin(theta_first_dif_min_rad)
    d_pinhole_m = 1.22 * lambd_m / sin_theta
    return d_pinhole_m


def calculate_fit_metrics(
        y_true,
        y_pred,
        n_model_params: int,
        sigma=None,
):
    """
    Вычисляет численные показатели качества аппроксимации.

    Parameters
    ----------
    y_true : array-like
        Исходные данные.
    y_pred : array-like
        Значения модели.
    n_model_params : int
        Число подбираемых параметров модели.
    sigma : array-like or None
        Стандартное отклонение измерений.
        Если задано, вычисляются chi2 и reduced chi2.

    Returns
    -------
    dict
        Словарь с метриками качества.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Размеры y_true и y_pred различаются: "
            f"{y_true.shape} != {y_pred.shape}"
        )

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if sigma is not None:
        sigma = np.broadcast_to(
            np.asarray(sigma, dtype=float),
            y_true.shape,
        )
        mask &= np.isfinite(sigma) & (sigma > 0)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if sigma is not None:
        sigma = sigma[mask]

    n = y_true.size
    p = n_model_params

    if n <= p:
        raise ValueError(
            f"Недостаточно точек: n={n}, число параметров p={p}"
        )

    residuals = y_true - y_pred

    # Сумма квадратов остатков
    sse = np.sum(residuals**2)

    # Средние ошибки
    mse = sse / n
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    max_abs_error = np.max(np.abs(residuals))

    # Нормированные ошибки
    peak = np.max(np.abs(y_true))
    value_range = np.ptp(y_true)

    nrmse_peak = rmse / peak if peak > 0 else np.nan
    nrmse_range = rmse / value_range if value_range > 0 else np.nan

    # Коэффициент детерминации
    ss_total = np.sum((y_true - np.mean(y_true))**2)

    if ss_total > 0:
        r2 = 1.0 - sse / ss_total
    else:
        r2 = np.nan

    if n > p + 1 and np.isfinite(r2):
        r2_adjusted = 1.0 - (
            (1.0 - r2) * (n - 1) / (n - p - 1)
        )
    else:
        r2_adjusted = np.nan

    # При вычислении информационных критериев учитываем
    # параметры модели и оценку дисперсии остатков.
    k = p + 1

    sse_safe = max(sse, np.finfo(float).tiny)
    log_likelihood_part = n * np.log(sse_safe / n)

    aic = log_likelihood_part + 2 * k
    bic = log_likelihood_part + k * np.log(n)

    if n > k + 1:
        aicc = aic + 2 * k * (k + 1) / (n - k - 1)
    else:
        aicc = np.nan

    # Среднее смещение модели
    mean_residual = np.mean(residuals)

    # Индикатор последовательной корреляции остатков.
    # Значение около 2 соответствует отсутствию выраженной
    # корреляции соседних остатков.
    if sse > 0 and n > 1:
        durbin_watson = (
            np.sum(np.diff(residuals)**2) / sse
        )
    else:
        durbin_watson = np.nan

    result = {
        "SSE": sse,
        "RMSE": rmse,
        "MAE": mae,
        "MaxAbsError": max_abs_error,
        "NRMSE_peak": nrmse_peak,
        "NRMSE_range": nrmse_range,
        "R2": r2,
        "R2_adjusted": r2_adjusted,
        "AIC": aic,
        "AICc": aicc,
        "BIC": bic,
        "MeanResidual": mean_residual,
        "DurbinWatson": durbin_watson,
    }

    if sigma is not None:
        chi2 = np.sum((residuals / sigma)**2)
        degrees_of_freedom = n - p

        result["Chi2"] = chi2
        result["ReducedChi2"] = (
            chi2 / degrees_of_freedom
        )

    return result


def print_parameter_errors(names, popt, pcov):
    errors = np.sqrt(np.diag(pcov))

    for name, value, error in zip(names, popt, errors):
        relative_error = (
            100 * error / abs(value)
            if value != 0
            else np.nan
        )

        print(
            f"{name}: "
            f"{value:.6g} ± {error:.3g}, "
            f"relative error = {relative_error:.2f}%"
        )



# hdr изображение
# ----- сборка изображения от большой выдержки к маленькой -----
def get_signal_cam_good_pix(
        filename_short,
        t_short,
        t_long,
        hdr_ready,
        mask_ready,
        files_mode = "bmp",
        top_thr = 254,
        low_thr  = 0
):
    # коэффициент пересчёта короткой экспозиции к длинной
    k_front = t_long / t_short   

    # читаем
    if files_mode == "bmp":
        img = Image.open(filename_short)
        short = np.array(img, dtype=np.float64)
    else:
        raise Exception(f"Wrong files_mode = {files_mode}")

    short_scaled = short * k_front

    # mask_front_short_good_pix = (short < threshold)
    mask_front_short_good_pix = (short >= low_thr) & (short <= top_thr)
    mask_not_ready = ~mask_ready
    mask_add_pix = mask_not_ready & mask_front_short_good_pix
    mask_ready_new = mask_ready | mask_add_pix
    hdr_ready = np.where(mask_add_pix, short_scaled, hdr_ready)

    return hdr_ready, mask_ready_new

def get_signal_cam_comb_many_exposure_hight2low(
        cfg: ExperimentConfig,
        files_mode = "bmp",
        top_thr: float = 254,
        low_thr: float = 0,
):
    raise NotImplementedError("Используйте аналогичную функцию " \
    "get_signal_cam_comb_many_exposure_low2hight с combine_exposures_longest_valid")
    W, H = cfg.W, cfg.H

    dx = cfg.cam_pixel_width_m
    dy = cfg.cam_pixel_width_m

    diag = np.sqrt((dx * W)**2 + (dy * H)**2)
    print(f'ДИАГОНАЛЬ МАТРИЦЫ: {diag*1e3} мм')

    print(f"matrix shift_x: {cfg.x_shift_pix*cfg.cam_pixel_width_m*1e6:.3f} um")
    print(f"matrix shift_y: {cfg.y_shift_pix*cfg.cam_pixel_width_m*1e6:.3f} um")

    exposure_time_arr = np.array(cfg.exposure_time_arr)
    exposure_time_arr = np.sort(exposure_time_arr)[::-1]
    t_long = exposure_time_arr[0]

    hdr_ready = np.zeros((H, W), dtype=np.float64)
    mask_ready = np.full_like(hdr_ready, False, dtype=np.bool)
    for exposure_time in exposure_time_arr:
        filename_short = cfg.dir_signal/f"{exposure_time}.bmp"
        hdr_ready, mask_ready = get_signal_cam_good_pix(
            filename_short,
            exposure_time,
            t_long,
            hdr_ready,
            mask_ready,
            files_mode=files_mode,
            top_thr=top_thr,
            low_thr=low_thr 
        )
        if np.sum(mask_ready) == mask_ready.size:
            print(f"Сигнал, готово, {exposure_time}")
            break
    print(f"Количество неопределенных пикселей в сигнале: {np.sum(~mask_ready)}/{mask_ready.size}")

    hdr_back_ready = np.zeros((H, W), dtype=np.float64)
    if cfg.dir_back is not None:
        mask_back_ready = np.full_like(hdr_ready, False, dtype=np.bool)
        for exposure_time in exposure_time_arr:
            filename_short = cfg.dir_back/f"{exposure_time}.bmp"
            hdr_back_ready, mask_back_ready = get_signal_cam_good_pix(
                filename_short,
                exposure_time,
                t_long,
                hdr_back_ready,
                mask_back_ready,
                files_mode=files_mode,
                top_thr=top_thr,
                low_thr=low_thr 
            )
            if np.sum(mask_back_ready) == mask_back_ready.size:
                print(f"Фон, готово, {exposure_time}")
                break
        print(f"Количество неопределенных пикселей в фоне: {np.sum(~mask_back_ready)}/{mask_back_ready.size}")
        
    signal = np.maximum(hdr_ready - hdr_back_ready, 0)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    cmap_style = 'gray'
    # # в зависимости от выдержки
    # img_front = img_front1
    # hdr = img_front

    ax1 = axes[0]
    im1 = ax1.imshow(hdr_ready, cmap=cmap_style)
    ax1.set_title('Сигнал. HDR')
    fig.colorbar(im1, ax=ax1) # Привязываем colorbar к конкретным осям

    ax2 = axes[1]
    im2 = ax2.imshow(hdr_back_ready, cmap=cmap_style)
    ax2.set_title(f'Фон. HDR')
    fig.colorbar(im2, ax=ax2)

    ax3 = axes[2]
    im3 = ax3.imshow(signal, cmap=cmap_style)
    ax3.set_title(f'Сигнал-Фон. HDR')
    fig.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.show()

    return signal


def combine_exposures_valid(
        directory: Path,
        exposure_time_arr,
        image_shape,
        back_level: float = 0,
        top_thr: float = 254,
        low_thr: float = 5,
        files_mode: str = "bmp",
        selection_mode="longest",
        exposure_coefs=None,
        background_directory: Path | None = None,
):
    """
    Для каждого пикселя выбирает наиболее длинную выдержку,
    на которой сигнал находится в допустимом диапазоне
    [low_thr, threshold].

    Если задан background_directory, фон вычитается из
    сырого сигнала на каждой выдержке до сборки HDR.

    Результат приводится к максимальной выдержке.
    """
    print(f"\ncombine_exposures_valid(): {back_level=}")

    exposure_time_arr = np.array(exposure_time_arr)
    exposure_time_arr = np.sort(exposure_time_arr)

    if exposure_time_arr.size == 0:
        raise ValueError("Empty exposure_time_arr")
    
    if low_thr > top_thr: 
        raise ValueError( f"low_thr={low_thr} must not be greater than " f"top_thr={top_thr}" )

    H, W = image_shape
    t_reference = exposure_time_arr[-1]

    hdr = np.zeros((H, W), dtype=np.float64)

    # Показывает, что для пикселя нашлась хотя бы одна
    # допустимая выдержка
    mask_valid_any = np.zeros((H, W), dtype=bool)

    # Изначально считаем, что каждый пиксель на всех выдержках 
    # был ниже или выше порога. Затем уточняем эти маски. 
    mask_always_below = np.ones((H, W), dtype=bool) 
    mask_always_above = np.ones((H, W), dtype=bool)

    for exposure_time in exposure_time_arr:
        filename = directory / f"{exposure_time}.bmp"

        if files_mode == "bmp":
            img = np.asarray(
                Image.open(filename),
                dtype=np.float64
            )
        else:
            raise ValueError(f"Wrong files_mode = {files_mode}")

        if img.shape != (H, W):
            raise ValueError(
                f"Wrong image shape for {filename}: "
                f"{img.shape}, expected {(H, W)}"
            )
        
        mask_saturated = img > top_thr

        if background_directory is not None:
            background_filename = background_directory / f"{exposure_time}.bmp"
            if files_mode == "bmp":
                background = np.asarray(
                    Image.open(background_filename),
                    dtype=np.float64
                )
            else:
                raise ValueError(f"Wrong files_mode = {files_mode}")

            if background.shape != (H, W):
                raise ValueError(
                    f"Wrong image shape for {background_filename}: "
                    f"{background.shape}, expected {(H, W)}"
                )

            mask_saturated_background = background > top_thr
            mask_saturated |= mask_saturated_background
            img -= background
        else:
            # вычитание собственных шумов камеры
            img -= back_level

        # plt.figure(figsize=(10, 8))
        # plt.imshow(img, cmap='gray')
        # plt.colorbar(label='Intensity')
        # plt.xlabel('Pixel X')
        # plt.ylabel('Pixel Y')
        # plt.title(f"{exposure_time} us")
        # plt.tight_layout()
        # filename = Path("debug")/f"{exposure_time}.png"
        # plt.savefig(filename)
        # plt.close()
        # print(f"save {filename}")

        # filename_bmp = Path("debug")/f"{exposure_time}.bmp"
        # if not np.all((img >= 0) & (img <= 255)):
        #     raise ValueError(
        #         f"Image values are outside [0, 255]: "
        #         f"min={img.min()}, max={img.max()}, "
        #         f"exposure_time={exposure_time}"
        #     )
        # img_bmp = img.astype(np.uint8)
        # Image.fromarray(img_bmp, mode="L").save(filename_bmp)
        # print(f"save {filename_bmp}")
        

        mask_below = img < low_thr
        mask_above = mask_saturated
        # Оставляем True только для пикселей, которые были 
        # ниже/выше порога на каждой обработанной выдержке. 
        mask_always_below &= mask_below
        mask_always_above &= mask_above
        
        mask_valid = (
            ~mask_saturated
            & (img >= low_thr)
        )

        img[img < 0] = 0
        
        if selection_mode == "l2h_longest":
            # Любая новая допустимая выдержка заменяет предыдущую.
            # Так как выдержки отсортированы по возрастанию,
            # в результате останется самая длинная.
            mask_add = mask_valid
        elif selection_mode == "l2h":
            # Заполняем только пиксели, для которых допустимая
            # выдержка ещё не была найдена.
            mask_add = mask_valid & ~mask_valid_any
        else:
            raise ValueError(f"Неправильный режим selection_mode={selection_mode}")

        if exposure_coefs is None:
            exposure_time_eff = np.floor(exposure_time / 23.4) * 23.4
            exposure_coef = t_reference / exposure_time_eff
        else:
            exposure_coef = exposure_coefs[exposure_time]
        img_scaled = img * exposure_coef

        # Выдержки идут по возрастанию.
        # Каждая более длинная допустимая выдержка заменяет предыдущую.
        hdr[mask_add] = img_scaled[mask_add]
        mask_valid_any |= mask_add
        print(f"\tex_time={exposure_time}, add {np.sum(mask_add)} pix, exposure_coef={exposure_coef}")
    print()

    mask_invalid = ~mask_valid_any
    # Пиксели, которые ни разу не были допустимыми, но при этом 
    # не были постоянно только ниже или только выше порога.
    # Например, на короткой выдержке значение было < low_thr,
    # а на следующей сразу стало > top_thr.
    mask_invalid_mixed = (
        mask_invalid &
        ~mask_always_below &
        ~mask_always_above
    )
    num_pixels = mask_valid_any.size
    num_invalid = np.count_nonzero(mask_invalid)
    num_always_below = np.count_nonzero(mask_always_below)
    num_always_above = np.count_nonzero(mask_always_above)
    num_invalid_mixed = np.count_nonzero(mask_invalid_mixed)

    print( "Количество пикселей без допустимой выдержки: " f"{num_invalid}/{num_pixels}" )
    print( f"Всегда ниже low_thr={low_thr}: " f"{num_always_below}/{num_pixels}" )
    print( f"Всегда выше top_thr={top_thr}: " f"{num_always_above}/{num_pixels}" )
    print( "Не попали в допустимый диапазон из-за перехода " f"между порогами: {num_invalid_mixed}/{num_pixels}" )

    return hdr, mask_valid_any


# сборка изображения
def get_signal_cam_comb_many_exposure_low2hight(
        cfg: ExperimentConfig,
        files_mode: str = "bmp",
        top_thr: float = 240,
        low_thr: float = 5,
        selection_mode="l2h_longest"
):
    exposure_time_arr = np.array(cfg.exposure_time_arr)
    exposure_time_arr = np.sort(cfg.exposure_time_arr)

    hdr_signal, mask_signal = combine_exposures_valid(
        directory=cfg.dir_signal,
        exposure_time_arr=exposure_time_arr,
        image_shape=(cfg.H, cfg.W),
        back_level=cfg.cam_hdr_back_level,
        top_thr=top_thr,
        low_thr=low_thr,
        files_mode=files_mode,
        selection_mode=selection_mode,
        exposure_coefs=cfg.cam_hdr_exposure_coefs,
    )

    if cfg.dir_back is not None:
        hdr_back, mask_back = combine_exposures_valid(
            directory=cfg.dir_back,
            exposure_time_arr=exposure_time_arr,
            image_shape=(cfg.H, cfg.W),
            back_level=cfg.cam_hdr_back_level,
            top_thr=top_thr,
            low_thr=low_thr,
            files_mode=files_mode,
            selection_mode=selection_mode,
            exposure_coefs=cfg.cam_hdr_exposure_coefs,
        )
    else:
        hdr_back = np.zeros_like(hdr_signal)
        mask_back = np.ones_like(mask_signal)

    mask_difference_valid = mask_signal & mask_back
    signal = np.zeros_like(hdr_signal)
    signal[mask_difference_valid] = np.maximum(
        hdr_signal[mask_difference_valid]
        - hdr_back[mask_difference_valid],
        0.0
    )
    print(f"Количество невалидных пикселей в итоговом изображении: {mask_difference_valid.size-np.sum(mask_difference_valid)}/{mask_difference_valid.size}")

    # if 

    return signal, mask_difference_valid


def get_signal_cam_comb_many_exposure_diff_one_exp(
        cfg: ExperimentConfig,
        files_mode: str = "bmp",
        top_thr: float = 240,
        low_thr: float = 5,
        selection_mode="l2h_longest"
):
    """Вычитает фон на каждой выдержке, затем собирает HDR."""
    return combine_exposures_valid(
        directory=cfg.dir_signal,
        background_directory=cfg.dir_back,
        exposure_time_arr=cfg.exposure_time_arr,
        image_shape=(cfg.H, cfg.W),
        back_level=cfg.cam_hdr_back_level,
        top_thr=top_thr,
        low_thr=low_thr,
        files_mode=files_mode,
        selection_mode=selection_mode,
        exposure_coefs=cfg.cam_hdr_exposure_coefs,
    )


def get_signal_cam_hdr(
        cfg: ExperimentConfig,
):
    mode = cfg.cam_hdr_mode
    diff_mode = cfg.cam_hdr_diff_mode
    files_mode = cfg.cam_hdr_files_mode
    top_thr = cfg.cam_hdr_top_thr
    low_thr = cfg.cam_hdr_low_thr

    if diff_mode == "per_exposure":
        print(f"get_signal_cam_hdr: {diff_mode=}")
        return get_signal_cam_comb_many_exposure_diff_one_exp(
            cfg,
            files_mode,
            top_thr,
            low_thr,
            mode,
        )
    elif diff_mode == "after_hdr":
        return get_signal_cam_comb_many_exposure_low2hight(
            cfg,
            files_mode,
            top_thr,
            low_thr,
            mode,
        )
    else:
        raise ValueError(f"Wrong cam_hdr_diff_mode={diff_mode!r}")



def get_signal_lin(
        bins_lin,
        filename,
        filename_dark,
        num_pix=None,
        shift_lin_m=None,
        pix_max_ampl=None,
        width_pix_x_m=None,
        width_pix_y_m=None,
        signal_type=None,
        mode=None,
        path_save_dir=None,
):
    df = pd.read_csv(
        filename,
        sep="\t",
        decimal=",",
    )
    pixels_data = df.iloc[:, 0].astype(int).to_numpy()
    amplitude = df.iloc[:, 1].to_numpy()
    pixels_data = pixels_data.max() - pixels_data

    pixels = np.linspace(0, num_pix, num_pix, endpoint=False, dtype=np.int64)
    signal_lin = np.zeros(num_pix, dtype=amplitude.dtype)
    signal_lin[pixels_data] = amplitude

    if filename_dark is not None:
        df_dark = pd.read_csv(
            filename_dark,
            sep="\t",
            decimal=",",
        )
        pixels_data_dark = df_dark.iloc[:, 0].astype(int).to_numpy()
        amplitude_dark = df_dark.iloc[:, 1].to_numpy()
        pixels_data_dark = pixels_data_dark.max() - pixels_data_dark

        assert np.all(pixels_data_dark == pixels_data), "У данных и фона разные пиксели"
        # TODO: на самом деле, это неотсортированный массив. 
        # Мб пиксели одни, но в разном порядке

        signal_lin[pixels_data_dark] -= amplitude_dark

    if path_save_dir is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(pixels_data, amplitude, label="signal")
        if filename_dark is not None:
            plt.plot(pixels_data_dark, amplitude_dark, label="dark")
        plt.plot(pixels, signal_lin, label="signal-dark")

    # print(f"{np.maximum(signal_lin[pixels_data], 0).shape=}")
    if mode == 1:
        signal_lin[pixels_data] -= np.min(signal_lin[pixels_data])
    elif mode == 2:
        # signal_lin[pixels_data] -= np.min(signal_lin[pixels_data])
        signal_lin[pixels_data] = np.maximum(signal_lin[pixels_data], 0)
        # plt.plot(pixels_data, np.maximum(signal_lin[pixels_data], 0), label="max")
    else:
        raise ValueError(f"Wrong value {mode=}")
    
    if path_save_dir is not None:
        plt.plot(pixels, signal_lin, label="signal_lin_after_mode")

    sigma_lin = 3
    signal_lin_orig = signal_lin.copy()
    signal_lin = gaussian_filter(signal_lin, sigma=sigma_lin)

    if path_save_dir is not None:
        plt.plot(pixels, signal_lin, label="signal_lin_final")
        plt.xlabel("pixel, num")
        plt.ylabel("signal")
        plt.title("Сигнал линейки")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_save_dir/f"lin_signal.png")
        plt.close()

    x_distance_m_lin = shift_lin_m + (pixels - pix_max_ampl) * width_pix_x_m

    bin_sums = []
    bin_centers = []
    for _, row in bins_lin.iterrows():
        r_in = row["r_in_m"]
        r_out = row["r_out_m"]
        mask = (x_distance_m_lin >= r_in) & (x_distance_m_lin < r_out)
        signal_sum = signal_lin[mask].sum()

        num_pix_into_bin = np.sum(mask)
        bin_area = num_pix_into_bin * width_pix_x_m * width_pix_y_m
        if signal_type == "intensity":
            if bin_area > 0:
                signal_sum /= bin_area
            elif bin_area == 0:
                if signal_sum != 0:
                    raise Exception("Calculation error")
            else:
                raise ValueError("Error")
        elif signal_type != "signal":
            raise ValueError(f"Wrong {signal_type=}")

        bin_sums.append(signal_sum)
        bin_centers.append(0.5 * (r_in + r_out))
    bin_sums = np.array(bin_sums)
    bin_centers = np.array(bin_centers)
    return bin_sums, bin_centers


def get_laser_center_pos_pix(
    detector_configuration_type,
    signal_shape,
    x_shift_pix,
    y_shift_pix,
):
    if detector_configuration_type == 0:
        center_laser_pix = (
            signal_shape[1] + x_shift_pix,
            signal_shape[0] // 2 + y_shift_pix
        )
    elif detector_configuration_type == 1:
        center_laser_pix = (
            -x_shift_pix,
            signal_shape[0] // 2 - y_shift_pix
        )
    else:
        raise ValueError(f"Wrong value {detector_configuration_type=}")
    return center_laser_pix


def get_signal_cam(bins_cam, filename_cam, filename_cam_dark,
    cam_pixel_width_m = None,
    x_shift_pix = None,
    y_shift_pix = None,
    plotted=True,
    path_save_dir=None,
    postfix="",
    data=None,
    signal_type=None,
    detector_configuration_type=None,
):
    if not (path_save_dir is None):
        path_save_dir = Path(path_save_dir)
        path_save_dir.mkdir(exist_ok=True, parents=True)
    
    if data is None:
        img_orig = Image.open(filename_cam)
        data = np.array(img_orig)
        print(f"Минимальное {data.min()} и максимальное {data.max()} значения в 'сигнал'")
        print(f"Количество ненулевых пикселей в 'сигнал': {np.sum(data > 0)}/{data.size}")

        if plotted:
            plt.figure(figsize=(10, 8))
            plt.imshow(data, cmap='gray')
            plt.colorbar(label='Intensity')
            plt.xlabel('Pixel X')
            plt.ylabel('Pixel Y')
            plt.title("Исходный сигнал")
            plt.tight_layout()
            if path_save_dir:
                plt.savefig(path_save_dir/f"cam_signal_orig_{postfix}.png")
            else:
                plt.show()
            plt.close()
        
        if filename_cam_dark is not None:
            img_dark = Image.open(filename_cam_dark)
            data_dark = np.array(img_dark)
            data = data.astype(np.int16) - data_dark.astype(np.int16)
            print(f"Минимальное {data_dark.min()} и максимальное {data_dark.max()} значения в 'фон'")
            print(f"Количество ненулевых пикселей в 'фон': {np.sum(data_dark > 0)}/{data_dark.size}")
            print(f"Минимальное {data.min()} и максимальное {data.max()} значения в 'сигнал-фон' до обрезания снизу")
            print(f"Количество отрицательных пикселей в 'сигнал-фон' до обрезания снизу: {np.sum(data < 0)}/{data.size}")

            if plotted:
                plt.figure(figsize=(10, 8))
                plt.imshow(data_dark, cmap='gray')
                plt.colorbar(label='Intensity')
                plt.xlabel('Pixel X')
                plt.ylabel('Pixel Y')
                plt.title("Фон")
                plt.tight_layout()
                if path_save_dir:
                    plt.savefig(path_save_dir/f"cam_signal_back_{postfix}.png")
                else:
                    plt.show()
                plt.close()

                plt.figure(figsize=(10, 8))
                plt.imshow(data, cmap='gray')
                plt.colorbar(label='Intensity')
                plt.xlabel('Pixel X')
                plt.ylabel('Pixel Y')
                plt.title("Разница")
                plt.tight_layout()
                if path_save_dir:
                    plt.savefig(path_save_dir/f"cam_signal_orig_minus_signal_back_{postfix}.png")
                else:
                    plt.show()
                plt.close()

            data = np.clip(data, 0, 255).astype(np.uint8)
    
    if plotted: 
        plt.figure(figsize=(10, 8))
        plt.imshow(data, cmap='gray')
        plt.colorbar(label='Intensity')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title("Сигнал")
        plt.tight_layout()
        if path_save_dir:
            plt.savefig(path_save_dir/f"cam_signal_{postfix}.png")
        else:
            plt.show()
        plt.close()


    center_laser_pix = get_laser_center_pos_pix(
        detector_configuration_type,
        data.shape,
        x_shift_pix,
        y_shift_pix,
    )
    print(f"matrix shift_x: {x_shift_pix*cam_pixel_width_m*1e6:.3f} um")
    print(f"matrix shift_y: {y_shift_pix*cam_pixel_width_m*1e6:.3f} um")
    print(f"center_laser_pix: {center_laser_pix}")

    ny, nx = data.shape
    y_pix, x_pix = np.indices((ny, nx))
    dx_m = (x_pix - center_laser_pix[0]) * cam_pixel_width_m
    dy_m = (y_pix - center_laser_pix[1]) * cam_pixel_width_m
    r_m = np.sqrt(dx_m**2 + dy_m**2)


    bin_signal = []
    for _, row in bins_cam.iterrows():
        r_in = row["r_in_m"]
        r_out = row["r_out_m"]
        mask = (r_m >= r_in) & (r_m < r_out)
        signal_sum = data[mask].sum()
        
        num_pix_into_bin = np.sum(mask)
        bin_area = num_pix_into_bin * cam_pixel_width_m**2
        if signal_type == "intensity":
            if bin_area > 0:
                signal_sum /= bin_area
            elif bin_area == 0:
                if signal_sum != 0:
                    raise Exception("Calculation error")
            else:
                raise ValueError("Error")
        elif signal_type != "signal":
            raise ValueError(f"Wrong {signal_type=}")

        bin_signal.append(signal_sum)

    if plotted:
        plt.figure(figsize=(10, 8))
        plt.imshow(data, cmap='gray')
        ax = plt.gca()
        for _, row in bins_cam.iterrows():
            r_in_pix = row["r_in_m"] / cam_pixel_width_m
            r_out_pix = row["r_out_m"] / cam_pixel_width_m
            ax.add_patch(
                Circle(
                    center_laser_pix,
                    r_in_pix,
                    fill=False,
                    color="red",
                    linewidth=0.5,
                )
            )
        # plt.xlim(0, nx)
        # plt.ylim(0, ny)
        plt.colorbar(label='Intensity')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title("Итоговый сигнал")
        plt.tight_layout()
        if path_save_dir:
            plt.savefig(path_save_dir/f"cam_signal_bins_{postfix}.png")
        else:
            plt.show()
        plt.close()

    # result_cam = bins_cam.copy()
    # bin_centers = (
    #     bins_cam["r_in_m"] + bins_cam["r_out_m"]
    # ) / 2
    # result_cam = pd.DataFrame({
    #     "r_center_m": bin_centers,
    #     "signal": bin_signal
    # })

    # return result_cam
    return np.array(bin_signal, dtype=np.float64)

def get_signal(bins_lin, filename_lin, bins_cam, filename_cam, filename_cam_dark):
    result_lin = get_signal_lin(bins_lin, filename_lin)
    result_cam = get_signal_cam(bins_cam, filename_cam, filename_cam_dark)
    b = result_lin + result_cam
    b = np.array(b)
    return b


def get_avg_img(dir_path):
    dir_path = Path(dir_path)
    # Найти все значения экспозиции
    exposures = sorted({
        file.stem.split("_")[0]
        for file in dir_path.glob("*_*.bmp")
    })

    for exp in exposures:
        files = sorted(dir_path.glob(f"{exp}_*.bmp"))

        imgs = [np.array(Image.open(f), dtype=np.float32) for f in files]

        avg_img = np.mean(imgs, axis=0)

        # Если исходные BMP 8-битные
        avg_img = np.round(avg_img).astype(np.uint8)

        out_file = dir_path / f"{exp}.bmp"
        Image.fromarray(avg_img).save(out_file)

        print(f"Сохранён {out_file}")


# ----- Восстановление -----
# ----- Частицы -----
def count_particles_in_volume(d_um, vol_frac_percent, volume_cm3=1.0):    
    d_um = np.asarray(d_um, float)
    f = np.asarray(vol_frac_percent, float) / 100.0
    V_total_um3 = volume_cm3 * 1e12  # 1 см³ = 10^12 мкм³
    N = f * (6 * V_total_um3) / (np.pi * d_um**3)
    return d_um, N


def get_count_particles_in_vol_1cm3(d_nm: float):
    d_gost = np.array([
        0.16981633596880844, 0.19293940929858155, 0.23532238664008012,
        0.32609716455167675, 0.47153213316004167, 0.7216327285875204,
        1.152395736105811, 1.6901532234248118, 2.4439389281550383,
        3.533902963216935, 5.109976362159404, 6.981399734261977,
        10.239221558345461, 15.893956967991773, 25.381509893008634,
        38.29677627097056, 54.59669133725947, 84.7483822636758,
        137.27043250982584, 204.20275714945305, 308.1104054657541,
        433.0622300986328, 591.6623334147585, 905.4799748284329
    ])
    y_gost = np.array([
        0.001788323913381938, 0.0010181875647686097, 0.000694204268698927,
        0.0004733111887782002, 0.00040425334330881034, 0.0004733111887782002,
        0.000678738593316342, 0.0011395974472006476, 0.001913376304890549,
        0.0032125457047220123, 0.0051561882769015945, 0.008657204784735615,
        0.012414620944861114, 0.019047739555785037, 0.029890842794669675,
        0.04690648359397067, 0.06430121483616155, 0.09645937035317578,
        0.15834655182844276, 0.23224646874885746, 0.34839696573920265,
        0.4565529925065831, 0.6258604313505132, 0.9602569863053698
    ])
    d_rel, N_rel = count_particles_in_volume(d_gost, y_gost) 
    d_um = d_nm / 1000  
    f = interp1d(np.log10(d_rel), np.log10(N_rel), kind='linear', fill_value="extrapolate")
    N_log10 = f(np.log10(d_um))
    N = 10 ** N_log10
    return N


# ----- Решение обратной задачи -----
def make_signal(b, noise, b_back):
    signal = b + noise - b_back
    mask = signal < 0
    signal[mask] = 0
    return signal


def difference_matrix(n: int, order: int) -> np.ndarray:
    if order == 0:
        return np.eye(n)
    elif order == 1:
        L = np.zeros((n - 1, n))
        for i in range(n - 1):
            L[i, i:i + 2] = [-1, 1]
        return L
    elif order == 2:
        L = np.zeros((n - 2, n))
        for i in range(n - 2):
            L[i, i:i + 3] = [1, -2, 1]
        return L
    elif order == 3:
        L = np.zeros((n - 3, n))
        for i in range(n - 3):
            L[i, i:i + 4] = [1, -3, 3, -1]
        return L
    else:
        raise ValueError("order должен быть 0, 1, 2 или 3")


def solve_tikhonov_nnls(A, b, L, alpha):
    """Решает min ||Ax-b||² + alpha||Lx||² при x >= 0."""
    A_aug = np.vstack([
        A,
        np.sqrt(alpha) * L,
    ])
    b_aug = np.concatenate([
        b,
        np.zeros(L.shape[0], dtype=b.dtype),
    ])

    x, _ = nnls(A_aug, b_aug)
    return x


def nnls_gcv(A, b, L, alpha, active_tol=1e-12):
    """
    Приближённая GCV с числом степеней свободы,
    вычисленным по активному набору NNLS.
    """
    x = solve_tikhonov_nnls(A, b, L, alpha)

    residual = A @ x - b
    residual_norm_sq = residual @ residual

    active = x > active_tol
    n_active = np.count_nonzero(active)

    if n_active == 0:
        trace_h = 0.0
    else:
        A_active = A[:, active]
        L_active = L[:, active]

        normal_matrix = (
            A_active.T @ A_active
            + alpha * (L_active.T @ L_active)
        )

        try:
            # trace(A M^{-1} A.T) = trace(M^{-1} A.T A)
            influence_matrix = np.linalg.solve(
                normal_matrix,
                A_active.T @ A_active,
            )
            trace_h = np.trace(influence_matrix)
        except np.linalg.LinAlgError:
            return x, np.inf

    denominator = A.shape[0] - trace_h

    if denominator <= np.finfo(float).eps:
        return x, np.inf

    gcv = residual_norm_sq / denominator**2
    return x, gcv


def find_best_alpha(A, b, L, alphas):
    gcv_curve = np.empty((len(alphas), 2), dtype=float)

    best_alpha = None
    best_gcv = np.inf

    for i, alpha in enumerate(alphas):
        try:
            _, gcv = nnls_gcv(A, b, L, alpha)
        except (ValueError, np.linalg.LinAlgError):
            gcv = np.inf

        gcv_curve[i] = alpha, gcv

        if np.isfinite(gcv) and gcv < best_gcv:
            best_gcv = gcv
            best_alpha = alpha

    if best_alpha is None:
        raise RuntimeError(
            "Не удалось найти допустимое значение alpha"
        )

    return best_alpha, gcv_curve


def inverse_solver(A, b):
    reg_order = REGULARIZATION_TYPE
    L = difference_matrix(A.shape[1], reg_order)

    if REGULARIZATION_ALPHA == "best":
        alpha_grid = np.logspace(-10, 40, 1000)
        best_alpha, gcv_curve = find_best_alpha(
            A, b, L, alpha_grid
        )

    elif isinstance(REGULARIZATION_ALPHA, (int, float)):
        best_alpha = float(REGULARIZATION_ALPHA)
        gcv_curve = np.array([[best_alpha, 1.0]])

        if best_alpha < 0:
            raise ValueError(
                "REGULARIZATION_ALPHA должен быть неотрицательным"
            )

    else:
        raise ValueError(
            "REGULARIZATION_ALPHA должен быть 'best' "
            "или неотрицательным числом"
        )

    x_nnls = solve_tikhonov_nnls(
        A, b, L, best_alpha
    )

    return x_nnls, best_alpha, gcv_curve


def chahine_inversion(b, K, num_iter=50, q0=None, tol=1e-6, verbose=False):
    """
    Chahine iterative inversion for particle size distribution.

    Solves the linear system: K * q = b, where
        b : measured intensity vector (length N, N = number of angular positions)
        K : scattering matrix (shape N x M, M = number of size classes)
        q : unknown size distribution (length M)

    The algorithm enforces non‑negativity and is stable for ill‑posed problems.

    Parameters
    ----------
    b : ndarray, shape (N,)
        Measured light intensity at each angle.
    K : ndarray, shape (N, M)
        Scattering matrix (kernel).
    num_iter : int, optional
        Maximum number of iterations (default 50).
    q0 : ndarray, shape (M,), optional
        Initial guess for the size distribution. If None, a uniform distribution is used.
    tol : float, optional
        Stopping criterion: relative change in q between iterations (default 1e-6).
    verbose : bool, optional
        Print progress if True (default False).

    Returns
    -------
    q : ndarray, shape (M,)
        Recovered particle‑size distribution (volume or number fraction per size class).

    Notes
    -----
    The implementation follows the standard Chahine update:
        q_j^{new} = q_j^{old} * ( sum_i K_{i,j} * (b_i / (K q)_i) ) / ( sum_i K_{i,j} )

    This matches the algorithm outlined in the paper (with corrected indexing).
    For improved conditioning, consider scaling the data and kernel as in Eq. (6)
    (e.g., using a = 1 or 2) before calling this function.
    """
    N, M = K.shape
    if b.shape[0] != N:
        raise ValueError("Length of b must match number of rows in K")

    # Initial guess
    if q0 is None:
        q = np.ones(M, dtype=float) / M   # uniform distribution
    else:
        q = q0.astype(float).copy()

    # Pre‑compute denominator for each size class: sum_i K[i, j]
    denom = np.sum(K, axis=0)   # shape (M,)
    # Avoid division by zero (if a column sum is zero, that size class is not measured)
    denom = np.where(denom < 1e-15, 1.0, denom)

    # Small epsilon to prevent division by zero in ratio
    eps = 1e-12

    for it in range(num_iter):
        # Compute modelled intensities
        I_calc = K @ q          # shape (N,)

        # Ratio of measured to modelled (element‑wise)
        ratio = b / (I_calc + eps)

        # Update each size class
        q_new = np.zeros(M)
        for j in range(M):
            numerator = np.sum(K[:, j] * ratio)
            q_new[j] = q[j] * numerator / denom[j]

        # Check convergence
        rel_change = np.linalg.norm(q_new - q) / (np.linalg.norm(q) + 1e-12)
        q = q_new

        if verbose:
            print(f"Iteration {it+1:3d}, relative change = {rel_change:.6e}")

        if rel_change < tol:
            break

    return q

from copy import deepcopy
def inverse_solver_type_1(
        A, b, noise, b_back, classes, 
        use_w_critical=True,
        use_chahine=False,
        use_conc_corr=True,
):
    print(f"inverse_solver_type_1: {use_w_critical=}, {use_chahine=}, {use_conc_corr=}")
    sizes = np.array([(item[0] + item[1])/2 for item in classes])
    
    signal = make_signal(b, noise, b_back)
    x_nnls, alpha_reg, gcv_curve = inverse_solver(A, signal)
    print(f"{alpha_reg=:.3e}")

    bin_edges = np.array(classes)
    bin_mins = bin_edges[:, 0]
    bin_maxs = bin_edges[:, 1]
    centers = (bin_mins + bin_maxs)/2

    restored_gost_reg = deepcopy(x_nnls)

    restored_gost_reg_iter = deepcopy(restored_gost_reg)
    if use_chahine:
        restored_gost_reg_iter = chahine_inversion(signal, A, num_iter=1000, q0=restored_gost_reg, tol=1e-6, verbose=False)

    restored_gost_reg_iter_w = deepcopy(restored_gost_reg_iter)
    if use_w_critical:
        restored_gost_reg_iter_w[restored_gost_reg_iter_w < W_CRITICAL] = 0
    
    restored_distr_reg = None
    restored_distr_reg_iter = None
    restored_distr_reg_iter_w = None
    if use_conc_corr:
        coef = get_count_particles_in_vol_1cm3(centers)
        restored_distr_reg = restored_gost_reg * coef
        restored_distr_reg_iter = restored_gost_reg_iter * coef
        restored_distr_reg_iter_w = restored_gost_reg_iter_w * coef
    
    return (
        sizes, 
        restored_distr_reg, 
        restored_distr_reg_iter, 
        restored_distr_reg_iter_w, 
        restored_gost_reg, 
        restored_gost_reg_iter, 
        restored_gost_reg_iter_w, 
        gcv_curve,
        alpha_reg
    )


def read_classes(filename):
    classes = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            min_size = float(parts[1])
            max_size = float(parts[2])
            classes.append((min_size, max_size))
    classes = np.array(classes)
    return classes


# построение графиков
def plot_gcv_curve(gcv_curve, alpha_reg, save_path):
    gcv_curve_cor = gcv_curve[gcv_curve[:, 1] < np.inf]
    plt.figure(figsize=(10, 5))
    plt.plot(gcv_curve[:, 0], gcv_curve[:, 1], label="gcv")
    plt.vlines([alpha_reg], np.min(gcv_curve_cor[:,1]), np.max(gcv_curve_cor[:,1]), 'black', '--', label="alpha_reg")
    plt.xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("gcv")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_cam_signal(
        signal_cam,
        save_path
):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(8, 6)
    )
    cmap_style = plt.colormaps["gray"].copy()
    cmap_style.set_bad(color="yellow")

    ax1 = axes
    im1 = ax1.imshow(
        signal_cam,
        cmap=cmap_style
    )
    ax1.set_title("Сигнал HDR")
    fig.colorbar(im1, ax=ax1)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_cam_signal_valid(
        signal_cam,
        cam_mask_valid,
        save_path
):
    signal_cam_masked = np.ma.masked_where(
        ~cam_mask_valid,
        signal_cam
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 6)
    )
    cmap_style = plt.colormaps["gray"].copy()
    cmap_style.set_bad(color="yellow")

    ax1 = axes[0]
    im1 = ax1.imshow(
        signal_cam,
        cmap=cmap_style
    )
    ax1.set_title("Сигнал HDR")
    fig.colorbar(im1, ax=ax1)

    ax2 = axes[1]
    im2 = ax2.imshow(
        signal_cam_masked,
        cmap=cmap_style
    )
    ax2.set_title("Сигнал HDR с валидными пикселями")
    fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_b_signal(b, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(b, marker="o")
    plt.xlabel("№ бинa")
    plt.ylabel("Суммарный сигнал")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def print_cam_pix_valid_per_bin(
    signal_mask,
    bins,
    cam_pixel_width_m,
    center_laser_pix,
    dir_save,
):
    print("\nВалидные пиксели в бинах камеры")
    dir_save = Path(dir_save)
    dir_save.mkdir(parents=True, exist_ok=True)
    path_save = dir_save / "cam_pix_valid_per_bin.txt"

    x_center, y_center = center_laser_pix
    height, width = signal_mask.shape
    y, x = np.indices((height, width))
    r2 = (x - x_center) ** 2 + (y - y_center) ** 2

    with open(path_save, "w", encoding="utf-8") as f:
        f.write(
            "bin\t"
            "r_in_um\t"
            "r_out_um\t"
            "valid_pixels\t"
            "total_pixels\t"
            "valid_percent\n"
        )

        for i, row in bins.iterrows():
            r_in_m = row["r_in_m"]
            r_out_m = row["r_out_m"]

            r_in_pix = r_in_m / cam_pixel_width_m
            r_out_pix = r_out_m / cam_pixel_width_m

            bin_mask = (
                (r2 >= r_in_pix ** 2)
                & (r2 < r_out_pix ** 2)
            )

            total_count = np.count_nonzero(bin_mask)
            valid_count = np.count_nonzero(
                bin_mask & signal_mask
            )

            valid_percent = (
                100 * valid_count / total_count
                if total_count > 0 else 0
            )

            print(
                f"bin {i}: "
                f"valid={valid_count}/{total_count}, "
                f"({valid_percent:.2f}%)"
            )

            r_in_um = r_in_m * 1e6
            r_out_um = r_out_m * 1e6
            f.write(
                f"{i}\t"
                f"{r_in_um:.2f}\t"
                f"{r_out_um:.2f}\t"
                f"{valid_count}\t"
                f"{total_count}\t"
                f"{valid_percent:.2f}\n"
            )

    print(f"Saved to: {path_save}\n")
