"""Render and publish the text configuration consumed by legacy ``code_git``."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

from experiment import Experiment, validate_safe_name
from errors import DarlError
from parameters import DarlStageParameters


class LegacyDarlConfigError(DarlError):
    """A legacy DARL configuration cannot be rendered or published."""


@dataclass(frozen=True)
class LegacyConfigArtifact:
    """The exact legacy configuration made available to ``code_git``."""

    config_name: str
    path: Path
    text: str


def default_code_git_dir() -> Path:
    """Return the sibling ``code_git`` checkout used by the current project."""
    return Path(__file__).resolve().parents[4] / "code_git"


def legacy_config_name(experiment: Experiment) -> str:
    """Build a deterministic safe config/case name for one experiment."""
    return validate_safe_name(f"auto_{experiment.name}", kind="конфигурации DARL")


def _number(value: int | float) -> str:
    """Preserve enough decimal digits for a lossless float round trip."""
    if type(value) is int:
        return str(value)
    return format(value, ".17g")


def _enabled(value: object | None) -> str:
    return "да" if value is not None else "нет"


def render_legacy_config(parameters: DarlStageParameters) -> str:
    """Render all fields understood by the reference ``load_config_to_state``."""
    if not isinstance(parameters, DarlStageParameters):
        raise LegacyDarlConfigError(
            "parameters: ожидался объект DarlStageParameters"
        )

    general = parameters.general
    darl = parameters.darl
    calibration = parameters.calibration_result
    quality = parameters.quality_control

    camera = general.detectors.camera
    camera_result = calibration.camera
    if camera is None:
        camera_diagonal_mm = 0.0
        camera_width_px = 0
        camera_height_px = 0
        camera_shift_x_um = 0.0
        camera_shift_y_um = 0.0
    else:
        camera_diagonal_mm = camera_result.matrix_diagonal_mm
        camera_width_px = camera.width_px
        camera_height_px = camera.height_px
        # Legacy preprocessing obtained these values in micrometres and then
        # stored metres by multiplying by 1e-6.  Division reverses that exact
        # floating-point operation; multiplication by 1e6 changes the final
        # bit for the reference Y shift.
        camera_shift_x_um = camera_result.x_shift_m / 1e-6
        camera_shift_y_um = camera_result.y_shift_m / 1e-6

    line = general.detectors.line_sensor
    line_darl = darl.detectors.line_sensor
    line_result = calibration.line_sensor
    if line is None:
        line_start_angle_deg = 0.0
        line_end_angle_deg = 0.0
        line_logarithmic_radius_percent = 0.0
        line_pixel_height_um = 0.0
    else:
        if line_darl is None or line_result is None:
            raise LegacyDarlConfigError(
                "Для включённой линейки отсутствуют параметры DARL или "
                "результат калибровки"
            )
        line_start_angle_deg = line_result.start_angle_deg
        line_end_angle_deg = line_result.end_angle_deg
        line_logarithmic_radius_percent = (
            line_darl.logarithmic_radius_percent
        )
        line_pixel_height_um = line_result.pixel_height_m * 1e6

    polarization = {
        "parallel": "Parallel",
        "perpendicular": "Perpendicular",
        "unpolarized": "Unpolarised",
    }[darl.laser.polarization]
    particle_type = {"sphere": 0, "rectangle": 1}[darl.particles.type]
    signal_type = {"signal": 0, "intensity": 1}[darl.signal.value_type]

    lines = [
        "Лазер:",
        f"длина_волны_(нм): {_number(general.instrument.wavelength_um * 1000)}",
        f"угол_(град): {_number(darl.laser.angle_deg)}",
        f"этап: {_number(darl.laser.stage)}",
        f"мощность_(Вт): {_number(darl.laser.power_w)}",
        f"поляризация: {polarization}",
        "",
        "Передняя_матрица:",
        f"включена: {_enabled(camera)}",
        f"диагональ_передней_матрицы_(мм): {_number(camera_diagonal_mm)}",
        f"количество_пикселей_длина_(шт): {_number(camera_width_px)}",
        f"количество_пикселей_ширина_(шт): {_number(camera_height_px)}",
        f"отступ_от_оптической_оси_(мкм): {_number(camera_shift_x_um)}",
        f"отступ_от_оптической_оси_Y(мкм): {_number(camera_shift_y_um)}",
        "",
        "Лог_линейка:",
        f"Включена: {_enabled(line)}",
        f"Начальный_угол_(град): {_number(line_start_angle_deg)}",
        f"Конечный_угол_(град): {_number(line_end_angle_deg)}",
        f"Лог_радиус_(%): {_number(line_logarithmic_radius_percent)}",
        f"Ширина_пикселя_(мкм): {_number(line_pixel_height_um)}",
        "",
        "Сигнал:",
        f"одна_частица: {int(darl.signal.one_particle)}",
        f"тип_сигнала: {signal_type}",
        "",
        "Частицы:",
        f"коэффициент_преломления(n): {_number(darl.particles.refractive_index)}",
        f"коэффициент_поглощения(k): {_number(darl.particles.absorption_coefficient)}",
        f"тип: {particle_type}",
        "отношение_сторон_прямоугольной_частицы: "
        f"{_number(darl.particles.rectangle_aspect_ratio)}",
        "",
        "Среда:",
        "внутри_кюветы_(n): "
        f"{_number(darl.medium.inside_cuvette_refractive_index)}",
        f"кювета_(n): {_number(darl.medium.cuvette_refractive_index)}",
        "снаружи_кюветы_(n): "
        f"{_number(darl.medium.outside_cuvette_refractive_index)}",
        "",
        "Классы_частиц:",
        f"Тип_разбиения: {_number(darl.particle_classes.split_type)}",
        "Минимальный_размер_частиц(нм): "
        f"{_number(darl.particle_classes.min_diameter_nm)}",
        "Максимальный_размер_класса(нм): "
        f"{_number(darl.particle_classes.max_diameter_nm)}",
        "Граница_Ми_Фраунгофер(нм): "
        f"{_number(darl.particle_classes.mie_fraunhofer_boundary_nm)}",
        "Размен_классов_в_Ми(нм): "
        f"{_number(darl.particle_classes.mie_class_size_nm)}",
        "Тип_лог_в_Фраунгофере: "
        f"{_number(darl.particle_classes.fraunhofer_log_type)}",
        "Лог_в_Фраунгофере(%): "
        f"{_number(darl.particle_classes.fraunhofer_log_percent)}",
        "",
        "Восстановление:",
        f"Тип: {_number(quality.restoration_type)}",
        "Граница_маленьких_частиц(нм): "
        f"{_number(quality.small_particle_boundary_nm)}",
        "",
        "Параметры_оценки_точности_восстановления:",
        f"Тип_оценки: {_number(quality.evaluation_type)}",
        f"Частота_классов: {_number(quality.class_frequency)}",
        "",
    ]
    return "\n".join(lines)


def write_legacy_config(
    code_git_dir: str | Path,
    config_name: str,
    text: str,
    *,
    overwrite: bool = True,
) -> LegacyConfigArtifact:
    """Atomically publish rendered text in ``code_git/data/configs``."""
    name = validate_safe_name(config_name, kind="конфигурации DARL")
    root = Path(code_git_dir).expanduser().resolve()
    if not root.is_dir():
        raise LegacyDarlConfigError(f"Директория code_git не найдена: {root}")
    configs = root / "data/configs"
    if not configs.is_dir():
        raise LegacyDarlConfigError(
            f"Директория конфигураций code_git не найдена: {configs}"
        )
    destination = configs / f"{name}.txt"
    if destination.is_symlink():
        raise LegacyDarlConfigError(
            f"Путь legacy-конфигурации не должен быть ссылкой: {destination}"
        )
    if destination.exists() and not overwrite:
        raise LegacyDarlConfigError(
            f"Legacy-конфигурация уже существует: {destination}"
        )

    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.",
            suffix=".tmp",
            dir=configs,
            text=True,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        if destination.exists() and not overwrite:
            raise LegacyDarlConfigError(
                f"Legacy-конфигурация уже существует: {destination}"
            )
        temporary.replace(destination)
    except (LegacyDarlConfigError, OSError) as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if isinstance(error, LegacyDarlConfigError):
            raise
        raise LegacyDarlConfigError(
            f"Не удалось записать legacy-конфигурацию {destination}: {error}"
        ) from error

    return LegacyConfigArtifact(name, destination, text)


def build_legacy_config(
    experiment: Experiment,
    parameters: DarlStageParameters,
    *,
    code_git_dir: str | Path | None = None,
    config_name: str | None = None,
    overwrite: bool = True,
) -> LegacyConfigArtifact:
    """Render and publish the legacy configuration for one experiment."""
    name = config_name or legacy_config_name(experiment)
    text = render_legacy_config(parameters)
    return write_legacy_config(
        code_git_dir or default_code_git_dir(),
        name,
        text,
        overwrite=overwrite,
    )
