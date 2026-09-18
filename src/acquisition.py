"""Line-sensor acquisition with atomic publication of completed signals."""

from pathlib import Path
import tempfile
import subprocess

from errors import ParametersError
from experiment import Experiment, validate_safe_name
from parameters import LineSensorAcquisition, load_measurement_parameters


LINE_PROGRAM = Path(__file__).resolve().parents[1] / "build" / "measure_lin"


def acquire_line_signal(directory: Path, settings: LineSensorAcquisition) -> Path:
    """Run the C++ acquisition program in an isolated output directory."""
    result = (directory / f"{settings.exposure_us}.txt").resolve()
    command = [str(LINE_PROGRAM), str(settings.n), settings.device,
               str(settings.exposure_us), str(settings.pixels), str(result)]
    try:
        completed = subprocess.run(command, cwd=directory, check=True,
                       capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Программа {LINE_PROGRAM} не найдена. Выполните cmake -S . -B build "
            "и cmake --build build --target measure_lin в real_data_auto."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Ошибка получения сигнала линейки (код {error.returncode}): "
            f"{error.stderr or error.stdout or 'нет диагностического сообщения'}"
        ) from error
    # The external void function reports failures on stderr instead of returning
    # an error status. Do not publish its output when it reports a failure.
    if completed.stderr.strip():
        raise RuntimeError(f"Ошибка получения сигнала линейки: {completed.stderr.strip()}")
    if not result.is_file() or result.stat().st_size == 0:
        raise RuntimeError("Программа линейки не сохранила сигнал")
    return result


def run_line_measurement(
    experiment_path: str | Path, measurement: str, *, background: bool = False,
) -> Path:
    """Validate settings and replace the signal only after successful acquisition."""
    validate_safe_name(measurement, kind="измерения")
    experiment = Experiment.open(experiment_path, for_acquisition=True)
    source = experiment.measurement_parameters_dir / f"{measurement}.yaml"
    parameters = load_measurement_parameters(source)
    if parameters.acquisition is None or parameters.acquisition.line_sensor is None:
        raise ParametersError(
            f"{source}: для measure_lin необходима секция acquisition.line_sensor"
        )
    settings = parameters.acquisition.line_sensor
    exposure = settings.exposure_us
    directory = experiment.data_dir / measurement / ("lin_back" if background else "lin")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{exposure}.txt"
    with tempfile.TemporaryDirectory(dir=directory, prefix=".measure_lin-") as staging:
        temporary = acquire_line_signal(Path(staging), settings)
        temporary.replace(destination)
    return destination
