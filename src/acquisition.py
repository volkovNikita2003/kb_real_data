"""Line-sensor acquisition with atomic publication of completed signals."""

from pathlib import Path
import tempfile

from errors import ParametersError
from experiment import Experiment, validate_safe_name
from parameters import load_measurement_parameters


def emulate_line_sensor(exposure_us: int, destination: Path) -> None:
    """Emulate acquisition by writing ten pixel values, one per line."""
    destination.write_text("".join(f"{pixel}\n" for pixel in range(10)), encoding="utf-8")


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
    exposure = parameters.acquisition.line_sensor.exposure_us
    directory = experiment.data_dir / measurement / ("lin_back" if background else "lin")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{exposure}.txt"
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        emulate_line_sensor(exposure, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
