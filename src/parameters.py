"""Strict YAML parameter models for the real-data processing pipeline.

Input YAML files are intentionally decoded explicitly.  This makes unknown
fields, implicit type conversions, and ambiguous defaults impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from errors import ParametersError
from experiment import Experiment, MeasurementParametersFile, RestoreProfile
from schema_validation import SchemaValidator
from yaml_support import YAML_OMIT_NONE, YamlError, dump_yaml, load_yaml, to_plain_data


SCHEMA_VERSION = 1
_VALIDATOR = SchemaValidator(ParametersError)

_mapping = _VALIDATOR.mapping
_fields = _VALIDATOR.fields
_string = _VALIDATOR.string
_boolean = _VALIDATOR.boolean
_integer = _VALIDATOR.integer
_number = _VALIDATOR.number
_choice = _VALIDATOR.choice


def _schema(data: dict[str, Any], path: str) -> int:
    return _VALIDATOR.version(
        data["schema_version"],
        f"{path}.schema_version",
        supported=SCHEMA_VERSION,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = load_yaml(path)
    except YamlError as error:
        raise ParametersError(str(error)) from error
    return _mapping(value, path.name)


@dataclass(frozen=True)
class InstrumentParameters:
    detector_position: str
    wavelength_um: float = 0.633
    focal_length_um: float = 400_000.0


@dataclass(frozen=True)
class CameraDetectorParameters:
    width_px: int = 2592
    height_px: int = 1944
    pixel_width_m: float = 1.8965517241379312e-6
    pixel_height_m: float = 1.8965517241379312e-6


@dataclass(frozen=True)
class LineSensorDetectorParameters:
    pixel_count: int = 3643
    pixel_width_m: float = 8e-6
    pixel_height_m: float = 0.0002


@dataclass(frozen=True)
class ExperimentDetectors:
    camera: CameraDetectorParameters | None = field(
        default=None, metadata={YAML_OMIT_NONE: True}
    )
    line_sensor: LineSensorDetectorParameters | None = field(
        default=None, metadata={YAML_OMIT_NONE: True}
    )

    def names(self) -> frozenset[str]:
        return frozenset(
            name
            for name in ("camera", "line_sensor")
            if getattr(self, name) is not None
        )


@dataclass(frozen=True)
class GeneralParameters:
    schema_version: int
    detectors: ExperimentDetectors
    instrument: InstrumentParameters


def load_general_parameters(path: str | Path) -> GeneralParameters:
    source = Path(path)
    root = _fields(
        _load_yaml(source),
        source.name,
        required={"schema_version", "detectors", "instrument"},
    )
    version = _schema(root, source.name)

    detector_data = _fields(
        root["detectors"],
        f"{source.name}.detectors",
        optional={"camera", "line_sensor"},
    )

    camera = None
    if "camera" in detector_data:
        path = f"{source.name}.detectors.camera"
        data = _fields(
            detector_data["camera"], path,
            optional={"width_px", "height_px", "pixel_width_m", "pixel_height_m"},
        )
        camera = CameraDetectorParameters(
            width_px=_integer(data.get("width_px", 2592), f"{path}.width_px", positive=True),
            height_px=_integer(data.get("height_px", 1944), f"{path}.height_px", positive=True),
            pixel_width_m=_number(
                data.get("pixel_width_m", 1.8965517241379312e-6),
                f"{path}.pixel_width_m", positive=True,
            ),
            pixel_height_m=_number(
                data.get("pixel_height_m", 1.8965517241379312e-6),
                f"{path}.pixel_height_m", positive=True,
            ),
        )
    line_sensor = None
    if "line_sensor" in detector_data:
        path = f"{source.name}.detectors.line_sensor"
        data = _fields(
            detector_data["line_sensor"], path,
            optional={"pixel_count", "pixel_width_m", "pixel_height_m"},
        )
        line_sensor = LineSensorDetectorParameters(
            pixel_count=_integer(data.get("pixel_count", 3643), f"{path}.pixel_count", positive=True),
            pixel_width_m=_number(data.get("pixel_width_m", 8e-6), f"{path}.pixel_width_m", positive=True),
            pixel_height_m=_number(data.get("pixel_height_m", 0.0002), f"{path}.pixel_height_m", positive=True),
        )

    detectors = ExperimentDetectors(
        camera=camera,
        line_sensor=line_sensor,
    )
    if not detectors.names():
        raise ParametersError(
            f"{source.name}.detectors: должен быть указан хотя бы один детектор"
        )

    instrument_data = _fields(
        root["instrument"],
        f"{source.name}.instrument",
        required={"detector_position"},
        optional={"wavelength_um", "focal_length_um"},
    )
    instrument = InstrumentParameters(
        detector_position=_choice(
            instrument_data["detector_position"],
            f"{source.name}.instrument.detector_position",
            {"old", "new"},
        ),
        wavelength_um=_number(
            instrument_data.get("wavelength_um", 0.633),
            f"{source.name}.instrument.wavelength_um",
            positive=True,
        ),
        focal_length_um=_number(
            instrument_data.get("focal_length_um", 400_000.0),
            f"{source.name}.instrument.focal_length_um",
            positive=True,
        ),
    )
    return GeneralParameters(version, detectors, instrument)


@dataclass(frozen=True)
class CalibrationHdrParameters:
    mode: str = "l2h"
    difference_mode: str = "after_hdr"
    background_level: float = 12.0
    low_threshold: float = 10.0
    top_threshold: float = 240.0


@dataclass(frozen=True)
class AutomaticCameraCalibration:
    pinhole_diameter_um: float = 200.0
    gaussian_sigma_px: float = 20.0
    correct_pixel_size: bool = True
    hdr: CalibrationHdrParameters = field(default_factory=CalibrationHdrParameters)


@dataclass(frozen=True)
class AutomaticLineSensorCalibration:
    pinhole_position_m: float
    signal_position_m: float
    pinhole_diameter_um: float = 200.0
    gaussian_sigma_px: float = 3.0
    time_offset_us: float = 2.0


@dataclass(frozen=True)
class CalibrationParameters:
    schema_version: int
    mode: str
    camera: AutomaticCameraCalibration | None = field(
        default=None, metadata={YAML_OMIT_NONE: True}
    )
    line_sensor: AutomaticLineSensorCalibration | None = field(
        default=None, metadata={YAML_OMIT_NONE: True}
    )


def load_calibration_parameters(path: str | Path) -> CalibrationParameters:
    source = Path(path)
    root = _fields(
        _load_yaml(source),
        source.name,
        required={"schema_version", "mode"},
        optional={"camera", "line_sensor"},
    )
    version = _schema(root, source.name)
    mode = _choice(root["mode"], f"{source.name}.mode", {"automatic"})

    camera = None
    if "camera" in root:
        data = _fields(
            root["camera"],
            f"{source.name}.camera",
            optional={
                "pinhole_diameter_um",
                "gaussian_sigma_px",
                "correct_pixel_size",
                "hdr",
            },
        )
        hdr_data = _fields(
            data.get("hdr", {}),
            f"{source.name}.camera.hdr",
            optional={
                "mode", "difference_mode", "background_level",
                "low_threshold", "top_threshold",
            },
        )
        hdr = CalibrationHdrParameters(
            mode=_choice(
                hdr_data.get("mode", "l2h"),
                f"{source.name}.camera.hdr.mode", {"l2h", "l2h_longest"},
            ),
            difference_mode=_choice(
                hdr_data.get("difference_mode", "after_hdr"),
                f"{source.name}.camera.hdr.difference_mode",
                {"after_hdr", "per_exposure"},
            ),
            background_level=_number(
                hdr_data.get("background_level", 12.0),
                f"{source.name}.camera.hdr.background_level", non_negative=True,
            ),
            low_threshold=_number(
                hdr_data.get("low_threshold", 10.0),
                f"{source.name}.camera.hdr.low_threshold", non_negative=True,
            ),
            top_threshold=_number(
                hdr_data.get("top_threshold", 240.0),
                f"{source.name}.camera.hdr.top_threshold", positive=True,
            ),
        )
        if hdr.low_threshold > hdr.top_threshold:
            raise ParametersError(
                f"{source.name}.camera.hdr: low_threshold не может быть больше top_threshold"
            )
        camera = AutomaticCameraCalibration(
            pinhole_diameter_um=_number(
                data.get("pinhole_diameter_um", 200.0),
                f"{source.name}.camera.pinhole_diameter_um",
                positive=True,
            ),
            gaussian_sigma_px=_number(
                data.get("gaussian_sigma_px", 20.0),
                f"{source.name}.camera.gaussian_sigma_px",
                positive=True,
            ),
            correct_pixel_size=_boolean(
                data.get("correct_pixel_size", True),
                f"{source.name}.camera.correct_pixel_size",
            ),
            hdr=hdr,
        )

    line_sensor = None
    if "line_sensor" in root:
        data = _fields(
            root["line_sensor"],
            f"{source.name}.line_sensor",
            required={"pinhole_position_m", "signal_position_m"},
            optional={
                "pinhole_diameter_um",
                "gaussian_sigma_px",
                "time_offset_us",
            },
        )
        line_sensor = AutomaticLineSensorCalibration(
            pinhole_position_m=_number(
                data["pinhole_position_m"],
                f"{source.name}.line_sensor.pinhole_position_m",
            ),
            signal_position_m=_number(
                data["signal_position_m"],
                f"{source.name}.line_sensor.signal_position_m",
            ),
            pinhole_diameter_um=_number(
                data.get("pinhole_diameter_um", 200.0),
                f"{source.name}.line_sensor.pinhole_diameter_um",
                positive=True,
            ),
            gaussian_sigma_px=_number(
                data.get("gaussian_sigma_px", 3.0),
                f"{source.name}.line_sensor.gaussian_sigma_px",
                positive=True,
            ),
            time_offset_us=_number(
                data.get("time_offset_us", 2.0),
                f"{source.name}.line_sensor.time_offset_us",
                non_negative=True,
            ),
        )
    return CalibrationParameters(version, mode, camera, line_sensor)


@dataclass(frozen=True)
class DarlCamera:
    pass


@dataclass(frozen=True)
class DarlLineSensor:
    logarithmic_radius_percent: float = 7.5


@dataclass(frozen=True)
class DarlDetectors:
    camera: DarlCamera | None = field(
        default=None, metadata={YAML_OMIT_NONE: True}
    )
    line_sensor: DarlLineSensor | None = field(
        default=None, metadata={YAML_OMIT_NONE: True}
    )

    def names(self) -> frozenset[str]:
        return frozenset(
            name
            for name in ("camera", "line_sensor")
            if getattr(self, name) is not None
        )


@dataclass(frozen=True)
class LaserParameters:
    angle_deg: float = 0.0
    stage: int = 1
    power_w: float = 30.0
    polarization: str = "parallel"


@dataclass(frozen=True)
class SignalParameters:
    one_particle: bool = False
    value_type: str = "signal"


@dataclass(frozen=True)
class ParticleParameters:
    refractive_index: float
    absorption_coefficient: float
    type: str
    rectangle_aspect_ratio: float = 1.5


@dataclass(frozen=True)
class MediumParameters:
    inside_cuvette_refractive_index: float
    cuvette_refractive_index: float
    outside_cuvette_refractive_index: float


@dataclass(frozen=True)
class ParticleClasses:
    split_type: int = 1
    min_diameter_nm: float = 100.0
    max_diameter_nm: float = 3_500_000.0
    mie_fraunhofer_boundary_nm: float = 10_000.0
    mie_class_size_nm: float = 300.0
    fraunhofer_log_type: int = 1
    fraunhofer_log_percent: float = 7.5


@dataclass(frozen=True)
class ExpectedDistribution:
    type: str
    mean_nm: float
    sigma_nm: float
    particle_count: float


@dataclass(frozen=True)
class MeasurementParameters:
    schema_version: int
    expected_distribution: ExpectedDistribution


@dataclass(frozen=True)
class DarlQualityControl:
    restoration_type: int = 1
    small_particle_boundary_nm: float = 300.0
    evaluation_type: int = 1
    class_frequency: int = 10


@dataclass(frozen=True)
class DarlParameters:
    schema_version: int
    detectors: DarlDetectors
    particles: ParticleParameters
    medium: MediumParameters
    laser: LaserParameters = field(default_factory=LaserParameters)
    signal: SignalParameters = field(default_factory=SignalParameters)
    particle_classes: ParticleClasses = field(default_factory=ParticleClasses)


def _load_darl_detectors(value: Any, path: str) -> DarlDetectors:
    root = _fields(value, path, optional={"camera", "line_sensor"})
    camera = None
    if "camera" in root:
        _fields(root["camera"], f"{path}.camera")
        camera = DarlCamera()
    line_sensor = None
    if "line_sensor" in root:
        data = _fields(
            root["line_sensor"],
            f"{path}.line_sensor",
            optional={"logarithmic_radius_percent"},
        )
        line_sensor = DarlLineSensor(
            _number(
                data.get("logarithmic_radius_percent", 7.5),
                f"{path}.line_sensor.logarithmic_radius_percent",
                positive=True,
            )
        )
    result = DarlDetectors(camera, line_sensor)
    if not result.names():
        raise ParametersError(f"{path}: должен быть указан хотя бы один детектор")
    return result


def load_darl_parameters(path: str | Path) -> DarlParameters:
    source = Path(path)
    root = _fields(
        _load_yaml(source),
        source.name,
        required={
            "schema_version",
            "detectors",
            "particles",
            "medium",
        },
        optional={"laser", "signal", "particle_classes"},
    )
    version = _schema(root, source.name)
    prefix = source.name
    detectors = _load_darl_detectors(root["detectors"], f"{prefix}.detectors")

    laser_data = _fields(
        root.get("laser", {}),
        f"{prefix}.laser",
        optional={"angle_deg", "stage", "power_w", "polarization"},
    )
    laser = LaserParameters(
        angle_deg=_number(laser_data.get("angle_deg", 0.0), f"{prefix}.laser.angle_deg"),
        stage=_integer(laser_data.get("stage", 1), f"{prefix}.laser.stage", positive=True),
        power_w=_number(laser_data.get("power_w", 30.0), f"{prefix}.laser.power_w", positive=True),
        polarization=_choice(
            laser_data.get("polarization", "parallel"),
            f"{prefix}.laser.polarization",
            {"parallel", "perpendicular", "unpolarized"},
        ),
    )
    signal_data = _fields(
        root.get("signal", {}),
        f"{prefix}.signal",
        optional={"one_particle", "value_type"},
    )
    signal = SignalParameters(
        one_particle=_boolean(
            signal_data.get("one_particle", False),
            f"{prefix}.signal.one_particle",
        ),
        value_type=_choice(
            signal_data.get("value_type", "signal"),
            f"{prefix}.signal.value_type",
            {"signal", "intensity"},
        ),
    )

    particle_data = _fields(
        root["particles"],
        f"{prefix}.particles",
        required={"refractive_index", "absorption_coefficient", "type"},
        optional={"rectangle_aspect_ratio"},
    )
    particle_type = _choice(
        particle_data["type"],
        f"{prefix}.particles.type",
        {"sphere", "rectangle"},
    )
    particles = ParticleParameters(
        refractive_index=_number(
            particle_data["refractive_index"],
            f"{prefix}.particles.refractive_index",
            positive=True,
        ),
        absorption_coefficient=_number(
            particle_data["absorption_coefficient"],
            f"{prefix}.particles.absorption_coefficient",
            non_negative=True,
        ),
        type=particle_type,
        rectangle_aspect_ratio=_number(
            particle_data.get("rectangle_aspect_ratio", 1.5),
            f"{prefix}.particles.rectangle_aspect_ratio",
            positive=True,
        ),
    )

    medium_data = _fields(
        root["medium"],
        f"{prefix}.medium",
        required={
            "inside_cuvette_refractive_index",
            "cuvette_refractive_index",
            "outside_cuvette_refractive_index",
        },
    )
    medium = MediumParameters(
        *(
            _number(medium_data[name], f"{prefix}.medium.{name}", positive=True)
            for name in (
                "inside_cuvette_refractive_index",
                "cuvette_refractive_index",
                "outside_cuvette_refractive_index",
            )
        )
    )

    class_data = _fields(
        root.get("particle_classes", {}),
        f"{prefix}.particle_classes",
        optional={
            "split_type",
            "min_diameter_nm",
            "max_diameter_nm",
            "mie_fraunhofer_boundary_nm",
            "mie_class_size_nm",
            "fraunhofer_log_type",
            "fraunhofer_log_percent",
        },
    )
    classes = ParticleClasses(
        split_type=_integer(
            class_data.get("split_type", 1),
            f"{prefix}.particle_classes.split_type",
        ),
        min_diameter_nm=_number(
            class_data.get("min_diameter_nm", 100.0),
            f"{prefix}.particle_classes.min_diameter_nm",
            positive=True,
        ),
        max_diameter_nm=_number(
            class_data.get("max_diameter_nm", 3_500_000.0),
            f"{prefix}.particle_classes.max_diameter_nm",
            positive=True,
        ),
        mie_fraunhofer_boundary_nm=_number(
            class_data.get("mie_fraunhofer_boundary_nm", 10_000.0),
            f"{prefix}.particle_classes.mie_fraunhofer_boundary_nm",
            positive=True,
        ),
        mie_class_size_nm=_number(
            class_data.get("mie_class_size_nm", 300.0),
            f"{prefix}.particle_classes.mie_class_size_nm",
            positive=True,
        ),
        fraunhofer_log_type=_integer(
            class_data.get("fraunhofer_log_type", 1),
            f"{prefix}.particle_classes.fraunhofer_log_type",
        ),
        fraunhofer_log_percent=_number(
            class_data.get("fraunhofer_log_percent", 7.5),
            f"{prefix}.particle_classes.fraunhofer_log_percent",
            positive=True,
        ),
    )
    if not (
        classes.min_diameter_nm
        < classes.mie_fraunhofer_boundary_nm
        < classes.max_diameter_nm
    ):
        raise ParametersError(
            f"{prefix}.particle_classes: требуется "
            "min_diameter_nm < mie_fraunhofer_boundary_nm < max_diameter_nm"
        )
    if classes.split_type != 1:
        raise ParametersError(f"{prefix}.particle_classes.split_type: поддерживается только 1")
    if classes.fraunhofer_log_type not in (1, 2):
        raise ParametersError(
            f"{prefix}.particle_classes.fraunhofer_log_type: допустимо 1 или 2"
        )

    return DarlParameters(
        version,
        detectors,
        particles,
        medium,
        laser,
        signal,
        classes,
    )


def load_measurement_parameters(
    source_file: str | Path | MeasurementParametersFile,
) -> MeasurementParameters:
    """Load an optional measurement-level expected distribution file."""
    source = Path(
        source_file.path
        if isinstance(source_file, MeasurementParametersFile)
        else source_file
    )
    root = _fields(
        _load_yaml(source),
        source.name,
        required={"schema_version", "expected_distribution"},
    )
    version = _schema(root, source.name)
    path = f"{source.name}.expected_distribution"
    data = _fields(
        root["expected_distribution"],
        path,
        required={"type", "mean_nm", "sigma_nm", "particle_count"},
    )
    distribution = ExpectedDistribution(
        type=_choice(data["type"], f"{path}.type", {"gaussian"}),
        mean_nm=_number(data["mean_nm"], f"{path}.mean_nm", positive=True),
        sigma_nm=_number(data["sigma_nm"], f"{path}.sigma_nm", positive=True),
        particle_count=_number(
            data["particle_count"],
            f"{path}.particle_count",
            positive=True,
        ),
    )
    return MeasurementParameters(version, distribution)


@dataclass(frozen=True)
class RestoreDetector:
    use_background: bool = True


@dataclass(frozen=True)
class RestoreDetectors:
    camera: RestoreDetector | None = field(
        default=None, metadata={YAML_OMIT_NONE: True}
    )
    line_sensor: RestoreDetector | None = field(
        default=None, metadata={YAML_OMIT_NONE: True}
    )

    def names(self) -> frozenset[str]:
        return frozenset(
            name
            for name in ("camera", "line_sensor")
            if getattr(self, name) is not None
        )


@dataclass(frozen=True)
class SolverParameters:
    type: str = "tikhonov_nnls"
    regularization_order: int = 1
    regularization_alpha: str | float = "best"
    use_chahine: bool = True
    use_concentration_correction: bool = True


@dataclass(frozen=True)
class ClassSliceParameters:
    drop_first: int = 10
    drop_last: int | None = None


@dataclass(frozen=True)
class RestoreParameters:
    schema_version: int
    detectors: RestoreDetectors
    solver: SolverParameters = field(default_factory=SolverParameters)
    class_slice: ClassSliceParameters = field(default_factory=ClassSliceParameters)


def load_restore_parameters(path: str | Path) -> RestoreParameters:
    source = Path(path)
    root = _fields(
        _load_yaml(source),
        source.name,
        required={"schema_version", "detectors"},
        optional={"solver", "class_slice"},
    )
    version = _schema(root, source.name)
    prefix = source.name
    detector_data = _fields(
        root["detectors"],
        f"{prefix}.detectors",
        optional={"camera", "line_sensor"},
    )

    def restore_detector(name: str) -> RestoreDetector | None:
        if name not in detector_data:
            return None
        data = _fields(
            detector_data[name],
            f"{prefix}.detectors.{name}",
            optional={"use_background"},
        )
        return RestoreDetector(
            _boolean(
                data.get("use_background", True),
                f"{prefix}.detectors.{name}.use_background",
            )
        )

    detectors = RestoreDetectors(
        restore_detector("camera"),
        restore_detector("line_sensor"),
    )
    if not detectors.names():
        raise ParametersError(f"{prefix}.detectors: должен быть указан хотя бы один детектор")

    solver_data = _fields(
        root.get("solver", {}),
        f"{prefix}.solver",
        optional={
            "type",
            "regularization_order",
            "regularization_alpha",
            "use_chahine",
            "use_concentration_correction",
        },
    )
    alpha_value = solver_data.get("regularization_alpha", "best")
    if isinstance(alpha_value, str):
        alpha: str | float = _choice(
            alpha_value,
            f"{prefix}.solver.regularization_alpha",
            {"best"},
        )
    else:
        alpha = _number(
            alpha_value,
            f"{prefix}.solver.regularization_alpha",
            positive=True,
        )
    solver = SolverParameters(
        type=_choice(
            solver_data.get("type", "tikhonov_nnls"),
            f"{prefix}.solver.type",
            {"tikhonov_nnls"},
        ),
        regularization_order=_integer(
            solver_data.get("regularization_order", 1),
            f"{prefix}.solver.regularization_order",
        ),
        regularization_alpha=alpha,
        use_chahine=_boolean(
            solver_data.get("use_chahine", True),
            f"{prefix}.solver.use_chahine",
        ),
        use_concentration_correction=_boolean(
            solver_data.get("use_concentration_correction", True),
            f"{prefix}.solver.use_concentration_correction",
        ),
    )
    if solver.regularization_order not in (0, 1, 2):
        raise ParametersError(
            f"{prefix}.solver.regularization_order: допустимо 0, 1 или 2"
        )

    slice_data = _fields(
        root.get("class_slice", {}),
        f"{prefix}.class_slice",
        optional={"drop_first", "drop_last"},
    )
    drop_last_value = slice_data.get("drop_last")
    drop_last = (
        None
        if drop_last_value is None
        else _integer(drop_last_value, f"{prefix}.class_slice.drop_last")
    )
    class_slice = ClassSliceParameters(
        drop_first=_integer(
            slice_data.get("drop_first", 10),
            f"{prefix}.class_slice.drop_first",
        ),
        drop_last=drop_last,
    )
    if class_slice.drop_first < 0 or (
        class_slice.drop_last is not None and class_slice.drop_last < 0
    ):
        raise ParametersError(f"{prefix}.class_slice: значения не могут быть отрицательными")
    return RestoreParameters(version, detectors, solver, class_slice)


@dataclass(frozen=True)
class CalibrationStageParameters:
    """Parameters consumed by calibration, without any DARL dependency."""

    general: GeneralParameters
    calibration: CalibrationParameters

    @classmethod
    def load(cls, experiment: Experiment) -> "CalibrationStageParameters":
        result = cls(
            load_general_parameters(experiment.general_parameters_file),
            load_calibration_parameters(experiment.calibration_parameters_file),
        )
        result.validate_consistency()
        return result

    def validate_consistency(self) -> None:
        expected = self.general.detectors.names()
        actual = frozenset(
            name for name in ("camera", "line_sensor")
            if getattr(self.calibration, name) is not None
        )
        if actual != expected:
            raise ParametersError(
                "calibration.yaml: набор секций детекторов должен совпадать "
                f"с general.yaml; ожидалось {sorted(expected)}, "
                f"получено {sorted(actual)}"
            )

    def effective_parameters(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "general": to_plain_data(self.general),
            "calibration": to_plain_data(self.calibration),
        }


@dataclass(frozen=True)
class ExperimentParameters:
    general: GeneralParameters
    calibration: CalibrationParameters
    darl: DarlParameters

    @classmethod
    def load(cls, experiment: Experiment) -> "ExperimentParameters":
        calibration_stage = CalibrationStageParameters.load(experiment)
        result = cls(
            calibration_stage.general,
            calibration_stage.calibration,
            load_darl_parameters(experiment.darl_parameters_file),
        )
        result.validate_consistency()
        return result

    def validate_consistency(self) -> None:
        expected = self.general.detectors.names()
        calibration = frozenset(
            name
            for name in ("camera", "line_sensor")
            if getattr(self.calibration, name) is not None
        )
        if calibration != expected:
            raise ParametersError(
                "calibration.yaml: набор секций детекторов должен совпадать "
                f"с general.yaml; ожидалось {sorted(expected)}, "
                f"получено {sorted(calibration)}"
            )
        modeled = self.darl.detectors.names()
        if modeled != expected:
            raise ParametersError(
                "darl.yaml: набор секций детекторов должен совпадать "
                f"с general.yaml; ожидалось {sorted(expected)}, "
                f"получено {sorted(modeled)}"
            )

    def load_restore_profile(self, profile: RestoreProfile) -> RestoreParameters:
        result = load_restore_parameters(profile.path)
        extra = result.detectors.names() - self.general.detectors.names()
        if extra:
            raise ParametersError(
                f"{profile.path.name}: профиль использует отсутствующие "
                f"детекторы: {', '.join(sorted(extra))}"
            )
        return result

    def effective_calibration(self) -> dict[str, Any]:
        return CalibrationStageParameters(
            self.general, self.calibration
        ).effective_parameters()

    def load_measurement(
        self,
        parameters_file: MeasurementParametersFile,
    ) -> MeasurementParameters:
        return load_measurement_parameters(parameters_file)

    def effective_matrix(
        self,
        calibration_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "general": to_plain_data(self.general),
            "darl": to_plain_data(self.darl),
            "calibration_result": to_plain_data(calibration_result),
        }

    def effective_quality_control(
        self,
        matrix_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "general": to_plain_data(self.general),
            "darl": to_plain_data(self.darl),
            "matrix_result": to_plain_data(matrix_result),
            "quality_control": to_plain_data(DarlQualityControl()),
        }

    def effective_expected_signal(
        self,
        parameters_file: MeasurementParametersFile,
        *,
        matrix_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        measurement = self.load_measurement(parameters_file)
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment": {
                "measurement": parameters_file.measurement_name,
            },
            "general": to_plain_data(self.general),
            "darl": to_plain_data(self.darl),
            "measurement": to_plain_data(measurement),
            "matrix_result": to_plain_data(matrix_result),
        }

    def effective_restore(
        self,
        profile: RestoreProfile,
        *,
        measurement_name: str,
        measurement_inputs: Mapping[str, Any],
        calibration_result: Mapping[str, Any],
        matrix_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        restore = self.load_restore_profile(profile)
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment": {
                "measurement": measurement_name,
                "restore_profile": profile.name,
            },
            "general": to_plain_data(self.general),
            "restore": to_plain_data(restore),
            "measurement_inputs": to_plain_data(measurement_inputs),
            "calibration_result": to_plain_data(calibration_result),
            "matrix_result": to_plain_data(matrix_result),
        }


def write_used_parameters(path: str | Path, parameters: Any) -> None:
    """Write a complete effective parameter mapping as stable safe YAML."""
    destination = Path(path)
    try:
        dump_yaml(destination, parameters, overwrite=False)
    except YamlError as error:
        if destination.exists():
            raise ParametersError(
                "Файл фактически использованных параметров уже существует: "
                f"{destination}"
            ) from error
        raise ParametersError(str(error)) from error
