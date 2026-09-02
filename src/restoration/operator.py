"""Resolve calculated and imported forward operators for restoration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

from darl import DarlResult
from errors import RestorationError
from experiment import Experiment
from parameters import RestoreOperatorParameters
from schema_validation import SchemaValidator
from yaml_support import YamlError, load_yaml


MANIFEST_SCHEMA_VERSION = 1


class ForwardOperatorError(RestorationError):
    """The selected forward-operator bundle is invalid or incomplete."""


_VALIDATOR = SchemaValidator(ForwardOperatorError)
_DETECTORS = ("camera", "line_sensor")


@dataclass(frozen=True)
class ForwardOperator:
    source: str
    name: str
    detectors: tuple[str, ...]
    signal_value_type: str
    matrix_file: Path
    particle_classes_file: Path
    detector_bin_files: Mapping[str, Path]
    background_signal_files: tuple[Path, ...]
    manifest_file: Path | None = None
    expected_signals: Mapping[str, Path] | None = None

    def expected_signal(self, measurement: str) -> Path | None:
        return (self.expected_signals or {}).get(measurement)

    def to_dict(self, experiment: Experiment) -> dict[str, Any]:
        def relative(path: Path) -> str:
            return path.resolve().relative_to(experiment.path).as_posix()

        artifacts = {
            "matrix": self.matrix_file,
            "particle_classes": self.particle_classes_file,
            **{
                f"detector_bins.{name}": path
                for name, path in self.detector_bin_files.items()
            },
        }
        result: dict[str, Any] = {
            "source": self.source,
            "name": self.name,
            "detectors": list(self.detectors),
            "signal": {"value_type": self.signal_value_type},
            "matrix_file": relative(self.matrix_file),
            "particle_classes_file": relative(self.particle_classes_file),
            "detector_bin_files": {
                name: relative(path)
                for name, path in self.detector_bin_files.items()
            },
            "background_signal_files": [
                relative(path) for path in self.background_signal_files
            ],
            "sha256": {
                name: _sha256(path) for name, path in artifacts.items()
                if path.is_file()
            },
        }
        if self.manifest_file is not None:
            result["manifest_file"] = relative(self.manifest_file)
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, relative: str, *, location: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ForwardOperatorError(f"{location}: ожидался непустой путь")
    path = Path(relative)
    if path.is_absolute():
        raise ForwardOperatorError(f"{location}: абсолютные пути запрещены")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ForwardOperatorError(
            f"{location}: путь выходит за пределы разрешённого каталога"
        ) from error
    return resolved


def from_darl(experiment: Experiment, result: DarlResult) -> ForwardOperator:
    root = experiment.darl_output_dir.resolve()
    bins_by_detector: dict[str, Path] = {}
    expected_names = {
        "camera": "bins_front_detector.txt",
        "line_sensor": "FrontDetectorLogLine_detector.txt",
    }
    available = set(result.detector_bin_files)
    for detector in result.detectors:
        filename = expected_names[detector]
        if filename not in available:
            raise ForwardOperatorError(
                f"darl.result.detector_bin_files: отсутствует файл для {detector}"
            )
        bins_by_detector[detector] = _inside(
            root, filename, location=f"darl.detector_bins.{detector}"
        )
    return ForwardOperator(
        source="darl",
        name=result.legacy_config_name,
        detectors=result.detectors,
        signal_value_type=result.signal.value_type,
        matrix_file=_inside(root, result.matrix_file, location="darl.matrix_file"),
        particle_classes_file=_inside(
            root, result.particle_classes_file,
            location="darl.particle_classes_file",
        ),
        detector_bin_files=bins_by_detector,
        background_signal_files=tuple(
            _inside(root, path, location="darl.background_signal_files")
            for path in result.background_signal_files
        ),
        expected_signals={
            item.name: _inside(
                root, item.modeled_signal,
                location=f"darl.distributions.{item.name}.modeled_signal",
            )
            for item in result.distributions
        },
    )


def from_manifest(experiment: Experiment, manifest: str) -> ForwardOperator:
    manifest_path = _inside(
        experiment.path, manifest, location="operator.manifest"
    )
    try:
        manifest_path.relative_to(experiment.operators_dir.resolve())
    except ValueError as error:
        raise ForwardOperatorError(
            "operator.manifest: manifest должен находиться внутри "
            "input_artifacts/operators"
        ) from error
    try:
        value = load_yaml(manifest_path)
    except YamlError as error:
        raise ForwardOperatorError(str(error)) from error
    root = _VALIDATOR.fields(
        value,
        "operator-manifest",
        required={"schema_version", "detectors", "signal", "files"},
    )
    _VALIDATOR.version(
        root["schema_version"], "schema_version",
        supported=MANIFEST_SCHEMA_VERSION,
    )
    name = manifest_path.parent.name
    raw_detectors = root["detectors"]
    if not isinstance(raw_detectors, list) or not raw_detectors:
        raise ForwardOperatorError("detectors: ожидался непустой список")
    if any(item not in _DETECTORS for item in raw_detectors):
        raise ForwardOperatorError("detectors: допустимы camera и line_sensor")
    if len(raw_detectors) != len(set(raw_detectors)):
        raise ForwardOperatorError("detectors: имена не должны повторяться")
    detectors = tuple(item for item in _DETECTORS if item in raw_detectors)

    signal = _VALIDATOR.fields(
        root["signal"], "signal", required={"value_type"}
    )
    value_type = _VALIDATOR.choice(
        signal["value_type"], "signal.value_type", {"signal", "intensity"}
    )
    files = _VALIDATOR.fields(
        root["files"], "files",
        required={"matrix", "particle_classes", "detector_bins"},
        optional={"background_signals"},
    )
    bundle = manifest_path.parent.resolve()
    raw_bins = _VALIDATOR.fields(
        files["detector_bins"], "files.detector_bins",
        required=set(detectors),
    )
    backgrounds = files.get("background_signals", [])
    if not isinstance(backgrounds, list) or any(
        not isinstance(item, str) or not item for item in backgrounds
    ):
        raise ForwardOperatorError(
            "files.background_signals: ожидался список непустых путей"
        )
    return ForwardOperator(
        source="file",
        name=name,
        detectors=detectors,
        signal_value_type=value_type,
        matrix_file=_inside(bundle, files["matrix"], location="files.matrix"),
        particle_classes_file=_inside(
            bundle, files["particle_classes"],
            location="files.particle_classes",
        ),
        detector_bin_files={
            detector: _inside(
                bundle, raw_bins[detector],
                location=f"files.detector_bins.{detector}",
            )
            for detector in detectors
        },
        background_signal_files=tuple(
            _inside(bundle, item, location="files.background_signals")
            for item in backgrounds
        ),
        manifest_file=manifest_path,
    )


def resolve_operator(
    experiment: Experiment,
    parameters: RestoreOperatorParameters,
    *,
    darl: DarlResult | None,
) -> ForwardOperator:
    if parameters.source == "darl":
        if darl is None:
            raise ForwardOperatorError(
                "Для operator.source: darl отсутствует результат DARL"
            )
        return from_darl(experiment, darl)
    if parameters.source == "file" and parameters.manifest is not None:
        return from_manifest(experiment, parameters.manifest)
    raise ForwardOperatorError("Некорректный источник аппаратной матрицы")
