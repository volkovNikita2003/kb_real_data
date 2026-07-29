"""Description of artifacts produced by one legacy DARL calculation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from errors import DarlError
from schema_validation import SchemaValidator
from yaml_support import YamlError, dump_yaml, load_yaml


RESULT_SCHEMA_VERSION = 1


class DarlResultError(DarlError):
    """The DARL result is incomplete, malformed, or cannot be stored."""


_VALIDATOR = SchemaValidator(DarlResultError)


@dataclass(frozen=True)
class DarlDistributionResult:
    name: str
    modeled_signal: str
    artifacts: tuple[str, ...]


@dataclass(frozen=True)
class DarlResult:
    schema_version: int
    legacy_config_name: str
    matrix_file: str
    particle_classes_file: str
    detector_bin_files: tuple[str, ...]
    background_signal_files: tuple[str, ...]
    distributions: tuple[DarlDistributionResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "legacy_config_name": self.legacy_config_name,
            "matrix_file": self.matrix_file,
            "particle_classes_file": self.particle_classes_file,
            "detector_bin_files": list(self.detector_bin_files),
            "background_signal_files": list(self.background_signal_files),
            "distributions": {
                item.name: {
                    "modeled_signal": item.modeled_signal,
                    "artifacts": list(item.artifacts),
                }
                for item in self.distributions
            },
        }


def collect_darl_result(
    directory: str | Path,
    *,
    legacy_config_name: str,
    detector_names: Iterable[str],
    distribution_names: tuple[str, ...],
) -> DarlResult:
    """Validate mandatory legacy artifacts and describe all generated files."""
    root = Path(directory)
    matrices = sorted(path.name for path in root.glob("matrix-*.npz"))
    if len(matrices) != 1:
        raise DarlResultError(
            f"Ожидался один файл аппаратной матрицы, найдено {len(matrices)}"
        )
    particle_classes = sorted(
        path.name for path in root.glob("particle_classes_lasser_*.txt")
    )
    if len(particle_classes) != 1:
        raise DarlResultError(
            "Ожидался один файл классов частиц, найдено "
            f"{len(particle_classes)}"
        )
    detector_files = {
        "camera": "bins_front_detector.txt",
        "line_sensor": "FrontDetectorLogLine_detector.txt",
    }
    detectors = frozenset(detector_names)
    unknown_detectors = detectors - detector_files.keys()
    if unknown_detectors:
        raise DarlResultError(
            "Неизвестные детекторы при проверке результата DARL: "
            + ", ".join(sorted(unknown_detectors))
        )
    expected_bin_files = {
        detector_files[name] for name in detectors
    }
    actual_bin_files = {
        path.name for path in root.iterdir()
        if path.is_file() and (
            path.name.startswith("bins_")
            or path.name.endswith("_detector.txt")
        )
    }
    missing_bin_files = expected_bin_files - actual_bin_files
    unexpected_bin_files = actual_bin_files - expected_bin_files
    if missing_bin_files or unexpected_bin_files:
        details: list[str] = []
        if missing_bin_files:
            details.append(
                "отсутствуют: " + ", ".join(sorted(missing_bin_files))
            )
        if unexpected_bin_files:
            details.append(
                "не ожидались: " + ", ".join(sorted(unexpected_bin_files))
            )
        raise DarlResultError(
            "Файлы бинов не соответствуют включённым детекторам ("
            + "; ".join(details)
            + ")"
        )
    bin_files = tuple(sorted(actual_bin_files))
    background_files = tuple(sorted(
        path.name for path in root.glob("background_signal_laser_*.txt")
    ))
    if not background_files:
        raise DarlResultError("Не найден ни один файл фонового сигнала")

    distributions: list[DarlDistributionResult] = []
    for name in distribution_names:
        distribution_dir = root / name
        modeled = distribution_dir / "modeled_signal.txt"
        if not modeled.is_file():
            raise DarlResultError(
                f"Не найден смоделированный сигнал распределения {name!r}: {modeled}"
            )
        artifacts = tuple(sorted(
            str(path.relative_to(root))
            for path in distribution_dir.rglob("*") if path.is_file()
        ))
        distributions.append(DarlDistributionResult(
            name=name,
            modeled_signal=str(modeled.relative_to(root)),
            artifacts=artifacts,
        ))
    return DarlResult(
        schema_version=RESULT_SCHEMA_VERSION,
        legacy_config_name=legacy_config_name,
        matrix_file=matrices[0],
        particle_classes_file=particle_classes[0],
        detector_bin_files=bin_files,
        background_signal_files=background_files,
        distributions=tuple(distributions),
    )


def save_darl_result(result: DarlResult, path: str | Path) -> None:
    if not isinstance(result, DarlResult):
        raise DarlResultError("result: ожидался объект DarlResult")
    try:
        dump_yaml(path, result.to_dict())
    except YamlError as error:
        raise DarlResultError(str(error)) from error


def load_darl_result(path: str | Path) -> DarlResult:
    """Load a result manifest written by :func:`save_darl_result`."""
    try:
        value = load_yaml(path)
    except YamlError as error:
        raise DarlResultError(str(error)) from error
    fields = _VALIDATOR.fields(
        value,
        "result.yaml",
        required={
            "schema_version", "legacy_config_name", "matrix_file",
            "particle_classes_file",
            "detector_bin_files", "background_signal_files", "distributions",
        },
    )
    _VALIDATOR.version(
        fields["schema_version"], "schema_version", supported=RESULT_SCHEMA_VERSION
    )
    for name in (
        "legacy_config_name", "matrix_file", "particle_classes_file",
    ):
        if not isinstance(fields[name], str) or not fields[name]:
            raise DarlResultError(f"{name}: ожидалась непустая строка")

    def strings(value: Any, location: str) -> tuple[str, ...]:
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise DarlResultError(f"{location}: ожидался список непустых строк")
        return tuple(value)

    detector_bins = strings(fields["detector_bin_files"], "detector_bin_files")
    if not detector_bins:
        raise DarlResultError(
            "detector_bin_files: должен быть указан хотя бы один файл"
        )
    allowed_detector_bins = {
        "bins_front_detector.txt",
        "FrontDetectorLogLine_detector.txt",
    }
    unknown_detector_bins = set(detector_bins) - allowed_detector_bins
    if unknown_detector_bins:
        raise DarlResultError(
            "detector_bin_files: неизвестные файлы: "
            + ", ".join(sorted(unknown_detector_bins))
        )
    if len(detector_bins) != len(set(detector_bins)):
        raise DarlResultError(
            "detector_bin_files: имена файлов не должны повторяться"
        )
    backgrounds = strings(
        fields["background_signal_files"], "background_signal_files"
    )
    if not backgrounds:
        raise DarlResultError(
            "background_signal_files: должен быть указан хотя бы один файл"
        )
    raw_distributions = fields["distributions"]
    if not isinstance(raw_distributions, dict):
        raise DarlResultError("distributions: ожидалось отображение")
    distributions: list[DarlDistributionResult] = []
    for name, item in raw_distributions.items():
        if not isinstance(name, str) or not name:
            raise DarlResultError("distributions: имя должно быть непустой строкой")
        data = _VALIDATOR.fields(
            item, f"distributions.{name}", required={"modeled_signal", "artifacts"}
        )
        modeled = data["modeled_signal"]
        if not isinstance(modeled, str) or not modeled:
            raise DarlResultError(
                f"distributions.{name}.modeled_signal: ожидалась непустая строка"
            )
        distributions.append(DarlDistributionResult(
            name, modeled,
            strings(data["artifacts"], f"distributions.{name}.artifacts"),
        ))
    return DarlResult(
        fields["schema_version"], fields["legacy_config_name"],
        fields["matrix_file"], fields["particle_classes_file"],
        detector_bins, backgrounds, tuple(distributions),
    )
