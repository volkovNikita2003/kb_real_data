"""Paths and structural rules for an experiment directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from errors import ExperimentStructureError


SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$", re.ASCII)
REQUIRED_ROOT_DIRECTORIES = frozenset({"data", "input_parameters"})
OPTIONAL_ROOT_DIRECTORIES = frozenset({"output"})
ALLOWED_ROOT_DIRECTORIES = REQUIRED_ROOT_DIRECTORIES | OPTIONAL_ROOT_DIRECTORIES


def validate_safe_name(name: str, *, kind: str) -> str:
    """Return *name* if it is safe to use as a path component."""
    if not SAFE_NAME_PATTERN.fullmatch(name):
        raise ExperimentStructureError(
            f"Некорректное имя {kind}: {name!r}. "
            "Разрешены латинские буквы, цифры, '_' и '-'; "
            "первый символ должен быть буквой или цифрой."
        )
    return name


@dataclass(frozen=True)
class Measurement:
    """Paths belonging to one measurement in an experiment."""

    name: str
    path: Path

    @property
    def camera_dir(self) -> Path:
        return self.path / "cam"

    @property
    def camera_background_dir(self) -> Path:
        return self.path / "cam_back"

    @property
    def line_dir(self) -> Path:
        return self.path / "lin"

    @property
    def line_background_dir(self) -> Path:
        return self.path / "lin_back"


@dataclass(frozen=True)
class RestoreProfile:
    """A named restore profile stored as a YAML file."""

    name: str
    path: Path


@dataclass(frozen=True)
class Experiment:
    """A structurally valid experiment and its conventional paths."""

    path: Path

    @classmethod
    def open(cls, path: str | Path) -> "Experiment":
        """Open an experiment and validate its root-level structure."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise ExperimentStructureError(
                f"Директория эксперимента не найдена: {resolved}"
            )
        if not resolved.is_dir():
            raise ExperimentStructureError(
                f"Путь эксперимента не является директорией: {resolved}"
            )

        validate_safe_name(resolved.name, kind="эксперимента")
        entries = {entry.name: entry for entry in resolved.iterdir()}
        missing = sorted(REQUIRED_ROOT_DIRECTORIES - entries.keys())
        if missing:
            raise ExperimentStructureError(
                "В эксперименте отсутствуют обязательные директории: "
                + ", ".join(missing)
            )

        unexpected = sorted(entries.keys() - ALLOWED_ROOT_DIRECTORIES)
        if unexpected:
            raise ExperimentStructureError(
                "В корне эксперимента обнаружены посторонние элементы: "
                + ", ".join(unexpected)
            )

        not_directories = sorted(
            name
            for name in ALLOWED_ROOT_DIRECTORIES
            if name in entries and not entries[name].is_dir()
        )
        if not_directories:
            raise ExperimentStructureError(
                "Следующие обязательные элементы должны быть директориями: "
                + ", ".join(not_directories)
            )

        experiment = cls(path=resolved)
        experiment._validate_data_root()
        return experiment

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def data_dir(self) -> Path:
        return self.path / "data"

    @property
    def input_parameters_dir(self) -> Path:
        return self.path / "input_parameters"

    @property
    def output_dir(self) -> Path:
        return self.path / "output"

    @property
    def calibration_dir(self) -> Path:
        return self.data_dir / "calibration"

    @property
    def calibration_camera_dir(self) -> Path:
        return self.calibration_dir / "cam"

    @property
    def calibration_line_dir(self) -> Path:
        return self.calibration_dir / "lin"

    @property
    def general_parameters_file(self) -> Path:
        return self.input_parameters_dir / "general.yaml"

    @property
    def calibration_parameters_file(self) -> Path:
        return self.input_parameters_dir / "calibration.yaml"

    @property
    def darl_parameters_file(self) -> Path:
        return self.input_parameters_dir / "darl.yaml"

    @property
    def restore_profiles_dir(self) -> Path:
        return self.input_parameters_dir / "restore_profiles"

    @property
    def calibration_output_dir(self) -> Path:
        return self.output_dir / "calibration"

    @property
    def darl_output_dir(self) -> Path:
        return self.output_dir / "darl"

    @property
    def restore_output_root(self) -> Path:
        return self.output_dir / "restore"

    @property
    def archive_dir(self) -> Path:
        return self.output_dir / "archive"

    def measurement_names(self) -> tuple[str, ...]:
        """Return measurement names in deterministic order."""
        return tuple(measurement.name for measurement in self.measurements())

    def measurements(self) -> tuple[Measurement, ...]:
        """Return all measurement directories except ``calibration``."""
        measurements: list[Measurement] = []
        for entry in sorted(self.data_dir.iterdir(), key=lambda item: item.name):
            if entry.name == "calibration":
                continue
            if not entry.is_dir():
                raise ExperimentStructureError(
                    f"В data/ разрешены только директории: {entry}"
                )
            validate_safe_name(entry.name, kind="измерения")
            measurements.append(Measurement(name=entry.name, path=entry))
        return tuple(measurements)

    def measurement(self, name: str) -> Measurement:
        """Return one existing measurement by name."""
        validate_safe_name(name, kind="измерения")
        path = self.data_dir / name
        if name == "calibration" or not path.is_dir():
            raise ExperimentStructureError(
                f"Измерение {name!r} не найдено в эксперименте {self.name!r}"
            )
        return Measurement(name=name, path=path)

    def restore_profile_names(self) -> tuple[str, ...]:
        """Return restore profile names in deterministic order."""
        return tuple(profile.name for profile in self.restore_profiles())

    def restore_profiles(self) -> tuple[RestoreProfile, ...]:
        """Return YAML restore profiles; an absent profile directory means none."""
        directory = self.restore_profiles_dir
        if not directory.exists():
            return ()
        if not directory.is_dir():
            raise ExperimentStructureError(
                f"Путь профилей восстановления не является директорией: {directory}"
            )

        profiles: list[RestoreProfile] = []
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            if not entry.is_file() or entry.suffix.lower() != ".yaml":
                raise ExperimentStructureError(
                    "В restore_profiles/ разрешены только YAML-файлы: "
                    f"{entry.name}"
                )
            validate_safe_name(entry.stem, kind="профиля восстановления")
            profiles.append(RestoreProfile(name=entry.stem, path=entry))
        return tuple(profiles)

    def restore_profile(self, name: str) -> RestoreProfile:
        """Return one existing restore profile by name."""
        validate_safe_name(name, kind="профиля восстановления")
        profiles = {profile.name: profile for profile in self.restore_profiles()}
        if name not in profiles:
            raise ExperimentStructureError(
                f"Профиль восстановления {name!r} не найден "
                f"в эксперименте {self.name!r}"
            )
        return profiles[name]

    def restore_output_dir(
        self,
        profile_name: str,
        measurement_name: str,
    ) -> Path:
        """Build the output path for a profile/measurement pair."""
        profile = self.restore_profile(profile_name)
        measurement = self.measurement(measurement_name)
        return self.restore_output_root / profile.name / measurement.name

    def _validate_data_root(self) -> None:
        calibration = self.calibration_dir
        if not calibration.exists():
            raise ExperimentStructureError(
                f"Отсутствует директория калибровки: {calibration}"
            )
        if not calibration.is_dir():
            raise ExperimentStructureError(
                f"Путь калибровки не является директорией: {calibration}"
            )
        # Iteration validates that every other data entry is a safe measurement.
        self.measurements()
