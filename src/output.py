"""Transactional creation and versioning of experiment results."""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from errors import OutputError
from experiment import Experiment


Clock = Callable[[], datetime]


def archive_timestamp(moment: datetime | None = None) -> str:
    """Return a filesystem-safe, sortable timestamp with microseconds."""
    value = moment or datetime.now().astimezone()
    # The archive layout documented by the project deliberately contains no
    # timezone suffix.  Microseconds make collisions in normal use unlikely.
    return value.strftime("%Y-%m-%dT%H-%M-%S.%f")


class OutputDirectory(AbstractContextManager[Path]):
    """Build one output directory and publish it only after successful work.

    The context yields a temporary directory next to the final target.  On a
    clean exit it is renamed to *target*.  An existing target is rejected,
    unless ``force=True``; in that case it is first moved to the mirrored
    archive path returned by :meth:`Experiment.archive_path_for`.
    """

    def __init__(
        self,
        experiment: Experiment,
        target: str | Path,
        *,
        force: bool = False,
        clock: Clock | None = None,
    ) -> None:
        self.experiment = experiment
        self.target = Path(target).expanduser().resolve()
        self.force = force
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._temporary: Path | None = None

        # Besides validating containment, this rejects output/ itself and any
        # path below output/archive/.
        try:
            experiment.archive_path_for(self.target)
        except ValueError as error:
            raise OutputError(str(error)) from error

    def __enter__(self) -> Path:
        if self.target.exists() or self.target.is_symlink():
            if not self.force:
                raise OutputError(
                    f"Каталог результатов уже существует: {self.target}. "
                    "Для повторного расчёта используйте --force."
                )
            if self.target.is_symlink() or not self.target.is_dir():
                raise OutputError(
                    f"Путь результата должен быть обычной директорией: {self.target}"
                )

        try:
            self.target.parent.mkdir(parents=True, exist_ok=True)
            temporary = tempfile.mkdtemp(
                prefix=f".{self.target.name}.tmp-",
                dir=self.target.parent,
            )
        except OSError as error:
            raise OutputError(
                f"Не удалось подготовить каталог результатов {self.target}: {error}"
            ) from error
        self._temporary = Path(temporary)
        return self._temporary

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        temporary = self._temporary
        if temporary is None:
            return False
        if exc_type is not None:
            shutil.rmtree(temporary, ignore_errors=True)
            return False

        archived: Path | None = None
        try:
            if self.target.exists():
                archived = self._archive_existing()
            temporary.rename(self.target)
        except Exception as error:
            shutil.rmtree(temporary, ignore_errors=True)
            if archived is not None and archived.exists() and not self.target.exists():
                try:
                    archived.rename(self.target)
                except OSError:
                    pass
            if isinstance(error, OutputError):
                raise
            raise OutputError(
                f"Не удалось сохранить результаты в {self.target}: {error}"
            ) from error
        return False

    def _archive_existing(self) -> Path:
        if not self.force:
            # Covers another process creating the target while work was in
            # progress.
            raise OutputError(
                f"Каталог результатов уже существует: {self.target}. "
                "Для повторного расчёта используйте --force."
            )
        archive_root = self.experiment.archive_path_for(self.target)
        archived = archive_root / archive_timestamp(self._clock())
        if archived.exists():
            raise OutputError(f"Версия архива уже существует: {archived}")
        try:
            archive_root.mkdir(parents=True, exist_ok=True)
            self.target.rename(archived)
        except OSError as error:
            raise OutputError(
                f"Не удалось перенести старые результаты в архив {archived}: {error}"
            ) from error
        return archived


def prepare_output_directory(
    experiment: Experiment,
    target: str | Path,
    *,
    force: bool = False,
) -> OutputDirectory:
    """Return a transaction for safely producing one result directory."""
    return OutputDirectory(experiment, target, force=force)
