"""Preserved legacy restoration baseline and its configuration adapter."""

from restoration.legacy.config import (
    LegacyRestoreConfigArtifact,
    LegacyRestoreConfigError,
    LegacyRestoreSolverSettings,
    build_legacy_restore_config,
)

__all__ = [
    "LegacyRestoreConfigArtifact",
    "LegacyRestoreConfigError",
    "LegacyRestoreSolverSettings",
    "build_legacy_restore_config",
]
