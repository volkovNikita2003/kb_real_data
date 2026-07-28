"""Project-specific exceptions."""


class ExperimentStructureError(ValueError):
    """The experiment directory does not follow the required structure."""


class ParametersError(ValueError):
    """An input parameter file is missing, malformed, or inconsistent."""


class OutputError(RuntimeError):
    """A result directory cannot be safely created or replaced."""
