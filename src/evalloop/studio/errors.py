"""Studio-specific errors. CLI maps these to a red message + exit 1."""


class StudioError(ValueError):
    """Invalid studio input, missing entity, or a failed job/run."""
