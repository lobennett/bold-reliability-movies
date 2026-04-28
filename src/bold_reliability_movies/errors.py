"""Typed exceptions raised by bold-reliability-movies."""

from __future__ import annotations


class BrmError(Exception):
    """Base class for all package errors."""


class MissingDependency(BrmError):
    """A required external dependency (e.g. ffmpeg) is not available."""


class InconsistentShapesError(BrmError):
    """Two frames in one group have different rendered shapes."""

    def __init__(
        self,
        frame_index: int,
        previous_shape: tuple[int, ...],
        current_shape: tuple[int, ...],
        suggestion: str,
    ) -> None:
        self.frame_index = frame_index
        self.previous_shape = previous_shape
        self.current_shape = current_shape
        self.suggestion = suggestion
        super().__init__(
            f"Frame {frame_index} has shape {current_shape}, previous frames "
            f"had shape {previous_shape}; cannot mix shapes in one video. "
            f"Suggestion: {suggestion}."
        )


class EncodeError(BrmError):
    """ffmpeg encoding failed."""


class GroupRejectedError(BrmError):
    """A FrameGroup was rejected: too many dropped frames or too few usable."""


class EmptyDiscoveryError(BrmError):
    """A FrameSource produced zero FrameGroups."""


class UnknownRendererError(BrmError):
    """A renderer name was requested that is not in the in-tree dispatch table."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        super().__init__(
            f"Unknown renderer {name!r}. Available: {', '.join(available)}."
        )
