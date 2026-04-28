"""Public dataclasses and Protocols crossing layer boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class Frame:
    """One frame of an output video.

    `path` is a 4D BOLD NIfTI; the pipeline computes its mean before passing
    the resulting 3D image to a Renderer. `label` is overlaid on the frame.
    `sort_key` is opaque to the renderer; the discovery layer uses it to
    order frames within a group.
    """

    path: Path
    label: str
    sort_key: tuple[Any, ...]


@dataclass(frozen=True)
class FrameGroup:
    """One output video (one MP4)."""

    name: str
    frames: list[Frame]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Renderer(Protocol):
    """A callable turning a mean image into an RGB frame.

    Implementations MUST return the same (H, W) for every call within a
    single video; (H, W, 3) uint8 arrays only.
    """

    def __call__(self, mean_img: nib.Nifti1Image, label: str) -> np.ndarray:
        ...


@runtime_checkable
class FrameSource(Protocol):
    """An object that produces FrameGroups from some input (BIDS dir, TSV, ...)."""

    def discover(self) -> list[FrameGroup]:
        ...
