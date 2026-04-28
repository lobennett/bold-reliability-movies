"""bold-reliability-movies — BIDS-aware BOLD reliability movies."""

from __future__ import annotations

__version__ = "0.1.0"

from bold_reliability_movies.discovery.fmriprep import FmriprepFrameSource
from bold_reliability_movies.discovery.manifest import ManifestFrameSource
from bold_reliability_movies.pipeline import make_video, make_videos
from bold_reliability_movies.renderers.mosaic import MosaicRenderer
from bold_reliability_movies.renderers.triplet import TripletRenderer
from bold_reliability_movies.types import Frame, FrameGroup, FrameSource, Renderer

__all__ = [
    "__version__",
    "Frame",
    "FrameGroup",
    "FrameSource",
    "Renderer",
    "MosaicRenderer",
    "TripletRenderer",
    "FmriprepFrameSource",
    "ManifestFrameSource",
    "make_video",
    "make_videos",
]
