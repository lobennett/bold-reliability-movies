"""Discovery layer: turn input sources into FrameGroups."""

from bold_reliability_movies.discovery.fmriprep import FmriprepFrameSource
from bold_reliability_movies.discovery.manifest import ManifestFrameSource

__all__ = ["FmriprepFrameSource", "ManifestFrameSource"]
