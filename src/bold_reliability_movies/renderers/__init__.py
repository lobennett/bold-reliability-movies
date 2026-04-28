"""In-tree renderer dispatch table.

Adding a CLI-visible renderer:
  1. Implement a callable matching the Renderer Protocol.
  2. Register it in REGISTRY below with a string name.

Library users can pass their own Renderer to make_video() without touching
this table.
"""

from __future__ import annotations

from bold_reliability_movies.errors import UnknownRendererError
from bold_reliability_movies.renderers.mosaic import MosaicRenderer
from bold_reliability_movies.renderers.triplet import TripletRenderer
from bold_reliability_movies.types import Renderer

REGISTRY: dict[str, Renderer] = {
    "mosaic": MosaicRenderer(),
    "triplet": TripletRenderer(),
}


def get_renderer(name: str) -> Renderer:
    if name not in REGISTRY:
        raise UnknownRendererError(name=name, available=list_renderers())
    return REGISTRY[name]


def list_renderers() -> list[str]:
    return sorted(REGISTRY.keys())


__all__ = ["REGISTRY", "get_renderer", "list_renderers", "MosaicRenderer", "TripletRenderer"]
