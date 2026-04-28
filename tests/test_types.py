from pathlib import Path

import numpy as np
import pytest

from bold_reliability_movies.types import (
    Frame,
    FrameGroup,
    FrameSource,
    Renderer,
)


def test_frame_is_frozen():
    f = Frame(path=Path("/tmp/x.nii.gz"), label="a", sort_key=(1,))
    with pytest.raises(AttributeError):  # FrozenInstanceError subclasses AttributeError
        f.label = "b"  # type: ignore[misc]


def test_frame_group_default_metadata():
    fg = FrameGroup(name="sub-s01", frames=[])
    assert fg.metadata == {}


def test_frame_group_holds_frames():
    f = Frame(path=Path("/tmp/x.nii.gz"), label="a", sort_key=(1,))
    fg = FrameGroup(name="sub-s01", frames=[f], metadata={"subject": "s01"})
    assert fg.frames[0].label == "a"
    assert fg.metadata["subject"] == "s01"


def test_renderer_protocol_accepts_callable():
    def renderer(mean_img, label):  # signature only
        return np.zeros((4, 4, 3), dtype=np.uint8)

    # runtime_checkable structural check: looks for __call__ attribute name,
    # NOT its signature. A function with the wrong arg count would also pass.
    assert isinstance(renderer, Renderer)


def test_frame_source_protocol_accepts_class_with_discover():
    class MySource:
        def discover(self) -> list[FrameGroup]:
            return []

    assert isinstance(MySource(), FrameSource)


def test_public_api_reexports():
    import bold_reliability_movies as brm

    assert hasattr(brm, "make_video")
    assert hasattr(brm, "make_videos")
    assert hasattr(brm, "Frame")
    assert hasattr(brm, "FrameGroup")
    assert hasattr(brm, "Renderer")
    assert hasattr(brm, "FrameSource")
    assert hasattr(brm, "MosaicRenderer")
    assert hasattr(brm, "TripletRenderer")
    assert hasattr(brm, "FmriprepFrameSource")
    assert hasattr(brm, "ManifestFrameSource")
    assert brm.__version__ == "0.1.0"
