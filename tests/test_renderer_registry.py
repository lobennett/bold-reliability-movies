from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from bold_reliability_movies.errors import UnknownRendererError
from bold_reliability_movies.renderers import REGISTRY, get_renderer, list_renderers


def test_registry_contains_mosaic_and_triplet() -> None:
    assert "mosaic" in REGISTRY
    assert "triplet" in REGISTRY


def test_get_renderer_returns_callable() -> None:
    r = get_renderer("mosaic")
    img = nib.Nifti1Image(np.zeros((8, 8, 4), dtype=np.float32), np.eye(4))
    out = r(img, "label")
    assert out.shape[2] == 3


def test_get_renderer_unknown_raises() -> None:
    with pytest.raises(UnknownRendererError) as ei:
        get_renderer("flatmap")
    assert "flatmap" in str(ei.value)
    assert "mosaic" in str(ei.value)


def test_list_renderers_sorted() -> None:
    names = list_renderers()
    assert names == sorted(names)
