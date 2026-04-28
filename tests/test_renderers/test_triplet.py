from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from bold_reliability_movies.renderers.triplet import TripletRenderer


def _mean_img(shape=(16, 16, 8)) -> nib.Nifti1Image:
    rng = np.random.default_rng(1)
    return nib.Nifti1Image(rng.normal(100, 5, size=shape).astype(np.float32), np.eye(4))


def test_triplet_returns_uint8_rgb() -> None:
    out = TripletRenderer()(_mean_img(), "a")
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3


def test_triplet_shape_constant() -> None:
    r = TripletRenderer()
    a = r(_mean_img((16, 16, 8)), "a")
    b = r(_mean_img((20, 20, 12)), "b")
    assert a.shape == b.shape


def test_triplet_deterministic() -> None:
    r = TripletRenderer()
    img = _mean_img()
    assert np.array_equal(r(img, "x"), r(img, "x"))


def test_triplet_rejects_4d_input() -> None:
    rng = np.random.default_rng(1)
    img4d = nib.Nifti1Image(
        rng.normal(100, 5, size=(8, 8, 4, 5)).astype(np.float32),
        np.eye(4),
    )
    with pytest.raises(ValueError) as ei:
        TripletRenderer()(img4d, "label")
    assert "3D" in str(ei.value)
