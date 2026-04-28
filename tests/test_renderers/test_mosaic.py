from __future__ import annotations

import nibabel as nib
import numpy as np

from bold_reliability_movies.renderers.mosaic import MosaicRenderer


def _mean_img(shape=(16, 16, 8)) -> nib.Nifti1Image:
    rng = np.random.default_rng(0)
    return nib.Nifti1Image(rng.normal(100, 5, size=shape).astype(np.float32), np.eye(4))


def test_mosaic_returns_uint8_rgb() -> None:
    r = MosaicRenderer()
    out = r(_mean_img(), "label")
    assert out.dtype == np.uint8
    assert out.ndim == 3
    assert out.shape[2] == 3


def test_mosaic_shape_constant_across_inputs() -> None:
    r = MosaicRenderer()
    a = r(_mean_img((16, 16, 8)), "a")
    b = r(_mean_img((16, 16, 12)), "b")
    assert a.shape == b.shape


def test_mosaic_deterministic() -> None:
    r = MosaicRenderer()
    img = _mean_img()
    a = r(img, "label")
    b = r(img, "label")
    assert np.array_equal(a, b)


def test_mosaic_label_changes_pixels() -> None:
    r = MosaicRenderer()
    img = _mean_img()
    a = r(img, "alpha")
    b = r(img, "beta")
    assert not np.array_equal(a, b)


def test_mosaic_custom_grid() -> None:
    r = MosaicRenderer(n_rows=2, n_cols=3)
    out = r(_mean_img(), "x")
    assert out.shape[2] == 3
