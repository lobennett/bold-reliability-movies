from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

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
    r_default = MosaicRenderer()           # 5×5 grid
    r_custom = MosaicRenderer(n_rows=2, n_cols=3)
    img = _mean_img()
    out_default = r_default(img, "x")
    out_custom = r_custom(img, "x")
    # Same fig_size + dpi → same H×W; constructor accepts the custom grid.
    assert out_default.shape == out_custom.shape
    assert out_custom.dtype == np.uint8
    # Custom grid lays out fewer panels — the rendered pixels MUST differ.
    assert not np.array_equal(out_default, out_custom)


def test_mosaic_rejects_4d_input() -> None:
    rng = np.random.default_rng(0)
    img4d = nib.Nifti1Image(
        rng.normal(100, 5, size=(8, 8, 4, 5)).astype(np.float32),
        np.eye(4),
    )
    with pytest.raises(ValueError) as ei:
        MosaicRenderer()(img4d, "label")
    assert "3D" in str(ei.value)
