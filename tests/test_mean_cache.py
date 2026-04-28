from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import pytest

from bold_reliability_movies.mean_cache import _cache_path_for, compute_mean


def test_compute_mean_returns_3d_image(tiny_bold: Path) -> None:
    img = compute_mean(tiny_bold)
    assert img.ndim == 3
    assert img.shape == (8, 8, 4)


def test_compute_mean_writes_cache_file(tiny_bold: Path) -> None:
    compute_mean(tiny_bold)
    cache = _cache_path_for(tiny_bold, cache_dir=None)
    assert cache.exists()


def test_compute_mean_uses_cache_on_second_call(tiny_bold: Path, monkeypatch) -> None:
    compute_mean(tiny_bold)
    from bold_reliability_movies import mean_cache

    called: list[bool] = []
    real_mean_img = mean_cache._compute_from_disk

    def spy(path: Path) -> nib.Nifti1Image:
        called.append(True)
        return real_mean_img(path)

    monkeypatch.setattr(mean_cache, "_compute_from_disk", spy)
    compute_mean(tiny_bold)
    assert called == []  # cache hit, no recompute


def test_compute_mean_invalidates_when_source_newer(tiny_bold: Path) -> None:
    compute_mean(tiny_bold)
    cache = _cache_path_for(tiny_bold, cache_dir=None)
    assert cache.exists()
    # Bump source mtime past cache mtime
    new_mtime = cache.stat().st_mtime + 10
    os.utime(tiny_bold, (new_mtime, new_mtime))
    cache_mtime_before = cache.stat().st_mtime
    compute_mean(tiny_bold)
    assert cache.stat().st_mtime > cache_mtime_before


def test_compute_mean_with_custom_cache_dir(tiny_bold: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    img = compute_mean(tiny_bold, cache_dir=cache_dir)
    assert img.shape == (8, 8, 4)
    cache = _cache_path_for(tiny_bold, cache_dir=cache_dir)
    assert cache.exists()
    assert cache.parent == cache_dir


def test_compute_mean_no_cache_argument(tiny_bold: Path) -> None:
    img = compute_mean(tiny_bold, use_cache=False)
    assert img.shape == (8, 8, 4)
    cache = _cache_path_for(tiny_bold, cache_dir=None)
    assert not cache.exists()


def test_compute_mean_corrupt_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.nii.gz"
    bad.write_bytes(b"not a nifti")
    with pytest.raises(nib.filebasedimages.ImageFileError):
        compute_mean(bad)
