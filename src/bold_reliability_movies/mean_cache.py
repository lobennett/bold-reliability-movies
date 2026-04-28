"""On-disk cache for per-run mean BOLD images.

Cache key: source NIfTI path + mtime. Cache file is `<bold>.mean.nii.gz`
next to the source by default, or under `cache_dir` if supplied. Atomic
writes (temp file + rename). Cache write failures are non-fatal — the
caller still gets the in-memory mean.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import nibabel as nib
from nilearn.image import mean_img

log = logging.getLogger(__name__)


def _cache_path_for(source: Path, cache_dir: Path | None) -> Path:
    if cache_dir is None:
        return source.with_name(source.name.replace(".nii.gz", "") + ".mean.nii.gz")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (source.name.replace(".nii.gz", "") + ".mean.nii.gz")


def _cache_is_fresh(cache: Path, source: Path) -> bool:
    if not cache.exists():
        return False
    return cache.stat().st_mtime >= source.stat().st_mtime


def _compute_from_disk(source: Path) -> nib.Nifti1Image:
    """Read a 4D BOLD NIfTI and return its temporal mean as a 3D image."""
    img: nib.Nifti1Image = mean_img(str(source), copy_header=True)
    return img


def _atomic_write(img: nib.Nifti1Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        suffix=".nii.gz", prefix=".tmp.", dir=str(dest.parent)
    )
    import os as _os

    _os.close(fd)
    tmp = Path(tmp_name)
    try:
        nib.save(img, str(tmp))
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def compute_mean(
    source: Path,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> nib.Nifti1Image:
    """Return the temporal mean of a 4D BOLD NIfTI, caching to disk by default."""
    source = Path(source)
    if not use_cache:
        return _compute_from_disk(source)

    cache = _cache_path_for(source, cache_dir)
    if _cache_is_fresh(cache, source):
        return nib.load(str(cache))  # type: ignore[return-value]

    img = _compute_from_disk(source)
    try:
        _atomic_write(img, cache)
    except OSError as exc:
        log.warning(
            "mean cache write failed for %s: %s — continuing in memory", source, exc
        )
    return img
