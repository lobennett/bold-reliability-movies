"""Shared pytest fixtures: synthetic NIfTI volumes + fake fmriprep tree."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _make_bold(path: Path, shape: tuple[int, int, int, int] = (8, 8, 4, 5)) -> Path:
    """Write a synthetic 4D BOLD NIfTI with a deterministic random signal."""
    seed = int.from_bytes(str(path).encode(), "little") % (2**32)
    rng = np.random.default_rng(seed=seed)
    base = rng.normal(loc=100, scale=2, size=shape).astype(np.float32)
    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(base, affine), str(path))
    return path


@pytest.fixture
def make_bold(tmp_path: Path) -> Callable[..., Path]:
    """Factory: write a tiny BOLD NIfTI at a path under tmp_path."""

    def _factory(
        name: str = "bold.nii.gz", shape: tuple[int, int, int, int] = (8, 8, 4, 5)
    ) -> Path:
        out = tmp_path / name
        out.parent.mkdir(parents=True, exist_ok=True)
        return _make_bold(out, shape=shape)

    return _factory


@pytest.fixture
def tiny_bold(make_bold: Callable[..., Path]) -> Path:
    """One synthetic BOLD NIfTI with a default tiny shape (8, 8, 4, 5)."""
    return make_bold("tiny_bold.nii.gz")


@pytest.fixture
def fake_fmriprep_tree(tmp_path: Path) -> Path:
    """Build a minimal fmriprep-derivatives tree.

    Structure:
      <root>/
        sub-s01/ses-01/func/sub-s01_ses-01_task-rest_run-1_desc-preproc_bold.nii.gz
        sub-s01/ses-01/func/sub-s01_ses-01_task-rest_run-2_desc-preproc_bold.nii.gz
        sub-s01/ses-02/func/sub-s01_ses-02_task-stroop_run-1_desc-preproc_bold.nii.gz
        sub-s02/ses-01/func/sub-s02_ses-01_task-rest_run-1_desc-preproc_bold.nii.gz
    """
    root = tmp_path / "deriv"
    layout = [
        ("sub-s01", "ses-01", "task-rest", 1),
        ("sub-s01", "ses-01", "task-rest", 2),
        ("sub-s01", "ses-02", "task-stroop", 1),
        ("sub-s02", "ses-01", "task-rest", 1),
    ]
    for sub, ses, task, run in layout:
        d = root / sub / ses / "func"
        d.mkdir(parents=True, exist_ok=True)
        fname = f"{sub}_{ses}_{task}_run-{run}_desc-preproc_bold.nii.gz"
        _make_bold(d / fname)
    return root


@pytest.fixture
def stub_renderer() -> Callable[..., np.ndarray]:
    """A renderer returning a constant (16, 16, 3) frame; label hashed into a pixel."""

    def _renderer(mean_img: nib.Nifti1Image, label: str) -> np.ndarray:
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        frame[0, 0, 0] = int.from_bytes(label.encode()[:4], "little") % 256
        return frame

    return _renderer
