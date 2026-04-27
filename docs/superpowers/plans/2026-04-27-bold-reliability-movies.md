# bold-reliability-movies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a v0.1 PyPI package `bold-reliability-movies` that turns BIDS/fMRIPrep derivatives (or a TSV manifest, or a list of NIfTIs) into MP4 "reliability movies" — one frame per BOLD run, mean-image rendered, cycled at a configurable fps.

**Architecture:** Three layers — discovery (BIDS/manifest → `FrameGroup`s) → pipeline (compute mean, render, encode) → renderer (in-tree mosaic + triplet, Protocol for third-party). Plain frozen dataclasses cross layer boundaries. CLI is `argparse` with `bids`/`list`/`render` subcommands. MP4 written by piping raw RGB frames into a `ffmpeg` subprocess.

**Tech Stack:** Python 3.10+, `uv` for project management, `hatchling` build backend, `nibabel` + `nilearn` for NIfTI/mean computation, `matplotlib` for in-tree rendering, `subprocess` + system `ffmpeg` for encoding, `pytest` + `ruff` + `mypy --strict` for QA, GitHub Actions for CI.

**Working directory for all tasks:** `/home/users/logben/bold-reliability-movies`. The repo is already `git init`'d on branch `main` with the design spec committed.

**Sherlock note:** all Python commands must be `uv run python ...` (not system `python`). `uv` requires `module load uv` once per shell on Sherlock.

---

## File map

```
bold-reliability-movies/
├── pyproject.toml                      # Task 1
├── README.md                           # Task 1 (skeleton), Task 15 (full)
├── LICENSE                             # Task 1
├── CHANGELOG.md                        # Task 15
├── CONTRIBUTING.md                     # Task 15
├── .gitignore                          # Task 1
├── .github/workflows/ci.yml            # Task 16
├── docs/superpowers/specs/<spec>.md    # already exists
├── docs/superpowers/plans/<plan>.md    # this file
├── src/bold_reliability_movies/
│   ├── __init__.py                     # Task 14
│   ├── errors.py                       # Task 2
│   ├── types.py                        # Task 3
│   ├── mean_cache.py                   # Task 5
│   ├── encode.py                       # Task 6
│   ├── pipeline.py                     # Task 10
│   ├── cli.py                          # Task 13
│   ├── discovery/
│   │   ├── __init__.py                 # Task 11
│   │   ├── fmriprep.py                 # Task 11
│   │   └── manifest.py                 # Task 12
│   └── renderers/
│       ├── __init__.py                 # Task 9
│       ├── mosaic.py                   # Task 7
│       └── triplet.py                  # Task 8
└── tests/
    ├── conftest.py                     # Task 4
    ├── test_errors.py                  # Task 2
    ├── test_types.py                   # Task 3
    ├── test_mean_cache.py              # Task 5
    ├── test_encode.py                  # Task 6
    ├── test_renderers/
    │   ├── __init__.py                 # Task 7
    │   ├── test_mosaic.py              # Task 7
    │   └── test_triplet.py             # Task 8
    ├── test_renderer_registry.py       # Task 9
    ├── test_pipeline.py                # Task 10
    ├── test_discovery_fmriprep.py      # Task 11
    ├── test_discovery_manifest.py      # Task 12
    └── test_cli.py                     # Task 13
```

---

## Task 1: Scaffold the package

**Goal:** A working `pyproject.toml`, the `src/` and `tests/` skeleton, LICENSE, .gitignore, and a stub README. Verify the package installs and `import bold_reliability_movies` works.

**Files:**
- Create: `pyproject.toml`
- Create: `LICENSE`
- Create: `.gitignore`
- Create: `README.md` (skeleton)
- Create: `src/bold_reliability_movies/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1.1: Initialize project with uv**

```bash
cd /home/users/logben/bold-reliability-movies
module load uv
uv init --package --name bold-reliability-movies --lib bold_reliability_movies
```

This creates `pyproject.toml`, `src/bold_reliability_movies/__init__.py`, and `.python-version`. The `--package --lib` combo gives a src-layout library project.

- [ ] **Step 1.2: Replace generated `pyproject.toml` with the project's full version**

Overwrite `pyproject.toml` with:

```toml
[project]
name = "bold-reliability-movies"
version = "0.1.0"
description = "BIDS-aware BOLD reliability movies (volume-space mosaic, extensible)."
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
authors = [{ name = "Logan Bennett", email = "logben@stanford.edu" }]
dependencies = [
    "nibabel>=5.0",
    "nilearn>=0.10",
    "matplotlib>=3.7",
    "numpy>=1.24",
]
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
    "License :: OSI Approved :: MIT License",
]

[project.urls]
Homepage = "https://github.com/lobennett/bold-reliability-movies"
Source = "https://github.com/lobennett/bold-reliability-movies"

[project.scripts]
bold-reliability-movies = "bold_reliability_movies.cli:main"
brm = "bold_reliability_movies.cli:main"

[project.optional-dependencies]
dev = ["pytest>=7", "pytest-cov", "ruff", "mypy"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP"]

[tool.mypy]
strict = true
files = ["src/bold_reliability_movies"]

[[tool.mypy.overrides]]
module = "nibabel.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "nilearn.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "matplotlib.*"
ignore_missing_imports = true

[tool.pytest.ini_options]
markers = [
    "slow: tests that perform real ffmpeg encoding (skipped by default)",
    "integration: tests that require real fMRIPrep/FreeSurfer derivatives",
]
addopts = "-m 'not slow and not integration'"
```

- [ ] **Step 1.3: Sync deps**

```bash
uv sync --all-extras
```

Expected: creates `.venv/`, installs nibabel/nilearn/matplotlib/numpy + dev deps. No errors.

- [ ] **Step 1.4: Write `LICENSE` (MIT)**

```
MIT License

Copyright (c) 2026 Logan Bennett

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 1.5: Write `.gitignore`**

```
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.eggs/
dist/
build/
.venv/
.coverage
.coverage.*
htmlcov/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.python-version
*.mp4
*.mean.nii.gz
.DS_Store
```

- [ ] **Step 1.6: Write `README.md` (skeleton — full version in Task 15)**

```markdown
# bold-reliability-movies

BIDS-aware BOLD reliability movies for visual fMRI QC.

> Status: pre-release. Full README in v0.1.0 release.
```

- [ ] **Step 1.7: Empty `src/bold_reliability_movies/__init__.py` and `tests/__init__.py`**

```bash
: > src/bold_reliability_movies/__init__.py
: > tests/__init__.py
```

(`uv init` may have populated `__init__.py`; truncate it.)

- [ ] **Step 1.8: Verify install**

Run: `uv run python -c "import bold_reliability_movies; print(bold_reliability_movies.__name__)"`
Expected output: `bold_reliability_movies`

- [ ] **Step 1.9: Verify pytest discovery (no tests yet)**

Run: `uv run pytest -q`
Expected: `no tests ran`. No errors.

- [ ] **Step 1.10: Commit**

```bash
git add pyproject.toml LICENSE .gitignore README.md uv.lock src/ tests/ .python-version
git commit -m "chore: scaffold package with uv (pyproject, license, src layout)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Typed exceptions in `errors.py`

**Goal:** Define every exception class the package raises, with one test asserting each is a real exception class with a sensible message.

**Files:**
- Create: `src/bold_reliability_movies/errors.py`
- Test: `tests/test_errors.py`

- [ ] **Step 2.1: Write the failing test**

`tests/test_errors.py`:

```python
import pytest

from bold_reliability_movies.errors import (
    BrmError,
    MissingDependency,
    InconsistentShapesError,
    EncodeError,
    EmptyDiscoveryError,
    UnknownRendererError,
)


def test_all_inherit_from_brm_error():
    for cls in (
        MissingDependency,
        InconsistentShapesError,
        EncodeError,
        EmptyDiscoveryError,
        UnknownRendererError,
    ):
        assert issubclass(cls, BrmError)


def test_brm_error_inherits_from_exception():
    assert issubclass(BrmError, Exception)


def test_inconsistent_shapes_error_carries_shapes():
    err = InconsistentShapesError(
        frame_index=4,
        previous_shape=(320, 320, 3),
        current_shape=(360, 360, 3),
        suggestion="use --group-by subject+task",
    )
    msg = str(err)
    assert "frame 4" in msg.lower()
    assert "320" in msg
    assert "360" in msg
    assert "--group-by subject+task" in msg


def test_unknown_renderer_lists_available():
    err = UnknownRendererError(name="flatmap", available=["mosaic", "triplet"])
    msg = str(err)
    assert "flatmap" in msg
    assert "mosaic" in msg
    assert "triplet" in msg
```

- [ ] **Step 2.2: Run the test to verify it fails**

Run: `uv run pytest tests/test_errors.py -v`
Expected: `ImportError` / `ModuleNotFoundError` for `bold_reliability_movies.errors`.

- [ ] **Step 2.3: Implement `errors.py`**

```python
"""Typed exceptions raised by bold-reliability-movies."""

from __future__ import annotations


class BrmError(Exception):
    """Base class for all package errors."""


class MissingDependency(BrmError):
    """A required external dependency (e.g. ffmpeg) is not available."""


class InconsistentShapesError(BrmError):
    """Two frames in one group have different rendered shapes."""

    def __init__(
        self,
        frame_index: int,
        previous_shape: tuple[int, ...],
        current_shape: tuple[int, ...],
        suggestion: str,
    ) -> None:
        self.frame_index = frame_index
        self.previous_shape = previous_shape
        self.current_shape = current_shape
        self.suggestion = suggestion
        super().__init__(
            f"Frame {frame_index} has shape {current_shape}, previous frames "
            f"had shape {previous_shape}; cannot mix shapes in one video. "
            f"Suggestion: {suggestion}."
        )


class EncodeError(BrmError):
    """ffmpeg encoding failed."""


class EmptyDiscoveryError(BrmError):
    """A FrameSource produced zero FrameGroups."""


class UnknownRendererError(BrmError):
    """A renderer name was requested that is not in the in-tree dispatch table."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        super().__init__(
            f"Unknown renderer {name!r}. Available: {', '.join(available)}."
        )
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `uv run pytest tests/test_errors.py -v`
Expected: 4 passed.

- [ ] **Step 2.5: Commit**

```bash
git add src/bold_reliability_movies/errors.py tests/test_errors.py
git commit -m "feat(errors): typed exceptions with structured messages

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Core types in `types.py`

**Goal:** `Frame`, `FrameGroup` frozen dataclasses; `Renderer` and `FrameSource` runtime-checkable Protocols.

**Files:**
- Create: `src/bold_reliability_movies/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 3.1: Write the failing test**

`tests/test_types.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from bold_reliability_movies.types import (
    Frame,
    FrameGroup,
    Renderer,
    FrameSource,
)


def test_frame_is_frozen():
    f = Frame(path=Path("/tmp/x.nii.gz"), label="a", sort_key=(1,))
    with pytest.raises(Exception):  # FrozenInstanceError on dataclass
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

    # Protocols are runtime-checkable; callables match the __call__ signature.
    assert isinstance(renderer, Renderer)


def test_frame_source_protocol_accepts_class_with_discover():
    class MySource:
        def discover(self) -> list[FrameGroup]:
            return []

    assert isinstance(MySource(), FrameSource)
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3.3: Implement `types.py`**

```python
"""Public dataclasses and Protocols crossing layer boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class Frame:
    """One frame of an output video.

    `path` is a 4D BOLD NIfTI; the pipeline computes its mean before passing
    the resulting 3D image to a Renderer. `label` is overlaid on the frame.
    `sort_key` is opaque to the renderer; the discovery layer uses it to
    order frames within a group.
    """

    path: Path
    label: str
    sort_key: tuple[Any, ...]


@dataclass(frozen=True)
class FrameGroup:
    """One output video (one MP4)."""

    name: str
    frames: list[Frame]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Renderer(Protocol):
    """A callable turning a mean image into an RGB frame.

    Implementations MUST return the same (H, W) for every call within a
    single video; (H, W, 3) uint8 arrays only.
    """

    def __call__(self, mean_img: nib.Nifti1Image, label: str) -> np.ndarray:
        ...


@runtime_checkable
class FrameSource(Protocol):
    """An object that produces FrameGroups from some input (BIDS dir, TSV, ...)."""

    def discover(self) -> list[FrameGroup]:
        ...
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `uv run pytest tests/test_types.py -v`
Expected: 5 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/bold_reliability_movies/types.py tests/test_types.py
git commit -m "feat(types): Frame, FrameGroup, Renderer + FrameSource Protocols

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Shared test fixtures in `conftest.py`

**Goal:** Synthetic NIfTI generator + fake fmriprep tree. Used by every downstream test.

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 4.1: Write the fixtures (no failing-test step — fixtures are shared infra; verified by downstream tests)**

`tests/conftest.py`:

```python
"""Shared pytest fixtures: synthetic NIfTI volumes + fake fmriprep tree."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _make_bold(path: Path, shape: tuple[int, int, int, int] = (8, 8, 4, 5)) -> Path:
    """Write a synthetic 4D BOLD NIfTI with a deterministic ramp signal."""
    rng = np.random.default_rng(seed=hash(str(path)) % (2**32))
    base = rng.normal(loc=100, scale=2, size=shape).astype(np.float32)
    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(base, affine), str(path))
    return path


@pytest.fixture
def make_bold(tmp_path: Path) -> Callable[..., Path]:
    """Factory: write a tiny BOLD NIfTI at a path under tmp_path."""

    def _factory(name: str = "bold.nii.gz", shape: tuple[int, int, int, int] = (8, 8, 4, 5)) -> Path:
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
        frame[0, 0, 0] = hash(label) % 256
        return frame

    return _renderer
```

- [ ] **Step 4.2: Verify fixtures importable**

Run: `uv run pytest --collect-only -q`
Expected: existing tests collected, no errors.

- [ ] **Step 4.3: Commit**

```bash
git add tests/conftest.py
git commit -m "test(conftest): synthetic BOLD + fake fmriprep tree + stub renderer

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Mean cache in `mean_cache.py`

**Goal:** `compute_mean(path, cache_dir=None) -> Nifti1Image` with on-disk cache, mtime invalidation, atomic write, and a non-fatal fallback when the cache dir is read-only.

**Files:**
- Create: `src/bold_reliability_movies/mean_cache.py`
- Test: `tests/test_mean_cache.py`

- [ ] **Step 5.1: Write the failing test**

`tests/test_mean_cache.py`:

```python
from __future__ import annotations

import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from bold_reliability_movies.mean_cache import compute_mean, _cache_path_for


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
    with pytest.raises(Exception):
        compute_mean(bad)
```

- [ ] **Step 5.2: Run test to verify it fails**

Run: `uv run pytest tests/test_mean_cache.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 5.3: Implement `mean_cache.py`**

```python
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
    return mean_img(str(source), copy_header=True)


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
        return nib.load(str(cache))

    img = _compute_from_disk(source)
    try:
        _atomic_write(img, cache)
    except OSError as exc:
        log.warning("mean cache write failed for %s: %s — continuing in memory", source, exc)
    return img
```

- [ ] **Step 5.4: Run test to verify it passes**

Run: `uv run pytest tests/test_mean_cache.py -v`
Expected: 7 passed.

- [ ] **Step 5.5: Commit**

```bash
git add src/bold_reliability_movies/mean_cache.py tests/test_mean_cache.py
git commit -m "feat(mean_cache): on-disk mean cache with mtime invalidation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Encoder in `encode.py`

**Goal:** `probe_ffmpeg()` returns the binary path or raises `MissingDependency`. `encode(frames, fps, out_path)` pipes raw RGB frames into `ffmpeg` and writes an MP4. Tests mock the subprocess.

**Files:**
- Create: `src/bold_reliability_movies/encode.py`
- Test: `tests/test_encode.py`

- [ ] **Step 6.1: Write the failing test**

`tests/test_encode.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from bold_reliability_movies.encode import encode, probe_ffmpeg
from bold_reliability_movies.errors import EncodeError, MissingDependency


def test_probe_ffmpeg_returns_path(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    assert probe_ffmpeg() == "/usr/bin/ffmpeg"


def test_probe_ffmpeg_raises_when_missing(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(MissingDependency) as ei:
        probe_ffmpeg()
    assert "ffmpeg" in str(ei.value).lower()


def _frames(n: int = 3, h: int = 8, w: int = 8) -> list[np.ndarray]:
    return [np.full((h, w, 3), i * 30, dtype=np.uint8) for i in range(n)]


def test_encode_invokes_ffmpeg_with_correct_args(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdin.write = MagicMock()
        proc.stdin.close = MagicMock()
        proc.wait = MagicMock(return_value=0)
        proc.stderr = MagicMock()
        proc.stderr.read = MagicMock(return_value=b"")
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")

    out = tmp_path / "out.mp4"
    encode(_frames(), fps=2, out_path=out)

    cmd = captured["cmd"]
    assert cmd[0] == "/usr/bin/ffmpeg"
    assert "-f" in cmd and "rawvideo" in cmd
    assert "-s" in cmd and "8x8" in cmd
    assert "-pix_fmt" in cmd and "rgb24" in cmd
    assert "-r" in cmd and "2" in cmd
    assert "-c:v" in cmd and "libx264" in cmd
    assert "yuv420p" in cmd
    assert str(out) in cmd


def test_encode_raises_on_nonzero_return(monkeypatch, tmp_path: Path) -> None:
    def fake_popen(cmd, *args, **kwargs):
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.wait = MagicMock(return_value=1)
        proc.stderr = MagicMock()
        proc.stderr.read = MagicMock(return_value=b"ffmpeg failed")
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")

    with pytest.raises(EncodeError) as ei:
        encode(_frames(), fps=2, out_path=tmp_path / "out.mp4")
    assert "ffmpeg failed" in str(ei.value)


def test_encode_rejects_empty_frame_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        encode([], fps=2, out_path=tmp_path / "out.mp4")


@pytest.mark.slow
def test_encode_real_ffmpeg(tmp_path: Path) -> None:
    """End-to-end real ffmpeg encode. Run with: pytest -m slow"""
    out = tmp_path / "real.mp4"
    encode(_frames(n=5, h=32, w=32), fps=2, out_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
```

- [ ] **Step 6.2: Run test to verify it fails**

Run: `uv run pytest tests/test_encode.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 6.3: Implement `encode.py`**

```python
"""MP4 encoder: pipes raw RGB frames into a system ffmpeg subprocess."""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bold_reliability_movies.errors import EncodeError, MissingDependency

log = logging.getLogger(__name__)

_INSTALL_HINTS = (
    "Install ffmpeg via one of:\n"
    "  macOS:        brew install ffmpeg\n"
    "  Ubuntu/Debian: sudo apt install ffmpeg\n"
    "  Conda:        conda install -c conda-forge ffmpeg\n"
    "  HPC modules:  module load ffmpeg"
)


def probe_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise MissingDependency(
            "ffmpeg not found on PATH (required to encode MP4 videos).\n"
            + _INSTALL_HINTS
        )
    return path


def encode(
    frames: Sequence[np.ndarray],
    fps: int,
    out_path: Path,
) -> None:
    """Encode RGB frames to MP4 at out_path. All frames must share (H, W, 3) shape."""
    if len(frames) == 0:
        raise ValueError("encode() requires at least one frame")

    ffmpeg = probe_ffmpeg()
    h, w, c = frames[0].shape
    if c != 3:
        raise ValueError(f"frames must be (H, W, 3) RGB; got {frames[0].shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(out_path),
    ]
    log.debug("ffmpeg cmd: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            proc.stdin.write(frame.tobytes())  # type: ignore[union-attr]
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise EncodeError(f"ffmpeg exited with rc={rc}: {err}")
```

- [ ] **Step 6.4: Run test to verify it passes**

Run: `uv run pytest tests/test_encode.py -v`
Expected: 5 passed (the slow test is skipped by default).

- [ ] **Step 6.5: Optionally run the real ffmpeg test**

Run: `uv run pytest tests/test_encode.py -m slow -v`
Expected: 1 passed if `ffmpeg` is on PATH; otherwise `MissingDependency` confirms the probe works against the real environment.

- [ ] **Step 6.6: Commit**

```bash
git add src/bold_reliability_movies/encode.py tests/test_encode.py
git commit -m "feat(encode): MP4 encoder via ffmpeg subprocess + probe

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Mosaic renderer

**Goal:** A renderer that tiles N evenly-spaced axial slices of a 3D mean image into a grid, normalizes intensity, and overlays a label. Returns a constant-shape `(H, W, 3) uint8` array.

**Files:**
- Create: `src/bold_reliability_movies/renderers/__init__.py` (empty for now; populated in Task 9)
- Create: `src/bold_reliability_movies/renderers/mosaic.py`
- Create: `tests/test_renderers/__init__.py` (empty)
- Create: `tests/test_renderers/test_mosaic.py`

- [ ] **Step 7.1: Write the failing test**

`tests/test_renderers/test_mosaic.py`:

```python
from __future__ import annotations

import numpy as np
import nibabel as nib

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
```

- [ ] **Step 7.2: Run test to verify it fails**

Run: `uv run pytest tests/test_renderers/test_mosaic.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 7.3: Implement `mosaic.py`**

```python
"""Default in-tree renderer: a grid of axial slices with a text label."""

from __future__ import annotations

import nibabel as nib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class MosaicRenderer:
    """Render a 3D mean image as an N×M grid of evenly-spaced axial slices."""

    def __init__(
        self,
        n_rows: int = 5,
        n_cols: int = 5,
        fig_size: tuple[float, float] = (8.0, 8.0),
        dpi: int = 100,
        cmap: str = "gray",
        vmin_pct: float = 1.0,
        vmax_pct: float = 99.0,
    ) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.fig_size = fig_size
        self.dpi = dpi
        self.cmap = cmap
        self.vmin_pct = vmin_pct
        self.vmax_pct = vmax_pct

    def __call__(self, mean_img: nib.Nifti1Image, label: str) -> np.ndarray:
        data = np.asarray(mean_img.get_fdata(), dtype=np.float32)
        n_slices = self.n_rows * self.n_cols
        nz = data.shape[2]
        z_idx = np.linspace(0, nz - 1, n_slices).astype(int)
        vmin = float(np.percentile(data, self.vmin_pct))
        vmax = float(np.percentile(data, self.vmax_pct))

        fig = Figure(figsize=self.fig_size, dpi=self.dpi, facecolor="black")
        canvas = FigureCanvasAgg(fig)
        for k, z in enumerate(z_idx):
            ax = fig.add_subplot(self.n_rows, self.n_cols, k + 1)
            ax.imshow(
                np.rot90(data[:, :, z]),
                cmap=self.cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05, hspace=0.05)
        fig.text(0.5, 0.96, label, ha="center", va="top", color="white", fontsize=14, family="monospace")

        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        rgb = rgba[:, :, :3].copy()
        return rgb
```

- [ ] **Step 7.4: Run test to verify it passes**

Run: `uv run pytest tests/test_renderers/test_mosaic.py -v`
Expected: 5 passed.

- [ ] **Step 7.5: Commit**

```bash
git add src/bold_reliability_movies/renderers/__init__.py src/bold_reliability_movies/renderers/mosaic.py tests/test_renderers/__init__.py tests/test_renderers/test_mosaic.py
git commit -m "feat(renderers): mosaic renderer (N×M axial-slice grid)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Triplet renderer

**Goal:** Axial + sagittal + coronal mid-cuts side by side, with a label. Parity with `neuro_workflow/qa/reliability.py`.

**Files:**
- Create: `src/bold_reliability_movies/renderers/triplet.py`
- Create: `tests/test_renderers/test_triplet.py`

- [ ] **Step 8.1: Write the failing test**

`tests/test_renderers/test_triplet.py`:

```python
from __future__ import annotations

import nibabel as nib
import numpy as np

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
```

- [ ] **Step 8.2: Run test to verify it fails**

Run: `uv run pytest tests/test_renderers/test_triplet.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 8.3: Implement `triplet.py`**

```python
"""Triplet renderer: mid-cut axial + sagittal + coronal panels."""

from __future__ import annotations

import nibabel as nib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class TripletRenderer:
    """Render a 3D mean image as three orthogonal mid-cuts side-by-side."""

    def __init__(
        self,
        fig_size: tuple[float, float] = (12.0, 4.5),
        dpi: int = 100,
        cmap: str = "gray",
        vmin_pct: float = 1.0,
        vmax_pct: float = 99.0,
    ) -> None:
        self.fig_size = fig_size
        self.dpi = dpi
        self.cmap = cmap
        self.vmin_pct = vmin_pct
        self.vmax_pct = vmax_pct

    def __call__(self, mean_img: nib.Nifti1Image, label: str) -> np.ndarray:
        data = np.asarray(mean_img.get_fdata(), dtype=np.float32)
        nx, ny, nz = data.shape
        vmin = float(np.percentile(data, self.vmin_pct))
        vmax = float(np.percentile(data, self.vmax_pct))

        fig = Figure(figsize=self.fig_size, dpi=self.dpi, facecolor="black")
        canvas = FigureCanvasAgg(fig)

        cuts = [
            ("axial",    np.rot90(data[:, :, nz // 2])),
            ("sagittal", np.rot90(data[nx // 2, :, :])),
            ("coronal",  np.rot90(data[:, ny // 2, :])),
        ]
        for k, (title, slc) in enumerate(cuts):
            ax = fig.add_subplot(1, 3, k + 1)
            ax.imshow(slc, cmap=self.cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(title, color="white", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.02, wspace=0.05)
        fig.text(0.5, 0.95, label, ha="center", va="top", color="white", fontsize=12, family="monospace")

        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        return rgba[:, :, :3].copy()
```

- [ ] **Step 8.4: Run test to verify it passes**

Run: `uv run pytest tests/test_renderers/test_triplet.py -v`
Expected: 3 passed.

- [ ] **Step 8.5: Commit**

```bash
git add src/bold_reliability_movies/renderers/triplet.py tests/test_renderers/test_triplet.py
git commit -m "feat(renderers): triplet renderer (axial+sagittal+coronal mid-cuts)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: In-tree renderer dispatch table

**Goal:** A name-to-instance registry the CLI uses. `get_renderer("mosaic")` returns a `MosaicRenderer()`. Unknown names raise `UnknownRendererError` with the available list.

**Files:**
- Modify: `src/bold_reliability_movies/renderers/__init__.py`
- Test: `tests/test_renderer_registry.py`

- [ ] **Step 9.1: Write the failing test**

`tests/test_renderer_registry.py`:

```python
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
```

- [ ] **Step 9.2: Run test to verify it fails**

Run: `uv run pytest tests/test_renderer_registry.py -v`
Expected: `ImportError` for `REGISTRY`.

- [ ] **Step 9.3: Implement `renderers/__init__.py`**

```python
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
```

- [ ] **Step 9.4: Run test to verify it passes**

Run: `uv run pytest tests/test_renderer_registry.py -v`
Expected: 4 passed.

- [ ] **Step 9.5: Run all tests to confirm nothing regressed**

Run: `uv run pytest -v`
Expected: all green.

- [ ] **Step 9.6: Commit**

```bash
git add src/bold_reliability_movies/renderers/__init__.py tests/test_renderer_registry.py
git commit -m "feat(renderers): in-tree dispatch table with get_renderer/list_renderers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Pipeline orchestration

**Goal:** `make_video(group, renderer, out_path, fps, *, cache_dir, use_cache)` and `make_videos(groups, ...)`. Per-group failure isolation, shape-mismatch detection, dropped-frame logging.

**Files:**
- Create: `src/bold_reliability_movies/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 10.1: Write the failing test**

`tests/test_pipeline.py`:

```python
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import nibabel as nib
import numpy as np
import pytest

from bold_reliability_movies.errors import InconsistentShapesError
from bold_reliability_movies.pipeline import make_video, make_videos
from bold_reliability_movies.types import Frame, FrameGroup


@pytest.fixture(autouse=True)
def _mock_ffmpeg(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")

    def fake_popen(cmd, *args, **kwargs):
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdin.write = MagicMock()
        proc.stdin.close = MagicMock()
        proc.wait = MagicMock(return_value=0)
        proc.stderr = MagicMock()
        proc.stderr.read = MagicMock(return_value=b"")
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)


def _frame(make_bold, name: str, label: str, sort_key=(0,)) -> Frame:
    return Frame(path=make_bold(name), label=label, sort_key=sort_key)


def test_make_video_runs_renderer_per_frame(make_bold, stub_renderer, tmp_path):
    calls: list[str] = []

    def renderer(img, label):
        calls.append(label)
        return stub_renderer(img, label)

    group = FrameGroup(
        name="g",
        frames=[
            _frame(make_bold, "a.nii.gz", "a"),
            _frame(make_bold, "b.nii.gz", "b"),
            _frame(make_bold, "c.nii.gz", "c"),
        ],
    )
    make_video(group, renderer=renderer, out_path=tmp_path / "g.mp4", fps=2)
    assert calls == ["a", "b", "c"]


def test_make_video_detects_shape_mismatch(make_bold, tmp_path):
    def renderer(img, label):
        if label == "b":
            return np.zeros((32, 32, 3), dtype=np.uint8)
        return np.zeros((16, 16, 3), dtype=np.uint8)

    group = FrameGroup(
        name="g",
        frames=[
            _frame(make_bold, "a.nii.gz", "a"),
            _frame(make_bold, "b.nii.gz", "b"),
        ],
    )
    with pytest.raises(InconsistentShapesError) as ei:
        make_video(group, renderer=renderer, out_path=tmp_path / "g.mp4", fps=2)
    assert ei.value.frame_index == 1


def test_make_videos_isolates_per_group_failures(make_bold, stub_renderer, tmp_path, caplog):
    bad_group = FrameGroup(
        name="bad",
        frames=[Frame(path=Path("/does/not/exist.nii.gz"), label="x", sort_key=(0,))],
    )
    good_group = FrameGroup(
        name="good",
        frames=[_frame(make_bold, "ok.nii.gz", "ok")],
    )
    caplog.set_level(logging.WARNING)
    summary = make_videos(
        [bad_group, good_group],
        renderer=stub_renderer,
        out_dir=tmp_path,
        fps=2,
    )
    assert summary.succeeded == ["good"]
    assert summary.failed == ["bad"]


def test_make_videos_drops_corrupt_frames_below_threshold(tmp_path, make_bold, stub_renderer):
    bad = tmp_path / "bad.nii.gz"
    bad.write_bytes(b"not a nifti")
    group = FrameGroup(
        name="g",
        frames=[
            Frame(path=bad, label="a", sort_key=(0,)),
            Frame(path=make_bold("b.nii.gz"), label="b", sort_key=(1,)),
            Frame(path=make_bold("c.nii.gz"), label="c", sort_key=(2,)),
        ],
    )
    summary = make_videos([group], renderer=stub_renderer, out_dir=tmp_path, fps=2)
    assert summary.succeeded == ["g"]


def test_make_videos_skips_group_with_too_few_frames(tmp_path, stub_renderer):
    bad = tmp_path / "bad.nii.gz"
    bad.write_bytes(b"not a nifti")
    group = FrameGroup(
        name="g",
        frames=[Frame(path=bad, label="a", sort_key=(0,))],
    )
    summary = make_videos([group], renderer=stub_renderer, out_dir=tmp_path, fps=2)
    assert summary.failed == ["g"]
```

- [ ] **Step 10.2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 10.3: Implement `pipeline.py`**

```python
"""Orchestration: turn FrameGroups into MP4 files."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bold_reliability_movies.encode import encode
from bold_reliability_movies.errors import InconsistentShapesError
from bold_reliability_movies.mean_cache import compute_mean
from bold_reliability_movies.types import Frame, FrameGroup, Renderer

log = logging.getLogger(__name__)

# When more than this fraction of frames in a group fail to render or load,
# we skip the entire group. Below the threshold, bad frames are dropped and
# the remaining frames are encoded.
_DROP_THRESHOLD = 0.5
_MIN_FRAMES = 2


@dataclass
class RunSummary:
    succeeded: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def _render_frames(
    frames: Sequence[Frame],
    renderer: Renderer,
    cache_dir: Path | None,
    use_cache: bool,
) -> tuple[list[np.ndarray], list[Frame]]:
    """Compute means and render. Returns (rgb_frames, kept_frames). Bad frames dropped."""
    rgb: list[np.ndarray] = []
    kept: list[Frame] = []
    for frame in frames:
        try:
            mean_img = compute_mean(frame.path, cache_dir=cache_dir, use_cache=use_cache)
        except Exception as exc:
            log.warning("dropping frame %s (%s): mean computation failed: %s", frame.label, frame.path, exc)
            continue
        try:
            arr = renderer(mean_img, frame.label)
        except Exception as exc:
            log.warning("dropping frame %s: renderer raised: %s", frame.label, exc)
            continue
        rgb.append(arr)
        kept.append(frame)
    return rgb, kept


def _check_shapes(rgb: Sequence[np.ndarray]) -> None:
    if not rgb:
        return
    first = rgb[0].shape
    for i, arr in enumerate(rgb[1:], start=1):
        if arr.shape != first:
            raise InconsistentShapesError(
                frame_index=i,
                previous_shape=first,
                current_shape=arr.shape,
                suggestion="use --group-by subject+task",
            )


def make_video(
    group: FrameGroup,
    *,
    renderer: Renderer,
    out_path: Path,
    fps: int,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> None:
    """Render a single FrameGroup to one MP4."""
    rgb, kept = _render_frames(group.frames, renderer, cache_dir, use_cache)
    n_total = len(group.frames)
    n_kept = len(kept)
    if n_total > 0 and (n_total - n_kept) / n_total > _DROP_THRESHOLD:
        raise RuntimeError(
            f"group {group.name!r}: {n_total - n_kept}/{n_total} frames failed (threshold {_DROP_THRESHOLD})"
        )
    if n_kept < _MIN_FRAMES:
        raise RuntimeError(f"group {group.name!r}: only {n_kept} usable frame(s); need >= {_MIN_FRAMES}")

    _check_shapes(rgb)
    log.info("encoding group %s (%d frames) → %s", group.name, n_kept, out_path)
    encode(rgb, fps=fps, out_path=out_path)


def make_videos(
    groups: Sequence[FrameGroup],
    *,
    renderer: Renderer,
    out_dir: Path,
    fps: int,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> RunSummary:
    """Render many FrameGroups; isolate per-group failures."""
    summary = RunSummary()
    out_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        out_path = out_dir / f"{group.name}.mp4"
        try:
            make_video(
                group,
                renderer=renderer,
                out_path=out_path,
                fps=fps,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
        except Exception as exc:
            log.error("group %s failed: %s", group.name, exc)
            summary.failed.append(group.name)
        else:
            summary.succeeded.append(group.name)
    return summary
```

- [ ] **Step 10.4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 5 passed.

- [ ] **Step 10.5: Commit**

```bash
git add src/bold_reliability_movies/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): make_video / make_videos with shape-check + per-group isolation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: fmriprep discovery

**Goal:** `FmriprepFrameSource(deriv_dir, *, group_by, filters, glob)` walks an fMRIPrep derivatives directory, parses BIDS entities from filenames, applies filters, and groups + sorts frames.

**Files:**
- Create: `src/bold_reliability_movies/discovery/__init__.py`
- Create: `src/bold_reliability_movies/discovery/fmriprep.py`
- Test: `tests/test_discovery_fmriprep.py`

- [ ] **Step 11.1: Write the failing test**

`tests/test_discovery_fmriprep.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from bold_reliability_movies.discovery.fmriprep import FmriprepFrameSource


def test_discovery_groups_by_subject_default(fake_fmriprep_tree: Path):
    src = FmriprepFrameSource(deriv_dir=fake_fmriprep_tree)
    groups = src.discover()
    names = sorted(g.name for g in groups)
    assert names == ["sub-s01", "sub-s02"]


def test_discovery_sort_order_by_session_task_run(fake_fmriprep_tree: Path):
    src = FmriprepFrameSource(deriv_dir=fake_fmriprep_tree)
    groups = {g.name: g for g in src.discover()}
    s01 = groups["sub-s01"]
    labels = [f.label for f in s01.frames]
    assert labels == [
        "ses-01 task-rest run-1",
        "ses-01 task-rest run-2",
        "ses-02 task-stroop run-1",
    ]


def test_discovery_filter_by_task(fake_fmriprep_tree: Path):
    src = FmriprepFrameSource(deriv_dir=fake_fmriprep_tree, filters={"task": "rest"})
    groups = {g.name: g for g in src.discover()}
    assert all(f.label.split()[1] == "task-rest" for g in groups.values() for f in g.frames)


def test_discovery_filter_by_subject(fake_fmriprep_tree: Path):
    src = FmriprepFrameSource(deriv_dir=fake_fmriprep_tree, filters={"sub": "s01"})
    groups = src.discover()
    assert {g.name for g in groups} == {"sub-s01"}


def test_discovery_group_by_subject_and_task(fake_fmriprep_tree: Path):
    src = FmriprepFrameSource(deriv_dir=fake_fmriprep_tree, group_by="subject+task")
    groups = sorted(g.name for g in src.discover())
    assert groups == ["sub-s01_task-rest", "sub-s01_task-stroop", "sub-s02_task-rest"]


def test_discovery_group_by_none_returns_one_group(fake_fmriprep_tree: Path):
    src = FmriprepFrameSource(deriv_dir=fake_fmriprep_tree, group_by="none")
    groups = src.discover()
    assert len(groups) == 1
    assert groups[0].name == "all"
    assert len(groups[0].frames) == 4


def test_discovery_skips_unparseable_filenames(tmp_path: Path):
    d = tmp_path / "deriv" / "sub-s01" / "ses-01" / "func"
    d.mkdir(parents=True)
    (d / "garbage.nii.gz").write_bytes(b"")
    (d / "sub-s01_ses-01_task-rest_run-1_desc-preproc_bold.nii.gz").write_bytes(b"")
    src = FmriprepFrameSource(deriv_dir=tmp_path / "deriv")
    groups = src.discover()
    # 1 group, 1 frame; the garbage file is silently skipped.
    assert len(groups) == 1
    assert len(groups[0].frames) == 1
```

- [ ] **Step 11.2: Run test to verify it fails**

Run: `uv run pytest tests/test_discovery_fmriprep.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 11.3: Implement `discovery/__init__.py`**

```python
"""Discovery layer: turn input sources into FrameGroups."""

from bold_reliability_movies.discovery.fmriprep import FmriprepFrameSource

__all__ = ["FmriprepFrameSource"]
```

- [ ] **Step 11.4: Implement `discovery/fmriprep.py`**

```python
"""FrameSource for fMRIPrep derivatives directories."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from bold_reliability_movies.types import Frame, FrameGroup

log = logging.getLogger(__name__)

# Match fMRIPrep preproc BOLD filenames; tolerant of optional space/echo/desc tokens.
_FILENAME_RE = re.compile(
    r"sub-(?P<sub>[A-Za-z0-9]+)_"
    r"ses-(?P<ses>[A-Za-z0-9]+)_"
    r"task-(?P<task>[A-Za-z0-9]+)_"
    r"run-(?P<run>\d+)_"
    r".*?desc-preproc_bold\.nii\.gz$"
)

GroupBy = Literal["subject", "subject+task", "subject+session", "none"]


@dataclass
class _Entities:
    sub: str
    ses: str
    task: str
    run: int

    @property
    def ses_num(self) -> int:
        m = re.search(r"\d+", self.ses)
        return int(m.group()) if m else 0


def _parse(name: str) -> _Entities | None:
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return _Entities(sub=m["sub"], ses=m["ses"], task=m["task"], run=int(m["run"]))


def _entities_match(ents: _Entities, filters: dict[str, str]) -> bool:
    for k, v in filters.items():
        if k == "sub" and ents.sub != v:
            return False
        if k == "ses" and ents.ses != v:
            return False
        if k == "task" and ents.task != v:
            return False
        if k == "run" and str(ents.run) != v:
            return False
    return True


def _group_key(ents: _Entities, mode: GroupBy) -> str:
    if mode == "subject":
        return f"sub-{ents.sub}"
    if mode == "subject+task":
        return f"sub-{ents.sub}_task-{ents.task}"
    if mode == "subject+session":
        return f"sub-{ents.sub}_ses-{ents.ses}"
    if mode == "none":
        return "all"
    raise ValueError(f"unknown group_by mode: {mode}")


@dataclass
class FmriprepFrameSource:
    """Walk a fMRIPrep-style derivatives directory and yield FrameGroups."""

    deriv_dir: Path
    group_by: GroupBy = "subject"
    filters: dict[str, str] = field(default_factory=dict)
    glob: str = "sub-*/ses-*/func/*_desc-preproc_bold.nii.gz"

    def discover(self) -> list[FrameGroup]:
        files = sorted(self.deriv_dir.glob(self.glob))
        buckets: dict[str, list[Frame]] = defaultdict(list)
        meta: dict[str, dict[str, str]] = defaultdict(dict)

        for fp in files:
            ents = _parse(fp.name)
            if ents is None:
                log.debug("skipping unparseable filename: %s", fp.name)
                continue
            if not _entities_match(ents, self.filters):
                continue
            key = _group_key(ents, self.group_by)
            label = f"ses-{ents.ses} task-{ents.task} run-{ents.run}"
            sort_key = (ents.sub, ents.ses_num, ents.task, ents.run)
            buckets[key].append(Frame(path=fp, label=label, sort_key=sort_key))
            meta[key].setdefault("subject", ents.sub)

        groups: list[FrameGroup] = []
        for key in sorted(buckets):
            frames = sorted(buckets[key], key=lambda f: f.sort_key)
            groups.append(FrameGroup(name=key, frames=frames, metadata=meta[key]))
        return groups
```

- [ ] **Step 11.5: Run test to verify it passes**

Run: `uv run pytest tests/test_discovery_fmriprep.py -v`
Expected: 7 passed.

- [ ] **Step 11.6: Commit**

```bash
git add src/bold_reliability_movies/discovery/__init__.py src/bold_reliability_movies/discovery/fmriprep.py tests/test_discovery_fmriprep.py
git commit -m "feat(discovery): FmriprepFrameSource with filters + group_by modes

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Manifest TSV discovery

**Goal:** `ManifestFrameSource(tsv_path)` reads a TSV with `path`, `label`, `group`, optional `sort_key`.

**Files:**
- Modify: `src/bold_reliability_movies/discovery/__init__.py`
- Create: `src/bold_reliability_movies/discovery/manifest.py`
- Test: `tests/test_discovery_manifest.py`

- [ ] **Step 12.1: Write the failing test**

`tests/test_discovery_manifest.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from bold_reliability_movies.discovery.manifest import ManifestFrameSource


def _write(tsv: Path, body: str) -> Path:
    tsv.write_text(body)
    return tsv


def test_manifest_groups_rows(tmp_path: Path):
    body = (
        "path\tlabel\tgroup\n"
        "/x/a.nii.gz\tses-01 r1\tsub-01\n"
        "/x/b.nii.gz\tses-01 r2\tsub-01\n"
        "/x/c.nii.gz\tses-01 r1\tsub-02\n"
    )
    tsv = _write(tmp_path / "m.tsv", body)
    groups = ManifestFrameSource(tsv_path=tsv).discover()
    names = sorted(g.name for g in groups)
    assert names == ["sub-01", "sub-02"]
    sub01 = next(g for g in groups if g.name == "sub-01")
    assert [f.label for f in sub01.frames] == ["ses-01 r1", "ses-01 r2"]


def test_manifest_skips_comment_lines(tmp_path: Path):
    body = (
        "# comment\n"
        "path\tlabel\tgroup\n"
        "# another comment\n"
        "/x/a.nii.gz\tlbl\tg1\n"
    )
    tsv = _write(tmp_path / "m.tsv", body)
    groups = ManifestFrameSource(tsv_path=tsv).discover()
    assert len(groups) == 1
    assert groups[0].frames[0].label == "lbl"


def test_manifest_default_sort_key_preserves_row_order(tmp_path: Path):
    body = (
        "path\tlabel\tgroup\n"
        "/x/c.nii.gz\tc\tg\n"
        "/x/a.nii.gz\ta\tg\n"
        "/x/b.nii.gz\tb\tg\n"
    )
    tsv = _write(tmp_path / "m.tsv", body)
    groups = ManifestFrameSource(tsv_path=tsv).discover()
    assert [f.label for f in groups[0].frames] == ["c", "a", "b"]


def test_manifest_explicit_sort_key(tmp_path: Path):
    body = (
        "path\tlabel\tgroup\tsort_key\n"
        "/x/c.nii.gz\tc\tg\t30\n"
        "/x/a.nii.gz\ta\tg\t10\n"
        "/x/b.nii.gz\tb\tg\t20\n"
    )
    tsv = _write(tmp_path / "m.tsv", body)
    groups = ManifestFrameSource(tsv_path=tsv).discover()
    assert [f.label for f in groups[0].frames] == ["a", "b", "c"]


def test_manifest_missing_required_column_errors(tmp_path: Path):
    body = "path\tlabel\n/x/a.nii.gz\tlbl\n"
    tsv = _write(tmp_path / "m.tsv", body)
    with pytest.raises(ValueError) as ei:
        ManifestFrameSource(tsv_path=tsv).discover()
    assert "group" in str(ei.value).lower()
```

- [ ] **Step 12.2: Run test to verify it fails**

Run: `uv run pytest tests/test_discovery_manifest.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 12.3: Implement `discovery/manifest.py`**

```python
"""FrameSource for tab-separated manifest files."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bold_reliability_movies.types import Frame, FrameGroup


_REQUIRED = ("path", "label", "group")


def _parse_sort_key(raw: str) -> tuple[Any, ...]:
    """Parse sort_key column. Numeric tokens become ints; strings stay strings."""
    parts = [p.strip() for p in raw.replace(",", "\t").split("\t") if p.strip()]
    out: list[Any] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            try:
                out.append(float(p))
            except ValueError:
                out.append(p)
    return tuple(out)


@dataclass
class ManifestFrameSource:
    """Read a TSV manifest. Required columns: path, label, group. Optional: sort_key."""

    tsv_path: Path

    def discover(self) -> list[FrameGroup]:
        with open(self.tsv_path, encoding="utf-8") as f:
            lines = [ln for ln in f if not ln.lstrip().startswith("#")]
        reader = csv.DictReader(lines, delimiter="\t")
        if reader.fieldnames is None:
            return []
        missing = [c for c in _REQUIRED if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"manifest {self.tsv_path} missing required column(s): {', '.join(missing)}"
            )

        buckets: dict[str, list[Frame]] = defaultdict(list)
        per_group_index: dict[str, int] = defaultdict(int)
        for row in reader:
            group = row["group"]
            idx = per_group_index[group]
            per_group_index[group] += 1
            sort_key: tuple[Any, ...]
            if "sort_key" in row and row["sort_key"]:
                sort_key = _parse_sort_key(row["sort_key"])
            else:
                sort_key = (idx,)
            buckets[group].append(
                Frame(path=Path(row["path"]), label=row["label"], sort_key=sort_key)
            )

        groups: list[FrameGroup] = []
        for name in sorted(buckets):
            frames = sorted(buckets[name], key=lambda f: f.sort_key)
            groups.append(FrameGroup(name=name, frames=frames))
        return groups
```

- [ ] **Step 12.4: Update `discovery/__init__.py`**

```python
"""Discovery layer: turn input sources into FrameGroups."""

from bold_reliability_movies.discovery.fmriprep import FmriprepFrameSource
from bold_reliability_movies.discovery.manifest import ManifestFrameSource

__all__ = ["FmriprepFrameSource", "ManifestFrameSource"]
```

- [ ] **Step 12.5: Run test to verify it passes**

Run: `uv run pytest tests/test_discovery_manifest.py -v`
Expected: 5 passed.

- [ ] **Step 12.6: Commit**

```bash
git add src/bold_reliability_movies/discovery/__init__.py src/bold_reliability_movies/discovery/manifest.py tests/test_discovery_manifest.py
git commit -m "feat(discovery): ManifestFrameSource (TSV with path/label/group/sort_key)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: CLI

**Goal:** `argparse`-based CLI with `bids`, `list`, `render` subcommands. Exit codes per spec. `brm --help` and `brm <subcmd> --help` produce clean output.

**Files:**
- Create: `src/bold_reliability_movies/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 13.1: Write the failing test**

`tests/test_cli.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bold_reliability_movies.cli import main


@pytest.fixture(autouse=True)
def _mock_ffmpeg(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")

    def fake_popen(cmd, *args, **kwargs):
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdin.write = MagicMock()
        proc.stdin.close = MagicMock()
        proc.wait = MagicMock(return_value=0)
        proc.stderr = MagicMock()
        proc.stderr.read = MagicMock(return_value=b"")
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)


def test_cli_no_args_exits_2(capsys):
    rc = main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "subcommand" in err.lower() or "usage" in err.lower()


def test_cli_unknown_renderer_exits_2(fake_fmriprep_tree: Path, capsys, tmp_path: Path):
    rc = main(
        [
            "bids",
            str(fake_fmriprep_tree),
            "--renderer", "flatmap",
            "--out", str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "flatmap" in err
    assert "mosaic" in err


def test_cli_empty_discovery_exits_3(tmp_path: Path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = main(["bids", str(empty), "--out", str(tmp_path / "out")])
    assert rc == 3


def test_cli_bids_runs_end_to_end(fake_fmriprep_tree: Path, tmp_path: Path):
    out = tmp_path / "movies"
    rc = main(
        [
            "bids",
            str(fake_fmriprep_tree),
            "--out", str(out),
            "--renderer", "mosaic",
            "--fps", "2",
            "--no-cache",
        ]
    )
    assert rc == 0


def test_cli_list_subcommand(tmp_path: Path, make_bold):
    a = make_bold("a.nii.gz")
    b = make_bold("b.nii.gz")
    tsv = tmp_path / "m.tsv"
    tsv.write_text(
        "path\tlabel\tgroup\n"
        f"{a}\tses-01 r1\tsub-01\n"
        f"{b}\tses-01 r2\tsub-01\n"
    )
    rc = main(["list", str(tsv), "--out", str(tmp_path / "movies"), "--no-cache"])
    assert rc == 0


def test_cli_render_subcommand(tmp_path: Path, make_bold):
    a = make_bold("a.nii.gz")
    b = make_bold("b.nii.gz")
    out = tmp_path / "render.mp4"
    rc = main(["render", str(a), str(b), "--out", str(out), "--no-cache"])
    assert rc == 0
```

- [ ] **Step 13.2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 13.3: Implement `cli.py`**

```python
"""argparse-based CLI: bids / list / render subcommands."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from bold_reliability_movies.discovery.fmriprep import FmriprepFrameSource
from bold_reliability_movies.discovery.manifest import ManifestFrameSource
from bold_reliability_movies.errors import (
    BrmError,
    EmptyDiscoveryError,
    UnknownRendererError,
)
from bold_reliability_movies.pipeline import make_video, make_videos
from bold_reliability_movies.renderers import get_renderer, list_renderers
from bold_reliability_movies.types import Frame, FrameGroup

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brm",
        description="BIDS-aware BOLD reliability movies.",
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    parser.add_argument("--quiet", action="store_true", help="WARNING-level logging")

    sub = parser.add_subparsers(dest="cmd", metavar="{bids,list,render}")

    # bids ----------------------------------------------------------------
    p_bids = sub.add_parser("bids", help="discover from an fMRIPrep derivatives dir")
    p_bids.add_argument("deriv_dir", type=Path)
    p_bids.add_argument("--out", type=Path, required=True, help="output dir")
    p_bids.add_argument("--fps", type=int, default=2)
    p_bids.add_argument("--renderer", default="mosaic", help=f"one of {list_renderers()}")
    p_bids.add_argument(
        "--group-by",
        choices=("subject", "subject+task", "subject+session", "none"),
        default="subject",
    )
    p_bids.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="ENTITY=VALUE",
        help="repeatable; e.g. --filter task=rest --filter sub=s01",
    )
    p_bids.add_argument("--cache", dest="cache", action="store_true", default=True)
    p_bids.add_argument("--no-cache", dest="cache", action="store_false")
    p_bids.add_argument("--cache-dir", type=Path, default=None)

    # list ----------------------------------------------------------------
    p_list = sub.add_parser("list", help="discover from a TSV manifest")
    p_list.add_argument("manifest", type=Path)
    p_list.add_argument("--out", type=Path, required=True)
    p_list.add_argument("--fps", type=int, default=2)
    p_list.add_argument("--renderer", default="mosaic")
    p_list.add_argument("--cache", dest="cache", action="store_true", default=True)
    p_list.add_argument("--no-cache", dest="cache", action="store_false")
    p_list.add_argument("--cache-dir", type=Path, default=None)

    # render --------------------------------------------------------------
    p_render = sub.add_parser("render", help="render a single video from positional NIfTIs")
    p_render.add_argument("niftis", type=Path, nargs="+")
    p_render.add_argument("--out", type=Path, required=True, help="output MP4 file")
    p_render.add_argument("--labels", nargs="+", default=None)
    p_render.add_argument("--renderer", default="mosaic")
    p_render.add_argument("--fps", type=int, default=2)
    p_render.add_argument("--cache", dest="cache", action="store_true", default=True)
    p_render.add_argument("--no-cache", dest="cache", action="store_false")
    p_render.add_argument("--cache-dir", type=Path, default=None)
    return parser


def _parse_filters(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--filter must be ENTITY=VALUE; got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def _setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    if quiet:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _cmd_bids(args: argparse.Namespace) -> int:
    try:
        renderer = get_renderer(args.renderer)
    except UnknownRendererError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    src = FmriprepFrameSource(
        deriv_dir=args.deriv_dir,
        group_by=args.group_by,
        filters=_parse_filters(args.filter),
    )
    groups = src.discover()
    if not groups:
        print(
            f"No matching BOLD files in {args.deriv_dir} (filters={_parse_filters(args.filter)}).",
            file=sys.stderr,
        )
        return 3
    summary = make_videos(
        groups,
        renderer=renderer,
        out_dir=args.out,
        fps=args.fps,
        cache_dir=args.cache_dir,
        use_cache=args.cache,
    )
    return 0 if not summary.failed else 1


def _cmd_list(args: argparse.Namespace) -> int:
    try:
        renderer = get_renderer(args.renderer)
    except UnknownRendererError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    src = ManifestFrameSource(tsv_path=args.manifest)
    groups = src.discover()
    if not groups:
        print(f"No rows in manifest {args.manifest}.", file=sys.stderr)
        return 3
    summary = make_videos(
        groups,
        renderer=renderer,
        out_dir=args.out,
        fps=args.fps,
        cache_dir=args.cache_dir,
        use_cache=args.cache,
    )
    return 0 if not summary.failed else 1


def _cmd_render(args: argparse.Namespace) -> int:
    try:
        renderer = get_renderer(args.renderer)
    except UnknownRendererError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    paths = [Path(p) for p in args.niftis]
    labels = args.labels if args.labels is not None else [p.stem for p in paths]
    if len(labels) != len(paths):
        print(
            f"--labels count ({len(labels)}) must match number of NIfTIs ({len(paths)})",
            file=sys.stderr,
        )
        return 2
    frames = [
        Frame(path=p, label=lbl, sort_key=(i,)) for i, (p, lbl) in enumerate(zip(paths, labels))
    ]
    group = FrameGroup(name=args.out.stem, frames=frames)
    try:
        make_video(
            group,
            renderer=renderer,
            out_path=args.out,
            fps=args.fps,
            cache_dir=args.cache_dir,
            use_cache=args.cache,
        )
    except BrmError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False), getattr(args, "quiet", False))

    if args.cmd is None:
        parser.print_usage(sys.stderr)
        print("error: a subcommand is required (bids | list | render)", file=sys.stderr)
        return 2

    if args.cmd == "bids":
        return _cmd_bids(args)
    if args.cmd == "list":
        return _cmd_list(args)
    if args.cmd == "render":
        return _cmd_render(args)
    parser.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 13.4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 6 passed.

- [ ] **Step 13.5: Run all tests**

Run: `uv run pytest -v`
Expected: all green.

- [ ] **Step 13.6: Commit**

```bash
git add src/bold_reliability_movies/cli.py tests/test_cli.py
git commit -m "feat(cli): bids / list / render subcommands with named exit codes

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Public API exports

**Goal:** `bold_reliability_movies/__init__.py` re-exports the stable surface so library users can `from bold_reliability_movies import make_video, Frame, FrameGroup, Renderer, FrameSource`.

**Files:**
- Modify: `src/bold_reliability_movies/__init__.py`

- [ ] **Step 14.1: Write the failing test (extend `tests/test_types.py`)**

Append to `tests/test_types.py`:

```python
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
```

- [ ] **Step 14.2: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py::test_public_api_reexports -v`
Expected: AttributeError.

- [ ] **Step 14.3: Implement `__init__.py`**

```python
"""bold-reliability-movies — BIDS-aware BOLD reliability movies."""

from __future__ import annotations

__version__ = "0.1.0"

from bold_reliability_movies.discovery.fmriprep import FmriprepFrameSource
from bold_reliability_movies.discovery.manifest import ManifestFrameSource
from bold_reliability_movies.pipeline import make_video, make_videos
from bold_reliability_movies.renderers.mosaic import MosaicRenderer
from bold_reliability_movies.renderers.triplet import TripletRenderer
from bold_reliability_movies.types import Frame, FrameGroup, FrameSource, Renderer

__all__ = [
    "__version__",
    "Frame",
    "FrameGroup",
    "FrameSource",
    "Renderer",
    "MosaicRenderer",
    "TripletRenderer",
    "FmriprepFrameSource",
    "ManifestFrameSource",
    "make_video",
    "make_videos",
]
```

- [ ] **Step 14.4: Run test to verify it passes**

Run: `uv run pytest tests/test_types.py::test_public_api_reexports -v`
Expected: 1 passed.

- [ ] **Step 14.5: Run full suite + lint + types**

```bash
uv run pytest -v
uv run ruff check
uv run mypy src/
```

Expected: all green. (Mypy overrides for `nibabel.*`, `nilearn.*`, `matplotlib.*` were added to `pyproject.toml` in Task 1, so missing-stub errors should not appear.)

- [ ] **Step 14.6: Commit**

```bash
git add src/bold_reliability_movies/__init__.py tests/test_types.py pyproject.toml
git commit -m "feat: public API surface in __init__.py + lint/type fixups

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: README, CHANGELOG, CONTRIBUTING

**Goal:** Real README with three install flows (uv, pip+venv, pip system), 60-second quickstart, custom-renderer example, attribution paragraph, link to spec.

**Files:**
- Modify: `README.md`
- Create: `CHANGELOG.md`
- Create: `CONTRIBUTING.md`

- [ ] **Step 15.1: Write `README.md`**

```markdown
# bold-reliability-movies

BIDS-aware BOLD reliability movies for visual fMRI QC. Cycle through the mean image of every BOLD run for a subject to spot dropout, drift, and across-run instability at a glance.

Inspired by Kendrick Kay's NSD inspection videos (`cvnlab/nsddatapaper/mainfigures/INSPECTIONS/GRANDVISUALIZATION`). This package is **not** a port of that code: it is a Python implementation that ships a volume-space mosaic renderer by default and exposes a `Renderer` Protocol so a flatmap (or any other) backend can be written by a third party.

## Install

### With uv (recommended)
\`\`\`bash
uv tool install bold-reliability-movies   # CLI available globally as `brm`
# or, in a project:
uv add bold-reliability-movies
\`\`\`

### With pip + venv
\`\`\`bash
python -m venv .venv && source .venv/bin/activate
pip install bold-reliability-movies
\`\`\`

### With pip (system Python — not recommended outside containers)
\`\`\`bash
pip install --user bold-reliability-movies
\`\`\`

### System dependency: ffmpeg
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Conda: `conda install -c conda-forge ffmpeg`
- HPC modules: `module load ffmpeg`

### From source (development)
\`\`\`bash
git clone https://github.com/lobennett/bold-reliability-movies
cd bold-reliability-movies
uv sync --dev          # uv flow
# or
pip install -e ".[dev]"  # pip flow
\`\`\`

## Quickstart

\`\`\`bash
# One video per subject, mosaic renderer, 2 fps
brm bids /path/to/fmriprep/derivatives --out movies/

# Filter to one task, group by subject+session
brm bids /path/to/fmriprep/derivatives --out movies/ \\
    --filter task=rest --group-by subject+session

# Custom inputs via TSV manifest
brm list manifest.tsv --out movies/

# Render an arbitrary list of NIfTIs into one video
brm render run1.nii.gz run2.nii.gz run3.nii.gz --out movie.mp4 \\
    --labels "ses-01 run-1" "ses-01 run-2" "ses-02 run-1"
\`\`\`

## Manifest format (`brm list`)

```
path	label	group	sort_key
/data/sub-01_ses-01_run-1.nii.gz	ses-01 run-1	sub-01	1
/data/sub-01_ses-01_run-2.nii.gz	ses-01 run-2	sub-01	2
```
- `path`, `label`, `group` required.
- `sort_key` optional; absent → row order preserved.
- Lines starting with `#` ignored.

## Custom renderers

A renderer is any callable matching the `Renderer` Protocol:

\`\`\`python
import nibabel as nib
import numpy as np
from bold_reliability_movies import Frame, FrameGroup, make_video

def my_renderer(mean_img: nib.Nifti1Image, label: str) -> np.ndarray:
    """Return (H, W, 3) uint8. H, W must be constant across calls."""
    data = mean_img.get_fdata()
    # ... your rendering ...
    return rgb_uint8

group = FrameGroup(
    name="sub-s03",
    frames=[Frame(path=Path("run1.nii.gz"), label="ses-01 run-1", sort_key=(1,))],
)
make_video(group, renderer=my_renderer, out_path="sub-s03.mp4", fps=2)
\`\`\`

To make your renderer visible to the `brm` CLI, register it in `src/bold_reliability_movies/renderers/__init__.py` and submit a PR. (External plugin loading via setuptools entry points will land when there's a real plugin author asking for it.)

## CLI exit codes

- `0` — success
- `1` — partial failure (some groups skipped or had dropped frames)
- `2` — misconfiguration (missing dep, bad arg, unknown renderer)
- `3` — no work found (empty discovery)

## Attribution

The "BOLD reliability movie" idea originates with Kendrick Kay (cvnlab), who showed per-session mean-BOLD videos at his Stanford talk and built the canonical version (an fsaverage flat-surface projection, MATLAB) inside the [NSD data paper](https://github.com/cvnlab/nsddatapaper) at `mainfigures/INSPECTIONS/GRANDVISUALIZATION/GRANDVISUALIZATIONnotes.m`. This package implements the same *concept* (cycle mean BOLD across runs for visual QC) in BIDS-Python form with a different default rendering. It does not reproduce his exact output.

## License

MIT. See `LICENSE`.

## Design

See `docs/superpowers/specs/2026-04-27-bold-reliability-movies-design.md` for the full architecture and decision record.
```

- [ ] **Step 15.2: Write `CHANGELOG.md`**

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-27

### Added
- Initial release.
- `Renderer` and `FrameSource` Protocols.
- In-tree renderers: `mosaic` (default), `triplet`.
- In-tree sources: `FmriprepFrameSource`, `ManifestFrameSource`.
- CLI: `brm bids|list|render` with named exit codes.
- On-disk mean cache with mtime invalidation.
- MP4 encoding via system ffmpeg.
```

- [ ] **Step 15.3: Write `CONTRIBUTING.md`**

```markdown
# Contributing

## Setup

\`\`\`bash
git clone https://github.com/lobennett/bold-reliability-movies
cd bold-reliability-movies
uv sync --dev
\`\`\`

## Tests

\`\`\`bash
uv run pytest -v                          # default suite (no slow/integration)
uv run pytest -m slow                     # real ffmpeg encode
uv run pytest -m integration              # requires real fmriprep derivatives
\`\`\`

## Lint and type-check

\`\`\`bash
uv run ruff check
uv run mypy src/
\`\`\`

## Adding a renderer

1. Implement a class with `__call__(mean_img, label) -> np.ndarray` in `src/bold_reliability_movies/renderers/<name>.py`.
2. Register it in `src/bold_reliability_movies/renderers/__init__.py`'s `REGISTRY`.
3. Add tests in `tests/test_renderers/test_<name>.py` covering: shape constancy across inputs, deterministic output, label changes pixels.
4. Update `CHANGELOG.md` under `[Unreleased]`.

## Adding a frame source

1. Implement a class with `discover() -> list[FrameGroup]` in `src/bold_reliability_movies/discovery/<name>.py`.
2. Re-export from `discovery/__init__.py`.
3. Add tests in `tests/test_discovery_<name>.py`.
4. If CLI-exposed, add a subcommand to `cli.py`.

## Commit style

Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`).
```

- [ ] **Step 15.4: Commit**

```bash
git add README.md CHANGELOG.md CONTRIBUTING.md
git commit -m "docs: README, CHANGELOG, CONTRIBUTING for v0.1.0

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: GitHub Actions CI

**Goal:** PR CI runs lint + type-check + the default test suite (no `slow`, no `integration`).

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 16.1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install ffmpeg
        run: sudo apt-get update && sudo apt-get install -y ffmpeg

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Sync deps
        run: uv sync --all-extras

      - name: Lint
        run: uv run ruff check

      - name: Type-check
        run: uv run mypy src/

      - name: Test
        run: uv run pytest -v
```

- [ ] **Step 16.2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: GitHub Actions — ruff, mypy, pytest on PR + main push

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 16.3: Final verification**

```bash
uv run pytest -v
uv run ruff check
uv run mypy src/
uv build
```

Expected:
- All tests pass.
- Ruff clean.
- Mypy clean.
- `dist/bold_reliability_movies-0.1.0-py3-none-any.whl` and `bold_reliability_movies-0.1.0.tar.gz` produced.

- [ ] **Step 16.4: Tag the v0.1.0 release locally**

```bash
git tag v0.1.0
git log --oneline | head -20
```

(Pushing to GitHub and publishing to PyPI are out-of-scope for this plan; do them by hand once you've created the GitHub repo and configured Trusted Publishing.)

---

## Self-review notes

- **Spec coverage:** every section of the spec maps to at least one task — types (T3), errors (T2), mean cache (T5), encode (T6), renderers (T7-T9), pipeline (T10), discovery fmriprep (T11), discovery manifest (T12), CLI (T13), public API (T14), packaging (T1), README/CHANGELOG/CONTRIBUTING (T15), CI (T16).
- **Out-of-spec items deferred to later versions:** entry-point plugin loading, parallelism, progress bars, surface renderer, integration tier in CI — explicitly called out as v1.1+ in the spec, no tasks here.
- **Type consistency:** `Renderer` signature `(mean_img: Nifti1Image, label: str) -> np.ndarray` used in T3 (definition), T7 (mosaic), T8 (triplet), T10 (pipeline), T13 (CLI). `FrameSource.discover() -> list[FrameGroup]` consistent across T11 and T12. `make_video` signature consistent T10/T13/T14.
