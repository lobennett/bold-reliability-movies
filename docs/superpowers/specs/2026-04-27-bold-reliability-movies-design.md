# bold-reliability-movies — Design Spec

**Date:** 2026-04-27
**Status:** Approved, ready for implementation planning
**Author:** Logan Bennett (logben@stanford.edu)

## 1. Purpose & framing

A small Python package that produces "BOLD reliability movies" — videos that
cycle through the mean image of every BOLD run for a subject, so a human can
visually inspect coverage, dropout, and across-run stability at a glance.

The idea is in the spirit of Kendrick Kay's NSD inspection videos
(`cvnlab/nsddatapaper/mainfigures/INSPECTIONS/GRANDVISUALIZATION`), which
render fsaverage-flatmap projections of per-session means and encode them at
30 fps. This package is **not** a port of that code. It is a BIDS-aware
Python implementation that ships a volume-space mosaic renderer by default
and exposes a `Renderer` Protocol so a flatmap (or any other) backend can be
written by a third party. Attribution to Kendrick's work is explicit in the
README; the package does not claim to reproduce his exact output.

The first audience is the cvnlab group, who can use it as-is on their data
or write their own renderer against the Protocol. A secondary path is
upstreaming a renderer or pipeline component into fMRIPrep at some future
point. Neither is a v1 requirement.

## 2. Scope

**In scope (v1):**
- Standalone Python package `bold-reliability-movies` (PyPI dist) /
  `bold_reliability_movies` (import name), MIT-licensed.
- BIDS-derivatives (fMRIPrep) and TSV-manifest input adapters.
- One in-tree renderer: axial mosaic (default). One additional in-tree
  renderer: axial+sagittal+coronal triplet (parity with the existing
  `neuro_workflow/qa/reliability.py`).
- MP4 output via ffmpeg. One video per group; default grouping = subject.
- CLI with `bids`, `list`, `render` subcommands.
- Renderer + FrameSource Protocols; library users can pass their own.
- On-disk mean cache (default on, `--no-cache` to disable, `--cache-dir`
  to relocate).
- Synthetic-fixture-based test suite, GitHub Actions CI.

**Out of scope (v1):**
- Surface / flatmap rendering (interface only; no implementation).
- Setuptools entry-point plugin loading. Will be added the first time an
  external author needs CLI-visible third-party renderers.
- Parallelism (`--jobs`).
- Progress bars / tqdm.
- Standard-space outputs other than what fMRIPrep already provides
  (the package consumes whatever volumes the source layer hands it).
- Real-data integration tests in CI (gated behind `pytest -m integration`).

## 3. Architecture

Three layers, communicating through plain frozen dataclasses.

```
┌─────────────────────────────────────────────────────────┐
│  CLI  (argparse subcommands: bids / list / render)      │
└──────────────────────┬──────────────────────────────────┘
                       │ list[FrameGroup]
┌──────────────────────▼──────────────────────────────────┐
│  Discovery layer       │ FrameSource Protocol           │
│  ─ fmriprep_derivatives│ (Path | TSV) → list[FrameGroup]│
│  ─ manifest_tsv        │                                │
└──────────────────────┬──────────────────────────────────┘
                       │ list[FrameGroup]
┌──────────────────────▼──────────────────────────────────┐
│  Pipeline (pure orchestration; no I/O policy)            │
│   1. compute_mean(nifti) → Nifti1Image  (cached)         │
│   2. renderer(mean_img, label) → np.ndarray (H,W,3)      │
│   3. encode(frames, fps, out_path) → MP4                 │
└──────────────────────┬──────────────────────────────────┘
                       │ Renderer Protocol
┌──────────────────────▼──────────────────────────────────┐
│  Renderers           │ mosaic (in-tree, default)        │
│                      │ triplet (in-tree)                │
│                      │ <user-written, library-only>     │
└─────────────────────────────────────────────────────────┘
```

### Module layout

```
src/bold_reliability_movies/
  __init__.py          # public re-exports: make_video, Renderer, FrameSource, FrameGroup, Frame
  types.py             # FrameGroup, Frame dataclasses; Renderer + FrameSource Protocols
  errors.py            # MissingDependency, InconsistentShapesError, etc.
  pipeline.py          # make_video(frames, renderer, out_path, fps) — only orchestration entry
  mean_cache.py        # compute_mean() with on-disk cache keyed by NIfTI path + mtime
  encode.py            # ffmpeg wrapper (matplotlib FuncAnimation + writer)
  discovery/
    __init__.py
    fmriprep.py        # FrameSource for fmriprep derivatives dir
    manifest.py        # FrameSource for TSV manifests
  renderers/
    __init__.py        # in-tree dispatch table: name → Renderer instance
    mosaic.py          # default
    triplet.py         # axial+sagittal+coronal
  cli.py               # argparse, subcommand dispatch
```

### Design rationale

- Discovery, rendering, encoding never share state; each can be tested in
  isolation with synthetic fixtures.
- The pipeline is unaware of where frames came from; the discovery layer is
  unaware of how frames will be rendered. This is the boundary that makes
  adding a flatmap renderer or a raw-BIDS source a pure addition.
- `FrameGroup` is the package's lingua franca — two fields (`name`,
  `frames`) plus opaque `metadata`. Anything that produces a list of these
  can drive the pipeline.
- `mean_cache.py` exists because mean computation is the expensive step and
  re-running on the same data should be free.

## 4. Data flow & types

### Core types (`types.py`)

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import nibabel as nib
import numpy as np

@dataclass(frozen=True)
class Frame:
    path: Path           # NIfTI to summarize (one frame in the output video)
    label: str           # text overlay, e.g. "ses-03 task-stroop run-2"
    sort_key: tuple      # used to order frames within a group; opaque to renderer

@dataclass(frozen=True)
class FrameGroup:
    name: str            # output stem, e.g. "sub-s03"  →  sub-s03.mp4
    frames: list[Frame]
    metadata: dict = field(default_factory=dict)  # subject, task, etc.; for logs/downstream

class FrameSource(Protocol):
    def discover(self) -> list[FrameGroup]: ...

class Renderer(Protocol):
    def __call__(self, mean_img: nib.Nifti1Image, label: str) -> np.ndarray:
        """Return an (H, W, 3) uint8 RGB image. H, W must be constant across calls."""
```

`Frame` and `FrameGroup` are frozen dataclasses (cheap to hash, safe to pass
between layers, easy to print). `Renderer` is a callable Protocol so a plain
function works (no class boilerplate) but classes with state — e.g., a
flatmap with a baked-in lookup table — also fit.

### End-to-end trace

For `bold-reliability-movies bids /path/to/derivatives --renderer mosaic --fps 2 --out movies/`:

1. **CLI parse:** resolves subcommand `bids`, instantiates
   `FmriprepFrameSource(deriv_dir, filters=...)`, looks up renderer name
   `"mosaic"` in the in-tree dispatch table → `MosaicRenderer()`.
2. **Discovery:** `source.discover()` walks
   `sub-*/ses-*/func/*_desc-preproc_bold.nii.gz`, parses BIDS entities,
   groups by subject (default), sorts each group's frames by
   `(ses_num, task, run)`. Returns `list[FrameGroup]`. Files where entities
   can't be parsed are skipped and logged.
3. **Per group**, the pipeline runs:
   - For each `Frame`: `mean_cache.compute_mean(frame.path)` →
     `Nifti1Image`. Cache check: if `<frame.path>.mean.nii.gz` exists and
     `mtime >= frame.path.mtime`, load it; else compute with
     `nilearn.image.mean_img`, save atomically (temp file + rename), return.
   - For each mean image: `rgb = renderer(mean_img, frame.label)` →
     `(H, W, 3) uint8`.
   - `encode(rgb_frames, fps, out / f"{group.name}.mp4")` — matplotlib
     `FuncAnimation` with the `ffmpeg` writer, `libx264`, `yuv420p`
     (browser-safe).
4. **Failure handling per group:** any exception inside a group is caught,
   logged with the group name and offending frame; that group is skipped;
   the loop continues. Exit code = 1 if any group failed, 0 otherwise.

### Constant-shape requirement

The renderer must produce constant-shape frames within a group. If a frame's
mean image has a different shape than the previous (e.g., one run was
acquired at different resolution), the mosaic renderer will produce
different pixel dimensions, and ffmpeg will refuse to encode. The pipeline
detects this **before** encoding and raises `InconsistentShapesError` with a
clear message naming the offending frame and suggesting
`--group-by subject+task` to separate. Padding silently to the max H,W
would hide real problems and is rejected.

### Mean cache

On by default. Cache files (~10 MB each, one 3D NIfTI per BOLD run) live
next to the source NIfTI (`<bold>.mean.nii.gz`). Disable with `--no-cache`,
relocate with `--cache-dir DIR` (useful when source is on read-only storage
like `/oak`). Cache key: source path + mtime. Atomic writes (temp file +
rename). Cache write failures are non-fatal — log a warning and proceed in
memory for that frame.

### Output organization

Default `--group-by subject`: one video per subject, frames ordered
`(session_num, task, run)`. Other modes:

- `--group-by subject+task`
- `--group-by subject+session`
- `--group-by none` — one video per group, with all frames concatenated
  (cross-subject grand visualization, the spirit of GRANDVISUALIZATION).

## 5. CLI

```
brm bids <bids_or_derivatives_dir> [--out DIR] [--fps N]
        [--renderer NAME] [--group-by MODE] [--filter ENTITY=VALUE...]
        [--cache | --no-cache] [--cache-dir DIR] [--verbose | --quiet]

brm list <manifest.tsv> [--out DIR] [--fps N] [--renderer NAME]
        [--cache | --no-cache] [--cache-dir DIR]

brm render <nifti> [<nifti> ...] [--out FILE] [--labels LABEL ...]
        [--renderer NAME] [--fps N]
```

The console-script entry exposes both `bold-reliability-movies` and `brm` as
the alias.

### Manifest TSV format (for `brm list`)

Columns: `path` (required, absolute or repo-relative), `label` (required,
human-readable), `group` (required, output filename stem), `sort_key`
(optional; if absent, row order is preserved). Lines starting with `#` are
comments. The BIDS adapter compiles internally to the same shape, so adding
a non-BIDS source is "write a TSV."

### Defaults

- `--fps 2`
- `--renderer mosaic`
- `--group-by subject`
- `--cache` (on)

## 6. Error handling

| Failure | Where | Behavior |
|---|---|---|
| Missing ffmpeg | `encode.py` startup probe | `MissingDependency` with install hints (`apt`, `brew`, `conda`, `module load`). Exit 2. Probed once per CLI run. |
| Unreadable / corrupt NIfTI | `mean_cache.compute_mean` | Log, drop frame from group. If group has <2 frames after, skip group. |
| Frame shape mismatch within group | Pipeline pre-encode | `InconsistentShapesError` with frame index, both shapes, suggestion to use `--group-by subject+task`. Group skipped. |
| Renderer raises | Pipeline | Caught per-frame. Log. If >50% frames fail, skip group; else drop bad frames and proceed. |
| Empty discovery | After `source.discover()` | Exit 3. Echo filter parameters. |
| Output dir missing | Pre-discovery | Auto-create (`mkdir parents=True exist_ok=True`). |
| Cache write fails | `mean_cache` | Warn, fall back to in-memory. Non-fatal. |
| Unknown renderer name | CLI dispatch | Exit 2. Print available renderers. |

### Exit codes

- 0: full success
- 1: partial failure (some groups skipped or had dropped frames)
- 2: misconfiguration (missing dep, bad arg, unknown renderer)
- 3: no work found (empty discovery)

### Logging

Python `logging` module. `INFO` default, `--verbose` for `DEBUG`, `--quiet`
for `WARNING`. One line per significant event (group start/complete, skipped
frame, skipped group). No progress bars.

### Determinism

Discovery + encoding produce byte-identical output for identical input. No
timestamps in filenames, no random seeds, no parallelism in v1.
`sha256sum movies/*.mp4` is a meaningful regression check.

## 7. Plugin contract

`Renderer` and `FrameSource` are `typing.Protocol` types — duck-typed,
no inheritance required. Library users pass their own:

```python
from bold_reliability_movies import make_video, FrameGroup, Frame

def my_renderer(mean_img, label):
    # ... returns (H, W, 3) uint8
    return rgb

make_video(
    group=FrameGroup(name="sub-s03", frames=[Frame(path, "ses-01 run-1", (1,))]),
    renderer=my_renderer,
    out_path=Path("sub-s03.mp4"),
    fps=2,
)
```

The CLI exposes only renderers in the in-tree dispatch table
(`renderers/__init__.py`). Adding a CLI-visible renderer in v1 means a PR
that registers it in that table.

### Promotion path

The first time an external author asks for CLI-visible third-party
renderers without forking, add a setuptools entry-point lookup in
`renderers/__init__.py`:

```python
import importlib.metadata
for ep in importlib.metadata.entry_points(group="bold_reliability_movies.renderers"):
    REGISTRY[ep.name] = ep.load()()
```

The Protocol stays the same. In-tree renderers stay registered the same
way. No breaking change for library users.

## 8. Testing

### Layout

```
tests/
  conftest.py                  # fixtures: tiny synthetic NIfTI, fake BIDS tree, stub renderer
  test_types.py                # dataclass behavior, Protocol shape via runtime_checkable
  test_mean_cache.py           # compute, hit/miss, mtime invalidation, atomic write
  test_discovery_fmriprep.py   # walks fake BIDS tree; grouping/sorting
  test_discovery_manifest.py   # TSV parsing, error rows
  test_renderers/
    test_mosaic.py             # constant shape, label drawn, deterministic bytes
    test_triplet.py            # same contract
  test_pipeline.py             # orchestration with stubs; shape-mismatch handling
  test_encode.py               # ffmpeg probe; monkeypatch the writer to skip real encode
  test_cli.py                  # argparse, subcommand dispatch, exit codes
```

### Key fixtures

- `tiny_bold(tmp_path)` — 4D NIfTI of shape `(8, 8, 4, 5)` with a known
  signal pattern, ~1 KB. Anywhere a test needs "a BOLD file."
- `fake_fmriprep_tree(tmp_path)` — builds
  `sub-s01/ses-01/func/sub-s01_ses-01_task-rest_run-1_desc-preproc_bold.nii.gz`
  etc. with `tiny_bold` files. Parametrized over single-session,
  multi-session, multi-task, missing-run cases.
- `stub_renderer` — returns a constant `(64, 64, 3)` array with the label
  hashed into a corner pixel. Lets pipeline tests assert "renderer was
  called with these labels in this order" without rendering pixels.

### Real vs. mocked

- **Real:** dataclass logic, BIDS entity parsing, sorting, mean computation
  on synthetic data, cache hit/miss/invalidation, shape-mismatch detection,
  CLI argparse, exit codes.
- **Mocked:** ffmpeg subprocess (`monkeypatch` on the matplotlib writer).
  Tests assert `encode` *would* call ffmpeg with the right args; real
  encoding is one `@pytest.mark.slow` test, skipped by default. Run with
  `pytest -m slow` before releases.
- **Skipped in CI:** real fmriprep / FreeSurfer derivatives. Live in
  `tests/integration/`, gated by `pytest -m integration`. Lands in v1.1
  with a hand-curated 1-subject 1-session fixture.

### Property-style checks

- For every renderer in the registry,
  `render(img_a, "a").shape == render(img_b, "b").shape` for any
  two compatible inputs. Parametrized over all in-tree renderers.
- Determinism: `render(img, label)` returns byte-identical arrays across
  two calls.

### Coverage target

90%+ on `pipeline.py`, `mean_cache.py`, `types.py`, `discovery/*`.
Renderers and `encode.py` get less; smoke tests + determinism check are
sufficient.

### CI

GitHub Actions, single matrix (Ubuntu, Python 3.11) for v1.

```yaml
- uv sync --dev
- uv run pytest -m "not slow and not integration"
- uv run ruff check
- uv run mypy src/
```

Add 3.10 / 3.12 / macOS once the package stabilizes.

## 9. Packaging & distribution

### Scaffolding

```bash
uv init --package --name bold-reliability-movies --lib bold_reliability_movies
cd bold-reliability-movies
uv add nibabel nilearn matplotlib numpy
uv add --dev pytest pytest-cov ruff mypy
```

### `pyproject.toml`

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
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
    "License :: OSI Approved :: MIT License",
]

[project.urls]
Homepage = "https://github.com/<you>/bold-reliability-movies"
Source = "https://github.com/<you>/bold-reliability-movies"

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

[tool.mypy]
strict = true
```

### System dependency

`ffmpeg` on `PATH`. Documented in README; CLI startup probe gives a clear
error if missing. Not pip-installable.

### README install section (three audiences)

```markdown
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

### System dependency
`ffmpeg` must be on `PATH`. Install with:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Conda: `conda install -c conda-forge ffmpeg`
- HPC modules: `module load ffmpeg`

### From source (development)
\`\`\`bash
git clone https://github.com/<you>/bold-reliability-movies
cd bold-reliability-movies
uv sync --dev          # uv flow
# or
pip install -e ".[dev]"  # pip flow
\`\`\`
```

### Release flow

- SemVer. v0.x = "API may shift"; v1.0 once an external user has used it.
- Build: `uv build` (wheel + sdist in `dist/`).
- Publish: `uv publish` (PyPI). Trusted publishing via GitHub Actions OIDC,
  no API tokens checked in.
- Tag-driven: pushing `v0.1.0` triggers build/test/publish.

### LICENSE

MIT. Compatible with cvnlab's likely licensing; broadest scientific reuse.

### Repo hygiene

- `.github/workflows/ci.yml` — test + lint on PR.
- `.github/workflows/release.yml` — build + publish on tag.
- `CHANGELOG.md` — keep-a-changelog format.
- `CONTRIBUTING.md` — short, points at writing-a-renderer doc.
- `README.md` — two-sentence framing, attribution paragraph naming
  Kendrick's NSD videos as inspiration without claiming reproduction,
  60-second quickstart, custom-renderer example, pointer to
  `docs/cli.md` and `docs/renderers.md`.

## 10. Relationship to existing code

`neuro_workflow/src/neuro_workflow/qa/reliability.py` (the seed for this
package) stays in place untouched until the new package is feature-equal.
Then `neuro_workflow.qa` switches to depending on
`bold-reliability-movies` and the in-tree implementation is removed. This
is a v1.1 task, not v1.

The existing `reliability.py` filter
`*space-T1w_desc-preproc_bold.nii.gz` matches zero files in the current
`fmriprep_25.2.4` derivatives (which are boldref-space only). The new
package's `FmriprepFrameSource` will default to matching
`desc-preproc_bold.nii.gz` regardless of `space-` token, with `--filter`
flags to narrow.

## 11. Non-goals (re-statement)

- This package does not reproduce Kendrick Kay's flatmap videos. It is
  inspired by them and shares the *concept* (cycle mean BOLD across runs
  for visual QC). Attribution is explicit. A future flatmap renderer is
  possible as a third-party `Renderer` Protocol implementation; that is
  not v1's job.
- This package does not run fMRIPrep, recon-all, or any preprocessing.
  It consumes outputs.
- This package does not fix surface outputs missing from the user's
  current `fmriprep_25.2.4` derivatives. That is a separate concern in
  the upstream `neuro_workflow` pipeline.
