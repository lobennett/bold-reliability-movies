# Contributing

## Setup

```bash
git clone https://github.com/lobennett/bold-reliability-movies
cd bold-reliability-movies
uv sync --dev
```

## Tests

```bash
uv run pytest -v                          # default suite (no slow/integration)
uv run pytest -m slow                     # real ffmpeg encode
uv run pytest -m integration              # requires real fmriprep derivatives
```

## Lint and type-check

```bash
uv run ruff check
uv run mypy src/
```

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
