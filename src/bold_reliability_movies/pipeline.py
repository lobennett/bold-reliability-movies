"""Orchestration: turn FrameGroups into MP4 files."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy.typing as npt

from bold_reliability_movies.encode import encode
from bold_reliability_movies.errors import BrmError, GroupRejectedError, InconsistentShapesError
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
) -> tuple[list[npt.NDArray[Any]], list[Frame]]:
    """Compute means and render. Returns (rgb_frames, kept_frames). Bad frames dropped."""
    rgb: list[npt.NDArray[Any]] = []
    kept: list[Frame] = []
    for frame in frames:
        try:
            mean_img = compute_mean(frame.path, cache_dir=cache_dir, use_cache=use_cache)
        except Exception as exc:
            log.warning(
                "dropping frame %s (%s): mean computation failed: %s",
                frame.label,
                frame.path,
                exc,
            )
            continue
        try:
            arr = renderer(mean_img, frame.label)
        except Exception as exc:
            log.warning("dropping frame %s: renderer raised: %s", frame.label, exc)
            continue
        rgb.append(arr)
        kept.append(frame)
    return rgb, kept


def _check_shapes(rgb: Sequence[npt.NDArray[Any]]) -> None:
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
    codec: str = "libx264",
) -> None:
    """Render a single FrameGroup to one MP4."""
    sorted_frames = sorted(group.frames, key=lambda f: f.sort_key)
    rgb, kept = _render_frames(sorted_frames, renderer, cache_dir, use_cache)
    n_total = len(group.frames)
    n_kept = len(kept)
    if n_total > 0 and (n_total - n_kept) / n_total > _DROP_THRESHOLD:
        raise GroupRejectedError(
            f"group {group.name!r}: {n_total - n_kept}/{n_total} frames failed "
            f"(threshold {_DROP_THRESHOLD})"
        )
    if n_kept < _MIN_FRAMES:
        raise GroupRejectedError(
            f"group {group.name!r}: only {n_kept} usable frame(s); need >= {_MIN_FRAMES}"
        )

    _check_shapes(rgb)
    log.info("encoding group %s (%d frames) → %s", group.name, n_kept, out_path)
    encode(rgb, fps=fps, out_path=out_path, codec=codec)


def make_videos(
    groups: Sequence[FrameGroup],
    *,
    renderer: Renderer,
    out_dir: Path,
    fps: int,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    codec: str = "libx264",
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
                codec=codec,
            )
        except (BrmError, OSError) as exc:
            log.error("group %s failed: %s", group.name, exc, exc_info=True)
            summary.failed.append(group.name)
        else:
            summary.succeeded.append(group.name)
    return summary
