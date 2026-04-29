from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

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
        frames=[
            _frame(make_bold, "ok1.nii.gz", "ok1"),
            _frame(make_bold, "ok2.nii.gz", "ok2"),
        ],
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


def test_make_videos_gif_codec_writes_gif_suffix(make_bold, stub_renderer, tmp_path):
    group = FrameGroup(
        name="g",
        frames=[
            Frame(path=make_bold("a.nii.gz"), label="a", sort_key=(1,)),
            Frame(path=make_bold("b.nii.gz"), label="b", sort_key=(2,)),
        ],
    )
    summary = make_videos(
        [group], renderer=stub_renderer, out_dir=tmp_path, fps=2, codec="gif"
    )
    assert summary.succeeded == ["g"]


def test_make_video_sorts_frames_by_sort_key(make_bold, tmp_path):
    calls: list[str] = []

    def renderer(img, label):
        calls.append(label)
        return np.zeros((16, 16, 3), dtype=np.uint8)

    # Frames inserted in REVERSE sort_key order
    group = FrameGroup(
        name="g",
        frames=[
            Frame(path=make_bold("c.nii.gz"), label="c", sort_key=(3,)),
            Frame(path=make_bold("a.nii.gz"), label="a", sort_key=(1,)),
            Frame(path=make_bold("b.nii.gz"), label="b", sort_key=(2,)),
        ],
    )
    make_video(group, renderer=renderer, out_path=tmp_path / "g.mp4", fps=2)
    assert calls == ["a", "b", "c"]
