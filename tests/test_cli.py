from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bold_reliability_movies.cli import main


@pytest.fixture(autouse=True)
def _mock_ffmpeg(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
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


def test_cli_render_with_mpeg4_codec(tmp_path: Path, make_bold):
    a = make_bold("a.nii.gz")
    b = make_bold("b.nii.gz")
    out = tmp_path / "render.mp4"
    rc = main(["render", str(a), str(b), "--out", str(out), "--no-cache", "--codec", "mpeg4"])
    assert rc == 0


def test_cli_render_gif_codec(tmp_path: Path, make_bold):
    a = make_bold("a.nii.gz")
    b = make_bold("b.nii.gz")
    out = tmp_path / "render.gif"
    rc = main(["render", str(a), str(b), "--out", str(out), "--no-cache", "--codec", "gif"])
    assert rc == 0
