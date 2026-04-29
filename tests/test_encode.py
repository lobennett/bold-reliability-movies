from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from bold_reliability_movies.encode import encode, probe_ffmpeg
from bold_reliability_movies.errors import EncodeError, MissingDependency


def test_probe_ffmpeg_returns_path(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    assert probe_ffmpeg() == "/usr/bin/ffmpeg"


def test_probe_ffmpeg_raises_when_missing(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(MissingDependency) as ei:
        probe_ffmpeg()
    assert "ffmpeg" in str(ei.value).lower()


def test_probe_ffmpeg_prefers_imageio_ffmpeg(monkeypatch) -> None:
    """When imageio_ffmpeg is importable, prefer its bundled binary."""
    import sys
    import types

    fake_module = types.ModuleType("imageio_ffmpeg")
    fake_module.get_ffmpeg_exe = lambda: "/fake/imageio/ffmpeg"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_module)

    # shutil.which would return something else, but we should never fall through
    monkeypatch.setattr("shutil.which", lambda name: "/system/ffmpeg")

    assert probe_ffmpeg() == "/fake/imageio/ffmpeg"


def test_probe_ffmpeg_falls_back_to_system(monkeypatch) -> None:
    """When imageio_ffmpeg is NOT installed, fall back to shutil.which."""
    import sys

    # Ensure imageio_ffmpeg import fails
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")

    assert probe_ffmpeg() == "/usr/bin/ffmpeg"


def test_probe_ffmpeg_missing_message_mentions_mp4_extra(monkeypatch) -> None:
    """Error message should hint at the [mp4] extra when nothing is found."""
    import sys

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(MissingDependency) as ei:
        probe_ffmpeg()
    msg = str(ei.value)
    assert "ffmpeg" in msg.lower()
    assert "[mp4]" in msg


def _frames(n: int = 3, h: int = 8, w: int = 8) -> list[np.ndarray]:
    return [np.full((h, w, 3), i * 30, dtype=np.uint8) for i in range(n)]


def test_encode_invokes_ffmpeg_with_correct_args(monkeypatch, tmp_path: Path) -> None:
    import sys

    captured: dict[str, object] = {}

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

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
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
    import sys

    def fake_popen(cmd, *args, **kwargs):
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.wait = MagicMock(return_value=1)
        proc.stderr = MagicMock()
        proc.stderr.read = MagicMock(return_value=b"ffmpeg failed")
        return proc

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")

    with pytest.raises(EncodeError) as ei:
        encode(_frames(), fps=2, out_path=tmp_path / "out.mp4")
    assert "ffmpeg failed" in str(ei.value)


def test_encode_rejects_empty_frame_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        encode([], fps=2, out_path=tmp_path / "out.mp4")


def test_encode_libx264_args(monkeypatch, tmp_path: Path) -> None:
    import sys

    captured: dict = {}

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.wait = MagicMock(return_value=0)
        proc.stderr = MagicMock()
        proc.stderr.read = MagicMock(return_value=b"")
        return proc

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    encode(_frames(), fps=2, out_path=tmp_path / "x.mp4", codec="libx264")
    cmd = captured["cmd"]
    assert "libx264" in cmd
    assert "-crf" in cmd
    assert "-pix_fmt" in cmd and "yuv420p" in cmd


def test_encode_mpeg4_args(monkeypatch, tmp_path: Path) -> None:
    import sys

    captured: dict = {}

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.wait = MagicMock(return_value=0)
        proc.stderr = MagicMock()
        proc.stderr.read = MagicMock(return_value=b"")
        return proc

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    encode(_frames(), fps=2, out_path=tmp_path / "x.mp4", codec="mpeg4")
    cmd = captured["cmd"]
    assert "mpeg4" in cmd
    assert "-q:v" in cmd
    # libx264-specific flags should NOT appear
    assert "libx264" not in cmd
    assert "-crf" not in cmd


def test_encode_gif_args(monkeypatch, tmp_path: Path) -> None:
    import sys

    captured: dict = {}

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.wait = MagicMock(return_value=0)
        proc.stderr = MagicMock()
        proc.stderr.read = MagicMock(return_value=b"")
        return proc

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    encode(_frames(), fps=2, out_path=tmp_path / "x.gif", codec="gif")
    cmd = captured["cmd"]
    assert "-loop" in cmd
    assert "0" in cmd
    # GIF should NOT have codec-specific video flags
    assert "libx264" not in cmd
    assert "-crf" not in cmd
    assert "-q:v" not in cmd


def test_encode_unknown_codec_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as ei:
        encode(_frames(), fps=2, out_path=tmp_path / "x.mp4", codec="vp9")
    assert "vp9" in str(ei.value)


@pytest.mark.slow
def test_encode_real_ffmpeg(tmp_path: Path) -> None:
    """End-to-end real ffmpeg encode. Run with: pytest -m slow"""
    out = tmp_path / "real.mp4"
    encode(_frames(n=5, h=32, w=32), fps=2, out_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
