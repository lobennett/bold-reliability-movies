"""MP4 encoder: pipes raw RGB frames into a system ffmpeg subprocess."""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

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
    frames: Sequence[npt.NDArray[Any]],
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
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if proc.stdin is None:
        raise RuntimeError("subprocess stdin not piped")
    try:
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise EncodeError(f"ffmpeg exited with rc={rc}: {err}")
