"""Synthetic demo: Gaussian blob drifting across runs → GIF via bold-reliability-movies."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from bold_reliability_movies import Frame, FrameGroup, MosaicRenderer, make_video

OUT_DIR = Path(__file__).parent / "output"
TMP_DIR = Path(__file__).parent / ".tmp_synth"


def write_synthetic_bold(
    path: Path, run_idx: int, shape: tuple[int, int, int, int] = (32, 32, 16, 5)
) -> None:
    """4D BOLD with a Gaussian blob that drifts position across runs."""
    nx, ny, nz, nt = shape
    x = np.arange(nx)[:, None, None]
    y = np.arange(ny)[None, :, None]
    z = np.arange(nz)[None, None, :]
    cx, cy, cz = nx // 2 + run_idx, ny // 2 + run_idx, nz // 2  # blob drifts
    sigma = 4.0
    blob = np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * sigma ** 2))
    rng = np.random.default_rng(seed=run_idx + 42)
    data = (blob[..., None] * 200 + rng.normal(50, 5, size=shape)).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(10):
        bold = TMP_DIR / f"run-{i:02d}_bold.nii.gz"
        write_synthetic_bold(bold, run_idx=i - 5)  # blob drifts -5..+4
        frames.append(Frame(path=bold, label=f"synthetic run-{i + 1:02d}", sort_key=(i,)))
    group = FrameGroup(name="synthetic_demo", frames=frames)
    out_path = OUT_DIR / "synthetic_demo.gif"
    make_video(
        group,
        renderer=MosaicRenderer(n_rows=4, n_cols=4, fig_size=(4, 4), dpi=80),
        out_path=out_path,
        fps=2,
        codec="gif",
        use_cache=False,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
