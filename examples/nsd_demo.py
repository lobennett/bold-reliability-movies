"""NSD subj01 reliability demo: per-session mean BOLD across 40 sessions.

Uses Kendrick Kay's Natural Scenes Dataset (https://naturalscenesdataset.org).
The package is inspired by Kendrick's NSD inspection videos
(cvnlab/nsddatapaper, mainfigures/INSPECTIONS/GRANDVISUALIZATION); using NSD
data here lets readers see what those videos look like in our package's
volume-space mosaic rendering.

The 40 frames of the output animation each show a 5x5 grid of axial slices
through subj01's per-session mean BOLD (1.8mm), spanning ~1 year of
scanning. Watch for shifts in coverage, intensity inhomogeneity, and
between-session alignment.

## Data setup (one-time)

Download from the public NSD AWS bucket (~32 MB, no sign-in required):

    mkdir -p examples/.nsd_cache
    aws s3 cp --no-sign-request --recursive \\
        --exclude "*" --include "mean_session*.nii.gz" \\
        s3://natural-scenes-dataset/nsddata/ppdata/subj01/func1pt8mm/ \\
        examples/.nsd_cache/

Then run:

    uv run python examples/nsd_demo.py

The output `examples/output/nsd_subj01.mp4` is committed to the repo, so
end-users can see the animation directly without re-running this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bold_reliability_movies import (
    Frame,
    FrameGroup,
    MosaicRenderer,
    make_video,
)

DEFAULT_DATA_DIR = Path(__file__).parent / ".nsd_cache"
OUT_DIR = Path(__file__).parent / "output"


def collect_frames(data_dir: Path) -> list[Frame]:
    """Walk data_dir for mean_session##.nii.gz, return sorted Frames."""
    frames: list[Frame] = []
    for i in range(1, 41):
        p = data_dir / f"mean_session{i:02d}.nii.gz"
        if not p.exists():
            continue
        frames.append(
            Frame(path=p, label=f"NSD subj01  session {i:02d}", sort_key=(i,))
        )
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory holding mean_session##.nii.gz (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "nsd_subj01.mp4",
        help="Output MP4 path",
    )
    parser.add_argument("--codec", default="libx264", help="ffmpeg codec")
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    frames = collect_frames(args.data_dir)
    if not frames:
        print(
            f"\nNo mean_session##.nii.gz found in {args.data_dir}\n"
            "\nDownload first (~32 MB):\n"
            "  mkdir -p examples/.nsd_cache\n"
            "  aws s3 cp --no-sign-request --recursive \\\n"
            "      --exclude '*' --include 'mean_session*.nii.gz' \\\n"
            "      s3://natural-scenes-dataset/nsddata/ppdata/subj01/func1pt8mm/ \\\n"
            "      examples/.nsd_cache/\n",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    group = FrameGroup(name="nsd_subj01", frames=frames)
    print(f"Rendering {len(frames)} sessions → {args.out}")
    make_video(
        group,
        renderer=MosaicRenderer(),  # default 5x5 axial mosaic
        out_path=args.out,
        fps=args.fps,
        codec=args.codec,
        use_cache=False,  # NSD files are already 3D means; cache is wasted
    )
    print(f"Wrote {args.out} ({args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
