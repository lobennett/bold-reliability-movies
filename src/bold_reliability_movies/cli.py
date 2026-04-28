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
    UnknownRendererError,
)
from bold_reliability_movies.pipeline import make_video, make_videos
from bold_reliability_movies.renderers import get_renderer, list_renderers
from bold_reliability_movies.types import Frame, FrameGroup, Renderer

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
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", force=True)


def _get_renderer_or_print(name: str) -> Renderer | None:
    try:
        return get_renderer(name)
    except UnknownRendererError as exc:
        print(str(exc), file=sys.stderr)
        return None


def _cmd_bids(args: argparse.Namespace) -> int:
    renderer = _get_renderer_or_print(args.renderer)
    if renderer is None:
        return 2
    filters = _parse_filters(args.filter)
    src = FmriprepFrameSource(
        deriv_dir=args.deriv_dir,
        group_by=args.group_by,
        filters=filters,
    )
    groups = src.discover()
    if not groups:
        print(
            f"No matching BOLD files in {args.deriv_dir} (filters={filters}).",
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
    renderer = _get_renderer_or_print(args.renderer)
    if renderer is None:
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
    renderer = _get_renderer_or_print(args.renderer)
    if renderer is None:
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
        Frame(path=p, label=lbl, sort_key=(i,))
        for i, (p, lbl) in enumerate(zip(paths, labels, strict=True))
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
