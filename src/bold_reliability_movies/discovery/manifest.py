"""FrameSource for tab-separated manifest files."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bold_reliability_movies.types import Frame, FrameGroup

_REQUIRED = ("path", "label", "group")


def _parse_sort_key(raw: str) -> tuple[Any, ...]:
    """Parse sort_key column. Numeric tokens become ints; strings stay strings."""
    parts = [p.strip() for p in raw.replace(",", "\t").split("\t") if p.strip()]
    out: list[Any] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            try:
                out.append(float(p))
            except ValueError:
                out.append(p)
    return tuple(out)


@dataclass
class ManifestFrameSource:
    """Read a TSV manifest. Required columns: path, label, group. Optional: sort_key."""

    tsv_path: Path

    def discover(self) -> list[FrameGroup]:
        with open(self.tsv_path, encoding="utf-8") as f:
            lines = [ln for ln in f if not ln.lstrip().startswith("#")]
        reader = csv.DictReader(lines, delimiter="\t")
        if reader.fieldnames is None:
            return []
        missing = [c for c in _REQUIRED if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"manifest {self.tsv_path} missing required column(s): {', '.join(missing)}"
            )

        buckets: dict[str, list[Frame]] = defaultdict(list)
        per_group_index: dict[str, int] = defaultdict(int)
        for row in reader:
            group = row["group"]
            idx = per_group_index[group]
            per_group_index[group] += 1
            sort_key: tuple[Any, ...]
            if "sort_key" in row and row["sort_key"]:
                sort_key = _parse_sort_key(row["sort_key"])
            else:
                sort_key = (idx,)
            buckets[group].append(
                Frame(path=Path(row["path"]), label=row["label"], sort_key=sort_key)
            )

        groups: list[FrameGroup] = []
        for name in sorted(buckets):
            frames = sorted(buckets[name], key=lambda f: f.sort_key)
            groups.append(FrameGroup(name=name, frames=frames))
        return groups
