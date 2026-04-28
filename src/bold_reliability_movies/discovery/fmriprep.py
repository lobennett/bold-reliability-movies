"""FrameSource for fMRIPrep derivatives directories."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from bold_reliability_movies.types import Frame, FrameGroup

log = logging.getLogger(__name__)

# Match fMRIPrep preproc BOLD filenames; tolerant of optional space/echo/desc tokens.
_FILENAME_RE = re.compile(
    r"sub-(?P<sub>[A-Za-z0-9]+)_"
    r"ses-(?P<ses>[A-Za-z0-9]+)_"
    r"task-(?P<task>[A-Za-z0-9]+)_"
    r"run-(?P<run>\d+)_"
    r".*?desc-preproc_bold\.nii\.gz$"
)

GroupBy = Literal["subject", "subject+task", "subject+session", "none"]


@dataclass
class _Entities:
    sub: str
    ses: str
    task: str
    run: int

    @property
    def ses_num(self) -> int:
        m = re.search(r"\d+", self.ses)
        return int(m.group()) if m else 0


def _parse(name: str) -> _Entities | None:
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return _Entities(sub=m["sub"], ses=m["ses"], task=m["task"], run=int(m["run"]))


def _entities_match(ents: _Entities, filters: dict[str, str]) -> bool:
    for k, v in filters.items():
        if k == "sub" and ents.sub != v:
            return False
        if k == "ses" and ents.ses != v:
            return False
        if k == "task" and ents.task != v:
            return False
        if k == "run" and str(ents.run) != v:
            return False
    return True


def _group_key(ents: _Entities, mode: GroupBy) -> str:
    if mode == "subject":
        return f"sub-{ents.sub}"
    if mode == "subject+task":
        return f"sub-{ents.sub}_task-{ents.task}"
    if mode == "subject+session":
        return f"sub-{ents.sub}_ses-{ents.ses}"
    if mode == "none":
        return "all"
    raise ValueError(f"unknown group_by mode: {mode}")


@dataclass
class FmriprepFrameSource:
    """Walk a fMRIPrep-style derivatives directory and yield FrameGroups."""

    deriv_dir: Path
    group_by: GroupBy = "subject"
    filters: dict[str, str] = field(default_factory=dict)
    glob: str = "sub-*/ses-*/func/*_desc-preproc_bold.nii.gz"

    def discover(self) -> list[FrameGroup]:
        files = sorted(self.deriv_dir.glob(self.glob))
        buckets: dict[str, list[Frame]] = defaultdict(list)
        meta: dict[str, dict[str, str]] = defaultdict(dict)

        for fp in files:
            ents = _parse(fp.name)
            if ents is None:
                log.debug("skipping unparseable filename: %s", fp.name)
                continue
            if not _entities_match(ents, self.filters):
                continue
            key = _group_key(ents, self.group_by)
            label = f"ses-{ents.ses} task-{ents.task} run-{ents.run}"
            sort_key = (ents.sub, ents.ses_num, ents.task, ents.run)
            buckets[key].append(Frame(path=fp, label=label, sort_key=sort_key))
            meta[key].setdefault("subject", ents.sub)

        groups: list[FrameGroup] = []
        for key in sorted(buckets):
            frames = sorted(buckets[key], key=lambda f: f.sort_key)
            groups.append(FrameGroup(name=key, frames=frames, metadata=meta[key]))
        return groups
