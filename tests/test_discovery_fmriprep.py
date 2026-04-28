from __future__ import annotations

from pathlib import Path

from bold_reliability_movies.discovery.fmriprep import FmriprepFrameSource


def test_discovery_groups_by_subject_default(fake_fmriprep_tree: Path):
    src = FmriprepFrameSource(deriv_dir=fake_fmriprep_tree)
    groups = src.discover()
    names = sorted(g.name for g in groups)
    assert names == ["sub-s01", "sub-s02"]


def test_discovery_sort_order_by_session_task_run(fake_fmriprep_tree: Path):
    src = FmriprepFrameSource(deriv_dir=fake_fmriprep_tree)
    groups = {g.name: g for g in src.discover()}
    s01 = groups["sub-s01"]
    labels = [f.label for f in s01.frames]
    assert labels == [
        "ses-01 task-rest run-1",
        "ses-01 task-rest run-2",
        "ses-02 task-stroop run-1",
    ]


def test_discovery_filter_by_task(fake_fmriprep_tree: Path):
    src = FmriprepFrameSource(deriv_dir=fake_fmriprep_tree, filters={"task": "rest"})
    groups = {g.name: g for g in src.discover()}
    assert all(f.label.split()[1] == "task-rest" for g in groups.values() for f in g.frames)


def test_discovery_filter_by_subject(fake_fmriprep_tree: Path):
    src = FmriprepFrameSource(deriv_dir=fake_fmriprep_tree, filters={"sub": "s01"})
    groups = src.discover()
    assert {g.name for g in groups} == {"sub-s01"}


def test_discovery_group_by_subject_and_task(fake_fmriprep_tree: Path):
    src = FmriprepFrameSource(deriv_dir=fake_fmriprep_tree, group_by="subject+task")
    groups = sorted(g.name for g in src.discover())
    assert groups == ["sub-s01_task-rest", "sub-s01_task-stroop", "sub-s02_task-rest"]


def test_discovery_group_by_none_returns_one_group(fake_fmriprep_tree: Path):
    src = FmriprepFrameSource(deriv_dir=fake_fmriprep_tree, group_by="none")
    groups = src.discover()
    assert len(groups) == 1
    assert groups[0].name == "all"
    assert len(groups[0].frames) == 5


def test_discovery_skips_unparseable_filenames(tmp_path: Path):
    d = tmp_path / "deriv" / "sub-s01" / "ses-01" / "func"
    d.mkdir(parents=True)
    (d / "garbage.nii.gz").write_bytes(b"")
    (d / "sub-s01_ses-01_task-rest_run-1_desc-preproc_bold.nii.gz").write_bytes(b"")
    src = FmriprepFrameSource(deriv_dir=tmp_path / "deriv")
    groups = src.discover()
    # 1 group, 1 frame; the garbage file is silently skipped.
    assert len(groups) == 1
    assert len(groups[0].frames) == 1
