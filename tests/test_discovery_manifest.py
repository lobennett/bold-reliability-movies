from __future__ import annotations

from pathlib import Path

import pytest

from bold_reliability_movies.discovery.manifest import ManifestFrameSource


def _write(tsv: Path, body: str) -> Path:
    tsv.write_text(body)
    return tsv


def test_manifest_groups_rows(tmp_path: Path):
    body = (
        "path\tlabel\tgroup\n"
        "/x/a.nii.gz\tses-01 r1\tsub-01\n"
        "/x/b.nii.gz\tses-01 r2\tsub-01\n"
        "/x/c.nii.gz\tses-01 r1\tsub-02\n"
    )
    tsv = _write(tmp_path / "m.tsv", body)
    groups = ManifestFrameSource(tsv_path=tsv).discover()
    names = sorted(g.name for g in groups)
    assert names == ["sub-01", "sub-02"]
    sub01 = next(g for g in groups if g.name == "sub-01")
    assert [f.label for f in sub01.frames] == ["ses-01 r1", "ses-01 r2"]


def test_manifest_skips_comment_lines(tmp_path: Path):
    body = (
        "# comment\n"
        "path\tlabel\tgroup\n"
        "# another comment\n"
        "/x/a.nii.gz\tlbl\tg1\n"
    )
    tsv = _write(tmp_path / "m.tsv", body)
    groups = ManifestFrameSource(tsv_path=tsv).discover()
    assert len(groups) == 1
    assert groups[0].frames[0].label == "lbl"


def test_manifest_default_sort_key_preserves_row_order(tmp_path: Path):
    body = (
        "path\tlabel\tgroup\n"
        "/x/c.nii.gz\tc\tg\n"
        "/x/a.nii.gz\ta\tg\n"
        "/x/b.nii.gz\tb\tg\n"
    )
    tsv = _write(tmp_path / "m.tsv", body)
    groups = ManifestFrameSource(tsv_path=tsv).discover()
    assert [f.label for f in groups[0].frames] == ["c", "a", "b"]


def test_manifest_explicit_sort_key(tmp_path: Path):
    body = (
        "path\tlabel\tgroup\tsort_key\n"
        "/x/c.nii.gz\tc\tg\t30\n"
        "/x/a.nii.gz\ta\tg\t10\n"
        "/x/b.nii.gz\tb\tg\t20\n"
    )
    tsv = _write(tmp_path / "m.tsv", body)
    groups = ManifestFrameSource(tsv_path=tsv).discover()
    assert [f.label for f in groups[0].frames] == ["a", "b", "c"]


def test_manifest_missing_required_column_errors(tmp_path: Path):
    body = "path\tlabel\n/x/a.nii.gz\tlbl\n"
    tsv = _write(tmp_path / "m.tsv", body)
    with pytest.raises(ValueError) as ei:
        ManifestFrameSource(tsv_path=tsv).discover()
    assert "group" in str(ei.value).lower()
