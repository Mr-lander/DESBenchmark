"""Tests for Taillard and FJSPLib-compatible instance loading."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import get_instance_families, load_fjsp, load_instance


FLEX_ROOT = PROJECT_ROOT / "external" / "benchmarks" / "flexible jobshop" / "instances"


@pytest.mark.parametrize(
    ("kind", "path"),
    [
        ("brandimarte", FLEX_ROOT / "Brandimarte1993" / "mk01.txt"),
        ("hurink", FLEX_ROOT / "HurinkJurischThole1994" / "edata" / "la01.txt"),
        ("dauzere", FLEX_ROOT / "DauzerePeresPaulli1994" / "01a.txt"),
        ("chambers_barnes", FLEX_ROOT / "ChambersBarnes1996" / "mt10c1.txt"),
        ("kacem", FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt"),
        ("fattahi", FLEX_ROOT / "FattahiMehrabadJolai2007" / "fattahi1.txt"),
        ("behnke", FLEX_ROOT / "BehnkeGeiger2012" / "behnke1.txt"),
    ],
)
def test_load_instance_supports_multiple_fjsp_families(kind: str, path: Path) -> None:
    instance = load_instance(path, kind)

    assert instance.n_jobs > 0
    assert instance.n_machines > 0
    assert len(instance.jobs) == instance.n_jobs
    assert all(job.operations for job in instance.jobs)
    assert all(op.options for job in instance.jobs for op in job.operations)


def test_load_fjsp_handles_float_average_flexibility_header() -> None:
    path = FLEX_ROOT / "HurinkJurischThole1994" / "edata" / "la01.txt"
    instance = load_fjsp(path)

    assert instance.n_jobs == 10
    assert instance.n_machines == 5


def test_instance_catalog_exposes_expected_families() -> None:
    families = {family.key: family for family in get_instance_families()}

    expected = {
        "taillard",
        "brandimarte",
        "hurink",
        "dauzere",
        "chambers_barnes",
        "kacem",
        "fattahi",
        "behnke",
    }
    assert expected.issubset(families.keys())
    assert families["brandimarte"].instance_count >= 10
