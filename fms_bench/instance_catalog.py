"""Local benchmark family catalog for the industrial FMS suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class InstanceFamily:
    key: str
    problem_kind: str
    source_dir: Path
    citation: str
    notes: str

    @property
    def instance_count(self) -> int:
        return len(list(self.source_dir.glob("**/*.txt")))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_instance_families() -> List[InstanceFamily]:
    """Return the local benchmark families that are useful for this project."""
    root = _repo_root()
    flexible_root = root / "external" / "benchmarks" / "flexible jobshop" / "instances"
    jobshop_root = root / "external" / "benchmarks" / "jobshop" / "instances"
    return [
        InstanceFamily(
            key="taillard",
            problem_kind="jssp",
            source_dir=jobshop_root / "Taillard1993" / "jobshop",
            citation="Taillard (1993)",
            notes="Static JSSP baseline family for fixed-routing control.",
        ),
        InstanceFamily(
            key="brandimarte",
            problem_kind="fjsp",
            source_dir=flexible_root / "Brandimarte1993",
            citation="Brandimarte (1993)",
            notes="Canonical FJSP backbone; good default cell template.",
        ),
        InstanceFamily(
            key="hurink",
            problem_kind="fjsp",
            source_dir=flexible_root / "HurinkJurischThole1994",
            citation="Hurink, Jurisch and Thole (1994)",
            notes="Flexibility-stress family with edata/rdata/sdata/vdata variants.",
        ),
        InstanceFamily(
            key="dauzere",
            problem_kind="fjsp",
            source_dir=flexible_root / "DauzerePeresPaulli1994",
            citation="Dauzere-Peres and Paulli (1994)",
            notes="General multiprocessor job-shop family with harder routing structure.",
        ),
        InstanceFamily(
            key="chambers_barnes",
            problem_kind="fjsp",
            source_dir=flexible_root / "ChambersBarnes1996",
            citation="Chambers and Barnes (1996)",
            notes="Tabu-search benchmark family with classic medium-sized cases.",
        ),
        InstanceFamily(
            key="kacem",
            problem_kind="fjsp",
            source_dir=flexible_root / "KacemHammadiBorne2002",
            citation="Kacem, Hammadi and Borne (2002)",
            notes="Small multi-objective-style FJSP cases; good for debugging and sanity checks.",
        ),
        InstanceFamily(
            key="fattahi",
            problem_kind="fjsp",
            source_dir=flexible_root / "FattahiMehrabadJolai2007",
            citation="Fattahi, Mehrabad and Jolai (2007)",
            notes="Industrial-style FJSP set; good second core family after Brandimarte.",
        ),
        InstanceFamily(
            key="behnke",
            problem_kind="fjsp",
            source_dir=flexible_root / "BehnkeGeiger2012",
            citation="Behnke and Geiger (2012)",
            notes="Large and difficult work-center family for scale and generalization stress.",
        ),
    ]


def recommended_holonic_suite() -> List[dict[str, object]]:
    """Return a pragmatic v1 suite recommendation for this benchmark."""
    return [
        {
            "stage": "core_cell_backbones",
            "families": ["brandimarte", "fattahi", "kacem"],
            "why": "Cover canonical, industrial-style, and small sanity-check FJSP cases.",
        },
        {
            "stage": "flexibility_stress",
            "families": ["hurink"],
            "why": "Use edata/rdata/vdata to vary routing flexibility without changing the overall problem class.",
        },
        {
            "stage": "scale_and_generalization",
            "families": ["dauzere", "behnke"],
            "why": "Stress transfer to larger and less toy-like routing structures.",
        },
        {
            "stage": "flat_baseline_reference",
            "families": ["taillard"],
            "why": "Provide a fixed-routing JSSP anchor so the benchmark is not only evaluated on flexible cases.",
        },
    ]
