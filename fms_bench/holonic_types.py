"""Dataclasses for the holonic benchmark environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .models import Job


@dataclass(frozen=True)
class PlantCellConfig:
    """Configuration for one cell in a holonic plant."""

    cell_id: int
    family_key: str
    instance_path: Path
    instance_kind: str = "fjsp"
    label: Optional[str] = None
    dispatch_policy: str = "spt"
    wip_cap: int = 1
    energy_cap_kwh: float = 10_000.0
    carbon_cap_kg: float = 10_000.0
    processing_time_scale: float = 1.0
    inbound_buffer_cap: int = 1_000_000
    ingress_transport_time_hours: float = 0.0
    ingress_transport_energy_kwh: float = 0.0


@dataclass(frozen=True)
class CellBudget:
    """Cell-level resource envelope."""

    cell_id: int
    wip_cap: int
    energy_cap_kwh: float
    carbon_cap_kg: float


@dataclass(frozen=True)
class PlantContract:
    """Plant-to-cell release contract for one job."""

    contract_id: int
    cell_id: int
    job_id: int
    released_at: float
    due_time: Optional[float]
    horizon_remaining: float


@dataclass(frozen=True)
class SharedBacklogItem:
    """One plant-level backlog item that can be assigned to compatible cells."""

    shared_backlog_id: int
    template_job_index: int
    candidate_cell_ids: List[int]


@dataclass(frozen=True)
class AssignmentContract:
    """Plant-to-cell assignment contract for one shared backlog item."""

    contract_id: int
    shared_backlog_id: int
    template_job_index: int
    cell_id: int
    job_id: int
    released_at: float
    due_time: Optional[float]
    horizon_remaining: float


@dataclass(frozen=True)
class HolonicEvent:
    """Typed benchmark-facing event emitted by the shared DES world."""

    time: float
    kind: str
    scope: str
    payload: Dict[str, int | float | str]

    def to_dict(self) -> Dict[str, int | float | str]:
        return {
            "time": self.time,
            "kind": self.kind,
            "scope": self.scope,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class HolonicMessage:
    """Typed benchmark-facing message between logical holons."""

    sender: str
    recipient: str
    kind: str
    payload: Dict[str, int | float | str | None]

    def to_dict(self) -> Dict[str, object]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "kind": self.kind,
            "payload": dict(self.payload),
        }




@dataclass(frozen=True)
class TreeMessage:
    """Typed message constrained to parent-child tree communication."""

    sender: str
    recipient: str
    kind: str
    payload: Dict[str, int | float | str | None]
    scope: str = "parent_child"

    def __post_init__(self) -> None:
        if self.scope != "parent_child":
            raise ValueError("TreeMessage scope must be parent_child")

    def to_dict(self) -> Dict[str, object]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "kind": self.kind,
            "scope": self.scope,
            "payload": dict(self.payload),
        }


@dataclass
class AreaFeedback:
    """Aggregated area summary exposed to the parent layer."""

    area_id: int
    label: str
    cell_ids: List[int]
    backlog_jobs: int
    active_jobs: int
    completed_jobs: int
    energy_kwh: float
    carbon_kg: float
    energy_overshoot_kwh: float
    carbon_overshoot_kg: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "area_id": self.area_id,
            "label": self.label,
            "cell_ids": list(self.cell_ids),
            "backlog_jobs": self.backlog_jobs,
            "active_jobs": self.active_jobs,
            "completed_jobs": self.completed_jobs,
            "energy_kwh": self.energy_kwh,
            "carbon_kg": self.carbon_kg,
            "energy_overshoot_kwh": self.energy_overshoot_kwh,
            "carbon_overshoot_kg": self.carbon_overshoot_kg,
        }


@dataclass
class AreaSpec:
    """Resolved area definition for a tree-holonic controller layer."""

    area_id: int
    label: str
    cell_ids: List[int]

@dataclass
class CellFeedback:
    """Aggregated cell summary exposed to the plant layer."""

    cell_id: int
    label: str
    family_key: str
    backlog_jobs: int
    released_jobs: int
    active_jobs: int
    completed_jobs: int
    ready_jobs: int
    inbound_jobs: int
    busy_machines: int
    down_machines: int
    energy_kwh: float
    energy_cost: float
    carbon_kg: float
    budget_cap_energy_kwh: float
    budget_cap_carbon_kg: float
    budget_cap_wip: int
    inbound_buffer_cap: int
    budget_remaining_energy_kwh: float
    budget_remaining_carbon_kg: float
    budget_remaining_wip: int
    inbound_buffer_remaining: int
    energy_overshoot_kwh: float
    carbon_overshoot_kg: float
    processing_time_scale: float

    def to_dict(self) -> Dict[str, float | int | str]:
        return {
            "cell_id": self.cell_id,
            "label": self.label,
            "family_key": self.family_key,
            "backlog_jobs": self.backlog_jobs,
            "released_jobs": self.released_jobs,
            "active_jobs": self.active_jobs,
            "completed_jobs": self.completed_jobs,
            "ready_jobs": self.ready_jobs,
            "inbound_jobs": self.inbound_jobs,
            "busy_machines": self.busy_machines,
            "down_machines": self.down_machines,
            "energy_kwh": self.energy_kwh,
            "energy_cost": self.energy_cost,
            "carbon_kg": self.carbon_kg,
            "budget_cap_energy_kwh": self.budget_cap_energy_kwh,
            "budget_cap_carbon_kg": self.budget_cap_carbon_kg,
            "budget_cap_wip": self.budget_cap_wip,
            "inbound_buffer_cap": self.inbound_buffer_cap,
            "budget_remaining_energy_kwh": self.budget_remaining_energy_kwh,
            "budget_remaining_carbon_kg": self.budget_remaining_carbon_kg,
            "budget_remaining_wip": self.budget_remaining_wip,
            "inbound_buffer_remaining": self.inbound_buffer_remaining,
            "energy_overshoot_kwh": self.energy_overshoot_kwh,
            "carbon_overshoot_kg": self.carbon_overshoot_kg,
            "processing_time_scale": self.processing_time_scale,
        }


@dataclass
class CellSpec:
    """Resolved cell definition with remapped jobs and machines."""

    cell_id: int
    label: str
    family_key: str
    instance_path: Path
    dispatch_policy: str
    machine_ids: List[int]
    template_job_ids: List[int]
    template_jobs: List[Job]
    budget: CellBudget
    processing_time_scale: float = 1.0
    inbound_buffer_cap: int = 1_000_000
    ingress_transport_time_hours: float = 0.0
    ingress_transport_energy_kwh: float = 0.0


@dataclass
class PlantAssembly:
    """Combined instance and topology metadata for the plant environment."""

    combined_jobs: List[Job]
    total_machine_count: int
    cells: Dict[int, CellSpec] = field(default_factory=dict)
    job_to_cell: Dict[int, int] = field(default_factory=dict)
    machine_to_cell: Dict[int, int] = field(default_factory=dict)

    @property
    def total_job_count(self) -> int:
        return len(self.combined_jobs)
