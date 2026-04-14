"""Core dataclasses for FMS-Bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OperationOption:
    machine_id: int
    proc_time: float


@dataclass
class Operation:
    op_id: int
    options: List[OperationOption]


@dataclass
class Job:
    job_id: int
    operations: List[Operation]
    due_time: Optional[float] = None
    release_time: float = 0.0
    current_op_idx: int = 0
    template_job_id: Optional[int] = None
    setup_family: Optional[str] = None


@dataclass
class Machine:
    machine_id: int
    idle_power_kw: float
    active_power_kw: float
    setup_power_kw: float = 0.0
    status: str = "idle"
    current_job: Optional[int] = None
    remaining_time: float = 0.0
    down_time: float = 0.0
    setup_remaining_time: float = 0.0
    pending_proc_time: float = 0.0
    last_setup_family: Optional[str] = None
    target_setup_family: Optional[str] = None


@dataclass
class Factory:
    factory_id: int
    machines: Dict[int, Machine] = field(default_factory=dict)
    queue: List[int] = field(default_factory=list)
