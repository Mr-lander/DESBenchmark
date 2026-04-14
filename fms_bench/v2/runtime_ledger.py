from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from fms_bench.holonic_types import TreeMessage

from .task_semantics import CommitmentStatus, Contract

REWARD_AUTHORITY_ID = "runtime_ledger_interval_v2"
SETTLEMENT_TRACE_CONTRACT = "cumulative_contract_history_trace"
SETTLEMENT_SEMANTICS_CONTRACT = "minimal_auto_settle_on_completed_bound_job"


@dataclass(frozen=True)
class RewardBreakdown:
    interval_start: float
    interval_end: float
    components: Mapping[str, float]
    energy_overshoot_kwh: float = 0.0
    carbon_overshoot_kg: float = 0.0

    @property
    def total(self) -> float:
        return float(sum(float(value) for value in self.components.values()))

    def to_dict(self) -> dict[str, float]:
        payload = {key: float(value) for key, value in self.components.items()}
        payload["energy_overshoot_kwh"] = float(self.energy_overshoot_kwh)
        payload["carbon_overshoot_kg"] = float(self.carbon_overshoot_kg)
        payload["total"] = self.total
        return payload


@dataclass
class ContractLedgerEntry:
    contract_id: str
    work_id: str
    sender: str
    recipient: str
    status: CommitmentStatus
    job_id: int | None = None
    total_operations: int | None = None
    completed_operations: int = 0
    accepted_by: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_trace_dict(self, timestamp: float) -> dict[str, object]:
        return {
            "timestamp": float(timestamp),
            "contract_id": self.contract_id,
            "work_id": self.work_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "status": self.status.value,
            "job_id": self.job_id,
            "completed_operations": int(self.completed_operations),
            "total_operations": self.total_operations,
            "accepted_by": self.accepted_by,
        }


class RuntimeLedger:
    """Auditable runtime ledger for contract status and interval reward."""

    def __init__(self) -> None:
        self._contracts: dict[str, ContractLedgerEntry] = {}
        self._job_to_contract: dict[int, str] = {}
        self._settlement_trace: list[dict[str, object]] = []
        self._violation_summary = {
            "energy_overshoot_magnitude_total": 0.0,
            "carbon_overshoot_magnitude_total": 0.0,
            "energy_overshoot_duration": 0.0,
            "carbon_overshoot_duration": 0.0,
            "energy_overshoot_frequency": 0,
            "carbon_overshoot_frequency": 0,
        }

    def apply_protocol_object(self, obj: object, timestamp: float | None = None) -> None:
        event_time = 0.0 if timestamp is None else float(timestamp)
        if isinstance(obj, Contract):
            entry = ContractLedgerEntry(
                contract_id=obj.contract_id,
                work_id=obj.work_id,
                sender=obj.sender,
                recipient=obj.recipient,
                status=CommitmentStatus.OFFERED,
                metadata=dict(obj.metadata),
            )
            self._contracts[obj.contract_id] = entry
            self._append_trace(obj.contract_id, event_time)
            return

        if isinstance(obj, TreeMessage) and obj.kind == "accept_contract":
            contract_id = str(obj.payload["contract_id"])
            self.mark_contract_active(
                contract_id=contract_id,
                accepted_by=obj.sender,
                timestamp=event_time,
            )
            return

        if isinstance(obj, TreeMessage) and obj.kind == "reject_contract":
            contract_id = str(obj.payload["contract_id"])
            entry = self._require_contract(contract_id)
            entry.status = CommitmentStatus.FAILED
            self._append_trace(contract_id, event_time)
            return

        if isinstance(obj, TreeMessage) and obj.kind == "escalate":
            contract_id = obj.payload.get("contract_id")
            if isinstance(contract_id, str) and contract_id in self._contracts:
                entry = self._contracts[contract_id]
                raw_status = str(obj.payload.get("status", CommitmentStatus.PARTIALLY_COMPLETED))
                try:
                    target_status = CommitmentStatus(raw_status)
                except ValueError:
                    target_status = CommitmentStatus.PARTIALLY_COMPLETED
                if entry.status != target_status:
                    entry.status = target_status
                    self._append_trace(contract_id, event_time)

    def bind_job(self, contract_id: str, job_id: int, total_operations: int) -> None:
        entry = self._require_contract(contract_id)
        entry.job_id = int(job_id)
        entry.total_operations = int(total_operations)
        self._job_to_contract[int(job_id)] = contract_id

    def mark_contract_active(
        self,
        contract_id: str,
        accepted_by: str,
        timestamp: float,
    ) -> None:
        entry = self._require_contract(contract_id)
        entry.accepted_by = accepted_by
        entry.status = CommitmentStatus.ACTIVE
        self._append_trace(contract_id, timestamp)

    def record_physical_progress(
        self,
        *,
        job_id: int,
        completed_operations: int,
        completed: bool,
        timestamp: float,
    ) -> None:
        contract_id = self._job_to_contract.get(int(job_id))
        if contract_id is None:
            return
        entry = self._contracts[contract_id]
        previous_completed_operations = int(entry.completed_operations)
        previous_status = entry.status
        entry.completed_operations = max(
            int(entry.completed_operations),
            int(completed_operations),
        )
        if completed:
            if entry.status != CommitmentStatus.SETTLED:
                entry.status = CommitmentStatus.COMPLETED
        elif (
            entry.completed_operations > 0
            and entry.status != CommitmentStatus.SETTLED
        ):
            entry.status = CommitmentStatus.PARTIALLY_COMPLETED
        if (
            int(entry.completed_operations) != previous_completed_operations
            or entry.status != previous_status
        ):
            self._append_trace(contract_id, timestamp)

    def mark_contract_settled(self, *, contract_id: str, timestamp: float) -> None:
        entry = self._require_contract(contract_id)
        if entry.status == CommitmentStatus.SETTLED:
            return
        if entry.status != CommitmentStatus.COMPLETED:
            return
        entry.status = CommitmentStatus.SETTLED
        self._append_trace(contract_id, timestamp)

    def mark_job_settled(self, *, job_id: int, timestamp: float) -> None:
        contract_id = self._job_to_contract.get(int(job_id))
        if contract_id is None:
            return
        self.mark_contract_settled(contract_id=contract_id, timestamp=timestamp)

    def contract_status(self, contract_id: str) -> CommitmentStatus:
        return self._require_contract(contract_id).status

    def settlement_trace(self) -> list[dict[str, object]]:
        return [dict(item) for item in self._settlement_trace]

    def settlement_summary(self) -> dict[str, int]:
        terminal_statuses = {
            CommitmentStatus.COMPLETED,
            CommitmentStatus.SETTLED,
            CommitmentStatus.VIOLATED,
            CommitmentStatus.FAILED,
            CommitmentStatus.CANCELLED,
            CommitmentStatus.EXPIRED,
        }
        settled_contracts = 0
        completed_unsettled_contracts = 0
        open_contracts = 0
        for entry in self._contracts.values():
            if entry.status == CommitmentStatus.SETTLED:
                settled_contracts += 1
            elif entry.status == CommitmentStatus.COMPLETED:
                completed_unsettled_contracts += 1
            elif entry.status not in terminal_statuses:
                open_contracts += 1
        return {
            "contracts_total": len(self._contracts),
            "settled_contracts": settled_contracts,
            "completed_unsettled_contracts": completed_unsettled_contracts,
            "open_contracts": open_contracts,
        }

    def bound_job_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._job_to_contract))

    def violation_summary(self) -> dict[str, float | int]:
        return dict(self._violation_summary)

    def finalize_interval(
        self,
        *,
        start_time: float,
        end_time: float,
        energy_kwh: float,
        carbon_kg: float,
        invalid_action: bool,
        terminated: bool,
        deadlocked: bool,
        tardiness_before: float,
        tardiness_after: float,
        energy_overshoot_kwh: float = 0.0,
        carbon_overshoot_kg: float = 0.0,
    ) -> RewardBreakdown:
        interval_duration = max(0.0, float(end_time) - float(start_time))
        if float(energy_overshoot_kwh) > 0.0:
            self._violation_summary["energy_overshoot_magnitude_total"] += float(
                energy_overshoot_kwh
            )
            self._violation_summary["energy_overshoot_duration"] += interval_duration
            self._violation_summary["energy_overshoot_frequency"] += 1
        if float(carbon_overshoot_kg) > 0.0:
            self._violation_summary["carbon_overshoot_magnitude_total"] += float(
                carbon_overshoot_kg
            )
            self._violation_summary["carbon_overshoot_duration"] += interval_duration
            self._violation_summary["carbon_overshoot_frequency"] += 1
        components = {
            "delta_time": -(float(end_time) - float(start_time)),
            "delta_tardiness": -(float(tardiness_after) - float(tardiness_before)),
            "energy_penalty": -0.01 * float(energy_kwh),
            "carbon_penalty": -0.001 * float(carbon_kg),
            "invalid_action_penalty": -0.1 if invalid_action else 0.0,
            "deadlock_penalty": -10.0 if deadlocked else 0.0,
            "terminal_bonus": 1.0 if terminated else 0.0,
        }
        return RewardBreakdown(
            interval_start=float(start_time),
            interval_end=float(end_time),
            components=components,
            energy_overshoot_kwh=float(energy_overshoot_kwh),
            carbon_overshoot_kg=float(carbon_overshoot_kg),
        )

    def _require_contract(self, contract_id: str) -> ContractLedgerEntry:
        if contract_id not in self._contracts:
            raise KeyError(f"Unknown runtime contract: {contract_id}")
        return self._contracts[contract_id]

    def _append_trace(self, contract_id: str, timestamp: float) -> None:
        entry = self._require_contract(contract_id)
        self._settlement_trace.append(entry.to_trace_dict(timestamp))
