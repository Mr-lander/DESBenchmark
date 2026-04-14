from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from fms_bench.holonic_types import TreeMessage

from .authority_modes import AuthorityMode, get_authority_mode
from .task_semantics import CommitmentStatus, Contract, PropagationAction
from .types import ActivationRecord


@dataclass(frozen=True)
class SettlementRecord:
    contract_id: str | None
    status: CommitmentStatus
    cost_breakdown: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EscalationRecord:
    target: str
    reason: str
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AreaDecision:
    area_id: str
    functional_loop: str
    propagation_action: PropagationAction
    issued_contract: Contract | None = None
    settlement: SettlementRecord | None = None
    escalation: EscalationRecord | None = None
    emitted_messages: tuple[TreeMessage, ...] = ()


class AreaRuntime:
    """Role-real area runtime that can propagate, settle, and regulate."""

    def __init__(
        self,
        area_id: str,
        child_ids: Sequence[str],
        authority_mode: str = "holonic_hybrid",
        area_local_reroute_enabled: bool | None = None,
    ) -> None:
        if not child_ids:
            raise ValueError("AreaRuntime requires at least one child holon")
        self.area_id = area_id
        self.child_ids = tuple(child_ids)
        self.authority_mode: AuthorityMode = get_authority_mode(authority_mode)
        self._area_local_reroute_enabled = (
            self.authority_mode.area_local_reroute_enabled
            if area_local_reroute_enabled is None
            else bool(area_local_reroute_enabled)
        )

    def handle_activation(self, record: ActivationRecord) -> AreaDecision:
        if record.agent_id != self.area_id:
            raise ValueError(
                f"Activation for {record.agent_id} cannot be handled by {self.area_id}"
            )

        if record.functional_loop == "commitment_management":
            return self._orchestrate_children(record)
        if record.functional_loop == "accountability_and_escalation":
            return self._settle_or_escalate(record)
        if record.functional_loop == "local_regulation":
            return self._regulate(record)

        return AreaDecision(
            area_id=self.area_id,
            functional_loop=record.functional_loop,
            propagation_action=PropagationAction.DEFER,
        )

    def _orchestrate_children(self, record: ActivationRecord) -> AreaDecision:
        message = self._extract_message(record)
        child_id = self._select_child(message)
        parent_contract_id = str(message.payload.get("contract_id", "unbound-contract"))
        work_id = str(
            message.payload.get(
                "work_id",
                message.payload.get("shared_backlog_id", parent_contract_id),
            )
        )
        issued_contract = Contract(
            contract_id=f"{self.area_id}::{parent_contract_id}::{child_id}",
            work_id=work_id,
            sender=self.area_id,
            recipient=child_id,
            metadata={
                "upstream_sender": message.sender,
                "upstream_kind": message.kind,
                "parent_contract_id": parent_contract_id,
            },
        )
        emitted_message = TreeMessage(
            sender=self.area_id,
            recipient=child_id,
            kind=self._child_message_kind(message),
            payload={
                **{
                    key: value
                    for key, value in dict(message.payload).items()
                    if key != "contract_id"
                },
                "contract_id": issued_contract.contract_id,
                "upstream_contract_id": parent_contract_id,
                "issuer": self.area_id,
                "work_id": work_id,
            },
        )
        return AreaDecision(
            area_id=self.area_id,
            functional_loop="sub_holon_orchestration",
            propagation_action=PropagationAction.ASSIGN_TO_SUBHOLON,
            issued_contract=issued_contract,
            emitted_messages=(emitted_message,),
        )

    def _settle_or_escalate(self, record: ActivationRecord) -> AreaDecision:
        message = self._extract_message(record)
        status = self._message_status(message)
        if message.kind == "reject_contract":
            reroute_decision = self._try_reroute_after_reject(message)
            if reroute_decision is not None:
                return reroute_decision
        settlement = SettlementRecord(
            contract_id=message.payload.get("contract_id"),
            status=status,
            cost_breakdown={"reported_cost": message.payload.get("cost")},
        )
        escalation = None
        emitted_messages: tuple[TreeMessage, ...] = ()
        propagation_action = PropagationAction.SETTLE

        if status in {
            CommitmentStatus.PARTIALLY_COMPLETED,
            CommitmentStatus.BLOCKED,
            CommitmentStatus.VIOLATED,
            CommitmentStatus.FAILED,
        }:
            upstream_contract_id = message.payload.get("upstream_contract_id")
            if not upstream_contract_id and message.kind == "reject_contract":
                upstream_contract_id, _ = self._decode_rejected_path(
                    str(message.payload.get("contract_id", ""))
                )
            attempted_area_ids = [
                str(area_id)
                for area_id in message.payload.get("attempted_area_ids", [])
                if isinstance(area_id, str)
            ]
            if self.area_id not in attempted_area_ids:
                attempted_area_ids.append(self.area_id)
            escalation = EscalationRecord(
                target="parent",
                reason=status.value,
                payload={
                    "area_id": self.area_id,
                    "contract_id": message.payload.get("contract_id"),
                    "status": status.value,
                    "upstream_contract_id": upstream_contract_id,
                    "work_id": message.payload.get("work_id"),
                    "shared_backlog_id": message.payload.get("shared_backlog_id"),
                    "attempted_child_ids": list(
                        message.payload.get("attempted_child_ids", [])
                    ),
                    "attempted_area_ids": attempted_area_ids,
                },
            )
            emitted_messages = (
                TreeMessage(
                    sender=self.area_id,
                    recipient="plant",
                    kind="escalate",
                    payload=dict(escalation.payload),
                ),
            )
            propagation_action = PropagationAction.ESCALATE

        return AreaDecision(
            area_id=self.area_id,
            functional_loop="accountability_and_escalation",
            propagation_action=propagation_action,
            settlement=settlement,
            escalation=escalation,
            emitted_messages=emitted_messages,
        )

    def _try_reroute_after_reject(self, message: TreeMessage) -> AreaDecision | None:
        if not self._area_local_reroute_enabled:
            return None
        rejected_contract_id = str(message.payload.get("contract_id", ""))
        if not rejected_contract_id:
            return None

        attempted_child_ids = [
            str(child_id)
            for child_id in message.payload.get("attempted_child_ids", [])
            if isinstance(child_id, str)
        ]
        upstream_contract_id = str(
            message.payload.get("upstream_contract_id", "")
        )
        work_id = str(message.payload.get("work_id", ""))
        if not upstream_contract_id:
            upstream_contract_id, decoded_attempts = self._decode_rejected_path(
                rejected_contract_id
            )
            if not attempted_child_ids:
                attempted_child_ids = decoded_attempts
        if not attempted_child_ids:
            attempted_child_ids = [str(message.sender)]
        if str(message.sender) not in attempted_child_ids:
            attempted_child_ids.append(str(message.sender))
        if not work_id:
            work_id = upstream_contract_id
        sibling_candidates = tuple(
            child_id
            for child_id in self.child_ids
            if child_id.startswith("cell_") and child_id not in set(attempted_child_ids)
        )
        if not sibling_candidates:
            return None

        reroute_child = sibling_candidates[0]
        attempted_area_ids = [
            str(area_id)
            for area_id in message.payload.get("attempted_area_ids", [])
            if isinstance(area_id, str)
        ]
        if self.area_id not in attempted_area_ids:
            attempted_area_ids.append(self.area_id)
        issued_contract = Contract(
            contract_id=f"{self.area_id}::{upstream_contract_id}::{reroute_child}",
            work_id=work_id,
            sender=self.area_id,
            recipient=reroute_child,
            metadata={
                "parent_contract_id": upstream_contract_id,
                "attempted_child_ids": [*attempted_child_ids, reroute_child],
                "attempted_area_ids": attempted_area_ids,
                "reroute_reason": "child_reject",
                "shared_backlog_id": message.payload.get("shared_backlog_id"),
            },
        )
        emitted_message = TreeMessage(
            sender=self.area_id,
            recipient=reroute_child,
            kind="assignment_contract",
            payload={
                "contract_id": issued_contract.contract_id,
                "upstream_contract_id": upstream_contract_id,
                "issuer": self.area_id,
                "work_id": work_id,
                "attempted_child_ids": [*attempted_child_ids, reroute_child],
                "attempted_area_ids": attempted_area_ids,
                "reroute_reason": "child_reject",
                "shared_backlog_id": message.payload.get("shared_backlog_id"),
            },
        )
        return AreaDecision(
            area_id=self.area_id,
            functional_loop="accountability_and_escalation",
            propagation_action=PropagationAction.REROUTE,
            issued_contract=issued_contract,
            emitted_messages=(emitted_message,),
        )

    def _regulate(self, record: ActivationRecord) -> AreaDecision:
        return AreaDecision(
            area_id=self.area_id,
            functional_loop="local_regulation",
            propagation_action=PropagationAction.DEFER,
        )

    def _select_child(self, message: TreeMessage) -> str:
        preferred_child = message.payload.get("preferred_child")
        if isinstance(preferred_child, str) and preferred_child in self.child_ids:
            return preferred_child
        return self.child_ids[0]

    @staticmethod
    def _extract_message(record: ActivationRecord) -> TreeMessage:
        message = record.context_handles.get("message")
        if not isinstance(message, TreeMessage):
            raise ValueError("AreaRuntime requires TreeMessage activation context")
        return message

    @staticmethod
    def _child_message_kind(message: TreeMessage) -> str:
        if "shared_backlog_id" in message.payload:
            return "assignment_contract"
        return "release_contract"

    @staticmethod
    def _message_status(message: TreeMessage) -> CommitmentStatus:
        if message.kind == "reject_contract":
            return CommitmentStatus.FAILED
        raw_status = str(message.payload.get("status", "partially_completed"))
        try:
            return CommitmentStatus(raw_status)
        except ValueError:
            return CommitmentStatus.PARTIALLY_COMPLETED

    @staticmethod
    def _decode_rejected_path(contract_id: str) -> tuple[str, list[str]]:
        tokens = [token for token in str(contract_id).split("::") if token]
        if len(tokens) < 3:
            return str(contract_id), []
        upstream_contract_id = "::".join(tokens[1:-1])
        attempted_child = tokens[-1]
        return upstream_contract_id, [attempted_child]
