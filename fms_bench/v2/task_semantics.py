from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class PropagationAction(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    DEFER = "defer"
    ASSIGN_TO_SUBHOLON = "assign_to_subholon"
    SPLIT = "split"
    MERGE = "merge"
    REROUTE = "reroute"
    REVISE_CONTRACT = "revise_contract"
    ESCALATE = "escalate"
    SETTLE = "settle"


class CommitmentStatus(StrEnum):
    PROPOSED = "proposed"
    OFFERED = "offered"
    ACCEPTED = "accepted"
    ACTIVE = "active"
    BLOCKED = "blocked"
    PARTIALLY_COMPLETED = "partially_completed"
    COMPLETED = "completed"
    VIOLATED = "violated"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    SETTLED = "settled"


PUBLIC_WORK_ITEM_CONTRACT = "next_fifo_shared_backlog_item_assignment_propagation"
IMPLEMENTED_PROPAGATION_ACTIONS = (
    PropagationAction.ASSIGN_TO_SUBHOLON,
    PropagationAction.DEFER,
    PropagationAction.REROUTE,
    PropagationAction.REVISE_CONTRACT,
    PropagationAction.ESCALATE,
    PropagationAction.SETTLE,
)
RESERVED_PROPAGATION_ACTIONS = (
    PropagationAction.ACCEPT,
    PropagationAction.REJECT,
    PropagationAction.SPLIT,
    PropagationAction.MERGE,
)
STABLE_PUBLIC_COMMITMENT_STATUSES = (
    CommitmentStatus.OFFERED,
    CommitmentStatus.ACTIVE,
    CommitmentStatus.PARTIALLY_COMPLETED,
    CommitmentStatus.COMPLETED,
    CommitmentStatus.FAILED,
    CommitmentStatus.SETTLED,
)


@dataclass(frozen=True)
class WorkPackage:
    work_id: str
    origin: str
    allowed_split: bool = False
    allowed_merge: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Contract:
    contract_id: str
    work_id: str
    sender: str
    recipient: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitmentStateMachine:
    status: CommitmentStatus

    @classmethod
    def initial(cls) -> "CommitmentStateMachine":
        return cls(status=CommitmentStatus.PROPOSED)

    def offer(self) -> "CommitmentStateMachine":
        return CommitmentStateMachine(status=CommitmentStatus.OFFERED)

    def accept(self) -> "CommitmentStateMachine":
        return CommitmentStateMachine(status=CommitmentStatus.ACCEPTED)

    def activate(self) -> "CommitmentStateMachine":
        return CommitmentStateMachine(status=CommitmentStatus.ACTIVE)

    def partial_complete(self) -> "CommitmentStateMachine":
        return CommitmentStateMachine(status=CommitmentStatus.PARTIALLY_COMPLETED)

    def settle(self) -> "CommitmentStateMachine":
        return CommitmentStateMachine(status=CommitmentStatus.SETTLED)
