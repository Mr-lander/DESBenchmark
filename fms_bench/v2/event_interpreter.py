from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from fms_bench.holonic_types import HolonicEvent, TreeMessage

from .types import ActivationRecord


@dataclass(frozen=True)
class DecisionEpoch:
    activation_timestamp: float
    records_by_agent: dict[str, ActivationRecord]

    @property
    def agents_to_act(self) -> tuple[str, ...]:
        return tuple(self.records_by_agent.keys())


def interpret_decision_epoch(
    *,
    world_time: float,
    events: Sequence[HolonicEvent] | Iterable[HolonicEvent],
    messages: Sequence[TreeMessage] | Iterable[TreeMessage] = (),
) -> DecisionEpoch:
    records_by_agent: dict[str, ActivationRecord] = {}

    for event in events:
        records_by_agent[event.scope] = ActivationRecord(
            agent_id=event.scope,
            activation_cause="world_event",
            functional_loop=_map_event_loop(event.kind),
            activation_timestamp=world_time,
            event_summary=(event,),
        )

    for message in messages:
        records_by_agent[message.recipient] = ActivationRecord(
            agent_id=message.recipient,
            activation_cause="parent_message",
            functional_loop=_map_message_loop(message.kind),
            activation_timestamp=world_time,
            context_handles={
                "sender": message.sender,
                "message": message,
            },
        )

    return DecisionEpoch(
        activation_timestamp=world_time,
        records_by_agent=records_by_agent,
    )


def _map_event_loop(kind: str) -> str:
    if kind in {"release_window_open", "release_needed"}:
        return "commitment_management"
    if kind in {"budget_alarm", "breakdown", "repair"}:
        return "local_regulation"
    return "local_regulation"


def _map_message_loop(kind: str) -> str:
    if kind in {
        "child_feedback",
        "child_completion",
        "status_update",
        "settlement",
        "reject_contract",
        "escalate",
    }:
        return "accountability_and_escalation"
    if kind in {"allocate", "release_contract", "assignment_contract"}:
        return "commitment_management"
    return "commitment_management"
