from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ActivationRecord:
    agent_id: str
    activation_cause: str
    functional_loop: str
    activation_timestamp: float
    decision_kind: str | None = None
    event_summary: tuple[object, ...] = ()
    context_handles: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LegalDecisionState:
    agent_id: str
    activation: ActivationRecord
    observation: Mapping[str, object]
    visible_protocol_objects: tuple[object, ...]
    legal_history_ref: str | None
