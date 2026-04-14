from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from fms_bench.holonic_types import TreeMessage

from .authority_modes import AuthorityMode, get_authority_mode
from .types import ActivationRecord


@dataclass(frozen=True)
class CellAction:
    action_kind: str
    selected_operation_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CellDecision:
    cell_id: str
    functional_loop: str
    action_kind: str
    selected_operation_id: str | None = None
    accepted_contract_id: str | None = None
    emitted_messages: tuple[TreeMessage, ...] = ()


class CellRuntime:
    """Benchmark-facing cell runtime with legal local choices."""

    def __init__(
        self,
        cell_id: str,
        authority_mode: str = "holonic_hybrid",
        allow_reject: bool | None = None,
    ) -> None:
        self.cell_id = cell_id
        self.authority_mode: AuthorityMode = get_authority_mode(authority_mode)
        self._allow_reject = (
            self.authority_mode.cell_contract_actions == "accept_reject"
            if allow_reject is None
            else bool(allow_reject)
        )

    def legal_actions(self, record: ActivationRecord) -> tuple[CellAction, ...]:
        self._validate_record(record)

        if record.functional_loop == "local_regulation":
            dispatch_options = record.context_handles.get("dispatch_options", ())
            return tuple(
                CellAction(
                    action_kind="dispatch",
                    selected_operation_id=str(option["operation_id"]),
                    metadata=dict(option),
                )
                for option in dispatch_options
            )

        if record.functional_loop == "commitment_management":
            message = self._extract_message(record)
            actions = [
                CellAction(
                    action_kind="accept_contract",
                    metadata={"contract_id": message.payload.get("contract_id")},
                )
            ]
            if self._allow_reject:
                actions.append(
                    CellAction(
                        action_kind="reject_contract",
                        metadata={"contract_id": message.payload.get("contract_id")},
                    )
                )
            return tuple(actions)

        return (CellAction(action_kind="defer"),)

    def handle_activation(
        self,
        record: ActivationRecord,
        action: CellAction | None = None,
    ) -> CellDecision:
        self._validate_record(record)

        if record.functional_loop == "commitment_management":
            message = self._extract_message(record)
            legal_actions = self.legal_actions(record)
            chosen_action = action or legal_actions[0]
            legal_action_kinds = {candidate.action_kind for candidate in legal_actions}
            if chosen_action.action_kind not in legal_action_kinds:
                chosen_action = legal_actions[0]
            contract_id = str(message.payload.get("contract_id"))
            if chosen_action.action_kind == "reject_contract":
                return CellDecision(
                    cell_id=self.cell_id,
                    functional_loop="commitment_management",
                    action_kind="reject_contract",
                    emitted_messages=(
                        TreeMessage(
                            sender=self.cell_id,
                            recipient=message.sender,
                            kind="reject_contract",
                            payload={"contract_id": contract_id},
                        ),
                    ),
                )
            return CellDecision(
                cell_id=self.cell_id,
                functional_loop="commitment_management",
                action_kind="accept_contract",
                accepted_contract_id=contract_id,
                emitted_messages=(
                    TreeMessage(
                        sender=self.cell_id,
                        recipient=message.sender,
                        kind="accept_contract",
                        payload={"contract_id": contract_id},
                    ),
                ),
            )

        legal_actions = self.legal_actions(record)
        chosen_action = action or legal_actions[0]
        return CellDecision(
            cell_id=self.cell_id,
            functional_loop=record.functional_loop,
            action_kind=chosen_action.action_kind,
            selected_operation_id=chosen_action.selected_operation_id,
        )

    def _validate_record(self, record: ActivationRecord) -> None:
        if record.agent_id != self.cell_id:
            raise ValueError(
                f"Activation for {record.agent_id} cannot be handled by {self.cell_id}"
            )

    @staticmethod
    def _extract_message(record: ActivationRecord) -> TreeMessage:
        message = record.context_handles.get("message")
        if not isinstance(message, TreeMessage):
            raise ValueError("CellRuntime requires TreeMessage activation context")
        return message
