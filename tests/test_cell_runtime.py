"""Tests for the v2 cell runtime boundary."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench.holonic_types import HolonicEvent, TreeMessage
from fms_bench.v2.cell_runtime import CellAction, CellRuntime
from fms_bench.v2.types import ActivationRecord


def _idle_activation() -> ActivationRecord:
    return ActivationRecord(
        agent_id="cell_0",
        activation_cause="world_event",
        functional_loop="local_regulation",
        activation_timestamp=11.0,
        event_summary=(
            HolonicEvent(
                time=11.0,
                kind="machine_idle",
                scope="cell_0",
                payload={"machine_id": 0},
            ),
        ),
        context_handles={
            "dispatch_options": (
                {"operation_id": "op_1", "machine_id": 0, "proc_time": 3.0},
                {"operation_id": "op_2", "machine_id": 0, "proc_time": 5.0},
            )
        },
    )


def _assignment_activation() -> ActivationRecord:
    return ActivationRecord(
        agent_id="cell_0",
        activation_cause="parent_message",
        functional_loop="commitment_management",
        activation_timestamp=12.0,
        context_handles={
            "message": TreeMessage(
                sender="area_0",
                recipient="cell_0",
                kind="assignment_contract",
                payload={"contract_id": "c1", "job_id": 3},
            )
        },
    )


def test_cell_runtime_receives_machine_idle_activation_and_returns_dispatch_action() -> None:
    runtime = CellRuntime(cell_id="cell_0")

    decision = runtime.handle_activation(_idle_activation())

    assert decision.action_kind == "dispatch"
    assert decision.selected_operation_id == "op_1"


def test_cell_runtime_exposes_legal_local_dispatch_choices() -> None:
    runtime = CellRuntime(cell_id="cell_0")

    legal_actions = runtime.legal_actions(_idle_activation())

    assert {action.selected_operation_id for action in legal_actions} == {"op_1", "op_2"}


def test_cell_runtime_can_accept_parent_assignment_contract() -> None:
    runtime = CellRuntime(cell_id="cell_0")

    decision = runtime.handle_activation(_assignment_activation())

    assert decision.action_kind == "accept_contract"
    assert decision.accepted_contract_id == "c1"


def test_cell_runtime_exposes_accept_and_reject_actions_for_commitment_management() -> None:
    runtime = CellRuntime(cell_id="cell_0")

    legal_actions = runtime.legal_actions(_assignment_activation())

    assert [action.action_kind for action in legal_actions] == [
        "accept_contract",
        "reject_contract",
    ]


def test_cell_runtime_hierarchical_mode_exposes_accept_only_contract_action() -> None:
    runtime = CellRuntime(cell_id="cell_0", authority_mode="hierarchical")

    legal_actions = runtime.legal_actions(_assignment_activation())

    assert [action.action_kind for action in legal_actions] == ["accept_contract"]


def test_cell_runtime_can_reject_parent_assignment_contract() -> None:
    runtime = CellRuntime(cell_id="cell_0")
    reject_action = runtime.legal_actions(_assignment_activation())[1]

    decision = runtime.handle_activation(_assignment_activation(), action=reject_action)

    assert decision.action_kind == "reject_contract"
    assert decision.accepted_contract_id is None
    assert len(decision.emitted_messages) == 1
    assert decision.emitted_messages[0].kind == "reject_contract"
    assert decision.emitted_messages[0].payload["contract_id"] == "c1"


def test_cell_runtime_accept_only_mode_disables_reject_and_falls_back_to_accept() -> None:
    runtime = CellRuntime(cell_id="cell_0", allow_reject=False)

    legal_actions = runtime.legal_actions(_assignment_activation())

    assert [action.action_kind for action in legal_actions] == ["accept_contract"]

    decision = runtime.handle_activation(
        _assignment_activation(),
        action=CellAction(action_kind="reject_contract"),
    )

    assert decision.action_kind == "accept_contract"
    assert decision.accepted_contract_id == "c1"
    assert decision.emitted_messages[0].kind == "accept_contract"
