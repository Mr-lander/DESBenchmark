"""Tests for the v2 area runtime boundary."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench.holonic_types import HolonicEvent, TreeMessage
from fms_bench.v2.area_runtime import AreaRuntime
from fms_bench.v2.task_semantics import CommitmentStatus, PropagationAction
from fms_bench.v2.types import ActivationRecord


def _contract_activation() -> ActivationRecord:
    return ActivationRecord(
        agent_id="area_0",
        activation_cause="parent_message",
        functional_loop="commitment_management",
        activation_timestamp=5.0,
        context_handles={
            "message": TreeMessage(
                sender="plant",
                recipient="area_0",
                kind="release_request",
                payload={
                    "contract_id": "plant-c1",
                    "work_id": "w1",
                    "shared_backlog_id": 3,
                },
            )
        },
    )


def _feedback_activation(status: str) -> ActivationRecord:
    return ActivationRecord(
        agent_id="area_0",
        activation_cause="parent_message",
        functional_loop="accountability_and_escalation",
        activation_timestamp=7.0,
        context_handles={
            "message": TreeMessage(
                sender="cell_0",
                recipient="area_0",
                kind="child_feedback",
                payload={
                    "contract_id": "area_0::plant-c1::cell_0",
                    "status": status,
                    "cost": 12.5,
                },
            )
        },
    )


def _reject_activation() -> ActivationRecord:
    return ActivationRecord(
        agent_id="area_0",
        activation_cause="parent_message",
        functional_loop="accountability_and_escalation",
        activation_timestamp=7.0,
        context_handles={
            "message": TreeMessage(
                sender="cell_0",
                recipient="area_0",
                kind="reject_contract",
                payload={"contract_id": "area_0::plant-c1::cell_0"},
            )
        },
    )


def test_area_runtime_can_issue_child_contract_without_being_a_cell_picker() -> None:
    runtime = AreaRuntime(area_id="area_0", child_ids=("cell_0", "area_1"))

    decision = runtime.handle_activation(_contract_activation())

    assert decision.propagation_action == PropagationAction.ASSIGN_TO_SUBHOLON
    assert decision.issued_contract is not None
    assert decision.issued_contract.recipient in {"cell_0", "area_1"}
    assert decision.functional_loop == "sub_holon_orchestration"


def test_area_runtime_can_settle_partial_completion_and_escalate() -> None:
    runtime = AreaRuntime(area_id="area_0", child_ids=("cell_0", "area_1"))

    decision = runtime.handle_activation(_feedback_activation(status="partially_completed"))

    assert decision.settlement is not None
    assert decision.settlement.status == CommitmentStatus.PARTIALLY_COMPLETED
    assert decision.escalation is not None
    assert decision.escalation.target == "parent"


def test_area_runtime_treats_reject_contract_as_failed_child_feedback() -> None:
    runtime = AreaRuntime(area_id="area_0", child_ids=("cell_0", "area_1"))

    decision = runtime.handle_activation(_reject_activation())

    assert decision.settlement is not None
    assert decision.settlement.status == CommitmentStatus.FAILED
    assert decision.escalation is not None
    assert decision.escalation.target == "parent"
    assert decision.emitted_messages[0].kind == "escalate"
    assert decision.emitted_messages[0].payload["status"] == "failed"


def test_area_runtime_reroutes_once_to_remaining_cell_after_reject() -> None:
    runtime = AreaRuntime(area_id="area_0", child_ids=("cell_0", "cell_1"))
    activation = ActivationRecord(
        agent_id="area_0",
        activation_cause="parent_message",
        functional_loop="accountability_and_escalation",
        activation_timestamp=7.0,
        context_handles={
            "message": TreeMessage(
                sender="cell_0",
                recipient="area_0",
                kind="reject_contract",
                payload={"contract_id": "area_0::plant-c1::cell_0"},
            )
        },
    )

    decision = runtime.handle_activation(activation)

    assert decision.propagation_action == PropagationAction.REROUTE
    assert decision.escalation is None
    assert decision.issued_contract is not None
    assert decision.issued_contract.recipient == "cell_1"
    assert decision.emitted_messages[0].kind == "assignment_contract"
    assert decision.emitted_messages[0].recipient == "cell_1"


def test_area_runtime_hierarchical_mode_escalates_reject_without_local_reroute() -> None:
    runtime = AreaRuntime(
        area_id="area_0",
        child_ids=("cell_0", "cell_1"),
        authority_mode="hierarchical",
    )

    decision = runtime.handle_activation(_reject_activation())

    assert decision.propagation_action == PropagationAction.ESCALATE
    assert decision.issued_contract is None
    assert decision.emitted_messages[0].kind == "escalate"


def test_area_runtime_escalates_when_reject_has_no_remaining_sibling() -> None:
    runtime = AreaRuntime(area_id="area_0", child_ids=("cell_0", "cell_1"))
    activation = ActivationRecord(
        agent_id="area_0",
        activation_cause="parent_message",
        functional_loop="accountability_and_escalation",
        activation_timestamp=7.0,
        context_handles={
            "message": TreeMessage(
                sender="cell_1",
                recipient="area_0",
                kind="reject_contract",
                payload={
                    "contract_id": "area_0::plant-c1::cell_1",
                    "upstream_contract_id": "plant-c1",
                    "attempted_child_ids": ["cell_0", "cell_1"],
                },
            )
        },
    )

    decision = runtime.handle_activation(activation)

    assert decision.propagation_action == PropagationAction.ESCALATE
    assert decision.escalation is not None
    assert decision.emitted_messages[0].kind == "escalate"


def test_area_runtime_escalation_payload_preserves_lineage_for_parent_revise() -> None:
    runtime = AreaRuntime(area_id="area_0", child_ids=("cell_0", "cell_1"))
    activation = ActivationRecord(
        agent_id="area_0",
        activation_cause="parent_message",
        functional_loop="accountability_and_escalation",
        activation_timestamp=7.0,
        context_handles={
            "message": TreeMessage(
                sender="cell_1",
                recipient="area_0",
                kind="reject_contract",
                payload={
                    "contract_id": "area_0::plant-c1::cell_1",
                    "upstream_contract_id": "plant-c1",
                    "work_id": "shared_backlog_3",
                    "shared_backlog_id": 3,
                    "attempted_child_ids": ["cell_0", "cell_1"],
                },
            )
        },
    )

    decision = runtime.handle_activation(activation)

    assert decision.propagation_action == PropagationAction.ESCALATE
    assert decision.emitted_messages[0].kind == "escalate"
    assert decision.emitted_messages[0].payload["upstream_contract_id"] == "plant-c1"
    assert decision.emitted_messages[0].payload["work_id"] == "shared_backlog_3"
    assert decision.emitted_messages[0].payload["shared_backlog_id"] == 3
    assert decision.emitted_messages[0].payload["attempted_area_ids"] == ["area_0"]


def test_area_runtime_can_disable_local_reroute_and_escalate_reject_immediately() -> None:
    runtime = AreaRuntime(
        area_id="area_0",
        child_ids=("cell_0", "cell_1"),
        area_local_reroute_enabled=False,
    )
    activation = ActivationRecord(
        agent_id="area_0",
        activation_cause="parent_message",
        functional_loop="accountability_and_escalation",
        activation_timestamp=7.0,
        context_handles={
            "message": TreeMessage(
                sender="cell_0",
                recipient="area_0",
                kind="reject_contract",
                payload={"contract_id": "area_0::plant-c1::cell_0"},
            )
        },
    )

    decision = runtime.handle_activation(activation)

    assert decision.propagation_action == PropagationAction.ESCALATE
    assert decision.issued_contract is None
    assert decision.escalation is not None
    assert decision.emitted_messages[0].kind == "escalate"


def test_area_runtime_keeps_world_events_inside_local_regulation_loop() -> None:
    runtime = AreaRuntime(area_id="area_0", child_ids=("cell_0", "area_1"))
    activation = ActivationRecord(
        agent_id="area_0",
        activation_cause="world_event",
        functional_loop="local_regulation",
        activation_timestamp=9.0,
        event_summary=(
            HolonicEvent(
                time=9.0,
                kind="budget_alarm",
                scope="area_0",
                payload={"budget_gap": 3},
            ),
        ),
    )

    decision = runtime.handle_activation(activation)

    assert decision.functional_loop == "local_regulation"
    assert decision.propagation_action == PropagationAction.DEFER
    assert decision.issued_contract is None
