from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from fms_bench.holonic_types import HolonicEvent, TreeMessage
from fms_bench.v2.event_interpreter import interpret_decision_epoch


def test_interpreter_groups_same_timestamp_events_into_one_decision_epoch() -> None:
    events = [
        HolonicEvent(time=5.0, kind="operation_finish", scope="cell_0", payload={}),
        HolonicEvent(time=5.0, kind="budget_alarm", scope="area_0", payload={}),
    ]

    epoch = interpret_decision_epoch(world_time=5.0, events=events)

    assert epoch.activation_timestamp == 5.0
    assert set(epoch.agents_to_act) == {"area_0", "cell_0"}


def test_world_events_are_distinct_from_parent_child_messages() -> None:
    events = [
        HolonicEvent(time=8.0, kind="budget_alarm", scope="area_0", payload={}),
    ]
    messages = [
        TreeMessage(
            sender="area_0",
            recipient="cell_0",
            kind="allocate",
            payload={},
        )
    ]

    epoch = interpret_decision_epoch(world_time=8.0, events=events, messages=messages)

    assert epoch.records_by_agent["area_0"].activation_cause == "world_event"
    assert epoch.records_by_agent["cell_0"].activation_cause == "parent_message"


def test_escalate_message_activates_accountability_loop_for_parent() -> None:
    epoch = interpret_decision_epoch(
        world_time=9.0,
        events=(),
        messages=(
            TreeMessage(
                sender="area_0",
                recipient="plant",
                kind="escalate",
                payload={"contract_id": "c1", "status": "failed"},
            ),
        ),
    )

    assert epoch.records_by_agent["plant"].activation_cause == "parent_message"
    assert (
        epoch.records_by_agent["plant"].functional_loop
        == "accountability_and_escalation"
    )
