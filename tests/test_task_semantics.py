from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_task_semantics_freezes_current_public_propagation_subset() -> None:
    from fms_bench.v2.task_semantics import (
        IMPLEMENTED_PROPAGATION_ACTIONS,
        PUBLIC_WORK_ITEM_CONTRACT,
        RESERVED_PROPAGATION_ACTIONS,
        STABLE_PUBLIC_COMMITMENT_STATUSES,
    )

    assert PUBLIC_WORK_ITEM_CONTRACT == "next_fifo_shared_backlog_item_assignment_propagation"
    assert [action.value for action in IMPLEMENTED_PROPAGATION_ACTIONS] == [
        "assign_to_subholon",
        "defer",
        "reroute",
        "revise_contract",
        "escalate",
        "settle",
    ]
    assert [action.value for action in RESERVED_PROPAGATION_ACTIONS] == [
        "accept",
        "reject",
        "split",
        "merge",
    ]
    assert [status.value for status in STABLE_PUBLIC_COMMITMENT_STATUSES] == [
        "offered",
        "active",
        "partially_completed",
        "completed",
        "failed",
        "settled",
    ]
