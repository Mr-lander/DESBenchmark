"""Tests for runtime accounting of contract settlement and reward intervals."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench.holonic_types import TreeMessage
from fms_bench.v2.runtime_ledger import RuntimeLedger
from fms_bench.v2.task_semantics import CommitmentStatus, Contract


def test_partial_physical_completion_does_not_imply_completed_contract() -> None:
    ledger = RuntimeLedger()
    contract = Contract(
        contract_id="c1",
        work_id="w1",
        sender="area_0",
        recipient="cell_0",
    )

    ledger.apply_protocol_object(contract)
    ledger.bind_job(contract_id="c1", job_id=3, total_operations=4)
    ledger.mark_contract_active(contract_id="c1", accepted_by="cell_0", timestamp=5.0)
    ledger.record_physical_progress(
        job_id=3,
        completed_operations=1,
        completed=False,
        timestamp=7.0,
    )

    assert ledger.contract_status("c1") == CommitmentStatus.PARTIALLY_COMPLETED


def test_reward_is_computed_from_interval_deltas_not_fixed_rounds() -> None:
    ledger = RuntimeLedger()

    reward = ledger.finalize_interval(
        start_time=5.0,
        end_time=8.0,
        energy_kwh=2.0,
        carbon_kg=1.0,
        invalid_action=False,
        terminated=False,
        deadlocked=False,
        tardiness_before=4.0,
        tardiness_after=5.5,
    )

    assert reward.components["delta_time"] == -3.0
    assert reward.components["delta_tardiness"] == -1.5
    assert reward.total < 0.0


def test_runtime_ledger_accumulates_overshoot_duration_and_frequency() -> None:
    ledger = RuntimeLedger()

    ledger.finalize_interval(
        start_time=0.0,
        end_time=2.0,
        energy_kwh=5.0,
        carbon_kg=1.0,
        invalid_action=False,
        terminated=False,
        deadlocked=False,
        tardiness_before=0.0,
        tardiness_after=0.0,
        energy_overshoot_kwh=1.5,
        carbon_overshoot_kg=0.0,
    )
    ledger.finalize_interval(
        start_time=2.0,
        end_time=5.0,
        energy_kwh=3.0,
        carbon_kg=6.0,
        invalid_action=False,
        terminated=False,
        deadlocked=False,
        tardiness_before=0.0,
        tardiness_after=0.0,
        energy_overshoot_kwh=0.0,
        carbon_overshoot_kg=2.0,
    )

    summary = ledger.violation_summary()

    assert summary["energy_overshoot_duration"] == 2.0
    assert summary["energy_overshoot_frequency"] == 1
    assert summary["carbon_overshoot_duration"] == 3.0
    assert summary["carbon_overshoot_frequency"] == 1
    assert summary["energy_overshoot_magnitude_total"] == 1.5
    assert summary["carbon_overshoot_magnitude_total"] == 2.0


def test_completed_contract_can_transition_to_settled_without_regression() -> None:
    ledger = RuntimeLedger()
    contract = Contract(
        contract_id="c1",
        work_id="w1",
        sender="area_0",
        recipient="cell_0",
    )

    ledger.apply_protocol_object(contract)
    ledger.bind_job(contract_id="c1", job_id=3, total_operations=4)
    ledger.mark_contract_active(contract_id="c1", accepted_by="cell_0", timestamp=5.0)
    ledger.record_physical_progress(
        job_id=3,
        completed_operations=4,
        completed=True,
        timestamp=7.0,
    )

    assert ledger.contract_status("c1") == CommitmentStatus.COMPLETED

    trace_len_before_settlement = len(ledger.settlement_trace())
    ledger.mark_contract_settled(contract_id="c1", timestamp=8.0)

    assert ledger.contract_status("c1") == CommitmentStatus.SETTLED
    assert ledger.settlement_trace()[-1]["status"] == "settled"
    assert len(ledger.settlement_trace()) == trace_len_before_settlement + 1

    ledger.record_physical_progress(
        job_id=3,
        completed_operations=4,
        completed=True,
        timestamp=9.0,
    )
    ledger.mark_contract_settled(contract_id="c1", timestamp=10.0)

    assert ledger.contract_status("c1") == CommitmentStatus.SETTLED
    assert len(ledger.settlement_trace()) == trace_len_before_settlement + 1


def test_runtime_ledger_reports_public_settlement_summary() -> None:
    ledger = RuntimeLedger()
    contract = Contract(
        contract_id="c1",
        work_id="w1",
        sender="area_0",
        recipient="cell_0",
    )

    ledger.apply_protocol_object(contract)
    ledger.bind_job(contract_id="c1", job_id=3, total_operations=2)
    ledger.mark_contract_active(contract_id="c1", accepted_by="cell_0", timestamp=1.0)
    ledger.record_physical_progress(
        job_id=3,
        completed_operations=2,
        completed=True,
        timestamp=2.0,
    )
    ledger.mark_contract_settled(contract_id="c1", timestamp=2.0)

    summary = ledger.settlement_summary()

    assert summary["contracts_total"] == 1
    assert summary["settled_contracts"] == 1
    assert summary["completed_unsettled_contracts"] == 0
    assert summary["open_contracts"] == 0


def test_failed_reject_escalation_does_not_regress_contract_status() -> None:
    ledger = RuntimeLedger()
    contract = Contract(
        contract_id="c1",
        work_id="w1",
        sender="area_0",
        recipient="cell_0",
    )

    ledger.apply_protocol_object(contract)
    ledger.apply_protocol_object(
        TreeMessage(
            sender="cell_0",
            recipient="area_0",
            kind="reject_contract",
            payload={"contract_id": "c1"},
        ),
        timestamp=3.0,
    )
    trace_len_before_escalation = len(ledger.settlement_trace())

    ledger.apply_protocol_object(
        TreeMessage(
            sender="area_0",
            recipient="plant",
            kind="escalate",
            payload={"contract_id": "c1", "status": "failed", "area_id": "area_0"},
        ),
        timestamp=4.0,
    )

    assert ledger.contract_status("c1") == CommitmentStatus.FAILED
    assert len(ledger.settlement_trace()) == trace_len_before_escalation
