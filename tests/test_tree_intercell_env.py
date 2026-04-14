"""Tests for the first inter-cell tree runtime environment slice."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import BenchmarkConfig, PlantCellConfig, TreeIntercellEnv
from fms_bench.arrival_profiles import apply_arrival_profile


FLEX_ROOT = PROJECT_ROOT / "external" / "benchmarks" / "flexible jobshop" / "instances"


def _make_config() -> BenchmarkConfig:
    config = BenchmarkConfig()
    config.horizon_hours = 500
    config.failure.mtbf_hours = 0.0
    config.failure.mttr_hours = 0.0
    return config


def _cell_configs(
    *,
    stage2_buffer_cap: int = 1,
    stage3_buffer_cap: int = 1,
) -> list[PlantCellConfig]:
    common = {
        "family_key": "kacem",
        "instance_path": FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
        "instance_kind": "kacem",
        "wip_cap": 2,
    }
    return [
        PlantCellConfig(
            cell_id=0,
            label="cell-0",
            processing_time_scale=0.8,
            inbound_buffer_cap=1,
            ingress_transport_time_hours=1.0,
            ingress_transport_energy_kwh=0.2,
            **common,
        ),
        PlantCellConfig(
            cell_id=1,
            label="cell-1",
            processing_time_scale=1.0,
            inbound_buffer_cap=stage2_buffer_cap,
            ingress_transport_time_hours=1.5,
            ingress_transport_energy_kwh=0.3,
            **common,
        ),
        PlantCellConfig(
            cell_id=2,
            label="cell-2",
            processing_time_scale=1.2,
            inbound_buffer_cap=stage3_buffer_cap,
            ingress_transport_time_hours=1.5,
            ingress_transport_energy_kwh=0.3,
            **common,
        ),
    ]


def make_intercell_env(
    *,
    stage2_buffer_cap: int = 1,
    stage3_buffer_cap: int = 1,
    max_in_flight_jobs: int = 1,
    arrival_profile_id: str | None = None,
    initial_visible_backlog: int | None = None,
    ) -> TreeIntercellEnv:
    config = _make_config()
    if arrival_profile_id is not None:
        config = apply_arrival_profile(config, arrival_profile_id)
    if initial_visible_backlog is not None:
        config.arrival.initial_visible_backlog = int(initial_visible_backlog)
    return TreeIntercellEnv(
        _cell_configs(
            stage2_buffer_cap=stage2_buffer_cap,
            stage3_buffer_cap=stage3_buffer_cap,
        ),
        area_to_cells={0: [0, 1], 1: [2]},
        config=config,
        seed=13,
        max_in_flight_jobs=max_in_flight_jobs,
    )


def _cell_configs_nine() -> list[PlantCellConfig]:
    common = {
        "family_key": "brandimarte",
        "instance_path": FLEX_ROOT / "Brandimarte1993" / "mk01.txt",
        "instance_kind": "brandimarte",
        "wip_cap": 2,
    }
    return [
        PlantCellConfig(cell_id=0, label="a0-front", processing_time_scale=0.84, inbound_buffer_cap=1, ingress_transport_time_hours=1.0, ingress_transport_energy_kwh=0.2, **common),
        PlantCellConfig(cell_id=1, label="a0-mid", processing_time_scale=1.02, inbound_buffer_cap=1, ingress_transport_time_hours=1.5, ingress_transport_energy_kwh=0.3, **common),
        PlantCellConfig(cell_id=2, label="a0-tail", processing_time_scale=1.20, inbound_buffer_cap=1, ingress_transport_time_hours=1.5, ingress_transport_energy_kwh=0.3, **common),
        PlantCellConfig(cell_id=3, label="a1-front", processing_time_scale=0.92, inbound_buffer_cap=1, ingress_transport_time_hours=1.0, ingress_transport_energy_kwh=0.2, **common),
        PlantCellConfig(cell_id=4, label="a1-mid", processing_time_scale=1.18, inbound_buffer_cap=1, ingress_transport_time_hours=1.5, ingress_transport_energy_kwh=0.3, **common),
        PlantCellConfig(cell_id=5, label="a1-tail", processing_time_scale=1.34, inbound_buffer_cap=1, ingress_transport_time_hours=1.5, ingress_transport_energy_kwh=0.3, **common),
        PlantCellConfig(cell_id=6, label="a2-front", processing_time_scale=0.80, inbound_buffer_cap=1, ingress_transport_time_hours=1.0, ingress_transport_energy_kwh=0.2, **common),
        PlantCellConfig(cell_id=7, label="a2-mid", processing_time_scale=0.98, inbound_buffer_cap=1, ingress_transport_time_hours=1.5, ingress_transport_energy_kwh=0.3, **common),
        PlantCellConfig(cell_id=8, label="a2-tail", processing_time_scale=1.16, inbound_buffer_cap=1, ingress_transport_time_hours=1.5, ingress_transport_energy_kwh=0.3, **common),
    ]


def test_tree_intercell_env_initial_stage_exposes_route_grammar_candidates() -> None:
    env = make_intercell_env()
    try:
        _, info = env.reset(seed=13)

        assert env.shared_backlog
        assert env.agents_to_act() == ["plant"]
        assert env._valid_release_cells() == [0, 1]
        assert info["plant"]["completed_jobs_total"] == 0
        assert info["route_stage_pending"][str(env.shared_backlog[0])] == 0
    finally:
        env.close()


def test_tree_intercell_env_stage_finish_requeues_same_order_for_next_stage() -> None:
    env = make_intercell_env()
    try:
        env.reset(seed=13)
        contract = env._release_next_job(0)
        env._advance_until_decision(auto_dispatch=False, stop_on_dispatch=True)
        option = env._dispatch_options_for_cell(0)[0]
        env.dispatch_job_to_machine(option.job_id, option.machine_id)

        transition = env._advance_until_decision(auto_dispatch=False, stop_on_dispatch=True)

        assert "stage_release_ready" in transition["events"]
        assert "transport_complete" not in transition["events"]
        assert contract.job_id in env.shared_backlog
        assert env.route_stage_index[contract.job_id] == 1
        assert env._valid_release_cells() == [1, 2]
        assert env.render()["last_contract"]["job_id"] == contract.job_id
        assert [message["kind"] for message in env.render()["messages"]] == [
            "assignment_request",
            "assignment_contract",
        ]
    finally:
        env.close()


def test_tree_intercell_env_downstream_full_causes_release_blocked_next_stage() -> None:
    env = make_intercell_env(stage2_buffer_cap=0, stage3_buffer_cap=0)
    try:
        env.reset(seed=13)
        contract = env._release_next_job(0)
        env._advance_until_decision(auto_dispatch=False, stop_on_dispatch=True)
        option = env._dispatch_options_for_cell(0)[0]
        env.dispatch_job_to_machine(option.job_id, option.machine_id)
        transition = env._advance_until_decision(auto_dispatch=False, stop_on_dispatch=True)

        assert "blocking_start" in transition["events"]
        assert "buffer_wait_start" in transition["events"]
        assert option.job_id not in env.shared_backlog
        assert env.blocked_backlog == [contract.job_id]
        assert transition["time_end"] == pytest.approx(env.simulator.time)
        assert env.route_stage_index[option.job_id] == 1
        assert env._valid_release_cells() == []
        assert env.agents_to_act() == []
        assert env.render()["action_mask"] == [0, 0]
        assert env.render()["plant"]["blocked_jobs_total"] == 1
        assert env.render()["route_stage_blocked"] == {str(contract.job_id): 1}
    finally:
        env.close()


def test_tree_intercell_env_buffer_slot_release_later_unblocks_waiting_job_and_accumulates_blocking_time() -> None:
    env = make_intercell_env(stage2_buffer_cap=1, stage3_buffer_cap=0)
    try:
        env.reset(seed=13)
        blocked_contract = env._release_next_job(0)
        env._cell_inbound_jobs[1].add(999)

        env._advance_until_decision(auto_dispatch=False, stop_on_dispatch=True)
        option = env._dispatch_options_for_cell(0)[0]
        env.dispatch_job_to_machine(option.job_id, option.machine_id)
        blocked_transition = env._advance_until_decision(auto_dispatch=False, stop_on_dispatch=True)

        assert blocked_contract.job_id == option.job_id
        assert "blocking_start" in blocked_transition["events"]
        assert blocked_contract.job_id not in env.shared_backlog
        assert env.blocked_backlog == [blocked_contract.job_id]

        env.simulator.schedule("noop", env.simulator.time + 3.0, {})
        env._advance_until_decision(auto_dispatch=False, stop_on_dispatch=True)
        env._cell_inbound_jobs[1].clear()
        unblocked_transition = env._advance_until_decision(auto_dispatch=False, stop_on_dispatch=True)
        info = env.render()

        assert "buffer_admit" in unblocked_transition["events"]
        assert "blocking_end" in unblocked_transition["events"]
        assert "stage_release_ready" in unblocked_transition["events"]
        assert blocked_contract.job_id not in env.blocked_backlog
        assert blocked_contract.job_id in env.shared_backlog
        assert info["blocking_time_total"] == pytest.approx(3.0)
        assert info["buffer_wait_time_total"] == pytest.approx(3.0)
    finally:
        env.close()


def test_tree_intercell_env_blocking_events_are_exposed_in_typed_trace() -> None:
    env = make_intercell_env(stage2_buffer_cap=0, stage3_buffer_cap=0)
    try:
        env.reset(seed=13)
        env._release_next_job(0)
        env._advance_until_decision(auto_dispatch=False, stop_on_dispatch=True)
        option = env._dispatch_options_for_cell(0)[0]
        env.dispatch_job_to_machine(option.job_id, option.machine_id)
        transition = env._advance_until_decision(auto_dispatch=False, stop_on_dispatch=True)

        kinds = [event.kind for event in env._typed_events(transition)]

        assert "buffer_wait_start" in kinds
        assert "blocking_start" in kinds
        assert "buffer_blocked" in kinds
    finally:
        env.close()


def test_tree_intercell_env_render_exposes_blocked_job_details() -> None:
    env = make_intercell_env(stage2_buffer_cap=0, stage3_buffer_cap=0)
    try:
        env.reset(seed=13)
        contract = env._release_next_job(0)
        env._advance_until_decision(auto_dispatch=False, stop_on_dispatch=True)
        option = env._dispatch_options_for_cell(0)[0]
        env.dispatch_job_to_machine(option.job_id, option.machine_id)
        env._last_transition = env._advance_until_decision(auto_dispatch=False, stop_on_dispatch=True)
        rendered = env.render()

        assert rendered["blocked_job_ids"] == [contract.job_id]
        assert rendered["blocked_job_details"][str(contract.job_id)]["candidate_cells"] == [1, 2]
        assert rendered["plant"]["buffer_waiting_jobs_total"] == 1
    finally:
        env.close()


def test_tree_intercell_env_nhpp_batch_arrivals_are_traceable_before_first_release() -> None:
    env = make_intercell_env(
        arrival_profile_id="arrival_nhpp_random_batch_nominal_v1",
        initial_visible_backlog=0,
    )
    try:
        _, info = env.reset(seed=13)

        assert env.shared_backlog == []
        assert info["plant"]["future_backlog_jobs_total"] > 0

        transition = env._advance_until_decision(auto_dispatch=False, stop_on_dispatch=True)
        rendered = env.render()

        assert "shared_backlog_arrival" in transition["events"]
        assert rendered["arrival_events_total"] >= 1
        assert rendered["arrived_jobs_total"] >= 2
        assert rendered["arrival_batch_size_mean"] >= 2.0
        assert rendered["realized_arrival_trace"]
    finally:
        env.close()


def test_tree_intercell_env_area_and_cell_feedback_do_not_double_count_completed_jobs() -> None:
    env = make_intercell_env()
    try:
        env.reset(seed=13)
        env.simulator.completed_jobs.add(0)
        env.route_previous_cell[0] = 0

        cell0 = env.render()["cell_summaries"][0]
        cell1 = env.render()["cell_summaries"][1]
        area0 = env.render()["area_summaries"][0]

        assert cell0["completed_jobs"] == 1
        assert cell1["completed_jobs"] == 0
        assert area0["completed_jobs"] == 1
    finally:
        env.close()


def test_tree_intercell_env_can_admit_second_dormant_job_when_inflight_limit_allows() -> None:
    env = make_intercell_env(max_in_flight_jobs=2)
    try:
        env.reset(seed=13)
        env._release_next_job(0)

        assert env.shared_backlog
        assert env.shared_backlog[0] == 1
        assert env._valid_release_cells() == [1]
    finally:
        env.close()


def test_tree_intercell_env_blocks_second_dormant_job_when_inflight_limit_is_exhausted() -> None:
    env = make_intercell_env(max_in_flight_jobs=1)
    try:
        env.reset(seed=13)
        env._release_next_job(0)

        assert env.shared_backlog
        assert env.shared_backlog[0] == 1
        assert env._valid_release_cells() == []
    finally:
        env.close()


def test_tree_intercell_env_accepts_custom_a3c9_route_grammar() -> None:
    env = TreeIntercellEnv(
        _cell_configs_nine(),
        area_to_cells={0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]},
        config=_make_config(),
        seed=13,
        max_in_flight_jobs=3,
        route_grammar=((0, 1, 3, 4, 6, 7), (1, 2, 4, 5, 7, 8), (2, 5, 8)),
    )
    try:
        _, info = env.reset(seed=13)

        assert env.area_ids == [0, 1, 2]
        assert sorted(env._valid_release_cells()) == [0, 1, 3, 4, 6, 7]
        assert env.render()["action_mask"] == [1, 1, 1]
        first_job_id = env.shared_backlog[0]
        assert info["route_stage_pending"][str(first_job_id)] == 0
    finally:
        env.close()
