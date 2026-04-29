"""Tests for the v2 interpreter-driven tree runtime environment."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import BenchmarkConfig, PlantCellConfig, TreeMessage
from fms_bench.v2.controller_payloads import build_controller_payload
from fms_bench.v2.event_interpreter import interpret_decision_epoch
from fms_bench.v2.tree_runtime_env import TreeRuntimeEnv
from fms_bench.v2.intercell_tree_runtime_env import IntercellTreeRuntimeEnv


FLEX_ROOT = PROJECT_ROOT / "external" / "benchmarks" / "flexible jobshop" / "instances"


def _make_config() -> BenchmarkConfig:
    config = BenchmarkConfig()
    config.horizon_hours = 200
    config.failure.mtbf_hours = 0.0
    config.failure.mttr_hours = 0.0
    return config


def _cell_configs() -> list[PlantCellConfig]:
    common = {
        "family_key": "kacem",
        "instance_path": FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
        "instance_kind": "kacem",
        "wip_cap": 2,
    }
    return [
        PlantCellConfig(
            cell_id=0,
            label="cell-fast",
            processing_time_scale=0.8,
            energy_cap_kwh=650.0,
            carbon_cap_kg=390.0,
            **common,
        ),
        PlantCellConfig(
            cell_id=1,
            label="cell-slow",
            processing_time_scale=1.4,
            energy_cap_kwh=900.0,
            carbon_cap_kg=540.0,
            **common,
        ),
        PlantCellConfig(
            cell_id=2,
            label="cell-medium",
            processing_time_scale=1.0,
            energy_cap_kwh=800.0,
            carbon_cap_kg=480.0,
            **common,
        ),
    ]


def make_tree_runtime_env(*, authority_mode: str = "holonic_hybrid") -> TreeRuntimeEnv:
    return TreeRuntimeEnv(
        _cell_configs(),
        area_to_cells={0: [0, 1], 1: [2]},
        config=_make_config(),
        seed=13,
        authority_mode=authority_mode,
    )


def make_single_cell_tree_runtime_env(
    *,
    authority_mode: str = "holonic_hybrid",
) -> TreeRuntimeEnv:
    return TreeRuntimeEnv(
        [
            PlantCellConfig(
                cell_id=0,
                family_key="kacem",
                instance_path=FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
                instance_kind="kacem",
                label="cell-solo",
                wip_cap=1,
            ),
        ],
        area_to_cells={0: [0]},
        config=_make_config(),
        seed=13,
        authority_mode=authority_mode,
    )


def make_zero_wip_tree_runtime_env() -> TreeRuntimeEnv:
    return TreeRuntimeEnv(
        [
            PlantCellConfig(
                cell_id=0,
                family_key="kacem",
                instance_path=FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
                instance_kind="kacem",
                label="cell-zero-wip",
                wip_cap=0,
            ),
        ],
        area_to_cells={0: [0]},
        config=_make_config(),
        seed=13,
    )


def make_two_cell_single_area_tree_runtime_env() -> TreeRuntimeEnv:
    common = {
        "family_key": "kacem",
        "instance_path": FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
        "instance_kind": "kacem",
        "wip_cap": 1,
    }
    return TreeRuntimeEnv(
        [
            PlantCellConfig(
                cell_id=0,
                label="cell-fast",
                processing_time_scale=0.8,
                energy_cap_kwh=650.0,
                carbon_cap_kg=390.0,
                **common,
            ),
            PlantCellConfig(
                cell_id=1,
                label="cell-slow",
                processing_time_scale=1.4,
                energy_cap_kwh=900.0,
                carbon_cap_kg=540.0,
                **common,
            ),
        ],
        area_to_cells={0: [0, 1]},
        config=_make_config(),
        seed=13,
    )


def _drive_until_cell_epoch(env: TreeRuntimeEnv) -> None:
    env.reset(seed=13)
    env.step_dict({"plant": 0})
    env.step_dict({"area_0": 0})


def _drive_until_multi_cell_local_epoch(env: TreeRuntimeEnv) -> None:
    env.reset(seed=13)
    env.step_dict({"plant": 0})
    env.step_dict({"area_0": 0})
    env.step_dict({"cell_0": 0})
    env.step_dict({"plant": 0})
    env.step_dict({"area_0": 1})
    env.step_dict({"cell_1": 0})


def _drive_until_three_cell_dispatch_pressure_epoch(env: TreeRuntimeEnv) -> None:
    env.reset(seed=13)
    for action_dict in (
        {"plant": 1},
        {"area_1": 0},
        {"cell_2": 0},
        {"plant": 0},
        {"area_0": 0},
        {"cell_0": 0},
        {"plant": 1},
        {"area_1": 0},
        {"cell_2": 0},
        {"plant": 0},
        {"area_0": 1},
        {"cell_1": 0},
    ):
        env.step_dict(action_dict)


def test_tree_runtime_env_uses_interpreted_decision_epoch_for_agents_to_act() -> None:
    env = make_tree_runtime_env()
    try:
        _, info = env.reset(seed=13)

        assert env.agents_to_act() == ["plant"]
        assert info["agents_to_act"] == ["plant"]
        assert info["current_decision_epoch"]["agents_to_act"] == ["plant"]
        assert info["current_decision_epoch"]["execution_mode"] == "single_agent"
        assert info["current_decision_epoch"]["supported_execution_modes"] == [
            "single_agent",
            "same_timestamp_multi_cell_local_regulation",
        ]
        assert info["activation_trace"][0]["agent_id"] == "plant"
        assert info["activation_trace"][0]["activation_cause"] == "world_event"
    finally:
        env.close()


def test_tree_runtime_env_freezes_public_architecture_contracts() -> None:
    assert TreeRuntimeEnv.RUNTIME_LAYER_CONTRACT == (
        "manufacturing_operations_and_distributed_production_control_benchmark_runtime"
    )
    assert TreeRuntimeEnv.SHARED_DES_RUNTIME_BOUNDARY_CONTRACT == (
        "one_shared_des_world_tree_runtime_boundary"
    )
    assert TreeRuntimeEnv.RUNTIME_AGENT_BOUNDARY_CONTRACT == (
        "plant_area_cell_are_public_runtime_agents"
    )
    assert TreeRuntimeEnv.PUBLIC_EPOCH_SEMANTICS_CONTRACT == (
        "serialized_public_epochs_plus_limited_same_timestamp_multi_cell_local_regulation"
    )
    assert TreeRuntimeEnv.PUBLIC_ROLE_MAPPING_CONTRACT == (
        "benchmark_facing_plant_area_cell_role_mapping"
    )
    assert TreeRuntimeEnv.LITERATURE_MAPPING_BOUNDARY_CONTRACT == (
        "isa95_level3_prosa_adacor_inspired_not_direct_implementation"
    )
    assert TreeRuntimeEnv.benchmark_role_mapping() == {
        "plant": "operations_management_release_and_escalation_role",
        "area": "supervisor_or_sub_holon_coordination_role",
        "cell": "operational_unit_commitment_and_local_dispatch_role",
    }


def test_tree_runtime_env_routes_pending_activation_from_plant_to_area() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        step = env.step_dict({"plant": 0})

        trace_by_agent = {
            record["agent_id"]: record
            for record in step.info["activation_trace"]
        }

        assert trace_by_agent["area_0"]["activation_cause"] == "parent_message"
        assert env.agents_to_act() == ["area_0"]
        assert step.reward == 0.0
        assert step.info["protocol_trace"]
    finally:
        env.close()


def test_tree_runtime_env_runtime_state_kind_marks_release_blocked_when_no_release_is_legal() -> None:
    env = make_zero_wip_tree_runtime_env()
    try:
        _, info = env.reset(seed=13)

        assert env.agents_to_act() == []
        assert info["runtime_state_kind"] == "release_blocked"
    finally:
        env.close()


def test_tree_runtime_env_routes_pending_activation_from_area_to_selected_child() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        step = env.step_dict({"area_0": 1})

        trace_by_agent = {
            record["agent_id"]: record
            for record in step.info["activation_trace"]
        }

        assert "activation_trace" in step.info
        assert "protocol_trace" in step.info
        assert "world_event_trace" in step.info
        assert "current_decision_epoch" in step.info
        assert trace_by_agent["cell_1"]["activation_cause"] == "parent_message"
        assert env.agents_to_act() == ["cell_1"]
        assert step.info["protocol_trace"][0]["sender"] == "area_0"
        assert step.info["protocol_trace"][0]["recipient"] == "cell_1"
        assert step.info["agents_to_act"] == env.agents_to_act()
    finally:
        env.close()


def test_tree_runtime_env_rejects_actions_for_non_current_agents() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)

        with pytest.raises(KeyError, match="current decision epoch"):
            env.step_dict({"area_0": 0})
    finally:
        env.close()


def test_tree_runtime_env_advances_back_to_plant_after_cell_accepts_contract() -> None:
    env = make_single_cell_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        step = env.step_dict({"cell_0": 0})

        assert step.info["current_decision_epoch"]["agents_to_act"] == ["cell_0"]
        assert env.agents_to_act() == ["cell_0"]
        assert (
            step.info["current_decision_epoch"]["records_by_agent"]["cell_0"]["functional_loop"]
            == "local_regulation"
        )
        assert any(
            message["kind"] == "accept_contract"
            for message in step.info["protocol_trace"]
        )
    finally:
        env.close()


def test_tree_runtime_env_exposes_cell_dispatch_epoch_after_commitment_acceptance() -> None:
    env = make_single_cell_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        step = env.step_dict({"cell_0": 0})

        record = step.info["current_decision_epoch"]["records_by_agent"]["cell_0"]
        assert "dispatch_options" in record["context_handles"]
        assert "machine_idle" in str(record["event_summary"])
        assert len(record["context_handles"]["dispatch_options"]) > 0
        assert set(env.action_mask("cell_0").tolist()) <= {0, 1}
        assert int(sum(env.action_mask("cell_0").tolist())) == len(record["context_handles"]["dispatch_options"])
    finally:
        env.close()


def test_tree_runtime_env_runtime_state_kind_marks_local_dispatch_active() -> None:
    env = make_single_cell_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        step = env.step_dict({"cell_0": 0})

        assert step.info["runtime_state_kind"] == "local_dispatch_active"
    finally:
        env.close()


def test_tree_runtime_env_controller_cell_summary_surfaces_inbound_buffer_fields() -> None:
    env = TreeRuntimeEnv(
        [
            PlantCellConfig(
                cell_id=0,
                family_key="kacem",
                instance_path=FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
                instance_kind="kacem",
                label="cell-solo",
                wip_cap=1,
                inbound_buffer_cap=1,
                ingress_transport_time_hours=2.0,
                ingress_transport_energy_kwh=0.5,
            ),
        ],
        area_to_cells={0: [0]},
        config=_make_config(),
        seed=13,
    )
    try:
        env.reset(seed=13)
        env._env._release_next_job(0)

        summary = env.controller_cell_summary(0)

        assert summary["inbound_jobs"] == 1
        assert summary["inbound_buffer_cap"] == 1
        assert summary["inbound_buffer_remaining"] == 0
    finally:
        env.close()


def test_intercell_tree_runtime_env_routes_stage_completion_back_to_plant_release() -> None:
    env = IntercellTreeRuntimeEnv(
        [
            PlantCellConfig(
                cell_id=0,
                family_key="kacem",
                instance_path=FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
                instance_kind="kacem",
                label="cell-0",
                wip_cap=2,
                inbound_buffer_cap=1,
                ingress_transport_time_hours=1.0,
                ingress_transport_energy_kwh=0.2,
            ),
            PlantCellConfig(
                cell_id=1,
                family_key="kacem",
                instance_path=FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
                instance_kind="kacem",
                label="cell-1",
                wip_cap=2,
                inbound_buffer_cap=1,
                ingress_transport_time_hours=1.5,
                ingress_transport_energy_kwh=0.3,
            ),
            PlantCellConfig(
                cell_id=2,
                family_key="kacem",
                instance_path=FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
                instance_kind="kacem",
                label="cell-2",
                wip_cap=2,
                inbound_buffer_cap=1,
                ingress_transport_time_hours=1.5,
                ingress_transport_energy_kwh=0.3,
            ),
        ],
        area_to_cells={0: [0, 1], 1: [2]},
        config=_make_config(),
        seed=13,
    )
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_0": 0})
        step = env.step_dict({"cell_0": 0})

        assert env.agents_to_act() == ["plant"]
        assert step.info["runtime_state_kind"] == "decision_active"
        assert step.info["world_event_trace"]
        assert any(
            event["kind"] == "stage_release_ready"
            for event in step.info["world_event_trace"]
        )
    finally:
        env.close()


def test_intercell_tree_runtime_env_marks_release_blocked_when_next_stage_has_no_buffer_capacity() -> None:
    env = IntercellTreeRuntimeEnv(
        [
            PlantCellConfig(
                cell_id=0,
                family_key="kacem",
                instance_path=FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
                instance_kind="kacem",
                label="cell-0",
                wip_cap=2,
                inbound_buffer_cap=1,
                ingress_transport_time_hours=1.0,
                ingress_transport_energy_kwh=0.2,
            ),
            PlantCellConfig(
                cell_id=1,
                family_key="kacem",
                instance_path=FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
                instance_kind="kacem",
                label="cell-1",
                wip_cap=2,
                inbound_buffer_cap=0,
                ingress_transport_time_hours=1.5,
                ingress_transport_energy_kwh=0.3,
            ),
            PlantCellConfig(
                cell_id=2,
                family_key="kacem",
                instance_path=FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
                instance_kind="kacem",
                label="cell-2",
                wip_cap=2,
                inbound_buffer_cap=0,
                ingress_transport_time_hours=1.5,
                ingress_transport_energy_kwh=0.3,
            ),
        ],
        area_to_cells={0: [0, 1], 1: [2]},
        config=_make_config(),
        seed=13,
    )
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_0": 0})
        step = env.step_dict({"cell_0": 0})

        assert env.agents_to_act() == []
        assert step.info["runtime_state_kind"] == "release_blocked"
    finally:
        env.close()


def test_intercell_tree_runtime_env_advance_idle_epoch_ignores_non_assigned_ready_options() -> None:
    env = IntercellTreeRuntimeEnv(
        _cell_configs(),
        area_to_cells={0: [0, 1], 1: [2]},
        config=_make_config(),
        seed=13,
        max_in_flight_jobs=2,
    )
    try:
        env.reset(seed=13)
        contract = env._env._release_next_job(0)
        assert env._env._handle_transport_complete(
            {"job_id": contract.job_id, "to_cell": 0, "stage_id": 0}
        )

        assigned_machine_ids = {
            option.machine_id for option in env._env._dispatch_options_for_cell(0)
        }
        assert assigned_machine_ids

        env._env.simulator.events.clear()
        env._env.shared_backlog.clear()
        for machine_id in assigned_machine_ids:
            machine = env._env.simulator.machines[machine_id]
            machine.status = "busy"
            machine.current_job = contract.job_id

        repair_machine_id = next(
            machine_id
            for machine_id in env._env.simulator.machine_ids
            if machine_id not in assigned_machine_ids
        )
        env._env.simulator.machines[repair_machine_id].status = "down"
        env._env.simulator.schedule("repair", env._env.simulator.time + 1.0, {"machine_id": repair_machine_id})

        env.current_decision_epoch = env._build_current_epoch()
        assert env.agents_to_act() == []

        step = env.advance_idle_epoch()

        assert step.truncated is False
        assert step.terminated is False
        assert env._env.simulator.time == pytest.approx(1.0)
        assert step.info["runtime_state_kind"] == "decision_active"
        assert any(
            event["kind"] == "machine_repair"
            for event in step.info["world_event_trace"]
        )
    finally:
        env.close()


def test_tree_runtime_env_dispatch_step_advances_world_after_local_regulation() -> None:
    env = make_single_cell_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_0": 0})

        step = env.step_dict({"cell_0": 0})

        assert step.reward == step.info["reward_components"]["total"]
        assert step.info["reward_components"]["delta_time"] < 0.0
        assert step.info["current_decision_epoch"]["activation_timestamp"] > 0.0
        assert step.info["current_decision_epoch"]["agents_to_act"] == ["cell_0"]
        assert (
            step.info["current_decision_epoch"]["records_by_agent"]["cell_0"]["functional_loop"]
            == "local_regulation"
        )
        assert env.agents_to_act() == ["cell_0"]
    finally:
        env.close()


def test_tree_runtime_env_exposes_same_timestamp_multi_cell_dispatch_epoch() -> None:
    env = make_two_cell_single_area_tree_runtime_env()
    try:
        _drive_until_multi_cell_local_epoch(env)

        assert env.agents_to_act() == ["cell_0", "cell_1"]
        assert env.render()["current_decision_epoch"]["execution_mode"] == (
            "same_timestamp_multi_cell_local_regulation"
        )
        assert (
            env.current_decision_epoch.records_by_agent["cell_0"].functional_loop
            == "local_regulation"
        )
        assert env.current_decision_epoch.records_by_agent["cell_1"].functional_loop == "local_regulation"
        assert int(sum(env.action_mask("cell_0").tolist())) > 0
        assert int(sum(env.action_mask("cell_1").tolist())) > 0
    finally:
        env.close()


def test_tree_runtime_env_accepts_multi_cell_step_dict_for_same_timestamp_epoch() -> None:
    env = make_two_cell_single_area_tree_runtime_env()
    try:
        _drive_until_multi_cell_local_epoch(env)

        step = env.step_dict({"cell_0": 0, "cell_1": 0})

        assert step.reward == step.info["reward_components"]["total"]
        assert step.info["reward_components"]["delta_time"] < 0.0
        assert step.info["current_decision_epoch"]["activation_timestamp"] > 0.0
        assert step.info["agents_to_act"]
        assert set(step.info["agents_to_act"]) != {"cell_0", "cell_1"}
    finally:
        env.close()


def test_tree_runtime_env_does_not_truncate_on_zero_time_followup_dispatch_epoch() -> None:
    env = make_tree_runtime_env()
    try:
        _drive_until_three_cell_dispatch_pressure_epoch(env)

        step = env.step_dict({"cell_0": 0, "cell_1": 0, "cell_2": 0})

        assert step.truncated is False
        assert step.terminated is False
        assert step.info["current_decision_epoch"]["agents_to_act"]
    finally:
        env.close()


def test_tree_runtime_env_explicitly_rejects_unsupported_multi_agent_epoch_modes() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.current_decision_epoch = interpret_decision_epoch(
            world_time=float(env.simulator.time),
            events=(),
            messages=(
                TreeMessage(
                    sender="plant",
                    recipient="area_0",
                    kind="assignment_request",
                    payload={"contract_id": "c0"},
                ),
                TreeMessage(
                    sender="plant",
                    recipient="area_1",
                    kind="assignment_request",
                    payload={"contract_id": "c1"},
                ),
            ),
        )

        with pytest.raises(NotImplementedError, match="unsupported_multi_agent"):
            env.step_dict({"area_0": 0, "area_1": 0})
    finally:
        env.close()


def test_tree_runtime_env_render_marks_deadlock_runtime_state_kind_when_world_is_deadlocked() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        original_render = env._env.render

        def _deadlocked_render() -> dict[str, object]:
            payload = dict(original_render())
            payload["deadlocked"] = True
            return payload

        env._env.render = _deadlocked_render  # type: ignore[method-assign]
        rendered = env.render()

        assert rendered["runtime_state_kind"] == "deadlock"
    finally:
        env.close()


def test_tree_runtime_env_reports_settlement_trace_and_reward_components() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 1})
        step = env.step_dict({"cell_1": 0})

        assert "settlement_trace" in step.info
        assert step.info["settlement_trace"]
        assert "reward_components" in step.info
        assert step.info["reward_components"]["total"] == step.reward
        assert any(
            entry["status"] in {"active", "partially_completed", "completed"}
            for entry in step.info["settlement_trace"]
        )
    finally:
        env.close()


def test_tree_runtime_env_exposes_explicit_settlement_summary() -> None:
    env = make_single_cell_tree_runtime_env()
    try:
        env.reset(seed=13)
        final_step = None

        for _ in range(200):
            agents = env.agents_to_act()
            assert agents
            final_step = env.step_dict({agents[0]: 0})
            if any(
                entry["status"] == "settled"
                for entry in final_step.info["settlement_trace"]
            ):
                break
            if final_step.terminated or final_step.truncated:
                break

        assert final_step is not None
        assert "settlement_summary" in final_step.info
        assert final_step.info["settlement_summary"]["contracts_total"] >= 1
        assert final_step.info["settlement_summary"]["settled_contracts"] >= 1
        assert any(
            entry["status"] == "settled"
            for entry in final_step.info["settlement_trace"]
        )
    finally:
        env.close()


def test_tree_runtime_env_exposes_reward_authority_on_public_info() -> None:
    env = make_tree_runtime_env()
    try:
        _, info = env.reset(seed=13)

        assert info["reward_authority"] == "runtime_ledger_interval_v2"

        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 1})
        step = env.step_dict({"cell_1": 0})

        assert step.info["reward_authority"] == "runtime_ledger_interval_v2"
    finally:
        env.close()


def test_tree_runtime_env_exposes_public_machine_summaries() -> None:
    env = make_tree_runtime_env()
    try:
        _, info = env.reset(seed=13)

        machine_summaries = info["machine_summaries"]
        assert machine_summaries
        first_machine = machine_summaries[str(min(int(machine_id) for machine_id in machine_summaries))]
        assert set(first_machine) >= {
            "machine_id",
            "cell_id",
            "status",
            "current_job",
            "remaining_time",
            "setup_remaining_time",
            "last_setup_family",
            "setup_power_kw",
        }
        assert first_machine["status"] == "idle"

        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 1})
        step = env.step_dict({"cell_1": 0})

        assert "machine_summaries" in step.info
        assert set(step.info["machine_summaries"]) == set(machine_summaries)
        assert {
            summary["status"]
            for summary in step.info["machine_summaries"].values()
        } <= {"idle", "busy", "down", "setup"}
    finally:
        env.close()


def test_tree_runtime_controller_payload_for_area_hides_sibling_state() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=7)
        env.step_dict({"plant": 0})

        payload = build_controller_payload(env, "area_0")

        assert payload["agent_id"] == "area_0"
        assert payload["role"] == "area"
        assert "world_event_trace" not in payload
        assert "reward_components" not in payload
        assert "current_decision_epoch" not in payload
        assert "visible_sibling_summaries" not in payload
        assert 2 not in payload["local_subtree_state"]["child_ids"]
    finally:
        env.close()


def test_tree_runtime_controller_payload_for_cell_only_contains_local_feedback() -> None:
    env = make_tree_runtime_env()
    try:
        _drive_until_cell_epoch(env)

        payload = build_controller_payload(env, "cell_0")

        assert payload["agent_id"] == "cell_0"
        assert payload["role"] == "cell"
        assert set(payload["feedback_view"]).issubset(
            {"progress", "capacity_risk", "completion", "escalation"}
        )
    finally:
        env.close()


def test_area_controller_payload_contains_aggregated_feedback_only() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})

        payload = build_controller_payload(env, "area_0")
        feedback = payload["feedback_view"]

        assert set(feedback) == {"progress", "capacity_risk", "completion", "escalation"}
        assert "world_event_trace" not in feedback
        assert feedback["capacity_risk"]["valid_child_ids"] == [0, 1]
    finally:
        env.close()


def test_cell_controller_payload_contains_local_event_and_parent_context() -> None:
    env = make_single_cell_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_0": 0})

        payload = build_controller_payload(env, "cell_0")

        assert "parent_context" in payload
        assert "current_contract" in payload["parent_context"]
        assert "local_env_events" in payload
        assert payload["local_env_events"]
        assert payload["local_env_events"][0]["kind"] == "machine_idle"
    finally:
        env.close()


def test_tree_runtime_env_cell_commitment_epoch_exposes_two_legal_actions() -> None:
    env = make_single_cell_tree_runtime_env()
    try:
        _drive_until_cell_epoch(env)

        assert env.agents_to_act() == ["cell_0"]
        assert env.current_decision_epoch.records_by_agent["cell_0"].functional_loop == "commitment_management"
        assert env.action_mask("cell_0").tolist() == [1, 1]

        payload = build_controller_payload(env, "cell_0")
        assert payload["activation"]["functional_loop"] == "commitment_management"
        assert payload["action_semantics"]["action_type"] == "commitment_management"
        assert [target["action_id"] for target in payload["action_semantics"]["action_targets"]] == [0, 1]
    finally:
        env.close()


def test_tree_runtime_env_hierarchical_mode_cell_commitment_is_accept_only() -> None:
    env = make_single_cell_tree_runtime_env(authority_mode="hierarchical")
    try:
        _drive_until_cell_epoch(env)

        assert env.agents_to_act() == ["cell_0"]
        assert env.action_mask("cell_0").tolist() == [1]

        payload = build_controller_payload(env, "cell_0")
        assert payload["authority_mode"]["mode_id"] == "hierarchical"
        assert payload["action_semantics"]["action_targets"] == [
            {
                "action_id": 0,
                "target_id": "cell_0",
                "action_kind": "accept_contract",
            }
        ]
    finally:
        env.close()


def test_tree_runtime_env_cell_reject_keeps_backlog_and_routes_to_area_accountability() -> None:
    env = make_single_cell_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})

        backlog_before = len(env._env.shared_backlog)
        assigned_before = len(env._env.assigned_clone_jobs)

        step = env.step_dict({"cell_0": 1})

        assert any(message["kind"] == "reject_contract" for message in step.info["protocol_trace"])
        assert len(env._env.shared_backlog) == backlog_before
        assert len(env._env.assigned_clone_jobs) == assigned_before
        assert env.runtime_ledger.bound_job_ids() == ()
        assert any(entry["status"] == "failed" for entry in step.info["settlement_trace"])
        assert step.info["current_decision_epoch"]["agents_to_act"] == ["area_0"]
        assert (
            step.info["current_decision_epoch"]["records_by_agent"]["area_0"]["functional_loop"]
            == "accountability_and_escalation"
        )
        assert env.agents_to_act() == ["area_0"]
    finally:
        env.close()


def test_tree_runtime_env_centralized_mode_releases_job_without_area_or_cell_commitment_epoch() -> None:
    env = make_tree_runtime_env(authority_mode="centralized")
    try:
        env.reset(seed=13)
        backlog_before = len(env._env.shared_backlog)

        step = env.step_dict({"plant": 0})

        assert len(env._env.shared_backlog) == backlog_before - 1
        assert env.runtime_ledger.bound_job_ids()
        assert not any(
            record["functional_loop"] == "commitment_management"
            and str(record["agent_id"]).startswith(("area_", "cell_"))
            for record in step.info["activation_trace"]
        )
        assert not any(agent.startswith("area_") for agent in env.agents_to_act())
        assert not any(
            agent.startswith("cell_")
            and env.current_decision_epoch.records_by_agent[agent].functional_loop
            == "commitment_management"
            for agent in env.agents_to_act()
        )
    finally:
        env.close()


def test_tree_runtime_env_area_accountability_after_reject_escalates_back_to_plant() -> None:
    env = make_single_cell_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_0": 1})

        area_payload = build_controller_payload(env, "area_0")
        assert area_payload["activation"]["functional_loop"] == "accountability_and_escalation"
        assert area_payload["action_mask"].tolist() == [1]
        assert area_payload["local_subtree_state"]["valid_children"] == []
        assert area_payload["valid_cells"] == []
        assert area_payload["action_semantics"]["action_type"] == "accountability_and_escalation"
        assert area_payload["action_semantics"]["action_targets"] == [
            {
                "action_id": 0,
                "target_id": "area_0",
                "action_kind": "process_child_feedback",
            }
        ]
        assert area_payload["visible_protocol_objects"][0]["kind"] == "reject_contract"

        step = env.step_dict({"area_0": 0})

        assert any(message["kind"] == "escalate" for message in step.info["protocol_trace"])
        assert step.info["current_decision_epoch"]["agents_to_act"] == ["plant"]
        assert (
            step.info["current_decision_epoch"]["records_by_agent"]["plant"]["functional_loop"]
            == "accountability_and_escalation"
        )
        assert env.agents_to_act() == ["plant"]
        assert any(entry["status"] == "failed" for entry in step.info["settlement_trace"])
    finally:
        env.close()


def test_tree_runtime_env_mediated_cnp_area_payload_exposes_bid_award_semantics() -> None:
    env = make_tree_runtime_env(authority_mode="mediated_heterarchical_cnp")
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})

        payload = build_controller_payload(env, "area_0")

        assert payload["authority_mode"]["mode_id"] == "mediated_heterarchical_cnp"
        assert payload["action_semantics"]["action_type"] == "award_bid"
        assert payload["local_subtree_state"]["candidate_bids"]
        assert [bid["cell_id"] for bid in payload["local_subtree_state"]["candidate_bids"]] == [0, 1]
        assert all(
            "bid_score" in target
            for target in payload["action_semantics"]["action_targets"]
        )
    finally:
        env.close()


def test_tree_runtime_env_mediated_cnp_reject_escalates_to_plant_without_area_local_reroute() -> None:
    env = make_tree_runtime_env(authority_mode="mediated_heterarchical_cnp")
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_0": 1})

        assert env.agents_to_act() == ["area_0"]

        step = env.step_dict({"area_0": 0})

        assert any(message["kind"] == "escalate" for message in step.info["protocol_trace"])
        assert not any(
            message["kind"] == "assignment_contract"
            for message in step.info["protocol_trace"]
        )
        assert env.agents_to_act() == ["plant"]
    finally:
        env.close()


def test_tree_runtime_env_area_accountability_can_reroute_once_to_sibling_cell() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_0": 1})

        area_payload = build_controller_payload(env, "area_0")
        assert area_payload["activation"]["functional_loop"] == "accountability_and_escalation"
        assert area_payload["action_mask"].tolist() == [1]
        assert area_payload["valid_cells"] == []

        step = env.step_dict({"area_0": 0})

        assert any(message["kind"] == "assignment_contract" for message in step.info["protocol_trace"])
        assert any(message["recipient"] == "cell_1" for message in step.info["protocol_trace"])
        assert step.info["current_decision_epoch"]["agents_to_act"] == ["cell_1"]
        assert (
            step.info["current_decision_epoch"]["records_by_agent"]["cell_1"]["functional_loop"]
            == "commitment_management"
        )
        assert env.agents_to_act() == ["cell_1"]
    finally:
        env.close()


def test_tree_runtime_env_second_reject_after_reroute_escalates_to_plant() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_0": 1})
        env.step_dict({"area_0": 0})

        assert env.agents_to_act() == ["cell_1"]
        env.step_dict({"cell_1": 1})
        assert env.agents_to_act() == ["area_0"]

        step = env.step_dict({"area_0": 0})

        assert any(message["kind"] == "escalate" for message in step.info["protocol_trace"])
        assert not any(
            message["kind"] == "assignment_contract" and message.get("recipient") == "cell_0"
            for message in step.info["protocol_trace"]
        )
        assert step.info["current_decision_epoch"]["agents_to_act"] == ["plant"]
        assert env.agents_to_act() == ["plant"]
    finally:
        env.close()


def test_tree_runtime_env_plant_accountability_can_revise_to_other_area() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_0": 1})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_1": 1})
        env.step_dict({"area_0": 0})

        assert env.agents_to_act() == ["plant"]

        plant_payload = build_controller_payload(env, "plant")
        assert plant_payload["activation"]["functional_loop"] == "accountability_and_escalation"
        assert plant_payload["valid_areas"] == [1]
        assert plant_payload["action_mask"].tolist() == [0, 1]
        assert plant_payload["visible_protocol_objects"][0]["kind"] == "escalate"
        assert plant_payload["action_semantics"]["action_type"] == "revise_contract"
        assert plant_payload["action_semantics"]["action_targets"] == [
            {
                "action_id": 1,
                "target_id": 1,
                "action_kind": "revise_contract",
            }
        ]

        step = env.step_dict({"plant": 1})

        revise_message = next(
            message
            for message in step.info["protocol_trace"]
            if message["kind"] == "revise_contract"
        )
        assert revise_message["recipient"] == "area_1"
        assert revise_message["payload"]["upstream_contract_id"].startswith("req@")
        assert revise_message["payload"]["attempted_area_ids"] == ["area_0", "area_1"]
        assert step.info["current_decision_epoch"]["agents_to_act"] == ["area_1"]
        assert (
            step.info["current_decision_epoch"]["records_by_agent"]["area_1"]["functional_loop"]
            == "commitment_management"
        )
    finally:
        env.close()


def test_tree_runtime_env_marks_plant_accountability_epoch_as_accountability_reroute() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_0": 1})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_1": 1})
        env.step_dict({"area_0": 0})

        assert env.agents_to_act() == ["plant"]
        assert (
            env.current_decision_epoch.records_by_agent["plant"].decision_kind
            == "accountability_reroute"
        )

        payload = build_controller_payload(env, "plant")

        assert payload["plant_decision_kind"] == "accountability_reroute"
        assert payload["action_semantics"]["action_type"] == "revise_contract"
    finally:
        env.close()


def test_tree_runtime_env_exhausted_plant_accountability_rebuilds_normal_epoch() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_0": 1})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_1": 1})
        env.step_dict({"area_0": 0})
        env.step_dict({"plant": 1})
        env.step_dict({"area_1": 2})
        env.step_dict({"cell_2": 1})
        env.step_dict({"area_1": 0})

        assert env.agents_to_act() == ["plant"]
        assert (
            env.current_decision_epoch.records_by_agent["plant"].decision_kind
            == "accountability_reroute"
        )

        step = env.step_dict({"plant": 0})

        assert step.info["accountability_reroute_exhausted"] is True
        assert env.agents_to_act() == ["plant"]
        assert (
            env.current_decision_epoch.records_by_agent["plant"].functional_loop
            == "commitment_management"
        )

        payload = build_controller_payload(env, "plant")
        assert payload["plant_decision_kind"] != "accountability_reroute"
        assert payload["action_semantics"]["action_type"] == "select_child_holon"
    finally:
        env.close()


def test_tree_runtime_env_truncates_protocol_livelock_after_repeated_reject_cycles() -> None:
    env = make_single_cell_tree_runtime_env()
    try:
        env.reset(seed=13)
        final_step = None

        for _ in range(200):
            active_agents = env.agents_to_act()
            assert active_agents
            agent_id = active_agents[0]
            mask = env.action_mask(agent_id).tolist()
            action = 1 if agent_id.startswith("cell_") and len(mask) > 1 and mask[1] else 0
            final_step = env.step_dict({agent_id: action})
            if final_step.truncated:
                break

        assert final_step is not None
        assert final_step.truncated is True
        assert final_step.terminated is False
        assert final_step.info["protocol_livelock_detected"] is True
        assert final_step.info["truncated_reason"] == "protocol_livelock"
        assert env.agents_to_act() == []
    finally:
        env.close()


def test_tree_runtime_env_revised_contract_flows_into_new_area_child_commitment() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_0": 1})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_1": 1})
        env.step_dict({"area_0": 0})
        env.step_dict({"plant": 1})

        step = env.step_dict({"area_1": 2})

        assignment_message = next(
            message
            for message in step.info["protocol_trace"]
            if message["kind"] == "assignment_contract"
        )
        assert assignment_message["recipient"] == "cell_2"
        assert assignment_message["payload"]["upstream_contract_id"].startswith("req@")
        assert assignment_message["payload"]["attempted_area_ids"] == ["area_0", "area_1"]
        assert step.info["current_decision_epoch"]["agents_to_act"] == ["cell_2"]
        assert (
            step.info["current_decision_epoch"]["records_by_agent"]["cell_2"]["functional_loop"]
            == "commitment_management"
        )
    finally:
        env.close()


def test_tree_runtime_env_plant_assignment_request_uses_current_fifo_shared_backlog_item() -> None:
    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        assert len(env._env.shared_backlog) >= 2
        first = int(env._env.shared_backlog[0])
        second = int(env._env.shared_backlog[1])
        env._env.shared_backlog[0] = second
        env._env.shared_backlog[1] = first

        step = env.step_dict({"plant": 0})

        assert step.info["protocol_trace"][0]["kind"] == "assignment_request"
        assert step.info["protocol_trace"][0]["payload"]["shared_backlog_id"] == second
        assert step.info["protocol_trace"][0]["payload"]["work_id"] == f"shared_backlog_{second}"
    finally:
        env.close()


def test_tree_runtime_controller_payload_does_not_depend_on_global_render_snapshot() -> None:
    def _forbidden_render() -> dict[str, object]:
        raise AssertionError("controller payload should not call env.render()")

    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        setattr(env, "render", _forbidden_render)
        plant_payload = build_controller_payload(env, "plant")
        assert plant_payload["local_subtree_state"]["valid_children"] == [0, 1]
    finally:
        env.close()

    env = make_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        setattr(env, "render", _forbidden_render)
        area_payload = build_controller_payload(env, "area_0")
        assert area_payload["local_subtree_state"]["valid_children"] == [0, 1]
    finally:
        env.close()

    env = make_single_cell_tree_runtime_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        setattr(env, "render", _forbidden_render)
        cell_payload = build_controller_payload(env, "cell_0")
        assert cell_payload["local_subtree_state"]["cell_id"] == 0
    finally:
        env.close()
