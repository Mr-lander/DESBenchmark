"""A3C9 runtime-contract tests for Category B inter-cell flow."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench.experiment_runner import (
    build_tree_runtime_v2_intercell_a3c9_probe_suite,
    make_regime_config,
)
from fms_bench.v2.controller_payloads import build_controller_payload
from fms_bench.v2.intercell_tree_runtime_env import IntercellTreeRuntimeEnv


_CANONICAL_ROUTE_GRAMMAR = (
    (0, 1, 3, 4, 6, 7),
    (1, 2, 4, 5, 7, 8),
    (0, 2, 3, 5, 6, 8),
)


def make_intercell_backlog_choice_env(
    *,
    authority_mode: str = "holonic_hybrid",
) -> IntercellTreeRuntimeEnv:
    suite = build_tree_runtime_v2_intercell_a3c9_probe_suite(PROJECT_ROOT)
    case = suite[0]
    return IntercellTreeRuntimeEnv(
        case.cell_configs,
        area_to_cells={0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]},
        config=make_regime_config("R0"),
        seed=13,
        max_in_flight_jobs=3,
        route_grammar=_CANONICAL_ROUTE_GRAMMAR,
        backlog_top_k=3,
        route_grammar_contract="category_b_a3c9_probe_v1",
        authority_mode=authority_mode,
    )


def _reserve_inbound(env: IntercellTreeRuntimeEnv, *cell_ids: int) -> None:
    for offset, cell_id in enumerate(cell_ids, start=1):
        env._env._cell_inbound_jobs[int(cell_id)].add(10000 + offset)


def test_intercell_tree_runtime_env_starts_with_backlog_selection_epoch() -> None:
    env = make_intercell_backlog_choice_env()
    try:
        _, info = env.reset(seed=13)

        assert env._env.area_to_cells == {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
        assert env._env.route_grammar == _CANONICAL_ROUTE_GRAMMAR
        assert env.agents_to_act() == ["plant"]
        plant_record = env.current_decision_epoch.records_by_agent["plant"]
        assert plant_record.decision_kind == "backlog_selection"

        payload = build_controller_payload(env, "plant")
        assert payload["plant_decision_kind"] == "backlog_selection"
        assert payload["action_semantics"]["action_type"] == "select_backlog_item"
        assert len(payload["valid_backlog_ids"]) == 3
        assert len(payload["visible_backlog_candidates"]) == 3
        for stage_cells in env._env.route_grammar:
            assert len(stage_cells) >= 6
            assert {env._env.cell_to_area[cell_id] for cell_id in stage_cells} == {0, 1, 2}
        assert info["activation_trace"][0]["plant_decision_kind"] == "backlog_selection"
    finally:
        env.close()


def test_intercell_runtime_exposes_authority_mode_in_reset_info_and_payload() -> None:
    env = make_intercell_backlog_choice_env(authority_mode="holonic_hybrid")
    try:
        _, info = env.reset(seed=13)

        assert info["authority_mode"]["mode_id"] == "holonic_hybrid"
        assert info["authority_mode"]["routing_authority_mode"] == "holonic_hybrid"
        assert info["authority_mode"]["cell_contract_actions"] == "accept_reject"

        payload = build_controller_payload(env, "plant")
        assert payload["authority_mode"]["mode_id"] == "holonic_hybrid"
        assert payload["authority_mode"]["area_local_reroute_enabled"] is True
        assert payload["authority_mode"]["mediated_bid_round_enabled"] is False
    finally:
        env.close()


def test_intercell_selected_backlog_is_not_head_alias_and_changes_area_surface() -> None:
    env = make_intercell_backlog_choice_env()
    try:
        env.reset(seed=13)
        backlog_ids = list(env._env.shared_backlog[:3])
        assert len(backlog_ids) == 3
        head_job = int(backlog_ids[0])
        second_job = int(backlog_ids[1])

        env._env.simulator.jobs[head_job].current_op_idx = 0
        env._env.simulator.jobs[second_job].current_op_idx = 1
        _reserve_inbound(env, 0, 1, 6, 7)

        before = build_controller_payload(env, "plant")
        head_candidate = next(
            candidate
            for candidate in before["visible_backlog_candidates"]
            if int(candidate["job_id"]) == head_job
        )
        second_candidate = next(
            candidate
            for candidate in before["visible_backlog_candidates"]
            if int(candidate["job_id"]) == second_job
        )
        assert head_candidate["valid_areas"] == [1]
        assert head_candidate["valid_cells_by_area"] == {1: [3, 4]}
        assert second_candidate["valid_areas"] == [0, 1, 2]
        assert second_candidate["valid_cells_by_area"] == {0: [2], 1: [4, 5], 2: [8]}
        assert second_candidate["job_id"] != env._env.shared_backlog[0]

        step = env.step_dict({"plant": 1})

        assert step.info["selected_backlog_id"] == second_job
        assert step.info["selected_backlog_id"] != head_job
        assert any(
            record["agent_id"] == "plant" and record["plant_decision_kind"] == "area_selection"
            for record in step.info["activation_trace"]
        )

        payload = build_controller_payload(env, "plant")
        assert payload["plant_decision_kind"] == "area_selection"
        assert payload["selected_backlog_id"] == second_job
        assert payload["selected_backlog_stage_id"] == 1
        assert payload["selected_backlog_valid_areas"] == [0, 1, 2]
        assert payload["selected_backlog_valid_cells_by_area"] == {0: [2], 1: [4, 5], 2: [8]}
        assert payload["selected_backlog_valid_areas"] != head_candidate["valid_areas"]
        assert (
            payload["selected_backlog_valid_cells_by_area"]
            != head_candidate["valid_cells_by_area"]
        )
    finally:
        env.close()


def test_intercell_backlog_selection_trace_exposes_candidate_and_downstream_surface() -> None:
    env = make_intercell_backlog_choice_env()
    try:
        _, info = env.reset(seed=13)
        plant_record = info["activation_trace"][0]

        assert plant_record["plant_decision_kind"] == "backlog_selection"
        assert "visible_backlog_candidates" in plant_record["context_handles"]
        assert "valid_backlog_ids" in plant_record["context_handles"]
        assert plant_record["context_handles"]["selected_backlog_id"] is None

        env._env.simulator.jobs[int(env._env.shared_backlog[1])].current_op_idx = 1
        _reserve_inbound(env, 0, 1, 6, 7)
        step = env.step_dict({"plant": 1})
        current_plant_record = step.info["current_decision_epoch"]["records_by_agent"]["plant"]

        assert current_plant_record["plant_decision_kind"] == "area_selection"
        assert current_plant_record["context_handles"]["selected_backlog_id"] == int(
            env.current_decision_epoch.records_by_agent["plant"].context_handles["selected_backlog_id"]
        )
        assert current_plant_record["context_handles"]["selected_backlog_valid_areas"] == [0, 1, 2]
        assert current_plant_record["context_handles"]["selected_backlog_valid_cells_by_area"] == {
            "0": [2],
            "1": [4, 5],
            "2": [8],
        }
        payload = build_controller_payload(env, "plant")
        assert payload["visible_backlog_candidates"]
        assert payload["selected_backlog_valid_areas"] == [0, 1, 2]
        assert payload["selected_backlog_valid_cells_by_area"] == {0: [2], 1: [4, 5], 2: [8]}
    finally:
        env.close()


def test_intercell_real_a3c9_area_activation_trace_exposes_valid_cells() -> None:
    env = make_intercell_backlog_choice_env()
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        step = env.step_dict({"plant": 0})

        area_record = step.info["activation_trace"][0]
        assert area_record["agent_id"] == "area_0"
        assert area_record["functional_loop"] == "commitment_management"
        assert area_record["context_handles"]["valid_cells"] == [0, 1]
        assert area_record["context_handles"]["valid_cell_count"] == 2
    finally:
        env.close()


def test_intercell_centralized_mode_uses_plant_cell_selection_and_skips_area_commitment() -> None:
    env = make_intercell_backlog_choice_env(authority_mode="centralized")
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})

        plant_payload = build_controller_payload(env, "plant")
        assert plant_payload["plant_decision_kind"] == "cell_selection"
        assert plant_payload["action_semantics"]["action_type"] == "select_execution_cell"
        assert plant_payload["valid_execution_cells"] == [0, 1, 3, 4, 6, 7]
        assert sorted(plant_payload["execution_cell_summaries"]) == [0, 1, 3, 4, 6, 7]

        selected_cell_id = int(plant_payload["action_semantics"]["action_targets"][0]["target_id"])
        step = env.step_dict({"plant": selected_cell_id})

        assert step.info["selected_cell_id"] == selected_cell_id
        assert step.info["current_decision_epoch"]["agents_to_act"] == [f"cell_{selected_cell_id}"]
        assert (
            step.info["current_decision_epoch"]["records_by_agent"][f"cell_{selected_cell_id}"][
                "functional_loop"
            ]
            == "local_regulation"
        )
        assert not any(
            record["agent_id"].startswith("area_")
            and record["functional_loop"] == "commitment_management"
            for record in step.info["activation_trace"]
        )
    finally:
        env.close()


def test_intercell_hierarchical_mode_cell_commitment_is_accept_only() -> None:
    env = make_intercell_backlog_choice_env(authority_mode="hierarchical")
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})

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


def test_intercell_mediated_cnp_area_payload_and_trace_show_bid_round() -> None:
    env = make_intercell_backlog_choice_env(authority_mode="mediated_heterarchical_cnp")
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"plant": 0})

        area_payload = build_controller_payload(env, "area_0")
        assert area_payload["action_semantics"]["action_type"] == "award_bid"
        assert area_payload["authority_mode"]["mediated_bid_round_enabled"] is True

        selected_cell_id = int(area_payload["action_semantics"]["action_targets"][0]["target_id"])
        step = env.step_dict({"area_0": selected_cell_id})

        protocol_kinds = [message["kind"] for message in step.info["protocol_trace"]]
        assert protocol_kinds[:4] == ["cfp", "bid", "award", "assignment_contract"]
        assert step.info["current_decision_epoch"]["agents_to_act"] == [f"cell_{selected_cell_id}"]
    finally:
        env.close()


def test_intercell_area_mask_uses_selected_backlog_not_fifo_head() -> None:
    env = make_intercell_backlog_choice_env()
    try:
        env.reset(seed=13)
        backlog_ids = list(env._env.shared_backlog[:3])
        head_job = int(backlog_ids[0])
        second_job = int(backlog_ids[1])

        env._env.simulator.jobs[head_job].current_op_idx = 0
        env._env.simulator.jobs[second_job].current_op_idx = 1
        _reserve_inbound(env, 0, 1, 6, 7)

        env.step_dict({"plant": 1})
        step = env.step_dict({"plant": 2})

        assert step.info["selected_backlog_id"] == second_job
        assert step.info["activation_trace"][0]["agent_id"] == "area_2"

        area_payload = build_controller_payload(env, "area_2")
        assert area_payload["valid_cells"] == [8]
        assert area_payload["local_subtree_state"]["valid_children"] == [8]

        mask = env.action_mask("area_2").tolist()
        assert sum(mask) == 1
        assert mask[8] == 1
    finally:
        env.close()
