from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from fms_bench.mas_baselines import ContractNetGreedyPolicy

if TYPE_CHECKING:
    from .tree_runtime_env import TreeRuntimeEnv
    from .types import ActivationRecord


PUBLIC_CONTROLLER_PAYLOAD_BASE_FIELDS = (
    "agent_id",
    "role",
    "authority_mode",
    "activation",
    "observation",
    "parent_context",
    "local_subtree_state",
    "feedback_view",
    "local_env_events",
    "action_mask",
    "action_semantics",
    "visible_protocol_objects",
)
PUBLIC_CONTROLLER_PAYLOAD_ROLE_ALIAS_FIELDS = {
    "plant": (
        "plant_decision_kind",
        "valid_areas",
        "visible_area_summaries",
        "area_summaries",
        "valid_backlog_ids",
        "visible_backlog_candidates",
        "selected_backlog_id",
        "selected_backlog_stage_id",
        "selected_backlog_valid_areas",
        "selected_backlog_valid_cells_by_area",
        "valid_execution_cells",
        "execution_cell_summaries",
    ),
    "area": (
        "area_id",
        "valid_cells",
        "visible_cell_summaries",
        "cell_summaries",
        "area_summary",
        "candidate_bids",
    ),
    "cell": (
        "cell_id",
        "cell_summary",
        "dispatch_options",
        "visible_machine_summaries",
    ),
}


def build_controller_payload(env: "TreeRuntimeEnv", agent_id: str) -> dict[str, object]:
    activation = env.current_decision_epoch.records_by_agent[agent_id]
    role = _agent_role(agent_id)
    local_subtree_state = _local_subtree_state(env, agent_id, activation)
    parent_context = _parent_context(env, activation)
    payload: dict[str, object] = {
        "agent_id": agent_id,
        "role": role,
        "authority_mode": env.authority_mode.to_dict(),
        "activation": env._serialize_record(activation),
        "observation": env.observe(agent_id),
        "parent_context": parent_context,
        "local_subtree_state": local_subtree_state,
        "feedback_view": _feedback_view(env, agent_id, local_subtree_state),
        "local_env_events": _local_env_events(env, activation),
        "action_mask": env.action_mask(agent_id),
        "action_semantics": _action_semantics(env, agent_id, activation, local_subtree_state),
        "visible_protocol_objects": _visible_protocol_objects(parent_context),
    }
    payload.update(_compatibility_aliases(env, agent_id, activation, local_subtree_state))
    return payload


def _agent_role(agent_id: str) -> str:
    if agent_id == "plant":
        return "plant"
    if agent_id.startswith("area_"):
        return "area"
    if agent_id.startswith("cell_"):
        return "cell"
    return "unknown"


def _parent_context(
    env: "TreeRuntimeEnv",
    activation: "ActivationRecord",
) -> dict[str, object]:
    context: dict[str, object] = {}
    for key, value in dict(activation.context_handles).items():
        if key == "dispatch_options":
            continue
        context[str(key)] = env._serialize_event(value)
    if "message" not in context and activation.agent_id.startswith("cell_"):
        current_contract = _current_contract_context(env, activation.agent_id)
        if current_contract is not None:
            context["current_contract"] = current_contract
    return context


def _current_contract_context(
    env: "TreeRuntimeEnv",
    agent_id: str,
) -> dict[str, object] | None:
    for entry in reversed(env.runtime_ledger.settlement_trace()):
        if str(entry.get("accepted_by")) != agent_id and str(entry.get("recipient")) != agent_id:
            continue
        return {
            "contract_id": str(entry.get("contract_id", "")),
            "work_id": str(entry.get("work_id", "")),
            "sender": str(entry.get("sender", "")),
            "recipient": str(entry.get("recipient", "")),
            "accepted_by": str(entry.get("accepted_by", "")),
            "status": str(entry.get("status", "")),
            "job_id": entry.get("job_id"),
            "completed_operations": int(entry.get("completed_operations", 0)),
            "total_operations": entry.get("total_operations"),
        }
    return None


def _local_subtree_state(
    env: "TreeRuntimeEnv",
    agent_id: str,
    activation: "ActivationRecord",
) -> dict[str, object]:
    if agent_id == "plant":
        plant_decision_kind = (
            str(activation.decision_kind)
            if activation.decision_kind is not None
            else str(dict(activation.context_handles).get("plant_decision_kind", "area_selection"))
        )
        valid_children = (
            []
            if plant_decision_kind == "backlog_selection"
            else env.controller_valid_area_ids(activation)
        )
        child_ids = [int(area_id) for area_id in sorted(env._env.area_ids)]
        child_summaries = {
            int(area_id): env.controller_area_summary(int(area_id))
            for area_id in child_ids
        }
        visible_backlog_candidates = env.controller_visible_backlog_candidates()
        valid_backlog_ids = env.controller_valid_backlog_ids()
        selected_backlog_id = dict(activation.context_handles).get("selected_backlog_id")
        selected_backlog_stage_id = dict(activation.context_handles).get("selected_backlog_stage_id")
        selected_backlog_valid_areas = [
            int(area_id)
            for area_id in dict(activation.context_handles).get("selected_backlog_valid_areas", ())
        ]
        selected_backlog_valid_cells_by_area = {
            int(area_id): [int(cell_id) for cell_id in cell_ids]
            for area_id, cell_ids in dict(
                dict(activation.context_handles).get("selected_backlog_valid_cells_by_area", {})
            ).items()
        }
        valid_execution_cells = [
            int(cell_id)
            for cell_ids in selected_backlog_valid_cells_by_area.values()
            for cell_id in cell_ids
        ]
        execution_cell_summaries = {
            int(cell_id): env.controller_cell_summary(int(cell_id))
            for cell_id in valid_execution_cells
        }
        return {
            "child_ids": child_ids,
            "valid_children": valid_children,
            "child_summaries": child_summaries,
            "plant_state": env.controller_plant_state(),
            "plant_decision_kind": plant_decision_kind,
            "valid_backlog_ids": valid_backlog_ids,
            "visible_backlog_candidates": visible_backlog_candidates,
            "selected_backlog_id": selected_backlog_id,
            "selected_backlog_stage_id": selected_backlog_stage_id,
            "selected_backlog_valid_areas": selected_backlog_valid_areas,
            "selected_backlog_valid_cells_by_area": selected_backlog_valid_cells_by_area,
            "valid_execution_cells": valid_execution_cells,
            "execution_cell_summaries": execution_cell_summaries,
        }

    if agent_id.startswith("area_"):
        area_id = int(agent_id.split("_", 1)[1])
        child_ids = [int(cell_id) for cell_id in env._env.area_to_cells[area_id]]
        cell_summaries = {
            cell_id: env.controller_cell_summary(cell_id)
            for cell_id in child_ids
        }
        valid_children = (
            []
            if activation.functional_loop == "accountability_and_escalation"
            else env.controller_valid_cell_ids(area_id)
        )
        area_summary = env.controller_area_summary(area_id)
        candidate_bids = []
        if (
            activation.functional_loop == "commitment_management"
            and env.authority_mode.mediated_bid_round_enabled
            and valid_children
        ):
            candidate_bids = _candidate_bids(env, area_id, valid_children, cell_summaries)
        return {
            "area_id": area_id,
            "child_ids": child_ids,
            "valid_children": valid_children,
            "child_summaries": cell_summaries,
            "area_summary": area_summary,
            "candidate_bids": candidate_bids,
        }

    if agent_id.startswith("cell_"):
        cell_id = int(agent_id.split("_", 1)[1])
        cell_summary = env.controller_cell_summary(cell_id)
        machine_summaries = env.controller_machine_summaries(cell_id=cell_id)
        return {
            "cell_id": cell_id,
            "child_ids": [],
            "cell_summary": cell_summary,
            "machine_summaries": machine_summaries,
        }

    return {"child_ids": []}


def _feedback_view(
    env: "TreeRuntimeEnv",
    agent_id: str,
    local_subtree_state: Mapping[str, object],
) -> dict[str, object]:
    settlement_entries = _settlement_entries(env, agent_id)
    if agent_id == "plant":
        plant_state = dict(local_subtree_state.get("plant_state", {}))
        progress = {
            "released_jobs_total": int(plant_state.get("released_jobs_total", 0)),
            "completed_jobs_total": int(plant_state.get("completed_jobs_total", 0)),
            "backlog_jobs_total": int(plant_state.get("backlog_jobs_total", 0)),
        }
        capacity_risk = {
            "admissible_for_new_work": bool(local_subtree_state.get("valid_children", [])),
            "busy_machines_total": int(plant_state.get("busy_machines_total", 0)),
            "down_machines_total": int(plant_state.get("down_machines_total", 0)),
        }
    elif agent_id.startswith("area_"):
        area_summary = dict(local_subtree_state.get("area_summary", {}))
        progress = {
            "backlog_jobs": int(area_summary.get("backlog_jobs", 0)),
            "active_jobs": int(area_summary.get("active_jobs", 0)),
            "completed_jobs": int(area_summary.get("completed_jobs", 0)),
        }
        capacity_risk = {
            "admissible_for_new_work": bool(local_subtree_state.get("valid_children", [])),
            "valid_child_ids": list(local_subtree_state.get("valid_children", [])),
            "energy_kwh": float(area_summary.get("energy_kwh", 0.0)),
            "carbon_kg": float(area_summary.get("carbon_kg", 0.0)),
        }
    else:
        cell_summary = dict(local_subtree_state.get("cell_summary", {}))
        capacity_risk = {
            "admissible_for_new_work": int(cell_summary.get("budget_remaining_wip", 0)) > 0,
            "wip_load": int(cell_summary.get("active_jobs", 0)),
            "local_blocked": int(cell_summary.get("budget_remaining_wip", 0)) <= 0,
            "machine_health_summary": {
                "busy_machines": int(cell_summary.get("busy_machines", 0)),
                "down_machines": int(cell_summary.get("down_machines", 0)),
            },
            "energy_overshoot": max(
                0.0,
                float(cell_summary.get("energy_kwh", 0.0))
                - float(cell_summary.get("budget_cap_energy_kwh", 0.0)),
            ),
            "carbon_overshoot": max(
                0.0,
                float(cell_summary.get("carbon_kg", 0.0))
                - float(cell_summary.get("budget_cap_carbon_kg", 0.0)),
            ),
        }
        progress = {
            "active_jobs": int(cell_summary.get("active_jobs", 0)),
            "ready_jobs": int(cell_summary.get("ready_jobs", 0)),
            "completed_jobs": int(cell_summary.get("completed_jobs", 0)),
        }
    return {
        "progress": progress,
        "capacity_risk": capacity_risk,
        "completion": _completion_feedback(settlement_entries),
        "escalation": [],
    }


def _completion_feedback(entries: list[dict[str, object]]) -> dict[str, object]:
    status_counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "tracked_contracts": entries[-5:],
        "status_counts": status_counts,
    }


def _settlement_entries(env: "TreeRuntimeEnv", agent_id: str) -> list[dict[str, object]]:
    entries = env.runtime_ledger.settlement_trace()
    if agent_id == "plant":
        return entries

    if agent_id.startswith("area_"):
        area_id = int(agent_id.split("_", 1)[1])
        subtree_agents = {f"cell_{cell_id}" for cell_id in env._env.area_to_cells[area_id]}
        subtree_agents.add(agent_id)
        return [
            dict(entry)
            for entry in entries
            if str(entry.get("sender")) in subtree_agents
            or str(entry.get("recipient")) in subtree_agents
            or str(entry.get("accepted_by")) in subtree_agents
        ]

    return [
        dict(entry)
        for entry in entries
        if agent_id
        in {
            str(entry.get("sender")),
            str(entry.get("recipient")),
            str(entry.get("accepted_by")),
        }
    ]


def _local_env_events(
    env: "TreeRuntimeEnv",
    activation: "ActivationRecord",
) -> list[object]:
    return [env._serialize_event(item) for item in activation.event_summary]


def _action_semantics(
    env: "TreeRuntimeEnv",
    agent_id: str,
    activation: "ActivationRecord",
    local_subtree_state: Mapping[str, object],
) -> dict[str, object]:
    if agent_id == "plant" and activation.functional_loop == "accountability_and_escalation":
        return {
            "action_type": "revise_contract",
            "target_role": "area",
            "plant_decision_kind": "accountability_reroute",
            "action_targets": [
                {
                    "action_id": int(env._env.area_index[area_id]),
                    "target_id": int(area_id),
                    "action_kind": "revise_contract",
                }
                for area_id in local_subtree_state.get("valid_children", [])
            ],
        }

    if agent_id == "plant":
        plant_decision_kind = str(local_subtree_state.get("plant_decision_kind", "area_selection"))
        if plant_decision_kind == "backlog_selection":
            return {
                "action_type": "select_backlog_item",
                "target_role": "shared_backlog_item",
                "plant_decision_kind": "backlog_selection",
                "action_targets": [
                    {
                        "action_id": int(candidate.get("backlog_index", index)),
                        "target_id": int(candidate.get("job_id", -1)),
                        "stage_id": int(candidate.get("stage_id", -1)),
                        "valid_areas": [int(area_id) for area_id in candidate.get("valid_areas", ())],
                        "valid_cells_by_area": {
                            int(area_id): [int(cell_id) for cell_id in cell_ids]
                            for area_id, cell_ids in dict(
                                candidate.get("valid_cells_by_area", {})
                            ).items()
                        },
                    }
                    for index, candidate in enumerate(
                        list(local_subtree_state.get("visible_backlog_candidates", ()))
                    )
                    if int(candidate.get("job_id", -1))
                    in {
                        int(job_id)
                        for job_id in local_subtree_state.get("valid_backlog_ids", ())
                    }
                ],
            }
        if plant_decision_kind == "cell_selection":
            return {
                "action_type": "select_execution_cell",
                "target_role": "cell",
                "plant_decision_kind": "cell_selection",
                "action_targets": [
                    {
                        "action_id": int(cell_id),
                        "target_id": int(cell_id),
                        "area_id": int(env._env.cell_to_area[int(cell_id)]),
                    }
                    for cell_id in local_subtree_state.get("valid_execution_cells", ())
                ],
            }
        return {
            "action_type": "select_child_holon",
            "target_role": "area",
            "plant_decision_kind": plant_decision_kind,
            "action_targets": [
                {
                    "action_id": int(env._env.area_index[area_id]),
                    "target_id": int(area_id),
                }
                for area_id in local_subtree_state.get("valid_children", [])
            ],
        }

    if agent_id.startswith("area_") and activation.functional_loop == "accountability_and_escalation":
        return {
            "action_type": "accountability_and_escalation",
            "target_role": "self",
            "action_targets": [
                {
                    "action_id": 0,
                    "target_id": agent_id,
                    "action_kind": "process_child_feedback",
                }
            ],
        }

    if agent_id.startswith("area_"):
        if env.authority_mode.mediated_bid_round_enabled and activation.functional_loop == "commitment_management":
            candidate_bids = list(local_subtree_state.get("candidate_bids", ()))
            bid_scores = {
                int(bid["cell_id"]): float(bid["score"])
                for bid in candidate_bids
            }
            bid_ranks = {
                int(bid["cell_id"]): int(bid["rank"])
                for bid in candidate_bids
            }
            return {
                "action_type": "award_bid",
                "target_role": "cell_bid",
                "action_targets": [
                    {
                        "action_id": int(cell_id),
                        "target_id": int(cell_id),
                        "action_kind": "award_contract",
                        "bid_score": bid_scores.get(int(cell_id), float("inf")),
                        "bid_rank": bid_ranks.get(int(cell_id), len(candidate_bids) + 1),
                    }
                    for cell_id in local_subtree_state.get("valid_children", [])
                ],
            }
        return {
            "action_type": "select_child_holon",
            "target_role": "cell",
            "action_targets": [
                {
                    "action_id": int(cell_id),
                    "target_id": int(cell_id),
                }
                for cell_id in local_subtree_state.get("valid_children", [])
            ],
        }

    if activation.functional_loop == "local_regulation":
        dispatch_options = [
            dict(option)
            for option in activation.context_handles.get("dispatch_options", ())
        ]
        return {
            "action_type": "dispatch_operation",
            "target_role": "machine_option",
            "action_targets": [
                {
                    "action_id": int(index),
                    "target_id": str(option.get("operation_id", index)),
                    "dispatch_option": dict(option),
                }
                for index, option in enumerate(dispatch_options)
            ],
        }

    return {
        "action_type": "commitment_management",
        "target_role": "self",
        "action_targets": [
            {
                "action_id": int(index),
                "target_id": agent_id,
                "action_kind": str(target.action_kind),
            }
            for index, target in enumerate(env._cell_runtimes[agent_id].legal_actions(activation))
        ],
    }


def _visible_protocol_objects(parent_context: Mapping[str, object]) -> tuple[object, ...]:
    message = parent_context.get("message")
    if message is None:
        return ()
    return (message,)


def _compatibility_aliases(
    env: "TreeRuntimeEnv",
    agent_id: str,
    activation: "ActivationRecord",
    local_subtree_state: Mapping[str, object],
) -> dict[str, object]:
    if agent_id == "plant":
        child_summaries = dict(local_subtree_state.get("child_summaries", {}))
        return {
            "plant_decision_kind": str(local_subtree_state.get("plant_decision_kind", "")),
            "valid_areas": list(local_subtree_state.get("valid_children", [])),
            "visible_area_summaries": child_summaries,
            "area_summaries": child_summaries,
            "valid_backlog_ids": [int(job_id) for job_id in local_subtree_state.get("valid_backlog_ids", ())],
            "visible_backlog_candidates": [
                dict(candidate)
                for candidate in local_subtree_state.get("visible_backlog_candidates", ())
            ],
            "selected_backlog_id": local_subtree_state.get("selected_backlog_id"),
            "selected_backlog_stage_id": local_subtree_state.get("selected_backlog_stage_id"),
            "selected_backlog_valid_areas": [
                int(area_id)
                for area_id in local_subtree_state.get("selected_backlog_valid_areas", ())
            ],
            "selected_backlog_valid_cells_by_area": {
                int(area_id): [int(cell_id) for cell_id in cell_ids]
                for area_id, cell_ids in dict(
                    local_subtree_state.get("selected_backlog_valid_cells_by_area", {})
                ).items()
            },
            "valid_execution_cells": [
                int(cell_id)
                for cell_id in local_subtree_state.get("valid_execution_cells", ())
            ],
            "execution_cell_summaries": {
                int(cell_id): dict(summary)
                for cell_id, summary in dict(
                    local_subtree_state.get("execution_cell_summaries", {})
                ).items()
            },
        }

    if agent_id.startswith("area_"):
        child_summaries = dict(local_subtree_state.get("child_summaries", {}))
        return {
            "area_id": int(local_subtree_state.get("area_id", -1)),
            "valid_cells": list(local_subtree_state.get("valid_children", [])),
            "visible_cell_summaries": child_summaries,
            "cell_summaries": child_summaries,
            "area_summary": dict(local_subtree_state.get("area_summary", {})),
            "candidate_bids": [
                dict(candidate)
                for candidate in local_subtree_state.get("candidate_bids", ())
            ],
        }

    dispatch_options = []
    if activation.functional_loop == "local_regulation":
        dispatch_options = [
            dict(option)
            for option in activation.context_handles.get("dispatch_options", ())
        ]
    return {
        "cell_id": int(local_subtree_state.get("cell_id", -1)),
        "cell_summary": dict(local_subtree_state.get("cell_summary", {})),
        "dispatch_options": dispatch_options,
        "visible_machine_summaries": dict(local_subtree_state.get("machine_summaries", {})),
    }


def _candidate_bids(
    env: "TreeRuntimeEnv",
    area_id: int,
    valid_children: list[int],
    cell_summaries: Mapping[int, Mapping[str, object]],
) -> list[dict[str, object]]:
    policy = ContractNetGreedyPolicy()
    policy.select_action(
        env.observe(f"area_{area_id}"),
        {
            "valid_cells": list(valid_children),
            "cell_summaries": {
                int(cell_id): dict(summary)
                for cell_id, summary in cell_summaries.items()
            },
        },
    )
    return [
        {
            "cell_id": int(bid.cell_id),
            "score": float(bid.score),
            "rank": int(index),
            "components": dict(bid.components),
        }
        for index, bid in enumerate(policy.last_bids, start=1)
    ]
