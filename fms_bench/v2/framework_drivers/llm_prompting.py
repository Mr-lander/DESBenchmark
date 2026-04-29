from __future__ import annotations

import json
import re
from typing import Mapping


OPENAI_COMPAT_BENCHMARK_SYSTEM_PROMPT = (
    "You are a benchmark controller agent. "
    "CRITICAL OUTPUT CONTRACT. "
    "Select exactly one legal integer action index from legal_actions. "
    "Use only the provided compact agent-local payload, including current context, visible feedback, "
    "decision_factors, and bounded working_memory when present. "
    'Reply with exactly one JSON object and nothing else. '
    'Valid schema: {"action": <index>}. '
    "No reasoning. No markdown. No prose."
)
# Keep the output budget large enough for providers that may emit an empty
# content string when forced to terminate too early on JSON-only replies.
# 512 leaves headroom for the longest current plant-level prompts while still
# keeping the contract focused on a single tiny JSON answer.
OPENAI_COMPAT_BENCHMARK_MAX_TOKENS = 512


def build_openai_compat_benchmark_prompt_payload(
    *,
    agent_id: str,
    agent_view: Mapping[str, object],
) -> dict[str, object]:
    role = str(agent_view.get("role", "unknown"))
    kind = _decision_kind(role=role, agent_view=agent_view)
    legal_actions = _legal_action_indices(agent_view)
    payload: dict[str, object] = {
        "agent": str(agent_id),
        "role": role,
        "kind": kind,
        "legal_actions": legal_actions,
        "decision_factors": _decision_factor_guidance(role=role, kind=kind),
        "options": _compact_options(
            role=role,
            kind=kind,
            legal_actions=legal_actions,
            agent_view=agent_view,
        ),
    }
    context = _compact_context(
        role=role,
        kind=kind,
        agent_view=agent_view,
    )
    if context:
        payload["context"] = context
    return payload


def build_openai_compat_benchmark_user_prompt(
    *,
    agent_id: str,
    agent_view: Mapping[str, object],
) -> str:
    return json.dumps(
        build_openai_compat_benchmark_prompt_payload(
            agent_id=agent_id,
            agent_view=agent_view,
        ),
        ensure_ascii=True,
        separators=(",", ":"),
    )


def build_openai_compat_benchmark_messages(
    *,
    agent_id: str,
    agent_view: Mapping[str, object],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": OPENAI_COMPAT_BENCHMARK_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_openai_compat_benchmark_user_prompt(
                agent_id=agent_id,
                agent_view=agent_view,
            ),
        },
    ]


def build_openai_compat_benchmark_generation_kwargs(
    *,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "temperature": float(temperature),
        "max_tokens": OPENAI_COMPAT_BENCHMARK_MAX_TOKENS,
    }
    # Qwen-plus models can spend a large hidden reasoning budget even for the
    # benchmark's tiny JSON-only action contract. Disabling reasoning keeps the
    # benchmark inside the existing timeout/cost envelope without changing the
    # visible output schema.
    normalized_model_name = str(model_name or "").strip().lower()
    if "qwen" in normalized_model_name:
        payload["reasoning"] = {
            "effort": "none",
            "exclude": True,
        }
    return payload


def parse_openai_compat_benchmark_action_text(content: str) -> int | None:
    normalized = _strip_code_fence(content)
    if not normalized:
        return None
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    action = payload.get("action")
    if isinstance(action, int):
        return int(action)
    if isinstance(action, str) and action.strip().lstrip("-").isdigit():
        return int(action.strip())
    return None


def _strip_code_fence(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _decision_kind(
    *,
    role: str,
    agent_view: Mapping[str, object],
) -> str:
    if role == "plant":
        return str(agent_view.get("plant_decision_kind", "area_selection"))
    action_semantics = dict(agent_view.get("action_semantics", {}))
    activation = dict(agent_view.get("activation", {}))
    if role == "area":
        return str(
            action_semantics.get(
                "action_type",
                activation.get("functional_loop", "select_child_holon"),
            )
        )
    if role == "cell":
        return str(
            activation.get(
                "functional_loop",
                action_semantics.get("action_type", "commitment_management"),
            )
        )
    return "unknown"


def _legal_action_indices(agent_view: Mapping[str, object]) -> list[int]:
    return [
        int(index)
        for index, enabled in enumerate(list(agent_view.get("action_mask", ())))
        if bool(enabled)
    ]


def _compact_context(
    *,
    role: str,
    kind: str,
    agent_view: Mapping[str, object],
) -> dict[str, object]:
    context: dict[str, object] = {}
    feedback = _compact_feedback_view(agent_view.get("feedback_view", {}))
    if feedback:
        context["feedback"] = feedback
    working_memory = _compact_working_memory(agent_view.get("working_memory", {}))
    if working_memory:
        context["working_memory"] = working_memory
    recent_event_types = _compact_recent_event_types(agent_view.get("local_env_events", ()))
    if recent_event_types:
        context["recent_event_types"] = recent_event_types
    if role == "plant":
        if agent_view.get("selected_backlog_id") is not None:
            context["selected_job_id"] = int(agent_view["selected_backlog_id"])
        if agent_view.get("selected_backlog_stage_id") is not None:
            context["selected_stage_id"] = int(agent_view["selected_backlog_stage_id"])
        if kind == "cell_selection":
            valid_cells = _int_list(agent_view.get("valid_execution_cells", ()))
            if valid_cells:
                context["valid_execution_cells"] = valid_cells
        return context

    if role == "area":
        if agent_view.get("area_id") is not None:
            context["area_id"] = int(agent_view["area_id"])
        area_summary = _compact_area_summary(agent_view.get("area_summary", {}))
        if area_summary:
            context["area_summary"] = area_summary
        return context

    if role == "cell" and kind != "local_regulation":
        cell_summary = _compact_cell_summary(agent_view.get("cell_summary", {}))
        if cell_summary:
            context["cell_summary"] = cell_summary
        feedback_view = dict(agent_view.get("feedback_view", {}))
        capacity_risk = dict(feedback_view.get("capacity_risk", {}))
        if "local_blocked" in capacity_risk:
            context["local_blocked"] = bool(capacity_risk.get("local_blocked", False))
        parent_context = dict(agent_view.get("parent_context", {}))
        current_contract = parent_context.get("current_contract")
        if isinstance(current_contract, Mapping):
            contract_context: dict[str, object] = {}
            for key in ("job_id", "completed_operations", "total_operations", "status"):
                if key in current_contract:
                    contract_context[str(key)] = current_contract[key]
            if contract_context:
                context["current_contract"] = contract_context
    return context


def _decision_factor_guidance(
    *,
    role: str,
    kind: str,
) -> list[str]:
    if role == "plant" and kind == "backlog_selection":
        return [
            "Prefer backlog choices that keep more downstream routing paths feasible.",
            "Account for recent reject, blocking, and overload signals in working_memory.",
            "Avoid choices that increase completion debt through weak late-stage routes.",
        ]
    if role == "plant" and kind == "cell_selection":
        return [
            "Compare candidate cells by current load, availability, and near-term disturbance risk.",
            "Use working_memory to avoid repeatedly sending work to recently blocked or rejected cells.",
            "Prefer the assignment that preserves future routing flexibility, not just immediate speed.",
        ]
    if role == "area" and kind in {"select_child_holon", "award_bid"}:
        return [
            "Balance bid or local speed against current backlog, busy/down machines, and overshoot risk.",
            "Use visible feedback and working_memory to avoid repeating unstable local reroutes.",
            "Prefer the child cell most likely to execute without immediate escalation.",
        ]
    if role == "cell" and kind == "local_regulation":
        return [
            "Compare proc_time with machine remaining/down time and current WIP pressure.",
            "Avoid dispatches that amplify blocking, setup burden, or repeated failure exposure.",
            "Prefer the machine that is feasible now and least likely to cause near-term regret.",
        ]
    if role == "cell":
        return [
            "Accept only when visible WIP, machine health, and budget risk support execution.",
            "Reject or revise when local blocking or overload makes timely completion unlikely.",
            "Use tracked contract status and recent outcomes to avoid over-commitment.",
        ]
    return [
        "Use current context, visible feedback, and working_memory only.",
        "Prefer feasible actions that reduce congestion and future reroute pressure.",
    ]


def _compact_feedback_view(raw_feedback_view: object) -> dict[str, object]:
    feedback_view = dict(raw_feedback_view) if isinstance(raw_feedback_view, Mapping) else {}
    if "capacity" in feedback_view or "completion_status_counts" in feedback_view:
        payload: dict[str, object] = {}
        progress = {
            str(key): _round_number(value)
            for key, value in dict(feedback_view.get("progress", {})).items()
            if isinstance(value, (int, float))
        }
        capacity = {
            str(key): _round_number(value)
            for key, value in dict(feedback_view.get("capacity", {})).items()
            if isinstance(value, (int, float, bool))
        }
        completion_status_counts = {
            str(key): int(value)
            for key, value in dict(feedback_view.get("completion_status_counts", {})).items()
            if isinstance(value, (int, float))
        }
        if progress:
            payload["progress"] = progress
        if capacity:
            payload["capacity"] = capacity
        if completion_status_counts:
            payload["completion_status_counts"] = completion_status_counts
        return payload

    progress = {
        str(key): _round_number(value)
        for key, value in dict(feedback_view.get("progress", {})).items()
        if isinstance(value, (int, float))
    }
    capacity_risk = dict(feedback_view.get("capacity_risk", {}))
    machine_health = dict(capacity_risk.get("machine_health_summary", {}))
    capacity: dict[str, object] = {}
    for key in (
        "admissible_for_new_work",
        "busy_machines_total",
        "down_machines_total",
        "energy_kwh",
        "carbon_kg",
        "wip_load",
        "local_blocked",
        "energy_overshoot",
        "carbon_overshoot",
    ):
        if key in capacity_risk:
            capacity[str(key)] = _round_number(capacity_risk[key])
    for key in ("busy_machines", "down_machines"):
        if key in machine_health:
            capacity[str(key)] = _round_number(machine_health[key])
    completion_status_counts = {
        str(key): int(value)
        for key, value in dict(
            dict(feedback_view.get("completion", {})).get("status_counts", {})
        ).items()
        if isinstance(value, (int, float))
    }
    payload: dict[str, object] = {}
    if progress:
        payload["progress"] = progress
    if capacity:
        payload["capacity"] = capacity
    if completion_status_counts:
        payload["completion_status_counts"] = completion_status_counts
    return payload


def _compact_working_memory(raw_working_memory: object) -> dict[str, object]:
    working_memory = dict(raw_working_memory) if isinstance(raw_working_memory, Mapping) else {}
    payload: dict[str, object] = {}
    recent_outcomes = _compact_recent_outcomes(working_memory.get("recent_outcomes", ()))
    if recent_outcomes:
        payload["recent_outcomes"] = recent_outcomes
    open_commitments = _compact_open_commitments(working_memory.get("open_commitments", ()))
    if open_commitments:
        payload["open_commitments"] = open_commitments
    rolled_summary = _compact_rolled_summary(working_memory.get("rolled_summary", {}))
    if rolled_summary:
        payload["rolled_summary"] = rolled_summary
    return payload


def _compact_recent_outcomes(raw_recent_outcomes: object) -> list[dict[str, object]]:
    outcomes: list[dict[str, object]] = []
    for raw_outcome in list(raw_recent_outcomes)[:6]:
        if not isinstance(raw_outcome, Mapping):
            continue
        outcome = dict(raw_outcome)
        item: dict[str, object] = {}
        for key in ("step", "kind", "action", "agent", "selected_target"):
            if key in outcome:
                item[str(key)] = _round_number(outcome[key])
        feedback = _compact_feedback_view(outcome.get("observed_feedback", {}))
        if feedback:
            item["observed_feedback"] = feedback
        event_types = _compact_recent_event_types(outcome.get("event_types", ()))
        if event_types:
            item["event_types"] = event_types
        if item:
            outcomes.append(item)
    return outcomes


def _compact_open_commitments(raw_commitments: object) -> list[dict[str, object]]:
    commitments: list[dict[str, object]] = []
    for raw_item in list(raw_commitments)[:4]:
        if not isinstance(raw_item, Mapping):
            continue
        item = dict(raw_item)
        commitment: dict[str, object] = {}
        for key in (
            "contract_id",
            "work_id",
            "job_id",
            "status",
            "sender",
            "recipient",
            "accepted_by",
            "completed_operations",
            "total_operations",
        ):
            if key in item:
                commitment[str(key)] = _round_number(item[key])
        if commitment:
            commitments.append(commitment)
    return commitments


def _compact_rolled_summary(raw_summary: object) -> dict[str, object]:
    summary = dict(raw_summary) if isinstance(raw_summary, Mapping) else {}
    payload: dict[str, object] = {}
    for key in (
        "compressed_count",
        "reject_count",
        "local_blocked_count",
        "mean_wip_load",
        "mean_energy_overshoot",
        "mean_carbon_overshoot",
        "mean_busy_machines",
        "mean_down_machines",
    ):
        if key in summary:
            payload[str(key)] = _round_number(summary[key])
    decision_counts_by_kind = {
        str(key): int(value)
        for key, value in dict(summary.get("decision_counts_by_kind", {})).items()
        if isinstance(value, (int, float))
    }
    if decision_counts_by_kind:
        payload["decision_counts_by_kind"] = decision_counts_by_kind
    completion_status_counts = {
        str(key): int(value)
        for key, value in dict(summary.get("completion_status_counts", {})).items()
        if isinstance(value, (int, float))
    }
    if completion_status_counts:
        payload["completion_status_counts"] = completion_status_counts
    repeated_targets: list[dict[str, object]] = []
    for raw_item in list(summary.get("top_repeated_targets", ()))[:3]:
        if not isinstance(raw_item, Mapping):
            continue
        item = dict(raw_item)
        target = item.get("target")
        count = item.get("count")
        if not isinstance(target, str) or not isinstance(count, (int, float)):
            continue
        repeated_targets.append({"target": target, "count": int(count)})
    if repeated_targets:
        payload["top_repeated_targets"] = repeated_targets
    return payload


def _compact_recent_event_types(raw_events: object) -> list[str]:
    event_types: list[str] = []
    for raw_event in list(raw_events)[:4]:
        if isinstance(raw_event, str) and raw_event:
            event_types.append(raw_event)
            continue
        if not isinstance(raw_event, Mapping):
            continue
        event = dict(raw_event)
        event_type = event.get("type") or event.get("kind") or event.get("event_type")
        if isinstance(event_type, str) and event_type:
            event_types.append(event_type)
    return event_types


def _compact_options(
    *,
    role: str,
    kind: str,
    legal_actions: list[int],
    agent_view: Mapping[str, object],
) -> list[dict[str, object]]:
    if role == "plant":
        if kind == "backlog_selection":
            return _compact_backlog_options(legal_actions=legal_actions, agent_view=agent_view)
        if kind == "cell_selection":
            return _compact_execution_cell_options(legal_actions=legal_actions, agent_view=agent_view)
        return _compact_area_options(legal_actions=legal_actions, agent_view=agent_view)

    if role == "area":
        if kind in {"select_child_holon", "award_bid"}:
            return _compact_area_cell_options(
                kind=kind,
                legal_actions=legal_actions,
                agent_view=agent_view,
            )
        return _compact_generic_action_targets(legal_actions=legal_actions, agent_view=agent_view)

    if role == "cell":
        if kind == "local_regulation":
            return _compact_dispatch_options(legal_actions=legal_actions, agent_view=agent_view)
        return _compact_commitment_actions(legal_actions=legal_actions, agent_view=agent_view)

    return _compact_generic_action_targets(legal_actions=legal_actions, agent_view=agent_view)


def _compact_backlog_options(
    *,
    legal_actions: list[int],
    agent_view: Mapping[str, object],
) -> list[dict[str, object]]:
    valid_backlog_ids = {
        int(job_id) for job_id in _int_list(agent_view.get("valid_backlog_ids", ()))
    }
    options: list[dict[str, object]] = []
    for index, raw_candidate in enumerate(list(agent_view.get("visible_backlog_candidates", ()))):
        candidate = dict(raw_candidate)
        action = int(candidate.get("backlog_index", index))
        job_id = int(candidate.get("job_id", -1))
        if action not in legal_actions or (valid_backlog_ids and job_id not in valid_backlog_ids):
            continue
        option: dict[str, object] = {
            "action": action,
            "job_id": job_id,
        }
        if "stage_id" in candidate:
            option["stage_id"] = int(candidate.get("stage_id", -1))
        valid_areas = _int_list(candidate.get("valid_areas", ()))
        if valid_areas:
            option["valid_areas"] = valid_areas
        valid_cells_by_area = _int_map_of_lists(candidate.get("valid_cells_by_area", {}))
        if valid_cells_by_area:
            option["valid_cells_by_area"] = valid_cells_by_area
        options.append(option)
    return options or [{"action": int(action)} for action in legal_actions]


def _compact_area_options(
    *,
    legal_actions: list[int],
    agent_view: Mapping[str, object],
) -> list[dict[str, object]]:
    area_summaries = _dict_map(agent_view.get("visible_area_summaries", agent_view.get("area_summaries", {})))
    valid_areas = set(_int_list(agent_view.get("valid_areas", ())))
    options: list[dict[str, object]] = []
    for target in _iter_legal_action_targets(legal_actions=legal_actions, agent_view=agent_view):
        area_id = int(target.get("target_id", target.get("action_id", -1)))
        if valid_areas and area_id not in valid_areas:
            continue
        option: dict[str, object] = {
            "action": int(target.get("action_id", area_id)),
            "area_id": area_id,
        }
        summary = _compact_area_summary(area_summaries.get(area_id, area_summaries.get(str(area_id), {})))
        if summary:
            option["summary"] = summary
        if "action_kind" in target:
            option["action_kind"] = str(target["action_kind"])
        options.append(option)
    if options:
        return options
    for area_id in sorted(valid_areas or set(legal_actions)):
        option = {
            "action": int(area_id),
            "area_id": int(area_id),
        }
        summary = _compact_area_summary(area_summaries.get(area_id, area_summaries.get(str(area_id), {})))
        if summary:
            option["summary"] = summary
        options.append(option)
    return options or [{"action": int(action)} for action in legal_actions]


def _compact_execution_cell_options(
    *,
    legal_actions: list[int],
    agent_view: Mapping[str, object],
) -> list[dict[str, object]]:
    cell_summaries = _dict_map(agent_view.get("execution_cell_summaries", agent_view.get("visible_cell_summaries", {})))
    valid_cells = set(_int_list(agent_view.get("valid_execution_cells", ())))
    options: list[dict[str, object]] = []
    for target in _iter_legal_action_targets(legal_actions=legal_actions, agent_view=agent_view):
        cell_id = int(target.get("target_id", target.get("action_id", -1)))
        if valid_cells and cell_id not in valid_cells:
            continue
        option: dict[str, object] = {
            "action": int(target.get("action_id", cell_id)),
            "cell_id": cell_id,
        }
        if "area_id" in target:
            option["area_id"] = int(target["area_id"])
        summary = _compact_cell_summary(cell_summaries.get(cell_id, cell_summaries.get(str(cell_id), {})))
        if summary:
            option["summary"] = summary
        options.append(option)
    if options:
        return options
    for cell_id in sorted(valid_cells or set(legal_actions)):
        option = {
            "action": int(cell_id),
            "cell_id": int(cell_id),
        }
        summary = _compact_cell_summary(cell_summaries.get(cell_id, cell_summaries.get(str(cell_id), {})))
        if summary:
            option["summary"] = summary
        options.append(option)
    return options or [{"action": int(action)} for action in legal_actions]


def _compact_area_cell_options(
    *,
    kind: str,
    legal_actions: list[int],
    agent_view: Mapping[str, object],
) -> list[dict[str, object]]:
    cell_summaries = _dict_map(agent_view.get("visible_cell_summaries", agent_view.get("cell_summaries", {})))
    valid_cells = set(_int_list(agent_view.get("valid_cells", ())))
    options: list[dict[str, object]] = []
    for target in _iter_legal_action_targets(legal_actions=legal_actions, agent_view=agent_view):
        cell_id = int(target.get("target_id", target.get("action_id", -1)))
        if valid_cells and cell_id not in valid_cells:
            continue
        option: dict[str, object] = {
            "action": int(target.get("action_id", cell_id)),
            "cell_id": cell_id,
        }
        summary = _compact_cell_summary(cell_summaries.get(cell_id, cell_summaries.get(str(cell_id), {})))
        if summary:
            option["summary"] = summary
        if kind == "award_bid":
            if "bid_score" in target:
                option["bid_score"] = float(target["bid_score"])
            if "bid_rank" in target:
                option["bid_rank"] = int(target["bid_rank"])
        options.append(option)
    if options:
        return options
    return [{"action": int(action)} for action in legal_actions]


def _compact_dispatch_options(
    *,
    legal_actions: list[int],
    agent_view: Mapping[str, object],
) -> list[dict[str, object]]:
    machine_summaries = _dict_map(agent_view.get("visible_machine_summaries", {}))
    options: list[dict[str, object]] = []
    action_targets = list(dict(agent_view.get("action_semantics", {})).get("action_targets", ()))
    if action_targets:
        for target in action_targets:
            action = int(target.get("action_id", -1))
            if action not in legal_actions:
                continue
            dispatch_option = dict(target.get("dispatch_option", {}))
            machine_id = int(dispatch_option.get("machine_id", -1))
            option: dict[str, object] = {
                "action": action,
                "machine_id": machine_id,
            }
            if "proc_time" in dispatch_option:
                option["proc_time"] = float(dispatch_option["proc_time"])
            machine = _compact_machine_summary(
                machine_summaries.get(machine_id, machine_summaries.get(str(machine_id), {}))
            )
            if machine:
                option["machine"] = machine
            options.append(option)
        if options:
            return options

    for index, raw_dispatch in enumerate(list(agent_view.get("dispatch_options", ()))):
        if index not in legal_actions:
            continue
        dispatch_option = dict(raw_dispatch)
        machine_id = int(dispatch_option.get("machine_id", -1))
        option = {
            "action": int(index),
            "machine_id": machine_id,
        }
        if "proc_time" in dispatch_option:
            option["proc_time"] = float(dispatch_option["proc_time"])
        machine = _compact_machine_summary(
            machine_summaries.get(machine_id, machine_summaries.get(str(machine_id), {}))
        )
        if machine:
            option["machine"] = machine
        options.append(option)
    return options or [{"action": int(action)} for action in legal_actions]


def _compact_commitment_actions(
    *,
    legal_actions: list[int],
    agent_view: Mapping[str, object],
) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    for target in _iter_legal_action_targets(legal_actions=legal_actions, agent_view=agent_view):
        option: dict[str, object] = {
            "action": int(target.get("action_id", -1)),
        }
        if "action_kind" in target:
            option["action_kind"] = str(target["action_kind"])
        options.append(option)
    return options or [{"action": int(action)} for action in legal_actions]


def _compact_generic_action_targets(
    *,
    legal_actions: list[int],
    agent_view: Mapping[str, object],
) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    for target in _iter_legal_action_targets(legal_actions=legal_actions, agent_view=agent_view):
        option: dict[str, object] = {
            "action": int(target.get("action_id", -1)),
        }
        if "target_id" in target:
            option["target_id"] = target["target_id"]
        if "action_kind" in target:
            option["action_kind"] = str(target["action_kind"])
        options.append(option)
    return options or [{"action": int(action)} for action in legal_actions]


def _iter_legal_action_targets(
    *,
    legal_actions: list[int],
    agent_view: Mapping[str, object],
) -> list[dict[str, object]]:
    action_targets = list(dict(agent_view.get("action_semantics", {})).get("action_targets", ()))
    targets: list[dict[str, object]] = []
    for raw_target in action_targets:
        target = dict(raw_target)
        action_id = int(target.get("action_id", -1))
        if action_id in legal_actions:
            targets.append(target)
    return targets


def _compact_area_summary(summary: object) -> dict[str, object]:
    source = dict(summary) if isinstance(summary, Mapping) else {}
    return _copy_present(
        source,
        (
            "active_jobs",
            "backlog_jobs",
            "busy_machines",
            "down_machines",
            "energy_kwh",
        ),
    )


def _compact_cell_summary(summary: object) -> dict[str, object]:
    source = dict(summary) if isinstance(summary, Mapping) else {}
    return _copy_present(
        source,
        (
            "processing_time_scale",
            "active_jobs",
            "backlog_jobs",
            "busy_machines",
            "down_machines",
        ),
    )


def _compact_machine_summary(summary: object) -> dict[str, object]:
    source = dict(summary) if isinstance(summary, Mapping) else {}
    return _copy_present(
        source,
        (
            "status",
            "remaining_time",
            "down_time",
        ),
    )


def _copy_present(
    source: Mapping[str, object],
    keys: tuple[str, ...],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key in keys:
        if key in source:
            payload[key] = source[key]
    return payload


def _round_number(value: object) -> object:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return round(float(value), 2)
    return value


def _int_list(raw_values: object) -> list[int]:
    if raw_values is None:
        return []
    return [int(value) for value in list(raw_values)]


def _int_map_of_lists(raw_mapping: object) -> dict[int, list[int]]:
    if not isinstance(raw_mapping, Mapping):
        return {}
    return {
        int(key): [int(value) for value in list(values)]
        for key, values in raw_mapping.items()
    }


def _dict_map(raw_mapping: object) -> dict[object, dict[str, object]]:
    if not isinstance(raw_mapping, Mapping):
        return {}
    return {
        key: dict(value) if isinstance(value, Mapping) else {}
        for key, value in raw_mapping.items()
    }
