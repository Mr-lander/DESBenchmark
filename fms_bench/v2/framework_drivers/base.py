from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib.util import find_spec
from typing import Any

from ..framework_envs import framework_env_payload


FRAMEWORK_ACTION_INPUT_CONTRACT = (
    "framework_action_selection_receives_per_agent_controller_payload_only"
)
FRAMEWORK_PAYLOAD_AUDIT_STATUS = (
    "checked_in_framework_driver_source_audit_no_artifact_debug_channel_reference_2026-03-28"
)
FRAMEWORK_DECISION_AUDIT_CONTRACT = (
    "per-agent-decision-source-trace-with-llm-fallback-and-no-legal-action-counts"
)
_WORKING_MEMORY_RECENT_LIMIT = 6
_WORKING_MEMORY_OPEN_COMMITMENT_LIMIT = 4
_WORKING_MEMORY_TOP_TARGET_LIMIT = 3
_WORKING_MEMORY_ROUND_DIGITS = 2
_TERMINAL_COMMITMENT_STATUSES = frozenset(
    {
        "completed",
        "done",
        "settled",
        "closed",
        "cancelled",
        "rejected",
        "failed",
    }
)


def _new_working_memory_summary() -> dict[str, Any]:
    return {
        "compressed_count": 0,
        "decision_counts_by_kind": {},
        "target_counts": {},
        "reject_count": 0,
        "local_blocked_count": 0,
        "completion_status_counts": {},
        "_sum_wip_load": 0.0,
        "_sum_energy_overshoot": 0.0,
        "_sum_carbon_overshoot": 0.0,
        "_sum_busy_machines": 0.0,
        "_sum_down_machines": 0.0,
    }


class FrameworkBackedTreeDriver:
    """Base driver for framework-shaped controllers over the public external session."""

    framework_name = "framework"
    python_package: str | None = None
    controller_name = "external_framework_base"
    uses_external_session = True

    def __init__(
        self,
        *,
        strict_framework: bool = False,
        controller_name: str | None = None,
    ) -> None:
        self.strict_framework = bool(strict_framework)
        self._framework_available = self._detect_framework_availability()
        self._native_runtime: Any | None = None
        if self.strict_framework and not self._framework_available:
            package = self.python_package or self.framework_name
            raise ImportError(
                f"{self.framework_name} driver requires optional dependency: {package}"
            )
        if controller_name is not None:
            self.controller_name = str(controller_name)
        self.reset_episode()

    @property
    def name(self) -> str:
        return str(self.controller_name)

    @property
    def framework_runtime_info(self) -> dict[str, object]:
        payload = {
            "framework_name": self.framework_name,
            "python_package": self.python_package,
            "available": bool(self._framework_available),
            "backend_mode": "native_framework" if self._native_runtime_enabled() else "contract_stub",
        }
        payload.update(
            framework_env_payload(
                framework_name=self.framework_name,
                python_package=self.python_package,
            )
        )
        return payload

    def reset_episode(self) -> None:
        self._teardown_native_runtime(self._native_runtime)
        self._native_runtime = None
        self._episode_state: dict[str, Any] = {}
        self._decision_source_trace: list[dict[str, object]] = []
        self._decision_failure_trace: list[dict[str, object]] = []
        self._decision_source_counts: dict[str, int] = defaultdict(int)
        self._decision_epoch_index = 0
        self._recent_working_memory: dict[str, list[dict[str, object]]] = defaultdict(list)
        self._rolled_working_memory: dict[str, dict[str, Any]] = defaultdict(
            _new_working_memory_summary
        )

    def close(self) -> None:
        self._teardown_native_runtime(self._native_runtime)
        self._native_runtime = None

    @property
    def decision_audit_payload(self) -> dict[str, object]:
        return {
            "framework_decision_audit_contract": FRAMEWORK_DECISION_AUDIT_CONTRACT,
            "decision_source_trace": [dict(item) for item in self._decision_source_trace],
            "decision_failure_trace": [dict(item) for item in self._decision_failure_trace],
            "decision_source_counts": {
                str(source): int(count)
                for source, count in self._decision_source_counts.items()
            },
        }

    @property
    def framework_decision_audit_info(self) -> dict[str, object]:
        counts = {
            str(source): int(count)
            for source, count in self._decision_source_counts.items()
        }
        return {
            "framework_decision_audit_contract": FRAMEWORK_DECISION_AUDIT_CONTRACT,
            "framework_decision_source_trace": [dict(item) for item in self._decision_source_trace],
            "framework_decision_failure_trace": [
                dict(item) for item in self._decision_failure_trace
            ],
            "framework_llm_decision_count": int(counts.get("llm", 0)),
            "framework_fallback_contract_decision_count": int(
                counts.get("fallback_contract", 0)
            ),
            "framework_no_legal_action_count": int(counts.get("no_legal_action", 0)),
            "decision_source_trace": [dict(item) for item in self._decision_source_trace],
            "decision_failure_trace": [dict(item) for item in self._decision_failure_trace],
            "llm_decision_count": int(counts.get("llm", 0)),
            "fallback_contract_decision_count": int(counts.get("fallback_contract", 0)),
            "no_legal_action_count": int(counts.get("no_legal_action", 0)),
        }

    def _record_decision_source(
        self,
        *,
        agent_id: str,
        source: str,
        action: int,
        legal_actions: Sequence[int] | None = None,
    ) -> None:
        key = str(source)
        self._decision_source_counts[key] = int(self._decision_source_counts.get(key, 0)) + 1
        record: dict[str, object] = {
            "agent_id": str(agent_id),
            "source": key,
            "action": int(action),
        }
        if legal_actions is not None:
            record["legal_actions"] = [int(item) for item in legal_actions]
        self._decision_source_trace.append(record)

    def _record_decision_failure(
        self,
        *,
        agent_id: str,
        attempt: int,
        failure_kind: str,
        legal_actions: Sequence[int] | None = None,
        raw_response: object | None = None,
        exception: BaseException | None = None,
    ) -> None:
        record: dict[str, object] = {
            "agent_id": str(agent_id),
            "attempt": int(attempt),
            "failure_kind": str(failure_kind),
        }
        if legal_actions is not None:
            record["legal_actions"] = [int(item) for item in legal_actions]
        if raw_response is not None:
            raw_text = str(raw_response)
            record["raw_response_excerpt"] = raw_text[:240]
            record["raw_response_length"] = len(raw_text)
        if exception is not None:
            record["exception_type"] = type(exception).__name__
            message = str(exception)
            if message:
                record["exception_message"] = message[:240]
        self._decision_failure_trace.append(record)

    def select_actions(self, decision: Mapping[str, object]) -> dict[str, int]:
        self._before_select_actions(decision)
        actions: dict[str, int] = {}
        for agent_id, agent_view in dict(decision.get("active_agents", {})).items():
            prompt_agent_view = self._augment_agent_view_for_prompt(
                agent_id=str(agent_id),
                agent_view=dict(agent_view),
            )
            actions[str(agent_id)] = int(
                self._select_framework_action(
                    agent_id=str(agent_id),
                    agent_view=prompt_agent_view,
                )
            )
        self._after_select_actions(decision, actions)
        self._record_working_memory(decision=decision, actions=actions)
        self._decision_epoch_index += 1
        return actions

    def _before_select_actions(self, decision: Mapping[str, object]) -> None:
        del decision

    def _after_select_actions(
        self,
        decision: Mapping[str, object],
        actions: Mapping[str, int],
    ) -> None:
        del decision, actions

    def _select_framework_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> int:
        raise NotImplementedError

    def _detect_framework_availability(self) -> bool:
        if self.python_package is None:
            return False
        return find_spec(self.python_package) is not None

    def _native_runtime_enabled(self) -> bool:
        return self._native_runtime is not None

    def _teardown_native_runtime(self, runtime: Any) -> None:
        self._teardown_native_runtime_value(runtime, visited=set())

    def _teardown_native_runtime_value(
        self,
        runtime: Any,
        *,
        visited: set[int],
    ) -> None:
        if runtime is None:
            return
        runtime_id = id(runtime)
        if runtime_id in visited:
            return
        visited.add(runtime_id)

        if isinstance(runtime, Mapping):
            for value in runtime.values():
                self._teardown_native_runtime_value(value, visited=visited)
            return

        if isinstance(runtime, Sequence) and not isinstance(runtime, (str, bytes, bytearray)):
            for value in runtime:
                self._teardown_native_runtime_value(value, visited=visited)
            return

        for method_name in ("close", "aclose", "stop", "shutdown", "cleanup"):
            method = getattr(runtime, method_name, None)
            if not callable(method):
                continue
            result = method()
            if inspect.isawaitable(result):
                asyncio.run(result)
            return

    def _augment_agent_view_for_prompt(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> dict[str, object]:
        payload = dict(agent_view)
        working_memory = self._build_working_memory_payload(agent_id)
        if working_memory:
            payload["working_memory"] = working_memory
        return payload

    def _build_working_memory_payload(self, agent_id: str) -> dict[str, object]:
        recent_outcomes = [
            dict(item) for item in self._recent_working_memory.get(str(agent_id), ())
        ]
        open_commitments = self._open_commitments_from_recent_outcomes(recent_outcomes)
        rolled_summary = self._render_rolled_summary(
            self._rolled_working_memory.get(str(agent_id), {})
        )
        payload: dict[str, object] = {}
        if recent_outcomes:
            payload["recent_outcomes"] = recent_outcomes[-_WORKING_MEMORY_RECENT_LIMIT:]
        if open_commitments:
            payload["open_commitments"] = open_commitments[:_WORKING_MEMORY_OPEN_COMMITMENT_LIMIT]
        if rolled_summary:
            payload["rolled_summary"] = rolled_summary
        return payload

    def _record_working_memory(
        self,
        *,
        decision: Mapping[str, object],
        actions: Mapping[str, int],
    ) -> None:
        for agent_id, raw_agent_view in dict(decision.get("active_agents", {})).items():
            agent_key = str(agent_id)
            if agent_key not in actions:
                continue
            recent_outcomes = self._recent_working_memory[agent_key]
            recent_outcomes.append(
                self._build_recent_outcome_record(
                    agent_id=agent_key,
                    agent_view=dict(raw_agent_view),
                    action=int(actions[agent_key]),
                )
            )
            while len(recent_outcomes) > _WORKING_MEMORY_RECENT_LIMIT:
                oldest = recent_outcomes.pop(0)
                self._merge_recent_outcome_into_summary(
                    self._rolled_working_memory[agent_key],
                    oldest,
                )

    def _build_recent_outcome_record(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
        action: int,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "step": int(self._decision_epoch_index),
            "kind": self._decision_kind(agent_view),
            "action": int(action),
            "agent": str(agent_id),
        }
        selected_target = self._selected_target_label(agent_view=agent_view, action=int(action))
        if selected_target:
            record["selected_target"] = selected_target
        observed_feedback = self._compact_feedback_for_memory(agent_view)
        if observed_feedback:
            record["observed_feedback"] = observed_feedback
        visible_commitments = self._extract_visible_commitments(agent_view)
        if visible_commitments:
            record["visible_commitments"] = visible_commitments
        event_types = self._compact_event_types(agent_view)
        if event_types:
            record["event_types"] = event_types
        return record

    @staticmethod
    def _decision_kind(agent_view: Mapping[str, object]) -> str:
        role = str(agent_view.get("role", "unknown"))
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

    @staticmethod
    def _selected_target_label(
        *,
        agent_view: Mapping[str, object],
        action: int,
    ) -> str | None:
        action_semantics = dict(agent_view.get("action_semantics", {}))
        action_type = str(action_semantics.get("action_type", ""))
        for raw_target in list(action_semantics.get("action_targets", ())):
            target = dict(raw_target)
            if int(target.get("action_id", -1)) != int(action):
                continue
            action_kind = target.get("action_kind")
            if isinstance(action_kind, str) and action_kind:
                return f"{action_type or 'action'}:{action_kind}"
            target_id = target.get("target_id")
            if target_id is None:
                break
            if action_type == "select_backlog_item":
                return f"job:{int(target_id)}"
            if action_type in {"select_child_holon", "revise_contract"}:
                return f"area:{int(target_id)}"
            if action_type in {"select_execution_cell", "award_bid"}:
                return f"cell:{int(target_id)}"
            return f"target:{target_id}"

        for index, raw_dispatch in enumerate(list(agent_view.get("dispatch_options", ()))):
            if int(index) != int(action):
                continue
            machine_id = dict(raw_dispatch).get("machine_id")
            if machine_id is not None:
                return f"machine:{int(machine_id)}"

        for index, raw_candidate in enumerate(list(agent_view.get("visible_backlog_candidates", ()))):
            candidate = dict(raw_candidate)
            if int(candidate.get("backlog_index", index)) != int(action):
                continue
            job_id = candidate.get("job_id")
            if job_id is not None:
                return f"job:{int(job_id)}"
        return None

    @staticmethod
    def _compact_feedback_for_memory(agent_view: Mapping[str, object]) -> dict[str, object]:
        feedback_view = dict(agent_view.get("feedback_view", {}))
        progress = {
            str(key): FrameworkBackedTreeDriver._round_prompt_number(value)
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
                capacity[str(key)] = FrameworkBackedTreeDriver._round_prompt_number(
                    capacity_risk[key]
                )
        for key in ("busy_machines", "down_machines"):
            if key in machine_health:
                capacity[str(key)] = FrameworkBackedTreeDriver._round_prompt_number(
                    machine_health[key]
                )
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

    @staticmethod
    def _extract_visible_commitments(agent_view: Mapping[str, object]) -> list[dict[str, object]]:
        parent_context = dict(agent_view.get("parent_context", {}))
        tracked_contracts = list(
            dict(dict(agent_view.get("feedback_view", {})).get("completion", {})).get(
                "tracked_contracts",
                (),
            )
        )
        current_contract = parent_context.get("current_contract")
        commitment_candidates: list[object] = []
        if current_contract is not None:
            commitment_candidates.append(current_contract)
        commitment_candidates.extend(tracked_contracts)

        visible_commitments: list[dict[str, object]] = []
        seen_keys: set[str] = set()
        for raw_item in commitment_candidates:
            if not isinstance(raw_item, Mapping):
                continue
            item = dict(raw_item)
            commitment = {
                str(key): item[key]
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
                )
                if key in item
            }
            if not commitment:
                continue
            identifier = str(
                commitment.get("contract_id")
                or commitment.get("work_id")
                or commitment.get("job_id")
                or f"anon-{len(visible_commitments)}"
            )
            if identifier in seen_keys:
                continue
            seen_keys.add(identifier)
            if "completed_operations" in commitment:
                commitment["completed_operations"] = int(commitment["completed_operations"])
            if "total_operations" in commitment and commitment["total_operations"] is not None:
                commitment["total_operations"] = int(commitment["total_operations"])
            visible_commitments.append(commitment)
        return visible_commitments[:_WORKING_MEMORY_OPEN_COMMITMENT_LIMIT]

    @staticmethod
    def _compact_event_types(agent_view: Mapping[str, object]) -> list[str]:
        event_types: list[str] = []
        for raw_event in list(agent_view.get("local_env_events", ())):
            if not isinstance(raw_event, Mapping):
                continue
            event = dict(raw_event)
            event_type = event.get("type") or event.get("kind") or event.get("event_type")
            if isinstance(event_type, str) and event_type:
                event_types.append(event_type)
        return event_types[:4]

    @staticmethod
    def _merge_recent_outcome_into_summary(
        summary: dict[str, Any],
        outcome: Mapping[str, object],
    ) -> None:
        summary["compressed_count"] = int(summary.get("compressed_count", 0)) + 1

        kind = str(outcome.get("kind", "unknown"))
        decision_counts = dict(summary.get("decision_counts_by_kind", {}))
        decision_counts[kind] = int(decision_counts.get(kind, 0)) + 1
        summary["decision_counts_by_kind"] = decision_counts

        selected_target = outcome.get("selected_target")
        if isinstance(selected_target, str) and selected_target:
            target_counts = dict(summary.get("target_counts", {}))
            target_counts[selected_target] = int(target_counts.get(selected_target, 0)) + 1
            summary["target_counts"] = target_counts
            if selected_target.endswith(":reject"):
                summary["reject_count"] = int(summary.get("reject_count", 0)) + 1

        observed_feedback = dict(outcome.get("observed_feedback", {}))
        capacity = dict(observed_feedback.get("capacity", {}))
        if bool(capacity.get("local_blocked", False)):
            summary["local_blocked_count"] = int(summary.get("local_blocked_count", 0)) + 1

        FrameworkBackedTreeDriver._accumulate_summary_mean(summary, "_sum_wip_load", capacity.get("wip_load"))
        FrameworkBackedTreeDriver._accumulate_summary_mean(
            summary,
            "_sum_energy_overshoot",
            capacity.get("energy_overshoot"),
        )
        FrameworkBackedTreeDriver._accumulate_summary_mean(
            summary,
            "_sum_carbon_overshoot",
            capacity.get("carbon_overshoot"),
        )
        FrameworkBackedTreeDriver._accumulate_summary_mean(
            summary,
            "_sum_busy_machines",
            capacity.get("busy_machines"),
        )
        FrameworkBackedTreeDriver._accumulate_summary_mean(
            summary,
            "_sum_down_machines",
            capacity.get("down_machines"),
        )

        completion_counts = dict(observed_feedback.get("completion_status_counts", {}))
        aggregate_counts = dict(summary.get("completion_status_counts", {}))
        for status, count in completion_counts.items():
            aggregate_counts[str(status)] = int(aggregate_counts.get(str(status), 0)) + int(count)
        summary["completion_status_counts"] = aggregate_counts

    @staticmethod
    def _accumulate_summary_mean(
        summary: dict[str, Any],
        key: str,
        value: object,
    ) -> None:
        if not isinstance(value, (int, float)):
            return
        summary[key] = float(summary.get(key, 0.0)) + float(value)

    @staticmethod
    def _render_rolled_summary(summary: Mapping[str, object]) -> dict[str, object]:
        compressed_count = int(summary.get("compressed_count", 0) or 0)
        if compressed_count <= 0:
            return {}
        target_counts = {
            str(key): int(value)
            for key, value in dict(summary.get("target_counts", {})).items()
        }
        top_targets = [
            {"target": target, "count": count}
            for target, count in sorted(
                target_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:_WORKING_MEMORY_TOP_TARGET_LIMIT]
        ]
        payload: dict[str, object] = {
            "compressed_count": compressed_count,
            "decision_counts_by_kind": {
                str(key): int(value)
                for key, value in dict(summary.get("decision_counts_by_kind", {})).items()
            },
            "reject_count": int(summary.get("reject_count", 0)),
            "local_blocked_count": int(summary.get("local_blocked_count", 0)),
        }
        if top_targets:
            payload["top_repeated_targets"] = top_targets
        for source_key, output_key in (
            ("_sum_wip_load", "mean_wip_load"),
            ("_sum_energy_overshoot", "mean_energy_overshoot"),
            ("_sum_carbon_overshoot", "mean_carbon_overshoot"),
            ("_sum_busy_machines", "mean_busy_machines"),
            ("_sum_down_machines", "mean_down_machines"),
        ):
            payload[output_key] = FrameworkBackedTreeDriver._round_prompt_number(
                float(summary.get(source_key, 0.0)) / float(compressed_count)
            )
        completion_status_counts = {
            str(key): int(value)
            for key, value in dict(summary.get("completion_status_counts", {})).items()
        }
        if completion_status_counts:
            payload["completion_status_counts"] = completion_status_counts
        return payload

    @staticmethod
    def _open_commitments_from_recent_outcomes(
        recent_outcomes: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        visible: list[dict[str, object]] = []
        seen: set[str] = set()
        for outcome in reversed(list(recent_outcomes)):
            for raw_commitment in list(outcome.get("visible_commitments", ())):
                if not isinstance(raw_commitment, Mapping):
                    continue
                commitment = dict(raw_commitment)
                status = str(commitment.get("status", "")).strip().lower()
                if status in _TERMINAL_COMMITMENT_STATUSES:
                    continue
                identifier = str(
                    commitment.get("contract_id")
                    or commitment.get("work_id")
                    or commitment.get("job_id")
                    or f"anon-{len(visible)}"
                )
                if identifier in seen:
                    continue
                seen.add(identifier)
                visible.append(commitment)
                if len(visible) >= _WORKING_MEMORY_OPEN_COMMITMENT_LIMIT:
                    return visible
        return visible

    @staticmethod
    def _round_prompt_number(value: object) -> object:
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return round(float(value), _WORKING_MEMORY_ROUND_DIGITS)
        return value
