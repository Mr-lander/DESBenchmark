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
        self._decision_source_counts: dict[str, int] = defaultdict(int)

    def close(self) -> None:
        self._teardown_native_runtime(self._native_runtime)
        self._native_runtime = None

    @property
    def decision_audit_payload(self) -> dict[str, object]:
        return {
            "framework_decision_audit_contract": FRAMEWORK_DECISION_AUDIT_CONTRACT,
            "decision_source_trace": [dict(item) for item in self._decision_source_trace],
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
            "framework_llm_decision_count": int(counts.get("llm", 0)),
            "framework_fallback_contract_decision_count": int(
                counts.get("fallback_contract", 0)
            ),
            "framework_no_legal_action_count": int(counts.get("no_legal_action", 0)),
            "decision_source_trace": [dict(item) for item in self._decision_source_trace],
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

    def select_actions(self, decision: Mapping[str, object]) -> dict[str, int]:
        self._before_select_actions(decision)
        actions: dict[str, int] = {}
        for agent_id, agent_view in dict(decision.get("active_agents", {})).items():
            actions[str(agent_id)] = int(
                self._select_framework_action(
                    agent_id=str(agent_id),
                    agent_view=dict(agent_view),
                )
            )
        self._after_select_actions(decision, actions)
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
