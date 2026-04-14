from __future__ import annotations

import contextlib
from collections import defaultdict
import io
import json
import os
import re
import sys
import time
from typing import Any
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import FrameworkBackedTreeDriver
from .deterministic_kernel import DeterministicCoordinationKernel
from .llm_prompting import build_openai_compat_benchmark_messages


CREWAI_LLM_BACKEND = "openai_compat"


def _legal_action_indices(agent_view: Mapping[str, object]) -> list[int]:
    mask = list(agent_view.get("action_mask", []))
    return [index for index, enabled in enumerate(mask) if bool(enabled)]


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _parse_llm_action_text(
    text: str,
    *,
    legal_actions: list[int],
) -> int | None:
    if not legal_actions:
        return 0

    normalized = _strip_code_fence(text)
    for candidate in (normalized, text.strip()):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, int) and int(parsed) in legal_actions:
            return int(parsed)
        if isinstance(parsed, Mapping):
            action = parsed.get("action")
            if isinstance(action, int) and int(action) in legal_actions:
                return int(action)
            if isinstance(action, str) and action.strip().lstrip("-").isdigit():
                parsed_int = int(action.strip())
                if parsed_int in legal_actions:
                    return parsed_int

    for match in re.findall(r"-?\d+", normalized):
        parsed_int = int(match)
        if parsed_int in legal_actions:
            return parsed_int
    return None


def _build_llm_messages(
    *,
    agent_id: str,
    agent_view: Mapping[str, object],
) -> list[dict[str, str]]:
    return build_openai_compat_benchmark_messages(
        agent_id=agent_id,
        agent_view=agent_view,
    )


def _extract_openai_compat_content(response_payload: Mapping[str, object]) -> str | None:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        return None
    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            text = item.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        if text_parts:
            return "".join(text_parts)
    return None


class CrewAITreeDriver(FrameworkBackedTreeDriver):
    """Framework-shaped driver that models manager-worker routing memory."""

    framework_name = "CrewAI"
    python_package = "crewai"
    controller_name = "external_framework_crewai"

    def __init__(
        self,
        *,
        manager_penalty: float = 2.0,
        strict_framework: bool = False,
        controller_name: str | None = None,
    ) -> None:
        self.manager_penalty = float(manager_penalty)
        self.kernel = DeterministicCoordinationKernel()
        self._runtime_serial = 0
        super().__init__(
            strict_framework=strict_framework,
            controller_name=controller_name,
        )

    def reset_episode(self) -> None:
        super().reset_episode()
        self._backlog_selection_counts: dict[int, int] = {}
        self._manager_counts: dict[str, dict[int, int]] = defaultdict(dict)
        if self._framework_available:
            self._runtime_serial += 1
            self._native_runtime = self._build_native_runtime()

    def _select_framework_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> int:
        if self._native_runtime_enabled():
            return self._select_native_action(
                agent_id=agent_id,
                agent_view=agent_view,
            )
        return self._select_contract_action(
            agent_id=agent_id,
            agent_view=agent_view,
        )

    def _select_contract_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> int:
        role = str(agent_view.get("role", "unknown"))
        if role == "plant":
            plant_decision_kind = str(agent_view.get("plant_decision_kind", "area_selection"))
            if plant_decision_kind == "backlog_selection":
                selected = self.kernel.select_backlog(
                    agent_view,
                    selection_counts=self._backlog_selection_counts,
                )
            else:
                selected = self.kernel.select_area(
                    agent_view,
                    selection_counts=self._manager_counts["plant"],
                    routing_penalty=self.manager_penalty,
                )
            self._apply_action_side_effects(
                agent_id=agent_id,
                agent_view=agent_view,
                selected_action=int(selected),
            )
            return selected
        if role == "area":
            selected = self.kernel.select_cell(
                agent_view,
                selection_counts=self._manager_counts[agent_id],
                routing_penalty=self.manager_penalty,
            )
            self._apply_action_side_effects(
                agent_id=agent_id,
                agent_view=agent_view,
                selected_action=int(selected),
            )
            return selected
        return self.kernel.select_dispatch(agent_view)

    def _build_native_runtime(self) -> dict[str, object]:
        os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
        os.environ.setdefault("OTEL_SDK_DISABLED", "true")
        from crewai.flow.flow import Flow, start

        runtime_tag = f"fms_bench_crewai_runtime_{self._runtime_serial}"
        driver = self

        class _DecisionFlow(Flow):
            def __init__(self, *, payload: Mapping[str, object]) -> None:
                super().__init__()
                self.state["payload"] = dict(payload)

            @start()
            def decide(self) -> dict[str, int]:
                payload = dict(self.state.get("payload", {}))
                action = driver._select_contract_action(
                    agent_id=str(payload.get("agent_id", "unknown")),
                    agent_view=dict(payload.get("agent_view", {})),
                )
                return {"action": int(action)}

        return {
            "runtime_tag": runtime_tag,
            "flow_class": _DecisionFlow,
        }

    def _select_native_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> int:
        assert isinstance(self._native_runtime, dict)
        flow_class = self._native_runtime["flow_class"]
        assert callable(flow_class)
        payload = {
            "agent_id": agent_id,
            "agent_view": dict(agent_view),
            "framework_name": self.framework_name,
        }
        flow = flow_class(payload=payload)
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            result = flow.kickoff()
        if isinstance(result, Mapping) and "action" in result:
            return int(result["action"])
        return int(result)

    def _increment_count(self, key: str, choice: int) -> None:
        counts = self._manager_counts[key]
        counts[int(choice)] = counts.get(int(choice), 0) + 1

    def _apply_action_side_effects(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
        selected_action: int,
    ) -> None:
        role = str(agent_view.get("role", "unknown"))
        if role == "plant":
            plant_decision_kind = str(agent_view.get("plant_decision_kind", "area_selection"))
            if plant_decision_kind == "backlog_selection":
                visible_candidates = list(agent_view.get("visible_backlog_candidates", ()))
                if 0 <= int(selected_action) < len(visible_candidates):
                    selected_job_id = int(
                        dict(visible_candidates[int(selected_action)]).get("job_id", -1)
                    )
                    if selected_job_id >= 0:
                        self._backlog_selection_counts[selected_job_id] = (
                            self._backlog_selection_counts.get(selected_job_id, 0) + 1
                        )
                return
            self._increment_count("plant", int(selected_action))
            return
        if role == "area":
            self._increment_count(agent_id, int(selected_action))


class CrewAILLMTreeDriver(CrewAITreeDriver):
    """CrewAI-shaped driver that queries an OpenAI-compatible endpoint without extra deps."""

    controller_name = "external_framework_crewai_llm"

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        manager_penalty: float = 2.0,
        strict_framework: bool = False,
        controller_name: str | None = None,
        request_timeout: float | None = None,
        request_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self.llm_backend = CREWAI_LLM_BACKEND
        self.model_name = str(model_name)
        self.llm_model = self.model_name
        self.base_url = str(base_url)
        self.llm_base_url = self.base_url
        self.api_key = None if api_key is None else str(api_key)
        self.request_timeout = None if request_timeout is None else float(request_timeout)
        self.request_retries = max(int(request_retries), 0)
        self.retry_backoff_seconds = max(float(retry_backoff_seconds), 0.0)
        super().__init__(
            manager_penalty=manager_penalty,
            strict_framework=strict_framework,
            controller_name=controller_name or self.controller_name,
        )

    @property
    def framework_runtime_info(self) -> dict[str, object]:
        payload = dict(super().framework_runtime_info)
        payload.update(
            {
                "llm_backend": self.llm_backend,
                "llm_model": self.llm_model,
                "llm_base_url": self.llm_base_url,
            }
        )
        return payload

    def _select_framework_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> int:
        if self._native_runtime_enabled():
            return self._select_native_action(
                agent_id=agent_id,
                agent_view=agent_view,
            )
        return self._select_llm_or_contract_action(
            agent_id=agent_id,
            agent_view=agent_view,
        )

    def _build_native_runtime(self) -> dict[str, object]:
        os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
        os.environ.setdefault("OTEL_SDK_DISABLED", "true")
        from crewai.flow.flow import Flow, start

        runtime_tag = f"fms_bench_crewai_runtime_{self._runtime_serial}"
        driver = self

        class _DecisionFlow(Flow):
            def __init__(self, *, payload: Mapping[str, object]) -> None:
                super().__init__()
                self.state["payload"] = dict(payload)

            @start()
            def decide(self) -> dict[str, int]:
                payload = dict(self.state.get("payload", {}))
                action = driver._select_llm_or_contract_action(
                    agent_id=str(payload.get("agent_id", "unknown")),
                    agent_view=dict(payload.get("agent_view", {})),
                )
                return {"action": int(action)}

        return {
            "runtime_tag": runtime_tag,
            "flow_class": _DecisionFlow,
        }

    def _select_llm_or_contract_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> int:
        legal_actions = _legal_action_indices(agent_view)
        if not legal_actions:
            self._record_decision_source(
                agent_id=agent_id,
                source="no_legal_action",
                action=0,
                legal_actions=legal_actions,
            )
            return 0

        for attempt in range(self.request_retries + 1):
            try:
                response_text = self._request_llm_action_text(
                    agent_id=agent_id,
                    agent_view=agent_view,
                )
            except Exception:
                response_text = None
            selected = (
                None
                if response_text is None
                else _parse_llm_action_text(
                    response_text,
                    legal_actions=legal_actions,
                )
            )
            if selected is not None:
                self._apply_action_side_effects(
                    agent_id=agent_id,
                    agent_view=agent_view,
                    selected_action=int(selected),
                )
                self._record_decision_source(
                    agent_id=agent_id,
                    source="llm",
                    action=int(selected),
                    legal_actions=legal_actions,
                )
                return int(selected)
            if attempt < self.request_retries and self.retry_backoff_seconds > 0.0:
                time.sleep(self.retry_backoff_seconds * float(attempt + 1))

        selected = self._select_contract_action(
            agent_id=agent_id,
            agent_view=agent_view,
        )
        self._record_decision_source(
            agent_id=agent_id,
            source="fallback_contract",
            action=int(selected),
            legal_actions=legal_actions,
        )
        return int(selected)

    def _request_llm_action_text(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> str | None:
        if not self.model_name or not self.base_url:
            return None

        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": _build_llm_messages(
                    agent_id=agent_id,
                    agent_view=agent_view,
                ),
                "temperature": 0.0,
                "max_tokens": 32,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key or 'EMPTY'}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            if self.request_timeout is None or self.request_timeout <= 0.0:
                with urlopen(request) as response:
                    raw = response.read().decode("utf-8")
            else:
                with urlopen(request, timeout=self.request_timeout) as response:
                    raw = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, OSError):
            return None

        try:
            response_payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(response_payload, Mapping):
            return None
        return _extract_openai_compat_content(response_payload)


def _register_crewai_llm_driver() -> None:
    current_module = sys.modules[__name__]
    for module_name in (
        "fms_bench.v2.framework_drivers",
        "fms_bench.v2",
        "fms_bench",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        setattr(module, "CrewAILLMTreeDriver", CrewAILLMTreeDriver)
        exported = getattr(module, "__all__", None)
        if isinstance(exported, list) and "CrewAILLMTreeDriver" not in exported:
            exported.append("CrewAILLMTreeDriver")
    setattr(current_module, "CrewAILLMTreeDriver", CrewAILLMTreeDriver)


_register_crewai_llm_driver()
