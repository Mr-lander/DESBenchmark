from __future__ import annotations

import asyncio
from collections import defaultdict
import json
import re
import sys
import time
from typing import Any, Mapping
from urllib import request as urllib_request

from .base import FrameworkBackedTreeDriver
from .deterministic_kernel import DeterministicCoordinationKernel
from .llm_prompting import (
    build_openai_compat_benchmark_generation_kwargs,
    build_openai_compat_benchmark_messages,
)

GOOGLE_ADK_LLM_BACKEND = "openai_compat"


def _encode_framework_payload(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


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


class _GoogleADKIngressAgent:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from google.adk.agents import BaseAgent
        from google.adk.events import Event, EventActions

        class _IngressAgent(BaseAgent):
            async def _run_async_impl(self, ctx):  # type: ignore[override]
                payload = {}
                if ctx.user_content and ctx.user_content.parts:
                    text_parts = [
                        part.text
                        for part in ctx.user_content.parts
                        if getattr(part, "text", None)
                    ]
                    if text_parts:
                        payload = json.loads("".join(text_parts))
                yield Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    branch=ctx.branch,
                    actions=EventActions(state_delta={"controller_payload": payload}),
                )

        return _IngressAgent(*args, **kwargs)


class _GoogleADKPolicyAgent:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from google.adk.agents import BaseAgent
        from google.adk.events import Event, EventActions
        from pydantic import Field

        class _PolicyAgent(BaseAgent):
            driver: "GoogleADKTreeDriver" = Field(exclude=True)

            async def _run_async_impl(self, ctx):  # type: ignore[override]
                payload = dict(ctx.session.state.get("controller_payload", {}))
                action = int(
                    self.driver._select_contract_action(
                        agent_id=str(payload.get("agent_id", "unknown")),
                        agent_view=dict(payload.get("agent_view", {})),
                    )
                )
                yield Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    branch=ctx.branch,
                    actions=EventActions(
                        state_delta={
                            "selected_action": int(action),
                            "selected_agent_id": str(payload.get("agent_id", "unknown")),
                        }
                    ),
                )

        return _PolicyAgent(*args, **kwargs)


class _GoogleADKLLMPolicyAgent:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from google.adk.agents import BaseAgent
        from google.adk.events import Event, EventActions
        from pydantic import Field

        class _PolicyAgent(BaseAgent):
            driver: Any = Field(exclude=True)

            async def _run_async_impl(self, ctx):  # type: ignore[override]
                payload = dict(ctx.session.state.get("controller_payload", {}))
                action = int(
                    self.driver._select_llm_or_contract_action(
                        agent_id=str(payload.get("agent_id", "unknown")),
                        agent_view=dict(payload.get("agent_view", {})),
                    )
                )
                yield Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    branch=ctx.branch,
                    actions=EventActions(
                        state_delta={
                            "selected_action": int(action),
                            "selected_agent_id": str(payload.get("agent_id", "unknown")),
                        }
                    ),
                )

        return _PolicyAgent(*args, **kwargs)


class GoogleADKTreeDriver(FrameworkBackedTreeDriver):
    """Framework-shaped driver that models workflow loop state and delegation counts."""

    framework_name = "Google ADK"
    python_package = "google.adk"
    controller_name = "external_framework_google_adk"

    def __init__(
        self,
        *,
        loop_penalty: float = 2.0,
        disturbance_weight: float = 1.5,
        strict_framework: bool = False,
        controller_name: str | None = None,
    ) -> None:
        self.loop_penalty = float(loop_penalty)
        self.disturbance_weight = float(disturbance_weight)
        self.kernel = DeterministicCoordinationKernel()
        self._runtime_serial = 0
        super().__init__(
            strict_framework=strict_framework,
            controller_name=controller_name,
        )

    def reset_episode(self) -> None:
        super().reset_episode()
        self._workflow_counts: dict[str, dict[int, int]] = defaultdict(dict)
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
                    selection_counts=self._workflow_counts["backlog"],
                )
                self._apply_action_side_effects(
                    agent_id=agent_id,
                    agent_view=agent_view,
                    selected_action=int(selected),
                )
                return selected
            selected = self.kernel.select_area(
                agent_view,
                selection_counts=self._workflow_counts["plant"],
                pressure_weight=self.disturbance_weight,
                routing_penalty=self.loop_penalty,
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
                selection_counts=self._workflow_counts[agent_id],
                pressure_weight=self.disturbance_weight,
                routing_penalty=self.loop_penalty,
            )
            self._apply_action_side_effects(
                agent_id=agent_id,
                agent_view=agent_view,
                selected_action=int(selected),
            )
            return selected
        return self.kernel.select_dispatch(
            agent_view,
            disturbance_weight=self.disturbance_weight,
        )

    def _build_native_runtime(self) -> dict[str, object]:
        from google.adk.agents import SequentialAgent
        from google.adk.runners import InMemoryRunner

        runtime_tag = f"fms_bench_google_adk_runtime_{self._runtime_serial}"
        root_agent = SequentialAgent(
            name=f"{runtime_tag}_root",
            sub_agents=[
                _GoogleADKIngressAgent(name=f"{runtime_tag}_ingress"),
                _GoogleADKPolicyAgent(
                    name=f"{runtime_tag}_policy",
                    driver=self,
                ),
            ],
        )
        return {
            "runner": InMemoryRunner(
                agent=root_agent,
                app_name=runtime_tag,
            ),
            "app_name": runtime_tag,
            "user_id": f"benchmark_{self._runtime_serial}",
            "runtime_tag": runtime_tag,
        }

    def _select_native_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> int:
        from google.genai import types

        async def _run_workflow() -> int:
            assert isinstance(self._native_runtime, dict)
            runner = self._native_runtime["runner"]
            app_name = str(self._native_runtime["app_name"])
            user_id = str(self._native_runtime["user_id"])
            runtime_tag = str(self._native_runtime["runtime_tag"])
            session_id = f"{runtime_tag}::{agent_id}"
            session = await runner.session_service.get_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
            )
            if session is None:
                await runner.session_service.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                )

            payload = _encode_framework_payload(
                {
                    "agent_id": agent_id,
                    "agent_view": dict(agent_view),
                    "framework_name": self.framework_name,
                }
            )
            async for _ in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=payload)],
                ),
            ):
                pass

            final_session = await runner.session_service.get_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
            )
            if final_session is None or "selected_action" not in final_session.state:
                raise RuntimeError(f"Google ADK native workflow produced no action for {agent_id}")
            return int(final_session.state["selected_action"])

        return int(asyncio.run(_run_workflow()))

    def _increment_count(self, key: str, choice: int) -> None:
        counts = self._workflow_counts[key]
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
                        backlog_counts = self._workflow_counts["backlog"]
                        backlog_counts[selected_job_id] = backlog_counts.get(selected_job_id, 0) + 1
                return
            self._increment_count("plant", int(selected_action))
            return
        if role == "area":
            self._increment_count(agent_id, int(selected_action))


class GoogleADKLLMTreeDriver(GoogleADKTreeDriver):
    """Google ADK-backed driver that queries an OpenAI-compatible endpoint via stdlib HTTP."""

    controller_name = "external_framework_google_adk_llm"

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        request_timeout: float | None = None,
        loop_penalty: float = 2.0,
        disturbance_weight: float = 1.5,
        strict_framework: bool = False,
        controller_name: str | None = None,
        request_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self.llm_backend = GOOGLE_ADK_LLM_BACKEND
        self.llm_model = str(model_name)
        self.llm_base_url = str(base_url)
        self._llm_api_key = None if api_key is None else str(api_key)
        self.request_timeout = (
            None if request_timeout is None else float(request_timeout)
        )
        self.request_retries = max(int(request_retries), 0)
        self.retry_backoff_seconds = max(float(retry_backoff_seconds), 0.0)
        super().__init__(
            loop_penalty=loop_penalty,
            disturbance_weight=disturbance_weight,
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

    def _build_native_runtime(self) -> dict[str, object]:
        from google.adk.agents import SequentialAgent
        from google.adk.runners import InMemoryRunner

        runtime_tag = f"fms_bench_google_adk_runtime_{self._runtime_serial}"
        root_agent = SequentialAgent(
            name=f"{runtime_tag}_root",
            sub_agents=[
                _GoogleADKIngressAgent(name=f"{runtime_tag}_ingress"),
                _GoogleADKLLMPolicyAgent(
                    name=f"{runtime_tag}_policy",
                    driver=self,
                ),
            ],
        )
        return {
            "runner": InMemoryRunner(
                agent=root_agent,
                app_name=runtime_tag,
            ),
            "app_name": runtime_tag,
            "user_id": f"benchmark_{self._runtime_serial}",
            "runtime_tag": runtime_tag,
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

        selected_action = self._select_openai_compat_action(
            agent_id=agent_id,
            agent_view=agent_view,
            legal_actions=legal_actions,
        )
        if selected_action is None:
            selected_action = self._select_contract_action(
                agent_id=agent_id,
                agent_view=agent_view,
            )
            self._record_decision_source(
                agent_id=agent_id,
                source="fallback_contract",
                action=int(selected_action),
                legal_actions=legal_actions,
            )
            return int(selected_action)

        self._apply_action_side_effects(
            agent_id=agent_id,
            agent_view=agent_view,
            selected_action=int(selected_action),
        )
        self._record_decision_source(
            agent_id=agent_id,
            source="llm",
            action=int(selected_action),
            legal_actions=legal_actions,
        )
        return int(selected_action)

    def _select_openai_compat_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
        legal_actions: list[int],
    ) -> int | None:
        for attempt in range(self.request_retries + 1):
            try:
                content = self._request_openai_compat_completion(
                    messages=_build_llm_messages(
                        agent_id=agent_id,
                        agent_view=agent_view,
                    )
                )
            except Exception:
                content = None

            selected = None if content is None else _parse_llm_action_text(
                content,
                legal_actions=legal_actions,
            )
            if selected is not None:
                return int(selected)
            if attempt < self.request_retries and self.retry_backoff_seconds > 0.0:
                time.sleep(self.retry_backoff_seconds * float(attempt + 1))
        return None

    def _request_openai_compat_completion(
        self,
        *,
        messages: list[dict[str, str]],
    ) -> str | None:
        if not self.llm_model or not self.llm_base_url:
            return None

        payload = {
            "model": self.llm_model,
            "messages": messages,
            **build_openai_compat_benchmark_generation_kwargs(
                model_name=self.llm_model,
                temperature=0.0,
            ),
        }
        headers = {
            "Content-Type": "application/json",
        }
        if self._llm_api_key is not None:
            headers["Authorization"] = f"Bearer {self._llm_api_key}"

        request = urllib_request.Request(
            url=self._chat_completions_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        if self.request_timeout is None:
            with urllib_request.urlopen(request) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        else:
            with urllib_request.urlopen(request, timeout=self.request_timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        return self._extract_openai_compat_content(response_payload)

    def _chat_completions_url(self) -> str:
        return self.llm_base_url.rstrip("/") + "/chat/completions"

    @staticmethod
    def _extract_openai_compat_content(response: Mapping[str, object]) -> str | None:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = dict(choices[0]).get("message")
        if not isinstance(message, Mapping):
            return None
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, Mapping):
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            if text_parts:
                return "".join(text_parts)
        return None


def _register_google_adk_llm_driver() -> None:
    driver_cls = GoogleADKLLMTreeDriver
    for module_name in (
        "fms_bench.v2.framework_drivers",
        "fms_bench.v2",
        "fms_bench",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "GoogleADKLLMTreeDriver", driver_cls)


_register_google_adk_llm_driver()
