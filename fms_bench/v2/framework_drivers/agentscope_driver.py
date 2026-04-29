from __future__ import annotations

import asyncio
from collections import defaultdict
import inspect
import json
import sys
from typing import Any, Mapping
from urllib import request as urllib_request

from .base import FrameworkBackedTreeDriver
from .deterministic_kernel import DeterministicCoordinationKernel
from .llm_prompting import (
    build_openai_compat_benchmark_generation_kwargs,
    build_openai_compat_benchmark_messages,
    parse_openai_compat_benchmark_action_text,
)


AGENTSCOPE_LLM_BACKEND = "openai_compat"


def _legal_action_indices(agent_view: Mapping[str, object]) -> list[int]:
    mask = list(agent_view.get("action_mask", []))
    return [
        index
        for index, enabled in enumerate(mask)
        if bool(enabled)
    ]


def _extract_chat_text(response: object) -> str:
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, Mapping):
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def _parse_llm_action_text(
    text: str,
    *,
    legal_actions: list[int],
) -> int | None:
    if not legal_actions:
        return 0
    parsed = parse_openai_compat_benchmark_action_text(text)
    if parsed is None or int(parsed) not in legal_actions:
        return None
    return int(parsed)


def _build_llm_messages(
    *,
    agent_id: str,
    agent_view: Mapping[str, object],
) -> list[dict[str, str]]:
    return build_openai_compat_benchmark_messages(
        agent_id=agent_id,
        agent_view=agent_view,
    )


class _AgentScopeRelayAgent:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from agentscope.agent import AgentBase

        class _RelayAgent(AgentBase):
            def __init__(self, benchmark_agent_id: str) -> None:
                self.name = f"{benchmark_agent_id}_relay"
                self._observed_messages: list[object] = []
                super().__init__()
                self.set_console_output_enabled(False)

            async def observe(self, msg: object) -> None:
                if isinstance(msg, list):
                    self._observed_messages.extend(msg)
                elif msg is not None:
                    self._observed_messages.append(msg)

            async def reply(self, msg: object | None = None) -> object:
                from agentscope.message import Msg

                incoming = msg[-1] if isinstance(msg, list) and msg else msg
                metadata = dict(getattr(incoming, "metadata", {}) or {})
                metadata["relay_observation_count"] = len(self._observed_messages)
                return Msg(
                    name=self.name,
                    content=f"relay::{metadata.get('agent_id', 'unknown')}",
                    role="assistant",
                    metadata=metadata,
                )

            async def handle_interrupt(self, *args: Any, **kwargs: Any) -> object:
                from agentscope.message import Msg

                del args, kwargs
                return Msg(
                    name=self.name,
                    content="relay_interrupt",
                    role="assistant",
                )

        return _RelayAgent(*args, **kwargs)


class _AgentScopeDecisionAgent:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from agentscope.agent import AgentBase

        class _DecisionAgent(AgentBase):
            def __init__(
                self,
                *,
                benchmark_agent_id: str,
                driver: "AgentScopeTreeDriver",
            ) -> None:
                self.name = f"{benchmark_agent_id}_decider"
                self._benchmark_agent_id = benchmark_agent_id
                self._driver = driver
                self._observed_messages: list[object] = []
                super().__init__()
                self.set_console_output_enabled(False)

            async def observe(self, msg: object) -> None:
                if isinstance(msg, list):
                    self._observed_messages.extend(msg)
                elif msg is not None:
                    self._observed_messages.append(msg)

            async def reply(self, msg: object | None = None) -> object:
                from agentscope.message import Msg

                incoming = msg[-1] if isinstance(msg, list) and msg else msg
                metadata = dict(getattr(incoming, "metadata", {}) or {})
                action = int(
                    self._driver._select_contract_action(
                        agent_id=str(metadata.get("agent_id", self._benchmark_agent_id)),
                        agent_view=dict(metadata.get("agent_view", {})),
                    )
                )
                metadata["action"] = action
                metadata["decision_observation_count"] = len(self._observed_messages)
                return Msg(
                    name=self.name,
                    content=str(action),
                    role="assistant",
                    metadata=metadata,
                )

            async def handle_interrupt(self, *args: Any, **kwargs: Any) -> object:
                from agentscope.message import Msg

                del args, kwargs
                return Msg(
                    name=self.name,
                    content="decision_interrupt",
                    role="assistant",
                )

        return _DecisionAgent(*args, **kwargs)


class _AgentScopeLLMDecisionAgent:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from agentscope.agent import AgentBase

        class _LLMDecisionAgent(AgentBase):
            def __init__(
                self,
                *,
                benchmark_agent_id: str,
                driver: "AgentScopeLLMTreeDriver",
            ) -> None:
                self.name = f"{benchmark_agent_id}_llm_decider"
                self._benchmark_agent_id = benchmark_agent_id
                self._driver = driver
                self._observed_messages: list[object] = []
                super().__init__()
                self.set_console_output_enabled(False)

            async def observe(self, msg: object) -> None:
                if isinstance(msg, list):
                    self._observed_messages.extend(msg)
                elif msg is not None:
                    self._observed_messages.append(msg)

            async def reply(self, msg: object | None = None) -> object:
                from agentscope.message import Msg

                incoming = msg[-1] if isinstance(msg, list) and msg else msg
                metadata = dict(getattr(incoming, "metadata", {}) or {})
                agent_id = str(metadata.get("agent_id", self._benchmark_agent_id))
                agent_view = dict(metadata.get("agent_view", {}))
                action = int(
                    await self._driver._select_llm_action_async(
                        agent_id=agent_id,
                        agent_view=agent_view,
                    )
                )
                metadata["action"] = action
                metadata["decision_observation_count"] = len(self._observed_messages)
                metadata["llm_backend"] = AGENTSCOPE_LLM_BACKEND
                return Msg(
                    name=self.name,
                    content=json.dumps({"action": int(action)}, ensure_ascii=True),
                    role="assistant",
                    metadata=metadata,
                )

            async def handle_interrupt(self, *args: Any, **kwargs: Any) -> object:
                from agentscope.message import Msg

                del args, kwargs
                return Msg(
                    name=self.name,
                    content="llm_decision_interrupt",
                    role="assistant",
                )

        return _LLMDecisionAgent(*args, **kwargs)


class AgentScopeTreeDriver(FrameworkBackedTreeDriver):
    """Framework-shaped driver that models routing and handoff memory."""

    framework_name = "AgentScope"
    python_package = "agentscope"
    controller_name = "external_framework_agentscope"

    def __init__(
        self,
        *,
        routing_penalty: float = 2.0,
        strict_framework: bool = False,
        controller_name: str | None = None,
    ) -> None:
        self.routing_penalty = float(routing_penalty)
        self.kernel = DeterministicCoordinationKernel()
        super().__init__(
            strict_framework=strict_framework,
            controller_name=controller_name,
        )
        if self._framework_available:
            self._native_runtime = self._build_native_runtime()

    def reset_episode(self) -> None:
        super().reset_episode()
        self._handoff_counts: dict[str, dict[int, int]] = defaultdict(dict)
        self._handoff_counts["backlog"] = {}
        if self._framework_available:
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
                    selection_counts=self._handoff_counts["backlog"],
                )
                visible_candidates = list(agent_view.get("visible_backlog_candidates", ()))
                if 0 <= int(selected) < len(visible_candidates):
                    selected_job_id = int(dict(visible_candidates[int(selected)]).get("job_id", -1))
                    if selected_job_id >= 0:
                        self._handoff_counts["backlog"][selected_job_id] = (
                        self._handoff_counts["backlog"].get(selected_job_id, 0) + 1
                        )
                return selected
            if plant_decision_kind == "cell_selection":
                selected = self.kernel.select_cell(
                    agent_view,
                    selection_counts=self._handoff_counts["plant"],
                    routing_penalty=self.routing_penalty,
                )
                self._handoff_counts["plant"][selected] = (
                    self._handoff_counts["plant"].get(selected, 0) + 1
                )
                return selected
            selected = self.kernel.select_area(
                agent_view,
                selection_counts=self._handoff_counts["plant"],
                routing_penalty=self.routing_penalty,
            )
            self._handoff_counts["plant"][selected] = (
                self._handoff_counts["plant"].get(selected, 0) + 1
            )
            return selected
        if role == "area":
            selected = self.kernel.select_cell(
                agent_view,
                selection_counts=self._handoff_counts[agent_id],
                routing_penalty=self.routing_penalty,
            )
            self._handoff_counts[agent_id][selected] = (
                self._handoff_counts[agent_id].get(selected, 0) + 1
            )
            return selected
        return self.kernel.select_dispatch(agent_view)

    def _build_native_runtime(self) -> dict[str, dict[str, object]]:
        return {}

    def _select_native_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> int:
        from agentscope.message import Msg
        from agentscope.pipeline import MsgHub

        pipeline_bundle = self._ensure_native_pipeline(agent_id)
        seed_msg = Msg(
            name=agent_id,
            content=f"agent_view::{agent_id}",
            role="user",
            metadata={
                "agent_id": agent_id,
                "agent_view": dict(agent_view),
                "framework_name": self.framework_name,
            },
        )

        async def _run_pipeline() -> object:
            participants = [
                pipeline_bundle["relay"],
                pipeline_bundle["decider"],
            ]
            async with MsgHub(
                participants=participants,
                announcement=seed_msg,
                enable_auto_broadcast=True,
                name=f"agentscope::{agent_id}",
            ):
                return await pipeline_bundle["pipeline"](seed_msg)

        result = asyncio.run(_run_pipeline())
        return int(result.metadata["action"])

    def _ensure_native_pipeline(self, agent_id: str) -> dict[str, object]:
        from agentscope.pipeline import SequentialPipeline

        assert isinstance(self._native_runtime, dict)
        if agent_id not in self._native_runtime:
            relay = _AgentScopeRelayAgent(agent_id)
            decider = _AgentScopeDecisionAgent(
                benchmark_agent_id=agent_id,
                driver=self,
            )
            self._native_runtime[agent_id] = {
                "relay": relay,
                "decider": decider,
                "pipeline": SequentialPipeline([relay, decider]),
            }
        return self._native_runtime[agent_id]


class AgentScopeLLMTreeDriver(AgentScopeTreeDriver):
    """AgentScope-backed driver that queries an OpenAI-compatible chat backend."""

    controller_name = "external_framework_agentscope_llm"

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        routing_penalty: float = 2.0,
        temperature: float = 0.0,
        strict_framework: bool = False,
        controller_name: str | None = None,
        request_timeout: float | None = None,
        request_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        retry_backoff_max_seconds: float = 60.0,
    ) -> None:
        self.llm_backend = AGENTSCOPE_LLM_BACKEND
        self.model_name = str(model_name)
        self.base_url = None if base_url is None else str(base_url)
        self.api_key = None if api_key is None else str(api_key)
        self.temperature = float(temperature)
        self.request_timeout = (
            None if request_timeout is None else float(request_timeout)
        )
        self.request_retries = max(int(request_retries), 0)
        self.retry_backoff_seconds = max(float(retry_backoff_seconds), 0.0)
        self.retry_backoff_max_seconds = max(float(retry_backoff_max_seconds), 0.0)
        self._llm_model: Any | None = None
        super().__init__(
            routing_penalty=routing_penalty,
            strict_framework=strict_framework,
            controller_name=controller_name or self.controller_name,
        )

    @property
    def framework_runtime_info(self) -> dict[str, object]:
        payload = dict(super().framework_runtime_info)
        payload.update(
            {
                "llm_backend": self.llm_backend,
                "llm_model": self.model_name,
                "llm_base_url": self.base_url,
            }
        )
        return payload

    def _select_framework_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> int:
        return self._select_llm_action(
            agent_id=agent_id,
            agent_view=agent_view,
        )

    def reset_episode(self) -> None:
        super().reset_episode()
        self._llm_model = None

    def _ensure_native_pipeline(self, agent_id: str) -> dict[str, object]:
        from agentscope.pipeline import SequentialPipeline

        assert isinstance(self._native_runtime, dict)
        if agent_id not in self._native_runtime:
            relay = _AgentScopeRelayAgent(agent_id)
            decider = _AgentScopeLLMDecisionAgent(
                benchmark_agent_id=agent_id,
                driver=self,
            )
            self._native_runtime[agent_id] = {
                "relay": relay,
                "decider": decider,
                "pipeline": SequentialPipeline([relay, decider]),
            }
        return self._native_runtime[agent_id]

    def _select_llm_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> int:
        return asyncio.run(
            self._select_llm_action_async(
                agent_id=agent_id,
                agent_view=agent_view,
            )
        )

    async def _select_llm_action_async(
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

        if not self._llm_configured():
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

        last_failure: dict[str, object] | None = None
        for attempt in range(self.request_retries + 1):
            selected: int | None = None
            current_failure: dict[str, object] | None = None
            try:
                response_text = await self._request_openai_compat_completion_async(
                    messages=_build_llm_messages(
                        agent_id=agent_id,
                        agent_view=agent_view,
                    )
                )
                if not response_text:
                    current_failure = {"failure_kind": "empty_response"}
                else:
                    selected = _parse_llm_action_text(
                        response_text,
                        legal_actions=legal_actions,
                    )
                    if selected is None:
                        current_failure = {
                            "failure_kind": "invalid_response",
                            "raw_response": response_text,
                        }
                if selected is not None:
                    self._record_decision_source(
                        agent_id=agent_id,
                        source="llm",
                        action=int(selected),
                        legal_actions=legal_actions,
                    )
                    return int(selected)
            except Exception as exc:
                current_failure = {
                    "failure_kind": "exception",
                    "exception": exc,
                }
            last_failure = current_failure
            delay = self._retry_backoff_delay_seconds(attempt)
            if attempt < self.request_retries and delay > 0.0:
                await asyncio.sleep(delay)

        if last_failure is not None:
            self._record_decision_failure(
                agent_id=agent_id,
                attempt=self.request_retries,
                failure_kind=str(last_failure["failure_kind"]),
                legal_actions=legal_actions,
                raw_response=last_failure.get("raw_response"),
                exception=(
                    last_failure.get("exception")
                    if isinstance(last_failure.get("exception"), BaseException)
                    else None
                ),
            )

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

    def _retry_backoff_delay_seconds(self, attempt: int) -> float:
        if self.retry_backoff_seconds <= 0.0:
            return 0.0
        delay = self.retry_backoff_seconds * float(2 ** max(int(attempt), 0))
        if self.retry_backoff_max_seconds > 0.0:
            delay = min(delay, self.retry_backoff_max_seconds)
        return delay

    async def _request_openai_compat_completion_async(
        self,
        *,
        messages: list[dict[str, str]],
    ) -> str | None:
        return await asyncio.to_thread(
            self._request_openai_compat_completion,
            messages=messages,
        )

    def _request_openai_compat_completion(
        self,
        *,
        messages: list[dict[str, str]],
    ) -> str | None:
        if not self.model_name or not self.base_url:
            return None

        payload = {
            "model": self.model_name,
            "messages": messages,
            **build_openai_compat_benchmark_generation_kwargs(
                model_name=self.model_name,
                temperature=self.temperature,
            ),
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        }
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib_request.Request(
            url=self._chat_completions_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        timeout = self.request_timeout
        if timeout is None or timeout <= 0.0:
            with urllib_request.urlopen(request) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        else:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        return self._extract_openai_compat_content(response_payload)

    def _chat_completions_url(self) -> str:
        assert self.base_url is not None
        return self.base_url.rstrip("/") + "/chat/completions"

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

    async def _invoke_llm_model(
        self,
        *,
        llm_model: Any,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> object:
        async def _resolve_response() -> object:
            response = llm_model(
                _build_llm_messages(
                    agent_id=agent_id,
                    agent_view=agent_view,
                )
            )
            if inspect.isawaitable(response):
                return await response
            return response

        if self.request_timeout is None or self.request_timeout <= 0.0:
            return await _resolve_response()
        return await asyncio.wait_for(
            _resolve_response(),
            timeout=self.request_timeout,
        )

    def _llm_configured(self) -> bool:
        return bool(self.model_name) and bool(self.base_url)

    def _build_llm_model(self) -> Any:
        from agentscope.model import OpenAIChatModel

        client_kwargs: dict[str, object] = {}
        if self.base_url is not None:
            client_kwargs["base_url"] = self.base_url
        if self.request_timeout is not None:
            client_kwargs["timeout"] = self.request_timeout
        return OpenAIChatModel(
            model_name=self.model_name,
            api_key=self.api_key,
            stream=False,
            client_kwargs=client_kwargs or None,
            generate_kwargs=build_openai_compat_benchmark_generation_kwargs(
                model_name=self.model_name,
                temperature=self.temperature,
            ),
        )


def _register_agentscope_llm_driver() -> None:
    driver_cls = AgentScopeLLMTreeDriver
    for module_name in (
        "fms_bench.v2.framework_drivers",
        "fms_bench.v2",
        "fms_bench",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "AgentScopeLLMTreeDriver", driver_cls)


_register_agentscope_llm_driver()
