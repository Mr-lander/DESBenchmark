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
from .deterministic_kernel import (
    DeterministicCoordinationKernel,
    _visible_cell_summaries,
    _visible_machine_summaries,
)
from .llm_prompting import (
    build_openai_compat_benchmark_generation_kwargs,
    build_openai_compat_benchmark_messages,
)

SPADE_LLM_BACKEND = "openai_compat"


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


class _SPADEDecisionBehaviour:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from spade.behaviour import OneShotBehaviour

        class _DecisionBehaviour(OneShotBehaviour):
            def __init__(
                self,
                *,
                benchmark_agent_id: str,
                driver: "SPADETreeDriver",
            ) -> None:
                self._benchmark_agent_id = benchmark_agent_id
                self._driver = driver
                self.action: int | None = None
                super().__init__()

            async def run(self) -> None:
                msg = await self.receive(timeout=0.01)
                payload = json.loads(msg.body) if msg is not None and msg.body else {}
                self.action = int(
                    self._driver._select_contract_action(
                        agent_id=str(payload.get("agent_id", self._benchmark_agent_id)),
                        agent_view=dict(payload.get("agent_view", {})),
                    )
                )
                self.set("last_action", int(self.action))

        return _DecisionBehaviour(*args, **kwargs)


class _SPADELLMDecisionBehaviour:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from spade.behaviour import OneShotBehaviour

        class _DecisionBehaviour(OneShotBehaviour):
            def __init__(
                self,
                *,
                benchmark_agent_id: str,
                driver: Any,
            ) -> None:
                self._benchmark_agent_id = benchmark_agent_id
                self._driver = driver
                self.action: int | None = None
                super().__init__()

            async def run(self) -> None:
                msg = await self.receive(timeout=0.01)
                payload = json.loads(msg.body) if msg is not None and msg.body else {}
                self.action = int(
                    self._driver._select_llm_or_contract_action(
                        agent_id=str(payload.get("agent_id", self._benchmark_agent_id)),
                        agent_view=dict(payload.get("agent_view", {})),
                    )
                )
                self.set("last_action", int(self.action))

        return _DecisionBehaviour(*args, **kwargs)


class SPADETreeDriver(FrameworkBackedTreeDriver):
    """Framework-shaped driver that models message-native disturbance propagation."""

    framework_name = "SPADE"
    python_package = "spade"
    controller_name = "external_framework_spade"

    def __init__(
        self,
        *,
        memory_decay: float = 0.7,
        disturbance_weight: float = 2.5,
        routing_penalty: float = 2.0,
        strict_framework: bool = False,
        controller_name: str | None = None,
    ) -> None:
        self.memory_decay = float(memory_decay)
        self.disturbance_weight = float(disturbance_weight)
        self.routing_penalty = float(routing_penalty)
        self.kernel = DeterministicCoordinationKernel()
        self._runtime_serial = 0
        self._runtime_tag = "benchmark_0"
        super().__init__(
            strict_framework=strict_framework,
            controller_name=controller_name,
        )

    def reset_episode(self) -> None:
        super().reset_episode()
        self._backlog_selection_counts: dict[int, int] = {}
        self._workflow_counts: dict[str, dict[int, int]] = defaultdict(dict)
        self._mailbox_area_pressures: dict[int, float] = {}
        self._mailbox_cell_pressures: dict[int, dict[int, float]] = defaultdict(dict)
        self._mailbox_machine_pressures: dict[int, dict[int, float]] = defaultdict(dict)
        if self._framework_available:
            self._runtime_serial += 1
            self._runtime_tag = f"benchmark_{self._runtime_serial}"
            self._native_runtime = self._build_native_runtime()

    def _before_select_actions(self, decision: Mapping[str, object]) -> None:
        del decision
        self._decay_memory()

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
                self._apply_action_side_effects(
                    agent_id=agent_id,
                    agent_view=agent_view,
                    selected_action=int(selected),
                )
                return selected
            selected = self.kernel.select_area(
                agent_view,
                area_pressures=self._mailbox_area_pressures,
                selection_counts=self._workflow_counts["plant"],
                pressure_weight=self.disturbance_weight,
                routing_penalty=self.routing_penalty,
            )
            self._apply_action_side_effects(
                agent_id=agent_id,
                agent_view=agent_view,
                selected_action=int(selected),
            )
            return selected
        if role == "area":
            area_id = int(agent_id.split("_", 1)[1])
            selected = self.kernel.select_cell(
                agent_view,
                cell_pressures=self._mailbox_cell_pressures.get(area_id, {}),
                selection_counts=self._workflow_counts[agent_id],
                pressure_weight=self.disturbance_weight,
                routing_penalty=self.routing_penalty,
            )
            self._apply_action_side_effects(
                agent_id=agent_id,
                agent_view=agent_view,
                selected_action=int(selected),
            )
            return selected
        if role == "cell":
            cell_id = int(agent_view.get("cell_id", -1))
            selected = self.kernel.select_dispatch(
                agent_view,
                machine_pressures=self._mailbox_machine_pressures.get(cell_id, {}),
                disturbance_weight=self.disturbance_weight,
            )
            self._apply_action_side_effects(
                agent_id=agent_id,
                agent_view=agent_view,
                selected_action=int(selected),
            )
            return selected
        return self.kernel.select_dispatch(agent_view)

    def _build_native_runtime(self) -> dict[str, object]:
        return {}

    def _select_native_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> int:
        from spade.message import Message

        async def _run_behaviour() -> int:
            agent = self._ensure_native_agent(agent_id)
            behaviour = _SPADEDecisionBehaviour(
                benchmark_agent_id=agent_id,
                driver=self,
            )
            behaviour.set_agent(agent)
            await behaviour.enqueue(
                Message(
                    sender=f"{self._runtime_tag}@localhost",
                    to=f"{agent_id}@{self._runtime_tag}.localhost",
                    body=_encode_framework_payload(
                        {
                            "agent_id": agent_id,
                            "agent_view": dict(agent_view),
                            "framework_name": self.framework_name,
                        }
                    ),
                )
            )
            await behaviour.run()
            if behaviour.action is None:
                raise RuntimeError(f"SPADE native decision behaviour produced no action for {agent_id}")
            return int(behaviour.action)

        return int(asyncio.run(_run_behaviour()))

    def _ensure_native_agent(self, agent_id: str) -> object:
        from spade.agent import Agent

        assert isinstance(self._native_runtime, dict)
        if agent_id not in self._native_runtime:
            self._native_runtime[agent_id] = Agent(
                jid=f"{agent_id}@{self._runtime_tag}.localhost",
                password="benchmark",
                verify_security=False,
            )
        return self._native_runtime[agent_id]

    def _update_area_memory(
        self,
        area_id: int,
        cell_summaries: Mapping[int, Mapping[str, object]],
    ) -> None:
        if not cell_summaries:
            return
        area_pressure, per_cell = self.kernel.area_pressures_from_cell_summaries(cell_summaries)
        self._mailbox_area_pressures[area_id] = max(
            float(self._mailbox_area_pressures.get(area_id, 0.0)),
            area_pressure,
        )
        memory = self._mailbox_cell_pressures[area_id]
        for cell_id, pressure in per_cell.items():
            memory[int(cell_id)] = max(float(memory.get(int(cell_id), 0.0)), float(pressure))

    def _update_machine_memory(
        self,
        cell_id: int,
        machine_summaries: Mapping[int, Mapping[str, object]],
    ) -> None:
        if cell_id < 0:
            return
        memory = self._mailbox_machine_pressures[cell_id]
        for machine_id, pressure in self.kernel.machine_pressures_from_machine_summaries(machine_summaries).items():
            memory[int(machine_id)] = max(float(memory.get(int(machine_id), 0.0)), float(pressure))

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
                        self._backlog_selection_counts[selected_job_id] = (
                            self._backlog_selection_counts.get(selected_job_id, 0) + 1
                        )
                return
            self._increment_count("plant", int(selected_action))
            return
        if role == "area":
            self._increment_count(agent_id, int(selected_action))
            area_id = int(agent_id.split("_", 1)[1])
            self._update_area_memory(area_id, _visible_cell_summaries(agent_view))
            return
        if role == "cell":
            cell_id = int(agent_view.get("cell_id", -1))
            self._update_machine_memory(cell_id, _visible_machine_summaries(agent_view))

    def _decay_memory(self) -> None:
        self._mailbox_area_pressures = {
            int(area_id): float(value) * self.memory_decay
            for area_id, value in self._mailbox_area_pressures.items()
            if float(value) * self.memory_decay > 1e-9
        }
        self._mailbox_cell_pressures = defaultdict(
            dict,
            {
                int(area_id): {
                    int(cell_id): float(value) * self.memory_decay
                    for cell_id, value in cell_memory.items()
                    if float(value) * self.memory_decay > 1e-9
                }
                for area_id, cell_memory in self._mailbox_cell_pressures.items()
                if any(float(value) * self.memory_decay > 1e-9 for value in cell_memory.values())
            },
        )
        self._mailbox_machine_pressures = defaultdict(
            dict,
            {
                int(cell_id): {
                    int(machine_id): float(value) * self.memory_decay
                    for machine_id, value in machine_memory.items()
                    if float(value) * self.memory_decay > 1e-9
                }
                for cell_id, machine_memory in self._mailbox_machine_pressures.items()
                if any(float(value) * self.memory_decay > 1e-9 for value in machine_memory.values())
            },
        )


class SPADELLMTreeDriver(SPADETreeDriver):
    """SPADE-backed driver that queries an OpenAI-compatible endpoint via stdlib HTTP."""

    controller_name = "external_framework_spade_llm"

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        request_timeout: float | None = None,
        memory_decay: float = 0.7,
        disturbance_weight: float = 2.5,
        strict_framework: bool = False,
        controller_name: str | None = None,
        request_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self.llm_backend = SPADE_LLM_BACKEND
        self.llm_model = str(model_name)
        self.llm_base_url = str(base_url)
        self._llm_api_key = None if api_key is None else str(api_key)
        self.request_timeout = (
            None if request_timeout is None else float(request_timeout)
        )
        self.request_retries = max(int(request_retries), 0)
        self.retry_backoff_seconds = max(float(retry_backoff_seconds), 0.0)
        super().__init__(
            memory_decay=memory_decay,
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

    def _select_native_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> int:
        from spade.message import Message

        async def _run_behaviour() -> int:
            agent = self._ensure_native_agent(agent_id)
            behaviour = _SPADELLMDecisionBehaviour(
                benchmark_agent_id=agent_id,
                driver=self,
            )
            behaviour.set_agent(agent)
            await behaviour.enqueue(
                Message(
                    sender=f"{self._runtime_tag}@localhost",
                    to=f"{agent_id}@{self._runtime_tag}.localhost",
                    body=_encode_framework_payload(
                        {
                            "agent_id": agent_id,
                            "agent_view": dict(agent_view),
                            "framework_name": self.framework_name,
                        }
                    ),
                )
            )
            await behaviour.run()
            if behaviour.action is None:
                raise RuntimeError(f"SPADE native decision behaviour produced no action for {agent_id}")
            return int(behaviour.action)

        return int(asyncio.run(_run_behaviour()))

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


def _register_spade_llm_driver() -> None:
    driver_cls = SPADELLMTreeDriver
    for module_name in (
        "fms_bench.v2.framework_drivers",
        "fms_bench.v2",
        "fms_bench",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "SPADELLMTreeDriver", driver_cls)


_register_spade_llm_driver()
