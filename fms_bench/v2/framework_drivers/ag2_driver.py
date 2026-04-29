from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from importlib.util import find_spec
import re
import sys
import time
from typing import Any
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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


AG2_LLM_BACKEND = "openai_compat"


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


class AG2TreeDriver(FrameworkBackedTreeDriver):
    """Framework-shaped driver that models handoffs with shared context memory."""

    framework_name = "AG2"
    python_package = "ag2"
    controller_name = "external_framework_ag2"

    def __init__(
        self,
        *,
        memory_decay: float = 0.7,
        disturbance_weight: float = 2.5,
        handoff_penalty: float = 1.5,
        strict_framework: bool = False,
        controller_name: str | None = None,
    ) -> None:
        self.memory_decay = float(memory_decay)
        self.disturbance_weight = float(disturbance_weight)
        self.handoff_penalty = float(handoff_penalty)
        self.kernel = DeterministicCoordinationKernel()
        self._runtime_serial = 0
        super().__init__(
            strict_framework=strict_framework,
            controller_name=controller_name,
        )

    def _detect_framework_availability(self) -> bool:
        return find_spec("ag2") is not None or find_spec("autogen") is not None

    def reset_episode(self) -> None:
        super().reset_episode()
        self._backlog_selection_counts: dict[int, int] = {}
        self._area_pressures: dict[int, float] = {}
        self._cell_pressures: dict[int, dict[int, float]] = defaultdict(dict)
        self._machine_pressures: dict[int, dict[int, float]] = defaultdict(dict)
        self._handoff_counts: dict[str, dict[int, int]] = defaultdict(dict)
        if self._framework_available:
            self._runtime_serial += 1
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
            else:
                selected = self.kernel.select_area(
                    agent_view,
                    area_pressures=self._area_pressures,
                    selection_counts=self._handoff_counts["plant"],
                    pressure_weight=self.disturbance_weight,
                    routing_penalty=self.handoff_penalty,
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
                cell_pressures=self._cell_pressures.get(area_id, {}),
                selection_counts=self._handoff_counts[agent_id],
                pressure_weight=self.disturbance_weight,
                routing_penalty=self.handoff_penalty,
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
                machine_pressures=self._machine_pressures.get(cell_id, {}),
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
        runtime_tag = f"fms_bench_ag2_runtime_{self._runtime_serial}"
        return {
            "runtime_tag": runtime_tag,
            "agents": {},
        }

    def _select_native_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> int:
        bundle = self._ensure_native_agents(agent_id)
        payload = {
            "agent_id": agent_id,
            "agent_view": dict(agent_view),
            "framework_name": self.framework_name,
        }
        bundle["latest_payload"] = payload
        reply = bundle["decider"].generate_reply(
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(payload, sort_keys=True),
                    "name": str(bundle["relay"].name),
                }
            ],
            sender=bundle["relay"],
        )
        if isinstance(reply, Mapping):
            if "action" in reply:
                return int(reply["action"])
            if "content" in reply:
                return self._parse_native_action(reply["content"])
        return self._parse_native_action(reply)

    def _ensure_native_agents(self, agent_id: str) -> dict[str, object]:
        try:
            from autogen import ConversableAgent, UserProxyAgent
        except ImportError:
            from ag2 import ConversableAgent, UserProxyAgent  # type: ignore[no-redef]

        assert isinstance(self._native_runtime, dict)
        agents = self._native_runtime.setdefault("agents", {})
        if agent_id in agents:
            return agents[agent_id]

        runtime_tag = str(self._native_runtime["runtime_tag"])
        relay = UserProxyAgent(
            name=f"{runtime_tag}_{agent_id}_relay",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        decider = ConversableAgent(
            name=f"{runtime_tag}_{agent_id}_decider",
            human_input_mode="NEVER",
            code_execution_config=False,
            llm_config=False,
            default_auto_reply="",
        )
        bundle: dict[str, object] = {
            "relay": relay,
            "decider": decider,
            "latest_payload": {},
        }

        def _reply(
            recipient: object,
            messages: object | None = None,
            sender: object | None = None,
            config: object | None = None,
        ) -> tuple[bool, str]:
            del recipient, messages, sender, config
            payload = dict(bundle.get("latest_payload", {}))
            action = self._select_contract_action(
                agent_id=str(payload.get("agent_id", agent_id)),
                agent_view=dict(payload.get("agent_view", {})),
            )
            return True, json.dumps({"action": int(action)})

        decider.register_reply(
            [type(relay), None],
            _reply,
            position=0,
            remove_other_reply_funcs=True,
        )
        agents[agent_id] = bundle
        return bundle

    @staticmethod
    def _parse_native_action(reply: object) -> int:
        if isinstance(reply, Mapping):
            if "action" in reply:
                return int(reply["action"])
            if "content" in reply:
                return AG2TreeDriver._parse_native_action(reply["content"])
        if isinstance(reply, str):
            try:
                payload = json.loads(reply)
            except json.JSONDecodeError:
                return int(reply)
            if isinstance(payload, Mapping) and "action" in payload:
                return int(payload["action"])
        return int(reply)

    def _increment_count(self, key: str, choice: int) -> None:
        counts = self._handoff_counts[key]
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
            area_id = int(agent_id.split("_", 1)[1])
            self._update_area_memory(area_id, _visible_cell_summaries(agent_view))
            self._increment_count(agent_id, int(selected_action))
            return
        if role == "cell":
            cell_id = int(agent_view.get("cell_id", -1))
            self._update_machine_memory(cell_id, _visible_machine_summaries(agent_view))

    def _update_area_memory(
        self,
        area_id: int,
        cell_summaries: Mapping[int, Mapping[str, object]],
    ) -> None:
        if not cell_summaries:
            return
        area_pressure, per_cell = self.kernel.area_pressures_from_cell_summaries(cell_summaries)
        self._area_pressures[area_id] = max(float(self._area_pressures.get(area_id, 0.0)), area_pressure)
        memory = self._cell_pressures[area_id]
        for cell_id, pressure in per_cell.items():
            memory[int(cell_id)] = max(float(memory.get(int(cell_id), 0.0)), float(pressure))

    def _update_machine_memory(
        self,
        cell_id: int,
        machine_summaries: Mapping[int, Mapping[str, object]],
    ) -> None:
        if cell_id < 0:
            return
        memory = self._machine_pressures[cell_id]
        for machine_id, pressure in self.kernel.machine_pressures_from_machine_summaries(machine_summaries).items():
            memory[int(machine_id)] = max(float(memory.get(int(machine_id), 0.0)), float(pressure))

    def _decay_memory(self) -> None:
        self._area_pressures = {
            int(area_id): float(value) * self.memory_decay
            for area_id, value in self._area_pressures.items()
            if float(value) * self.memory_decay > 1e-9
        }
        self._cell_pressures = defaultdict(
            dict,
            {
                int(area_id): {
                    int(cell_id): float(value) * self.memory_decay
                    for cell_id, value in cell_memory.items()
                    if float(value) * self.memory_decay > 1e-9
                }
                for area_id, cell_memory in self._cell_pressures.items()
                if any(float(value) * self.memory_decay > 1e-9 for value in cell_memory.values())
            },
        )
        self._machine_pressures = defaultdict(
            dict,
            {
                int(cell_id): {
                    int(machine_id): float(value) * self.memory_decay
                    for machine_id, value in machine_memory.items()
                    if float(value) * self.memory_decay > 1e-9
                }
                for cell_id, machine_memory in self._machine_pressures.items()
                if any(float(value) * self.memory_decay > 1e-9 for value in machine_memory.values())
            },
        )


class AG2LLMTreeDriver(AG2TreeDriver):
    """AG2-shaped driver that queries an OpenAI-compatible endpoint without extra deps."""

    controller_name = "external_framework_ag2_llm"

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        request_timeout: float | None = None,
        memory_decay: float = 0.7,
        disturbance_weight: float = 2.5,
        handoff_penalty: float = 1.5,
        strict_framework: bool = False,
        controller_name: str | None = None,
        request_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self.llm_backend = AG2_LLM_BACKEND
        self.model_name = str(model_name)
        self.llm_model = self.model_name
        self.base_url = str(base_url)
        self.llm_base_url = self.base_url
        self.api_key = None if api_key is None else str(api_key)
        self.request_timeout = None if request_timeout is None else float(request_timeout)
        self.request_retries = max(int(request_retries), 0)
        self.retry_backoff_seconds = max(float(retry_backoff_seconds), 0.0)
        super().__init__(
            memory_decay=memory_decay,
            disturbance_weight=disturbance_weight,
            handoff_penalty=handoff_penalty,
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
                **build_openai_compat_benchmark_generation_kwargs(
                    model_name=self.model_name,
                    temperature=0.0,
                ),
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

    def _ensure_native_agents(self, agent_id: str) -> dict[str, object]:
        try:
            from autogen import ConversableAgent, UserProxyAgent
        except ImportError:
            from ag2 import ConversableAgent, UserProxyAgent  # type: ignore[no-redef]

        assert isinstance(self._native_runtime, dict)
        agents = self._native_runtime.setdefault("agents", {})
        if agent_id in agents:
            return agents[agent_id]

        runtime_tag = str(self._native_runtime["runtime_tag"])
        relay = UserProxyAgent(
            name=f"{runtime_tag}_{agent_id}_relay",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        decider = ConversableAgent(
            name=f"{runtime_tag}_{agent_id}_llm_decider",
            human_input_mode="NEVER",
            code_execution_config=False,
            llm_config=False,
            default_auto_reply="",
        )
        bundle: dict[str, object] = {
            "relay": relay,
            "decider": decider,
            "latest_payload": {},
        }

        def _reply(
            recipient: object,
            messages: object | None = None,
            sender: object | None = None,
            config: object | None = None,
        ) -> tuple[bool, str]:
            del recipient, messages, sender, config
            payload = dict(bundle.get("latest_payload", {}))
            action = self._select_llm_or_contract_action(
                agent_id=str(payload.get("agent_id", agent_id)),
                agent_view=dict(payload.get("agent_view", {})),
            )
            return True, json.dumps({"action": int(action)})

        decider.register_reply(
            [type(relay), None],
            _reply,
            position=0,
            remove_other_reply_funcs=True,
        )
        agents[agent_id] = bundle
        return bundle


def _register_ag2_llm_driver() -> None:
    current_module = sys.modules[__name__]
    for module_name in (
        "fms_bench.v2.framework_drivers",
        "fms_bench.v2",
        "fms_bench",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        setattr(module, "AG2LLMTreeDriver", AG2LLMTreeDriver)
        exported = getattr(module, "__all__", None)
        if isinstance(exported, list) and "AG2LLMTreeDriver" not in exported:
            exported.append("AG2LLMTreeDriver")
    setattr(current_module, "AG2LLMTreeDriver", AG2LLMTreeDriver)


_register_ag2_llm_driver()
