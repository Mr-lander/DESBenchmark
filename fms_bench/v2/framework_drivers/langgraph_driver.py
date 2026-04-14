from __future__ import annotations

from collections.abc import Mapping
from collections import defaultdict
import json
import sys
import time
from typing import Any, TypedDict

from .base import FrameworkBackedTreeDriver
from .deterministic_kernel import (
    DeterministicCoordinationKernel,
    _visible_cell_summaries,
    _visible_machine_summaries,
)
from .llm_prompting import (
    OPENAI_COMPAT_BENCHMARK_MAX_TOKENS,
    build_openai_compat_benchmark_messages,
    build_openai_compat_benchmark_user_prompt,
    parse_openai_compat_benchmark_action_text,
)


class _LangGraphDecisionState(TypedDict, total=False):
    agent_id: str
    agent_view: dict[str, object]
    action: int


def _legal_action_indices(agent_view: Mapping[str, object]) -> list[int]:
    mask = list(agent_view.get("action_mask", []))
    return [index for index, enabled in enumerate(mask) if bool(enabled)]


def _build_llm_messages(
    *,
    agent_id: str,
    agent_view: Mapping[str, object],
) -> list[dict[str, str]]:
    return build_openai_compat_benchmark_messages(
        agent_id=agent_id,
        agent_view=agent_view,
    )


class LangGraphTreeDriver(FrameworkBackedTreeDriver):
    """Framework-shaped driver that models graph-state memory propagation."""

    framework_name = "LangGraph"
    python_package = "langgraph"
    controller_name = "external_framework_langgraph"

    def __init__(
        self,
        *,
        memory_decay: float = 0.7,
        disturbance_weight: float = 2.5,
        strict_framework: bool = False,
        controller_name: str | None = None,
    ) -> None:
        self.memory_decay = float(memory_decay)
        self.disturbance_weight = float(disturbance_weight)
        self.kernel = DeterministicCoordinationKernel()
        super().__init__(
            strict_framework=strict_framework,
            controller_name=controller_name,
        )
        self._compiled_graph = self._build_native_graph() if self._framework_available else None
        self._native_runtime = self._compiled_graph

    def reset_episode(self) -> None:
        super().reset_episode()
        self._backlog_selection_counts: dict[int, int] = {}
        self._area_pressures: dict[int, float] = {}
        self._cell_pressures: dict[int, dict[int, float]] = defaultdict(dict)
        self._machine_pressures: dict[int, dict[int, float]] = defaultdict(dict)
        if self._framework_available:
            compiled_graph = getattr(self, "_compiled_graph", None)
            if compiled_graph is None:
                compiled_graph = self._build_native_graph()
                self._compiled_graph = compiled_graph
            self._native_runtime = compiled_graph

    def _before_select_actions(self, decision: Mapping[str, object]) -> None:
        del decision
        self._decay_memory()

    def _select_framework_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> int:
        if self._compiled_graph is not None:
            result = self._compiled_graph.invoke(
                {
                    "agent_id": str(agent_id),
                    "agent_view": dict(agent_view),
                }
            )
            return int(result["action"])
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
                visible_candidates = list(agent_view.get("visible_backlog_candidates", ()))
                if 0 <= int(selected) < len(visible_candidates):
                    selected_job_id = int(dict(visible_candidates[int(selected)]).get("job_id", -1))
                    if selected_job_id >= 0:
                        self._backlog_selection_counts[selected_job_id] = (
                            self._backlog_selection_counts.get(selected_job_id, 0) + 1
                        )
                return selected
            if plant_decision_kind == "cell_selection":
                selected = self.kernel.select_cell(
                    agent_view,
                    cell_pressures=self._cell_pressures.get(-1, {}),
                    pressure_weight=self.disturbance_weight,
                )
                self._update_plant_cell_memory(_visible_cell_summaries(agent_view))
                return selected
            return self.kernel.select_area(
                agent_view,
                area_pressures=self._area_pressures,
                pressure_weight=self.disturbance_weight,
            )
        if role == "area":
            area_id = int(agent_id.split("_", 1)[1])
            selected = self.kernel.select_cell(
                agent_view,
                cell_pressures=self._cell_pressures.get(area_id, {}),
                pressure_weight=self.disturbance_weight,
            )
            self._update_area_memory(area_id, _visible_cell_summaries(agent_view))
            return selected
        if role == "cell":
            cell_id = int(agent_view.get("cell_id", -1))
            selected = self.kernel.select_dispatch(
                agent_view,
                machine_pressures=self._machine_pressures.get(cell_id, {}),
                disturbance_weight=self.disturbance_weight,
            )
            self._update_machine_memory(cell_id, _visible_machine_summaries(agent_view))
            return selected
        return self.kernel.select_dispatch(agent_view)

    def _build_native_graph(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(_LangGraphDecisionState)
        graph.add_node("decide", self._native_decide_node)
        graph.add_edge(START, "decide")
        graph.add_edge("decide", END)
        return graph.compile(name="langgraph_tree_driver")

    def _native_decide_node(
        self,
        state: _LangGraphDecisionState,
    ) -> dict[str, int]:
        action = self._select_contract_action(
            agent_id=str(state["agent_id"]),
            agent_view=dict(state["agent_view"]),
        )
        return {"action": int(action)}

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

    def _update_plant_cell_memory(
        self,
        cell_summaries: Mapping[int, Mapping[str, object]],
    ) -> None:
        if not cell_summaries:
            return
        _, per_cell = self.kernel.area_pressures_from_cell_summaries(cell_summaries)
        memory = self._cell_pressures[-1]
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


class LangGraphLLMTreeDriver(LangGraphTreeDriver):
    """LangGraph-backed driver that can delegate agent-local action choice to an LLM."""

    controller_name = "external_framework_langgraph_llm"

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
        self.llm_backend = "openai_compat"
        self.llm_model = str(model_name)
        self.llm_base_url = str(base_url)
        self._llm_api_key = None if api_key is None else str(api_key)
        self.request_timeout = (
            None if request_timeout is None else float(request_timeout)
        )
        self.request_retries = max(int(request_retries), 0)
        self.retry_backoff_seconds = max(float(retry_backoff_seconds), 0.0)
        self._llm_client = self._build_openai_compat_client()
        super().__init__(
            memory_decay=memory_decay,
            disturbance_weight=disturbance_weight,
            strict_framework=strict_framework,
            controller_name=controller_name,
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

    def _build_native_graph(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(_LangGraphDecisionState)
        graph.add_node("decide", self._native_llm_decide_node)
        graph.add_edge(START, "decide")
        graph.add_edge("decide", END)
        return graph.compile(name="langgraph_llm_tree_driver")

    def _native_llm_decide_node(
        self,
        state: _LangGraphDecisionState,
    ) -> dict[str, int]:
        action = self._select_llm_or_contract_action(
            agent_id=str(state["agent_id"]),
            agent_view=dict(state["agent_view"]),
        )
        return {"action": int(action)}

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

        role = str(agent_view.get("role", "unknown"))
        if role == "area":
            area_id = int(agent_id.split("_", 1)[1])
            self._update_area_memory(area_id, _visible_cell_summaries(agent_view))
        elif role == "cell":
            cell_id = int(agent_view.get("cell_id", -1))
            self._update_machine_memory(cell_id, _visible_machine_summaries(agent_view))
        self._record_decision_source(
            agent_id=agent_id,
            source="llm",
            action=int(selected_action),
            legal_actions=legal_actions,
        )
        return int(selected_action)

    def _build_openai_compat_client(self) -> Any | None:
        if not self.llm_model or not self.llm_base_url:
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None
        try:
            client_kwargs: dict[str, object] = {
                "api_key": self._llm_api_key,
                "base_url": self.llm_base_url,
            }
            if self.request_timeout is not None:
                client_kwargs["timeout"] = self.request_timeout
            return OpenAI(**client_kwargs)
        except TypeError:
            if self.request_timeout is None:
                return None
            try:
                return OpenAI(
                    api_key=self._llm_api_key,
                    base_url=self.llm_base_url,
                )
            except TypeError:
                # Keep the driver usable against older or absent OpenAI client installs.
                return None

    def _select_openai_compat_action(
        self,
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
        legal_actions: list[int],
    ) -> int | None:
        if self._llm_client is None:
            return None

        for attempt in range(self.request_retries + 1):
            try:
                response = self._llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=_build_llm_messages(
                        agent_id=agent_id,
                        agent_view=agent_view,
                    ),
                    temperature=0.0,
                    max_tokens=OPENAI_COMPAT_BENCHMARK_MAX_TOKENS,
                )
            except Exception:
                response = None

            content = None if response is None else self._extract_openai_compat_content(response)
            selected = None if content is None else self._parse_action_from_content(content)
            if selected is not None and int(selected) in legal_actions:
                return selected
            if attempt < self.request_retries and self.retry_backoff_seconds > 0.0:
                time.sleep(self.retry_backoff_seconds * float(attempt + 1))
        return None

    @staticmethod
    def _render_llm_prompt(
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> str:
        return build_openai_compat_benchmark_user_prompt(
            agent_id=agent_id,
            agent_view=agent_view,
        )

    @staticmethod
    def _extract_openai_compat_content(response: Any) -> str | None:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        if message is None:
            return None
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, Mapping):
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
                    continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
            if text_parts:
                return "".join(text_parts)
        return None

    @staticmethod
    def _parse_action_from_content(content: str) -> int | None:
        return parse_openai_compat_benchmark_action_text(content)


def _register_langgraph_llm_driver_symbol() -> None:
    current_module = sys.modules[__name__]
    for module_name in (
        "fms_bench.v2.framework_drivers",
        "fms_bench.v2",
        "fms_bench",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        setattr(module, "LangGraphLLMTreeDriver", LangGraphLLMTreeDriver)
        exported = getattr(module, "__all__", None)
        if isinstance(exported, list) and "LangGraphLLMTreeDriver" not in exported:
            exported.append("LangGraphLLMTreeDriver")
    setattr(current_module, "LangGraphLLMTreeDriver", LangGraphLLMTreeDriver)


_register_langgraph_llm_driver_symbol()
