from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping


def _mask_valid_ids(agent_view: Mapping[str, object]) -> list[int]:
    return [
        int(index)
        for index, enabled in enumerate(list(agent_view.get("action_mask", [])))
        if int(enabled) > 0
    ]


def _visible_area_summaries(agent_view: Mapping[str, object]) -> dict[int, dict[str, Any]]:
    raw = dict(agent_view.get("visible_area_summaries", agent_view.get("area_summaries", {})))
    return {
        int(area_id): dict(summary)
        for area_id, summary in raw.items()
    }


def _visible_cell_summaries(agent_view: Mapping[str, object]) -> dict[int, dict[str, Any]]:
    raw = dict(agent_view.get("visible_cell_summaries", agent_view.get("cell_summaries", {})))
    return {
        int(cell_id): dict(summary)
        for cell_id, summary in raw.items()
    }


def _visible_machine_summaries(agent_view: Mapping[str, object]) -> dict[int, dict[str, Any]]:
    raw = dict(agent_view.get("visible_machine_summaries", {}))
    return {
        int(machine_id): dict(summary)
        for machine_id, summary in raw.items()
    }


def _first_valid_action(agent_view: Mapping[str, object]) -> int:
    valid = _mask_valid_ids(agent_view)
    return int(valid[0]) if valid else 0


def _ratio_if_present(summary: Mapping[str, object], value_key: str, cap_key: str) -> float:
    if cap_key not in summary:
        return 0.0
    return float(summary.get(value_key, 0.0)) / max(1.0, float(summary.get(cap_key, 1.0)))


@dataclass(frozen=True)
class ExternalFlatTreeDriver:
    """Role-agnostic external driver using only the serialized legal action mask."""

    controller_name: str = "external_tree_flat_mask_first"
    uses_external_session: bool = True

    def select_actions(self, decision: Mapping[str, object]) -> dict[str, int]:
        return {
            str(agent_id): _first_valid_action(dict(agent_view))
            for agent_id, agent_view in dict(decision["active_agents"]).items()
        }

    @property
    def name(self) -> str:
        return self.controller_name


@dataclass(frozen=True)
class ExternalHierarchicalTreeDriver:
    """Role-specialized external tree driver over the serialized legal slice."""

    controller_name: str = "external_tree_hierarchical_throughput"
    uses_external_session: bool = True

    def select_actions(self, decision: Mapping[str, object]) -> dict[str, int]:
        actions: dict[str, int] = {}
        for agent_id, agent_view in dict(decision["active_agents"]).items():
            agent_payload = dict(agent_view)
            role = str(agent_payload.get("role", "unknown"))
            if role == "plant":
                actions[str(agent_id)] = self._select_area(agent_payload)
            elif role == "area":
                actions[str(agent_id)] = self._select_cell(agent_payload)
            else:
                actions[str(agent_id)] = _first_valid_action(agent_payload)
        return actions

    def _select_area(self, agent_view: Mapping[str, object]) -> int:
        valid_areas = [int(area_id) for area_id in agent_view.get("valid_areas", _mask_valid_ids(agent_view))]
        if not valid_areas:
            return 0
        area_summaries = _visible_area_summaries(agent_view)
        return int(
            min(
                valid_areas,
                key=lambda area_id: self._area_score(area_summaries.get(area_id, {}), area_id),
            )
        )

    def _select_cell(self, agent_view: Mapping[str, object]) -> int:
        valid_cells = [int(cell_id) for cell_id in agent_view.get("valid_cells", _mask_valid_ids(agent_view))]
        if not valid_cells:
            return 0
        cell_summaries = _visible_cell_summaries(agent_view)
        return int(
            min(
                valid_cells,
                key=lambda cell_id: self._cell_score(cell_summaries.get(cell_id, {}), cell_id),
            )
        )

    @staticmethod
    def _area_score(summary: Mapping[str, object], area_id: int) -> tuple[float, float, float, int]:
        return (
            float(summary.get("active_jobs", 0.0)),
            float(summary.get("backlog_jobs", 0.0)),
            float(summary.get("energy_kwh", 0.0)),
            area_id,
        )

    @staticmethod
    def _cell_score(summary: Mapping[str, object], cell_id: int) -> tuple[float, int]:
        proc_scale = float(summary.get("processing_time_scale", 1.0))
        active = float(summary.get("active_jobs", 0.0))
        busy = float(summary.get("busy_machines", 0.0))
        backlog = float(summary.get("backlog_jobs", 0.0))
        composite = proc_scale * (1.0 + active + 0.5 * busy + 0.25 * backlog)
        return (composite, cell_id)

    @property
    def name(self) -> str:
        return self.controller_name


@dataclass(frozen=True)
class ExternalContractNetTreeDriver:
    """Budget-aware external MAS-like tree driver over serialized runtime slices."""

    controller_name: str = "external_tree_contract_net_budgeted"
    uses_external_session: bool = True

    def select_actions(self, decision: Mapping[str, object]) -> dict[str, int]:
        actions: dict[str, int] = {}
        for agent_id, agent_view in dict(decision["active_agents"]).items():
            agent_payload = dict(agent_view)
            role = str(agent_payload.get("role", "unknown"))
            if role == "plant":
                actions[str(agent_id)] = self._select_area(agent_payload)
            elif role == "area":
                actions[str(agent_id)] = self._select_cell(agent_payload)
            else:
                actions[str(agent_id)] = _first_valid_action(agent_payload)
        return actions

    def _select_area(self, agent_view: Mapping[str, object]) -> int:
        valid_areas = [int(area_id) for area_id in agent_view.get("valid_areas", _mask_valid_ids(agent_view))]
        if not valid_areas:
            return 0
        area_summaries = _visible_area_summaries(agent_view)
        return int(
            min(
                valid_areas,
                key=lambda area_id: self._area_score(area_summaries.get(area_id, {}), area_id),
            )
        )

    def _select_cell(self, agent_view: Mapping[str, object]) -> int:
        valid_cells = [int(cell_id) for cell_id in agent_view.get("valid_cells", _mask_valid_ids(agent_view))]
        if not valid_cells:
            return 0
        cell_summaries = _visible_cell_summaries(agent_view)
        return int(
            min(
                valid_cells,
                key=lambda cell_id: self._cell_bid_score(cell_summaries.get(cell_id, {}), cell_id),
            )
        )

    @staticmethod
    def _area_score(summary: Mapping[str, object], area_id: int) -> tuple[float, float, float, float, int]:
        return (
            float(summary.get("active_jobs", 0.0)),
            float(summary.get("backlog_jobs", 0.0)),
            float(summary.get("energy_kwh", 0.0)),
            float(summary.get("carbon_kg", 0.0)),
            area_id,
        )

    @staticmethod
    def _cell_bid_score(summary: Mapping[str, object], cell_id: int) -> tuple[float, int]:
        active_jobs = float(summary.get("active_jobs", 0.0))
        busy_machines = float(summary.get("busy_machines", 0.0))
        down_machines = float(summary.get("down_machines", 0.0))
        backlog_jobs = float(summary.get("backlog_jobs", 0.0))
        proc_scale = float(summary.get("processing_time_scale", 1.0))
        energy = float(summary.get("energy_kwh", 0.0))
        carbon = float(summary.get("carbon_kg", 0.0))
        energy_cap = max(1.0, float(summary.get("budget_cap_energy_kwh", 1.0)))
        carbon_cap = max(1.0, float(summary.get("budget_cap_carbon_kg", 1.0)))
        wip_cap = max(1.0, float(summary.get("budget_cap_wip", 1.0)))
        remaining_energy = max(1.0, energy_cap - energy)
        remaining_carbon = max(1.0, carbon_cap - carbon)

        score = (
            proc_scale * (1.0 + active_jobs + 0.5 * busy_machines + 0.25 * backlog_jobs)
            + 4.0 * down_machines
            + 2.0 * (active_jobs / wip_cap)
            + 1.5 * (energy / energy_cap)
            + 1.5 * (carbon / carbon_cap)
            + 2.5 * ((120.0 * proc_scale) / remaining_energy)
            + 2.5 * ((72.0 * proc_scale) / remaining_carbon)
        )
        return (float(score), cell_id)

    @property
    def name(self) -> str:
        return self.controller_name


@dataclass
class ExternalBlackboardTreeDriver:
    """Stateful external framework-style tree driver with shared routing memory."""

    controller_name: str = "external_tree_blackboard_balanced"
    memory_penalty: float = 2.0
    uses_external_session: bool = True

    def __post_init__(self) -> None:
        self._selection_counts: dict[str, dict[int, int]] = defaultdict(dict)

    def reset_episode(self) -> None:
        self._selection_counts.clear()

    def select_actions(self, decision: Mapping[str, object]) -> dict[str, int]:
        actions: dict[str, int] = {}
        for agent_id, agent_view in dict(decision["active_agents"]).items():
            agent_payload = dict(agent_view)
            role = str(agent_payload.get("role", "unknown"))
            if role == "plant":
                action = self._select_area(agent_payload)
            elif role == "area":
                action = self._select_cell(str(agent_id), agent_payload)
            else:
                action = _first_valid_action(agent_payload)
            actions[str(agent_id)] = int(action)
        return actions

    def _select_area(self, agent_view: Mapping[str, object]) -> int:
        valid_areas = [int(area_id) for area_id in agent_view.get("valid_areas", _mask_valid_ids(agent_view))]
        if not valid_areas:
            return 0
        area_summaries = _visible_area_summaries(agent_view)
        selected = int(
            min(
                valid_areas,
                key=lambda area_id: self._area_memory_score(area_summaries.get(area_id, {}), area_id),
            )
        )
        self._selection_counts["plant"][selected] = self._selection_counts["plant"].get(selected, 0) + 1
        return selected

    def _select_cell(self, agent_id: str, agent_view: Mapping[str, object]) -> int:
        valid_cells = [int(cell_id) for cell_id in agent_view.get("valid_cells", _mask_valid_ids(agent_view))]
        if not valid_cells:
            return 0
        cell_summaries = _visible_cell_summaries(agent_view)
        selected = int(
            min(
                valid_cells,
                key=lambda cell_id: self._cell_memory_score(agent_id, cell_summaries.get(cell_id, {}), cell_id),
            )
        )
        self._selection_counts[agent_id][selected] = self._selection_counts[agent_id].get(selected, 0) + 1
        return selected

    def _area_memory_score(self, summary: Mapping[str, object], area_id: int) -> tuple[float, float, float, float]:
        active_jobs, backlog_jobs, energy_kwh, _ = ExternalHierarchicalTreeDriver._area_score(summary, area_id)
        penalty = float(self._selection_counts["plant"].get(area_id, 0)) * float(self.memory_penalty)
        return (active_jobs + penalty, backlog_jobs, energy_kwh, float(area_id))

    def _cell_memory_score(
        self,
        agent_id: str,
        summary: Mapping[str, object],
        cell_id: int,
    ) -> tuple[float, float]:
        composite, _ = ExternalHierarchicalTreeDriver._cell_score(summary, cell_id)
        penalty = float(self._selection_counts[agent_id].get(cell_id, 0)) * float(self.memory_penalty)
        return (composite + penalty, float(cell_id))

    @property
    def name(self) -> str:
        return self.controller_name


@dataclass
class ExternalAdaptiveHolonTreeDriver:
    """ADACOR-style external driver with transient cooldowns over legal runtime slices."""

    controller_name: str = "external_tree_adacor_adaptive"
    transient_cooldown_steps: int = 2
    disturbance_threshold: float = 4.0
    transient_pressure_weight: float = 0.75
    transient_cooldown_weight: float = 2.0
    uses_external_session: bool = True

    def __post_init__(self) -> None:
        self._transient_cooldowns: dict[str, int] = defaultdict(int)

    def reset_episode(self) -> None:
        self._transient_cooldowns.clear()

    def select_actions(self, decision: Mapping[str, object]) -> dict[str, int]:
        self._age_transient_cooldowns()
        actions: dict[str, int] = {}
        for agent_id, agent_view in dict(decision["active_agents"]).items():
            agent_payload = dict(agent_view)
            role = str(agent_payload.get("role", "unknown"))
            if role == "plant":
                action = self._select_area(str(agent_id), agent_payload)
            elif role == "area":
                action = self._select_cell(str(agent_id), agent_payload)
            else:
                action = _first_valid_action(agent_payload)
            actions[str(agent_id)] = int(action)
        return actions

    def _select_area(self, controller_key: str, agent_view: Mapping[str, object]) -> int:
        valid_areas = [int(area_id) for area_id in agent_view.get("valid_areas", _mask_valid_ids(agent_view))]
        if not valid_areas:
            return 0
        area_summaries = _visible_area_summaries(agent_view)
        if self._is_transient(controller_key):
            selected = int(
                min(
                    valid_areas,
                    key=lambda area_id: self._transient_area_score(
                        controller_key,
                        area_summaries.get(area_id, {}),
                        area_id,
                    ),
                )
            )
        else:
            selected = int(
                min(
                    valid_areas,
                    key=lambda area_id: self._normal_area_score(area_summaries.get(area_id, {}), area_id),
                )
            )
        pressure = self._area_view_pressure(area_summaries)
        self._arm_transient_if_disturbed(controller_key=controller_key, pressure=pressure)
        return selected

    def _select_cell(self, controller_key: str, agent_view: Mapping[str, object]) -> int:
        valid_cells = [int(cell_id) for cell_id in agent_view.get("valid_cells", _mask_valid_ids(agent_view))]
        if not valid_cells:
            return 0
        cell_summaries = _visible_cell_summaries(agent_view)
        if self._is_transient(controller_key):
            selected = int(
                min(
                    valid_cells,
                    key=lambda cell_id: self._transient_cell_score(
                        controller_key,
                        cell_summaries.get(cell_id, {}),
                        cell_id,
                    ),
                )
            )
        else:
            selected = int(
                min(
                    valid_cells,
                    key=lambda cell_id: self._normal_cell_score(cell_summaries.get(cell_id, {}), cell_id),
                )
            )
        pressure = self._cell_view_pressure(cell_summaries)
        self._arm_transient_if_disturbed(controller_key=controller_key, pressure=pressure)
        return selected

    def _normal_area_score(self, summary: Mapping[str, object], area_id: int) -> tuple[float, float, float, int]:
        return ExternalHierarchicalTreeDriver._area_score(summary, area_id)

    def _normal_cell_score(self, summary: Mapping[str, object], cell_id: int) -> tuple[float, int]:
        return ExternalHierarchicalTreeDriver._cell_score(summary, cell_id)

    def _transient_area_score(
        self,
        controller_key: str,
        summary: Mapping[str, object],
        area_id: int,
    ) -> tuple[float, float, float, int]:
        base = self._normal_area_score(summary, area_id)
        pressure = self._area_disturbance_pressure(summary)
        score = (
            base[0]
            + self.transient_pressure_weight * pressure
            + self.transient_cooldown_weight * self._transient_cooldowns.get(controller_key, 0)
        )
        return (float(score), base[1], base[2], area_id)

    def _transient_cell_score(
        self,
        controller_key: str,
        summary: Mapping[str, object],
        cell_id: int,
    ) -> tuple[float, int]:
        base = self._normal_cell_score(summary, cell_id)
        pressure = self._cell_disturbance_pressure(summary)
        score = (
            base[0]
            + self.transient_pressure_weight * pressure
            + self.transient_cooldown_weight * self._transient_cooldowns.get(controller_key, 0)
        )
        return (float(score), cell_id)

    def _arm_transient_if_disturbed(self, *, controller_key: str, pressure: float) -> None:
        if pressure < self.disturbance_threshold:
            return
        self._transient_cooldowns[controller_key] = int(self.transient_cooldown_steps)

    def _age_transient_cooldowns(self) -> None:
        for controller_key, cooldown in list(self._transient_cooldowns.items()):
            if cooldown <= 1:
                self._transient_cooldowns.pop(controller_key, None)
            else:
                self._transient_cooldowns[controller_key] = int(cooldown - 1)

    def _is_transient(self, controller_key: str) -> bool:
        return int(self._transient_cooldowns.get(controller_key, 0)) > 0

    def _area_view_pressure(self, summaries: Mapping[int, Mapping[str, object]]) -> float:
        pressures = [self._area_disturbance_pressure(summary) for summary in summaries.values()]
        return float(max(pressures, default=0.0))

    def _cell_view_pressure(self, summaries: Mapping[int, Mapping[str, object]]) -> float:
        pressures = [self._cell_disturbance_pressure(summary) for summary in summaries.values()]
        return float(max(pressures, default=0.0))

    def _area_disturbance_pressure(self, summary: Mapping[str, object]) -> float:
        active_jobs = float(summary.get("active_jobs", 0.0))
        backlog_jobs = float(summary.get("backlog_jobs", 0.0))
        busy_machines = float(summary.get("busy_machines", 0.0))
        down_machines = float(summary.get("down_machines", 0.0))
        cell_count = max(1.0, float(len(list(summary.get("cell_ids", ())))))
        if "budget_cap_wip" in summary:
            wip_scale = max(1.0, float(summary.get("budget_cap_wip", 1.0)))
        else:
            # Plant-visible area summaries do not currently export full budget envelopes.
            wip_scale = cell_count
        return (
            4.0 * down_machines
            + 1.5 * (active_jobs / wip_scale)
            + 1.0 * (backlog_jobs / wip_scale)
            + 0.75 * busy_machines
            + 1.25 * _ratio_if_present(summary, "energy_kwh", "budget_cap_energy_kwh")
            + 1.25 * _ratio_if_present(summary, "carbon_kg", "budget_cap_carbon_kg")
        )

    def _cell_disturbance_pressure(self, summary: Mapping[str, object]) -> float:
        active_jobs = float(summary.get("active_jobs", 0.0))
        busy_machines = float(summary.get("busy_machines", 0.0))
        down_machines = float(summary.get("down_machines", 0.0))
        backlog_jobs = float(summary.get("backlog_jobs", 0.0))
        energy_kwh = float(summary.get("energy_kwh", 0.0))
        carbon_kg = float(summary.get("carbon_kg", 0.0))
        energy_cap = max(1.0, float(summary.get("budget_cap_energy_kwh", 1.0)))
        carbon_cap = max(1.0, float(summary.get("budget_cap_carbon_kg", 1.0)))
        wip_cap = max(1.0, float(summary.get("budget_cap_wip", 1.0)))
        return (
            5.0 * down_machines
            + 1.5 * (active_jobs / wip_cap)
            + 1.0 * (backlog_jobs / wip_cap)
            + 0.75 * busy_machines
            + 1.0 * (energy_kwh / energy_cap)
            + 1.0 * (carbon_kg / carbon_cap)
        )

    @property
    def name(self) -> str:
        return self.controller_name


@dataclass
class ExternalSupervisorReactiveTreeDriver:
    """Stateful supervisor-style external driver that propagates disturbance memory."""

    controller_name: str = "external_tree_supervisor_reactive"
    memory_decay: float = 0.7
    disturbance_weight: float = 2.5
    uses_external_session: bool = True

    def __post_init__(self) -> None:
        self._area_pressures: dict[int, float] = {}
        self._cell_pressures: dict[int, dict[int, float]] = defaultdict(dict)
        self._machine_pressures: dict[int, dict[int, float]] = defaultdict(dict)

    def reset_episode(self) -> None:
        self._area_pressures.clear()
        self._cell_pressures.clear()
        self._machine_pressures.clear()

    def select_actions(self, decision: Mapping[str, object]) -> dict[str, int]:
        self._decay_memory()
        actions: dict[str, int] = {}
        for agent_id, agent_view in dict(decision["active_agents"]).items():
            agent_payload = dict(agent_view)
            role = str(agent_payload.get("role", "unknown"))
            if role == "plant":
                action = self._select_area(agent_payload)
            elif role == "area":
                action = self._select_cell(str(agent_id), agent_payload)
            elif role == "cell":
                action = self._select_dispatch(agent_payload)
            else:
                action = _first_valid_action(agent_payload)
            actions[str(agent_id)] = int(action)
        return actions

    def _select_area(self, agent_view: Mapping[str, object]) -> int:
        valid_areas = [int(area_id) for area_id in agent_view.get("valid_areas", _mask_valid_ids(agent_view))]
        if not valid_areas:
            return 0
        area_summaries = _visible_area_summaries(agent_view)
        return int(
            min(
                valid_areas,
                key=lambda area_id: self._area_score(area_summaries.get(area_id, {}), area_id),
            )
        )

    def _select_cell(self, agent_id: str, agent_view: Mapping[str, object]) -> int:
        valid_cells = [int(cell_id) for cell_id in agent_view.get("valid_cells", _mask_valid_ids(agent_view))]
        if not valid_cells:
            return 0
        area_id = int(agent_id.split("_", 1)[1])
        cell_summaries = _visible_cell_summaries(agent_view)
        selected = int(
            min(
                valid_cells,
                key=lambda cell_id: self._cell_score(area_id, cell_summaries.get(cell_id, {}), cell_id),
            )
        )
        self._update_area_memory(area_id, cell_summaries)
        return selected

    def _select_dispatch(self, agent_view: Mapping[str, object]) -> int:
        dispatch_options = list(agent_view.get("dispatch_options", ()))
        if not dispatch_options:
            return _first_valid_action(agent_view)
        cell_id = int(agent_view.get("cell_id", -1))
        machine_summaries = _visible_machine_summaries(agent_view)
        selected_index = int(
            min(
                range(len(dispatch_options)),
                key=lambda index: self._dispatch_score(
                    cell_id=cell_id,
                    option=dict(dispatch_options[index]),
                    machine_summaries=machine_summaries,
                ),
            )
        )
        self._update_machine_memory(cell_id, machine_summaries)
        return selected_index

    def _area_score(self, summary: Mapping[str, object], area_id: int) -> tuple[float, float, float, int]:
        active, backlog, energy, _ = ExternalHierarchicalTreeDriver._area_score(summary, area_id)
        remembered = float(self._area_pressures.get(area_id, 0.0))
        composite = active + self.disturbance_weight * remembered
        return (float(composite), backlog, energy, area_id)

    def _cell_score(
        self,
        area_id: int,
        summary: Mapping[str, object],
        cell_id: int,
    ) -> tuple[float, int]:
        base, _ = ExternalHierarchicalTreeDriver._cell_score(summary, cell_id)
        live_pressure = self._cell_disturbance_pressure(summary)
        remembered = float(self._cell_pressures.get(area_id, {}).get(cell_id, 0.0))
        composite = base + self.disturbance_weight * (live_pressure + remembered)
        return (float(composite), cell_id)

    def _dispatch_score(
        self,
        *,
        cell_id: int,
        option: Mapping[str, object],
        machine_summaries: Mapping[int, Mapping[str, object]],
    ) -> tuple[float, int]:
        machine_id = int(option.get("machine_id", -1))
        proc_time = float(option.get("proc_time", 0.0))
        live_pressure = self._machine_disturbance_pressure(machine_summaries.get(machine_id, {}))
        remembered = float(self._machine_pressures.get(cell_id, {}).get(machine_id, 0.0))
        composite = proc_time + self.disturbance_weight * (live_pressure + remembered)
        return (float(composite), machine_id)

    def _update_area_memory(
        self,
        area_id: int,
        cell_summaries: Mapping[int, Mapping[str, object]],
    ) -> None:
        if not cell_summaries:
            return
        remembered_area = max(
            self._cell_disturbance_pressure(summary)
            for summary in cell_summaries.values()
        )
        self._area_pressures[area_id] = max(
            float(self._area_pressures.get(area_id, 0.0)),
            float(remembered_area),
        )
        area_cell_memory = self._cell_pressures[area_id]
        for cell_id, summary in cell_summaries.items():
            area_cell_memory[int(cell_id)] = max(
                float(area_cell_memory.get(int(cell_id), 0.0)),
                self._cell_disturbance_pressure(summary),
            )

    def _update_machine_memory(
        self,
        cell_id: int,
        machine_summaries: Mapping[int, Mapping[str, object]],
    ) -> None:
        if cell_id < 0:
            return
        cell_machine_memory = self._machine_pressures[cell_id]
        for machine_id, summary in machine_summaries.items():
            cell_machine_memory[int(machine_id)] = max(
                float(cell_machine_memory.get(int(machine_id), 0.0)),
                self._machine_disturbance_pressure(summary),
            )

    def _decay_memory(self) -> None:
        self._area_pressures = {
            area_id: float(value) * float(self.memory_decay)
            for area_id, value in self._area_pressures.items()
            if float(value) * float(self.memory_decay) > 1e-9
        }
        self._cell_pressures = defaultdict(
            dict,
            {
                int(area_id): {
                    int(cell_id): float(value) * float(self.memory_decay)
                    for cell_id, value in cell_memory.items()
                    if float(value) * float(self.memory_decay) > 1e-9
                }
                for area_id, cell_memory in self._cell_pressures.items()
                if any(float(value) * float(self.memory_decay) > 1e-9 for value in cell_memory.values())
            },
        )
        self._machine_pressures = defaultdict(
            dict,
            {
                int(cell_id): {
                    int(machine_id): float(value) * float(self.memory_decay)
                    for machine_id, value in machine_memory.items()
                    if float(value) * float(self.memory_decay) > 1e-9
                }
                for cell_id, machine_memory in self._machine_pressures.items()
                if any(float(value) * float(self.memory_decay) > 1e-9 for value in machine_memory.values())
            },
        )

    @staticmethod
    def _cell_disturbance_pressure(summary: Mapping[str, object]) -> float:
        active_jobs = float(summary.get("active_jobs", 0.0))
        busy_machines = float(summary.get("busy_machines", 0.0))
        backlog_jobs = float(summary.get("backlog_jobs", 0.0))
        down_machines = float(summary.get("down_machines", 0.0))
        return (
            4.0 * down_machines
            + 1.0 * busy_machines
            + 0.5 * backlog_jobs
            + 0.25 * active_jobs
        )

    @staticmethod
    def _machine_disturbance_pressure(summary: Mapping[str, object]) -> float:
        status = str(summary.get("status", "idle"))
        down_time = float(summary.get("down_time", 0.0))
        remaining_time = float(summary.get("remaining_time", 0.0))
        status_penalty = 0.0
        if status == "down":
            status_penalty = 4.0
        elif status == "busy":
            status_penalty = 1.5
        return status_penalty + 0.5 * down_time + 0.1 * remaining_time

    @property
    def name(self) -> str:
        return self.controller_name
