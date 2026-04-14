"""Tree-holonic controller baselines for flat, hierarchical, and MAS-style comparisons."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .baselines import BaselinePolicy


class _BaseTreePolicy(BaselinePolicy):
    def _agent_id(self, info: Dict[str, Any]) -> str:
        return str(info.get("agent_id", ""))

    def _mask_valid_ids(self, info: Dict[str, Any]) -> List[int]:
        mask = np.asarray(info.get("action_mask", []), dtype=np.int32)
        return [int(idx) for idx in np.flatnonzero(mask > 0)]

    def _visible_area_summaries(self, info: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        raw = info.get("visible_area_summaries", info.get("area_summaries", {}))
        return {int(area_id): dict(summary) for area_id, summary in dict(raw).items()}

    def _visible_cell_summaries(self, info: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        raw = info.get("visible_cell_summaries", info.get("cell_summaries", {}))
        return {int(cell_id): dict(summary) for cell_id, summary in dict(raw).items()}

    def _active_area_ids(self, info: Dict[str, Any]) -> List[int]:
        valid_areas = info.get("valid_areas")
        if valid_areas is not None:
            return [int(area_id) for area_id in valid_areas]
        if self._agent_id(info) == "plant":
            return self._mask_valid_ids(info)
        return [
            int(agent.split("_", 1)[1])
            for agent in info.get("agents_to_act", [])
            if agent.startswith("area_")
        ]

    def _valid_cells_for_area(self, info: Dict[str, Any], area_id: int) -> List[int]:
        visible_cell_summaries = self._visible_cell_summaries(info)
        if visible_cell_summaries:
            valid_cells = set(int(cell_id) for cell_id in info.get("valid_cells", visible_cell_summaries))
            return [
                int(cell_id)
                for cell_id in sorted(visible_cell_summaries)
                if int(cell_id) in valid_cells
            ]
        valid_cells = set(int(cell_id) for cell_id in info.get("valid_cells", []))
        area_summary = info.get("area_summaries", {}).get(area_id, {})
        return [
            int(cell_id)
            for cell_id in area_summary.get("cell_ids", [])
            if int(cell_id) in valid_cells
        ]

    def _first_valid_action(self, info: Dict[str, Any]) -> int:
        valid = self._mask_valid_ids(info)
        return int(valid[0]) if valid else 0

    def _build_action_dict(
        self,
        info: Dict[str, Any],
        area_id: int,
        cell_id: int,
    ) -> Dict[str, int]:
        action = {
            "plant": int(area_id),
            f"area_{area_id}": int(cell_id),
        }
        for agent in info.get("agents_to_act", []):
            if agent.startswith("cell_"):
                action[agent] = 0
        return action


class FlatTreePolicy(_BaseTreePolicy):
    """Flat adapter that only follows public masks and valid controller ids."""

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> Dict[str, int]:
        if self._agent_id(info):
            return self._first_valid_action(info)
        mask = np.asarray(info.get("action_mask", []), dtype=np.int32)
        valid_areas = np.flatnonzero(mask > 0)
        area_id = int(valid_areas[0]) if len(valid_areas) else 0
        valid_cells = self._valid_cells_for_area(info, area_id)
        cell_id = int(valid_cells[0]) if valid_cells else 0
        return self._build_action_dict(info, area_id=area_id, cell_id=cell_id)

    @property
    def name(self) -> str:
        return "tree_flat_mask_first"


class HierarchicalTreePolicy(_BaseTreePolicy):
    """Hierarchical baseline using area aggregation then cell throughput/load selection."""

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> Dict[str, int]:
        agent_id = self._agent_id(info)
        if agent_id == "plant":
            valid_areas = self._active_area_ids(info)
            if not valid_areas:
                return 0
            area_summaries = self._visible_area_summaries(info)
            return int(min(valid_areas, key=lambda area_id: self._area_score(area_summaries.get(area_id, {}), area_id)))
        if agent_id.startswith("area_"):
            area_id = int(agent_id.split("_", 1)[1])
            valid_cells = self._valid_cells_for_area(info, area_id)
            if not valid_cells:
                return 0
            cell_summaries = self._visible_cell_summaries(info)
            return int(min(valid_cells, key=lambda cell_id: self._cell_score(cell_summaries.get(cell_id, {}), cell_id)))
        if agent_id.startswith("cell_"):
            return 0

        valid_areas = self._active_area_ids(info)
        if not valid_areas:
            return {"plant": 0}

        area_summaries = self._visible_area_summaries(info)
        selected_area = min(valid_areas, key=lambda area_id: self._area_score(area_summaries.get(area_id, {}), area_id))
        valid_cells = self._valid_cells_for_area(info, selected_area)
        cell_summaries = self._visible_cell_summaries(info)
        selected_cell = min(valid_cells, key=lambda cell_id: self._cell_score(cell_summaries.get(cell_id, {}), cell_id)) if valid_cells else 0
        return self._build_action_dict(info, area_id=selected_area, cell_id=selected_cell)

    def _area_score(self, summary: Dict[str, Any], area_id: int) -> tuple[float, float, float, int]:
        return (
            float(summary.get("active_jobs", 0.0)),
            float(summary.get("backlog_jobs", 0.0)),
            float(summary.get("energy_kwh", 0.0)),
            area_id,
        )

    def _cell_score(self, summary: Dict[str, Any], cell_id: int) -> tuple[float, int]:
        proc_scale = float(summary.get("processing_time_scale", 1.0))
        active = float(summary.get("active_jobs", 0.0))
        busy = float(summary.get("busy_machines", 0.0))
        backlog = float(summary.get("backlog_jobs", 0.0))
        composite = proc_scale * (1.0 + active + 0.5 * busy + 0.25 * backlog)
        return (composite, cell_id)

    @property
    def name(self) -> str:
        return "tree_hierarchical_throughput"


class TreeContractNetPolicy(_BaseTreePolicy):
    """Tree-only MAS-style baseline using budget-aware parent-child bidding."""

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> Dict[str, int]:
        agent_id = self._agent_id(info)
        if agent_id == "plant":
            valid_areas = self._active_area_ids(info)
            if not valid_areas:
                return 0
            area_summaries = self._visible_area_summaries(info)
            return int(min(valid_areas, key=lambda area_id: self._plant_area_bid_score(area_summaries.get(area_id, {}), area_id)))
        if agent_id.startswith("area_"):
            area_id = int(agent_id.split("_", 1)[1])
            valid_cells = self._valid_cells_for_area(info, area_id)
            if not valid_cells:
                return 0
            cell_summaries = self._visible_cell_summaries(info)
            return int(min(valid_cells, key=lambda cell_id: self._cell_bid_score(cell_summaries.get(cell_id, {}), cell_id)))
        if agent_id.startswith("cell_"):
            return 0

        valid_areas = self._active_area_ids(info)
        if not valid_areas:
            return {"plant": 0}

        selected_area = min(valid_areas, key=lambda area_id: self._area_bid_score(info, area_id))
        valid_cells = self._valid_cells_for_area(info, selected_area)
        selected_cell = min(valid_cells, key=lambda cell_id: self._cell_bid_score(info.get("cell_summaries", {}).get(cell_id, {}), cell_id)) if valid_cells else 0
        return self._build_action_dict(info, area_id=selected_area, cell_id=selected_cell)

    def _area_bid_score(self, info: Dict[str, Any], area_id: int) -> tuple[float, int]:
        valid_cells = self._valid_cells_for_area(info, area_id)
        if not valid_cells:
            return (float("inf"), area_id)
        best_cell_score = min(
            self._cell_bid_score(self._visible_cell_summaries(info).get(cell_id, {}), cell_id)[0]
            for cell_id in valid_cells
        )
        return (best_cell_score, area_id)

    def _plant_area_bid_score(self, summary: Dict[str, Any], area_id: int) -> tuple[float, float, float, float, int]:
        return (
            float(summary.get("active_jobs", 0.0)),
            float(summary.get("backlog_jobs", 0.0)),
            float(summary.get("energy_kwh", 0.0)),
            float(summary.get("carbon_kg", 0.0)),
            area_id,
        )

    def _cell_bid_score(self, summary: Dict[str, Any], cell_id: int) -> tuple[float, int]:
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
        return "tree_contract_net_budgeted"
