"""Public tree-holonic benchmark environment over the shared DES kernel."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from gymnasium import spaces

from .config import BenchmarkConfig
from .holonic_types import AreaFeedback, PlantCellConfig, PlantContract, TreeMessage
from .plant_env import HolonicPlantEnv, PlantStepResult


class TreeHolonicEnv(HolonicPlantEnv):
    """Expose Plant / Area / Cell controllers over the shared plant simulator."""

    def __init__(
        self,
        cell_configs: Iterable[PlantCellConfig],
        area_to_cells: Mapping[int, Sequence[int]],
        config: Optional[BenchmarkConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(cell_configs, config=config, seed=seed)
        self.area_to_cells = self._validate_area_to_cells(area_to_cells)
        self.area_ids = sorted(self.area_to_cells)
        self.area_index = {area_id: idx for idx, area_id in enumerate(self.area_ids)}
        self.index_to_area = {idx: area_id for area_id, idx in self.area_index.items()}
        self.cell_to_area = {
            cell_id: area_id
            for area_id, cell_ids in self.area_to_cells.items()
            for cell_id in cell_ids
        }
        self.action_space = spaces.Discrete(max(1, len(self.area_ids)))

    def observe(self, agent: str = "plant") -> np.ndarray:
        if agent == "plant":
            return self._build_observation()
        if agent.startswith("area_"):
            area_id = int(agent.split("_", 1)[1])
            return self._build_area_observation(area_id)
        return super().observe(agent)

    def action_mask(self, agent: str = "plant") -> np.ndarray:
        if agent == "plant":
            mask = np.zeros(self.action_space.n, dtype=np.int8)
            for area_id in self._valid_release_areas():
                mask[self.area_index[area_id]] = 1
            if not mask.any() and len(mask) > 0:
                mask[0] = 1
            return mask

        if agent.startswith("area_"):
            area_id = int(agent.split("_", 1)[1])
            mask = np.zeros(len(self.assembly.cells), dtype=np.int8)
            valid_cells = set(self._valid_release_cells())
            for cell_id in self.area_to_cells[area_id]:
                if cell_id in valid_cells:
                    mask[cell_id] = 1
            if not mask.any() and self.area_to_cells[area_id]:
                mask[self.area_to_cells[area_id][0]] = 1
            return mask

        if agent.startswith("cell_"):
            return np.ones(1, dtype=np.int8)

        raise ValueError(f"Unknown agent for action_mask(): {agent}")

    def agents_to_act(self) -> List[str]:
        if self._terminated or self._truncated:
            return []

        agents = ["plant"]
        valid_areas = self._valid_release_areas()
        agents.extend(f"area_{area_id}" for area_id in valid_areas)
        for area_id in valid_areas:
            agents.extend(f"cell_{cell_id}" for cell_id in self.area_to_cells[area_id])
        return agents

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        result = self._tree_step(int(action), action_dict={})
        return result.observation, result.reward, result.terminated, result.truncated, result.info

    def step_dict(self, action_dict: Dict[str, int]) -> PlantStepResult:
        agents = self.agents_to_act()
        if not agents:
            raise RuntimeError("No controller action is expected at the current decision epoch")
        if "plant" not in action_dict:
            raise KeyError("action_dict must include a 'plant' action")
        return self._tree_step(int(action_dict["plant"]), action_dict=action_dict)

    def _tree_step(self, plant_action: int, action_dict: Dict[str, int]) -> PlantStepResult:
        if self._terminated or self._truncated:
            raise RuntimeError("Episode already ended. Call reset() before step().")

        invalid_action = False
        contract: Optional[PlantContract] = None
        selected_area_id: Optional[int] = None
        selected_cell_id: Optional[int] = None
        valid_areas = self._valid_release_areas()

        if valid_areas:
            valid_area_indices = [self.area_index[area_id] for area_id in valid_areas]
            if plant_action in valid_area_indices:
                selected_area_id = self.index_to_area[plant_action]
            else:
                selected_area_id = valid_areas[0]
                invalid_action = True

            valid_cells = [
                cell_id
                for cell_id in self.area_to_cells[selected_area_id]
                if cell_id in self._valid_release_cells()
            ]
            if valid_cells:
                area_agent = f"area_{selected_area_id}"
                area_action = int(action_dict.get(area_agent, valid_cells[0]))
                if area_action in valid_cells:
                    selected_cell_id = area_action
                else:
                    selected_cell_id = valid_cells[0]
                    invalid_action = True
                contract = self._release_next_job(selected_cell_id)
        elif plant_action != 0:
            invalid_action = True

        if invalid_action:
            self._invalid_actions += 1

        transition = self._advance_until_plant_decision()
        self._last_transition = transition
        self._last_contract = contract

        terminated = self._is_done()
        truncated = False
        if not terminated:
            if self.simulator.time >= float(self.config.horizon_hours):
                truncated = True
            if bool(transition["deadlocked"]):
                truncated = True

        self._terminated = terminated
        self._truncated = truncated

        delta_t = float(transition["time_end"]) - float(transition["time_start"])
        reward = -delta_t
        reward -= 0.01 * float(transition["energy_kwh_step"])
        reward -= 0.001 * float(transition["carbon_kg_step"])
        if invalid_action:
            reward -= 0.1
        if terminated:
            reward += 1.0

        observation = self._build_observation()
        info = self._build_info(transition, invalid_action=invalid_action, contract=contract)
        info["selected_area_id"] = selected_area_id
        info["selected_cell_id"] = selected_cell_id
        return PlantStepResult(
            observation=observation,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _valid_release_areas(self) -> List[int]:
        valid_cells = set(self._valid_release_cells())
        return [
            area_id
            for area_id in self.area_ids
            if any(cell_id in valid_cells for cell_id in self.area_to_cells[area_id])
        ]

    def _area_feedback(self, area_id: int) -> AreaFeedback:
        cell_feedbacks = [self._cell_feedback(cell_id) for cell_id in self.area_to_cells[area_id]]
        return AreaFeedback(
            area_id=area_id,
            label=f"area_{area_id}",
            cell_ids=list(self.area_to_cells[area_id]),
            backlog_jobs=sum(item.backlog_jobs for item in cell_feedbacks),
            active_jobs=sum(item.active_jobs for item in cell_feedbacks),
            completed_jobs=sum(item.completed_jobs for item in cell_feedbacks),
            energy_kwh=sum(item.energy_kwh for item in cell_feedbacks),
            carbon_kg=sum(item.carbon_kg for item in cell_feedbacks),
            energy_overshoot_kwh=sum(item.energy_overshoot_kwh for item in cell_feedbacks),
            carbon_overshoot_kg=sum(item.carbon_overshoot_kg for item in cell_feedbacks),
        )

    def _build_area_observation(self, area_id: int) -> np.ndarray:
        feedback = self._area_feedback(area_id)
        cell_feedbacks = [self._cell_feedback(cell_id) for cell_id in self.area_to_cells[area_id]]
        area_machine_count = max(
            1,
            sum(len(self.assembly.cells[cell_id].machine_ids) for cell_id in self.area_to_cells[area_id]),
        )
        area_wip_cap = max(
            1,
            sum(self.assembly.cells[cell_id].budget.wip_cap for cell_id in self.area_to_cells[area_id]),
        )
        area_job_count = max(
            1,
            sum(len(self.assembly.cells[cell_id].template_job_ids) for cell_id in self.area_to_cells[area_id]),
        )
        area_energy_cap = max(
            1.0,
            sum(self.assembly.cells[cell_id].budget.energy_cap_kwh for cell_id in self.area_to_cells[area_id]),
        )
        area_carbon_cap = max(
            1.0,
            sum(self.assembly.cells[cell_id].budget.carbon_cap_kg for cell_id in self.area_to_cells[area_id]),
        )
        busy_machines = sum(item.busy_machines for item in cell_feedbacks)
        down_machines = sum(item.down_machines for item in cell_feedbacks)

        obs = np.zeros(8, dtype=np.float32)
        obs[0] = float(min(1.0, feedback.backlog_jobs / max(1, self.assembly.total_job_count)))
        obs[1] = float(min(1.0, feedback.active_jobs / area_wip_cap))
        obs[2] = float(min(1.0, feedback.completed_jobs / area_job_count))
        obs[3] = float(min(1.0, busy_machines / area_machine_count))
        obs[4] = float(min(1.0, down_machines / area_machine_count))
        obs[5] = float(min(1.0, feedback.energy_kwh / area_energy_cap))
        obs[6] = float(min(1.0, feedback.carbon_kg / area_carbon_cap))
        obs[7] = float(min(1.0, self.simulator.time / max(1.0, float(self.config.horizon_hours))))
        return obs

    def _typed_messages(self, contract: Optional[PlantContract]) -> List[TreeMessage]:
        if contract is None:
            return []

        area_id = self.cell_to_area[contract.cell_id]
        payload = {
            "contract_id": contract.contract_id,
            "job_id": contract.job_id,
            "cell_id": contract.cell_id,
            "area_id": area_id,
            "released_at": contract.released_at,
            "due_time": contract.due_time,
            "horizon_remaining": contract.horizon_remaining,
        }
        return [
            TreeMessage(
                sender="plant",
                recipient=f"area_{area_id}",
                kind="release_request",
                payload=payload,
            ),
            TreeMessage(
                sender=f"area_{area_id}",
                recipient=f"cell_{contract.cell_id}",
                kind="release_contract",
                payload=payload,
            ),
        ]

    def _build_info(
        self,
        transition: Dict[str, Any],
        invalid_action: bool,
        contract: Optional[PlantContract],
    ) -> Dict[str, Any]:
        info = super()._build_info(transition, invalid_action=invalid_action, contract=contract)
        info["valid_areas"] = self._valid_release_areas()
        info["area_summaries"] = {
            area_id: self._area_feedback(area_id).to_dict()
            for area_id in self.area_ids
        }
        return info

    def _validate_area_to_cells(self, area_to_cells: Mapping[int, Sequence[int]]) -> Dict[int, List[int]]:
        if not area_to_cells:
            raise ValueError("TreeHolonicEnv requires at least one area")

        normalized = {int(area_id): [int(cell_id) for cell_id in cell_ids] for area_id, cell_ids in area_to_cells.items()}
        expected_cells = set(self.assembly.cells)
        seen_cells = set()
        for area_id, cell_ids in normalized.items():
            if area_id < 0:
                raise ValueError("Area ids must be non-negative")
            if not cell_ids:
                raise ValueError(f"Area {area_id} must include at least one cell")
            for cell_id in cell_ids:
                if cell_id not in expected_cells:
                    raise ValueError(f"Unknown cell id in area mapping: {cell_id}")
                if cell_id in seen_cells:
                    raise ValueError(f"Cell {cell_id} assigned to multiple areas")
                seen_cells.add(cell_id)

        if seen_cells != expected_cells:
            missing = sorted(expected_cells - seen_cells)
            raise ValueError(f"Cells missing from area mapping: {missing}")
        return {area_id: list(cell_ids) for area_id, cell_ids in sorted(normalized.items())}
