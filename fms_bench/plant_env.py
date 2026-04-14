"""Holonic plant environment built on top of the shared DES kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .baselines import SPT_Policy
from .config import BenchmarkConfig
from .constraints import EnergyModel
from .data_loader import InstanceData
from .holonic_types import CellFeedback, HolonicEvent, HolonicMessage, PlantAssembly, PlantCellConfig, PlantContract
from .simulator import DispatchOption, EventDrivenSimulator
from .suite_builder import build_plant_assembly


@dataclass
class PlantStepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class HolonicPlantEnv(gym.Env):
    """Plant-level environment with cell-level automatic dispatch."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cell_configs: Iterable[PlantCellConfig],
        config: Optional[BenchmarkConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config or BenchmarkConfig()
        self.base_seed = seed
        self.assembly: PlantAssembly = build_plant_assembly(list(cell_configs), config=self.config)
        if not self.assembly.cells:
            raise ValueError("HolonicPlantEnv requires at least one cell")

        combined_instance = InstanceData(
            jobs=self.assembly.combined_jobs,
            n_jobs=self.assembly.total_job_count,
            n_machines=self.assembly.total_machine_count,
        )
        self.simulator = EventDrivenSimulator(combined_instance, self.config)
        self.energy_model = EnergyModel(
            peak_window=self.config.energy.peak_window,
            peak_price=self.config.energy.peak_price,
            offpeak_price=self.config.energy.offpeak_price,
        )
        self.cell_policies = {cell_id: SPT_Policy() for cell_id in self.assembly.cells}

        self.action_space = spaces.Discrete(len(self.assembly.cells))
        obs_dim = 3 + 6 * len(self.assembly.cells)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._terminated = False
        self._truncated = False
        self._contract_seq = 0
        self._released_jobs: set[int] = set()
        self._backlogs: Dict[int, List[int]] = {}
        self._cell_energy_kwh: Dict[int, float] = {}
        self._cell_energy_cost: Dict[int, float] = {}
        self._cell_carbon_kg: Dict[int, float] = {}
        self._cell_setup_time: Dict[int, float] = {}
        self._cell_setup_energy_kwh: Dict[int, float] = {}
        self._last_contract: Optional[PlantContract] = None
        self._last_transition: Dict[str, Any] = {}
        self._invalid_actions = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        del options

        effective_seed = seed if seed is not None else self.base_seed
        self.simulator.reset(seed=effective_seed, schedule_job_arrivals=False)
        self._terminated = False
        self._truncated = False
        self._contract_seq = 0
        self._released_jobs = set()
        self._backlogs = {
            cell_id: list(cell.template_job_ids)
            for cell_id, cell in self.assembly.cells.items()
        }
        self._cell_energy_kwh = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._cell_energy_cost = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._cell_carbon_kg = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._cell_setup_time = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._cell_setup_energy_kwh = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._last_contract = None
        self._last_transition = {
            "time_start": 0.0,
            "time_end": 0.0,
            "events": [],
            "deadlocked": False,
            "setup_time_step": 0.0,
            "energy_kwh_step": 0.0,
            "energy_cost_step": 0.0,
            "carbon_kg_step": 0.0,
        }
        self._invalid_actions = 0

        obs = self._build_observation()
        info = self._build_info(self._last_transition, invalid_action=False, contract=None)
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._terminated or self._truncated:
            raise RuntimeError("Episode already ended. Call reset() before step().")

        action = int(action)
        invalid_action = False
        contract: Optional[PlantContract] = None
        valid_cells = self._valid_release_cells()

        if valid_cells:
            if action not in valid_cells:
                action = valid_cells[0]
                invalid_action = True
            contract = self._release_next_job(action)
        elif action != 0:
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

        obs = self._build_observation()
        info = self._build_info(transition, invalid_action=invalid_action, contract=contract)
        return obs, float(reward), terminated, truncated, info

    def observe(self, agent: str = "plant") -> np.ndarray:
        """Return a benchmark-facing observation slice for one agent."""
        if agent == "plant":
            return self._build_observation()
        if agent.startswith("cell_"):
            cell_id = int(agent.split("_", 1)[1])
            return self._build_cell_observation(cell_id)
        raise ValueError(f"Unknown agent for observe(): {agent}")

    def action_mask(self, agent: str = "plant") -> np.ndarray:
        """Return a binary action mask for the requested agent."""
        if agent != "plant":
            raise ValueError(f"Unknown agent for action_mask(): {agent}")
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        valid_cells = self._valid_release_cells()
        if valid_cells:
            for cell_id in valid_cells:
                if 0 <= cell_id < len(mask):
                    mask[cell_id] = 1
        elif len(mask) > 0:
            mask[0] = 1
        return mask

    def agents_to_act(self) -> List[str]:
        """Return benchmark-facing agents expected to act at the current decision epoch."""
        if self._terminated or self._truncated:
            return []
        return ["plant"]

    def step_dict(self, action_dict: Dict[str, int]) -> PlantStepResult:
        """Dict-based stepping API for future HRL and MAS-style controllers."""
        agents = self.agents_to_act()
        if agents != ["plant"]:
            raise RuntimeError("No plant action is expected at the current decision epoch")
        if "plant" not in action_dict:
            raise KeyError("action_dict must include a 'plant' action")
        observation, reward, terminated, truncated, info = self.step(int(action_dict["plant"]))
        return PlantStepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _release_next_job(self, cell_id: int) -> PlantContract:
        backlog = self._backlogs[cell_id]
        if not backlog:
            raise RuntimeError(f"Cell {cell_id} has no remaining backlog")
        job_id = backlog.pop(0)
        self._released_jobs.add(job_id)
        self.simulator.activate_job(job_id, release_time=self.simulator.time)

        self._contract_seq += 1
        job = self.simulator.jobs[job_id]
        return PlantContract(
            contract_id=self._contract_seq,
            cell_id=cell_id,
            job_id=job_id,
            released_at=self.simulator.time,
            due_time=job.due_time,
            horizon_remaining=max(0.0, float(self.config.horizon_hours) - self.simulator.time),
        )

    def _advance_until_plant_decision(self) -> Dict[str, Any]:
        return self._advance_until_decision(auto_dispatch=True, stop_on_dispatch=False)

    def _advance_until_decision(
        self,
        *,
        auto_dispatch: bool,
        stop_on_dispatch: bool,
    ) -> Dict[str, Any]:
        energy_kwh_step = 0.0
        energy_cost_step = 0.0
        carbon_kg_step = 0.0
        setup_time_step = 0.0
        event_log: List[str] = []
        event_records: List[Dict[str, Any]] = []
        start_time = self.simulator.time
        deadlocked = False

        def on_advance(start: float, end: float, machines: Dict[int, Any]) -> None:
            nonlocal energy_kwh_step, energy_cost_step, carbon_kg_step
            interval = self.energy_model.interval_energy(machines, start, end)
            energy_kwh_step += interval["energy_kwh"]
            energy_cost_step += interval["energy_cost"]
            carbon_kg_step += interval["energy_kwh"] * self.config.carbon.kg_per_kwh

            for cell_id, cell in self.assembly.cells.items():
                subset = {mid: machines[mid] for mid in cell.machine_ids}
                cell_interval = self.energy_model.interval_energy(subset, start, end)
                self._cell_energy_kwh[cell_id] += cell_interval["energy_kwh"]
                self._cell_energy_cost[cell_id] += cell_interval["energy_cost"]
                self._cell_carbon_kg[cell_id] += cell_interval["energy_kwh"] * self.config.carbon.kg_per_kwh

        while True:
            if self._is_done():
                break

            valid_cells = self._valid_release_cells()
            if valid_cells:
                break

            dispatch_options = self.simulator.get_dispatch_options()
            if dispatch_options:
                if stop_on_dispatch:
                    break
                if auto_dispatch:
                    option = self._select_local_dispatch_option(dispatch_options)
                    if option is not None:
                        self.dispatch_job_to_machine(option.job_id, option.machine_id)
                        continue

            transition = self.simulator.run_until_decision(
                advance_hook=on_advance,
                stop_condition=lambda: bool(self._valid_release_cells())
                or (stop_on_dispatch and bool(self.simulator.get_dispatch_options())),
            )
            event_log.extend(list(transition["events"]))
            event_records.extend(list(transition.get("event_records", [])))
            for event_record in transition.get("event_records", []):
                if str(event_record.get("kind", "")) != "setup_complete":
                    continue
                payload = dict(event_record.get("payload", {}))
                setup_duration = float(payload.get("setup_duration", 0.0))
                setup_time_step += setup_duration
                machine_id = payload.get("machine_id")
                if isinstance(machine_id, int):
                    cell_id = self.assembly.machine_to_cell.get(machine_id)
                    if cell_id is not None:
                        self._cell_setup_time[cell_id] += setup_duration
                        machine = self.simulator.machines.get(machine_id)
                        if machine is not None:
                            self._cell_setup_energy_kwh[cell_id] += (
                                setup_duration * float(machine.setup_power_kw)
                            )
            if (
                float(transition["time_end"]) == float(transition["time_start"])
                and not transition["events"]
                and not self._is_done()
            ):
                deadlocked = True
                break

        return {
            "time_start": start_time,
            "time_end": self.simulator.time,
            "events": event_log,
            "event_records": event_records,
            "deadlocked": deadlocked,
            "setup_time_step": setup_time_step,
            "energy_kwh_step": energy_kwh_step,
            "energy_cost_step": energy_cost_step,
            "carbon_kg_step": carbon_kg_step,
        }

    def _select_local_dispatch_option(
        self,
        dispatch_options: Optional[List[DispatchOption]] = None,
    ) -> Optional[DispatchOption]:
        if dispatch_options is None:
            dispatch_options = self.simulator.get_dispatch_options()
        if not dispatch_options:
            return None

        for cell_id in sorted(self.assembly.cells):
            cell_options = [
                opt
                for opt in dispatch_options
                if self.assembly.machine_to_cell[opt.machine_id] == cell_id
            ]
            if not cell_options:
                continue
            policy = self.cell_policies[cell_id]
            local_action = policy.select_action(
                np.zeros(1, dtype=np.float32),
                {"dispatch_options": [self._option_to_dict(opt) for opt in cell_options]},
            )
            if 0 <= local_action < len(cell_options):
                return cell_options[local_action]
            return cell_options[0]
        return None

    def _dispatch_options_for_cell(self, cell_id: int) -> List[DispatchOption]:
        return [
            option
            for option in self.simulator.get_dispatch_options()
            if self.assembly.machine_to_cell.get(option.machine_id) == int(cell_id)
        ]

    def dispatch_job_to_machine(self, job_id: int, machine_id: int) -> bool:
        return self.simulator.dispatch(job_id, machine_id)

    def _valid_release_cells(self) -> List[int]:
        valid: List[int] = []
        for cell_id, cell in self.assembly.cells.items():
            if not self._backlogs[cell_id]:
                continue
            feedback = self._cell_feedback(cell_id)
            if feedback.active_jobs >= cell.budget.wip_cap:
                continue
            valid.append(cell_id)
        return valid

    def _cell_feedback(self, cell_id: int) -> CellFeedback:
        cell = self.assembly.cells[cell_id]
        cell_job_ids = set(cell.template_job_ids)
        ready_jobs = [jid for jid in self.simulator.ready_jobs if jid in cell_job_ids]
        processing_jobs = [
            jid
            for jid in (machine.current_job for machine in self.simulator.machines.values())
            if jid is not None and jid in cell_job_ids
        ]
        active_jobs = len(set(ready_jobs).union(processing_jobs))
        completed_jobs = len([jid for jid in self.simulator.completed_jobs if jid in cell_job_ids])
        busy_machines = sum(
            1
            for mid in cell.machine_ids
            if self.simulator.machines[mid].status in {"busy", "setup"}
        )
        down_machines = sum(
            1 for mid in cell.machine_ids if self.simulator.machines[mid].status == "down"
        )

        remaining_wip = max(0, cell.budget.wip_cap - active_jobs)
        remaining_energy = max(0.0, cell.budget.energy_cap_kwh - self._cell_energy_kwh[cell_id])
        remaining_carbon = max(0.0, cell.budget.carbon_cap_kg - self._cell_carbon_kg[cell_id])
        energy_overshoot = max(0.0, self._cell_energy_kwh[cell_id] - cell.budget.energy_cap_kwh)
        carbon_overshoot = max(0.0, self._cell_carbon_kg[cell_id] - cell.budget.carbon_cap_kg)

        return CellFeedback(
            cell_id=cell_id,
            label=cell.label,
            family_key=cell.family_key,
            backlog_jobs=len(self._backlogs[cell_id]),
            released_jobs=len(cell.template_job_ids) - len(self._backlogs[cell_id]),
            active_jobs=active_jobs,
            completed_jobs=completed_jobs,
            ready_jobs=len(ready_jobs),
            inbound_jobs=0,
            busy_machines=busy_machines,
            down_machines=down_machines,
            energy_kwh=self._cell_energy_kwh[cell_id],
            energy_cost=self._cell_energy_cost[cell_id],
            carbon_kg=self._cell_carbon_kg[cell_id],
            budget_cap_energy_kwh=cell.budget.energy_cap_kwh,
            budget_cap_carbon_kg=cell.budget.carbon_cap_kg,
            budget_cap_wip=cell.budget.wip_cap,
            inbound_buffer_cap=cell.inbound_buffer_cap,
            budget_remaining_energy_kwh=remaining_energy,
            budget_remaining_carbon_kg=remaining_carbon,
            budget_remaining_wip=remaining_wip,
            inbound_buffer_remaining=cell.inbound_buffer_cap,
            energy_overshoot_kwh=energy_overshoot,
            carbon_overshoot_kg=carbon_overshoot,
            processing_time_scale=cell.processing_time_scale,
        )

    def _build_observation(self) -> np.ndarray:
        total_jobs = max(1, self.assembly.total_job_count)
        horizon = max(1.0, float(self.config.horizon_hours))
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[0] = float(min(1.0, self.simulator.time / horizon))
        obs[1] = float(min(1.0, len(self._released_jobs) / total_jobs))
        obs[2] = float(min(1.0, len(self.simulator.completed_jobs) / total_jobs))

        for idx, cell_id in enumerate(sorted(self.assembly.cells)):
            feedback = self._cell_feedback(cell_id)
            base = 3 + idx * 6
            machine_count = max(1, len(self.assembly.cells[cell_id].machine_ids))
            obs[base] = float(min(1.0, feedback.backlog_jobs / total_jobs))
            obs[base + 1] = float(min(1.0, feedback.active_jobs / max(1, self.assembly.cells[cell_id].budget.wip_cap)))
            obs[base + 2] = float(min(1.0, feedback.busy_machines / machine_count))
            obs[base + 3] = float(min(1.0, feedback.down_machines / machine_count))
            obs[base + 4] = float(
                min(
                    1.0,
                    feedback.energy_kwh / max(1.0, self.assembly.cells[cell_id].budget.energy_cap_kwh),
                )
            )
            obs[base + 5] = float(
                min(
                    1.0,
                    feedback.carbon_kg / max(1.0, self.assembly.cells[cell_id].budget.carbon_cap_kg),
                )
            )
        return obs

    def _build_cell_observation(self, cell_id: int) -> np.ndarray:
        cell = self.assembly.cells[cell_id]
        feedback = self._cell_feedback(cell_id)
        machine_count = max(1, len(cell.machine_ids))
        obs = np.zeros(8, dtype=np.float32)
        obs[0] = float(min(1.0, feedback.backlog_jobs / max(1, self.assembly.total_job_count)))
        obs[1] = float(min(1.0, feedback.active_jobs / max(1, cell.budget.wip_cap)))
        obs[2] = float(min(1.0, feedback.completed_jobs / max(1, len(cell.template_job_ids))))
        obs[3] = float(min(1.0, feedback.busy_machines / machine_count))
        obs[4] = float(min(1.0, feedback.down_machines / machine_count))
        obs[5] = float(min(1.0, feedback.energy_kwh / max(1.0, cell.budget.energy_cap_kwh)))
        obs[6] = float(min(1.0, feedback.carbon_kg / max(1.0, cell.budget.carbon_cap_kg)))
        obs[7] = float(min(1.0, self.simulator.time / max(1.0, float(self.config.horizon_hours))))
        return obs

    def _typed_events(self, transition: Dict[str, Any]) -> List[HolonicEvent]:
        events: List[HolonicEvent] = []
        event_records = list(transition.get("event_records", []))
        if not event_records:
            event_records = [
                {
                    "time": float(transition["time_end"]),
                    "kind": kind,
                    "payload": {},
                }
                for kind in transition["events"]
            ]

        for item in event_records:
            kind = str(item.get("kind", ""))
            raw_payload = dict(item.get("payload", {}))
            if kind == "finish":
                event_kind = "operation_finish"
                scope = "cell"
            elif kind == "arrival":
                event_kind = "job_arrival"
                scope = "plant"
            elif kind == "shared_backlog_arrival":
                event_kind = "shared_backlog_arrival"
                scope = "plant"
            elif kind == "setup_start":
                event_kind = "machine_setup_start"
                scope = "resource"
            elif kind == "setup_complete":
                event_kind = "machine_setup_complete"
                scope = "resource"
            elif kind == "breakdown":
                event_kind = "machine_breakdown"
                scope = "resource"
            elif kind == "repair":
                event_kind = "machine_repair"
                scope = "resource"
            else:
                event_kind = str(kind)
                scope = "des"
            payload = self._public_event_payload(kind, raw_payload)
            events.append(
                HolonicEvent(
                    time=float(item.get("time", transition["time_end"])),
                    kind=event_kind,
                    scope=scope,
                    payload=payload,
                )
            )
        return events

    def _public_event_payload(self, kind: str, payload: Dict[str, Any]) -> Dict[str, int | float | str]:
        public_payload = {
            key: value
            for key, value in payload.items()
            if key != "token"
        }

        machine_id = public_payload.get("machine_id")
        if isinstance(machine_id, int):
            cell_id = self.assembly.machine_to_cell.get(machine_id)
            if cell_id is not None:
                public_payload["cell_id"] = int(cell_id)

        job_id = public_payload.get("job_id")
        if isinstance(job_id, int) and "cell_id" not in public_payload:
            cell_id = self.assembly.job_to_cell.get(job_id)
            if cell_id is not None:
                public_payload["cell_id"] = int(cell_id)

        if kind == "arrival" and isinstance(job_id, int):
            public_payload.setdefault("job_id", int(job_id))
        if kind == "finish":
            if isinstance(job_id, int):
                public_payload["job_id"] = int(job_id)
            if isinstance(public_payload.get("op_idx"), int):
                public_payload["op_idx"] = int(public_payload["op_idx"])
        if kind in {"setup_start", "setup_complete", "breakdown", "repair"} and isinstance(machine_id, int):
            public_payload["machine_id"] = int(machine_id)

        normalized: Dict[str, int | float | str] = {}
        for key, value in public_payload.items():
            if isinstance(value, (int, float, str)):
                normalized[str(key)] = value
        return normalized

    def _typed_messages(self, contract: Optional[PlantContract]) -> List[HolonicMessage]:
        if contract is None:
            return []
        return [
            HolonicMessage(
                sender="plant",
                recipient=f"cell_{contract.cell_id}",
                kind="release_contract",
                payload={
                    "contract_id": contract.contract_id,
                    "job_id": contract.job_id,
                    "cell_id": contract.cell_id,
                    "released_at": contract.released_at,
                    "due_time": contract.due_time,
                    "horizon_remaining": contract.horizon_remaining,
                },
            )
        ]

    def _build_info(
        self,
        transition: Dict[str, Any],
        invalid_action: bool,
        contract: Optional[PlantContract],
    ) -> Dict[str, Any]:
        cell_summaries = {
            cell_id: self._cell_feedback(cell_id).to_dict()
            for cell_id in sorted(self.assembly.cells)
        }
        typed_events = self._typed_events(transition)
        typed_messages = self._typed_messages(contract)
        return {
            "time": self.simulator.time,
            "events": list(transition["events"]),
            "deadlocked": bool(transition["deadlocked"]),
            "invalid_action": invalid_action,
            "invalid_actions_total": self._invalid_actions,
            "valid_cells": self._valid_release_cells(),
            "action_mask": self.action_mask("plant").tolist(),
            "agents_to_act": self.agents_to_act(),
            "last_contract": None if contract is None else contract.__dict__.copy(),
            "typed_events": [event.to_dict() for event in typed_events],
            "messages": [message.to_dict() for message in typed_messages],
            "plant": {
                "released_jobs_total": len(self._released_jobs),
                "completed_jobs_total": len(self.simulator.completed_jobs),
                "backlog_jobs_total": sum(len(backlog) for backlog in self._backlogs.values()),
                "busy_machines_total": sum(
                    1
                    for machine in self.simulator.machines.values()
                    if machine.status in {"busy", "setup"}
                ),
                "setup_machines_total": sum(
                    1 for machine in self.simulator.machines.values() if machine.status == "setup"
                ),
                "down_machines_total": sum(
                    1 for machine in self.simulator.machines.values() if machine.status == "down"
                ),
            },
            "cell_summaries": cell_summaries,
            "machine_summaries": self._machine_summaries(),
            "setup_time_step": float(transition.get("setup_time_step", 0.0)),
            "setup_time_total": float(sum(self._cell_setup_time.values())),
            "setup_energy_kwh_total": float(sum(self._cell_setup_energy_kwh.values())),
            "setup_profile_id": self.config.setup_profile_id,
            "energy_kwh_step": float(transition["energy_kwh_step"]),
            "energy_cost_step": float(transition["energy_cost_step"]),
            "carbon_kg_step": float(transition["carbon_kg_step"]),
        }

    @staticmethod
    def _option_to_dict(option: DispatchOption) -> Dict[str, int | float]:
        return {
            "job_id": option.job_id,
            "machine_id": option.machine_id,
            "proc_time": option.proc_time,
        }

    def _dispatch_option_to_public_dict(self, option: DispatchOption) -> Dict[str, int | float | str]:
        job = self.simulator.jobs.get(option.job_id)
        op_idx = int(job.current_op_idx) if job is not None else 0
        cell_id = self.assembly.machine_to_cell.get(option.machine_id)
        return {
            "operation_id": f"job_{option.job_id}_op_{op_idx}_machine_{option.machine_id}",
            "job_id": int(option.job_id),
            "machine_id": int(option.machine_id),
            "cell_id": None if cell_id is None else int(cell_id),
            "op_idx": int(op_idx),
            "proc_time": float(option.proc_time),
        }

    def _machine_summaries(self) -> Dict[str, Dict[str, Any]]:
        summaries: Dict[str, Dict[str, Any]] = {}
        for machine_id in sorted(self.simulator.machines):
            machine = self.simulator.machines[machine_id]
            cell_id = self.assembly.machine_to_cell.get(machine_id)
            summaries[str(machine_id)] = {
                "machine_id": int(machine_id),
                "cell_id": None if cell_id is None else int(cell_id),
                "status": str(machine.status),
                "current_job": None if machine.current_job is None else int(machine.current_job),
                "remaining_time": float(machine.remaining_time),
                "setup_remaining_time": float(machine.setup_remaining_time),
                "down_time": float(machine.down_time),
                "last_setup_family": "" if machine.last_setup_family is None else str(machine.last_setup_family),
                "target_setup_family": "" if machine.target_setup_family is None else str(machine.target_setup_family),
                "idle_power_kw": float(machine.idle_power_kw),
                "active_power_kw": float(machine.active_power_kw),
                "setup_power_kw": float(machine.setup_power_kw),
            }
        return summaries

    def _is_done(self) -> bool:
        return (
            all(len(backlog) == 0 for backlog in self._backlogs.values())
            and len(self.simulator.completed_jobs) >= self.assembly.total_job_count
        )

    def render(self) -> Dict[str, Any]:
        return self._build_info(self._last_transition, invalid_action=False, contract=self._last_contract)

    def close(self) -> None:
        return None
