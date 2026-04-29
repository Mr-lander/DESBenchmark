"""Gymnasium environment wrapper for DESBench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import BenchmarkConfig
from .constraints import CarbonTracker, EnergyModel
from .data_loader import InstanceData, load_instance
from .simulator import DispatchOption, EventDrivenSimulator


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class FMSBenchmarkEnv(gym.Env):
    """Event-driven Flexible Manufacturing System benchmark environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance_path: str | Path,
        instance_kind: str = "brandimarte",
        config: Optional[BenchmarkConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.instance_path = Path(instance_path)
        self.instance_kind = instance_kind
        self.config = config or BenchmarkConfig()
        self.instance: InstanceData = load_instance(self.instance_path, self.instance_kind)
        self.base_seed = seed

        obs_dim = 5 + 3 * max(1, self.instance.n_machines)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(max(1, self.instance.n_jobs * self.instance.n_machines))

        self.simulator = EventDrivenSimulator(self.instance, self.config)
        self.energy_model = EnergyModel(
            peak_window=self.config.energy.peak_window,
            peak_price=self.config.energy.peak_price,
            offpeak_price=self.config.energy.offpeak_price,
        )
        self.carbon_tracker = CarbonTracker(
            kg_per_kwh=self.config.carbon.kg_per_kwh,
            weekly_cap_tons=self.config.carbon.weekly_cap_tons,
        )

        self.total_energy_kwh = 0.0
        self.total_energy_cost = 0.0
        self._terminated = False
        self._truncated = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        del options

        effective_seed = seed if seed is not None else self.base_seed
        self.simulator.reset(seed=effective_seed)
        self.energy_model = EnergyModel(
            peak_window=self.config.energy.peak_window,
            peak_price=self.config.energy.peak_price,
            offpeak_price=self.config.energy.offpeak_price,
        )
        self.carbon_tracker = CarbonTracker(
            kg_per_kwh=self.config.carbon.kg_per_kwh,
            weekly_cap_tons=self.config.carbon.weekly_cap_tons,
        )
        self.total_energy_kwh = 0.0
        self.total_energy_cost = 0.0
        self._terminated = False
        self._truncated = False

        transition = self._advance_until_decision()
        obs = self._build_observation()
        info = self._build_info(transition, invalid_action=False, dispatched=None)
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._terminated or self._truncated:
            raise RuntimeError("Episode already ended. Call reset() before step().")

        action = int(action)
        dispatch_options = self.simulator.get_dispatch_options()
        invalid_action = False
        dispatched: Optional[Dict[str, int | float]] = None

        if dispatch_options:
            if 0 <= action < len(dispatch_options):
                selected = dispatch_options[action]
            else:
                selected = dispatch_options[0]
                invalid_action = True
            if self.simulator.dispatch(selected.job_id, selected.machine_id):
                dispatched = {
                    "job_id": selected.job_id,
                    "machine_id": selected.machine_id,
                    "proc_time": selected.proc_time,
                }
            else:
                invalid_action = True
        elif action != 0:
            invalid_action = True

        transition = self._advance_until_decision()

        terminated = self.simulator.is_done()
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
        if terminated:
            reward += 1.0

        obs = self._build_observation()
        info = self._build_info(transition, invalid_action=invalid_action, dispatched=dispatched)
        return obs, float(reward), terminated, truncated, info

    def _advance_until_decision(self) -> Dict[str, Any]:
        energy_kwh_step = 0.0
        energy_cost_step = 0.0
        carbon_kg_step = 0.0

        def on_advance(start: float, end: float, machines: Dict[int, Any]) -> None:
            nonlocal energy_kwh_step, energy_cost_step, carbon_kg_step
            interval = self.energy_model.interval_energy(machines, start, end)
            energy_kwh_step += interval["energy_kwh"]
            energy_cost_step += interval["energy_cost"]
            carbon_kg_step += self.carbon_tracker.record(interval["energy_kwh"])

        transition = self.simulator.run_until_decision(advance_hook=on_advance)
        transition["energy_kwh_step"] = energy_kwh_step
        transition["energy_cost_step"] = energy_cost_step
        transition["carbon_kg_step"] = carbon_kg_step

        self.total_energy_kwh += energy_kwh_step
        self.total_energy_cost += energy_cost_step
        return transition

    def _build_observation(self) -> np.ndarray:
        n_jobs = max(1, self.instance.n_jobs)
        n_machines = max(1, self.instance.n_machines)
        horizon = max(1.0, float(self.config.horizon_hours))

        machine_ids = sorted(self.simulator.machines.keys())
        machines = [self.simulator.machines[mid] for mid in machine_ids]
        busy = sum(1 for m in machines if m.status == "busy")
        down = sum(1 for m in machines if m.status == "down")

        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[0] = float(min(1.0, self.simulator.time / horizon))
        obs[1] = float(min(1.0, len(self.simulator.completed_jobs) / n_jobs))
        obs[2] = float(min(1.0, len(self.simulator.ready_jobs) / n_jobs))
        obs[3] = float(min(1.0, busy / n_machines))
        obs[4] = float(min(1.0, down / n_machines))

        for idx in range(n_machines):
            base = 5 + idx * 3
            if idx >= len(machines):
                continue
            machine = machines[idx]
            if machine.status == "idle":
                status_value = 0.0
            elif machine.status == "busy":
                status_value = 0.5
            else:
                status_value = 1.0
            obs[base] = status_value
            obs[base + 1] = float(min(1.0, machine.remaining_time / horizon))
            if machine.current_job is None:
                obs[base + 2] = 0.0
            else:
                obs[base + 2] = float((machine.current_job + 1) / (self.instance.n_jobs + 1))

        return obs

    def _build_info(
        self,
        transition: Dict[str, Any],
        invalid_action: bool,
        dispatched: Optional[Dict[str, int | float]],
    ) -> Dict[str, Any]:
        carbon_state = self.carbon_tracker.status()
        return {
            "time": self.simulator.time,
            "events": list(transition["events"]),
            "deadlocked": bool(transition["deadlocked"]),
            "invalid_action": invalid_action,
            "dispatched": dispatched,
            "ready_jobs": sorted(self.simulator.ready_jobs),
            "dispatch_options": [self._option_to_dict(opt) for opt in self.simulator.get_dispatch_options()],
            "energy_kwh_step": float(transition["energy_kwh_step"]),
            "energy_cost_step": float(transition["energy_cost_step"]),
            "carbon_kg_step": float(transition["carbon_kg_step"]),
            "energy_kwh_total": float(self.total_energy_kwh),
            "energy_cost_total": float(self.total_energy_cost),
            "carbon_total_kg": float(carbon_state["carbon_total_kg"]),
            "carbon_cap_kg": carbon_state["carbon_cap_kg"],
            "carbon_remaining_kg": carbon_state["carbon_remaining_kg"],
            "carbon_cap_exceeded": carbon_state["carbon_cap_exceeded"],
        }

    @staticmethod
    def _option_to_dict(option: DispatchOption) -> Dict[str, int | float]:
        return {
            "job_id": option.job_id,
            "machine_id": option.machine_id,
            "proc_time": option.proc_time,
        }

    def render(self) -> Dict[str, Any]:
        return {
            "time": self.simulator.time,
            "ready_jobs": sorted(self.simulator.ready_jobs),
            "completed_jobs": sorted(self.simulator.completed_jobs),
            "machines": {
                mid: {
                    "status": machine.status,
                    "current_job": machine.current_job,
                    "remaining_time": machine.remaining_time,
                }
                for mid, machine in self.simulator.machines.items()
            },
        }

    def close(self) -> None:
        return None


IndustrialFMS = FMSBenchmarkEnv
