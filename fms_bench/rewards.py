"""Reward decomposition for DESBench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import BenchmarkConfig
from .models import Job


@dataclass
class RewardComponents:
    """Decomposed reward components for interpretability."""

    time_penalty: float = 0.0
    makespan_bonus: float = 0.0
    tardiness_penalty: float = 0.0
    energy_penalty: float = 0.0
    carbon_penalty: float = 0.0
    invalid_action_penalty: float = 0.0
    deadlock_penalty: float = 0.0

    def total(self) -> float:
        return (
            self.time_penalty
            + self.makespan_bonus
            + self.tardiness_penalty
            + self.energy_penalty
            + self.carbon_penalty
            + self.invalid_action_penalty
            + self.deadlock_penalty
        )


class RewardModel:
    """Compute decomposed rewards for DESBench."""

    def __init__(
        self,
        config: BenchmarkConfig,
        time_weight: float = 1.0,
        makespan_weight: float = 1.0,
        tardiness_weight: float = 0.5,
        energy_weight: float = 0.01,
        carbon_weight: float = 0.001,
        invalid_action_weight: float = 0.1,
        deadlock_weight: float = 10.0,
    ) -> None:
        self.config = config
        self.time_weight = time_weight
        self.makespan_weight = makespan_weight
        self.tardiness_weight = tardiness_weight
        self.energy_weight = energy_weight
        self.carbon_weight = carbon_weight
        self.invalid_action_weight = invalid_action_weight
        self.deadlock_weight = deadlock_weight

    def compute(
        self,
        delta_t: float,
        energy_kwh: float,
        carbon_kg: float,
        invalid_action: bool = False,
        deadlocked: bool = False,
        completed_jobs: Optional[list[Job]] = None,
        all_jobs: Optional[list[Job]] = None,
        terminated: bool = False,
    ) -> RewardComponents:
        """Compute decomposed reward for a step."""
        components = RewardComponents()

        # Time penalty (negative reward for time elapsed)
        components.time_penalty = -self.time_weight * delta_t

        # Energy penalty
        components.energy_penalty = -self.energy_weight * energy_kwh

        # Carbon penalty
        components.carbon_penalty = -self.carbon_weight * carbon_kg

        # Invalid action penalty
        if invalid_action:
            components.invalid_action_penalty = -self.invalid_action_weight

        # Deadlock penalty
        if deadlocked:
            components.deadlock_penalty = -self.deadlock_weight

        # Terminal rewards
        if terminated and completed_jobs is not None and all_jobs is not None:
            # Makespan bonus (negative makespan is better)
            if completed_jobs:
                makespan = max(job.release_time for job in completed_jobs) if completed_jobs else 0.0
                # Normalize by horizon
                horizon = float(self.config.horizon_hours)
                components.makespan_bonus = -self.makespan_weight * (makespan / horizon)

            # Tardiness penalty
            tardiness = 0.0
            for job in completed_jobs:
                if job.due_time is not None:
                    completion_time = job.release_time  # Simplified; actual completion time should be tracked
                    if completion_time > job.due_time:
                        tardiness += completion_time - job.due_time
            components.tardiness_penalty = -self.tardiness_weight * (tardiness / max(1.0, horizon))

        return components

    def compute_step_reward(
        self,
        delta_t: float,
        energy_kwh: float,
        carbon_kg: float,
        invalid_action: bool = False,
        deadlocked: bool = False,
    ) -> float:
        """Simple step reward without terminal bonuses."""
        components = self.compute(
            delta_t=delta_t,
            energy_kwh=energy_kwh,
            carbon_kg=carbon_kg,
            invalid_action=invalid_action,
            deadlocked=deadlocked,
            terminated=False,
        )
        return components.total()

    def compute_terminal_reward(
        self,
        completed_jobs: list[Job],
        all_jobs: list[Job],
        makespan: float,
        total_tardiness: float,
    ) -> RewardComponents:
        """Compute terminal reward components."""
        horizon = float(self.config.horizon_hours)
        components = RewardComponents()

        # Makespan bonus
        components.makespan_bonus = -self.makespan_weight * (makespan / horizon)

        # Tardiness penalty
        components.tardiness_penalty = -self.tardiness_weight * (total_tardiness / max(1.0, horizon))

        return components


def default_reward_model(config: BenchmarkConfig) -> RewardModel:
    """Create a default reward model with balanced weights."""
    return RewardModel(
        config=config,
        time_weight=1.0,
        makespan_weight=1.0,
        tardiness_weight=0.5,
        energy_weight=0.01,
        carbon_weight=0.001,
        invalid_action_weight=0.1,
        deadlock_weight=10.0,
    )
