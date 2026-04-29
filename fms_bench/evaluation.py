"""Evaluation protocol for DESBench."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import numpy as np

from .config import BenchmarkConfig
from .env import IndustrialFMS
from .models import Job


@dataclass
class EpisodeMetrics:
    """Metrics collected during an episode."""

    episode_id: int
    seed: int
    makespan: float
    total_tardiness: float
    total_energy_kwh: float
    total_energy_cost: float
    total_carbon_kg: float
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    deadlocked: bool
    invalid_actions: int = 0
    job_completion_times: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "makespan": self.makespan,
            "total_tardiness": self.total_tardiness,
            "total_energy_kwh": self.total_energy_kwh,
            "total_energy_cost": self.total_energy_cost,
            "total_carbon_kg": self.total_carbon_kg,
            "total_reward": self.total_reward,
            "steps": self.steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "deadlocked": self.deadlocked,
            "invalid_actions": self.invalid_actions,
            "job_completion_times": self.job_completion_times,
        }


@dataclass
class EvaluationResults:
    """Aggregated evaluation results."""

    instance_path: str
    config: BenchmarkConfig
    episodes: List[EpisodeMetrics]
    policy_name: str = "unknown"

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not self.episodes:
            return {}

        makespans = [e.makespan for e in self.episodes]
        rewards = [e.total_reward for e in self.episodes]
        energies = [e.total_energy_kwh for e in self.episodes]
        carbons = [e.total_carbon_kg for e in self.episodes]

        return {
            "instance": self.instance_path,
            "policy": self.policy_name,
            "num_episodes": len(self.episodes),
            "makespan_mean": float(np.mean(makespans)),
            "makespan_std": float(np.std(makespans)),
            "makespan_min": float(np.min(makespans)),
            "makespan_max": float(np.max(makespans)),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "energy_mean": float(np.mean(energies)),
            "carbon_mean": float(np.mean(carbons)),
            "success_rate": sum(1 for e in self.episodes if e.terminated and not e.deadlocked) / len(self.episodes),
            "deadlock_rate": sum(1 for e in self.episodes if e.deadlocked) / len(self.episodes),
            "truncation_rate": sum(1 for e in self.episodes if e.truncated) / len(self.episodes),
        }

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        data = {
            "summary": self.summary(),
            "episodes": [e.to_dict() for e in self.episodes],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationResults":
        """Load results from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        # Simplified reconstruction - in production, would need full config reconstruction
        return cls(
            instance_path=data["summary"]["instance"],
            config=BenchmarkConfig(),
            episodes=[],  # Would need proper reconstruction
            policy_name=data["summary"]["policy"],
        )


class Evaluator:
    """Run evaluation episodes for DESBench."""

    def __init__(
        self,
        instance_path: str | Path,
        instance_kind: str = "brandimarte",
        config: Optional[BenchmarkConfig] = None,
    ) -> None:
        self.instance_path = Path(instance_path)
        self.instance_kind = instance_kind
        self.config = config or BenchmarkConfig()

    def run_episode(
        self,
        policy: Any,
        episode_id: int,
        seed: int,
        track_job_times: bool = True,
    ) -> EpisodeMetrics:
        """Run a single evaluation episode."""
        env = IndustrialFMS(
            instance_path=self.instance_path,
            instance_kind=self.instance_kind,
            config=self.config,
            seed=seed,
        )

        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        invalid_actions = 0
        job_completion_times: Dict[int, float] = {}

        while True:
            # Get action from policy
            if hasattr(policy, "select_action"):
                action = policy.select_action(obs, info)
            elif hasattr(policy, "act"):
                action = policy.act(obs, info)
            elif callable(policy):
                action = policy(obs, info)
            else:
                # Default: random action
                dispatch_options = info.get("dispatch_options", [])
                if dispatch_options:
                    action = np.random.randint(0, len(dispatch_options))
                else:
                    action = 0

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if info.get("invalid_action", False):
                invalid_actions += 1

            # Track job completion times
            if track_job_times and "dispatched" in info and info["dispatched"]:
                job_id = info["dispatched"].get("job_id")
                if job_id is not None:
                    # This is dispatch time, not completion time
                    pass

            # Check for completed jobs
            completed_job_ids = info.get("ready_jobs", [])
            # Note: would need more sophisticated tracking for actual completion times

            if terminated or truncated:
                break

        # Compute final metrics
        makespan = info.get("time", 0.0)
        total_tardiness = 0.0  # Would need due time tracking
        total_energy_kwh = info.get("energy_kwh_total", 0.0)
        total_energy_cost = info.get("energy_cost_total", 0.0)
        total_carbon_kg = info.get("carbon_total_kg", 0.0)

        env.close()

        return EpisodeMetrics(
            episode_id=episode_id,
            seed=seed,
            makespan=makespan,
            total_tardiness=total_tardiness,
            total_energy_kwh=total_energy_kwh,
            total_energy_cost=total_energy_cost,
            total_carbon_kg=total_carbon_kg,
            total_reward=total_reward,
            steps=steps,
            terminated=terminated,
            truncated=truncated,
            deadlocked=info.get("deadlocked", False),
            invalid_actions=invalid_actions,
            job_completion_times=job_completion_times,
        )

    def evaluate(
        self,
        policy: Any,
        num_episodes: int = 10,
        seeds: Optional[List[int]] = None,
        policy_name: str = "unknown",
        verbose: bool = True,
    ) -> EvaluationResults:
        """Run multiple evaluation episodes."""
        if seeds is None:
            seeds = list(range(num_episodes))

        episodes: List[EpisodeMetrics] = []

        for i, seed in enumerate(seeds):
            if verbose:
                print(f"Running episode {i + 1}/{len(seeds)} (seed={seed})...")

            metrics = self.run_episode(policy, episode_id=i, seed=seed)
            episodes.append(metrics)

            if verbose:
                summary = metrics.to_dict()
                print(
                    f"  makespan={summary['makespan']:.2f}, "
                    f"reward={summary['total_reward']:.2f}, "
                    f"energy={summary['total_energy_kwh']:.2f} kWh"
                )

        return EvaluationResults(
            instance_path=str(self.instance_path),
            config=self.config,
            episodes=episodes,
            policy_name=policy_name,
        )


def evaluate_policy(
    policy: Any,
    instance_path: str | Path,
    instance_kind: str = "brandimarte",
    config: Optional[BenchmarkConfig] = None,
    num_episodes: int = 10,
    policy_name: str = "unknown",
) -> EvaluationResults:
    """Convenience function to evaluate a policy."""
    evaluator = Evaluator(
        instance_path=instance_path,
        instance_kind=instance_kind,
        config=config,
    )
    return evaluator.evaluate(
        policy=policy,
        num_episodes=num_episodes,
        policy_name=policy_name,
    )
