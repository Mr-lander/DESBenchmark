from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .external_controller_adapter import ExternalControllerSession
from .learned_tree_driver import _ROLE_DIMS, _candidate_encoding, _softmax


@dataclass
class PolicyGradientTreeDriver:
    """Role-conditioned linear policy over the public external-session boundary."""

    role_weights: Mapping[str, Sequence[float] | np.ndarray]
    controller_name: str = "policy_gradient_tree_driver"
    temperature: float = 1.0
    uses_external_session: bool = True
    training_stats: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.role_weights = {
            str(role): np.asarray(values, dtype=np.float32)
            for role, values in dict(self.role_weights).items()
        }

    @property
    def name(self) -> str:
        return self.controller_name

    @property
    def framework_runtime_info(self) -> dict[str, object]:
        return {
            "framework_name": "policy_gradient_tree_driver",
            "backend_mode": "reinforce",
            "python_package": "numpy",
            "optional_dependency": "",
            "uv_project_environment": "",
            "llm_backend": "",
            "llm_model": "",
            "llm_base_url": "",
        }

    @property
    def decision_audit_payload(self) -> dict[str, object]:
        return {
            "framework_decision_audit_contract": "policy_gradient_non_llm_decision_audit",
            "decision_source_counts": {
                "llm": 0,
                "fallback_contract": 0,
                "no_legal_action": 0,
            },
            "decision_source_trace": [],
        }

    def select_actions(self, decision: Mapping[str, object]) -> dict[str, int]:
        actions: dict[str, int] = {}
        for agent_id, agent_view in dict(decision["active_agents"]).items():
            candidate_ids, candidate_features = _candidate_encoding(dict(agent_view))
            if not candidate_ids:
                actions[str(agent_id)] = 0
                continue
            role = str(dict(agent_view).get("role", "cell"))
            scores = self._score_candidates(role, candidate_features)
            best_index = int(np.argmax(scores))
            actions[str(agent_id)] = int(candidate_ids[best_index])
        return actions

    def sample_actions(
        self,
        decision: Mapping[str, object],
        *,
        rng: np.random.Generator,
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        actions: dict[str, int] = {}
        traces: list[dict[str, Any]] = []
        for agent_id, agent_view in dict(decision["active_agents"]).items():
            candidate_ids, candidate_features = _candidate_encoding(dict(agent_view))
            if not candidate_ids:
                actions[str(agent_id)] = 0
                continue
            role = str(dict(agent_view).get("role", "cell"))
            scores = self._score_candidates(role, candidate_features)
            probabilities = _softmax(scores / max(1e-6, float(self.temperature)))
            chosen_index = int(rng.choice(len(candidate_ids), p=probabilities))
            actions[str(agent_id)] = int(candidate_ids[chosen_index])
            traces.append(
                {
                    "role": role,
                    "candidate_features": np.asarray(candidate_features, dtype=np.float32),
                    "probabilities": probabilities.astype(np.float32),
                    "chosen_index": chosen_index,
                }
            )
        return actions, traces

    def save_artifact(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_payload(), ensure_ascii=True, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_artifact(cls, path: str | Path) -> "PolicyGradientTreeDriver":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            role_weights=dict(payload["role_weights"]),
            controller_name=str(payload.get("controller_name", "policy_gradient_tree_driver")),
            temperature=float(payload.get("temperature", 1.0)),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "controller_name": self.controller_name,
            "temperature": float(self.temperature),
            "role_weights": {
                role: [float(value) for value in weights]
                for role, weights in self.role_weights.items()
            },
        }

    def _score_candidates(
        self,
        role: str,
        candidate_features: Sequence[Sequence[float]],
    ) -> np.ndarray:
        weights = self.role_weights.get(role)
        if weights is None:
            return np.zeros(len(candidate_features), dtype=np.float32)
        features = np.asarray(candidate_features, dtype=np.float32)
        return features @ np.asarray(weights, dtype=np.float32)


def train_policy_gradient_tree_driver(
    *,
    env_factory: Callable[[int], Any],
    train_seeds: Sequence[int],
    epochs: int,
    episodes_per_epoch: int = 1,
    learning_rate: float = 0.02,
    gamma: float = 1.0,
    baseline_momentum: float = 0.9,
    temperature: float = 1.0,
    policy_seed: int = 0,
    max_decisions_per_episode: int | None = None,
    controller_name: str = "policy_gradient_tree_driver",
    save_path: str | Path | None = None,
) -> PolicyGradientTreeDriver:
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    if int(episodes_per_epoch) <= 0:
        raise ValueError("episodes_per_epoch must be positive")
    if not train_seeds:
        raise ValueError("train_seeds must not be empty")

    rng = np.random.default_rng(int(policy_seed))
    driver = PolicyGradientTreeDriver(
        role_weights={
            role: np.zeros(dim, dtype=np.float32)
            for role, dim in _ROLE_DIMS.items()
        },
        controller_name=controller_name,
        temperature=temperature,
    )
    role_baselines = {role: 0.0 for role in _ROLE_DIMS}
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_seed_trace: list[int] = []

    total_episodes = int(epochs) * int(episodes_per_epoch)
    for episode_index in range(total_episodes):
        seed_value = int(train_seeds[episode_index % len(train_seeds)])
        trajectory, total_return = _rollout_training_episode(
            env_factory=env_factory,
            driver=driver,
            seed=seed_value,
            gamma=float(gamma),
            rng=rng,
            max_decisions=max_decisions_per_episode,
        )
        _apply_reinforce_update(
            driver=driver,
            trajectory=trajectory,
            learning_rate=float(learning_rate),
            baseline_momentum=float(baseline_momentum),
            role_baselines=role_baselines,
        )
        episode_returns.append(float(total_return))
        episode_lengths.append(len(trajectory))
        episode_seed_trace.append(int(seed_value))

    driver.training_stats = {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "epochs": int(epochs),
        "episodes_per_epoch": int(episodes_per_epoch),
        "total_episodes": int(total_episodes),
        "learning_rate": float(learning_rate),
        "gamma": float(gamma),
        "train_seeds": [int(seed) for seed in train_seeds],
        "episode_seed_trace": episode_seed_trace,
    }
    if save_path is not None:
        driver.save_artifact(save_path)
    return driver


def load_policy_gradient_tree_driver(path: str | Path) -> PolicyGradientTreeDriver:
    return PolicyGradientTreeDriver.load_artifact(path)


def _rollout_training_episode(
    *,
    env_factory: Callable[[int], Any],
    driver: PolicyGradientTreeDriver,
    seed: int,
    gamma: float,
    rng: np.random.Generator,
    max_decisions: int | None,
) -> tuple[list[dict[str, Any]], float]:
    env = env_factory(int(seed))
    rewards: list[float] = []
    trajectory: list[dict[str, Any]] = []
    try:
        session = ExternalControllerSession(env)
        decision = session.reset(seed=int(seed))
        decision_count = 0
        while not decision["terminated"] and not decision["truncated"]:
            if max_decisions is not None and decision_count >= int(max_decisions):
                break
            actions, traces = driver.sample_actions(decision, rng=rng)
            transition = session.step(actions)
            reward = float(transition["reward"])
            rewards.append(reward)
            for trace in traces:
                trajectory.append(
                    {
                        "role": str(trace["role"]),
                        "candidate_features": trace["candidate_features"],
                        "probabilities": trace["probabilities"],
                        "chosen_index": int(trace["chosen_index"]),
                        "reward_index": len(rewards) - 1,
                    }
                )
            decision = transition["current_decision"]
            decision_count += 1
    finally:
        env.close()

    discounted_returns = _discounted_returns(rewards, gamma)
    for item in trajectory:
        item["return"] = float(discounted_returns[int(item["reward_index"])])
    return trajectory, float(sum(rewards))


def _discounted_returns(rewards: Sequence[float], gamma: float) -> list[float]:
    values = [0.0 for _ in rewards]
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + float(gamma) * running
        values[index] = running
    return values


def _apply_reinforce_update(
    *,
    driver: PolicyGradientTreeDriver,
    trajectory: Sequence[Mapping[str, Any]],
    learning_rate: float,
    baseline_momentum: float,
    role_baselines: dict[str, float],
) -> None:
    gradients = {
        role: np.zeros_like(weights, dtype=np.float32)
        for role, weights in driver.role_weights.items()
    }
    counts = {role: 0 for role in driver.role_weights}

    for item in trajectory:
        role = str(item["role"])
        features = np.asarray(item["candidate_features"], dtype=np.float32)
        probabilities = np.asarray(item["probabilities"], dtype=np.float32)
        chosen_index = int(item["chosen_index"])
        return_value = float(item["return"])
        baseline = float(role_baselines.get(role, 0.0))
        advantage = return_value - baseline

        expected_features = np.sum(features * probabilities[:, None], axis=0)
        selected_features = features[chosen_index]
        gradients[role] += (selected_features - expected_features) * np.float32(advantage)
        counts[role] += 1
        role_baselines[role] = (
            float(baseline_momentum) * baseline
            + (1.0 - float(baseline_momentum)) * return_value
        )

    for role, weights in driver.role_weights.items():
        if counts[role] == 0:
            continue
        driver.role_weights[role] = np.asarray(weights, dtype=np.float32) + (
            np.float32(learning_rate / float(counts[role])) * gradients[role]
        )


ReinforceTreeDriver = PolicyGradientTreeDriver
train_reinforce_tree_driver = train_policy_gradient_tree_driver
load_reinforce_tree_driver = load_policy_gradient_tree_driver
