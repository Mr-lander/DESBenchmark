from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .external_controller_adapter import (
    ExternalControllerSession,
    ExternalDecisionDriverLike,
    _reset_external_driver_episode,
)


_ROLE_DIMS = {
    "plant": 6,
    "area": 11,
    "cell": 1,
}


@dataclass(frozen=True)
class DecisionSupervisionExample:
    role: str
    agent_id: str
    candidate_action_ids: tuple[int, ...]
    candidate_features: tuple[tuple[float, ...], ...]
    selected_action_id: int

    @property
    def selected_action_index(self) -> int:
        return self.candidate_action_ids.index(self.selected_action_id)


@dataclass
class BehaviorClonedTreeDriver:
    """Role-conditioned learned scorer over the external legal boundary."""

    role_weights: dict[str, np.ndarray]
    controller_name: str = "bc_tree_driver"
    uses_external_session: bool = True

    def select_actions(self, decision: Mapping[str, object]) -> dict[str, int]:
        actions: dict[str, int] = {}
        for agent_id, agent_view in dict(decision["active_agents"]).items():
            candidate_ids, candidate_features = _candidate_encoding(dict(agent_view))
            if not candidate_ids:
                actions[str(agent_id)] = 0
                continue
            role = str(agent_view.get("role", "cell"))
            scores = self._score_candidates(role, candidate_features)
            best_index = int(np.argmax(scores))
            actions[str(agent_id)] = int(candidate_ids[best_index])
        return actions

    @property
    def name(self) -> str:
        return self.controller_name

    def _score_candidates(
        self,
        role: str,
        candidate_features: Sequence[Sequence[float]],
    ) -> np.ndarray:
        weights = self.role_weights.get(role)
        if weights is None:
            return np.zeros(len(candidate_features), dtype=np.float32)
        features = np.asarray(candidate_features, dtype=np.float32)
        return features @ weights.astype(np.float32)


@dataclass
class MLPBehaviorClonedTreeDriver:
    """Role-conditioned nonlinear scorer over the external legal boundary."""

    role_hidden_weights: dict[str, np.ndarray]
    role_hidden_bias: dict[str, np.ndarray]
    role_output_weights: dict[str, np.ndarray]
    role_output_bias: dict[str, float]
    controller_name: str = "mlp_bc_tree_driver"
    uses_external_session: bool = True

    def select_actions(self, decision: Mapping[str, object]) -> dict[str, int]:
        actions: dict[str, int] = {}
        for agent_id, agent_view in dict(decision["active_agents"]).items():
            candidate_ids, candidate_features = _candidate_encoding(dict(agent_view))
            if not candidate_ids:
                actions[str(agent_id)] = 0
                continue
            role = str(agent_view.get("role", "cell"))
            scores = self._score_candidates(role, candidate_features)
            best_index = int(np.argmax(scores))
            actions[str(agent_id)] = int(candidate_ids[best_index])
        return actions

    @property
    def name(self) -> str:
        return self.controller_name

    def _score_candidates(
        self,
        role: str,
        candidate_features: Sequence[Sequence[float]],
    ) -> np.ndarray:
        hidden_weights = self.role_hidden_weights.get(role)
        hidden_bias = self.role_hidden_bias.get(role)
        output_weights = self.role_output_weights.get(role)
        output_bias = self.role_output_bias.get(role)
        if hidden_weights is None or hidden_bias is None or output_weights is None or output_bias is None:
            return np.zeros(len(candidate_features), dtype=np.float32)
        features = np.asarray(candidate_features, dtype=np.float32)
        hidden = np.tanh(features @ hidden_weights.astype(np.float32) + hidden_bias.astype(np.float32))
        return hidden @ output_weights.astype(np.float32) + np.float32(output_bias)


def collect_behavior_cloning_examples(
    *,
    teacher: ExternalDecisionDriverLike | Callable[[Mapping[str, object]], Mapping[str, int]],
    env_factory: Callable[[int], Any],
    seeds: Iterable[int],
    max_decisions: int | None = None,
) -> list[DecisionSupervisionExample]:
    """Collect legal teacher decisions from the external benchmark boundary."""

    examples: list[DecisionSupervisionExample] = []
    total_decisions = 0
    max_count = None if max_decisions is None else int(max_decisions)

    for seed in seeds:
        env = env_factory(int(seed))
        try:
            session = ExternalControllerSession(env)
            _reset_external_driver_episode(teacher)
            decision = session.reset(seed=int(seed))
            while not decision["terminated"] and not decision["truncated"]:
                actions = _call_teacher(teacher, decision)
                for agent_id, agent_view in dict(decision["active_agents"]).items():
                    candidate_ids, candidate_features = _candidate_encoding(dict(agent_view))
                    if not candidate_ids:
                        continue
                    selected_action = int(actions[str(agent_id)])
                    if selected_action not in candidate_ids:
                        raise ValueError(
                            f"Teacher selected illegal action {selected_action} for {agent_id}; "
                            f"valid={candidate_ids}"
                        )
                    examples.append(
                        DecisionSupervisionExample(
                            role=str(agent_view.get("role", "cell")),
                            agent_id=str(agent_id),
                            candidate_action_ids=tuple(int(item) for item in candidate_ids),
                            candidate_features=tuple(
                                tuple(float(value) for value in feature_row)
                                for feature_row in candidate_features
                            ),
                            selected_action_id=selected_action,
                        )
                    )
                    total_decisions += 1
                    if max_count is not None and total_decisions >= max_count:
                        return examples
                transition = session.step(actions)
                decision = transition["current_decision"]
        finally:
            env.close()
    return examples


def fit_behavior_cloned_tree_driver(
    examples: Sequence[DecisionSupervisionExample],
    *,
    epochs: int = 200,
    lr: float = 0.05,
    weight_decay: float = 0.0,
    controller_name: str = "bc_tree_driver",
) -> BehaviorClonedTreeDriver:
    """Fit a role-conditioned linear scorer from teacher decisions."""

    if not examples:
        raise ValueError("Cannot fit a behavior-cloned tree driver from an empty dataset")

    role_weights: dict[str, np.ndarray] = {}
    for role, dim in _ROLE_DIMS.items():
        role_weights[role] = np.zeros(dim, dtype=np.float32)

    for _ in range(int(epochs)):
        gradients = {
            role: np.zeros_like(weights, dtype=np.float32)
            for role, weights in role_weights.items()
        }
        counts = {role: 0 for role in _ROLE_DIMS}
        for example in examples:
            role = example.role
            features = np.asarray(example.candidate_features, dtype=np.float32)
            weights = role_weights[role]
            logits = features @ weights
            probabilities = _softmax(logits)
            grad_logits = probabilities
            grad_logits[example.selected_action_index] -= 1.0
            gradients[role] += features.T @ grad_logits
            counts[role] += 1

        for role, weights in role_weights.items():
            if counts[role] == 0:
                continue
            grad = gradients[role] / float(counts[role])
            if weight_decay:
                grad = grad + float(weight_decay) * weights
            role_weights[role] = weights - float(lr) * grad.astype(np.float32)

    return BehaviorClonedTreeDriver(
        role_weights=role_weights,
        controller_name=controller_name,
    )


def fit_mlp_behavior_cloned_tree_driver(
    examples: Sequence[DecisionSupervisionExample],
    *,
    hidden_dim: int = 16,
    epochs: int = 300,
    lr: float = 0.03,
    weight_decay: float = 0.0,
    seed: int = 0,
    controller_name: str = "mlp_bc_tree_driver",
) -> MLPBehaviorClonedTreeDriver:
    """Fit a role-conditioned one-hidden-layer scorer from teacher decisions."""

    if not examples:
        raise ValueError("Cannot fit an MLP behavior-cloned tree driver from an empty dataset")

    hidden_size = int(hidden_dim)
    if hidden_size <= 0:
        raise ValueError("hidden_dim must be positive")

    rng = np.random.default_rng(int(seed))
    role_hidden_weights: dict[str, np.ndarray] = {}
    role_hidden_bias: dict[str, np.ndarray] = {}
    role_output_weights: dict[str, np.ndarray] = {}
    role_output_bias: dict[str, float] = {}

    for role, dim in _ROLE_DIMS.items():
        scale_in = np.sqrt(2.0 / max(1.0, float(dim + hidden_size)))
        scale_out = np.sqrt(2.0 / max(1.0, float(hidden_size + 1)))
        role_hidden_weights[role] = (rng.standard_normal((dim, hidden_size)) * scale_in).astype(np.float32)
        role_hidden_bias[role] = np.zeros(hidden_size, dtype=np.float32)
        role_output_weights[role] = (rng.standard_normal(hidden_size) * scale_out).astype(np.float32)
        role_output_bias[role] = 0.0

    for _ in range(int(epochs)):
        grad_hidden_weights = {
            role: np.zeros_like(weights, dtype=np.float32)
            for role, weights in role_hidden_weights.items()
        }
        grad_hidden_bias = {
            role: np.zeros_like(bias, dtype=np.float32)
            for role, bias in role_hidden_bias.items()
        }
        grad_output_weights = {
            role: np.zeros_like(weights, dtype=np.float32)
            for role, weights in role_output_weights.items()
        }
        grad_output_bias = {role: 0.0 for role in _ROLE_DIMS}
        counts = {role: 0 for role in _ROLE_DIMS}

        for example in examples:
            role = example.role
            features = np.asarray(example.candidate_features, dtype=np.float32)
            hidden_weights = role_hidden_weights[role]
            hidden_bias = role_hidden_bias[role]
            output_weights = role_output_weights[role]
            output_bias = role_output_bias[role]

            preact = features @ hidden_weights + hidden_bias
            hidden = np.tanh(preact)
            logits = hidden @ output_weights + np.float32(output_bias)
            probabilities = _softmax(logits)
            grad_logits = probabilities
            grad_logits[example.selected_action_index] -= 1.0

            grad_output_weights[role] += hidden.T @ grad_logits
            grad_output_bias[role] += float(np.sum(grad_logits))

            grad_hidden = np.outer(grad_logits, output_weights)
            grad_preact = grad_hidden * (1.0 - np.square(hidden))
            grad_hidden_weights[role] += features.T @ grad_preact
            grad_hidden_bias[role] += np.sum(grad_preact, axis=0).astype(np.float32)
            counts[role] += 1

        for role in _ROLE_DIMS:
            if counts[role] == 0:
                continue
            scale = 1.0 / float(counts[role])

            hidden_weight_grad = grad_hidden_weights[role] * scale
            hidden_bias_grad = grad_hidden_bias[role] * scale
            output_weight_grad = grad_output_weights[role] * scale
            output_bias_grad = grad_output_bias[role] * scale

            if weight_decay:
                hidden_weight_grad = hidden_weight_grad + float(weight_decay) * role_hidden_weights[role]
                output_weight_grad = output_weight_grad + float(weight_decay) * role_output_weights[role]

            role_hidden_weights[role] = role_hidden_weights[role] - float(lr) * hidden_weight_grad.astype(np.float32)
            role_hidden_bias[role] = role_hidden_bias[role] - float(lr) * hidden_bias_grad.astype(np.float32)
            role_output_weights[role] = role_output_weights[role] - float(lr) * output_weight_grad.astype(np.float32)
            role_output_bias[role] = float(role_output_bias[role] - float(lr) * output_bias_grad)

    return MLPBehaviorClonedTreeDriver(
        role_hidden_weights=role_hidden_weights,
        role_hidden_bias=role_hidden_bias,
        role_output_weights=role_output_weights,
        role_output_bias=role_output_bias,
        controller_name=controller_name,
    )


def _call_teacher(
    teacher: ExternalDecisionDriverLike | Callable[[Mapping[str, object]], Mapping[str, int]],
    decision: Mapping[str, object],
) -> Mapping[str, int]:
    if hasattr(teacher, "select_actions"):
        return teacher.select_actions(decision)
    return teacher(decision)


def _candidate_encoding(agent_view: Mapping[str, object]) -> tuple[list[int], list[list[float]]]:
    role = str(agent_view.get("role", "cell"))
    if role == "plant":
        return _plant_candidate_encoding(agent_view)
    if role == "area":
        return _area_candidate_encoding(agent_view)
    return _cell_candidate_encoding(agent_view)


def _plant_candidate_encoding(agent_view: Mapping[str, object]) -> tuple[list[int], list[list[float]]]:
    valid_areas = [int(area_id) for area_id in agent_view.get("valid_areas", _mask_valid_ids(agent_view))]
    area_summaries = {
        int(area_id): dict(summary)
        for area_id, summary in dict(agent_view.get("visible_area_summaries", agent_view.get("area_summaries", {}))).items()
    }
    features: list[list[float]] = []
    for area_id in valid_areas:
        summary = area_summaries.get(area_id, {})
        cell_ids = summary.get("cell_ids", [])
        features.append(
            [
                1.0,
                float(summary.get("active_jobs", 0.0)),
                float(summary.get("backlog_jobs", 0.0)),
                float(summary.get("energy_kwh", 0.0)) / 1000.0,
                float(summary.get("carbon_kg", 0.0)) / 1000.0,
                float(len(cell_ids)),
            ]
        )
    return valid_areas, features


def _area_candidate_encoding(agent_view: Mapping[str, object]) -> tuple[list[int], list[list[float]]]:
    valid_cells = [int(cell_id) for cell_id in agent_view.get("valid_cells", _mask_valid_ids(agent_view))]
    cell_summaries = {
        int(cell_id): dict(summary)
        for cell_id, summary in dict(agent_view.get("visible_cell_summaries", agent_view.get("cell_summaries", {}))).items()
    }
    features: list[list[float]] = []
    for cell_id in valid_cells:
        summary = cell_summaries.get(cell_id, {})
        energy_cap = max(1.0, float(summary.get("budget_cap_energy_kwh", 1.0)))
        carbon_cap = max(1.0, float(summary.get("budget_cap_carbon_kg", 1.0)))
        wip_cap = max(1.0, float(summary.get("budget_cap_wip", 1.0)))
        proc_scale = float(summary.get("processing_time_scale", 1.0))
        energy = float(summary.get("energy_kwh", 0.0))
        carbon = float(summary.get("carbon_kg", 0.0))
        remaining_energy = max(1.0, energy_cap - energy)
        remaining_carbon = max(1.0, carbon_cap - carbon)
        active_jobs = float(summary.get("active_jobs", 0.0))
        features.append(
            [
                1.0,
                proc_scale,
                active_jobs,
                float(summary.get("busy_machines", 0.0)),
                float(summary.get("down_machines", 0.0)),
                float(summary.get("backlog_jobs", 0.0)),
                energy / energy_cap,
                carbon / carbon_cap,
                active_jobs / wip_cap,
                (120.0 * proc_scale) / remaining_energy,
                (72.0 * proc_scale) / remaining_carbon,
            ]
        )
    return valid_cells, features


def _cell_candidate_encoding(agent_view: Mapping[str, object]) -> tuple[list[int], list[list[float]]]:
    valid_actions = _mask_valid_ids(agent_view)
    if not valid_actions:
        return [], []
    return valid_actions, [[1.0] for _ in valid_actions]


def _mask_valid_ids(agent_view: Mapping[str, object]) -> list[int]:
    return [
        int(index)
        for index, enabled in enumerate(list(agent_view.get("action_mask", [])))
        if int(enabled) > 0
    ]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    weights = np.exp(shifted, dtype=np.float32)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full_like(weights, fill_value=1.0 / len(weights), dtype=np.float32)
    return (weights / total).astype(np.float32)
