"""Evaluation helpers for the holonic plant benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional
import csv
import json
import time

import numpy as np

from .config import BenchmarkConfig
from .holonic_types import PlantCellConfig
from .plant_env import HolonicPlantEnv


_STRUCTURED_VIOLATION_KEYS = (
    'energy_overshoot_magnitude_total',
    'carbon_overshoot_magnitude_total',
    'energy_overshoot_duration',
    'carbon_overshoot_duration',
    'energy_overshoot_frequency',
    'carbon_overshoot_frequency',
)
STRUCTURED_VIOLATION_SUMMARY_FIELDS = _STRUCTURED_VIOLATION_KEYS
SUITE_SUMMARY_STRUCTURED_VIOLATION_METRICS = tuple(
    f'{key}_mean' for key in STRUCTURED_VIOLATION_SUMMARY_FIELDS
)
SUITE_SUMMARY_SETTLEMENT_METRICS = (
    'settled_contract_count_mean',
    'open_contract_count_mean',
    'completed_unsettled_contract_count_mean',
)
EVALUATOR_SUMMARY_CONTRACT = 'derived_summary_not_reward_authority'
TARDINESS_METRIC_CONTRACTS = {
    'tardiness_mean': 'completed_job_tardiness_only',
    'unfinished_overdue_tardiness_mean': 'unfinished_overdue_jobs_only',
    'tardiness_with_completion_debt_mean': 'completed_tardiness_plus_unfinished_overdue_debt',
}
LEGACY_SUMMARY_METRIC_CONTRACTS = {
    'violation_magnitude_mean': 'legacy_mixed_unit_compatibility_metric',
}
DECISION_SOURCE_KEYS = (
    'llm',
    'fallback_contract',
    'no_legal_action',
)
ProgressCallback = Callable[[dict[str, Any]], None]


class ZeroFallbackViolationError(RuntimeError):
    """Raised when strict real-llm evaluation observes any fallback decision."""

    def __init__(
        self,
        *,
        policy_name: str,
        benchmark_item_id: str,
        episode_id: int,
        seed: int,
        fallback_count: int,
    ) -> None:
        self.policy_name = str(policy_name)
        self.benchmark_item_id = str(benchmark_item_id)
        self.episode_id = int(episode_id)
        self.seed = int(seed)
        self.fallback_count = int(fallback_count)
        super().__init__(
            "fallback_contract decisions are disallowed for strict real-llm evaluation: "
            f"policy={self.policy_name} "
            f"benchmark_item_id={self.benchmark_item_id} "
            f"episode_id={self.episode_id} "
            f"seed={self.seed} "
            f"fallback_count={self.fallback_count}"
        )


class DisallowedDecisionFailureError(RuntimeError):
    """Raised when response-backed evaluation sees a non-whitelisted failure kind."""

    def __init__(
        self,
        *,
        policy_name: str,
        benchmark_item_id: str,
        episode_id: int,
        seed: int,
        disallowed_failure_kinds: Iterable[str],
        allowed_failure_kinds: Iterable[str],
    ) -> None:
        self.policy_name = str(policy_name)
        self.benchmark_item_id = str(benchmark_item_id)
        self.episode_id = int(episode_id)
        self.seed = int(seed)
        self.disallowed_failure_kinds = tuple(
            sorted(str(kind) for kind in disallowed_failure_kinds)
        )
        self.allowed_failure_kinds = tuple(
            sorted(str(kind) for kind in allowed_failure_kinds)
        )
        super().__init__(
            "framework decision failures are disallowed for response-backed real-llm evaluation: "
            f"policy={self.policy_name} "
            f"benchmark_item_id={self.benchmark_item_id} "
            f"episode_id={self.episode_id} "
            f"seed={self.seed} "
            f"allowed_failure_kinds={list(self.allowed_failure_kinds)} "
            f"disallowed_failure_kinds={list(self.disallowed_failure_kinds)}"
        )


def _compute_total_tardiness(
    due_times: Mapping[int, Optional[float]],
    completion_times: Mapping[int, float],
) -> float:
    """Compute tardiness over completed jobs with known due dates."""
    total = 0.0
    for job_id, completion_time in completion_times.items():
        due_time = due_times.get(job_id)
        if due_time is None:
            continue
        total += max(0.0, float(completion_time) - float(due_time))
    return total


def _compute_completion_debt(
    *,
    jobs: Mapping[int, Any],
    completed_job_ids: Iterable[int],
    current_time: float,
) -> dict[str, float | int]:
    """Compute residual work and overdue debt visible at episode stop time."""
    completed = {int(job_id) for job_id in completed_job_ids}
    unfinished_overdue_tardiness = 0.0
    completion_debt_jobs = 0
    completion_debt_operations = 0
    for job_id, job in jobs.items():
        if int(job_id) in completed:
            continue
        completion_debt_jobs += 1
        total_operations = len(getattr(job, 'operations', ()))
        completed_operations = min(
            max(0, int(getattr(job, 'current_op_idx', 0))),
            total_operations,
        )
        completion_debt_operations += max(0, total_operations - completed_operations)
        due_time = getattr(job, 'due_time', None)
        if due_time is None:
            continue
        unfinished_overdue_tardiness += max(0.0, float(current_time) - float(due_time))
    return {
        'unfinished_overdue_tardiness': float(unfinished_overdue_tardiness),
        'completion_debt_jobs': int(completion_debt_jobs),
        'completion_debt_operations': int(completion_debt_operations),
    }


def _parse_benchmark_identity(benchmark_item_id: str) -> dict[str, str]:
    suite_id = ''
    split_id = ''
    for token in str(benchmark_item_id).split('__'):
        if '=' not in token:
            continue
        key, value = token.split('=', 1)
        if key == 'suite':
            suite_id = value
        elif key == 'split':
            split_id = value
    return {
        'suite_id': suite_id,
        'split_id': split_id,
    }


def _derive_failure_metrics_from_world_event_trace(
    world_event_trace: Iterable[Mapping[str, Any]],
) -> dict[str, float | int]:
    breakdown_count = 0
    downtime_total = 0.0
    repair_time_total = 0.0
    seen_events: set[tuple[float, str, tuple[tuple[str, str], ...]]] = set()
    for event in world_event_trace:
        kind = str(event.get('kind', ''))
        payload = dict(event.get('payload', {}))
        dedupe_key = (
            float(event.get('time', 0.0)),
            kind,
            tuple(sorted((str(key), str(value)) for key, value in payload.items())),
        )
        if dedupe_key in seen_events:
            continue
        seen_events.add(dedupe_key)
        if kind == 'machine_breakdown':
            breakdown_count += 1
        elif kind == 'machine_repair':
            repair_time = float(payload.get('repair_duration', payload.get('downtime_duration', 0.0)))
            downtime = float(payload.get('downtime_duration', repair_time))
            repair_time_total += repair_time
            downtime_total += downtime
    return {
        'breakdown_count_total': int(breakdown_count),
        'downtime_total': float(downtime_total),
        'repair_time_total': float(repair_time_total),
    }


def _derive_branching_metrics_from_activation_trace(
    activation_trace: Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    def _normalize_int_list(values: Any) -> list[int]:
        if not isinstance(values, (list, tuple)):
            return []
        return [int(value) for value in values]

    def _message_payload(context_handles: Mapping[str, Any]) -> dict[str, Any]:
        message = context_handles.get('message')
        if isinstance(message, Mapping):
            payload = message.get('payload', {})
            return dict(payload) if isinstance(payload, Mapping) else {}
        payload = getattr(message, 'payload', None)
        return dict(payload) if isinstance(payload, Mapping) else {}

    def _area_valid_cells(
        *,
        agent_id: str,
        context_handles: Mapping[str, Any],
    ) -> list[int]:
        direct = _normalize_int_list(context_handles.get('valid_cells', ()))
        if direct:
            return direct
        payload = _message_payload(context_handles)
        candidate_cells = _normalize_int_list(payload.get('candidate_cell_ids', ()))
        if candidate_cells:
            return candidate_cells
        valid_cells_by_area = payload.get('selected_backlog_valid_cells_by_area', {})
        if isinstance(valid_cells_by_area, Mapping):
            area_id = agent_id.split('_', 1)[1]
            nested = valid_cells_by_area.get(area_id, valid_cells_by_area.get(int(area_id), ()))
            return _normalize_int_list(nested)
        return []

    valid_backlog_counts: list[float] = []
    valid_area_counts: list[float] = []
    valid_cell_counts: list[float] = []
    dispatch_option_counts: list[float] = []

    for record in activation_trace:
        agent_id = str(record.get('agent_id', ''))
        functional_loop = str(record.get('functional_loop', ''))
        context_handles = dict(record.get('context_handles', {}))
        plant_decision_kind = str(
            record.get(
                'plant_decision_kind',
                context_handles.get('plant_decision_kind', ''),
            )
        )
        if agent_id == 'plant':
            if plant_decision_kind == 'backlog_selection':
                valid_backlog_ids = list(context_handles.get('valid_backlog_ids', ()))
                visible_candidates = list(context_handles.get('visible_backlog_candidates', ()))
                candidate_count = len(valid_backlog_ids) if valid_backlog_ids else len(visible_candidates)
                valid_backlog_counts.append(float(candidate_count))
            elif plant_decision_kind == 'area_selection':
                valid_areas = list(
                    context_handles.get(
                        'selected_backlog_valid_areas',
                        context_handles.get('valid_areas', ()),
                    )
                )
                valid_area_counts.append(float(len(valid_areas)))
            continue
        if agent_id.startswith('area_') and functional_loop == 'commitment_management':
            valid_cells = _area_valid_cells(
                agent_id=agent_id,
                context_handles=context_handles,
            )
            valid_cell_counts.append(float(len(valid_cells)))
            continue
        if agent_id.startswith('cell_') and functional_loop == 'local_regulation':
            dispatch_options = list(context_handles.get('dispatch_options', ()))
            dispatch_option_counts.append(float(len(dispatch_options)))

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def _rate(values: list[float]) -> float:
        return float(np.mean([value > 1.0 for value in values])) if values else 0.0

    return {
        'valid_backlog_count': _mean(valid_backlog_counts),
        'backlog_choice_count': float(len(valid_backlog_counts)),
        'plant_valid_area_count': _mean(valid_area_counts),
        'area_valid_cell_count': _mean(valid_cell_counts),
        'dispatch_option_count': _mean(dispatch_option_counts),
        'backlog_multi_action_epoch_rate': _rate(valid_backlog_counts),
        'plant_multi_action_epoch_rate': _rate(valid_area_counts),
        'area_multi_action_epoch_rate': _rate(valid_cell_counts),
        'cell_multi_action_epoch_rate': _rate(dispatch_option_counts),
    }


@dataclass
class HolonicEpisodeMetrics:
    """Metrics collected for one holonic plant rollout."""

    episode_id: int
    seed: int
    policy_name: str
    benchmark_item_id: str
    makespan: float
    total_tardiness: float
    completed_jobs: int
    throughput: float
    total_energy_kwh: float
    total_carbon_kg: float
    budget_violation_count: int
    budget_violation_magnitude: float
    decision_steps: int
    coordination_message_count: int
    active_agent_slots: int
    runtime_seconds: float
    total_reward: float
    terminated: bool
    truncated: bool
    deadlocked: bool
    invalid_actions: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'episode_id': self.episode_id,
            'seed': self.seed,
            'policy_name': self.policy_name,
            'benchmark_item_id': self.benchmark_item_id,
            'makespan': self.makespan,
            'total_tardiness': self.total_tardiness,
            'completed_jobs': self.completed_jobs,
            'throughput': self.throughput,
            'total_energy_kwh': self.total_energy_kwh,
            'total_carbon_kg': self.total_carbon_kg,
            'budget_violation_count': self.budget_violation_count,
            'budget_violation_magnitude': self.budget_violation_magnitude,
            'decision_steps': self.decision_steps,
            'coordination_message_count': self.coordination_message_count,
            'active_agent_slots': self.active_agent_slots,
            'runtime_seconds': self.runtime_seconds,
            'total_reward': self.total_reward,
            'terminated': self.terminated,
            'truncated': self.truncated,
            'deadlocked': self.deadlocked,
            'invalid_actions': self.invalid_actions,
        }
        data.update(self.extra)
        return data


@dataclass
class HolonicEvaluationResults:
    """Aggregated results for a holonic evaluation run."""

    benchmark_item_id: str
    policy_name: str
    seeds: List[int]
    episodes: List[HolonicEpisodeMetrics]

    def summary(self) -> Dict[str, Any]:
        if not self.episodes:
            return {
                'benchmark_item_id': self.benchmark_item_id,
                'policy_name': self.policy_name,
                'num_episodes': 0,
            }

        makespans = np.array([e.makespan for e in self.episodes], dtype=float)
        tardiness = np.array([e.total_tardiness for e in self.episodes], dtype=float)
        completed = np.array([e.completed_jobs for e in self.episodes], dtype=float)
        throughputs = np.array([e.throughput for e in self.episodes], dtype=float)
        energies = np.array([e.total_energy_kwh for e in self.episodes], dtype=float)
        carbons = np.array([e.total_carbon_kg for e in self.episodes], dtype=float)
        violation_counts = np.array([e.budget_violation_count for e in self.episodes], dtype=float)
        violations = np.array([e.budget_violation_magnitude for e in self.episodes], dtype=float)
        decision_steps = np.array([e.decision_steps for e in self.episodes], dtype=float)
        coordination_messages = np.array([e.coordination_message_count for e in self.episodes], dtype=float)
        active_agent_slots = np.array([e.active_agent_slots for e in self.episodes], dtype=float)
        runtimes = np.array([e.runtime_seconds for e in self.episodes], dtype=float)
        branching_metrics = [
            _derive_branching_metrics_from_activation_trace(
                e.extra.get('activation_trace', [])
            )
            for e in self.episodes
        ]
        valid_backlog_counts = np.array(
            [metrics['valid_backlog_count'] for metrics in branching_metrics],
            dtype=float,
        )
        backlog_choice_counts = np.array(
            [metrics['backlog_choice_count'] for metrics in branching_metrics],
            dtype=float,
        )
        plant_valid_area_counts = np.array(
            [metrics['plant_valid_area_count'] for metrics in branching_metrics],
            dtype=float,
        )
        area_valid_cell_counts = np.array(
            [metrics['area_valid_cell_count'] for metrics in branching_metrics],
            dtype=float,
        )
        dispatch_option_counts = np.array(
            [metrics['dispatch_option_count'] for metrics in branching_metrics],
            dtype=float,
        )
        backlog_multi_action_rates = np.array(
            [metrics['backlog_multi_action_epoch_rate'] for metrics in branching_metrics],
            dtype=float,
        )
        plant_multi_action_rates = np.array(
            [metrics['plant_multi_action_epoch_rate'] for metrics in branching_metrics],
            dtype=float,
        )
        area_multi_action_rates = np.array(
            [metrics['area_multi_action_epoch_rate'] for metrics in branching_metrics],
            dtype=float,
        )
        cell_multi_action_rates = np.array(
            [metrics['cell_multi_action_epoch_rate'] for metrics in branching_metrics],
            dtype=float,
        )
        activation_counts = np.array([float(e.extra.get('activation_record_count', 0.0)) for e in self.episodes], dtype=float)
        settlement_counts = np.array([float(e.extra.get('settlement_event_count', 0.0)) for e in self.episodes], dtype=float)
        settled_contract_counts = np.array(
            [float(e.extra.get('settled_contract_count', 0.0)) for e in self.episodes],
            dtype=float,
        )
        open_contract_counts = np.array(
            [float(e.extra.get('open_contract_count', 0.0)) for e in self.episodes],
            dtype=float,
        )
        completed_unsettled_contract_counts = np.array(
            [float(e.extra.get('completed_unsettled_contract_count', 0.0)) for e in self.episodes],
            dtype=float,
        )
        unfinished_overdue_tardiness = np.array(
            [float(e.extra.get('unfinished_overdue_tardiness', 0.0)) for e in self.episodes],
            dtype=float,
        )
        completion_debt_jobs = np.array(
            [float(e.extra.get('completion_debt_jobs', 0.0)) for e in self.episodes],
            dtype=float,
        )
        completion_debt_operations = np.array(
            [float(e.extra.get('completion_debt_operations', 0.0)) for e in self.episodes],
            dtype=float,
        )
        tardiness_with_completion_debt = np.array(
            [float(e.extra.get('tardiness_with_completion_debt', e.total_tardiness)) for e in self.episodes],
            dtype=float,
        )
        decision_source_counts = {
            source: np.array(
                [float(e.extra.get(f'framework_{source}_decision_count', 0.0)) for e in self.episodes],
                dtype=float,
            )
            for source in DECISION_SOURCE_KEYS
        }
        energy_overshoot_magnitude_total = np.array(
            [float(e.extra.get('energy_overshoot_magnitude_total', 0.0)) for e in self.episodes],
            dtype=float,
        )
        carbon_overshoot_magnitude_total = np.array(
            [float(e.extra.get('carbon_overshoot_magnitude_total', 0.0)) for e in self.episodes],
            dtype=float,
        )
        energy_overshoot_duration = np.array(
            [float(e.extra.get('energy_overshoot_duration', 0.0)) for e in self.episodes],
            dtype=float,
        )
        carbon_overshoot_duration = np.array(
            [float(e.extra.get('carbon_overshoot_duration', 0.0)) for e in self.episodes],
            dtype=float,
        )
        energy_overshoot_frequency = np.array(
            [float(e.extra.get('energy_overshoot_frequency', 0.0)) for e in self.episodes],
            dtype=float,
        )
        carbon_overshoot_frequency = np.array(
            [float(e.extra.get('carbon_overshoot_frequency', 0.0)) for e in self.episodes],
            dtype=float,
        )
        setup_times = np.array(
            [float(e.extra.get('setup_time_total', 0.0)) for e in self.episodes],
            dtype=float,
        )
        transport_times = np.array(
            [float(e.extra.get('transport_time_total', 0.0)) for e in self.episodes],
            dtype=float,
        )
        setup_energies = np.array(
            [float(e.extra.get('setup_energy_kwh_total', 0.0)) for e in self.episodes],
            dtype=float,
        )
        transport_energies = np.array(
            [float(e.extra.get('transport_energy_kwh_total', 0.0)) for e in self.episodes],
            dtype=float,
        )
        transport_moves = np.array(
            [float(e.extra.get('transport_moves_total', 0.0)) for e in self.episodes],
            dtype=float,
        )
        inter_cell_moves = np.array(
            [float(e.extra.get('inter_cell_moves_total', e.extra.get('transport_moves_total', 0.0))) for e in self.episodes],
            dtype=float,
        )
        blocking_times = np.array(
            [float(e.extra.get('blocking_time_total', 0.0)) for e in self.episodes],
            dtype=float,
        )
        buffer_wait_times = np.array(
            [float(e.extra.get('buffer_wait_time_total', 0.0)) for e in self.episodes],
            dtype=float,
        )
        arrival_events = np.array(
            [float(e.extra.get('arrival_events_total', 0.0)) for e in self.episodes],
            dtype=float,
        )
        arrived_jobs = np.array(
            [float(e.extra.get('arrived_jobs_total', 0.0)) for e in self.episodes],
            dtype=float,
        )
        arrival_batch_sizes = np.array(
            [float(e.extra.get('arrival_batch_size_mean', 0.0)) for e in self.episodes],
            dtype=float,
        )
        breakdown_counts = np.array(
            [float(e.extra.get('breakdown_count_total', 0.0)) for e in self.episodes],
            dtype=float,
        )
        downtime_totals = np.array(
            [float(e.extra.get('downtime_total', 0.0)) for e in self.episodes],
            dtype=float,
        )
        repair_time_totals = np.array(
            [float(e.extra.get('repair_time_total', 0.0)) for e in self.episodes],
            dtype=float,
        )

        return {
            'benchmark_item_id': self.benchmark_item_id,
            'suite_id': str(self.episodes[0].extra.get('suite_id', '')),
            'split_id': str(self.episodes[0].extra.get('split_id', '')),
            'policy_name': self.policy_name,
            'num_episodes': len(self.episodes),
            'controller_interface_mode': str(self.episodes[0].extra.get('controller_interface_mode', 'in_process')),
            'framework_name': str(self.episodes[0].extra.get('framework_name', '')),
            'framework_backend_mode': str(self.episodes[0].extra.get('framework_backend_mode', '')),
            'framework_python_package': str(self.episodes[0].extra.get('framework_python_package', '')),
            'framework_optional_dependency': str(self.episodes[0].extra.get('framework_optional_dependency', '')),
            'framework_uv_project_environment': str(self.episodes[0].extra.get('framework_uv_project_environment', '')),
            'framework_llm_backend': str(self.episodes[0].extra.get('framework_llm_backend', '')),
            'framework_llm_model': str(self.episodes[0].extra.get('framework_llm_model', '')),
            'framework_llm_base_url': str(self.episodes[0].extra.get('framework_llm_base_url', '')),
            'framework_decision_audit_contract': str(
                self.episodes[0].extra.get('framework_decision_audit_contract', '')
            ),
            'reward_authority': str(self.episodes[0].extra.get('reward_authority', '')),
            'numeric_profile_id': str(self.episodes[0].extra.get('numeric_profile_id', '')),
            'numeric_profile_provenance_class': str(
                self.episodes[0].extra.get('numeric_profile_provenance_class', '')
            ),
            'numeric_profile_projection_kind': str(
                self.episodes[0].extra.get('numeric_profile_projection_kind', '')
            ),
            'numeric_profile_scope': str(self.episodes[0].extra.get('numeric_profile_scope', '')),
            'numeric_profile_source_ref': str(
                self.episodes[0].extra.get('numeric_profile_source_ref', '')
            ),
            'due_date_profile_id': str(self.episodes[0].extra.get('due_date_profile_id', '')),
            'due_date_profile_rule_family': str(
                self.episodes[0].extra.get('due_date_profile_rule_family', '')
            ),
            'due_date_profile_provenance_class': str(
                self.episodes[0].extra.get('due_date_profile_provenance_class', '')
            ),
            'due_date_profile_source_ref': str(
                self.episodes[0].extra.get('due_date_profile_source_ref', '')
            ),
            'setup_profile_id': str(self.episodes[0].extra.get('setup_profile_id', '')),
            'setup_profile_provenance_class': str(
                self.episodes[0].extra.get('setup_profile_provenance_class', '')
            ),
            'setup_profile_scope': str(self.episodes[0].extra.get('setup_profile_scope', '')),
            'setup_profile_source_ref': str(
                self.episodes[0].extra.get('setup_profile_source_ref', '')
            ),
            'arrival_profile_id': str(self.episodes[0].extra.get('arrival_profile_id', '')),
            'arrival_profile_provenance_class': str(
                self.episodes[0].extra.get('arrival_profile_provenance_class', '')
            ),
            'arrival_profile_scope': str(self.episodes[0].extra.get('arrival_profile_scope', '')),
            'arrival_profile_source_ref': str(
                self.episodes[0].extra.get('arrival_profile_source_ref', '')
            ),
            'failure_profile_id': str(self.episodes[0].extra.get('failure_profile_id', '')),
            'failure_profile_provenance_class': str(
                self.episodes[0].extra.get('failure_profile_provenance_class', '')
            ),
            'failure_profile_scope': str(self.episodes[0].extra.get('failure_profile_scope', '')),
            'failure_profile_source_ref': str(
                self.episodes[0].extra.get('failure_profile_source_ref', '')
            ),
            'seeds': list(self.seeds),
            'success_rate': float(np.mean([e.terminated and not e.deadlocked and not e.truncated for e in self.episodes])),
            'deadlock_rate': float(np.mean([e.deadlocked for e in self.episodes])),
            'truncation_rate': float(np.mean([e.truncated for e in self.episodes])),
            'makespan_mean': float(np.mean(makespans)),
            'makespan_std': float(np.std(makespans)),
            'tardiness_mean': float(np.mean(tardiness)),
            'unfinished_overdue_tardiness_mean': float(np.mean(unfinished_overdue_tardiness)),
            'completion_debt_jobs_mean': float(np.mean(completion_debt_jobs)),
            'completion_debt_operations_mean': float(np.mean(completion_debt_operations)),
            'tardiness_with_completion_debt_mean': float(np.mean(tardiness_with_completion_debt)),
            'completed_jobs_mean': float(np.mean(completed)),
            'throughput_mean': float(np.mean(throughputs)),
            'energy_mean': float(np.mean(energies)),
            'carbon_mean': float(np.mean(carbons)),
            'violation_count_mean': float(np.mean(violation_counts)),
            'violation_magnitude_mean': float(np.mean(violations)),
            'energy_overshoot_magnitude_total_mean': float(np.mean(energy_overshoot_magnitude_total)),
            'carbon_overshoot_magnitude_total_mean': float(np.mean(carbon_overshoot_magnitude_total)),
            'energy_overshoot_duration_mean': float(np.mean(energy_overshoot_duration)),
            'carbon_overshoot_duration_mean': float(np.mean(carbon_overshoot_duration)),
            'energy_overshoot_frequency_mean': float(np.mean(energy_overshoot_frequency)),
            'carbon_overshoot_frequency_mean': float(np.mean(carbon_overshoot_frequency)),
            'setup_time_mean': float(np.mean(setup_times)),
            'setup_energy_kwh_mean': float(np.mean(setup_energies)),
            'transport_time_mean': float(np.mean(transport_times)),
            'transport_energy_kwh_mean': float(np.mean(transport_energies)),
            'transport_moves_mean': float(np.mean(transport_moves)),
            'inter_cell_moves_mean': float(np.mean(inter_cell_moves)),
            'blocking_time_mean': float(np.mean(blocking_times)),
            'buffer_wait_time_mean': float(np.mean(buffer_wait_times)),
            'arrival_events_mean': float(np.mean(arrival_events)),
            'arrived_jobs_mean': float(np.mean(arrived_jobs)),
            'arrival_batch_size_mean': float(np.mean(arrival_batch_sizes)),
            'breakdown_count_mean': float(np.mean(breakdown_counts)),
            'downtime_mean': float(np.mean(downtime_totals)),
            'repair_time_mean': float(np.mean(repair_time_totals)),
            'valid_backlog_count_mean': float(np.mean(valid_backlog_counts)),
            'backlog_choice_count_mean': float(np.mean(backlog_choice_counts)),
            'plant_valid_area_count_mean': float(np.mean(plant_valid_area_counts)),
            'area_valid_cell_count_mean': float(np.mean(area_valid_cell_counts)),
            'dispatch_option_count_mean': float(np.mean(dispatch_option_counts)),
            'backlog_multi_action_epoch_rate': float(np.mean(backlog_multi_action_rates)),
            'plant_multi_action_epoch_rate': float(np.mean(plant_multi_action_rates)),
            'area_multi_action_epoch_rate': float(np.mean(area_multi_action_rates)),
            'cell_multi_action_epoch_rate': float(np.mean(cell_multi_action_rates)),
            'decision_steps_mean': float(np.mean(decision_steps)),
            'coordination_message_count_mean': float(np.mean(coordination_messages)),
            'active_agent_slots_mean': float(np.mean(active_agent_slots)),
            'runtime_mean_seconds': float(np.mean(runtimes)),
            'activation_record_count_mean': float(np.mean(activation_counts)),
            'settlement_event_count_mean': float(np.mean(settlement_counts)),
            'settled_contract_count_mean': float(np.mean(settled_contract_counts)),
            'open_contract_count_mean': float(np.mean(open_contract_counts)),
            'completed_unsettled_contract_count_mean': float(np.mean(completed_unsettled_contract_counts)),
            'framework_llm_decision_count_mean': float(np.mean(decision_source_counts['llm'])),
            'framework_fallback_contract_decision_count_mean': float(
                np.mean(decision_source_counts['fallback_contract'])
            ),
            'framework_no_legal_action_count_mean': float(
                np.mean(decision_source_counts['no_legal_action'])
            ),
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            'summary': self.summary(),
            'episodes': [episode.to_dict() for episode in self.episodes],
        }
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def save_episode_csv(self, path: str | Path) -> None:
        path = Path(path)
        rows = [episode.to_dict() for episode in self.episodes]
        if not rows:
            path.write_text('', encoding='utf-8')
            return
        fieldnames = list(rows[0].keys())
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


class HolonicEvaluator:
    """Multi-seed evaluator for plant-level holonic policies."""

    def __init__(
        self,
        cell_configs: Iterable[PlantCellConfig],
        config: Optional[BenchmarkConfig] = None,
        benchmark_item_id: str = 'unspecified',
        env_cls: type[HolonicPlantEnv] = HolonicPlantEnv,
        progress_callback: ProgressCallback | None = None,
        progress_decision_interval: int = 5,
        require_zero_fallback: bool = False,
        allowed_framework_failure_kinds: Optional[Iterable[str]] = None,
    ) -> None:
        self.cell_configs = list(cell_configs)
        self.config = config or BenchmarkConfig()
        self.benchmark_item_id = benchmark_item_id
        self.benchmark_identity = _parse_benchmark_identity(benchmark_item_id)
        self.env_cls = env_cls
        self.progress_callback = progress_callback
        self.progress_decision_interval = max(int(progress_decision_interval), 1)
        self.require_zero_fallback = bool(require_zero_fallback)
        self.allowed_framework_failure_kinds = (
            None
            if allowed_framework_failure_kinds is None
            else frozenset(str(kind) for kind in allowed_framework_failure_kinds)
        )

    def _select_action(self, policy: Any, observation: np.ndarray, info: Dict[str, Any]) -> Any:
        if hasattr(policy, 'select_action'):
            return policy.select_action(observation, info)
        if hasattr(policy, 'act'):
            return policy.act(observation, info)
        if callable(policy):
            return policy(observation, info)
        valid_cells = info.get('valid_cells', [])
        return int(valid_cells[0]) if valid_cells else 0

    @staticmethod
    def _budget_violations(info: Dict[str, Any]) -> tuple[int, float]:
        violation_summary = HolonicEvaluator._structured_violation_summary(info)
        if any(float(violation_summary[key]) > 0.0 for key in _STRUCTURED_VIOLATION_KEYS):
            count = int(
                float(violation_summary['energy_overshoot_frequency'])
                + float(violation_summary['carbon_overshoot_frequency'])
            )
            magnitude = float(
                float(violation_summary['energy_overshoot_magnitude_total'])
                + float(violation_summary['carbon_overshoot_magnitude_total'])
            )
            return count, magnitude

        count = 0
        magnitude = 0.0
        for summary in info.get('cell_summaries', {}).values():
            energy_excess = max(0.0, float(summary['energy_kwh']) - float(summary['budget_cap_energy_kwh']))
            carbon_excess = max(0.0, float(summary['carbon_kg']) - float(summary['budget_cap_carbon_kg']))
            wip_excess = max(0.0, float(summary['active_jobs']) - float(summary['budget_cap_wip']))
            for excess in (energy_excess, carbon_excess, wip_excess):
                if excess > 0.0:
                    count += 1
                    magnitude += excess
        return count, magnitude

    def run_episode(self, policy: Any, episode_id: int, seed: int) -> HolonicEpisodeMetrics:
        env = self.env_cls(self.cell_configs, config=self.config, seed=seed)
        start_wall_clock = time.perf_counter()
        try:
            rollout = self._run_episode_rollout(
                env=env,
                policy=policy,
                episode_id=episode_id,
                seed=seed,
            )
            controller_interface_mode = self._controller_interface_mode(policy)
            total_reward = float(rollout['total_reward'])
            decision_steps = int(rollout['decision_steps'])
            coordination_message_count = int(rollout['coordination_message_count'])
            active_agent_slots = int(rollout['active_agent_slots'])
            activation_record_count = int(rollout['activation_record_count'])
            episode_activation_trace = list(rollout['episode_activation_trace'])
            episode_settlement_trace = list(rollout['episode_settlement_trace'])
            episode_protocol_trace = list(rollout['episode_protocol_trace'])
            episode_world_event_trace = list(rollout['episode_world_event_trace'])
            episode_state_trace = list(rollout['episode_state_trace'])
            terminated = bool(rollout['terminated'])
            truncated = bool(rollout['truncated'])
            final_info = dict(rollout['final_info'])

            settlement_event_count = len(episode_settlement_trace)
            completed_jobs = int(final_info['plant']['completed_jobs_total'])
            makespan = float(final_info['time'])
            throughput = float(completed_jobs / makespan) if makespan > 0.0 else float(completed_jobs)
            violation_count, violation_magnitude = self._budget_violations(final_info)
            due_times = {job_id: job.due_time for job_id, job in env.simulator.jobs.items()}
            total_tardiness = _compute_total_tardiness(
                due_times=due_times,
                completion_times=env.simulator.job_completion_times,
            )
            completion_debt = _compute_completion_debt(
                jobs=env.simulator.jobs,
                completed_job_ids=env.simulator.completed_jobs,
                current_time=makespan,
            )
            unfinished_overdue_tardiness = float(
                completion_debt['unfinished_overdue_tardiness']
            )
            completion_debt_jobs = int(completion_debt['completion_debt_jobs'])
            completion_debt_operations = int(
                completion_debt['completion_debt_operations']
            )
            policy_name = getattr(policy, 'name', getattr(policy, '__name__', 'unknown'))
            runtime_seconds = float(time.perf_counter() - start_wall_clock)
            framework_runtime_info = self._framework_runtime_info(policy)
            decision_audit = self._framework_decision_audit(policy)
            decision_source_counts = dict(decision_audit.get('decision_source_counts', {}))
            fallback_count = int(decision_source_counts.get('fallback_contract', 0))
            if self.require_zero_fallback and fallback_count > 0:
                raise ZeroFallbackViolationError(
                    policy_name=policy_name,
                    benchmark_item_id=self.benchmark_item_id,
                    episode_id=episode_id,
                    seed=seed,
                    fallback_count=fallback_count,
                )
            violation_summary = self._structured_violation_summary(final_info)
            failure_metrics = _derive_failure_metrics_from_world_event_trace(
                episode_world_event_trace
            )
            branching_metrics = _derive_branching_metrics_from_activation_trace(
                episode_activation_trace
            )
            episode_metrics = HolonicEpisodeMetrics(
                episode_id=episode_id,
                seed=seed,
                policy_name=policy_name,
                benchmark_item_id=self.benchmark_item_id,
                makespan=makespan,
                total_tardiness=total_tardiness,
                completed_jobs=completed_jobs,
                throughput=throughput,
                total_energy_kwh=float(sum(summary['energy_kwh'] for summary in final_info['cell_summaries'].values())),
                total_carbon_kg=float(sum(summary['carbon_kg'] for summary in final_info['cell_summaries'].values())),
                budget_violation_count=violation_count,
                budget_violation_magnitude=violation_magnitude,
                decision_steps=decision_steps,
                coordination_message_count=coordination_message_count,
                active_agent_slots=active_agent_slots,
                runtime_seconds=runtime_seconds,
                total_reward=float(total_reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                deadlocked=bool(final_info.get('deadlocked', False)),
                invalid_actions=int(final_info.get('invalid_actions_total', 0)),
                extra={
                    'suite_id': str(self.benchmark_identity.get('suite_id', '')),
                    'split_id': str(self.benchmark_identity.get('split_id', '')),
                    'controller_interface_mode': controller_interface_mode,
                    'framework_name': str(framework_runtime_info.get('framework_name', '')),
                    'framework_backend_mode': str(framework_runtime_info.get('backend_mode', '')),
                    'framework_python_package': str(framework_runtime_info.get('python_package', '')),
                    'framework_optional_dependency': str(framework_runtime_info.get('optional_dependency', '')),
                    'framework_uv_project_environment': str(framework_runtime_info.get('uv_project_environment', '')),
                    'framework_llm_backend': str(framework_runtime_info.get('llm_backend', '')),
                    'framework_llm_model': str(framework_runtime_info.get('llm_model', '')),
                    'framework_llm_base_url': str(framework_runtime_info.get('llm_base_url', '')),
                    'framework_decision_audit_contract': str(
                        decision_audit.get('framework_decision_audit_contract', '')
                    ),
                    'framework_decision_source_trace': list(
                        decision_audit.get('decision_source_trace', [])
                    ),
                    'framework_decision_failure_trace': list(
                        decision_audit.get('decision_failure_trace', [])
                    ),
                    'framework_decision_source_counts': decision_source_counts,
                    'framework_llm_decision_count': int(decision_source_counts.get('llm', 0)),
                    'framework_fallback_contract_decision_count': int(
                        decision_source_counts.get('fallback_contract', 0)
                    ),
                    'framework_no_legal_action_count': int(
                        decision_source_counts.get('no_legal_action', 0)
                    ),
                    'numeric_profile_id': str(self.config.numeric_profile_id),
                    'numeric_profile_provenance_class': str(
                        self.config.numeric_profile_provenance_class
                    ),
                    'numeric_profile_projection_kind': str(
                        self.config.numeric_profile_projection_kind
                    ),
                    'numeric_profile_scope': str(self.config.numeric_profile_scope),
                    'numeric_profile_source_ref': str(self.config.numeric_profile_source_ref),
                    'due_date_profile_id': str(self.config.due_date_profile_id),
                    'due_date_profile_rule_family': str(self.config.due_date_profile_rule_family),
                    'due_date_profile_provenance_class': str(
                        self.config.due_date_profile_provenance_class
                    ),
                    'due_date_profile_source_ref': str(self.config.due_date_profile_source_ref),
                    'setup_profile_id': str(self.config.setup_profile_id),
                    'setup_profile_provenance_class': str(
                        self.config.setup_profile_provenance_class
                    ),
                    'setup_profile_scope': str(self.config.setup_profile_scope),
                    'setup_profile_source_ref': str(self.config.setup_profile_source_ref),
                    'arrival_profile_id': str(self.config.arrival_profile_id),
                    'arrival_profile_provenance_class': str(
                        self.config.arrival_profile_provenance_class
                    ),
                    'arrival_profile_scope': str(self.config.arrival_profile_scope),
                    'arrival_profile_source_ref': str(self.config.arrival_profile_source_ref),
                    'arrival_process': str(self.config.arrival.process),
                    'arrival_lambda_low': float(self.config.arrival.lambda_low),
                    'arrival_lambda_high': float(self.config.arrival.lambda_high),
                    'arrival_nhpp_cycle_hours': float(self.config.arrival.nhpp_cycle_hours),
                    'arrival_batch_size_distribution': str(
                        self.config.arrival.batch_size_distribution
                    ),
                    'arrival_batch_size_low': int(self.config.arrival.batch_size_low),
                    'arrival_batch_size_high': int(self.config.arrival.batch_size_high),
                    'arrival_batch_size_poisson_lambda': float(
                        self.config.arrival.batch_size_poisson_lambda
                    ),
                    'failure_profile_id': str(self.config.failure_profile_id),
                    'failure_profile_provenance_class': str(
                        self.config.failure_profile_provenance_class
                    ),
                    'failure_profile_scope': str(self.config.failure_profile_scope),
                    'failure_profile_source_ref': str(self.config.failure_profile_source_ref),
                    'failure_process': str(self.config.failure.process),
                    'failure_mtbf_hours': float(self.config.failure.mtbf_hours),
                    'failure_mttr_hours': float(self.config.failure.mttr_hours),
                    'failure_weibull_shape': float(self.config.failure.weibull_shape),
                    'failure_load_dependent_alpha': float(
                        self.config.failure.load_dependent_alpha
                    ),
                    'failure_load_reference_proc_hours': float(
                        self.config.failure.load_reference_proc_hours
                    ),
                    'backlog_jobs_total': int(final_info['plant']['backlog_jobs_total']),
                    'inbound_jobs_total': int(final_info['plant'].get('inbound_jobs_total', 0)),
                    'down_machines_total': int(final_info['plant']['down_machines_total']),
                    'arrival_events_total': int(final_info.get('arrival_events_total', 0)),
                    'arrived_jobs_total': int(final_info.get('arrived_jobs_total', 0)),
                    'arrival_batch_size_mean': float(final_info.get('arrival_batch_size_mean', 0.0)),
                    'realized_arrival_trace': list(final_info.get('realized_arrival_trace', [])),
                    'setup_time_total': float(final_info.get('setup_time_total', 0.0)),
                    'setup_energy_kwh_total': float(final_info.get('setup_energy_kwh_total', 0.0)),
                    'blocking_time_total': float(final_info.get('blocking_time_total', 0.0)),
                    'buffer_wait_time_total': float(final_info.get('buffer_wait_time_total', 0.0)),
                    'transport_time_total': float(final_info.get('transport_time_total', 0.0)),
                    'transport_energy_kwh_total': float(
                        final_info.get('transport_energy_kwh_total', 0.0)
                    ),
                    'transport_moves_total': int(final_info.get('transport_moves_total', 0)),
                    'inter_cell_moves_total': int(
                        final_info.get('inter_cell_moves_total', final_info.get('transport_moves_total', 0))
                    ),
                    'valid_backlog_count': float(branching_metrics['valid_backlog_count']),
                    'backlog_choice_count': float(branching_metrics['backlog_choice_count']),
                    'plant_valid_area_count': float(branching_metrics['plant_valid_area_count']),
                    'area_valid_cell_count': float(branching_metrics['area_valid_cell_count']),
                    'dispatch_option_count': float(branching_metrics['dispatch_option_count']),
                    'backlog_multi_action_epoch_rate': float(
                        branching_metrics['backlog_multi_action_epoch_rate']
                    ),
                    'plant_multi_action_epoch_rate': float(
                        branching_metrics['plant_multi_action_epoch_rate']
                    ),
                    'area_multi_action_epoch_rate': float(
                        branching_metrics['area_multi_action_epoch_rate']
                    ),
                    'cell_multi_action_epoch_rate': float(
                        branching_metrics['cell_multi_action_epoch_rate']
                    ),
                    **failure_metrics,
                    'activation_record_count': int(activation_record_count),
                    'settlement_event_count': int(settlement_event_count),
                    'settlement_summary': self._jsonable_mapping(
                        final_info.get('settlement_summary', {})
                    ),
                    'settled_contract_count': int(
                        dict(final_info.get('settlement_summary', {})).get(
                            'settled_contracts', 0
                        )
                    ),
                    'open_contract_count': int(
                        dict(final_info.get('settlement_summary', {})).get(
                            'open_contracts', 0
                        )
                    ),
                    'completed_unsettled_contract_count': int(
                        dict(final_info.get('settlement_summary', {})).get(
                            'completed_unsettled_contracts', 0
                        )
                    ),
                    'activation_trace': episode_activation_trace,
                    'settlement_trace': episode_settlement_trace,
                    'protocol_trace': episode_protocol_trace,
                    'world_event_trace': episode_world_event_trace,
                    'state_trace': episode_state_trace,
                    'reward_authority': str(final_info.get('reward_authority', '')),
                    'violation_summary': dict(violation_summary),
                    **{
                        key: violation_summary[key]
                        for key in _STRUCTURED_VIOLATION_KEYS
                    },
                    'unfinished_overdue_tardiness': unfinished_overdue_tardiness,
                    'completion_debt_jobs': completion_debt_jobs,
                    'completion_debt_operations': completion_debt_operations,
                    'tardiness_with_completion_debt': float(
                        total_tardiness + unfinished_overdue_tardiness
                    ),
                },
            )
            self._emit_progress(
                event='episode_completed',
                policy=policy,
                episode_id=episode_id,
                seed=seed,
                decision_steps=decision_steps,
                coordination_message_count=coordination_message_count,
                active_agent_slots=active_agent_slots,
                info=final_info,
                runtime_seconds=runtime_seconds,
                terminated=terminated,
                truncated=truncated,
            )
            return episode_metrics
        finally:
            env.close()

    def _run_episode_rollout(
        self,
        *,
        env: Any,
        policy: Any,
        episode_id: int,
        seed: int,
    ) -> Dict[str, Any]:
        if self._uses_external_session(policy):
            return self._run_external_session_rollout(
                env=env,
                policy=policy,
                episode_id=episode_id,
                seed=seed,
            )
        return self._run_native_rollout(
            env=env,
            policy=policy,
            episode_id=episode_id,
            seed=seed,
        )

    def _run_native_rollout(
        self,
        *,
        env: Any,
        policy: Any,
        episode_id: int,
        seed: int,
    ) -> Dict[str, Any]:
        observation, info = env.reset(seed=seed)
        controller_adapter = self._maybe_make_controller_adapter(env)
        total_reward = 0.0
        decision_steps = 0
        coordination_message_count = 0
        active_agent_slots = 0
        activation_record_count = len(info.get('activation_trace', []))
        episode_activation_trace = list(info.get('activation_trace', []))
        episode_settlement_trace = list(info.get('settlement_trace', []))
        episode_protocol_trace = list(info.get('protocol_trace', []))
        episode_world_event_trace = list(info.get('world_event_trace', []))
        episode_state_trace = [self._extract_public_state_snapshot(info)]
        terminated = False
        truncated = False
        final_info = info
        self._emit_progress(
            event='episode_started',
            policy=policy,
            episode_id=episode_id,
            seed=seed,
            decision_steps=decision_steps,
            coordination_message_count=coordination_message_count,
            active_agent_slots=active_agent_slots,
            info=info,
            active_agents=info.get('agents_to_act', []),
            terminated=terminated,
            truncated=truncated,
        )

        while not (terminated or truncated):
            active_agent_slots += len(info.get('agents_to_act', []))
            if not info.get('agents_to_act', []):
                if hasattr(env, 'advance_idle_epoch'):
                    step = env.advance_idle_epoch()
                    observation = step.observation
                    reward = step.reward
                    terminated = step.terminated
                    truncated = step.truncated
                    info = step.info
                    total_reward += float(reward)
                    protocol_trace = info.get('protocol_trace')
                    if protocol_trace is not None:
                        coordination_message_count += len(protocol_trace)
                    else:
                        coordination_message_count += len(info.get('messages', []))
                    activation_record_count += len(info.get('activation_trace', []))
                    episode_activation_trace.extend(info.get('activation_trace', []))
                    settlement_trace = info.get('settlement_trace')
                    if settlement_trace is not None:
                        episode_settlement_trace = list(settlement_trace)
                    episode_protocol_trace.extend(info.get('protocol_trace', []))
                    episode_world_event_trace.extend(info.get('world_event_trace', []))
                    episode_state_trace.append(self._extract_public_state_snapshot(info))
                    final_info = info
                    continue
                raise RuntimeError('No controller action is available and the environment cannot auto-advance')
            if controller_adapter is not None:
                action = controller_adapter.step_controller(policy)
                self._enforce_allowed_framework_failure_kinds(
                    policy=policy,
                    episode_id=episode_id,
                    seed=seed,
                )
                step = controller_adapter.step_dict(action)
                observation = step.observation
                reward = step.reward
                terminated = step.terminated
                truncated = step.truncated
                info = step.info
            else:
                action = self._select_action(policy, observation, info)
                self._enforce_allowed_framework_failure_kinds(
                    policy=policy,
                    episode_id=episode_id,
                    seed=seed,
                )
                if isinstance(action, dict) and hasattr(env, 'step_dict'):
                    step = env.step_dict(action)
                    observation = step.observation
                    reward = step.reward
                    terminated = step.terminated
                    truncated = step.truncated
                    info = step.info
                else:
                    observation, reward, terminated, truncated, info = env.step(int(action))
            protocol_trace = info.get('protocol_trace')
            if protocol_trace is not None:
                coordination_message_count += len(protocol_trace)
            else:
                coordination_message_count += len(info.get('messages', []))
            activation_record_count += len(info.get('activation_trace', []))
            episode_activation_trace.extend(info.get('activation_trace', []))
            settlement_trace = info.get('settlement_trace')
            if settlement_trace is not None:
                episode_settlement_trace = list(settlement_trace)
            episode_protocol_trace.extend(info.get('protocol_trace', []))
            episode_world_event_trace.extend(info.get('world_event_trace', []))
            episode_state_trace.append(self._extract_public_state_snapshot(info))
            total_reward += float(reward)
            decision_steps += 1
            final_info = info
            self._emit_progress_heartbeat(
                policy=policy,
                episode_id=episode_id,
                seed=seed,
                decision_steps=decision_steps,
                coordination_message_count=coordination_message_count,
                active_agent_slots=active_agent_slots,
                info=info,
                active_agents=info.get('agents_to_act', []),
                terminated=terminated,
                truncated=truncated,
            )

        return {
            'total_reward': total_reward,
            'decision_steps': decision_steps,
            'coordination_message_count': coordination_message_count,
            'active_agent_slots': active_agent_slots,
            'activation_record_count': activation_record_count,
            'episode_activation_trace': episode_activation_trace,
            'episode_settlement_trace': episode_settlement_trace,
            'episode_protocol_trace': episode_protocol_trace,
            'episode_world_event_trace': episode_world_event_trace,
            'episode_state_trace': episode_state_trace,
            'terminated': terminated,
            'truncated': truncated,
            'final_info': final_info,
        }

    def _run_external_session_rollout(
        self,
        *,
        env: Any,
        policy: Any,
        episode_id: int,
        seed: int,
    ) -> Dict[str, Any]:
        from .v2.external_controller_adapter import (
            ExternalControllerSession,
            _reset_external_driver_episode,
        )

        if not hasattr(env, 'current_decision_epoch'):
            raise TypeError('External-session policies require the v2 runtime path')

        session = ExternalControllerSession(env)
        _reset_external_driver_episode(policy)
        decision = session.reset(seed=seed)
        info = dict(session.current_artifact_payload()['info'])
        total_reward = 0.0
        decision_steps = 0
        coordination_message_count = 0
        active_agent_slots = 0
        activation_record_count = len(info.get('activation_trace', []))
        episode_activation_trace = list(info.get('activation_trace', []))
        episode_settlement_trace = list(info.get('settlement_trace', []))
        episode_protocol_trace = list(info.get('protocol_trace', []))
        episode_world_event_trace = list(info.get('world_event_trace', []))
        episode_state_trace = [self._extract_public_state_snapshot(info)]
        terminated = bool(decision.get('terminated', False))
        truncated = bool(decision.get('truncated', False))
        final_info = info
        self._emit_progress(
            event='episode_started',
            policy=policy,
            episode_id=episode_id,
            seed=seed,
            decision_steps=decision_steps,
            coordination_message_count=coordination_message_count,
            active_agent_slots=active_agent_slots,
            info=info,
            active_agents=decision.get('agents_to_act', []),
            terminated=terminated,
            truncated=truncated,
        )

        while not (terminated or truncated):
            if not decision.get('agents_to_act', []):
                terminated = bool(decision.get('terminated', terminated))
                truncated = bool(decision.get('truncated', truncated))
                break
            active_agent_slots += len(decision.get('agents_to_act', []))
            actions = self._select_external_actions(policy, decision)
            self._enforce_allowed_framework_failure_kinds(
                policy=policy,
                episode_id=episode_id,
                seed=seed,
            )
            transition = session.step(actions)
            reward = float(transition['reward'])
            terminated = bool(transition['terminated'])
            truncated = bool(transition['truncated'])
            current_decision = dict(transition['current_decision'])
            info = dict(session.current_artifact_payload()['info'])
            coordination_message_count += len(info.get('protocol_trace', []))
            activation_record_count += len(info.get('activation_trace', []))
            episode_activation_trace.extend(info.get('activation_trace', []))
            settlement_trace = info.get('settlement_trace')
            if settlement_trace is not None:
                episode_settlement_trace = list(settlement_trace)
            episode_protocol_trace.extend(info.get('protocol_trace', []))
            episode_world_event_trace.extend(info.get('world_event_trace', []))
            episode_state_trace.append(self._extract_public_state_snapshot(info))
            total_reward += reward
            decision_steps += 1
            final_info = info
            decision = current_decision
            self._emit_progress_heartbeat(
                policy=policy,
                episode_id=episode_id,
                seed=seed,
                decision_steps=decision_steps,
                coordination_message_count=coordination_message_count,
                active_agent_slots=active_agent_slots,
                info=info,
                active_agents=decision.get('agents_to_act', []),
                terminated=terminated,
                truncated=truncated,
            )

        return {
            'total_reward': total_reward,
            'decision_steps': decision_steps,
            'coordination_message_count': coordination_message_count,
            'active_agent_slots': active_agent_slots,
            'activation_record_count': activation_record_count,
            'episode_activation_trace': episode_activation_trace,
            'episode_settlement_trace': episode_settlement_trace,
            'episode_protocol_trace': episode_protocol_trace,
            'episode_world_event_trace': episode_world_event_trace,
            'episode_state_trace': episode_state_trace,
            'terminated': terminated,
            'truncated': truncated,
            'final_info': final_info,
        }

    @staticmethod
    def _extract_public_state_snapshot(info: Mapping[str, Any]) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            'time': float(info.get('time', 0.0)),
            'agents_to_act': [str(agent_id) for agent_id in info.get('agents_to_act', [])],
            'plant': HolonicEvaluator._jsonable_mapping(info.get('plant', {})),
            'cell_summaries': HolonicEvaluator._jsonable_mapping(info.get('cell_summaries', {})),
        }
        if 'machine_summaries' in info:
            snapshot['machine_summaries'] = HolonicEvaluator._jsonable_mapping(info.get('machine_summaries', {}))
        if 'area_summaries' in info:
            snapshot['area_summaries'] = HolonicEvaluator._jsonable_mapping(info.get('area_summaries', {}))
        if 'reward_components' in info:
            snapshot['reward_components'] = HolonicEvaluator._jsonable_mapping(info.get('reward_components', {}))
        if 'valid_cells' in info:
            snapshot['valid_cells'] = [int(cell_id) for cell_id in info.get('valid_cells', [])]
        if 'valid_areas' in info:
            snapshot['valid_areas'] = [int(area_id) for area_id in info.get('valid_areas', [])]
        if 'valid_backlog_ids' in info:
            snapshot['valid_backlog_ids'] = [int(job_id) for job_id in info.get('valid_backlog_ids', [])]
        if 'visible_backlog_candidates' in info:
            snapshot['visible_backlog_candidates'] = HolonicEvaluator._jsonable_mapping(
                info.get('visible_backlog_candidates', [])
            )
        if 'selected_backlog_id' in info:
            snapshot['selected_backlog_id'] = info.get('selected_backlog_id')
        if 'selected_backlog_stage_id' in info:
            snapshot['selected_backlog_stage_id'] = info.get('selected_backlog_stage_id')
        if 'selected_backlog_valid_areas' in info:
            snapshot['selected_backlog_valid_areas'] = [
                int(area_id) for area_id in info.get('selected_backlog_valid_areas', [])
            ]
        if 'selected_backlog_valid_cells_by_area' in info:
            snapshot['selected_backlog_valid_cells_by_area'] = HolonicEvaluator._jsonable_mapping(
                info.get('selected_backlog_valid_cells_by_area', {})
            )
        if 'plant_decision_kind' in info:
            snapshot['plant_decision_kind'] = str(info.get('plant_decision_kind', ''))
        if 'current_decision_epoch' in info:
            snapshot['current_decision_epoch'] = HolonicEvaluator._jsonable_mapping(info.get('current_decision_epoch', {}))
        return snapshot

    @staticmethod
    def _structured_violation_summary(info: Mapping[str, Any]) -> dict[str, float | int]:
        summary = info.get('violation_summary', {})
        if not isinstance(summary, Mapping):
            summary = {}
        normalized: dict[str, float | int] = {}
        for key in _STRUCTURED_VIOLATION_KEYS:
            value = summary.get(key, 0.0)
            if key.endswith('_frequency'):
                normalized[key] = int(value)
            else:
                normalized[key] = float(value)
        return normalized

    @staticmethod
    def _jsonable_mapping(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): HolonicEvaluator._jsonable_mapping(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [HolonicEvaluator._jsonable_mapping(item) for item in value]
        if isinstance(value, tuple):
            return [HolonicEvaluator._jsonable_mapping(item) for item in value]
        return value

    @staticmethod
    def _maybe_make_controller_adapter(env: Any) -> Any | None:
        if not hasattr(env, 'current_decision_epoch'):
            return None
        if not hasattr(env, 'step_dict'):
            return None
        from .v2.controller_api import TreeControllerAdapter

        return TreeControllerAdapter(env)

    @staticmethod
    def _uses_external_session(policy: Any) -> bool:
        if bool(getattr(policy, 'uses_external_session', False)):
            return True
        return hasattr(policy, 'select_actions')

    @staticmethod
    def _select_external_actions(policy: Any, decision: Dict[str, Any]) -> Any:
        if hasattr(policy, 'select_actions'):
            return policy.select_actions(decision)
        if callable(policy):
            return policy(decision)
        raise TypeError(f'Unsupported external-session policy: {type(policy)!r}')

    @staticmethod
    def _controller_interface_mode(policy: Any) -> str:
        if HolonicEvaluator._uses_external_session(policy):
            return 'external_session'
        return 'in_process'

    @staticmethod
    def _framework_runtime_info(policy: Any) -> Mapping[str, Any]:
        payload = getattr(policy, 'framework_runtime_info', None)
        if isinstance(payload, Mapping):
            return payload
        return {}

    @staticmethod
    def _framework_decision_audit(policy: Any) -> Mapping[str, Any]:
        payload = getattr(policy, 'decision_audit_payload', None)
        if isinstance(payload, Mapping):
            return payload
        return {}

    def _enforce_allowed_framework_failure_kinds(
        self,
        *,
        policy: Any,
        episode_id: int,
        seed: int,
    ) -> None:
        if self.allowed_framework_failure_kinds is None:
            return
        decision_audit = self._framework_decision_audit(policy)
        failure_trace = decision_audit.get('decision_failure_trace', [])
        if not isinstance(failure_trace, list):
            return
        disallowed_failure_kinds = {
            str(entry.get('failure_kind', '')).strip()
            for entry in failure_trace
            if isinstance(entry, Mapping)
            and str(entry.get('failure_kind', '')).strip()
            and str(entry.get('failure_kind', '')).strip()
            not in self.allowed_framework_failure_kinds
        }
        if not disallowed_failure_kinds:
            return
        raise DisallowedDecisionFailureError(
            policy_name=str(getattr(policy, 'name', getattr(policy, '__name__', 'unknown'))),
            benchmark_item_id=self.benchmark_item_id,
            episode_id=episode_id,
            seed=seed,
            disallowed_failure_kinds=disallowed_failure_kinds,
            allowed_failure_kinds=self.allowed_framework_failure_kinds,
        )

    def _emit_progress(
        self,
        *,
        event: str,
        policy: Any,
        episode_id: int,
        seed: int,
        decision_steps: int,
        coordination_message_count: int,
        active_agent_slots: int,
        info: Mapping[str, Any] | None = None,
        active_agents: Iterable[Any] | None = None,
        runtime_seconds: float | None = None,
        terminated: bool | None = None,
        truncated: bool | None = None,
    ) -> None:
        if self.progress_callback is None:
            return
        decision_source_counts = dict(
            self._framework_decision_audit(policy).get('decision_source_counts', {})
        )
        payload: dict[str, Any] = {
            'event': str(event),
            'policy_name': str(getattr(policy, 'name', getattr(policy, '__name__', 'unknown'))),
            'episode_id': int(episode_id),
            'seed': int(seed),
            'decision_steps': int(decision_steps),
            'coordination_message_count': int(coordination_message_count),
            'active_agent_slots': int(active_agent_slots),
            'framework_llm_decision_count': int(decision_source_counts.get('llm', 0)),
            'framework_fallback_contract_decision_count': int(
                decision_source_counts.get('fallback_contract', 0)
            ),
            'framework_no_legal_action_count': int(
                decision_source_counts.get('no_legal_action', 0)
            ),
        }
        if runtime_seconds is not None:
            payload['runtime_seconds'] = float(runtime_seconds)
        if terminated is not None:
            payload['terminated'] = bool(terminated)
        if truncated is not None:
            payload['truncated'] = bool(truncated)
        if active_agents is not None:
            serialized_active_agents = [str(agent_id) for agent_id in active_agents]
            payload['agents_to_act'] = serialized_active_agents
            payload['active_agent_count'] = len(serialized_active_agents)
        if info is not None:
            plant_info = info.get('plant', {})
            plant_payload = plant_info if isinstance(plant_info, Mapping) else {}
            payload.update(
                {
                    'sim_time': float(info.get('time', 0.0)),
                    'completed_jobs_total': int(plant_payload.get('completed_jobs_total', 0)),
                    'backlog_jobs_total': int(plant_payload.get('backlog_jobs_total', 0)),
                    'inbound_jobs_total': int(plant_payload.get('inbound_jobs_total', 0)),
                    'down_machines_total': int(plant_payload.get('down_machines_total', 0)),
                }
            )
            if 'plant_decision_kind' in info:
                payload['plant_decision_kind'] = str(info.get('plant_decision_kind', ''))
        self.progress_callback(payload)

    def _emit_progress_heartbeat(
        self,
        *,
        policy: Any,
        episode_id: int,
        seed: int,
        decision_steps: int,
        coordination_message_count: int,
        active_agent_slots: int,
        info: Mapping[str, Any] | None,
        active_agents: Iterable[Any] | None,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if decision_steps <= 0:
            return
        should_emit = (
            decision_steps == 1
            or decision_steps % self.progress_decision_interval == 0
            or terminated
            or truncated
        )
        if not should_emit:
            return
        self._emit_progress(
            event='episode_heartbeat',
            policy=policy,
            episode_id=episode_id,
            seed=seed,
            decision_steps=decision_steps,
            coordination_message_count=coordination_message_count,
            active_agent_slots=active_agent_slots,
            info=info,
            active_agents=active_agents,
            terminated=terminated,
            truncated=truncated,
        )

    def evaluate(self, policy: Any, seeds: Iterable[int]) -> HolonicEvaluationResults:
        seed_list = [int(seed) for seed in seeds]
        policy_name = getattr(policy, 'name', getattr(policy, '__name__', 'unknown'))
        episodes = [self.run_episode(policy, episode_id=i, seed=seed) for i, seed in enumerate(seed_list)]
        return HolonicEvaluationResults(
            benchmark_item_id=self.benchmark_item_id,
            policy_name=policy_name,
            seeds=seed_list,
            episodes=episodes,
        )


def evaluate_holonic_policy(
    policy: Any,
    cell_configs: Iterable[PlantCellConfig],
    config: Optional[BenchmarkConfig] = None,
    seeds: Optional[Iterable[int]] = None,
    benchmark_item_id: str = 'unspecified',
    env_cls: type[HolonicPlantEnv] = HolonicPlantEnv,
    progress_callback: ProgressCallback | None = None,
    progress_decision_interval: int = 5,
    require_zero_fallback: bool = False,
    allowed_framework_failure_kinds: Optional[Iterable[str]] = None,
) -> HolonicEvaluationResults:
    """Convenience wrapper for plant-level holonic evaluation."""
    evaluator = HolonicEvaluator(
        cell_configs=cell_configs,
        config=config,
        benchmark_item_id=benchmark_item_id,
        env_cls=env_cls,
        progress_callback=progress_callback,
        progress_decision_interval=progress_decision_interval,
        require_zero_fallback=require_zero_fallback,
        allowed_framework_failure_kinds=allowed_framework_failure_kinds,
    )
    seed_list = list(seeds) if seeds is not None else [0]
    return evaluator.evaluate(policy=policy, seeds=seed_list)
