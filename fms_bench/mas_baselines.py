"""MAS-like benchmark baselines built on the public holonic interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .baselines import BaselinePolicy


@dataclass(frozen=True)
class ContractNetBid:
    """One cell proposal produced during a Contract-Net round."""

    cell_id: int
    score: float
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            'cell_id': self.cell_id,
            'score': self.score,
            'components': dict(self.components),
        }


class _BaseContractNetPolicy(BaselinePolicy):
    """Shared logic for plant-side Contract-Net allocation baselines."""

    def __init__(self) -> None:
        self.last_bids: List[ContractNetBid] = []
        self.last_award_cell_id: int | None = None

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        valid_cells = list(info.get('valid_cells', []))
        if not valid_cells:
            self.last_bids = []
            self.last_award_cell_id = None
            return 0

        cell_summaries = info.get('cell_summaries', {})
        bids = [self._build_bid(cell_id, cell_summaries.get(cell_id, {})) for cell_id in valid_cells]
        bids.sort(key=lambda bid: (bid.score, bid.cell_id))
        self.last_bids = bids
        self.last_award_cell_id = bids[0].cell_id
        return bids[0].cell_id

    def _build_bid(self, cell_id: int, summary: Dict[str, Any]) -> ContractNetBid:
        active_jobs = float(summary.get('active_jobs', 0))
        busy_machines = float(summary.get('busy_machines', 0))
        down_machines = float(summary.get('down_machines', 0))
        backlog_jobs = float(summary.get('backlog_jobs', 0))
        completed_jobs = float(summary.get('completed_jobs', 0))

        components = {
            'active_jobs': active_jobs,
            'busy_machines': 0.5 * busy_machines,
            'down_machines': 4.0 * down_machines,
            'backlog_jobs': 0.25 * backlog_jobs,
            'completed_jobs_credit': -0.05 * completed_jobs,
        }
        components.update(self._resource_components(summary))
        score = float(sum(components.values()))
        return ContractNetBid(cell_id=cell_id, score=score, components=components)

    def _resource_components(self, summary: Dict[str, Any]) -> Dict[str, float]:
        return {}


class ContractNetGreedyPolicy(_BaseContractNetPolicy):
    """Greedy Contract-Net allocator using local load bids only."""

    @property
    def name(self) -> str:
        return 'contract_net_greedy'


class ContractNetBudgetedPolicy(_BaseContractNetPolicy):
    """Contract-Net allocator with current and projected budget penalties."""

    def _resource_components(self, summary: Dict[str, Any]) -> Dict[str, float]:
        energy = float(summary.get('energy_kwh', 0.0))
        carbon = float(summary.get('carbon_kg', 0.0))
        active_jobs = float(summary.get('active_jobs', 0.0))
        proc_scale = float(summary.get('processing_time_scale', 1.0))
        energy_cap = max(1.0, float(summary.get('budget_cap_energy_kwh', 1.0)))
        carbon_cap = max(1.0, float(summary.get('budget_cap_carbon_kg', 1.0)))
        wip_cap = max(1.0, float(summary.get('budget_cap_wip', 1.0)))

        remaining_energy = max(1.0, energy_cap - energy)
        remaining_carbon = max(1.0, carbon_cap - carbon)
        projected_energy = 120.0 * proc_scale
        projected_carbon = 72.0 * proc_scale

        return {
            'wip_pressure': 2.0 * (active_jobs / wip_cap),
            'energy_pressure': 1.5 * (energy / energy_cap),
            'carbon_pressure': 1.5 * (carbon / carbon_cap),
            'projected_energy_risk': 2.5 * (projected_energy / remaining_energy),
            'projected_carbon_risk': 2.5 * (projected_carbon / remaining_carbon),
        }

    @property
    def name(self) -> str:
        return 'contract_net_budgeted'


class ContractNetThroughputPolicy(_BaseContractNetPolicy):
    """Contract-Net allocator that prices local load through cell processing speed."""

    def _build_bid(self, cell_id: int, summary: Dict[str, Any]) -> ContractNetBid:
        active_jobs = float(summary.get('active_jobs', 0))
        busy_machines = float(summary.get('busy_machines', 0))
        down_machines = float(summary.get('down_machines', 0))
        backlog_jobs = float(summary.get('backlog_jobs', 0))
        completed_jobs = float(summary.get('completed_jobs', 0))
        proc_scale = float(summary.get('processing_time_scale', 1.0))

        components = {
            'speed_weighted_load': proc_scale * (1.0 + active_jobs + 0.5 * busy_machines + 0.25 * backlog_jobs),
            'down_machines': 4.0 * down_machines,
            'completed_jobs_credit': -0.05 * completed_jobs,
        }
        score = float(sum(components.values()))
        return ContractNetBid(cell_id=cell_id, score=score, components=components)

    @property
    def name(self) -> str:
        return 'contract_net_throughput'
