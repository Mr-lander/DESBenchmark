"""Baseline heuristics for FMS-Bench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from .simulator import DispatchOption


class BaselinePolicy(ABC):
    """Base class for baseline scheduling policies."""

    @abstractmethod
    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        """Select an action given observation and environment info."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return policy name."""
        pass


class RandomPolicy(BaselinePolicy):
    """Random action selection."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        dispatch_options = info.get("dispatch_options", [])
        if not dispatch_options:
            return 0
        return int(self.rng.integers(0, len(dispatch_options)))

    @property
    def name(self) -> str:
        return "random"


class FIFO_Policy(BaselinePolicy):
    """First-In-First-Out: select the job that arrived earliest."""

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        dispatch_options = info.get("dispatch_options", [])
        if not dispatch_options:
            return 0

        # dispatch_options can be list of dict or list of DispatchOption
        def get_job_id(opt):
            return opt.job_id if hasattr(opt, 'job_id') else opt['job_id']

        # Sort by job_id (assuming lower job_id arrived earlier)
        sorted_options = sorted(dispatch_options, key=get_job_id)
        # Return index of first option in original list
        first_option = sorted_options[0]
        first_job_id = get_job_id(first_option)
        for i, opt in enumerate(dispatch_options):
            if get_job_id(opt) == first_job_id:
                return i
        return 0

    @property
    def name(self) -> str:
        return "fifo"


class SPT_Policy(BaselinePolicy):
    """Shortest Processing Time: select job-machine pair with minimum processing time."""

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        dispatch_options = info.get("dispatch_options", [])
        if not dispatch_options:
            return 0

        # dispatch_options can be list of dict or list of DispatchOption
        def get_proc_time(opt):
            return opt.proc_time if hasattr(opt, 'proc_time') else opt['proc_time']

        # Sort by processing time
        sorted_options = sorted(dispatch_options, key=get_proc_time)
        first_option = sorted_options[0]
        first_proc_time = get_proc_time(first_option)

        # Return index of first option with minimum proc_time
        for i, opt in enumerate(dispatch_options):
            if get_proc_time(opt) == first_proc_time:
                return i
        return 0

    @property
    def name(self) -> str:
        return "spt"


class LPT_Policy(BaselinePolicy):
    """Longest Processing Time: select job-machine pair with maximum processing time."""

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        dispatch_options = info.get("dispatch_options", [])
        if not dispatch_options:
            return 0

        def get_proc_time(opt):
            return opt.proc_time if hasattr(opt, 'proc_time') else opt['proc_time']

        # Sort by processing time (descending)
        sorted_options = sorted(dispatch_options, key=lambda x: -get_proc_time(x))
        first_option = sorted_options[0]
        first_proc_time = get_proc_time(first_option)

        for i, opt in enumerate(dispatch_options):
            if get_proc_time(opt) == first_proc_time:
                return i
        return 0

    @property
    def name(self) -> str:
        return "lpt"


class EDD_Policy(BaselinePolicy):
    """Earliest Due Date: select job with earliest due time."""

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        dispatch_options = info.get("dispatch_options", [])
        if not dispatch_options:
            return 0

        def get_job_id(opt):
            return opt.job_id if hasattr(opt, 'job_id') else opt['job_id']

        # Need to get job due times from info or maintain state
        # For now, fall back to FIFO
        sorted_options = sorted(dispatch_options, key=get_job_id)
        first_option = sorted_options[0]
        first_job_id = get_job_id(first_option)

        for i, opt in enumerate(dispatch_options):
            if get_job_id(opt) == first_job_id:
                return i
        return 0

    @property
    def name(self) -> str:
        return "edd"


class LeastFlexible_Policy(BaselinePolicy):
    """Least Flexible Job: select job with fewest available machine options."""

    def __init__(self, job_options: Optional[Dict[int, int]] = None) -> None:
        """
        Initialize with job flexibility info.

        Args:
            job_options: Mapping from job_id to number of available machines
        """
        self.job_options = job_options or {}

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        dispatch_options = info.get("dispatch_options", [])
        if not dispatch_options:
            return 0

        def get_job_id(opt):
            return opt.job_id if hasattr(opt, 'job_id') else opt['job_id']

        # Group by job and count options
        job_option_counts: Dict[int, int] = {}
        for opt in dispatch_options:
            job_id = get_job_id(opt)
            job_option_counts[job_id] = job_option_counts.get(job_id, 0) + 1

        # Select job with fewest options
        min_options = min(job_option_counts.values())
        least_flexible_jobs = [jid for jid, cnt in job_option_counts.items() if cnt == min_options]

        # Among least flexible, pick by job_id
        target_job = min(least_flexible_jobs)

        # Find first option for this job
        for i, opt in enumerate(dispatch_options):
            if get_job_id(opt) == target_job:
                return i
        return 0

    @property
    def name(self) -> str:
        return "least_flexible"


class EnergyAware_Policy(BaselinePolicy):
    """Energy-aware: prefer machines that are already busy (avoid idle power waste)."""

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        dispatch_options = info.get("dispatch_options", [])
        if not dispatch_options:
            return 0

        def get_machine_id(opt):
            return opt.machine_id if hasattr(opt, 'machine_id') else opt['machine_id']

        # Get machine states
        machines = info.get("machines", {})

        # Score each option
        scored_options: List[tuple[int, float]] = []
        for i, opt in enumerate(dispatch_options):
            machine_id = get_machine_id(opt)
            machine_info = machines.get(machine_id, {})
            status = machine_info.get("status", "idle")

            # Prefer busy machines (already consuming active power)
            if status == "busy":
                score = 0.0  # Best
            elif status == "idle":
                score = 1.0  # Would transition to active
            else:  # down
                score = 10.0  # Avoid

            scored_options.append((i, score))

        # Select minimum score
        scored_options.sort(key=lambda x: x[1])
        return scored_options[0][0]

    @property
    def name(self) -> str:
        return "energy_aware"


class LeastLoadedCellPolicy(BaselinePolicy):
    """Plant-level heuristic: release to the least-loaded valid cell."""

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        valid_cells = list(info.get("valid_cells", []))
        if not valid_cells:
            return 0

        cell_summaries = info.get("cell_summaries", {})

        def score(cell_id: int) -> tuple[int, int, int, int]:
            summary = cell_summaries.get(cell_id, {})
            active = int(summary.get("active_jobs", 0))
            busy = int(summary.get("busy_machines", 0))
            backlog = int(summary.get("backlog_jobs", 0))
            return (active, busy, -backlog, cell_id)

        return min(valid_cells, key=score)

    @property
    def name(self) -> str:
        return "least_loaded_cell"


class ThroughputAwareCellPolicy(BaselinePolicy):
    """Plant-level heuristic that favors faster compatible cells under similar load."""

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        valid_cells = list(info.get("valid_cells", []))
        if not valid_cells:
            return 0

        cell_summaries = info.get("cell_summaries", {})

        def score(cell_id: int) -> tuple[float, int]:
            summary = cell_summaries.get(cell_id, {})
            proc_scale = float(summary.get("processing_time_scale", 1.0))
            active = float(summary.get("active_jobs", 0))
            busy = float(summary.get("busy_machines", 0))
            backlog = float(summary.get("backlog_jobs", 0))
            composite = proc_scale * (1.0 + active + 0.5 * busy + 0.25 * backlog)
            return (composite, cell_id)

        return min(valid_cells, key=score)

    @property
    def name(self) -> str:
        return "throughput_aware_cell"


class FlatMaskFirstPolicy(BaselinePolicy):
    """Flat-style plant baseline using only observation and action mask."""

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        action_mask = info.get("action_mask")
        if action_mask is None:
            valid_cells = list(info.get("valid_cells", []))
            return int(valid_cells[0]) if valid_cells else 0
        mask = np.asarray(action_mask, dtype=np.int32)
        valid = np.flatnonzero(mask > 0)
        return int(valid[0]) if len(valid) else 0

    @property
    def name(self) -> str:
        return "flat_mask_first"


class CompositePolicy(BaselinePolicy):
    """Composite policy that combines multiple heuristics with weights."""

    def __init__(
        self,
        policies: List[BaselinePolicy],
        weights: Optional[List[float]] = None,
    ) -> None:
        if len(policies) == 0:
            raise ValueError("Must provide at least one policy")
        self.policies = policies
        self.weights = weights or [1.0] * len(policies)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def select_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        dispatch_options = info.get("dispatch_options", [])
        if not dispatch_options:
            return 0

        # Each policy votes for an action
        votes: Dict[int, float] = {}
        for policy, weight in zip(self.policies, self.weights):
            action = policy.select_action(observation, info)
            votes[action] = votes.get(action, 0.0) + weight

        # Select action with highest votes
        best_action = max(votes.keys(), key=lambda a: votes[a])
        return best_action

    @property
    def name(self) -> str:
        names = "+".join(p.name for p in self.policies)
        return f"composite({names})"


def get_baseline(name: str, **kwargs: Any) -> BaselinePolicy:
    """Factory function to create baseline policies by name."""
    name = name.lower()

    if name == "random":
        return RandomPolicy(**kwargs)
    elif name == "fifo":
        return FIFO_Policy(**kwargs)
    elif name == "spt":
        return SPT_Policy(**kwargs)
    elif name == "lpt":
        return LPT_Policy(**kwargs)
    elif name == "edd":
        return EDD_Policy(**kwargs)
    elif name == "least_flexible":
        return LeastFlexible_Policy(**kwargs)
    elif name == "energy_aware":
        return EnergyAware_Policy(**kwargs)
    elif name == "least_loaded_cell":
        return LeastLoadedCellPolicy(**kwargs)
    elif name == "flat_mask_first":
        return FlatMaskFirstPolicy(**kwargs)
    elif name == "throughput_aware_cell":
        return ThroughputAwareCellPolicy(**kwargs)
    elif name == "contract_net_greedy":
        from .mas_baselines import ContractNetGreedyPolicy
        return ContractNetGreedyPolicy(**kwargs)
    elif name == "contract_net_budgeted":
        from .mas_baselines import ContractNetBudgetedPolicy
        return ContractNetBudgetedPolicy(**kwargs)
    elif name == "contract_net_throughput":
        from .mas_baselines import ContractNetThroughputPolicy
        return ContractNetThroughputPolicy(**kwargs)
    else:
        raise ValueError(f"Unknown baseline policy: {name}")
