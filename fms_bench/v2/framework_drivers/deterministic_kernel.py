from __future__ import annotations

from typing import Any, Mapping


def _mask_valid_ids(agent_view: Mapping[str, object]) -> list[int]:
    return [
        int(index)
        for index, enabled in enumerate(list(agent_view.get("action_mask", [])))
        if int(enabled) > 0
    ]


def _visible_area_summaries(agent_view: Mapping[str, object]) -> dict[int, dict[str, Any]]:
    raw = dict(agent_view.get("visible_area_summaries", agent_view.get("area_summaries", {})))
    return {int(area_id): dict(summary) for area_id, summary in raw.items()}


def _visible_backlog_candidates(agent_view: Mapping[str, object]) -> list[dict[str, Any]]:
    return [dict(candidate) for candidate in list(agent_view.get("visible_backlog_candidates", ()))]


def _visible_cell_summaries(agent_view: Mapping[str, object]) -> dict[int, dict[str, Any]]:
    raw = dict(
        agent_view.get(
            "visible_cell_summaries",
            agent_view.get("execution_cell_summaries", agent_view.get("cell_summaries", {})),
        )
    )
    return {int(cell_id): dict(summary) for cell_id, summary in raw.items()}


def _visible_machine_summaries(agent_view: Mapping[str, object]) -> dict[int, dict[str, Any]]:
    raw = dict(agent_view.get("visible_machine_summaries", {}))
    return {int(machine_id): dict(summary) for machine_id, summary in raw.items()}


def _first_valid_action(agent_view: Mapping[str, object]) -> int:
    valid = _mask_valid_ids(agent_view)
    return int(valid[0]) if valid else 0


class DeterministicCoordinationKernel:
    """Shared legal-view-only scoring kernel for framework-backed ablation families."""

    @staticmethod
    def select_backlog(
        agent_view: Mapping[str, object],
        *,
        selection_counts: Mapping[int, int] | None = None,
    ) -> int:
        valid_backlog_ids = {
            int(job_id) for job_id in list(agent_view.get("valid_backlog_ids", ()))
        }
        candidates = _visible_backlog_candidates(agent_view)
        if not candidates:
            return _first_valid_action(agent_view)
        counts = dict(selection_counts or {})
        valid_indices = [
            index
            for index, candidate in enumerate(candidates)
            if int(candidate.get("job_id", -1)) in valid_backlog_ids
        ]
        if not valid_indices:
            return 0
        return int(
            min(
                valid_indices,
                key=lambda index: (
                    int(candidates[index].get("stage_id", 0)),
                    len(list(candidates[index].get("valid_areas", ()))),
                    int(counts.get(int(candidates[index].get("job_id", -1)), 0)),
                    int(candidates[index].get("job_id", -1)),
                ),
            )
        )

    @staticmethod
    def select_area(
        agent_view: Mapping[str, object],
        *,
        area_pressures: Mapping[int, float] | None = None,
        selection_counts: Mapping[int, int] | None = None,
        pressure_weight: float = 0.0,
        routing_penalty: float = 0.0,
    ) -> int:
        valid_areas = [int(area_id) for area_id in agent_view.get("valid_areas", _mask_valid_ids(agent_view))]
        if not valid_areas:
            return 0
        area_summaries = _visible_area_summaries(agent_view)
        pressures = dict(area_pressures or {})
        counts = dict(selection_counts or {})
        return int(
            min(
                valid_areas,
                key=lambda area_id: DeterministicCoordinationKernel._area_score(
                    area_summaries.get(area_id, {}),
                    area_id=area_id,
                    remembered_pressure=float(pressures.get(area_id, 0.0)),
                    pressure_weight=pressure_weight,
                    selection_count=int(counts.get(area_id, 0)),
                    routing_penalty=routing_penalty,
                ),
            )
        )

    @staticmethod
    def select_cell(
        agent_view: Mapping[str, object],
        *,
        cell_pressures: Mapping[int, float] | None = None,
        selection_counts: Mapping[int, int] | None = None,
        pressure_weight: float = 0.0,
        routing_penalty: float = 0.0,
    ) -> int:
        valid_cells = [
            int(cell_id)
            for cell_id in agent_view.get(
                "valid_cells",
                agent_view.get("valid_execution_cells", _mask_valid_ids(agent_view)),
            )
        ]
        if not valid_cells:
            return 0
        cell_summaries = _visible_cell_summaries(agent_view)
        pressures = dict(cell_pressures or {})
        counts = dict(selection_counts or {})
        return int(
            min(
                valid_cells,
                key=lambda cell_id: DeterministicCoordinationKernel._cell_score(
                    cell_summaries.get(cell_id, {}),
                    cell_id=cell_id,
                    remembered_pressure=float(pressures.get(cell_id, 0.0)),
                    pressure_weight=pressure_weight,
                    selection_count=int(counts.get(cell_id, 0)),
                    routing_penalty=routing_penalty,
                ),
            )
        )

    @staticmethod
    def select_dispatch(
        agent_view: Mapping[str, object],
        *,
        machine_pressures: Mapping[int, float] | None = None,
        disturbance_weight: float = 0.0,
    ) -> int:
        dispatch_options = list(agent_view.get("dispatch_options", ()))
        if not dispatch_options:
            return _first_valid_action(agent_view)
        machine_summaries = _visible_machine_summaries(agent_view)
        remembered = dict(machine_pressures or {})
        return int(
            min(
                range(len(dispatch_options)),
                key=lambda index: DeterministicCoordinationKernel._dispatch_score(
                    option=dict(dispatch_options[index]),
                    machine_summaries=machine_summaries,
                    remembered_pressure=float(
                        remembered.get(int(dict(dispatch_options[index]).get("machine_id", -1)), 0.0)
                    ),
                    disturbance_weight=disturbance_weight,
                ),
            )
        )

    @staticmethod
    def cell_disturbance_pressure(summary: Mapping[str, object]) -> float:
        active_jobs = float(summary.get("active_jobs", 0.0))
        busy_machines = float(summary.get("busy_machines", 0.0))
        backlog_jobs = float(summary.get("backlog_jobs", 0.0))
        down_machines = float(summary.get("down_machines", 0.0))
        return (
            4.0 * down_machines
            + 1.0 * busy_machines
            + 0.5 * backlog_jobs
            + 0.25 * active_jobs
        )

    @staticmethod
    def area_disturbance_pressure(summary: Mapping[str, object]) -> float:
        active_jobs = float(summary.get("active_jobs", 0.0))
        backlog_jobs = float(summary.get("backlog_jobs", 0.0))
        busy_machines = float(summary.get("busy_machines", 0.0))
        down_machines = float(summary.get("down_machines", 0.0))
        return (
            4.0 * down_machines
            + 0.75 * busy_machines
            + 0.5 * backlog_jobs
            + 0.25 * active_jobs
        )

    @staticmethod
    def machine_disturbance_pressure(summary: Mapping[str, object]) -> float:
        status = str(summary.get("status", "idle"))
        down_time = float(summary.get("down_time", 0.0))
        remaining_time = float(summary.get("remaining_time", 0.0))
        status_penalty = 0.0
        if status == "down":
            status_penalty = 4.0
        elif status == "busy":
            status_penalty = 1.5
        return status_penalty + 0.5 * down_time + 0.1 * remaining_time

    @staticmethod
    def area_pressures_from_cell_summaries(
        cell_summaries: Mapping[int, Mapping[str, object]],
    ) -> tuple[float, dict[int, float]]:
        per_cell = {
            int(cell_id): DeterministicCoordinationKernel.cell_disturbance_pressure(summary)
            for cell_id, summary in cell_summaries.items()
        }
        return float(max(per_cell.values(), default=0.0)), per_cell

    @staticmethod
    def machine_pressures_from_machine_summaries(
        machine_summaries: Mapping[int, Mapping[str, object]],
    ) -> dict[int, float]:
        return {
            int(machine_id): DeterministicCoordinationKernel.machine_disturbance_pressure(summary)
            for machine_id, summary in machine_summaries.items()
        }

    @staticmethod
    def _area_score(
        summary: Mapping[str, object],
        *,
        area_id: int,
        remembered_pressure: float,
        pressure_weight: float,
        selection_count: int,
        routing_penalty: float,
    ) -> tuple[float, float, float, int]:
        active_jobs = float(summary.get("active_jobs", 0.0))
        backlog_jobs = float(summary.get("backlog_jobs", 0.0))
        energy_kwh = float(summary.get("energy_kwh", 0.0))
        live_pressure = DeterministicCoordinationKernel.area_disturbance_pressure(summary)
        composite = (
            active_jobs
            + pressure_weight * (live_pressure + remembered_pressure)
            + routing_penalty * float(selection_count)
        )
        return (float(composite), backlog_jobs, energy_kwh, area_id)

    @staticmethod
    def _cell_score(
        summary: Mapping[str, object],
        *,
        cell_id: int,
        remembered_pressure: float,
        pressure_weight: float,
        selection_count: int,
        routing_penalty: float,
    ) -> tuple[float, int]:
        proc_scale = float(summary.get("processing_time_scale", 1.0))
        active_jobs = float(summary.get("active_jobs", 0.0))
        busy_machines = float(summary.get("busy_machines", 0.0))
        backlog_jobs = float(summary.get("backlog_jobs", 0.0))
        base = proc_scale * (1.0 + active_jobs + 0.5 * busy_machines + 0.25 * backlog_jobs)
        live_pressure = DeterministicCoordinationKernel.cell_disturbance_pressure(summary)
        composite = (
            base
            + pressure_weight * (live_pressure + remembered_pressure)
            + routing_penalty * float(selection_count)
        )
        return (float(composite), cell_id)

    @staticmethod
    def _dispatch_score(
        *,
        option: Mapping[str, object],
        machine_summaries: Mapping[int, Mapping[str, object]],
        remembered_pressure: float,
        disturbance_weight: float,
    ) -> tuple[float, int]:
        machine_id = int(option.get("machine_id", -1))
        proc_time = float(option.get("proc_time", 0.0))
        live_pressure = DeterministicCoordinationKernel.machine_disturbance_pressure(
            machine_summaries.get(machine_id, {})
        )
        composite = proc_time + disturbance_weight * (live_pressure + remembered_pressure)
        return (float(composite), machine_id)
