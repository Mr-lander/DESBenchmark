"""First minimal inter-cell tree benchmark environment."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from gymnasium import spaces

from .arrival_profiles import schedule_online_arrival_batches
from .config import BenchmarkConfig
from .constraints import EnergyModel
from .data_loader import InstanceData
from .holonic_types import (
    AssignmentContract,
    AreaFeedback,
    CellFeedback,
    HolonicEvent,
    PlantAssembly,
    PlantCellConfig,
    TreeMessage,
)
from .models import Job, Operation, OperationOption
from .simulator import DispatchOption, EventDrivenSimulator
from .suite_builder import build_plant_assembly


class TreeIntercellEnv:
    """Minimal inter-cell routing env reusing the public Plant/Area/Cell boundary."""

    ROUTE_GRAMMAR = (
        (0, 1),
        (1, 2),
        (0, 2),
    )

    def __init__(
        self,
        cell_configs: Iterable[PlantCellConfig],
        area_to_cells: Mapping[int, Sequence[int]],
        config: Optional[BenchmarkConfig] = None,
        seed: Optional[int] = None,
        max_in_flight_jobs: int = 1,
        route_grammar: Optional[Sequence[Sequence[int]]] = None,
        backlog_top_k: int = 1,
        route_grammar_contract: str | None = None,
    ) -> None:
        self.config = config or BenchmarkConfig()
        self.base_seed = seed
        self.max_in_flight_jobs = max(1, int(max_in_flight_jobs))
        self.backlog_top_k = max(1, int(backlog_top_k))
        self.route_grammar_contract = "" if route_grammar_contract is None else str(route_grammar_contract)
        self.assembly: PlantAssembly = build_plant_assembly(list(cell_configs), config=self.config)
        self.area_to_cells = self._validate_area_to_cells(area_to_cells)
        self.area_ids = sorted(self.area_to_cells)
        self.area_index = {area_id: idx for idx, area_id in enumerate(self.area_ids)}
        self.index_to_area = {idx: area_id for area_id, idx in self.area_index.items()}
        self.cell_to_area = {
            cell_id: area_id
            for area_id, cell_ids in self.area_to_cells.items()
            for cell_id in cell_ids
        }
        self.route_grammar = self._validate_route_grammar(route_grammar)
        self.logical_jobs = self._build_intercell_jobs()
        combined_instance = InstanceData(
            jobs=self.logical_jobs,
            n_jobs=len(self.logical_jobs),
            n_machines=self.assembly.total_machine_count,
        )
        self.simulator = EventDrivenSimulator(combined_instance, self.config)
        self.simulator.register_external_event_handler("shared_backlog_arrival", self._handle_shared_backlog_arrival)
        self.simulator.register_external_event_handler("transport_start", self._handle_transport_start)
        self.simulator.register_external_event_handler("transport_complete", self._handle_transport_complete)
        self.energy_model = EnergyModel(
            peak_window=self.config.energy.peak_window,
            peak_price=self.config.energy.peak_price,
            offpeak_price=self.config.energy.offpeak_price,
        )

        self.action_space = spaces.Discrete(max(1, len(self.area_ids)))
        obs_dim = 3 + 6 * len(self.assembly.cells)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.template_job_count = len(self.logical_jobs)
        self.shared_backlog: List[int] = []
        self._future_shared_backlog_ids: List[int] = []
        self._shared_backlog_arrival_times: Dict[int, float] = {}
        self._realized_arrival_trace: List[Dict[str, Any]] = []
        self.blocked_backlog: List[int] = []
        self.assigned_clone_jobs: set[int] = set()
        self.assigned_templates: Dict[int, int] = {}
        self.route_stage_index: Dict[int, int] = {}
        self.route_previous_cell: Dict[int, Optional[int]] = {}
        self.current_assignment_cell: Dict[int, int] = {}
        self._job_blocking_started_at: Dict[int, float] = {}
        self._cell_assigned_jobs: Dict[int, set[int]] = {cell_id: set() for cell_id in self.assembly.cells}
        self._cell_inbound_jobs: Dict[int, set[int]] = {cell_id: set() for cell_id in self.assembly.cells}
        self._cell_in_transit_jobs: Dict[int, set[int]] = {cell_id: set() for cell_id in self.assembly.cells}
        self._cell_energy_kwh: Dict[int, float] = {}
        self._cell_energy_cost: Dict[int, float] = {}
        self._cell_carbon_kg: Dict[int, float] = {}
        self._cell_setup_time: Dict[int, float] = {}
        self._cell_setup_energy_kwh: Dict[int, float] = {}
        self._cell_transport_time_hours: Dict[int, float] = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._cell_transport_energy_kwh: Dict[int, float] = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._transport_moves_total = 0
        self._blocking_time_total = 0.0
        self._buffer_wait_time_total = 0.0
        self._contract_seq = 0
        self._last_contract: Optional[AssignmentContract] = None
        self._last_transition: Dict[str, Any] = {}
        self._terminated = False
        self._truncated = False
        self._invalid_actions = 0

    def _build_intercell_jobs(self) -> List[Job]:
        reference_cell = self.assembly.cells[min(self.assembly.cells)]
        logical_jobs: List[Job] = []
        if len(reference_cell.template_jobs) == 0:
            raise ValueError("TreeIntercellEnv requires at least one template job")
        eligible_template_indices = [
            template_idx
            for template_idx, template_job in enumerate(reference_cell.template_jobs)
            if len(template_job.operations) >= len(self.route_grammar)
        ]
        if not eligible_template_indices:
            raise ValueError(
                "TreeIntercellEnv requires at least one template job with enough operations for the configured route grammar"
            )
        for logical_job_id, template_idx in enumerate(eligible_template_indices):
            template_job = reference_cell.template_jobs[template_idx]
            operations: List[Operation] = []
            for stage_idx, candidate_cells in enumerate(self.route_grammar):
                source_op_idx = self._route_source_operation_index(
                    stage_idx=stage_idx,
                    source_operation_count=len(template_job.operations),
                )
                stage_options: List[OperationOption] = []
                for cell_id in candidate_cells:
                    cell_job = self.assembly.cells[cell_id].template_jobs[template_idx]
                    stage_op = cell_job.operations[source_op_idx]
                    stage_options.extend(
                        OperationOption(
                            machine_id=int(option.machine_id),
                            proc_time=float(option.proc_time),
                        )
                        for option in stage_op.options
                    )
                operations.append(Operation(op_id=stage_idx, options=stage_options))
            logical_jobs.append(
                Job(
                    job_id=logical_job_id,
                    operations=operations,
                    due_time=template_job.due_time,
                    release_time=0.0,
                    current_op_idx=0,
                    template_job_id=template_idx,
                    setup_family=template_job.setup_family,
                )
            )
        return logical_jobs

    def _route_source_operation_index(self, stage_idx: int, source_operation_count: int) -> int:
        if source_operation_count < len(self.route_grammar):
            raise ValueError("Inter-cell public routing requires enough source operations")
        if len(self.route_grammar) == 1:
            return 0
        scaled_position = stage_idx * (source_operation_count - 1) / (len(self.route_grammar) - 1)
        return int(round(scaled_position))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        del options
        self.simulator.reset(seed=seed if seed is not None else self.base_seed, schedule_job_arrivals=False)
        self.shared_backlog = []
        self._future_shared_backlog_ids = []
        self._shared_backlog_arrival_times = {}
        self._realized_arrival_trace = []
        self.blocked_backlog = []
        self.assigned_clone_jobs = set()
        self.assigned_templates = {}
        self.route_stage_index = {job.job_id: 0 for job in self.logical_jobs}
        self.route_previous_cell = {job.job_id: None for job in self.logical_jobs}
        self.current_assignment_cell = {}
        self._job_blocking_started_at = {}
        self._cell_assigned_jobs = {cell_id: set() for cell_id in self.assembly.cells}
        self._cell_inbound_jobs = {cell_id: set() for cell_id in self.assembly.cells}
        self._cell_in_transit_jobs = {cell_id: set() for cell_id in self.assembly.cells}
        self._cell_energy_kwh = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._cell_energy_cost = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._cell_carbon_kg = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._cell_setup_time = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._cell_setup_energy_kwh = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._cell_transport_time_hours = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._cell_transport_energy_kwh = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._transport_moves_total = 0
        self._blocking_time_total = 0.0
        self._buffer_wait_time_total = 0.0
        self._contract_seq = 0
        self._last_contract = None
        self._terminated = False
        self._truncated = False
        self._invalid_actions = 0
        self._last_transition = {
            "time_start": 0.0,
            "time_end": 0.0,
            "events": [],
            "event_records": [],
            "deadlocked": False,
            "setup_time_step": 0.0,
            "energy_kwh_step": 0.0,
            "energy_cost_step": 0.0,
            "carbon_kg_step": 0.0,
        }
        self._initialize_shared_backlog_visibility()
        obs = self._build_observation()
        info = self._build_info(self._last_transition, invalid_action=False, contract=None)
        return obs, info

    def _initialize_shared_backlog_visibility(self) -> None:
        all_backlog_ids = [job.job_id for job in self.logical_jobs]
        if self.config.arrival.process in {"hpp_singleton", "nhpp_singleton", "nhpp_random_batch"}:
            initial_visible = max(
                0,
                min(int(self.config.arrival.initial_visible_backlog), len(all_backlog_ids)),
            )
            self.shared_backlog = list(all_backlog_ids[:initial_visible])
            future_backlog_ids = list(all_backlog_ids[initial_visible:])
            self._future_shared_backlog_ids = list(future_backlog_ids)
            for event in schedule_online_arrival_batches(
                future_backlog_ids,
                config=self.config,
                rng=self.simulator.rng,
                start_time=float(self.simulator.time),
            ):
                for shared_backlog_id in event.shared_backlog_ids:
                    self._shared_backlog_arrival_times[int(shared_backlog_id)] = float(event.time)
                payload = {
                    "shared_backlog_ids": ",".join(str(item) for item in event.shared_backlog_ids),
                    "arrival_event_id": int(event.event_id),
                    "arrival_batch_size": int(event.batch_size),
                    "arrival_process": str(event.process),
                }
                if event.batch_size == 1:
                    payload["shared_backlog_id"] = int(event.shared_backlog_ids[0])
                self.simulator.schedule(
                    "shared_backlog_arrival",
                    float(event.time),
                    payload,
                )
            return
        self.shared_backlog = list(all_backlog_ids)

    def observe(self, agent: str = "plant") -> np.ndarray:
        if agent == "plant":
            return self._build_observation()
        if agent.startswith("area_"):
            return self._build_area_observation(int(agent.split("_", 1)[1]))
        if agent.startswith("cell_"):
            return self._build_cell_observation(int(agent.split("_", 1)[1]))
        raise ValueError(f"Unknown agent for observe(): {agent}")

    def action_mask(self, agent: str = "plant") -> np.ndarray:
        if agent == "plant":
            mask = np.zeros(self.action_space.n, dtype=np.int8)
            for area_id in self._valid_release_areas():
                mask[self.area_index[area_id]] = 1
            return mask
        if agent.startswith("area_"):
            area_id = int(agent.split("_", 1)[1])
            mask = np.zeros(len(self.assembly.cells), dtype=np.int8)
            valid_cells = set(self._valid_release_cells())
            for cell_id in self.area_to_cells[area_id]:
                if cell_id in valid_cells:
                    mask[cell_id] = 1
            return mask
        if agent.startswith("cell_"):
            return np.ones(1, dtype=np.int8)
        raise ValueError(f"Unknown agent for action_mask(): {agent}")

    def agents_to_act(self) -> List[str]:
        if self._terminated or self._truncated:
            return []
        valid_areas = self._valid_release_areas()
        if not valid_areas:
            return []
        return ["plant"]

    def _stage_candidates(self, job_id: int) -> tuple[int, ...]:
        stage_idx = int(self.simulator.jobs[job_id].current_op_idx)
        if not 0 <= stage_idx < len(self.route_grammar):
            return ()
        return tuple(int(cell_id) for cell_id in self.route_grammar[stage_idx])

    def _reserved_inbound_jobs(self, cell_id: int) -> int:
        return len(self._cell_inbound_jobs[cell_id].union(self._cell_in_transit_jobs[cell_id]))

    def _admissible_cells_for_job(self, job_id: int) -> List[int]:
        valid: List[int] = []
        for cell_id in self._stage_candidates(job_id):
            cell = self.assembly.cells[cell_id]
            if self._reserved_inbound_jobs(cell_id) >= cell.inbound_buffer_cap:
                continue
            valid.append(cell_id)
        return valid

    def _has_active_route_work(self) -> bool:
        return bool(self._open_route_job_ids())

    def _open_route_job_ids(self) -> set[int]:
        return {
            int(job_id)
            for job_id in self.assigned_clone_jobs
            if int(job_id) not in self.simulator.completed_jobs
        }

    def _valid_release_cells(self) -> List[int]:
        if not self.shared_backlog:
            return []
        job_id = int(self.shared_backlog[0])
        return self._valid_release_cells_for_job(job_id)

    def _valid_release_cells_for_job(self, job_id: int) -> List[int]:
        if (
            job_id not in self.assigned_clone_jobs
            and len(self._open_route_job_ids()) >= self.max_in_flight_jobs
        ):
            return []
        return self._admissible_cells_for_job(job_id)

    def _valid_release_areas(self) -> List[int]:
        if not self.shared_backlog:
            return []
        return self._valid_release_areas_for_job(int(self.shared_backlog[0]))

    def _valid_release_areas_for_job(self, job_id: int) -> List[int]:
        valid_cells = set(self._valid_release_cells_for_job(job_id))
        return [
            area_id
            for area_id in self.area_ids
            if any(cell_id in valid_cells for cell_id in self.area_to_cells[area_id])
        ]

    def visible_backlog_candidates(self) -> List[Dict[str, Any]]:
        if not self.shared_backlog:
            return []
        candidates: List[Dict[str, Any]] = []
        visible_job_ids = [int(job_id) for job_id in self.shared_backlog[: self.backlog_top_k]]
        for backlog_index, job_id in enumerate(visible_job_ids):
            valid_cells = set(self._valid_release_cells_for_job(job_id))
            valid_cells_by_area = {
                int(area_id): [
                    int(cell_id)
                    for cell_id in self.area_to_cells[area_id]
                    if int(cell_id) in valid_cells
                ]
                for area_id in self.area_ids
                if any(int(cell_id) in valid_cells for cell_id in self.area_to_cells[area_id])
            }
            candidates.append(
                {
                    "job_id": int(job_id),
                    "backlog_index": int(backlog_index),
                    "stage_id": int(self.simulator.jobs[job_id].current_op_idx),
                    "candidate_cells": [int(cell_id) for cell_id in self._stage_candidates(job_id)],
                    "valid_areas": [int(area_id) for area_id in valid_cells_by_area],
                    "valid_cells_by_area": valid_cells_by_area,
                    "is_head_of_line": backlog_index == 0,
                }
            )
        return candidates

    def valid_backlog_ids(self) -> List[int]:
        return [
            int(candidate["job_id"])
            for candidate in self.visible_backlog_candidates()
            if list(candidate.get("valid_areas", ()))
        ]

    def _release_next_job(
        self,
        cell_id: int,
        job_id: int | None = None,
    ) -> AssignmentContract:
        if not self.shared_backlog:
            raise RuntimeError("No remaining inter-cell route items")
        selected_job_id = int(self.shared_backlog[0] if job_id is None else job_id)
        if selected_job_id not in self.shared_backlog:
            raise RuntimeError(f"Requested backlog job is not currently in shared backlog: {selected_job_id}")
        self.shared_backlog.remove(selected_job_id)
        stage_idx = int(self.simulator.jobs[selected_job_id].current_op_idx)
        self.assigned_clone_jobs.add(selected_job_id)
        self.assigned_templates[selected_job_id] = cell_id
        self.current_assignment_cell[selected_job_id] = cell_id
        self._cell_assigned_jobs[cell_id].add(selected_job_id)
        self._cell_in_transit_jobs[cell_id].add(selected_job_id)
        cell = self.assembly.cells[cell_id]
        transport_time_hours = float(cell.ingress_transport_time_hours)
        transport_energy_kwh = float(cell.ingress_transport_energy_kwh)
        self._cell_transport_time_hours[cell_id] += transport_time_hours
        self._cell_transport_energy_kwh[cell_id] += transport_energy_kwh
        self._transport_moves_total += 1
        from_cell = self.route_previous_cell.get(selected_job_id)
        self.simulator.schedule(
            "transport_start",
            self.simulator.time,
            {
                "job_id": selected_job_id,
                "stage_id": stage_idx,
                "from_cell": -1 if from_cell is None else int(from_cell),
                "to_cell": int(cell_id),
                "transport_time_hours": transport_time_hours,
                "transport_energy_kwh": transport_energy_kwh,
            },
        )
        self.simulator.schedule(
            "transport_complete",
            self.simulator.time + transport_time_hours,
            {
                "job_id": selected_job_id,
                "stage_id": stage_idx,
                "from_cell": -1 if from_cell is None else int(from_cell),
                "to_cell": int(cell_id),
                "transport_time_hours": transport_time_hours,
                "transport_energy_kwh": transport_energy_kwh,
            },
        )

        self._contract_seq += 1
        job = self.simulator.jobs[selected_job_id]
        self.route_stage_index[selected_job_id] = stage_idx
        contract = AssignmentContract(
            contract_id=self._contract_seq,
            shared_backlog_id=selected_job_id,
            template_job_index=job.template_job_id or selected_job_id,
            cell_id=cell_id,
            job_id=selected_job_id,
            released_at=self.simulator.time,
            due_time=job.due_time,
            horizon_remaining=max(0.0, float(self.config.horizon_hours) - self.simulator.time),
        )
        self._last_contract = contract
        return contract

    def _handle_transport_start(self, payload: Dict[str, int | float | str]) -> bool:
        return True

    def _handle_shared_backlog_arrival(self, payload: Dict[str, int | float | str]) -> bool:
        if "shared_backlog_ids" in payload:
            shared_backlog_ids = [
                int(item)
                for item in str(payload["shared_backlog_ids"]).split(",")
                if str(item).strip()
            ]
        else:
            shared_backlog_ids = [int(payload["shared_backlog_id"])]
        admitted_ids: List[int] = []
        for shared_backlog_id in shared_backlog_ids:
            if shared_backlog_id in self.assigned_clone_jobs:
                continue
            if shared_backlog_id in self.shared_backlog:
                continue
            if shared_backlog_id not in self._future_shared_backlog_ids:
                continue
            self._future_shared_backlog_ids.remove(shared_backlog_id)
            self._shared_backlog_arrival_times.pop(shared_backlog_id, None)
            self.shared_backlog.append(shared_backlog_id)
            admitted_ids.append(shared_backlog_id)
        if not admitted_ids:
            return False
        self.shared_backlog.sort()
        self._realized_arrival_trace.append(
            {
                "time": float(self.simulator.time),
                "arrival_event_id": int(payload.get("arrival_event_id", len(self._realized_arrival_trace))),
                "process": str(payload.get("arrival_process", self.config.arrival.process)),
                "batch_size": int(payload.get("arrival_batch_size", len(admitted_ids))),
                "shared_backlog_ids": list(admitted_ids),
            }
        )
        return True

    def _handle_transport_complete(self, payload: Dict[str, int | float | str]) -> bool:
        job_id = int(payload["job_id"])
        cell_id = int(payload["to_cell"])
        self._cell_in_transit_jobs[cell_id].discard(job_id)
        self._cell_inbound_jobs[cell_id].add(job_id)
        return self.simulator.activate_job(
            job_id,
            release_time=self.simulator.time,
            arrival_payload={
                "cell_id": cell_id,
                "arrival_source": "inter_cell_transport",
                "stage_id": int(payload["stage_id"]),
            },
        )

    def dispatch_job_to_machine(self, job_id: int, machine_id: int) -> bool:
        dispatched = self.simulator.dispatch(job_id, machine_id)
        if not dispatched:
            return False
        cell_id = self.assembly.machine_to_cell.get(machine_id)
        if cell_id is not None:
            self._cell_inbound_jobs[cell_id].discard(job_id)
        return True

    def _dispatch_options_for_cell(self, cell_id: int) -> List[DispatchOption]:
        return [
            option
            for option in self.simulator.get_dispatch_options()
            if self.assembly.machine_to_cell.get(option.machine_id) == int(cell_id)
            and self.current_assignment_cell.get(option.job_id) == int(cell_id)
        ]

    def _runtime_dispatch_options(self) -> List[DispatchOption]:
        return [
            option
            for option in self.simulator.get_dispatch_options()
            if self.current_assignment_cell.get(option.job_id)
            == self.assembly.machine_to_cell.get(option.machine_id)
        ]

    def _advance_until_plant_decision(self) -> Dict[str, Any]:
        return self._advance_until_decision(auto_dispatch=False, stop_on_dispatch=False)

    def _advance_until_decision(
        self,
        *,
        auto_dispatch: bool,
        stop_on_dispatch: bool,
        dispatch_scope: str = "raw",
    ) -> Dict[str, Any]:
        del auto_dispatch
        energy_kwh_step = 0.0
        energy_cost_step = 0.0
        carbon_kg_step = 0.0
        setup_time_step = 0.0
        event_log: List[str] = []
        event_records: List[Dict[str, Any]] = []
        start_time = self.simulator.time
        deadlocked = False

        if dispatch_scope == "runtime":
            dispatch_options_provider = self._runtime_dispatch_options
        elif dispatch_scope == "raw":
            dispatch_options_provider = self.simulator.get_dispatch_options
        else:
            raise ValueError(f"Unsupported dispatch_scope: {dispatch_scope}")

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
            promoted_records = self._promote_blocked_jobs_if_admissible()
            if promoted_records:
                event_log.extend(str(record["kind"]) for record in promoted_records)
                event_records.extend(promoted_records)
            if self._is_done():
                break
            if self._valid_release_cells():
                break
            dispatch_options = dispatch_options_provider()
            if dispatch_options and stop_on_dispatch:
                break

            transition = self.simulator.run_until_decision(
                advance_hook=on_advance,
                stop_condition=lambda: bool(self._valid_release_cells())
                or (stop_on_dispatch and bool(dispatch_options_provider())),
                dispatch_condition=lambda: stop_on_dispatch
                and bool(dispatch_options_provider()),
            )
            event_log.extend(list(transition["events"]))
            event_records.extend(list(transition.get("event_records", [])))
            for event_record in transition.get("event_records", []):
                kind = str(event_record.get("kind", ""))
                payload = dict(event_record.get("payload", {}))
                if kind == "setup_complete":
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
                if kind == "finish":
                    for synthetic_record in self._handle_stage_finish(payload):
                        event_log.append(str(synthetic_record["kind"]))
                        event_records.append(synthetic_record)
            if (
                float(transition["time_end"]) == float(transition["time_start"])
                and not transition["events"]
                and not self._is_done()
            ):
                deadlocked = bool(self.blocked_backlog) and not bool(self.shared_backlog)
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

    def _handle_stage_finish(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        job_id = int(payload["job_id"])
        machine_id = int(payload["machine_id"])
        cell_id = int(self.assembly.machine_to_cell[machine_id])
        self.route_previous_cell[job_id] = cell_id
        self._cell_assigned_jobs[cell_id].discard(job_id)
        self.current_assignment_cell.pop(job_id, None)
        self._cell_inbound_jobs[cell_id].discard(job_id)
        self._cell_in_transit_jobs[cell_id].discard(job_id)
        if job_id in self.simulator.completed_jobs:
            self.route_stage_index[job_id] = len(self.route_grammar)
            return []
        self.simulator.ready_jobs.discard(job_id)
        self.route_stage_index[job_id] = int(self.simulator.jobs[job_id].current_op_idx)
        stage_id = int(self.route_stage_index[job_id])
        admissible_cells = self._admissible_cells_for_job(job_id)
        if admissible_cells:
            self.shared_backlog.insert(0, job_id)
            return [
                {
                    "time": float(self.simulator.time),
                    "kind": "stage_release_ready",
                    "payload": {
                        "job_id": int(job_id),
                        "stage_id": stage_id,
                        "from_cell": int(cell_id),
                        "candidate_cells": ",".join(str(item) for item in self._stage_candidates(job_id)),
                        "admissible_cells": ",".join(str(item) for item in admissible_cells),
                        "release_scope": "post_operation_routing",
                    },
                }
            ]
        self.blocked_backlog.append(job_id)
        self._job_blocking_started_at[job_id] = float(self.simulator.time)
        blocked_payload = {
            "job_id": int(job_id),
            "stage_id": stage_id,
            "from_cell": int(cell_id),
            "candidate_cells": ",".join(str(item) for item in self._stage_candidates(job_id)),
            "blocking_reason": "downstream_inbound_buffer_full",
        }
        return [
            {
                "time": float(self.simulator.time),
                "kind": "buffer_wait_start",
                "payload": dict(blocked_payload),
            },
            {
                "time": float(self.simulator.time),
                "kind": "blocking_start",
                "payload": dict(blocked_payload),
            },
        ]

    def _promote_blocked_jobs_if_admissible(self) -> List[Dict[str, Any]]:
        if not self.blocked_backlog:
            return []
        promoted_jobs: List[int] = []
        promoted_records: List[Dict[str, Any]] = []
        for job_id in list(self.blocked_backlog):
            admissible_cells = self._admissible_cells_for_job(job_id)
            if not admissible_cells:
                continue
            promoted_jobs.append(job_id)
            start_time = float(self._job_blocking_started_at.pop(job_id, self.simulator.time))
            blocked_duration = max(0.0, float(self.simulator.time) - start_time)
            self._blocking_time_total += blocked_duration
            self._buffer_wait_time_total += blocked_duration
            stage_id = int(self.route_stage_index.get(job_id, self.simulator.jobs[job_id].current_op_idx))
            from_cell = self.route_previous_cell.get(job_id)
            payload = {
                "job_id": int(job_id),
                "stage_id": stage_id,
                "from_cell": -1 if from_cell is None else int(from_cell),
                "candidate_cells": ",".join(str(item) for item in self._stage_candidates(job_id)),
                "admissible_cells": ",".join(str(item) for item in admissible_cells),
                "blocked_duration_hours": blocked_duration,
            }
            promoted_records.extend(
                [
                    {
                        "time": float(self.simulator.time),
                        "kind": "buffer_admit",
                        "payload": dict(payload),
                    },
                    {
                        "time": float(self.simulator.time),
                        "kind": "blocking_end",
                        "payload": dict(payload),
                    },
                    {
                        "time": float(self.simulator.time),
                        "kind": "stage_release_ready",
                        "payload": {
                            **dict(payload),
                            "release_scope": "post_blocking_routing",
                        },
                    },
                ]
            )
        if not promoted_jobs:
            return []
        promoted_set = set(promoted_jobs)
        self.blocked_backlog = [job_id for job_id in self.blocked_backlog if job_id not in promoted_set]
        self.shared_backlog = promoted_jobs + self.shared_backlog
        return promoted_records

    def _cell_feedback(self, cell_id: int) -> CellFeedback:
        cell = self.assembly.cells[cell_id]
        cell_job_ids = set(self._cell_assigned_jobs[cell_id])
        ready_jobs = [jid for jid in self.simulator.ready_jobs if jid in cell_job_ids]
        processing_jobs = [
            jid
            for jid in (machine.current_job for machine in self.simulator.machines.values())
            if jid is not None and jid in cell_job_ids
        ]
        active_jobs = len(set(ready_jobs).union(processing_jobs))
        backlog_jobs = sum(1 for job_id in self.shared_backlog if cell_id in self._stage_candidates(job_id))
        completed_jobs = sum(
            1
            for job_id in self.simulator.completed_jobs
            if self.route_previous_cell.get(int(job_id)) == cell_id
        )
        inbound_jobs = self._reserved_inbound_jobs(cell_id)
        busy_machines = sum(
            1 for mid in cell.machine_ids if self.simulator.machines[mid].status in {"busy", "setup"}
        )
        down_machines = sum(
            1 for mid in cell.machine_ids if self.simulator.machines[mid].status == "down"
        )
        remaining_wip = max(0, cell.budget.wip_cap - active_jobs)
        remaining_inbound = max(0, cell.inbound_buffer_cap - inbound_jobs)
        remaining_energy = max(0.0, cell.budget.energy_cap_kwh - self._cell_energy_kwh[cell_id])
        remaining_carbon = max(0.0, cell.budget.carbon_cap_kg - self._cell_carbon_kg[cell_id])
        energy_overshoot = max(0.0, self._cell_energy_kwh[cell_id] - cell.budget.energy_cap_kwh)
        carbon_overshoot = max(0.0, self._cell_carbon_kg[cell_id] - cell.budget.carbon_cap_kg)
        return CellFeedback(
            cell_id=cell_id,
            label=cell.label,
            family_key=cell.family_key,
            backlog_jobs=backlog_jobs,
            released_jobs=len(self._cell_assigned_jobs[cell_id]),
            active_jobs=active_jobs,
            completed_jobs=completed_jobs,
            ready_jobs=len(ready_jobs),
            inbound_jobs=inbound_jobs,
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
            inbound_buffer_remaining=remaining_inbound,
            energy_overshoot_kwh=energy_overshoot,
            carbon_overshoot_kg=carbon_overshoot,
            processing_time_scale=cell.processing_time_scale,
        )

    def _area_feedback(self, area_id: int) -> AreaFeedback:
        cell_feedbacks = [self._cell_feedback(cell_id) for cell_id in self.area_to_cells[area_id]]
        area_cell_ids = set(self.area_to_cells[area_id])
        backlog_jobs = sum(
            1
            for job_id in self.shared_backlog
            if any(cell_id in area_cell_ids for cell_id in self._stage_candidates(job_id))
        )
        completed_jobs = sum(
            1
            for job_id in self.simulator.completed_jobs
            if self.route_previous_cell.get(int(job_id)) in area_cell_ids
        )
        return AreaFeedback(
            area_id=area_id,
            label=f"area_{area_id}",
            cell_ids=list(self.area_to_cells[area_id]),
            backlog_jobs=backlog_jobs,
            active_jobs=sum(item.active_jobs for item in cell_feedbacks),
            completed_jobs=completed_jobs,
            energy_kwh=sum(item.energy_kwh for item in cell_feedbacks),
            carbon_kg=sum(item.carbon_kg for item in cell_feedbacks),
            energy_overshoot_kwh=sum(item.energy_overshoot_kwh for item in cell_feedbacks),
            carbon_overshoot_kg=sum(item.carbon_overshoot_kg for item in cell_feedbacks),
        )

    def _build_observation(self) -> np.ndarray:
        total_jobs = max(1, self.template_job_count)
        horizon = max(1.0, float(self.config.horizon_hours))
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[0] = float(min(1.0, self.simulator.time / horizon))
        obs[1] = float(min(1.0, len(self.assigned_clone_jobs) / total_jobs))
        obs[2] = float(min(1.0, len(self.simulator.completed_jobs) / total_jobs))
        for idx, cell_id in enumerate(sorted(self.assembly.cells)):
            feedback = self._cell_feedback(cell_id)
            base = 3 + idx * 6
            machine_count = max(1, len(self.assembly.cells[cell_id].machine_ids))
            obs[base] = float(min(1.0, feedback.backlog_jobs / total_jobs))
            obs[base + 1] = float(min(1.0, feedback.active_jobs / max(1, self.assembly.cells[cell_id].budget.wip_cap)))
            obs[base + 2] = float(min(1.0, feedback.busy_machines / machine_count))
            obs[base + 3] = float(min(1.0, feedback.down_machines / machine_count))
            obs[base + 4] = float(min(1.0, feedback.energy_kwh / max(1.0, self.assembly.cells[cell_id].budget.energy_cap_kwh)))
            obs[base + 5] = float(min(1.0, feedback.carbon_kg / max(1.0, self.assembly.cells[cell_id].budget.carbon_cap_kg)))
        return obs

    def _build_cell_observation(self, cell_id: int) -> np.ndarray:
        cell = self.assembly.cells[cell_id]
        feedback = self._cell_feedback(cell_id)
        machine_count = max(1, len(cell.machine_ids))
        obs = np.zeros(8, dtype=np.float32)
        obs[0] = float(min(1.0, feedback.backlog_jobs / max(1, self.template_job_count)))
        obs[1] = float(min(1.0, feedback.active_jobs / max(1, cell.budget.wip_cap)))
        obs[2] = float(min(1.0, feedback.completed_jobs / max(1, self.template_job_count)))
        obs[3] = float(min(1.0, feedback.busy_machines / machine_count))
        obs[4] = float(min(1.0, feedback.down_machines / machine_count))
        obs[5] = float(min(1.0, feedback.energy_kwh / max(1.0, cell.budget.energy_cap_kwh)))
        obs[6] = float(min(1.0, feedback.carbon_kg / max(1.0, cell.budget.carbon_cap_kg)))
        obs[7] = float(min(1.0, self.simulator.time / max(1.0, float(self.config.horizon_hours))))
        return obs

    def _build_area_observation(self, area_id: int) -> np.ndarray:
        feedback = self._area_feedback(area_id)
        cell_feedbacks = [self._cell_feedback(cell_id) for cell_id in self.area_to_cells[area_id]]
        area_machine_count = max(1, sum(len(self.assembly.cells[cell_id].machine_ids) for cell_id in self.area_to_cells[area_id]))
        area_wip_cap = max(1, sum(self.assembly.cells[cell_id].budget.wip_cap for cell_id in self.area_to_cells[area_id]))
        area_energy_cap = max(1.0, sum(self.assembly.cells[cell_id].budget.energy_cap_kwh for cell_id in self.area_to_cells[area_id]))
        area_carbon_cap = max(1.0, sum(self.assembly.cells[cell_id].budget.carbon_cap_kg for cell_id in self.area_to_cells[area_id]))
        busy_machines = sum(item.busy_machines for item in cell_feedbacks)
        down_machines = sum(item.down_machines for item in cell_feedbacks)
        obs = np.zeros(8, dtype=np.float32)
        obs[0] = float(min(1.0, feedback.backlog_jobs / max(1, self.template_job_count)))
        obs[1] = float(min(1.0, feedback.active_jobs / area_wip_cap))
        obs[2] = float(min(1.0, feedback.completed_jobs / max(1, self.template_job_count)))
        obs[3] = float(min(1.0, busy_machines / area_machine_count))
        obs[4] = float(min(1.0, down_machines / area_machine_count))
        obs[5] = float(min(1.0, feedback.energy_kwh / area_energy_cap))
        obs[6] = float(min(1.0, feedback.carbon_kg / area_carbon_cap))
        obs[7] = float(min(1.0, self.simulator.time / max(1.0, float(self.config.horizon_hours))))
        return obs

    def _typed_events(self, transition: Dict[str, Any]) -> List[HolonicEvent]:
        events: List[HolonicEvent] = []
        for item in transition.get("event_records", []):
            kind = str(item.get("kind", ""))
            payload = {
                str(key): value
                for key, value in dict(item.get("payload", {})).items()
                if isinstance(value, (int, float, str))
            }
            if kind == "transport_start":
                event_kind = "transport_start"
                scope = "plant"
            elif kind == "transport_complete":
                event_kind = "transport_complete"
                scope = "plant"
            elif kind == "stage_release_ready":
                event_kind = "stage_release_ready"
                scope = "plant"
            elif kind == "buffer_wait_start":
                event_kind = "buffer_wait_start"
                scope = "plant"
            elif kind == "buffer_admit":
                event_kind = "buffer_admit"
                scope = "plant"
            elif kind == "blocking_start":
                event_kind = "blocking_start"
                scope = "plant"
            elif kind == "blocking_end":
                event_kind = "blocking_end"
                scope = "plant"
            elif kind == "arrival":
                event_kind = "job_arrival"
                scope = "plant"
            elif kind == "finish":
                event_kind = "operation_finish"
                scope = "cell"
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
                event_kind = kind
                scope = "des"
            events.append(
                HolonicEvent(
                    time=float(item.get("time", transition.get("time_end", 0.0))),
                    kind=event_kind,
                    scope=scope,
                    payload=payload,
                )
            )
            if kind == "blocking_start":
                events.append(
                    HolonicEvent(
                        time=float(item.get("time", transition.get("time_end", 0.0))),
                        kind="buffer_blocked",
                        scope="plant",
                        payload={**payload, "alias_of": "blocking_start"},
                    )
                )
            elif kind == "buffer_admit":
                events.append(
                    HolonicEvent(
                        time=float(item.get("time", transition.get("time_end", 0.0))),
                        kind="buffer_released",
                        scope="plant",
                        payload={**payload, "alias_of": "buffer_admit"},
                    )
                )
        return events

    def _typed_messages(self, contract: Optional[AssignmentContract]) -> List[TreeMessage]:
        if contract is None:
            return []
        area_id = self.cell_to_area[contract.cell_id]
        payload = {
            "contract_id": contract.contract_id,
            "shared_backlog_id": contract.shared_backlog_id,
            "template_job_index": contract.template_job_index,
            "job_id": contract.job_id,
            "cell_id": contract.cell_id,
            "area_id": area_id,
            "released_at": contract.released_at,
            "due_time": contract.due_time,
            "horizon_remaining": contract.horizon_remaining,
            "stage_id": self.route_stage_index.get(contract.job_id, 0),
            "from_cell": self.route_previous_cell.get(contract.job_id),
        }
        return [
            TreeMessage(
                sender="plant",
                recipient=f"area_{area_id}",
                kind="assignment_request",
                payload=payload,
            ),
            TreeMessage(
                sender=f"area_{area_id}",
                recipient=f"cell_{contract.cell_id}",
                kind="assignment_contract",
                payload=payload,
            ),
        ]

    def build_assignment_request_payload(self, shared_backlog_id: int, selected_area_id: int) -> Dict[str, Any]:
        job_id = int(shared_backlog_id)
        return {
            "stage_id": int(self.route_stage_index.get(job_id, self.simulator.jobs[job_id].current_op_idx)),
            "from_cell": self.route_previous_cell.get(job_id),
            "candidate_cell_ids": [
                int(cell_id)
                for cell_id in self._stage_candidates(job_id)
                if self.cell_to_area[int(cell_id)] == int(selected_area_id)
            ],
        }

    def _dispatch_option_to_public_dict(self, option: DispatchOption) -> Dict[str, int | float | str]:
        job = self.simulator.jobs.get(option.job_id)
        op_idx = int(job.current_op_idx) if job is not None else 0
        cell_id = self.assembly.machine_to_cell.get(option.machine_id)
        return {
            "operation_id": f"job_{option.job_id}_stage_{op_idx}_machine_{option.machine_id}",
            "job_id": int(option.job_id),
            "machine_id": int(option.machine_id),
            "cell_id": None if cell_id is None else int(cell_id),
            "op_idx": int(op_idx),
            "stage_id": int(op_idx),
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

    def _build_info(
        self,
        transition: Dict[str, Any],
        invalid_action: bool,
        contract: Optional[AssignmentContract],
    ) -> Dict[str, Any]:
        return {
            "time": self.simulator.time,
            "events": list(transition["events"]),
            "deadlocked": bool(transition["deadlocked"]),
            "invalid_action": invalid_action,
            "invalid_actions_total": self._invalid_actions,
            "valid_cells": self._valid_release_cells(),
            "valid_areas": self._valid_release_areas(),
            "valid_backlog_ids": self.valid_backlog_ids(),
            "visible_backlog_candidates": self.visible_backlog_candidates(),
            "action_mask": self.action_mask("plant").tolist(),
            "agents_to_act": self.agents_to_act(),
            "last_contract": None if contract is None else contract.__dict__.copy(),
            "typed_events": [event.to_dict() for event in self._typed_events(transition)],
            "messages": [message.to_dict() for message in self._typed_messages(contract)],
            "plant": {
                "released_jobs_total": len(self.assigned_clone_jobs),
                "completed_jobs_total": len(self.simulator.completed_jobs),
                "backlog_jobs_total": len(self.shared_backlog),
                "future_backlog_jobs_total": len(self._future_shared_backlog_ids),
                "blocked_jobs_total": len(self.blocked_backlog),
                "buffer_waiting_jobs_total": len(self.blocked_backlog),
                "inbound_jobs_total": sum(self._reserved_inbound_jobs(cell_id) for cell_id in self.assembly.cells),
                "busy_machines_total": sum(
                    1 for machine in self.simulator.machines.values() if machine.status in {"busy", "setup"}
                ),
                "setup_machines_total": sum(
                    1 for machine in self.simulator.machines.values() if machine.status == "setup"
                ),
                "down_machines_total": sum(
                    1 for machine in self.simulator.machines.values() if machine.status == "down"
                ),
                "dormant_jobs_total": len(self.simulator.jobs) - len(self.assigned_clone_jobs),
            },
            "cell_summaries": {
                cell_id: self._cell_feedback(cell_id).to_dict()
                for cell_id in sorted(self.assembly.cells)
            },
            "area_summaries": {
                area_id: self._area_feedback(area_id).to_dict()
                for area_id in self.area_ids
            },
            "machine_summaries": self._machine_summaries(),
            "setup_time_step": float(transition.get("setup_time_step", 0.0)),
            "setup_time_total": float(sum(self._cell_setup_time.values())),
            "setup_energy_kwh_total": float(sum(self._cell_setup_energy_kwh.values())),
            "setup_profile_id": self.config.setup_profile_id,
            "setup_profile_provenance_class": self.config.setup_profile_provenance_class,
            "setup_profile_scope": self.config.setup_profile_scope,
            "setup_profile_source_ref": self.config.setup_profile_source_ref,
            "arrival_profile_id": self.config.arrival_profile_id,
            "arrival_profile_provenance_class": self.config.arrival_profile_provenance_class,
            "arrival_profile_scope": self.config.arrival_profile_scope,
            "arrival_profile_source_ref": self.config.arrival_profile_source_ref,
            "arrival_events_total": int(len(self._realized_arrival_trace)),
            "arrived_jobs_total": int(
                sum(int(entry.get('batch_size', 0)) for entry in self._realized_arrival_trace)
            ),
            "arrival_batch_size_mean": float(
                (
                    sum(float(entry.get('batch_size', 0)) for entry in self._realized_arrival_trace)
                    / len(self._realized_arrival_trace)
                )
                if self._realized_arrival_trace
                else 0.0
            ),
            "realized_arrival_trace": [
                {
                    'time': float(entry['time']),
                    'arrival_event_id': int(entry['arrival_event_id']),
                    'process': str(entry['process']),
                    'batch_size': int(entry['batch_size']),
                    'shared_backlog_ids': list(entry['shared_backlog_ids']),
                }
                for entry in self._realized_arrival_trace
            ],
            "energy_kwh_step": float(transition.get("energy_kwh_step", 0.0)),
            "energy_cost_step": float(transition.get("energy_cost_step", 0.0)),
            "carbon_kg_step": float(transition.get("carbon_kg_step", 0.0)),
            "blocking_time_total": float(self._blocking_time_total),
            "buffer_wait_time_total": float(self._buffer_wait_time_total),
            "transport_time_total": float(sum(self._cell_transport_time_hours.values())),
            "transport_energy_kwh_total": float(sum(self._cell_transport_energy_kwh.values())),
            "transport_moves_total": int(self._transport_moves_total),
            "inter_cell_moves_total": int(self._transport_moves_total),
            "route_stage_pending": {
                str(job_id): int(self.route_stage_index.get(job_id, self.simulator.jobs[job_id].current_op_idx))
                for job_id in self.shared_backlog
            },
            "route_stage_blocked": {
                str(job_id): int(self.route_stage_index.get(job_id, self.simulator.jobs[job_id].current_op_idx))
                for job_id in self.blocked_backlog
            },
            "blocked_job_ids": [int(job_id) for job_id in self.blocked_backlog],
            "blocked_job_details": {
                str(job_id): {
                    "stage_id": int(self.route_stage_index.get(job_id, self.simulator.jobs[job_id].current_op_idx)),
                    "from_cell": self.route_previous_cell.get(job_id),
                    "candidate_cells": [int(cell_id) for cell_id in self._stage_candidates(job_id)],
                    "blocked_since": float(self._job_blocking_started_at.get(job_id, self.simulator.time)),
                    "blocked_duration_hours": max(
                        0.0,
                        float(self.simulator.time) - float(self._job_blocking_started_at.get(job_id, self.simulator.time)),
                    ),
                }
                for job_id in self.blocked_backlog
            },
        }

    def _is_done(self) -> bool:
        return (
            len(self.simulator.completed_jobs) >= len(self.logical_jobs)
            and not self.shared_backlog
            and not self._future_shared_backlog_ids
            and not self.blocked_backlog
            and all(not jobs for jobs in self._cell_inbound_jobs.values())
            and all(not jobs for jobs in self._cell_in_transit_jobs.values())
        )

    def render(self) -> Dict[str, Any]:
        return self._build_info(self._last_transition, invalid_action=False, contract=self._last_contract)

    def close(self) -> None:
        return None

    def _validate_area_to_cells(self, area_to_cells: Mapping[int, Sequence[int]]) -> Dict[int, List[int]]:
        if not area_to_cells:
            raise ValueError("TreeIntercellEnv requires at least one area")
        normalized = {int(area_id): [int(cell_id) for cell_id in cell_ids] for area_id, cell_ids in area_to_cells.items()}
        expected_cells = set(self.assembly.cells)
        seen_cells = set()
        for area_id, cell_ids in normalized.items():
            if area_id < 0:
                raise ValueError("Area ids must be non-negative")
            if not cell_ids:
                raise ValueError(f"Area {area_id} must include at least one cell")
            for cell_id in cell_ids:
                if cell_id not in expected_cells:
                    raise ValueError(f"Unknown cell id in area mapping: {cell_id}")
                if cell_id in seen_cells:
                    raise ValueError(f"Cell {cell_id} assigned to multiple areas")
                seen_cells.add(cell_id)
        if seen_cells != expected_cells:
            missing = sorted(expected_cells - seen_cells)
            raise ValueError(f"Cells missing from area mapping: {missing}")
        return {area_id: list(cell_ids) for area_id, cell_ids in sorted(normalized.items())}

    def _validate_route_grammar(
        self,
        route_grammar: Optional[Sequence[Sequence[int]]],
    ) -> tuple[tuple[int, ...], ...]:
        raw = route_grammar or self.ROUTE_GRAMMAR
        normalized: list[tuple[int, ...]] = []
        expected_cells = set(self.assembly.cells)
        for stage_idx, candidate_cells in enumerate(raw):
            stage_cells = tuple(int(cell_id) for cell_id in candidate_cells)
            if not stage_cells:
                raise ValueError(f"route grammar stage {stage_idx} must not be empty")
            for cell_id in stage_cells:
                if cell_id not in expected_cells:
                    raise ValueError(f"route grammar references unknown cell id: {cell_id}")
            normalized.append(stage_cells)
        if self.route_grammar_contract == "a3c9_canonical_non_degenerate_v1":
            if len(self.area_ids) != 3 or len(self.assembly.cells) != 9:
                raise ValueError("A3C9 route grammar contract requires exactly three areas and nine cells")
            for stage_idx, stage_cells in enumerate(normalized):
                if len(stage_cells) < 6:
                    raise ValueError(
                        f"A3C9 route grammar stage {stage_idx} must expose at least six candidate cells"
                    )
                covered_areas = {self.cell_to_area[int(cell_id)] for cell_id in stage_cells}
                if len(covered_areas) < 3:
                    raise ValueError(
                        f"A3C9 route grammar stage {stage_idx} must span all three areas"
                    )
        return tuple(normalized)
