"""Shared-backlog plant environment with assignment-sensitive plant actions."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .arrival_profiles import schedule_online_arrival_batches
from .config import BenchmarkConfig
from .holonic_types import AssignmentContract, CellFeedback, HolonicMessage, PlantCellConfig, SharedBacklogItem
from .plant_env import HolonicPlantEnv


class SharedBacklogPlantEnv(HolonicPlantEnv):
    """Plant env variant where each plant action assigns the next global backlog item to a cell."""

    def __init__(
        self,
        cell_configs: Iterable[PlantCellConfig],
        config: Optional[BenchmarkConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(cell_configs=cell_configs, config=config, seed=seed)
        self.shared_backlog_items: List[SharedBacklogItem] = self._build_shared_backlog_items()
        self.shared_backlog_item_lookup: Dict[int, SharedBacklogItem] = {
            item.shared_backlog_id: item for item in self.shared_backlog_items
        }
        self.template_job_count = len(self.shared_backlog_items)
        self.clone_job_lookup = {
            (cell_id, template_idx): cell.template_job_ids[template_idx]
            for cell_id, cell in self.assembly.cells.items()
            for template_idx in range(self.template_job_count)
        }
        self.shared_backlog: List[int] = []
        self._future_shared_backlog_ids: List[int] = []
        self._shared_backlog_arrival_times: Dict[int, float] = {}
        self._realized_arrival_trace: List[Dict[str, Any]] = []
        self.assigned_clone_jobs: set[int] = set()
        self.assigned_templates: Dict[int, int] = {}
        self._cell_assigned_jobs: Dict[int, set[int]] = {cell_id: set() for cell_id in self.assembly.cells}
        self._cell_inbound_jobs: Dict[int, set[int]] = {cell_id: set() for cell_id in self.assembly.cells}
        self._cell_transport_time_hours: Dict[int, float] = {
            cell_id: 0.0 for cell_id in self.assembly.cells
        }
        self._cell_transport_energy_kwh: Dict[int, float] = {
            cell_id: 0.0 for cell_id in self.assembly.cells
        }
        self._transport_moves_total = 0
        self.simulator.register_external_event_handler(
            "shared_backlog_arrival",
            self._handle_shared_backlog_arrival,
        )

    def _build_shared_backlog_items(self) -> List[SharedBacklogItem]:
        cells = [self.assembly.cells[cell_id] for cell_id in sorted(self.assembly.cells)]
        if not cells:
            raise ValueError('SharedBacklogPlantEnv requires at least one cell')

        template_count = len(cells[0].template_job_ids)
        reference_signature = self._job_signature(cells[0].template_jobs)
        for cell in cells[1:]:
            if len(cell.template_job_ids) != template_count:
                raise ValueError('SharedBacklogPlantEnv requires replicated cells with equal job counts')
            if self._job_signature(cell.template_jobs) != reference_signature:
                raise ValueError('SharedBacklogPlantEnv requires structurally compatible replicated cells')

        candidate_cells = sorted(self.assembly.cells)
        return [
            SharedBacklogItem(
                shared_backlog_id=template_idx,
                template_job_index=template_idx,
                candidate_cell_ids=candidate_cells,
            )
            for template_idx in range(template_count)
        ]

    @staticmethod
    def _job_signature(jobs: List[Any]) -> tuple[tuple[int, tuple[int, ...]], ...]:
        signature = []
        for job in jobs:
            signature.append(
                (
                    len(job.operations),
                    tuple(len(op.options) for op in job.operations),
                )
            )
        return tuple(signature)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.shared_backlog = []
        self._future_shared_backlog_ids = []
        self._shared_backlog_arrival_times = {}
        self._realized_arrival_trace = []
        self.assigned_clone_jobs = set()
        self.assigned_templates = {}
        self._cell_assigned_jobs = {cell_id: set() for cell_id in self.assembly.cells}
        self._cell_inbound_jobs = {cell_id: set() for cell_id in self.assembly.cells}
        self._cell_transport_time_hours = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._cell_transport_energy_kwh = {cell_id: 0.0 for cell_id in self.assembly.cells}
        self._transport_moves_total = 0
        self._backlogs = {cell_id: [] for cell_id in self.assembly.cells}
        self._initialize_shared_backlog_visibility()
        obs = self._build_observation()
        info = self._build_info(self._last_transition, invalid_action=False, contract=None)
        return obs, info

    def _initialize_shared_backlog_visibility(self) -> None:
        all_backlog_ids = [item.shared_backlog_id for item in self.shared_backlog_items]
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

    def _handle_shared_backlog_arrival(
        self,
        payload: Dict[str, int | float | str],
    ) -> bool:
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
            if shared_backlog_id in self.assigned_templates:
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

    def _release_next_job(self, cell_id: int, job_id: Optional[int] = None) -> AssignmentContract:
        if not self.shared_backlog:
            raise RuntimeError('No remaining shared backlog items')
        shared_backlog_id = int(self.shared_backlog[0] if job_id is None else job_id)
        if shared_backlog_id not in self.shared_backlog:
            raise RuntimeError(
                f"Requested shared backlog item is not currently visible: {shared_backlog_id}"
            )
        self.shared_backlog.remove(shared_backlog_id)
        backlog_item = self.shared_backlog_item_lookup[shared_backlog_id]
        template_idx = int(backlog_item.template_job_index)
        job_id = self.clone_job_lookup[(cell_id, template_idx)]
        self._released_jobs.add(job_id)
        self.assigned_clone_jobs.add(job_id)
        self.assigned_templates[shared_backlog_id] = cell_id
        self._cell_assigned_jobs[cell_id].add(job_id)
        self._cell_inbound_jobs[cell_id].add(job_id)
        cell = self.assembly.cells[cell_id]
        transport_time_hours = float(cell.ingress_transport_time_hours)
        transport_energy_kwh = float(cell.ingress_transport_energy_kwh)
        self._cell_transport_time_hours[cell_id] += transport_time_hours
        self._cell_transport_energy_kwh[cell_id] += transport_energy_kwh
        self._transport_moves_total += 1
        self.simulator.activate_job(
            job_id,
            release_time=self.simulator.time + transport_time_hours,
            arrival_payload={
                "cell_id": cell_id,
                "arrival_source": "ingress_transport",
                "transport_time_hours": transport_time_hours,
                "transport_energy_kwh": transport_energy_kwh,
            },
        )

        self._contract_seq += 1
        job = self.simulator.jobs[job_id]
        return AssignmentContract(
            contract_id=self._contract_seq,
            shared_backlog_id=shared_backlog_id,
            template_job_index=template_idx,
            cell_id=cell_id,
            job_id=job_id,
            released_at=self.simulator.time,
            due_time=job.due_time,
            horizon_remaining=max(0.0, float(self.config.horizon_hours) - self.simulator.time),
        )

    def _valid_release_cells(self) -> List[int]:
        if not self.shared_backlog:
            return []
        valid: List[int] = []
        for cell_id, cell in self.assembly.cells.items():
            feedback = self._cell_feedback(cell_id)
            if feedback.active_jobs >= cell.budget.wip_cap:
                continue
            if feedback.inbound_jobs >= cell.inbound_buffer_cap:
                continue
            valid.append(cell_id)
        return valid

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
        completed_jobs = len([jid for jid in self.simulator.completed_jobs if jid in cell_job_ids])
        inbound_jobs = len(self._cell_inbound_jobs[cell_id])
        busy_machines = sum(
            1 for mid in cell.machine_ids if self.simulator.machines[mid].status == 'busy'
        )
        down_machines = sum(
            1 for mid in cell.machine_ids if self.simulator.machines[mid].status == 'down'
        )

        remaining_wip = max(0, cell.budget.wip_cap - active_jobs)
        remaining_inbound_buffer = max(0, cell.inbound_buffer_cap - inbound_jobs)
        remaining_energy = max(0.0, cell.budget.energy_cap_kwh - self._cell_energy_kwh[cell_id])
        remaining_carbon = max(0.0, cell.budget.carbon_cap_kg - self._cell_carbon_kg[cell_id])
        energy_overshoot = max(0.0, self._cell_energy_kwh[cell_id] - cell.budget.energy_cap_kwh)
        carbon_overshoot = max(0.0, self._cell_carbon_kg[cell_id] - cell.budget.carbon_cap_kg)

        return CellFeedback(
            cell_id=cell_id,
            label=cell.label,
            family_key=cell.family_key,
            backlog_jobs=len(self.shared_backlog),
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
            inbound_buffer_remaining=remaining_inbound_buffer,
            energy_overshoot_kwh=energy_overshoot,
            carbon_overshoot_kg=carbon_overshoot,
            processing_time_scale=cell.processing_time_scale,
        )

    def dispatch_job_to_machine(self, job_id: int, machine_id: int) -> bool:
        dispatched = self.simulator.dispatch(job_id, machine_id)
        if not dispatched:
            return False
        cell_id = self.assembly.job_to_cell.get(job_id)
        if cell_id is not None:
            self._cell_inbound_jobs[cell_id].discard(job_id)
        return True

    def _build_observation(self) -> np.ndarray:
        total_jobs = max(1, self.template_job_count)
        horizon = max(1.0, float(self.config.horizon_hours))
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        completed_assigned = len(self.simulator.completed_jobs.intersection(self.assigned_clone_jobs))
        obs[0] = float(min(1.0, self.simulator.time / horizon))
        obs[1] = float(min(1.0, len(self.assigned_clone_jobs) / total_jobs))
        obs[2] = float(min(1.0, completed_assigned / total_jobs))

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

    def _typed_messages(self, contract: Optional[AssignmentContract]) -> List[HolonicMessage]:
        if contract is None:
            return []
        return [
            HolonicMessage(
                sender='plant',
                recipient=f'cell_{contract.cell_id}',
                kind='assignment_contract',
                payload={
                    'contract_id': contract.contract_id,
                    'shared_backlog_id': contract.shared_backlog_id,
                    'template_job_index': contract.template_job_index,
                    'job_id': contract.job_id,
                    'cell_id': contract.cell_id,
                    'released_at': contract.released_at,
                    'due_time': contract.due_time,
                    'horizon_remaining': contract.horizon_remaining,
                },
            )
        ]

    def _build_info(
        self,
        transition: Dict[str, Any],
        invalid_action: bool,
        contract: Optional[AssignmentContract],
    ) -> Dict[str, Any]:
        info = super()._build_info(transition, invalid_action=invalid_action, contract=contract)
        completed_assigned = len(self.simulator.completed_jobs.intersection(self.assigned_clone_jobs))
        info['plant']['released_jobs_total'] = len(self.assigned_clone_jobs)
        info['plant']['completed_jobs_total'] = completed_assigned
        info['plant']['backlog_jobs_total'] = len(self.shared_backlog)
        info['plant']['future_backlog_jobs_total'] = len(self._future_shared_backlog_ids)
        info['plant']['inbound_jobs_total'] = sum(len(jobs) for jobs in self._cell_inbound_jobs.values())
        info['plant']['dormant_jobs_total'] = len(self.simulator.jobs) - len(self.assigned_clone_jobs)
        info['arrival_profile_id'] = self.config.arrival_profile_id
        info['arrival_profile_provenance_class'] = self.config.arrival_profile_provenance_class
        info['arrival_profile_scope'] = self.config.arrival_profile_scope
        info['arrival_profile_source_ref'] = self.config.arrival_profile_source_ref
        info['arrival_events_total'] = int(len(self._realized_arrival_trace))
        info['arrived_jobs_total'] = int(
            sum(int(entry.get('batch_size', 0)) for entry in self._realized_arrival_trace)
        )
        info['arrival_batch_size_mean'] = float(
            (
                sum(float(entry.get('batch_size', 0)) for entry in self._realized_arrival_trace)
                / len(self._realized_arrival_trace)
            )
            if self._realized_arrival_trace
            else 0.0
        )
        info['realized_arrival_trace'] = [
            {
                'time': float(entry['time']),
                'arrival_event_id': int(entry['arrival_event_id']),
                'process': str(entry['process']),
                'batch_size': int(entry['batch_size']),
                'shared_backlog_ids': list(entry['shared_backlog_ids']),
            }
            for entry in self._realized_arrival_trace
        ]
        info['transport_time_total'] = float(sum(self._cell_transport_time_hours.values()))
        info['transport_energy_kwh_total'] = float(sum(self._cell_transport_energy_kwh.values()))
        info['transport_moves_total'] = int(self._transport_moves_total)
        info['shared_backlog_remaining'] = list(self.shared_backlog)
        info['future_shared_backlog_remaining'] = list(self._future_shared_backlog_ids)
        return info

    def _is_done(self) -> bool:
        return (
            (not self.shared_backlog)
            and (not self._future_shared_backlog_ids)
            and self.assigned_clone_jobs.issubset(self.simulator.completed_jobs)
        )
