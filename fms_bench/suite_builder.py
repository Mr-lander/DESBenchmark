"""Build multi-cell plant assemblies from public benchmark instances."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Iterable, List

from .config import BenchmarkConfig
from .data_loader import InstanceData, load_instance
from .due_date_profiles import apply_due_date_profile, synthesize_due_time
from .holonic_types import CellBudget, CellSpec, PlantAssembly, PlantCellConfig
from .models import Job, Operation, OperationOption
from .tree_builder import build_tree_topology


def _infer_local_machine_ids(instance: InstanceData) -> List[int]:
    observed = sorted(
        {
            option.machine_id
            for job in instance.jobs
            for op in job.operations
            for option in op.options
        }
    )
    if not observed:
        return list(range(instance.n_machines))
    if len(observed) >= instance.n_machines:
        return observed
    if min(observed) == 0:
        fallback = set(range(instance.n_machines))
    else:
        fallback = set(range(1, instance.n_machines + 1))
    return sorted(set(observed).union(fallback))


def _remap_job(
    job: Job,
    job_id: int,
    machine_map: Dict[int, int],
    proc_time_scale: float = 1.0,
    due_date_profile_id: str | None = None,
) -> Job:
    operations: List[Operation] = []
    for op in job.operations:
        options = [
            OperationOption(machine_id=machine_map[option.machine_id], proc_time=float(option.proc_time) * float(proc_time_scale))
            for option in op.options
        ]
        operations.append(Operation(op_id=op.op_id, options=options))

    remapped = Job(
        job_id=job_id,
        operations=operations,
        due_time=job.due_time,
        release_time=job.release_time,
        current_op_idx=0,
        template_job_id=job.template_job_id if job.template_job_id is not None else job.job_id,
        setup_family=job.setup_family or f"job_type_{job.job_id}",
    )
    if remapped.due_time is None:
        remapped.due_time = synthesize_due_time(
            operations=operations,
            release_time=float(remapped.release_time),
            config=apply_due_date_profile(BenchmarkConfig(), due_date_profile_id),
        )
    return remapped


def build_plant_assembly(
    cell_configs: Iterable[PlantCellConfig],
    config: BenchmarkConfig | None = None,
) -> PlantAssembly:
    """Create a multi-cell plant assembly from benchmark instance configs."""
    benchmark_config = config or BenchmarkConfig()
    benchmark_config = apply_due_date_profile(
        benchmark_config,
        benchmark_config.due_date_profile_id,
    )
    cells: Dict[int, CellSpec] = {}
    combined_jobs: List[Job] = []
    job_to_cell: Dict[int, int] = {}
    machine_to_cell: Dict[int, int] = {}

    next_job_id = 0
    next_machine_id = 0

    for cell_config in cell_configs:
        instance = load_instance(cell_config.instance_path, cell_config.instance_kind)
        local_machine_ids = _infer_local_machine_ids(instance)
        machine_map = {local_id: next_machine_id + idx for idx, local_id in enumerate(local_machine_ids)}
        global_machine_ids = list(machine_map.values())
        for machine_id in global_machine_ids:
            machine_to_cell[machine_id] = cell_config.cell_id

        template_job_ids: List[int] = []
        template_jobs: List[Job] = []
        for job in instance.jobs:
            remapped = _remap_job(
                job,
                job_id=next_job_id,
                machine_map=machine_map,
                proc_time_scale=cell_config.processing_time_scale,
                due_date_profile_id=benchmark_config.due_date_profile_id,
            )
            combined_jobs.append(copy.deepcopy(remapped))
            template_jobs.append(copy.deepcopy(remapped))
            template_job_ids.append(next_job_id)
            job_to_cell[next_job_id] = cell_config.cell_id
            next_job_id += 1

        label = cell_config.label or f"cell-{cell_config.cell_id}"
        cells[cell_config.cell_id] = CellSpec(
            cell_id=cell_config.cell_id,
            label=label,
            family_key=cell_config.family_key,
            instance_path=Path(cell_config.instance_path),
            dispatch_policy=cell_config.dispatch_policy,
            machine_ids=global_machine_ids,
            template_job_ids=template_job_ids,
            template_jobs=template_jobs,
            budget=CellBudget(
                cell_id=cell_config.cell_id,
                wip_cap=cell_config.wip_cap,
                energy_cap_kwh=cell_config.energy_cap_kwh,
                carbon_cap_kg=cell_config.carbon_cap_kg,
            ),
            processing_time_scale=cell_config.processing_time_scale,
            inbound_buffer_cap=cell_config.inbound_buffer_cap,
            ingress_transport_time_hours=float(cell_config.ingress_transport_time_hours),
            ingress_transport_energy_kwh=float(cell_config.ingress_transport_energy_kwh),
        )
        next_machine_id += len(global_machine_ids)

    return PlantAssembly(
        combined_jobs=combined_jobs,
        total_machine_count=next_machine_id,
        cells=cells,
        job_to_cell=job_to_cell,
        machine_to_cell=machine_to_cell,
    )



def build_tree_area_mapping(cell_configs: Iterable[PlantCellConfig], area_count: int) -> Dict[int, List[int]]:
    """Return a stable area-to-cells map for tree-holonic benchmark items."""
    topology = build_tree_topology(list(cell_configs), area_count=area_count)
    return topology.area_to_cells
