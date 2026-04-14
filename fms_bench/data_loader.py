"""Parsers for standard scheduling instances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .models import Job, Operation, OperationOption


@dataclass
class InstanceData:
    jobs: List[Job]
    n_jobs: int
    n_machines: int


def _parse_line_numbers(line: str) -> List[int]:
    return [int(x) for x in line.strip().split() if x.strip()]


FJSP_KIND_ALIASES = {
    "brandimarte",
    "fjsp",
    "hurink",
    "hurink_edata",
    "hurink_rdata",
    "hurink_sdata",
    "hurink_vdata",
    "dauzere",
    "dauzere_peres_paulli",
    "chambers",
    "barnes",
    "chambers_barnes",
    "kacem",
    "fattahi",
    "behnke",
}


def load_taillard(path: str | Path) -> InstanceData:
    """Parse Taillard JSSP format."""
    path = Path(path)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    n_jobs, n_machines = _parse_line_numbers(lines[0])[:2]
    jobs: List[Job] = []
    for job_id in range(n_jobs):
        nums = _parse_line_numbers(lines[1 + job_id])
        ops: List[Operation] = []
        for op_id in range(n_machines):
            machine = nums[2 * op_id]
            proc = nums[2 * op_id + 1]
            ops.append(Operation(op_id=op_id, options=[OperationOption(machine_id=machine, proc_time=proc)]))
        jobs.append(Job(job_id=job_id, operations=ops))
    return InstanceData(jobs=jobs, n_jobs=n_jobs, n_machines=n_machines)


def load_fjsp(path: str | Path) -> InstanceData:
    """Parse FJSPLib-compatible FJSP format.

    This loader works for Brandimarte and the other common FJSPLib families
    because they share the same per-job line structure. The header may contain
    an extra average-flexibility field, which can be an integer or a float.
    """
    path = Path(path)
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty FJSP instance: {path}")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid FJSP header in {path}: {lines[0]!r}")

    n_jobs = int(float(header[0]))
    n_machines = int(float(header[1]))
    if len(lines) < 1 + n_jobs:
        raise ValueError(f"Expected {n_jobs} job rows in {path}, found {len(lines) - 1}")

    jobs: List[Job] = []

    for job_id in range(n_jobs):
        nums = _parse_line_numbers(lines[1 + job_id])
        if not nums:
            raise ValueError(f"Empty job row {job_id} in {path}")

        idx = 0
        op_count = nums[idx]
        idx += 1
        ops: List[Operation] = []
        for op_id in range(op_count):
            if idx >= len(nums):
                raise ValueError(f"Unexpected end of job row {job_id} in {path}")
            k = nums[idx]
            idx += 1
            options: List[OperationOption] = []
            for _ in range(k):
                if idx + 1 >= len(nums):
                    raise ValueError(f"Incomplete option list in job row {job_id} of {path}")
                machine = nums[idx]
                proc = nums[idx + 1]
                idx += 2
                options.append(OperationOption(machine_id=machine, proc_time=proc))
            ops.append(Operation(op_id=op_id, options=options))
        jobs.append(Job(job_id=job_id, operations=ops))
    return InstanceData(jobs=jobs, n_jobs=n_jobs, n_machines=n_machines)


def load_brandimarte(path: str | Path) -> InstanceData:
    """Backward-compatible alias for the common FJSPLib parser."""
    return load_fjsp(path)


def load_instance(path: str | Path, kind: str) -> InstanceData:
    kind = kind.lower()
    if kind in ("taillard", "jssp"):
        return load_taillard(path)
    if kind in FJSP_KIND_ALIASES:
        return load_fjsp(path)
    raise ValueError(f"Unknown instance kind: {kind}")
