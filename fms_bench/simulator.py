"""Event-driven discrete-event simulation core for FMS-Bench."""

from __future__ import annotations

import copy
import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import BenchmarkConfig
from .data_loader import InstanceData
from .distributions import sample_exponential
from .models import Job, Machine

EPS = 1e-9


@dataclass
class Event:
    time: float
    kind: str
    payload: Dict[str, int | float | str] = field(default_factory=dict)
    priority: int = 0


class EventQueue:
    """Priority queue ordered by event time, then priority, then insertion order."""

    def __init__(self) -> None:
        self._heap: List[tuple[float, int, int, Event]] = []
        self._seq = 0

    def clear(self) -> None:
        self._heap.clear()
        self._seq = 0

    def push(self, event: Event) -> None:
        heapq.heappush(self._heap, (event.time, event.priority, self._seq, event))
        self._seq += 1

    def pop(self) -> Optional[Event]:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[-1]

    def peek_time(self) -> Optional[float]:
        if not self._heap:
            return None
        return self._heap[0][0]

    def __len__(self) -> int:
        return len(self._heap)


@dataclass(frozen=True)
class DispatchOption:
    job_id: int
    machine_id: int
    proc_time: float


AdvanceHook = Callable[[float, float, Dict[int, Machine]], None]
StopHook = Callable[[], bool]
ExternalEventHandler = Callable[[Dict[str, int | float | str]], bool]


class EventDrivenSimulator:
    """DES core that jumps directly to next event times."""

    EVENT_PRIORITY = {
        "setup_start": 0,
        "finish": 1,
        "setup_complete": 2,
        "arrival": 3,
        "breakdown": 4,
        "repair": 5,
    }

    def __init__(
        self,
        instance: InstanceData,
        config: BenchmarkConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.instance = instance
        self.config = config
        self.rng = rng or np.random.default_rng()

        self.time: float = 0.0
        self.jobs: Dict[int, Job] = {}
        self.machines: Dict[int, Machine] = {}
        self.ready_jobs: set[int] = set()
        self.completed_jobs: set[int] = set()
        self.job_completion_times: Dict[int, float] = {}
        self.machine_tokens: Dict[int, int] = {}
        self.machine_failure_tokens: Dict[int, int] = {}
        self.external_event_handlers: Dict[str, ExternalEventHandler] = {}
        self.events = EventQueue()
        self.machine_ids = self._infer_machine_ids()

    def _infer_machine_ids(self) -> List[int]:
        observed = sorted(
            {
                option.machine_id
                for job in self.instance.jobs
                for op in job.operations
                for option in op.options
            }
        )
        if not observed:
            return list(range(self.instance.n_machines))
        if len(observed) >= self.instance.n_machines:
            return observed
        if min(observed) == 0:
            fallback = set(range(self.instance.n_machines))
        else:
            fallback = set(range(1, self.instance.n_machines + 1))
        return sorted(set(observed).union(fallback))

    def reset(
        self,
        seed: Optional[int] = None,
        schedule_job_arrivals: bool = True,
    ) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.time = 0.0
        self.events.clear()
        self.ready_jobs.clear()
        self.completed_jobs.clear()
        self.job_completion_times.clear()
        self.machine_tokens = {machine_id: 0 for machine_id in self.machine_ids}
        self.machine_failure_tokens = {machine_id: 0 for machine_id in self.machine_ids}
        self.jobs = {job.job_id: copy.deepcopy(job) for job in self.instance.jobs}

        active_power_kw = float(self.config.energy.active_power_kw)
        idle_power_kw = active_power_kw * self.config.energy.idle_ratio
        setup_power_kw = active_power_kw * float(self.config.setup.setup_power_ratio)
        self.machines = {
            machine_id: Machine(
                machine_id=machine_id,
                idle_power_kw=idle_power_kw,
                active_power_kw=active_power_kw,
                setup_power_kw=setup_power_kw,
            )
            for machine_id in self.machine_ids
        }

        if schedule_job_arrivals:
            for job in self.jobs.values():
                self.schedule(
                    kind="arrival",
                    time=max(0.0, float(job.release_time)),
                    payload={"job_id": job.job_id},
                )


    def activate_job(
        self,
        job_id: int,
        release_time: Optional[float] = None,
        arrival_payload: Optional[Dict[str, int | float | str]] = None,
    ) -> bool:
        """Make a known job available to the simulator.

        If the release time is now or in the past, the job becomes ready
        immediately. Otherwise an arrival event is scheduled for the future.
        """
        job = self.jobs.get(job_id)
        if job is None:
            return False
        if release_time is not None:
            job.release_time = float(release_time)
        if job_id in self.completed_jobs:
            return False
        target_time = max(self.time, float(job.release_time))
        if target_time <= self.time + EPS:
            return self._handle_arrival(job_id)
        payload: Dict[str, int | float | str] = {"job_id": job_id}
        if arrival_payload is not None:
            payload.update(arrival_payload)
            payload["job_id"] = job_id
        self.schedule("arrival", target_time, payload)
        return True

    def _setup_family_for_job(self, job: Job) -> str:
        base_family = job.setup_family or f"job_type_{job.template_job_id or job.job_id}"
        if self.config.setup.rule == "operation_change":
            return f"{base_family}:op_{job.current_op_idx}"
        return base_family

    def _setup_duration(
        self,
        *,
        machine: Machine,
        job: Job,
        proc_time: float,
        target_setup_family: str,
    ) -> float:
        if not self.config.setup.enabled:
            return 0.0
        if machine.last_setup_family is None:
            return 0.0
        if machine.last_setup_family == target_setup_family:
            return float(self.config.setup.same_family_time_hours)
        return max(
            float(self.config.setup.minimum_time_hours),
            float(self.config.setup.time_ratio_to_proc) * float(proc_time),
        )

    def _start_processing(
        self,
        *,
        machine_id: int,
        job_id: int,
        op_idx: int,
        proc_time: float,
        target_setup_family: str,
        keep_token: bool = False,
    ) -> None:
        machine = self.machines[machine_id]
        machine.status = "busy"
        machine.current_job = job_id
        machine.remaining_time = proc_time
        machine.setup_remaining_time = 0.0
        machine.pending_proc_time = 0.0
        machine.last_setup_family = target_setup_family
        machine.target_setup_family = None

        if not keep_token:
            self.machine_tokens[machine_id] += 1
        self.schedule(
            "finish",
            self.time + proc_time,
            {
                "machine_id": machine_id,
                "job_id": job_id,
                "op_idx": op_idx,
                "token": self.machine_tokens[machine_id],
            },
        )
        if self.config.failure.process != "none" and self.config.failure.mtbf_hours > 0.0:
            self._schedule_breakdown(machine_id, proc_time=proc_time)

    def is_done(self) -> bool:
        return len(self.completed_jobs) >= len(self.jobs)

    def is_deadlocked(self) -> bool:
        return (not self.is_done()) and len(self.events) == 0 and len(self.get_dispatch_options()) == 0

    def register_external_event_handler(
        self,
        kind: str,
        handler: ExternalEventHandler,
    ) -> None:
        self.external_event_handlers[str(kind)] = handler

    def schedule(
        self,
        kind: str,
        time: float,
        payload: Optional[Dict[str, int | float | str]] = None,
    ) -> None:
        payload = payload or {}
        priority = self.EVENT_PRIORITY.get(kind, 99)
        self.events.push(Event(time=float(time), kind=kind, payload=payload, priority=priority))

    def _schedule_breakdown(self, machine_id: int, proc_time: float | None = None) -> None:
        if self.config.failure.process == "none" or self.config.failure.mtbf_hours <= 0.0:
            return
        self.machine_failure_tokens[machine_id] += 1
        payload: Dict[str, int | float | str] = {
            "machine_id": machine_id,
            "token": self.machine_failure_tokens[machine_id],
            "failure_profile_id": str(self.config.failure_profile_id),
            "failure_process": str(self.config.failure.process),
        }
        if self.config.failure.process == "exponential":
            delay = sample_exponential(self.config.failure.mtbf_hours, self.rng)
        elif self.config.failure.process == "weibull":
            shape = max(float(self.config.failure.weibull_shape), EPS)
            scale = float(self.config.failure.mtbf_hours) / math.gamma(1.0 + (1.0 / shape))
            delay = float(scale * self.rng.weibull(shape))
            payload["weibull_shape"] = float(shape)
        elif self.config.failure.process == "load_dependent":
            reference = max(float(self.config.failure.load_reference_proc_hours), EPS)
            effective_proc_time = max(float(proc_time or reference), EPS)
            alpha = max(float(self.config.failure.load_dependent_alpha), 0.0)
            load_multiplier = max(1.0, 1.0 + alpha * ((effective_proc_time / reference) - 1.0))
            effective_mtbf = max(EPS, float(self.config.failure.mtbf_hours) / load_multiplier)
            delay = sample_exponential(effective_mtbf, self.rng)
            payload["load_multiplier"] = float(load_multiplier)
            payload["effective_mtbf_hours"] = float(effective_mtbf)
            payload["proc_time_hours"] = float(effective_proc_time)
        else:
            raise ValueError(f"Unsupported failure process: {self.config.failure.process}")
        delay = max(delay, EPS)
        self.schedule(
            "breakdown",
            self.time + delay,
            payload,
        )

    def _schedule_repair(self, machine_id: int) -> None:
        if self.config.failure.mttr_hours <= 0.0:
            self.schedule(
                "repair",
                self.time,
                {
                    "machine_id": machine_id,
                    "repair_duration": 0.0,
                    "downtime_duration": 0.0,
                    "failure_profile_id": str(self.config.failure_profile_id),
                },
            )
            return
        delay = sample_exponential(self.config.failure.mttr_hours, self.rng)
        delay = max(delay, EPS)
        self.schedule(
            "repair",
            self.time + delay,
            {
                "machine_id": machine_id,
                "repair_duration": float(delay),
                "downtime_duration": float(delay),
                "failure_profile_id": str(self.config.failure_profile_id),
            },
        )

    def get_dispatch_options(self) -> List[DispatchOption]:
        options: List[DispatchOption] = []
        for job_id in sorted(self.ready_jobs):
            job = self.jobs[job_id]
            if job.current_op_idx >= len(job.operations):
                continue
            op = job.operations[job.current_op_idx]
            for option in op.options:
                machine = self.machines.get(option.machine_id)
                if machine is not None and machine.status == "idle":
                    options.append(
                        DispatchOption(
                            job_id=job_id,
                            machine_id=option.machine_id,
                            proc_time=float(option.proc_time),
                        )
                    )
        options.sort(key=lambda x: (x.job_id, x.machine_id, x.proc_time))
        return options

    def dispatch(self, job_id: int, machine_id: int) -> bool:
        if job_id not in self.ready_jobs:
            return False
        machine = self.machines.get(machine_id)
        if machine is None or machine.status != "idle":
            return False

        job = self.jobs[job_id]
        if job.current_op_idx >= len(job.operations):
            return False

        proc_time: Optional[float] = None
        for option in job.operations[job.current_op_idx].options:
            if option.machine_id == machine_id:
                proc_time = float(option.proc_time)
                break
        if proc_time is None:
            return False

        self.ready_jobs.remove(job_id)
        target_setup_family = self._setup_family_for_job(job)
        setup_duration = self._setup_duration(
            machine=machine,
            job=job,
            proc_time=proc_time,
            target_setup_family=target_setup_family,
        )

        if setup_duration > EPS:
            self.machine_tokens[machine_id] += 1
            machine.status = "setup"
            machine.current_job = job_id
            machine.remaining_time = setup_duration
            machine.setup_remaining_time = setup_duration
            machine.pending_proc_time = proc_time
            machine.target_setup_family = target_setup_family
            self.schedule(
                "setup_start",
                self.time,
                {
                    "machine_id": machine_id,
                    "job_id": job_id,
                    "op_idx": job.current_op_idx,
                    "token": self.machine_tokens[machine_id],
                    "from_setup_family": "" if machine.last_setup_family is None else machine.last_setup_family,
                    "to_setup_family": target_setup_family,
                    "setup_duration": setup_duration,
                },
            )
            self.schedule(
                "setup_complete",
                self.time + setup_duration,
                {
                    "machine_id": machine_id,
                    "job_id": job_id,
                    "op_idx": job.current_op_idx,
                    "token": self.machine_tokens[machine_id],
                    "proc_time": proc_time,
                    "from_setup_family": "" if machine.last_setup_family is None else machine.last_setup_family,
                    "to_setup_family": target_setup_family,
                    "setup_duration": setup_duration,
                },
            )
            return True

        self._start_processing(
            machine_id=machine_id,
            job_id=job_id,
            op_idx=job.current_op_idx,
            proc_time=proc_time,
            target_setup_family=target_setup_family,
        )
        return True

    def run_until_decision(
        self,
        advance_hook: Optional[AdvanceHook] = None,
        stop_condition: Optional[StopHook] = None,
    ) -> Dict[str, object]:
        start_time = self.time
        event_kinds: List[str] = []
        event_records: List[Dict[str, object]] = []

        while True:
            if self.is_done():
                break
            if stop_condition is not None and stop_condition():
                break
            if self.get_dispatch_options():
                break

            event = self.events.pop()
            if event is None:
                break

            self._process_event(event, event_kinds, event_records, advance_hook)
            # Drain all same-time events so the state at self.time is consistent.
            while True:
                next_time = self.events.peek_time()
                if next_time is None or abs(next_time - self.time) > EPS:
                    break
                same_time_event = self.events.pop()
                if same_time_event is None:
                    break
                self._process_event(same_time_event, event_kinds, event_records, None)

        return {
            "time_start": start_time,
            "time_end": self.time,
            "events": event_kinds,
            "event_records": event_records,
            "deadlocked": self.is_deadlocked(),
        }

    def _process_event(
        self,
        event: Event,
        event_kinds: List[str],
        event_records: List[Dict[str, object]],
        advance_hook: Optional[AdvanceHook],
    ) -> None:
        if event.time < self.time - EPS:
            return
        if event.time > self.time:
            if advance_hook is not None:
                advance_hook(self.time, event.time, self.machines)
            self._advance_time(event.time)
        if self._handle_event(event):
            event_kinds.append(event.kind)
            event_records.append(
                {
                    "time": float(self.time),
                    "kind": str(event.kind),
                    "payload": dict(event.payload),
                }
            )

    def _advance_time(self, new_time: float) -> None:
        dt = new_time - self.time
        if dt <= 0.0:
            self.time = max(self.time, new_time)
            return

        for machine in self.machines.values():
            if machine.status == "busy":
                machine.remaining_time = max(0.0, machine.remaining_time - dt)
            elif machine.status == "setup":
                machine.remaining_time = max(0.0, machine.remaining_time - dt)
                machine.setup_remaining_time = max(0.0, machine.setup_remaining_time - dt)
            elif machine.status == "down":
                machine.down_time += dt
        self.time = new_time

    def _handle_event(self, event: Event) -> bool:
        kind = event.kind
        payload = event.payload
        if kind == "arrival":
            return self._handle_arrival(int(payload["job_id"]))
        if kind == "setup_start":
            return self._handle_setup_start(
                machine_id=int(payload["machine_id"]),
                job_id=int(payload["job_id"]),
                token=int(payload["token"]),
            )
        if kind == "finish":
            return self._handle_finish(
                machine_id=int(payload["machine_id"]),
                job_id=int(payload["job_id"]),
                op_idx=int(payload["op_idx"]),
                token=int(payload["token"]),
            )
        if kind == "setup_complete":
            return self._handle_setup_complete(
                machine_id=int(payload["machine_id"]),
                job_id=int(payload["job_id"]),
                op_idx=int(payload["op_idx"]),
                token=int(payload["token"]),
                proc_time=float(payload["proc_time"]),
                target_setup_family=str(payload.get("to_setup_family", "")),
            )
        if kind == "breakdown":
            return self._handle_breakdown(int(payload["machine_id"]), int(payload.get("token", -1)))
        if kind == "repair":
            return self._handle_repair(int(payload["machine_id"]))
        handler = self.external_event_handlers.get(kind)
        if handler is not None:
            return bool(handler(dict(payload)))
        return False

    def _processing_jobs(self) -> set[int]:
        return {m.current_job for m in self.machines.values() if m.current_job is not None}

    def _handle_arrival(self, job_id: int) -> bool:
        if job_id in self.completed_jobs:
            return False
        job = self.jobs.get(job_id)
        if job is None or job.current_op_idx >= len(job.operations):
            return False
        if job_id in self.ready_jobs:
            return False
        if job_id in self._processing_jobs():
            return False
        self.ready_jobs.add(job_id)
        return True

    def _handle_setup_start(self, machine_id: int, job_id: int, token: int) -> bool:
        machine = self.machines.get(machine_id)
        if machine is None:
            return False
        if self.machine_tokens.get(machine_id, -1) != token:
            return False
        return machine.status == "setup" and machine.current_job == job_id

    def _handle_setup_complete(
        self,
        machine_id: int,
        job_id: int,
        op_idx: int,
        token: int,
        proc_time: float,
        target_setup_family: str,
    ) -> bool:
        machine = self.machines.get(machine_id)
        if machine is None:
            return False
        if self.machine_tokens.get(machine_id, -1) != token:
            return False
        if machine.status != "setup" or machine.current_job != job_id:
            return False

        machine.setup_remaining_time = 0.0
        machine.pending_proc_time = 0.0
        machine.target_setup_family = None
        self._start_processing(
            machine_id=machine_id,
            job_id=job_id,
            op_idx=op_idx,
            proc_time=proc_time,
            target_setup_family=target_setup_family,
            keep_token=True,
        )
        return True

    def _handle_finish(self, machine_id: int, job_id: int, op_idx: int, token: int) -> bool:
        machine = self.machines.get(machine_id)
        if machine is None:
            return False
        if self.machine_tokens.get(machine_id, -1) != token:
            return False
        if machine.status != "busy" or machine.current_job != job_id:
            return False

        job = self.jobs[job_id]
        if job.current_op_idx != op_idx:
            return False

        machine.status = "idle"
        machine.current_job = None
        machine.remaining_time = 0.0
        self.machine_failure_tokens[machine_id] += 1

        job.current_op_idx += 1
        if job.current_op_idx >= len(job.operations):
            self.completed_jobs.add(job_id)
            self.job_completion_times[job_id] = self.time
            self.ready_jobs.discard(job_id)
        else:
            self.ready_jobs.add(job_id)
        return True

    def _handle_breakdown(self, machine_id: int, token: int = -1) -> bool:
        machine = self.machines.get(machine_id)
        if machine is None:
            return False
        if token >= 0 and self.machine_failure_tokens.get(machine_id, -1) != token:
            return False
        if machine.status != "busy":
            return False

        if machine.current_job is not None:
            # Invalidate outstanding finish event for this processing slice.
            self.machine_tokens[machine_id] += 1
        self.machine_failure_tokens[machine_id] += 1
        machine.status = "down"
        self._schedule_repair(machine_id)
        return True

    def _handle_repair(self, machine_id: int) -> bool:
        machine = self.machines.get(machine_id)
        if machine is None:
            return False
        if machine.status != "down":
            return False

        if machine.current_job is not None and machine.remaining_time > EPS:
            machine.status = "busy"
            current_job = machine.current_job
            job = self.jobs[current_job]
            self.schedule(
                "finish",
                self.time + machine.remaining_time,
                {
                    "machine_id": machine_id,
                    "job_id": current_job,
                    "op_idx": job.current_op_idx,
                    "token": self.machine_tokens[machine_id],
                },
            )
            self._schedule_breakdown(machine_id, proc_time=machine.remaining_time)
        else:
            machine.status = "idle"
            machine.current_job = None
            machine.remaining_time = 0.0
            self.machine_failure_tokens[machine_id] += 1

        return True
