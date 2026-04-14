"""Tests for named failure benchmark scenario profiles."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import BenchmarkConfig
from fms_bench.data_loader import InstanceData
from fms_bench.failure_profiles import apply_failure_profile, list_failure_profiles
from fms_bench.models import Job, Operation, OperationOption
from fms_bench.simulator import EventDrivenSimulator


def _failure_instance() -> InstanceData:
    return InstanceData(
        jobs=[
            Job(
                job_id=0,
                operations=[
                    Operation(
                        op_id=0,
                        options=[OperationOption(machine_id=0, proc_time=10.0)],
                    )
                ],
            ),
        ],
        n_jobs=1,
        n_machines=1,
    )


def test_apply_failure_profile_sets_weibull_metadata_and_parameters() -> None:
    config = apply_failure_profile(BenchmarkConfig(), "failure_weibull_aging_nominal_v1")

    assert config.failure.process == "weibull"
    assert config.failure.weibull_shape > 1.0
    assert config.failure_profile_id == "failure_weibull_aging_nominal_v1"
    assert config.failure_profile_scope == "breakdown,repair,aging"


def test_list_failure_profiles_exposes_none_exponential_weibull_and_load_dependent_catalog() -> None:
    profiles = list_failure_profiles()

    assert [profile["profile_id"] for profile in profiles] == [
        "failure_none_v1",
        "failure_exponential_nominal_v1",
        "failure_weibull_aging_nominal_v1",
        "failure_load_dependent_nominal_v1",
    ]


def test_simulator_schedules_load_dependent_breakdown_with_profile_payload() -> None:
    config = apply_failure_profile(BenchmarkConfig(), "failure_load_dependent_nominal_v1")
    config.failure.mtbf_hours = 0.25
    config.failure.mttr_hours = 0.5

    simulator = EventDrivenSimulator(_failure_instance(), config)
    simulator.reset(seed=7, schedule_job_arrivals=False)
    assert simulator.activate_job(0, release_time=0.0) is True
    assert simulator.dispatch(0, 0) is True

    breakdown_events = [
        event
        for _, _, _, event in simulator.events._heap
        if event.kind == "breakdown"
    ]

    assert len(breakdown_events) == 1
    assert breakdown_events[0].payload["failure_profile_id"] == "failure_load_dependent_nominal_v1"
    assert float(breakdown_events[0].payload["load_multiplier"]) > 1.0
