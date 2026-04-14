"""Named arrival profiles for shared-backlog benchmark scenario generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

from .config import (
    BenchmarkConfig,
    DEFAULT_ARRIVAL_PROFILE_ID,
    DEFAULT_ARRIVAL_PROFILE_PROVENANCE_CLASS,
    DEFAULT_ARRIVAL_PROFILE_SCOPE,
    DEFAULT_ARRIVAL_PROFILE_SOURCE_REF,
)


@dataclass(frozen=True)
class ArrivalProfileSpec:
    profile_id: str
    label: str
    provenance_class: str
    scope: str
    source_ref: str
    process: str
    rate_per_hour: float
    initial_visible_backlog: int
    lambda_low: float = 0.0
    lambda_high: float = 0.0
    batch_size_distribution: str = "singleton"
    batch_size_low: int = 1
    batch_size_high: int = 1
    batch_size_poisson_lambda: float = 1.0
    nhpp_cycle_hours: float = 24.0
    source_urls: tuple[str, ...] = ()

    def public_metadata(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "label": self.label,
            "provenance_class": self.provenance_class,
            "scope": self.scope,
            "source_ref": self.source_ref,
            "source_urls": list(self.source_urls),
        }


_ARRIVAL_PROFILE_REGISTRY: dict[str, ArrivalProfileSpec] = {
    DEFAULT_ARRIVAL_PROFILE_ID: ArrivalProfileSpec(
        profile_id=DEFAULT_ARRIVAL_PROFILE_ID,
        label="Static shared-backlog visibility with no timed online arrivals",
        provenance_class=DEFAULT_ARRIVAL_PROFILE_PROVENANCE_CLASS,
        scope=DEFAULT_ARRIVAL_PROFILE_SCOPE,
        source_ref=DEFAULT_ARRIVAL_PROFILE_SOURCE_REF,
        process="static_release",
        rate_per_hour=0.0,
        initial_visible_backlog=10_000,
    ),
    "arrival_hpp_singleton_nominal_v1": ArrivalProfileSpec(
        profile_id="arrival_hpp_singleton_nominal_v1",
        label="Homogeneous Poisson singleton online arrivals over shared backlog items",
        provenance_class="benchmark_scenario_profile",
        scope="shared_backlog_visibility,timed_online_arrivals",
        source_ref="internal_arrival_hpp_singleton_nominal_v1",
        process="hpp_singleton",
        rate_per_hour=1.5,
        initial_visible_backlog=1,
    ),
    "arrival_nhpp_singleton_nominal_v1": ArrivalProfileSpec(
        profile_id="arrival_nhpp_singleton_nominal_v1",
        label="Nonhomogeneous Poisson singleton online arrivals over shared backlog items",
        provenance_class="benchmark_scenario_profile",
        scope="shared_backlog_visibility,timed_online_arrivals,nhpp",
        source_ref="internal_arrival_nhpp_singleton_nominal_v1",
        process="nhpp_singleton",
        rate_per_hour=1.2,
        lambda_low=0.4,
        lambda_high=1.8,
        nhpp_cycle_hours=12.0,
        initial_visible_backlog=1,
    ),
    "arrival_nhpp_random_batch_nominal_v1": ArrivalProfileSpec(
        profile_id="arrival_nhpp_random_batch_nominal_v1",
        label="Nonhomogeneous Poisson compound-batch online arrivals over shared backlog items",
        provenance_class="benchmark_scenario_profile",
        scope="shared_backlog_visibility,timed_online_arrivals,nhpp,compound_batch",
        source_ref="internal_arrival_nhpp_random_batch_nominal_v1",
        process="nhpp_random_batch",
        rate_per_hour=1.2,
        lambda_low=0.5,
        lambda_high=2.4,
        batch_size_distribution="fixed",
        batch_size_low=2,
        batch_size_high=2,
        nhpp_cycle_hours=12.0,
        initial_visible_backlog=1,
    ),
    "arrival_nhpp_random_batch_overlap_v1": ArrivalProfileSpec(
        profile_id="arrival_nhpp_random_batch_overlap_v1",
        label="Overlap-pressure NHPP compound-batch arrivals with a wider initial visible backlog",
        provenance_class="benchmark_scenario_profile",
        scope="shared_backlog_visibility,timed_online_arrivals,nhpp,compound_batch,overlap_pressure",
        source_ref="internal_arrival_nhpp_random_batch_overlap_v1",
        process="nhpp_random_batch",
        rate_per_hour=1.6,
        lambda_low=0.7,
        lambda_high=3.0,
        batch_size_distribution="fixed",
        batch_size_low=2,
        batch_size_high=2,
        nhpp_cycle_hours=12.0,
        initial_visible_backlog=3,
    ),
    "arrival_nhpp_random_batch_overlap_a3c9_v2": ArrivalProfileSpec(
        profile_id="arrival_nhpp_random_batch_overlap_a3c9_v2",
        label="A3C9 overlap-pressure NHPP compound-batch arrivals with stronger early shared-backlog branching",
        provenance_class="benchmark_scenario_profile",
        scope="shared_backlog_visibility,timed_online_arrivals,nhpp,compound_batch,overlap_pressure",
        source_ref="internal_arrival_nhpp_random_batch_overlap_a3c9_v2",
        process="nhpp_random_batch",
        rate_per_hour=1.8,
        lambda_low=0.7,
        lambda_high=3.4,
        batch_size_distribution="fixed",
        batch_size_low=2,
        batch_size_high=2,
        nhpp_cycle_hours=12.0,
        initial_visible_backlog=3,
    ),
}


@dataclass(frozen=True)
class ArrivalBatchEventSpec:
    event_id: int
    time: float
    shared_backlog_ids: tuple[int, ...]
    process: str

    @property
    def batch_size(self) -> int:
        return len(self.shared_backlog_ids)


def get_arrival_profile(profile_id: str) -> ArrivalProfileSpec:
    try:
        return _ARRIVAL_PROFILE_REGISTRY[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown arrival profile: {profile_id}") from exc


def list_arrival_profiles() -> list[dict[str, object]]:
    return [profile.public_metadata() for profile in _ARRIVAL_PROFILE_REGISTRY.values()]


def apply_arrival_profile(
    config: BenchmarkConfig,
    profile_id: str | None,
) -> BenchmarkConfig:
    if profile_id is None:
        return config

    profile = get_arrival_profile(profile_id)
    config.arrival.process = profile.process
    config.arrival.rate_per_hour = float(profile.rate_per_hour)
    config.arrival.initial_visible_backlog = int(profile.initial_visible_backlog)
    config.arrival.lambda_low = float(profile.lambda_low)
    config.arrival.lambda_high = float(profile.lambda_high)
    config.arrival.batch_size_distribution = str(profile.batch_size_distribution)
    config.arrival.batch_size_low = int(profile.batch_size_low)
    config.arrival.batch_size_high = int(profile.batch_size_high)
    config.arrival.batch_size_poisson_lambda = float(profile.batch_size_poisson_lambda)
    config.arrival.nhpp_cycle_hours = float(profile.nhpp_cycle_hours)
    config.arrival_profile_id = profile.profile_id
    config.arrival_profile_provenance_class = profile.provenance_class
    config.arrival_profile_scope = profile.scope
    config.arrival_profile_source_ref = profile.source_ref
    return config


def schedule_online_arrival_batches(
    backlog_ids: Iterable[int],
    *,
    config: BenchmarkConfig,
    rng: object,
    start_time: float = 0.0,
) -> list[ArrivalBatchEventSpec]:
    remaining_ids = [int(backlog_id) for backlog_id in backlog_ids]
    process = str(config.arrival.process)
    if process == "static_release" or not remaining_ids:
        return []

    current_time = float(start_time)
    events: list[ArrivalBatchEventSpec] = []
    event_id = 0
    while remaining_ids:
        current_time += _sample_interarrival_delay(config=config, rng=rng, current_time=current_time)
        batch_size = min(len(remaining_ids), _sample_arrival_batch_size(config=config, rng=rng))
        shared_backlog_ids = tuple(remaining_ids[:batch_size])
        remaining_ids = remaining_ids[batch_size:]
        events.append(
            ArrivalBatchEventSpec(
                event_id=event_id,
                time=float(current_time),
                shared_backlog_ids=shared_backlog_ids,
                process=process,
            )
        )
        event_id += 1
    return events


def _sample_interarrival_delay(
    *,
    config: BenchmarkConfig,
    rng: object,
    current_time: float,
) -> float:
    process = str(config.arrival.process)
    if process == "hpp_singleton":
        rate = max(float(config.arrival.rate_per_hour), 1e-9)
        return max(float(rng.exponential(1.0 / rate)), 1e-9)
    if process in {"nhpp_singleton", "nhpp_random_batch"}:
        lambda_max = max(float(config.arrival.lambda_low), float(config.arrival.lambda_high), 1e-9)
        proposal_time = float(current_time)
        while True:
            proposal_time += max(float(rng.exponential(1.0 / lambda_max)), 1e-9)
            accept_ratio = _nhpp_rate(config=config, time_hours=proposal_time) / lambda_max
            if float(rng.random()) <= accept_ratio:
                return max(proposal_time - float(current_time), 1e-9)
    raise ValueError(f"Unsupported arrival process: {process}")


def _sample_arrival_batch_size(*, config: BenchmarkConfig, rng: object) -> int:
    distribution = str(config.arrival.batch_size_distribution)
    if distribution == "singleton":
        return 1
    if distribution == "fixed":
        return max(1, int(config.arrival.batch_size_low))
    if distribution == "uniform_inclusive":
        low = int(config.arrival.batch_size_low)
        high = max(low, int(config.arrival.batch_size_high))
        return max(1, int(rng.integers(low, high + 1)))
    if distribution == "poisson_plus_one":
        lam = max(float(config.arrival.batch_size_poisson_lambda), 1e-9)
        return max(1, 1 + int(rng.poisson(lam)))
    raise ValueError(f"Unsupported arrival batch distribution: {distribution}")


def _nhpp_rate(*, config: BenchmarkConfig, time_hours: float) -> float:
    cycle = max(float(config.arrival.nhpp_cycle_hours), 1e-9)
    phase = (float(time_hours) % cycle) / cycle
    midpoint = 0.5 * (float(config.arrival.lambda_high) + float(config.arrival.lambda_low))
    amplitude = 0.5 * (float(config.arrival.lambda_high) - float(config.arrival.lambda_low))
    return max(1e-9, midpoint + amplitude * math.sin(2.0 * math.pi * phase))
