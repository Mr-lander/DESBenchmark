"""Configuration defaults for FMS-Bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

DEFAULT_NUMERIC_PROFILE_ID = "benchmark_default_v1"
DEFAULT_NUMERIC_PROFILE_PROVENANCE_CLASS = "engineering_placeholder"
DEFAULT_NUMERIC_PROFILE_PROJECTION_KIND = "generic_benchmark_default"
DEFAULT_NUMERIC_PROFILE_SCOPE = "energy_power,tariff,carbon"
DEFAULT_NUMERIC_PROFILE_SOURCE_REF = "internal_benchmark_defaults_v1"
NUMERIC_PROFILE_METADATA_FIELDS = (
    "numeric_profile_id",
    "numeric_profile_provenance_class",
    "numeric_profile_projection_kind",
    "numeric_profile_scope",
    "numeric_profile_source_ref",
)
DEFAULT_DUE_DATE_PROFILE_ID = "twk_nominal_k2_default_v1"
DEFAULT_DUE_DATE_PROFILE_RULE_FAMILY = "TWK"
DEFAULT_DUE_DATE_PROFILE_PROVENANCE_CLASS = "literature_supported_scenario_profile"
DEFAULT_DUE_DATE_PROFILE_SOURCE_REF = "gordon_proth_chu_2001_twk_family"
DUE_DATE_PROFILE_METADATA_FIELDS = (
    "due_date_profile_id",
    "due_date_profile_rule_family",
    "due_date_profile_provenance_class",
    "due_date_profile_source_ref",
)
DEFAULT_SETUP_PROFILE_ID = "setup_none_v1"
DEFAULT_SETUP_PROFILE_PROVENANCE_CLASS = "benchmark_scenario_profile"
DEFAULT_SETUP_PROFILE_SCOPE = "setup_time,setup_power"
DEFAULT_SETUP_PROFILE_SOURCE_REF = "internal_setup_defaults_v1"
SETUP_PROFILE_METADATA_FIELDS = (
    "setup_profile_id",
    "setup_profile_provenance_class",
    "setup_profile_scope",
    "setup_profile_source_ref",
)
DEFAULT_ARRIVAL_PROFILE_ID = "arrival_static_release_v1"
DEFAULT_ARRIVAL_PROFILE_PROVENANCE_CLASS = "benchmark_scenario_profile"
DEFAULT_ARRIVAL_PROFILE_SCOPE = "shared_backlog_visibility"
DEFAULT_ARRIVAL_PROFILE_SOURCE_REF = "internal_arrival_static_release_v1"
ARRIVAL_PROFILE_METADATA_FIELDS = (
    "arrival_profile_id",
    "arrival_profile_provenance_class",
    "arrival_profile_scope",
    "arrival_profile_source_ref",
)
DEFAULT_FAILURE_PROFILE_ID = "failure_none_v1"
DEFAULT_FAILURE_PROFILE_PROVENANCE_CLASS = "benchmark_scenario_profile"
DEFAULT_FAILURE_PROFILE_SCOPE = "breakdown,repair"
DEFAULT_FAILURE_PROFILE_SOURCE_REF = "internal_failure_none_v1"
FAILURE_PROFILE_METADATA_FIELDS = (
    "failure_profile_id",
    "failure_profile_provenance_class",
    "failure_profile_scope",
    "failure_profile_source_ref",
)


@dataclass
class OrderArrivalConfig:
    """Poisson or compound Poisson order arrivals."""

    process: str = "static_release"
    rate_per_hour: float = 0.0
    initial_visible_backlog: int = 10_000
    lambda_low: float = 0.5
    lambda_high: float = 5.0
    burst_multiplier: float = 10.0
    batch_size_distribution: str = "singleton"
    batch_size_low: int = 1
    batch_size_high: int = 1
    batch_size_poisson_lambda: float = 1.0
    nhpp_cycle_hours: float = 24.0


@dataclass
class ProcessingTimeConfig:
    """Gamma variability with coefficient of variation."""

    distribution: str = "gamma"
    cv_low: float = 0.2
    cv_high: float = 0.5


@dataclass
class FailureConfig:
    """Exponential failure and repair."""

    process: str = "none"
    mtbf_hours: float = 500.0
    mttr_hours: float = 4.0
    weibull_shape: float = 1.0
    load_dependent_alpha: float = 0.0
    load_reference_proc_hours: float = 4.0


@dataclass
class EnergyConfig:
    """TOU pricing and machine power model."""

    peak_window: Tuple[int, int] = (8, 22)
    peak_price: float = 0.20
    offpeak_price: float = 0.10
    active_power_kw: float = 10.0
    idle_ratio: float = 0.20


@dataclass
class CarbonConfig:
    """Weekly carbon cap and conversion."""

    weekly_cap_tons: float = 1.9
    kg_per_kwh: float = 0.6


@dataclass
class DueDateConfig:
    """Named due-date assignment configuration."""

    rule_family: str = DEFAULT_DUE_DATE_PROFILE_RULE_FAMILY
    allowance_factor: float = 2.0
    use_release_time: bool = False
    minimum_due_time: float = 1.0


@dataclass
class SetupConfig:
    """Named setup/changeover benchmark scenario configuration."""

    enabled: bool = False
    rule: str = "job_family_change"
    time_ratio_to_proc: float = 0.0
    minimum_time_hours: float = 0.0
    same_family_time_hours: float = 0.0
    setup_power_ratio: float = 0.0


@dataclass
class BenchmarkConfig:
    """Top-level benchmark settings."""

    horizon_hours: int = 168
    decision_interval_minutes: int = 10
    arrival: OrderArrivalConfig = field(default_factory=OrderArrivalConfig)
    processing: ProcessingTimeConfig = field(default_factory=ProcessingTimeConfig)
    failure: FailureConfig = field(default_factory=FailureConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    carbon: CarbonConfig = field(default_factory=CarbonConfig)
    due_date: DueDateConfig = field(default_factory=DueDateConfig)
    setup: SetupConfig = field(default_factory=SetupConfig)
    numeric_profile_id: str = DEFAULT_NUMERIC_PROFILE_ID
    numeric_profile_provenance_class: str = DEFAULT_NUMERIC_PROFILE_PROVENANCE_CLASS
    numeric_profile_projection_kind: str = DEFAULT_NUMERIC_PROFILE_PROJECTION_KIND
    numeric_profile_scope: str = DEFAULT_NUMERIC_PROFILE_SCOPE
    numeric_profile_source_ref: str = DEFAULT_NUMERIC_PROFILE_SOURCE_REF
    due_date_profile_id: str = DEFAULT_DUE_DATE_PROFILE_ID
    due_date_profile_rule_family: str = DEFAULT_DUE_DATE_PROFILE_RULE_FAMILY
    due_date_profile_provenance_class: str = DEFAULT_DUE_DATE_PROFILE_PROVENANCE_CLASS
    due_date_profile_source_ref: str = DEFAULT_DUE_DATE_PROFILE_SOURCE_REF
    setup_profile_id: str = DEFAULT_SETUP_PROFILE_ID
    setup_profile_provenance_class: str = DEFAULT_SETUP_PROFILE_PROVENANCE_CLASS
    setup_profile_scope: str = DEFAULT_SETUP_PROFILE_SCOPE
    setup_profile_source_ref: str = DEFAULT_SETUP_PROFILE_SOURCE_REF
    arrival_profile_id: str = DEFAULT_ARRIVAL_PROFILE_ID
    arrival_profile_provenance_class: str = DEFAULT_ARRIVAL_PROFILE_PROVENANCE_CLASS
    arrival_profile_scope: str = DEFAULT_ARRIVAL_PROFILE_SCOPE
    arrival_profile_source_ref: str = DEFAULT_ARRIVAL_PROFILE_SOURCE_REF
    failure_profile_id: str = DEFAULT_FAILURE_PROFILE_ID
    failure_profile_provenance_class: str = DEFAULT_FAILURE_PROFILE_PROVENANCE_CLASS
    failure_profile_scope: str = DEFAULT_FAILURE_PROFILE_SCOPE
    failure_profile_source_ref: str = DEFAULT_FAILURE_PROFILE_SOURCE_REF
