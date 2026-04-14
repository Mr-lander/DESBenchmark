"""Named failure/disturbance profiles for benchmark scenario generation."""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    BenchmarkConfig,
    DEFAULT_FAILURE_PROFILE_ID,
    DEFAULT_FAILURE_PROFILE_PROVENANCE_CLASS,
    DEFAULT_FAILURE_PROFILE_SCOPE,
    DEFAULT_FAILURE_PROFILE_SOURCE_REF,
)


@dataclass(frozen=True)
class FailureProfileSpec:
    profile_id: str
    label: str
    provenance_class: str
    scope: str
    source_ref: str
    process: str
    mtbf_hours: float
    mttr_hours: float
    weibull_shape: float = 1.0
    load_dependent_alpha: float = 0.0
    load_reference_proc_hours: float = 4.0
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


_FAILURE_PROFILE_REGISTRY: dict[str, FailureProfileSpec] = {
    DEFAULT_FAILURE_PROFILE_ID: FailureProfileSpec(
        profile_id=DEFAULT_FAILURE_PROFILE_ID,
        label="No breakdown or repair disturbance profile",
        provenance_class=DEFAULT_FAILURE_PROFILE_PROVENANCE_CLASS,
        scope=DEFAULT_FAILURE_PROFILE_SCOPE,
        source_ref=DEFAULT_FAILURE_PROFILE_SOURCE_REF,
        process="none",
        mtbf_hours=0.0,
        mttr_hours=0.0,
    ),
    "failure_exponential_nominal_v1": FailureProfileSpec(
        profile_id="failure_exponential_nominal_v1",
        label="Nominal exponential breakdown and repair benchmark profile",
        provenance_class="benchmark_scenario_profile",
        scope="breakdown,repair,exponential",
        source_ref="internal_failure_exponential_nominal_v1",
        process="exponential",
        mtbf_hours=30.0,
        mttr_hours=3.0,
    ),
    "failure_weibull_aging_nominal_v1": FailureProfileSpec(
        profile_id="failure_weibull_aging_nominal_v1",
        label="Aging-style Weibull breakdown profile with exponential repair",
        provenance_class="benchmark_scenario_profile",
        scope="breakdown,repair,aging",
        source_ref="internal_failure_weibull_aging_nominal_v1",
        process="weibull",
        mtbf_hours=30.0,
        mttr_hours=3.0,
        weibull_shape=1.8,
    ),
    "failure_load_dependent_nominal_v1": FailureProfileSpec(
        profile_id="failure_load_dependent_nominal_v1",
        label="Load-dependent hazard profile with exponential repair",
        provenance_class="benchmark_scenario_profile",
        scope="breakdown,repair,load_dependent",
        source_ref="internal_failure_load_dependent_nominal_v1",
        process="load_dependent",
        mtbf_hours=30.0,
        mttr_hours=3.0,
        load_dependent_alpha=1.0,
        load_reference_proc_hours=4.0,
    ),
}


def get_failure_profile(profile_id: str) -> FailureProfileSpec:
    try:
        return _FAILURE_PROFILE_REGISTRY[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown failure profile: {profile_id}") from exc


def list_failure_profiles() -> list[dict[str, object]]:
    return [profile.public_metadata() for profile in _FAILURE_PROFILE_REGISTRY.values()]


def apply_failure_profile(
    config: BenchmarkConfig,
    profile_id: str | None,
) -> BenchmarkConfig:
    if profile_id is None:
        return config

    profile = get_failure_profile(profile_id)
    config.failure.process = profile.process
    config.failure.mtbf_hours = float(profile.mtbf_hours)
    config.failure.mttr_hours = float(profile.mttr_hours)
    config.failure.weibull_shape = float(profile.weibull_shape)
    config.failure.load_dependent_alpha = float(profile.load_dependent_alpha)
    config.failure.load_reference_proc_hours = float(profile.load_reference_proc_hours)
    config.failure_profile_id = profile.profile_id
    config.failure_profile_provenance_class = profile.provenance_class
    config.failure_profile_scope = profile.scope
    config.failure_profile_source_ref = profile.source_ref
    return config
