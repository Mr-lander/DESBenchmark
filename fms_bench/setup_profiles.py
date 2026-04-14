"""Named setup/changeover profiles for benchmark scenario generation."""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    BenchmarkConfig,
    DEFAULT_SETUP_PROFILE_ID,
    DEFAULT_SETUP_PROFILE_PROVENANCE_CLASS,
    DEFAULT_SETUP_PROFILE_SCOPE,
    DEFAULT_SETUP_PROFILE_SOURCE_REF,
)


@dataclass(frozen=True)
class SetupProfileSpec:
    profile_id: str
    label: str
    provenance_class: str
    scope: str
    source_ref: str
    enabled: bool
    rule: str
    time_ratio_to_proc: float
    minimum_time_hours: float
    same_family_time_hours: float
    setup_power_ratio: float
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


_SETUP_PROFILE_REGISTRY: dict[str, SetupProfileSpec] = {
    DEFAULT_SETUP_PROFILE_ID: SetupProfileSpec(
        profile_id=DEFAULT_SETUP_PROFILE_ID,
        label="No explicit setup/changeover benchmark profile",
        provenance_class=DEFAULT_SETUP_PROFILE_PROVENANCE_CLASS,
        scope=DEFAULT_SETUP_PROFILE_SCOPE,
        source_ref=DEFAULT_SETUP_PROFILE_SOURCE_REF,
        enabled=False,
        rule="job_family_change",
        time_ratio_to_proc=0.0,
        minimum_time_hours=0.0,
        same_family_time_hours=0.0,
        setup_power_ratio=0.0,
    ),
    "setup_family_change_nominal_v1": SetupProfileSpec(
        profile_id="setup_family_change_nominal_v1",
        label="Nominal family-change proxy with proportional setup time",
        provenance_class="benchmark_scenario_profile",
        scope="setup_time,setup_power",
        source_ref="internal_setup_family_change_nominal_v1",
        enabled=True,
        rule="job_family_change",
        time_ratio_to_proc=0.25,
        minimum_time_hours=0.25,
        same_family_time_hours=0.0,
        setup_power_ratio=0.6,
    ),
    "setup_family_change_heavy_v1": SetupProfileSpec(
        profile_id="setup_family_change_heavy_v1",
        label="Heavy family-change proxy with larger proportional setup time",
        provenance_class="benchmark_scenario_profile",
        scope="setup_time,setup_power",
        source_ref="internal_setup_family_change_heavy_v1",
        enabled=True,
        rule="job_family_change",
        time_ratio_to_proc=0.5,
        minimum_time_hours=0.5,
        same_family_time_hours=0.0,
        setup_power_ratio=0.75,
    ),
}


def get_setup_profile(profile_id: str) -> SetupProfileSpec:
    try:
        return _SETUP_PROFILE_REGISTRY[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown setup profile: {profile_id}") from exc


def list_setup_profiles() -> list[dict[str, object]]:
    return [profile.public_metadata() for profile in _SETUP_PROFILE_REGISTRY.values()]


def apply_setup_profile(
    config: BenchmarkConfig,
    profile_id: str | None,
) -> BenchmarkConfig:
    if profile_id is None:
        return config

    profile = get_setup_profile(profile_id)
    config.setup.enabled = bool(profile.enabled)
    config.setup.rule = profile.rule
    config.setup.time_ratio_to_proc = float(profile.time_ratio_to_proc)
    config.setup.minimum_time_hours = float(profile.minimum_time_hours)
    config.setup.same_family_time_hours = float(profile.same_family_time_hours)
    config.setup.setup_power_ratio = float(profile.setup_power_ratio)
    config.setup_profile_id = profile.profile_id
    config.setup_profile_provenance_class = profile.provenance_class
    config.setup_profile_scope = profile.scope
    config.setup_profile_source_ref = profile.source_ref
    return config
