"""Named due-date assignment profiles for benchmark scenario generation."""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    BenchmarkConfig,
    DEFAULT_DUE_DATE_PROFILE_ID,
    DEFAULT_DUE_DATE_PROFILE_PROVENANCE_CLASS,
    DEFAULT_DUE_DATE_PROFILE_RULE_FAMILY,
    DEFAULT_DUE_DATE_PROFILE_SOURCE_REF,
)
from .models import Operation


@dataclass(frozen=True)
class DueDateProfileSpec:
    profile_id: str
    label: str
    rule_family: str
    provenance_class: str
    source_ref: str
    source_urls: tuple[str, ...]
    allowance_factor: float
    use_release_time: bool = False
    minimum_due_time: float = 1.0

    def public_metadata(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "rule_family": self.rule_family,
            "provenance_class": self.provenance_class,
            "source_ref": self.source_ref,
            "source_urls": list(self.source_urls),
        }


_DUE_DATE_PROFILE_REGISTRY: dict[str, DueDateProfileSpec] = {
    DEFAULT_DUE_DATE_PROFILE_ID: DueDateProfileSpec(
        profile_id=DEFAULT_DUE_DATE_PROFILE_ID,
        label="TWK family due-date fallback with factor 2 on nominal shortest-path work content",
        rule_family=DEFAULT_DUE_DATE_PROFILE_RULE_FAMILY,
        provenance_class=DEFAULT_DUE_DATE_PROFILE_PROVENANCE_CLASS,
        source_ref=DEFAULT_DUE_DATE_PROFILE_SOURCE_REF,
        source_urls=(
            "https://doi.org/10.1080/09537280110069621",
            "https://doi.org/10.1016/S0377-2217(01)00181-3",
        ),
        allowance_factor=2.0,
        use_release_time=False,
        minimum_due_time=1.0,
    ),
    "twk_nominal_k3_relaxed_v1": DueDateProfileSpec(
        profile_id="twk_nominal_k3_relaxed_v1",
        label="TWK family relaxed due-date profile with factor 3 on nominal shortest-path work content",
        rule_family="TWK",
        provenance_class="literature_supported_scenario_profile",
        source_ref="gordon_proth_chu_2001_twk_family",
        source_urls=(
            "https://doi.org/10.1080/09537280110069621",
            "https://doi.org/10.1016/S0377-2217(01)00181-3",
        ),
        allowance_factor=3.0,
        use_release_time=False,
        minimum_due_time=1.0,
    ),
}


def get_due_date_profile(profile_id: str) -> DueDateProfileSpec:
    try:
        return _DUE_DATE_PROFILE_REGISTRY[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown due-date profile: {profile_id}") from exc


def list_due_date_profiles() -> list[dict[str, object]]:
    return [profile.public_metadata() for profile in _DUE_DATE_PROFILE_REGISTRY.values()]


def apply_due_date_profile(
    config: BenchmarkConfig,
    profile_id: str | None,
) -> BenchmarkConfig:
    if profile_id is None:
        return config

    profile = get_due_date_profile(profile_id)
    config.due_date_profile_id = profile.profile_id
    config.due_date.rule_family = profile.rule_family
    config.due_date.allowance_factor = float(profile.allowance_factor)
    config.due_date.use_release_time = bool(profile.use_release_time)
    config.due_date.minimum_due_time = float(profile.minimum_due_time)
    config.due_date_profile_rule_family = profile.rule_family
    config.due_date_profile_provenance_class = profile.provenance_class
    config.due_date_profile_source_ref = profile.source_ref
    return config


def synthesize_due_time(
    *,
    operations: list[Operation],
    release_time: float,
    config: BenchmarkConfig,
) -> float:
    nominal_work_content = sum(
        min(option.proc_time for option in op.options)
        for op in operations
        if op.options
    )

    due_time = max(
        float(config.due_date.minimum_due_time),
        float(config.due_date.allowance_factor) * float(nominal_work_content),
    )
    if config.due_date.use_release_time:
        due_time += max(0.0, float(release_time))
    return float(due_time)
