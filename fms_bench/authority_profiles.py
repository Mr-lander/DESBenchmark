"""Named numeric profile overlays and minimal provenance metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import (
    BenchmarkConfig,
    DEFAULT_NUMERIC_PROFILE_ID,
    DEFAULT_NUMERIC_PROFILE_PROJECTION_KIND,
    DEFAULT_NUMERIC_PROFILE_PROVENANCE_CLASS,
    DEFAULT_NUMERIC_PROFILE_SCOPE,
    DEFAULT_NUMERIC_PROFILE_SOURCE_REF,
)


@dataclass(frozen=True)
class NumericProfileSpec:
    profile_id: str
    label: str
    provenance_class: str
    projection_kind: str
    scope: str
    source_ref: str
    source_urls: tuple[str, ...] = ()
    notes: str = ""
    energy_overrides: dict[str, Any] = field(default_factory=dict)
    carbon_overrides: dict[str, Any] = field(default_factory=dict)

    def public_metadata(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "label": self.label,
            "provenance_class": self.provenance_class,
            "projection_kind": self.projection_kind,
            "scope": self.scope,
            "source_ref": self.source_ref,
            "source_urls": list(self.source_urls),
            "notes": self.notes,
        }


_NUMERIC_PROFILE_REGISTRY: dict[str, NumericProfileSpec] = {
    DEFAULT_NUMERIC_PROFILE_ID: NumericProfileSpec(
        profile_id=DEFAULT_NUMERIC_PROFILE_ID,
        label="Benchmark default numeric profile",
        provenance_class=DEFAULT_NUMERIC_PROFILE_PROVENANCE_CLASS,
        projection_kind=DEFAULT_NUMERIC_PROFILE_PROJECTION_KIND,
        scope=DEFAULT_NUMERIC_PROFILE_SCOPE,
        source_ref=DEFAULT_NUMERIC_PROFILE_SOURCE_REF,
        notes=(
            "Internal benchmark defaults for machine power, tariff, and carbon placeholders. "
            "These are not authority-backed industrial truth values."
        ),
        energy_overrides={
            "peak_window": (8, 22),
            "peak_price": 0.20,
            "offpeak_price": 0.10,
            "active_power_kw": 10.0,
            "idle_ratio": 0.20,
        },
        carbon_overrides={
            "weekly_cap_tons": 1.9,
            "kg_per_kwh": 0.6,
        },
    ),
    "us_ca_pge_e20_camx_summer_2026q1": NumericProfileSpec(
        profile_id="us_ca_pge_e20_camx_summer_2026q1",
        label="PG&E E-20 summer TOU + EPA eGRID2023 CAMX overlay",
        provenance_class="authority_profiled_scenario",
        projection_kind="official_tariff_and_grid_factor_overlay",
        scope="tariff,carbon",
        source_ref="pge_e20_effective_2026-03-01__epa_egrid2023_camx",
        source_urls=(
            "https://www.pge.com/tariffs/assets/pdf/tariffbook/ELEC_SCHEDS_E-20.pdf",
            "https://www.epa.gov/egrid/summary-data",
            "https://www.epa.gov/system/files/documents/2025-01/egrid2023_technical_guide.pdf",
        ),
        notes=(
            "Simplified two-band summer peak/off-peak overlay derived from PG&E Schedule E-20 "
            "and EPA eGRID2023 CAMX. Machine power and idle ratio remain benchmark defaults."
        ),
        energy_overrides={
            "peak_window": (12, 18),
            "peak_price": 0.12385,
            "offpeak_price": 0.11840,
        },
        carbon_overrides={
            "kg_per_kwh": 0.19503700802971002,
        },
    ),
    "us_ca_pge_e20_camx_summer_2026q1__haas_vf2_spindle_proxy_v1": NumericProfileSpec(
        profile_id="us_ca_pge_e20_camx_summer_2026q1__haas_vf2_spindle_proxy_v1",
        label="PG&E E-20 + EPA eGRID2023 CAMX + Haas VF-2 spindle-power proxy overlay",
        provenance_class="authority_profiled_scenario_with_vendor_machine_proxy",
        projection_kind="official_tariff_grid_and_spindle_power_proxy_overlay",
        scope="machine_power_proxy,tariff,carbon",
        source_ref="pge_e20_effective_2026-03-01__epa_egrid2023_camx__haas_vf2_spec_page",
        source_urls=(
            "https://www.pge.com/tariffs/assets/pdf/tariffbook/ELEC_SCHEDS_E-20.pdf",
            "https://www.epa.gov/egrid/summary-data",
            "https://www.epa.gov/system/files/documents/2025-01/egrid2023_technical_guide.pdf",
            "https://www.haascnc.com/machines/vertical-mills/vf-series/models/small/vf-2.html",
        ),
        notes=(
            "Tariff and carbon values follow the PG&E E-20 and EPA eGRID2023 CAMX overlay. "
            "active_power_kw is an opt-in vendor spindle-power proxy using the Haas VF-2 "
            "standard spindle rating of 22.4 kW; it is not a measured busy-state factory load. "
            "idle_ratio remains the benchmark default."
        ),
        energy_overrides={
            "peak_window": (12, 18),
            "peak_price": 0.12385,
            "offpeak_price": 0.11840,
            "active_power_kw": 22.4,
        },
        carbon_overrides={
            "kg_per_kwh": 0.19503700802971002,
        },
    ),
}


def get_numeric_profile(profile_id: str) -> NumericProfileSpec:
    try:
        return _NUMERIC_PROFILE_REGISTRY[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown numeric profile: {profile_id}") from exc


def list_numeric_profiles() -> list[dict[str, object]]:
    return [profile.public_metadata() for profile in _NUMERIC_PROFILE_REGISTRY.values()]


def apply_numeric_profile(
    config: BenchmarkConfig,
    profile_id: str | None,
) -> BenchmarkConfig:
    if profile_id is None:
        return config

    profile = get_numeric_profile(profile_id)
    for key, value in profile.energy_overrides.items():
        setattr(config.energy, key, value)
    for key, value in profile.carbon_overrides.items():
        setattr(config.carbon, key, value)
    config.numeric_profile_id = profile.profile_id
    config.numeric_profile_provenance_class = profile.provenance_class
    config.numeric_profile_projection_kind = profile.projection_kind
    config.numeric_profile_scope = profile.scope
    config.numeric_profile_source_ref = profile.source_ref
    return config
