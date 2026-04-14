"""Tests for named numeric profile overlays and provenance metadata."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FLEX_ROOT = PROJECT_ROOT / "external" / "benchmarks" / "flexible jobshop" / "instances"


def test_benchmark_config_defaults_to_named_placeholder_profile() -> None:
    from fms_bench.config import BenchmarkConfig

    config = BenchmarkConfig()

    assert config.energy.active_power_kw == 10.0
    assert config.numeric_profile_id == "benchmark_default_v1"
    assert config.numeric_profile_provenance_class == "engineering_placeholder"
    assert config.numeric_profile_projection_kind == "generic_benchmark_default"
    assert config.numeric_profile_scope == "energy_power,tariff,carbon"
    assert config.numeric_profile_source_ref == "internal_benchmark_defaults_v1"


def test_authority_profile_overlay_updates_tariff_and_carbon_but_not_power_defaults() -> None:
    from fms_bench.authority_profiles import apply_numeric_profile
    from fms_bench.config import BenchmarkConfig

    config = apply_numeric_profile(
        BenchmarkConfig(),
        "us_ca_pge_e20_camx_summer_2026q1",
    )

    assert config.energy.peak_window == (12, 18)
    assert config.energy.peak_price == 0.12385
    assert config.energy.offpeak_price == 0.11840
    assert config.carbon.kg_per_kwh == 0.19503700802971002
    assert config.energy.active_power_kw == 10.0
    assert config.energy.idle_ratio == 0.20
    assert config.numeric_profile_id == "us_ca_pge_e20_camx_summer_2026q1"
    assert config.numeric_profile_provenance_class == "authority_profiled_scenario"
    assert config.numeric_profile_projection_kind == "official_tariff_and_grid_factor_overlay"
    assert config.numeric_profile_scope == "tariff,carbon"
    assert config.numeric_profile_source_ref == "pge_e20_effective_2026-03-01__epa_egrid2023_camx"


def test_machine_power_proxy_profile_updates_active_power_with_explicit_proxy_metadata() -> None:
    from fms_bench.authority_profiles import apply_numeric_profile
    from fms_bench.config import BenchmarkConfig

    config = apply_numeric_profile(
        BenchmarkConfig(),
        "us_ca_pge_e20_camx_summer_2026q1__haas_vf2_spindle_proxy_v1",
    )

    assert config.energy.peak_window == (12, 18)
    assert config.energy.peak_price == 0.12385
    assert config.energy.offpeak_price == 0.11840
    assert config.carbon.kg_per_kwh == 0.19503700802971002
    assert config.energy.active_power_kw == 22.4
    assert config.energy.idle_ratio == 0.20
    assert (
        config.numeric_profile_id
        == "us_ca_pge_e20_camx_summer_2026q1__haas_vf2_spindle_proxy_v1"
    )
    assert (
        config.numeric_profile_provenance_class
        == "authority_profiled_scenario_with_vendor_machine_proxy"
    )
    assert (
        config.numeric_profile_projection_kind
        == "official_tariff_grid_and_spindle_power_proxy_overlay"
    )
    assert config.numeric_profile_scope == "machine_power_proxy,tariff,carbon"
    assert (
        config.numeric_profile_source_ref
        == "pge_e20_effective_2026-03-01__epa_egrid2023_camx__haas_vf2_spec_page"
    )


def test_available_numeric_profiles_expose_release_facing_metadata() -> None:
    from fms_bench.authority_profiles import list_numeric_profiles

    by_id = {profile["profile_id"]: profile for profile in list_numeric_profiles()}

    assert set(by_id) >= {
        "benchmark_default_v1",
        "us_ca_pge_e20_camx_summer_2026q1",
        "us_ca_pge_e20_camx_summer_2026q1__haas_vf2_spindle_proxy_v1",
    }
    authority = by_id["us_ca_pge_e20_camx_summer_2026q1"]
    assert authority["source_ref"] == "pge_e20_effective_2026-03-01__epa_egrid2023_camx"
    assert authority["source_urls"]
    assert "pge.com" in " ".join(authority["source_urls"]).lower()
    assert "epa.gov" in " ".join(authority["source_urls"]).lower()
    machine_proxy = by_id["us_ca_pge_e20_camx_summer_2026q1__haas_vf2_spindle_proxy_v1"]
    assert (
        machine_proxy["provenance_class"]
        == "authority_profiled_scenario_with_vendor_machine_proxy"
    )
    assert (
        machine_proxy["projection_kind"]
        == "official_tariff_grid_and_spindle_power_proxy_overlay"
    )
    assert machine_proxy["scope"] == "machine_power_proxy,tariff,carbon"
    assert (
        machine_proxy["source_ref"]
        == "pge_e20_effective_2026-03-01__epa_egrid2023_camx__haas_vf2_spec_page"
    )
    assert "haascnc.com" in " ".join(machine_proxy["source_urls"]).lower()
    assert "proxy" in str(machine_proxy["notes"]).lower()


def test_simulator_machine_power_reads_from_config_energy_defaults() -> None:
    from fms_bench import BenchmarkConfig, HolonicPlantEnv, PlantCellConfig

    config = BenchmarkConfig()
    config.failure.mtbf_hours = 0.0
    config.failure.mttr_hours = 0.0
    config.energy.active_power_kw = 7.5
    config.energy.idle_ratio = 0.4
    env = HolonicPlantEnv(
        cell_configs=[
            PlantCellConfig(
                cell_id=0,
                family_key="kacem",
                instance_path=FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
                instance_kind="kacem",
                label="cell-kacem",
                wip_cap=1,
            ),
        ],
        config=config,
        seed=13,
    )

    try:
        env.reset(seed=13)
        machine = next(iter(env.simulator.machines.values()))
        assert machine.active_power_kw == 7.5
        assert machine.idle_power_kw == 3.0
    finally:
        env.close()
