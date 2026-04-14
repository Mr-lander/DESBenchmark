"""Tests for setup/changeover profile overlays."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import BenchmarkConfig
from fms_bench.config import SETUP_PROFILE_METADATA_FIELDS
from fms_bench.setup_profiles import apply_setup_profile, list_setup_profiles


def test_setup_profiles_apply_nominal_family_change_overlay() -> None:
    config = BenchmarkConfig()

    updated = apply_setup_profile(config, "setup_family_change_nominal_v1")

    assert updated.setup.enabled is True
    assert updated.setup.rule == "job_family_change"
    assert updated.setup.time_ratio_to_proc == 0.25
    assert updated.setup.minimum_time_hours == 0.25
    assert updated.setup.setup_power_ratio == 0.6
    assert updated.setup_profile_id == "setup_family_change_nominal_v1"
    assert updated.setup_profile_provenance_class == "benchmark_scenario_profile"
    assert updated.setup_profile_scope == "setup_time,setup_power"


def test_setup_profiles_surface_public_metadata_fields() -> None:
    profiles = list_setup_profiles()

    assert SETUP_PROFILE_METADATA_FIELDS == (
        "setup_profile_id",
        "setup_profile_provenance_class",
        "setup_profile_scope",
        "setup_profile_source_ref",
    )
    assert [profile["profile_id"] for profile in profiles] == [
        "setup_none_v1",
        "setup_family_change_nominal_v1",
        "setup_family_change_heavy_v1",
    ]
