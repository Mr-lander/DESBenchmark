"""Tests for named due-date synthesis profiles."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from fms_bench import BenchmarkConfig, PlantCellConfig, build_plant_assembly
from fms_bench.due_date_profiles import (
    apply_due_date_profile,
    list_due_date_profiles,
    synthesize_due_time,
)
from fms_bench.models import Operation, OperationOption


def test_benchmark_config_defaults_to_named_twk_due_date_profile() -> None:
    config = BenchmarkConfig()

    assert config.due_date.rule_family == "TWK"
    assert config.due_date.allowance_factor == pytest.approx(2.0)
    assert config.due_date.use_release_time is False
    assert config.due_date.minimum_due_time == pytest.approx(1.0)
    assert config.due_date_profile_id == "twk_nominal_k2_default_v1"
    assert config.due_date_profile_provenance_class == "literature_supported_scenario_profile"
    assert config.due_date_profile_rule_family == "TWK"
    assert config.due_date_profile_source_ref == "gordon_proth_chu_2001_twk_family"


def test_default_due_date_profile_preserves_legacy_two_x_nominal_behavior() -> None:
    config = BenchmarkConfig()
    operations = [
        Operation(op_id=0, options=[OperationOption(machine_id=1, proc_time=3.0)]),
        Operation(op_id=1, options=[OperationOption(machine_id=1, proc_time=2.0)]),
    ]

    due_time = synthesize_due_time(
        operations=operations,
        release_time=0.0,
        config=config,
    )

    assert due_time == pytest.approx(10.0)


def test_alternate_due_date_profile_can_change_twk_allowance_factor() -> None:
    config = apply_due_date_profile(BenchmarkConfig(), "twk_nominal_k3_relaxed_v1")
    operations = [
        Operation(op_id=0, options=[OperationOption(machine_id=1, proc_time=3.0)]),
        Operation(op_id=1, options=[OperationOption(machine_id=1, proc_time=2.0)]),
    ]

    due_time = synthesize_due_time(
        operations=operations,
        release_time=0.0,
        config=config,
    )

    assert due_time == pytest.approx(15.0)
    assert config.due_date_profile_id == "twk_nominal_k3_relaxed_v1"
    assert config.due_date_profile_provenance_class == "literature_supported_scenario_profile"
    assert config.due_date_profile_rule_family == "TWK"


def test_build_plant_assembly_uses_configured_due_date_profile_for_missing_due_times(
    tmp_path: Path,
) -> None:
    instance_path = tmp_path / "toy.txt"
    instance_path.write_text("1 1\n1 1 1 5\n", encoding="utf-8")
    config = apply_due_date_profile(BenchmarkConfig(), "twk_nominal_k3_relaxed_v1")

    assembly = build_plant_assembly(
        [
            PlantCellConfig(
                cell_id=0,
                family_key="toy",
                instance_path=instance_path,
                instance_kind="fjsp",
            )
        ],
        config=config,
    )

    assert assembly.combined_jobs[0].due_time == pytest.approx(15.0)
    assert assembly.cells[0].template_jobs[0].due_time == pytest.approx(15.0)


def test_due_date_profile_registry_exposes_named_profiles() -> None:
    by_id = {profile["profile_id"]: profile for profile in list_due_date_profiles()}

    assert set(by_id) >= {"twk_nominal_k2_default_v1", "twk_nominal_k3_relaxed_v1"}
    assert by_id["twk_nominal_k2_default_v1"]["rule_family"] == "TWK"
    assert by_id["twk_nominal_k2_default_v1"]["source_urls"]
