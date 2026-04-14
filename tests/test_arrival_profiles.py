"""Tests for named arrival benchmark scenario profiles."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import BenchmarkConfig
from fms_bench.arrival_profiles import apply_arrival_profile, list_arrival_profiles


def test_apply_arrival_profile_sets_hpp_singleton_metadata_and_parameters() -> None:
    config = apply_arrival_profile(
        BenchmarkConfig(),
        "arrival_hpp_singleton_nominal_v1",
    )

    assert config.arrival.process == "hpp_singleton"
    assert config.arrival.rate_per_hour > 0.0
    assert config.arrival.initial_visible_backlog == 1
    assert config.arrival_profile_id == "arrival_hpp_singleton_nominal_v1"
    assert config.arrival_profile_provenance_class == "benchmark_scenario_profile"
    assert config.arrival_profile_scope == "shared_backlog_visibility,timed_online_arrivals"
    assert (
        config.arrival_profile_source_ref
        == "internal_arrival_hpp_singleton_nominal_v1"
    )


def test_apply_arrival_profile_sets_nhpp_batch_metadata_and_parameters() -> None:
    config = apply_arrival_profile(
        BenchmarkConfig(),
        "arrival_nhpp_random_batch_nominal_v1",
    )

    assert config.arrival.process == "nhpp_random_batch"
    assert config.arrival.lambda_low > 0.0
    assert config.arrival.lambda_high > config.arrival.lambda_low
    assert config.arrival.batch_size_distribution == "fixed"
    assert config.arrival.batch_size_low == 2
    assert config.arrival.batch_size_high == 2
    assert config.arrival.initial_visible_backlog == 1
    assert config.arrival_profile_id == "arrival_nhpp_random_batch_nominal_v1"
    assert config.arrival_profile_scope == (
        "shared_backlog_visibility,timed_online_arrivals,nhpp,compound_batch"
    )


def test_apply_arrival_profile_sets_overlap_nhpp_batch_parameters() -> None:
    config = apply_arrival_profile(
        BenchmarkConfig(),
        "arrival_nhpp_random_batch_overlap_v1",
    )

    assert config.arrival.process == "nhpp_random_batch"
    assert config.arrival.lambda_low == 0.7
    assert config.arrival.lambda_high == 3.0
    assert config.arrival.batch_size_distribution == "fixed"
    assert config.arrival.batch_size_low == 2
    assert config.arrival.batch_size_high == 2
    assert config.arrival.initial_visible_backlog == 3
    assert config.arrival_profile_id == "arrival_nhpp_random_batch_overlap_v1"
    assert config.arrival_profile_scope == (
        "shared_backlog_visibility,timed_online_arrivals,nhpp,compound_batch,overlap_pressure"
    )


def test_apply_arrival_profile_sets_a3c9_overlap_nhpp_batch_parameters() -> None:
    config = apply_arrival_profile(
        BenchmarkConfig(),
        "arrival_nhpp_random_batch_overlap_a3c9_v2",
    )

    assert config.arrival.process == "nhpp_random_batch"
    assert config.arrival.rate_per_hour == 1.8
    assert config.arrival.lambda_low == 0.7
    assert config.arrival.lambda_high == 3.4
    assert config.arrival.batch_size_distribution == "fixed"
    assert config.arrival.batch_size_low == 2
    assert config.arrival.batch_size_high == 2
    assert config.arrival.initial_visible_backlog == 3
    assert config.arrival_profile_id == "arrival_nhpp_random_batch_overlap_a3c9_v2"
    assert config.arrival_profile_scope == (
        "shared_backlog_visibility,timed_online_arrivals,nhpp,compound_batch,overlap_pressure"
    )


def test_list_arrival_profiles_exposes_static_and_hpp_catalog() -> None:
    profiles = list_arrival_profiles()

    assert [profile["profile_id"] for profile in profiles] == [
        "arrival_static_release_v1",
        "arrival_hpp_singleton_nominal_v1",
        "arrival_nhpp_singleton_nominal_v1",
        "arrival_nhpp_random_batch_nominal_v1",
        "arrival_nhpp_random_batch_overlap_v1",
        "arrival_nhpp_random_batch_overlap_a3c9_v2",
    ]
    assert profiles[0]["scope"] == "shared_backlog_visibility"
    assert profiles[1]["scope"] == "shared_backlog_visibility,timed_online_arrivals"
    assert profiles[2]["scope"] == "shared_backlog_visibility,timed_online_arrivals,nhpp"
    assert profiles[3]["scope"] == (
        "shared_backlog_visibility,timed_online_arrivals,nhpp,compound_batch"
    )
    assert profiles[4]["scope"] == (
        "shared_backlog_visibility,timed_online_arrivals,nhpp,compound_batch,overlap_pressure"
    )
    assert profiles[5]["scope"] == (
        "shared_backlog_visibility,timed_online_arrivals,nhpp,compound_batch,overlap_pressure"
    )
