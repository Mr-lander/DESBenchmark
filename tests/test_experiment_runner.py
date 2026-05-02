"""Tests for the first benchmark experiment runner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable
import subprocess
import sys
import csv
import json

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fms_bench.experiment_runner as experiment_runner_module
from fms_bench import (
    ContractNetBudgetedPolicy,
    ContractNetGreedyPolicy,
    FlatMaskFirstPolicy,
    FlatTreePolicy,
    HierarchicalTreePolicy,
    ThroughputAwareCellPolicy,
    TreeContractNetPolicy,
)
from fms_bench.experiment_runner import (
    ExperimentCaseRunResult,
    HolonicExperimentCase,
    build_core_v1_small_suite,
    build_tree_runtime_v2_intercell_a5c12_harder_candidates_suite,
    build_tree_runtime_v2_intercell_a3c9_dynamics_budget_sweep_suite,
    build_tree_runtime_v2_intercell_a3c9_probe_suite,
    build_tree_runtime_v2_intercell_a3c9_dynamics_suite,
    build_tree_runtime_v2_intercell_a3c9_nightmare_suite,
    build_tree_runtime_v2_intercell_a3c9_wide_suite,
    build_tree_runtime_v2_intercell_blocking_probe_suite,
    build_tree_runtime_v2_intercell_brandimarte_probe_suite,
    build_tree_runtime_v2_intercell_dynamics_extension_suite,
    build_tree_runtime_v2_intercell_harder_congestion_candidates_suite,
    build_tree_runtime_v2_intercell_harder_congestion_suite,
    build_tree_runtime_v2_intercell_small_suite,
    build_tree_runtime_v2_intercell_wide_suite,
    build_tree_benchmark_suite,
    build_tree_runtime_v2_ladder_suite,
    build_tree_runtime_v2_pilot_suite,
    build_tree_runtime_v2_small_suite,
    build_tree_runtime_v2_wide_suite,
    build_shared_backlog_v1_small_suite,
    make_level_config,
    run_experiment_case,
    run_experiment_suite,
)
from fms_bench.holonic_evaluation import (
    HolonicEpisodeMetrics,
    HolonicEvaluationResults,
    ZeroFallbackViolationError,
)


def _write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _demo_case(case_id: str) -> HolonicExperimentCase:
    return HolonicExperimentCase(
        case_id=case_id,
        suite_tier="demo_suite",
        level="R0",
        benchmark_item_id=f"suite=demo_suite__case={case_id}",
        cell_configs=[],
        seeds=[13],
    )


def test_make_level_config_supports_only_currently_implemented_levels() -> None:
    l0 = make_level_config('L0')
    l2 = make_level_config('L2')
    l4 = make_level_config('L4')

    assert l0.failure.mtbf_hours == 0.0
    assert l2.failure.mtbf_hours > 0.0
    assert l4.failure.mtbf_hours == 0.0


def test_make_level_config_accepts_optional_numeric_profile_overlay() -> None:
    config = experiment_runner_module.make_level_config(
        "L0",
        numeric_profile_id="us_ca_pge_e20_camx_summer_2026q1",
    )

    assert config.numeric_profile_id == "us_ca_pge_e20_camx_summer_2026q1"
    assert config.energy.peak_window == (12, 18)
    assert config.energy.peak_price == 0.12385
    assert config.energy.offpeak_price == 0.11840
    assert config.carbon.kg_per_kwh == 0.19503700802971002
    assert config.energy.active_power_kw == 10.0


def test_make_level_config_accepts_optional_due_date_profile_overlay() -> None:
    config = experiment_runner_module.make_level_config(
        "L0",
        due_date_profile_id="twk_nominal_k3_relaxed_v1",
    )

    assert config.due_date.allowance_factor == pytest.approx(3.0)
    assert config.due_date_profile_id == "twk_nominal_k3_relaxed_v1"
    assert config.due_date_profile_rule_family == "TWK"


def test_build_core_v1_small_suite_contains_l0_l2_l4_only() -> None:
    suite = build_core_v1_small_suite(PROJECT_ROOT)

    levels = sorted({case.level for case in suite})
    assert levels == ['L0', 'L2', 'L4']
    assert len(suite) >= 3


def test_run_experiment_case_writes_summary_artifacts() -> None:
    suite = build_core_v1_small_suite(PROJECT_ROOT)
    case = next(item for item in suite if item.level == 'L0')
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner'

    result = run_experiment_case(
        case=case,
        policies=[ContractNetGreedyPolicy()],
        output_dir=out_dir,
    )

    assert result.case_id == case.case_id
    assert result.summary_csv.exists()
    assert result.rows
    assert result.rows[0]['policy_name'] == 'contract_net_greedy'


def test_run_experiment_case_writes_progress_artifacts_before_final_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = replace(_demo_case("case_progress"), level="L0")
    policy = FlatTreePolicy()

    def fake_evaluate_holonic_policy(
        *,
        policy: object,
        cell_configs: object,
        config: object,
        seeds: object,
        benchmark_item_id: str,
        env_cls: object,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        progress_decision_interval: int = 5,
        require_zero_fallback: bool = False,
    ) -> HolonicEvaluationResults:
        del (
            cell_configs,
            config,
            seeds,
            env_cls,
            progress_decision_interval,
            require_zero_fallback,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "episode_started",
                    "episode_id": 0,
                    "seed": 13,
                    "decision_steps": 0,
                    "coordination_message_count": 0,
                    "active_agent_slots": 0,
                    "framework_llm_decision_count": 0,
                    "framework_fallback_contract_decision_count": 0,
                    "framework_no_legal_action_count": 0,
                    "sim_time": 0.0,
                    "completed_jobs_total": 0,
                    "active_agent_count": 1,
                }
            )
            progress_callback(
                {
                    "event": "episode_heartbeat",
                    "episode_id": 0,
                    "seed": 13,
                    "decision_steps": 3,
                    "coordination_message_count": 3,
                    "active_agent_slots": 3,
                    "framework_llm_decision_count": 2,
                    "framework_fallback_contract_decision_count": 1,
                    "framework_no_legal_action_count": 0,
                    "sim_time": 4.5,
                    "completed_jobs_total": 1,
                    "active_agent_count": 1,
                }
            )
            progress_callback(
                {
                    "event": "episode_completed",
                    "episode_id": 0,
                    "seed": 13,
                    "decision_steps": 3,
                    "coordination_message_count": 3,
                    "active_agent_slots": 3,
                    "framework_llm_decision_count": 2,
                    "framework_fallback_contract_decision_count": 1,
                    "framework_no_legal_action_count": 0,
                    "sim_time": 4.5,
                    "completed_jobs_total": 1,
                    "active_agent_count": 0,
                    "runtime_seconds": 0.25,
                    "terminated": True,
                    "truncated": False,
                }
            )
        episode = HolonicEpisodeMetrics(
            episode_id=0,
            seed=13,
            policy_name=str(getattr(policy, "name", "policy")),
            benchmark_item_id=benchmark_item_id,
            makespan=4.5,
            total_tardiness=0.0,
            completed_jobs=1,
            throughput=0.2,
            total_energy_kwh=0.0,
            total_carbon_kg=0.0,
            budget_violation_count=0,
            budget_violation_magnitude=0.0,
            decision_steps=3,
            coordination_message_count=3,
            active_agent_slots=3,
            runtime_seconds=0.25,
            total_reward=1.0,
            terminated=True,
            truncated=False,
            deadlocked=False,
            extra={
                "suite_id": "demo_suite",
                "split_id": "",
                "controller_interface_mode": "in_process",
                "framework_name": "",
                "framework_backend_mode": "",
                "framework_python_package": "",
                "framework_optional_dependency": "",
                "framework_uv_project_environment": "",
                "framework_llm_backend": "",
                "framework_llm_model": "",
                "framework_llm_base_url": "",
                "framework_decision_audit_contract": "",
                "framework_decision_source_trace": [],
                "framework_decision_source_counts": {
                    "llm": 2,
                    "fallback_contract": 1,
                    "no_legal_action": 0,
                },
                "framework_llm_decision_count": 2,
                "framework_fallback_contract_decision_count": 1,
                "framework_no_legal_action_count": 0,
                "activation_trace": [],
                "settlement_trace": [],
                "protocol_trace": [],
                "world_event_trace": [],
                "state_trace": [],
                "reward_authority": "",
            },
        )
        return HolonicEvaluationResults(
            benchmark_item_id=benchmark_item_id,
            policy_name=str(getattr(policy, "name", "policy")),
            seeds=[13],
            episodes=[episode],
        )

    monkeypatch.setattr(
        experiment_runner_module,
        "evaluate_holonic_policy",
        fake_evaluate_holonic_policy,
    )

    result = run_experiment_case(case=case, policies=[policy], output_dir=tmp_path)

    progress_snapshot = result.summary_csv.parent / f"{policy.name}_progress.json"
    progress_log = result.summary_csv.parent / f"{policy.name}_progress.jsonl"
    assert progress_snapshot.exists()
    assert progress_log.exists()

    snapshot_payload = json.loads(progress_snapshot.read_text(encoding="utf-8"))
    assert snapshot_payload["event"] == "policy_completed"
    assert snapshot_payload["framework_llm_decision_count"] == 2

    with progress_log.open("r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    assert [event["event"] for event in events] == [
        "policy_started",
        "episode_started",
        "episode_heartbeat",
        "episode_completed",
        "policy_completed",
    ]


def test_run_experiment_case_surfaces_numeric_profile_metadata_in_summary() -> None:
    suite = build_tree_runtime_v2_small_suite(
        PROJECT_ROOT,
        numeric_profile_id="us_ca_pge_e20_camx_summer_2026q1",
    )
    case = next(item for item in suite if item.regime_id == "R0")
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_numeric_profile'

    result = run_experiment_case(
        case=case,
        policies=[FlatTreePolicy()],
        output_dir=out_dir,
    )

    row = result.rows[0]
    assert row["numeric_profile_id"] == "us_ca_pge_e20_camx_summer_2026q1"
    assert row["numeric_profile_provenance_class"] == "authority_profiled_scenario"
    assert row["numeric_profile_projection_kind"] == "official_tariff_and_grid_factor_overlay"
    assert row["numeric_profile_scope"] == "tariff,carbon"
    assert row["numeric_profile_source_ref"] == "pge_e20_effective_2026-03-01__epa_egrid2023_camx"


def test_run_experiment_case_persists_decision_failure_trace_on_policy_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = replace(_demo_case("case_policy_failed_trace"), level="L0")
    policy = FlatTreePolicy()
    policy.decision_audit_payload = {
        "framework_decision_audit_contract": "framework_decision_audit_v1",
        "decision_source_trace": [
            {
                "agent_id": "area_3",
                "source": "fallback_contract",
                "action": 1,
                "legal_actions": [0, 1, 2],
            }
        ],
        "decision_failure_trace": [
            {
                "agent_id": "area_3",
                "attempt": 0,
                "failure_kind": "exception",
                "legal_actions": [0, 1, 2],
                "exception_type": "TimeoutError",
                "exception_message": "request timed out",
            }
        ],
        "decision_source_counts": {
            "llm": 184,
            "fallback_contract": 1,
            "no_legal_action": 0,
        },
    }

    def fake_evaluate_holonic_policy(
        *,
        policy: object,
        cell_configs: object,
        config: object,
        seeds: object,
        benchmark_item_id: str,
        env_cls: object,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        progress_decision_interval: int = 5,
        require_zero_fallback: bool = False,
    ) -> HolonicEvaluationResults:
        del (
            policy,
            cell_configs,
            config,
            seeds,
            env_cls,
            progress_callback,
            progress_decision_interval,
            require_zero_fallback,
        )
        raise ZeroFallbackViolationError(
            policy_name="tree_flat_mask_first",
            benchmark_item_id=benchmark_item_id,
            episode_id=7,
            seed=347,
            fallback_count=1,
        )

    monkeypatch.setattr(
        experiment_runner_module,
        "evaluate_holonic_policy",
        fake_evaluate_holonic_policy,
    )

    with pytest.raises(ZeroFallbackViolationError):
        run_experiment_case(case=case, policies=[policy], output_dir=tmp_path)

    progress_snapshot = (
        tmp_path / case.case_id / f"{policy.name}_progress.json"
    )
    progress_log = tmp_path / case.case_id / f"{policy.name}_progress.jsonl"

    snapshot_payload = json.loads(progress_snapshot.read_text(encoding="utf-8"))
    assert snapshot_payload["event"] == "policy_failed"
    assert snapshot_payload["framework_llm_decision_count"] == 184
    assert snapshot_payload["framework_fallback_contract_decision_count"] == 1
    assert snapshot_payload["framework_decision_failure_trace"] == [
        {
            "agent_id": "area_3",
            "attempt": 0,
            "failure_kind": "exception",
            "legal_actions": [0, 1, 2],
            "exception_type": "TimeoutError",
            "exception_message": "request timed out",
        }
    ]

    with progress_log.open("r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    assert events[-1]["event"] == "policy_failed"
    assert events[-1]["framework_decision_source_counts"] == {
        "llm": 184,
        "fallback_contract": 1,
        "no_legal_action": 0,
    }


def test_run_experiment_case_surfaces_due_date_profile_metadata_in_summary() -> None:
    suite = build_tree_runtime_v2_small_suite(
        PROJECT_ROOT,
        due_date_profile_id="twk_nominal_k3_relaxed_v1",
    )
    case = next(item for item in suite if item.regime_id == "R0")
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_due_date_profile'

    result = run_experiment_case(
        case=case,
        policies=[FlatTreePolicy()],
        output_dir=out_dir,
    )

    row = result.rows[0]
    assert row["due_date_profile_id"] == "twk_nominal_k3_relaxed_v1"
    assert row["due_date_profile_provenance_class"] == "literature_supported_scenario_profile"
    assert row["due_date_profile_rule_family"] == "TWK"
    assert row["due_date_profile_source_ref"] == "gordon_proth_chu_2001_twk_family"


def test_run_experiment_case_surfaces_case_budget_provenance_metadata_in_summary() -> None:
    suite = build_tree_runtime_v2_small_suite(PROJECT_ROOT)
    case = next(item for item in suite if item.regime_id == "R0")
    out_dir = PROJECT_ROOT / "tests" / "_tmp_experiment_runner_case_budget_metadata"

    result = run_experiment_case(
        case=case,
        policies=[FlatTreePolicy()],
        output_dir=out_dir,
    )

    row = result.rows[0]
    assert row["case_budget_provenance_class"] == "benchmark_case_design"
    assert row["case_budget_scope"] == "wip,energy,carbon"
    assert (
        row["case_budget_source_ref"]
        == "experiment_runner_tree_runtime_v2_case_cell_budgets_r20260327"
    )


def test_run_experiment_suite_checkpoints_partial_suite_summary_before_late_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "suite"
    cases = [_demo_case("case_a"), _demo_case("case_b")]

    def fake_run_experiment_case(
        case: HolonicExperimentCase,
        policies: object,
        output_dir: Path,
        require_zero_fallback: bool = False,
    ) -> ExperimentCaseRunResult:
        del require_zero_fallback
        if case.case_id == "case_b":
            raise RuntimeError("synthetic late teardown failure")

        row = {
            "policy_name": "external_framework_spade",
            "case_id": case.case_id,
            "suite_tier": "demo_suite",
            "controller_interface_mode": "external_session",
            "framework_name": "SPADE",
            "framework_backend_mode": "native_framework",
            "makespan_mean": "10.0",
        }
        summary_csv = output_dir / case.case_id / "summary.csv"
        artifacts_csv = output_dir / case.case_id / "artifacts.csv"
        _write_summary_csv(summary_csv, [row])
        _write_summary_csv(
            artifacts_csv,
            [
                {
                    "case_id": case.case_id,
                    "policy_name": row["policy_name"],
                    "results_json": "",
                }
            ],
        )
        return ExperimentCaseRunResult(
            case_id=case.case_id,
            summary_csv=summary_csv,
            artifact_manifest=artifacts_csv,
            rows=[row],
        )

    monkeypatch.setattr(experiment_runner_module, "run_experiment_case", fake_run_experiment_case)

    with pytest.raises(RuntimeError, match="synthetic late teardown failure"):
        run_experiment_suite(cases=cases, policies=[object()], output_dir=output_dir)

    summary_path = output_dir / "suite_summary.csv"
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["case_id"] for row in rows] == ["case_a"]


def test_recover_experiment_suite_summary_rebuilds_missing_suite_summary_from_case_dirs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "suite"
    rows = [
        {
            "policy_name": "external_framework_agentscope",
            "case_id": "case_a",
            "suite_tier": "demo_suite",
            "controller_interface_mode": "external_session",
            "framework_name": "AgentScope",
            "framework_backend_mode": "native_framework",
            "makespan_mean": "11.0",
        },
        {
            "policy_name": "external_framework_agentscope",
            "case_id": "case_b",
            "suite_tier": "demo_suite",
            "controller_interface_mode": "external_session",
            "framework_name": "AgentScope",
            "framework_backend_mode": "native_framework",
            "makespan_mean": "13.0",
        },
    ]
    _write_summary_csv(output_dir / "case_a" / "summary.csv", [rows[0]])
    _write_summary_csv(output_dir / "case_b" / "summary.csv", [rows[1]])

    summary_path = experiment_runner_module.recover_experiment_suite_summary(
        output_dir=output_dir,
        cases=[_demo_case("case_a"), _demo_case("case_b")],
        policies=[type("Policy", (), {"name": "external_framework_agentscope"})()],
    )

    assert summary_path == output_dir / "suite_summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        recovered_rows = list(csv.DictReader(f))
    assert [row["case_id"] for row in recovered_rows] == ["case_a", "case_b"]


def test_merge_suite_summary_csvs_combines_rows_and_rejects_duplicate_case_policy_pairs(
    tmp_path: Path,
) -> None:
    summary_a = tmp_path / "agentscope.csv"
    summary_b = tmp_path / "langgraph.csv"
    duplicate = tmp_path / "duplicate.csv"
    merged = tmp_path / "merged.csv"
    row_a = {
        "policy_name": "external_framework_agentscope",
        "case_id": "case_a",
        "suite_tier": "demo_suite",
        "controller_interface_mode": "external_session",
        "framework_name": "AgentScope",
        "framework_backend_mode": "native_framework",
        "makespan_mean": "11.0",
    }
    row_b = {
        "policy_name": "external_framework_langgraph",
        "case_id": "case_a",
        "suite_tier": "demo_suite",
        "controller_interface_mode": "external_session",
        "framework_name": "LangGraph",
        "framework_backend_mode": "native_framework",
        "makespan_mean": "9.0",
    }
    _write_summary_csv(summary_a, [row_a])
    _write_summary_csv(summary_b, [row_b])
    _write_summary_csv(duplicate, [dict(row_a)])

    merged_path = experiment_runner_module.merge_suite_summary_csvs(
        summary_csv_paths=[summary_a, summary_b],
        output_csv=merged,
    )

    assert merged_path == merged
    with merged.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["policy_name"] for row in rows] == [
        "external_framework_agentscope",
        "external_framework_langgraph",
    ]

    with pytest.raises(ValueError, match="duplicate case/policy rows"):
        experiment_runner_module.merge_suite_summary_csvs(
            summary_csv_paths=[summary_a, duplicate],
            output_csv=tmp_path / "dup_merged.csv",
        )



def test_example_script_runs_as_cli_without_module_import_errors() -> None:
    script_path = PROJECT_ROOT / 'examples' / 'run_core_v1_small.py'

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                "import runpy; "
                f"runpy.run_path(r'{script_path}', run_name='not_main')"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_build_shared_backlog_v1_small_suite_contains_scaled_kacem_cases() -> None:
    suite = build_shared_backlog_v1_small_suite(PROJECT_ROOT)

    levels = sorted({case.level for case in suite})
    compositions = sorted({case.case_id.split('__')[1] for case in suite})
    assert levels == ['L0', 'L2', 'L4']
    assert compositions == ['fattahi_fastslow', 'kacem_budgettrap', 'kacem_fastslow']
    assert all('shared_backlog_v1_small' in case.case_id for case in suite)
    assert all(case.env_name == 'shared_backlog' for case in suite)


def test_run_shared_backlog_case_writes_summary_artifacts() -> None:
    suite = build_shared_backlog_v1_small_suite(PROJECT_ROOT)
    case = next(item for item in suite if item.level == 'L0')
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_shared_backlog'

    result = run_experiment_case(
        case=case,
        policies=[ThroughputAwareCellPolicy(), ContractNetGreedyPolicy()],
        output_dir=out_dir,
    )

    assert result.case_id == case.case_id
    assert result.summary_csv.exists()
    assert len(result.rows) == 2
    policy_names = {row['policy_name'] for row in result.rows}
    assert policy_names == {'throughput_aware_cell', 'contract_net_greedy'}


def test_shared_backlog_example_script_runs_as_cli_without_module_import_errors() -> None:
    script_path = PROJECT_ROOT / 'examples' / 'run_shared_backlog_v1_small.py'

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                "import runpy; "
                f"runpy.run_path(r'{script_path}', run_name='not_main')"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_budgettrap_l4_case_creates_tighter_fast_cell_budget() -> None:
    suite = build_shared_backlog_v1_small_suite(PROJECT_ROOT)
    case = next(item for item in suite if item.case_id == 'shared_backlog_v1_small__kacem_budgettrap__L4')

    fast_cell, slow_cell = case.cell_configs
    assert fast_cell.energy_cap_kwh < slow_cell.energy_cap_kwh
    assert fast_cell.processing_time_scale < slow_cell.processing_time_scale


def test_budgettrap_case_separates_budget_aware_and_throughput_baselines() -> None:
    suite = build_shared_backlog_v1_small_suite(PROJECT_ROOT)
    case = next(item for item in suite if item.case_id == 'shared_backlog_v1_small__kacem_budgettrap__L4')
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_budgettrap'

    result = run_experiment_case(
        case=case,
        policies=[ThroughputAwareCellPolicy(), ContractNetBudgetedPolicy()],
        output_dir=out_dir,
    )

    rows = {row['policy_name']: row for row in result.rows}
    assert float(rows['contract_net_budgeted']['violation_magnitude_mean']) < float(rows['throughput_aware_cell']['violation_magnitude_mean'])


def test_tree_benchmark_case_id_contains_topology_and_regime() -> None:
    case = build_tree_benchmark_suite(PROJECT_ROOT)[0]

    assert "topology=" in case.benchmark_item_id
    assert "regime=" in case.benchmark_item_id
    assert case.env_name == "tree_shared_backlog"


def test_run_tree_benchmark_case_writes_summary_artifacts() -> None:
    suite = build_tree_benchmark_suite(PROJECT_ROOT)
    case = next(item for item in suite if item.regime_id == 'R0')
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree'

    result = run_experiment_case(
        case=case,
        policies=[FlatTreePolicy()],
        output_dir=out_dir,
    )

    assert result.case_id == case.case_id
    assert result.summary_csv.exists()
    assert result.rows
    assert result.rows[0]['policy_name'] == 'tree_flat_mask_first'


def test_tree_benchmark_example_script_runs_as_cli_without_module_import_errors() -> None:
    script_path = PROJECT_ROOT / 'examples' / 'run_tree_benchmark_v1_small.py'

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                "import runpy; "
                f"runpy.run_path(r'{script_path}', run_name='not_main')"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_tree_runtime_v2_external_adapter_example_script_imports_cleanly() -> None:
    script_path = PROJECT_ROOT / 'examples' / 'run_tree_runtime_v2_external_adapter.py'

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                "import runpy; "
                f"runpy.run_path(r'{script_path}', run_name='not_main')"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_tree_runtime_v2_external_family_example_script_imports_cleanly() -> None:
    script_path = PROJECT_ROOT / 'examples' / 'run_tree_runtime_v2_external_family.py'

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                "import runpy; "
                f"runpy.run_path(r'{script_path}', run_name='not_main')"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_tree_runtime_v2_framework_native_example_script_imports_cleanly() -> None:
    script_path = PROJECT_ROOT / 'examples' / 'run_tree_runtime_v2_framework_native.py'

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                "import runpy; "
                f"runpy.run_path(r'{script_path}', run_name='not_main')"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_tree_runtime_v2_category_b_example_script_imports_cleanly() -> None:
    script_path = PROJECT_ROOT / 'examples' / 'run_tree_runtime_v2_category_b.py'

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                "import runpy; "
                f"runpy.run_path(r'{script_path}', run_name='not_main')"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_category_b_runner_defaults_to_intercell_wide_langgraph() -> None:
    import examples.run_tree_runtime_v2_category_b as script_module

    args = script_module._parse_args([])

    assert args.framework == "langgraph"
    assert args.suite == "intercell_wide"


def test_category_b_runner_accepts_intercell_blocking_probe_suite() -> None:
    import examples.run_tree_runtime_v2_category_b as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "langgraph",
            "--suite",
            "intercell_blocking_probe",
        ]
    )

    assert args.suite == "intercell_blocking_probe"


def test_category_b_runner_accepts_intercell_a3c9_probe_suite() -> None:
    import examples.run_tree_runtime_v2_category_b as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "langgraph",
            "--suite",
            "intercell_a3c9_probe",
        ]
    )

    assert args.suite == "intercell_a3c9_probe"


def test_category_b_runner_accepts_a3c9_family_suites() -> None:
    import examples.run_tree_runtime_v2_category_b as script_module

    parsed_wide = script_module._parse_args(
        [
            "--framework",
            "langgraph",
            "--suite",
            "intercell_a3c9_wide",
        ]
    )
    parsed_dynamics = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_a3c9_dynamics",
        ]
    )
    parsed_nightmare = script_module._parse_args(
        [
            "--framework",
            "langgraph",
            "--suite",
            "intercell_a3c9_nightmare",
        ]
    )

    assert parsed_wide.suite == "intercell_a3c9_wide"
    assert parsed_dynamics.suite == "intercell_a3c9_dynamics"
    assert parsed_nightmare.suite == "intercell_a3c9_nightmare"


def test_category_b_runner_forwards_into_framework_native_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import examples.run_tree_runtime_v2_category_b as script_module

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        script_module.framework_native_runner,
        "main",
        lambda argv=None: captured.setdefault("argv", list(argv or [])),
    )

    script_module.main(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_wide",
            "--seed-limit",
            "1",
            "--llm-backend",
            "openai_compat",
            "--llm-model",
            "qwen-test",
            "--llm-base-url",
            "http://127.0.0.1:30000/v1",
        ]
    )

    assert captured["argv"] == [
        "--framework",
        "agentscope",
        "--suite",
        "intercell_wide",
        "--seed-limit",
        "1",
        "--llm-backend",
        "openai_compat",
        "--llm-model",
        "qwen-test",
        "--llm-base-url",
        "http://127.0.0.1:30000/v1",
    ]


def test_framework_native_runner_threads_profile_args_into_suite_builder() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    captured: dict[str, object] = {}

    def fake_builder(
        repo_root: Path,
        *,
        numeric_profile_id: str | None = None,
        due_date_profile_id: str | None = None,
        seed_split: str = "test",
    ) -> list[object]:
        captured["repo_root"] = repo_root
        captured["numeric_profile_id"] = numeric_profile_id
        captured["due_date_profile_id"] = due_date_profile_id
        captured["seed_split"] = seed_split
        return []

    suite_builders = dict(script_module.SUITE_BUILDERS)
    suite_builders["small"] = fake_builder

    args = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "small",
            "--numeric-profile",
            "us_ca_pge_e20_camx_summer_2026q1",
            "--due-date-profile",
            "twk_nominal_k3_relaxed_v1",
        ]
    )
    script_module._build_suite_from_args(args, suite_builders=suite_builders)

    assert captured["repo_root"] == PROJECT_ROOT
    assert captured["numeric_profile_id"] == "us_ca_pge_e20_camx_summer_2026q1"
    assert captured["due_date_profile_id"] == "twk_nominal_k3_relaxed_v1"


def test_framework_native_runner_parses_llm_backend_args() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "langgraph",
            "--suite",
            "small",
            "--llm-backend",
            "openai_compat",
            "--llm-model",
            "Qwen3.5-27B-Claude-Opus-4.6-Distill-Q4_K_M.gguf",
            "--llm-base-url",
            "http://127.0.0.1:30000/v1",
            "--llm-api-key",
            "EMPTY",
            "--llm-timeout",
            "12.5",
            "--llm-retries",
            "6",
            "--llm-retry-backoff-seconds",
            "1.5",
            "--llm-retry-backoff-max-seconds",
            "60",
        ]
    )

    assert args.llm_backend == "openai_compat"
    assert args.llm_model == "Qwen3.5-27B-Claude-Opus-4.6-Distill-Q4_K_M.gguf"
    assert args.llm_base_url == "http://127.0.0.1:30000/v1"
    assert args.llm_api_key == "EMPTY"
    assert args.llm_timeout == 12.5
    assert args.llm_retries == 6
    assert args.llm_retry_backoff_seconds == 1.5
    assert args.llm_retry_backoff_max_seconds == 60.0


def test_framework_case_shard_runner_parses_llm_retry_args() -> None:
    import examples.run_tree_runtime_v2_framework_case_shard as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "langgraph",
            "--suite",
            "intercell_a5c12_harder",
            "--case-id",
            "case-a",
            "--llm-backend",
            "openai_compat",
            "--llm-model",
            "qwen-test",
            "--llm-base-url",
            "http://127.0.0.1:30000/v1",
            "--llm-retries",
            "7",
            "--llm-retry-backoff-seconds",
            "2",
            "--llm-retry-backoff-max-seconds",
            "60",
            "--output-dir",
            "unused-output",
        ]
    )

    assert args.llm_retries == 7
    assert args.llm_retry_backoff_seconds == 2.0
    assert args.llm_retry_backoff_max_seconds == 60.0


def test_framework_native_runner_parses_require_zero_fallback_flag() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_a5c12_harder",
            "--require-zero-fallback",
        ]
    )

    assert args.require_zero_fallback is True


def test_framework_native_runner_parses_strict_real_llm_flag() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_a5c12_harder",
            "--strict-real-llm",
        ]
    )

    assert args.strict_real_llm is True


def test_framework_native_runner_parses_real_llm_response_backed_flag() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_a5c12_harder",
            "--real-llm-response-backed",
        ]
    )

    assert args.real_llm_response_backed is True


def test_framework_native_runner_threads_llm_timeout_into_supported_llm_drivers() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    ag2_driver = script_module._build_llm_driver(
        "ag2",
        strict_framework=False,
        llm_model="qwen-test",
        llm_base_url="http://127.0.0.1:30000/v1",
        llm_api_key="EMPTY",
        llm_timeout_seconds=12.5,
        llm_request_retries=6,
        llm_retry_backoff_seconds=1.5,
        llm_retry_backoff_max_seconds=60.0,
    )
    agentscope_driver = script_module._build_llm_driver(
        "agentscope",
        strict_framework=False,
        llm_model="qwen-test",
        llm_base_url="http://127.0.0.1:30000/v1",
        llm_api_key="EMPTY",
        llm_timeout_seconds=12.5,
        llm_request_retries=6,
        llm_retry_backoff_seconds=1.5,
        llm_retry_backoff_max_seconds=60.0,
    )
    crewai_driver = script_module._build_llm_driver(
        "crewai",
        strict_framework=False,
        llm_model="qwen-test",
        llm_base_url="http://127.0.0.1:30000/v1",
        llm_api_key="EMPTY",
        llm_timeout_seconds=12.5,
    )
    google_adk_driver = script_module._build_llm_driver(
        "google_adk",
        strict_framework=False,
        llm_model="qwen-test",
        llm_base_url="http://127.0.0.1:30000/v1",
        llm_api_key="EMPTY",
        llm_timeout_seconds=12.5,
    )
    langgraph_driver = script_module._build_llm_driver(
        "langgraph",
        strict_framework=False,
        llm_model="qwen-test",
        llm_base_url="http://127.0.0.1:30000/v1",
        llm_api_key="EMPTY",
        llm_timeout_seconds=12.5,
        llm_request_retries=6,
        llm_retry_backoff_seconds=1.5,
        llm_retry_backoff_max_seconds=60.0,
    )
    spade_driver = script_module._build_llm_driver(
        "spade",
        strict_framework=False,
        llm_model="qwen-test",
        llm_base_url="http://127.0.0.1:30000/v1",
        llm_api_key="EMPTY",
        llm_timeout_seconds=12.5,
    )

    assert ag2_driver.request_timeout == 12.5
    assert agentscope_driver.request_timeout == 12.5
    assert agentscope_driver.request_retries == 6
    assert agentscope_driver.retry_backoff_seconds == 1.5
    assert agentscope_driver.retry_backoff_max_seconds == 60.0
    assert crewai_driver.request_timeout == 12.5
    assert google_adk_driver.request_timeout == 12.5
    assert langgraph_driver.request_timeout == 12.5
    assert langgraph_driver.request_retries == 6
    assert langgraph_driver.retry_backoff_seconds == 1.5
    assert langgraph_driver.retry_backoff_max_seconds == 60.0
    assert spade_driver.request_timeout == 12.5


def test_framework_native_runner_parses_case_and_seed_limits() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "small",
            "--case-limit",
            "2",
            "--seed-limit",
            "3",
        ]
    )

    assert args.case_limit == 2
    assert args.seed_limit == 3


def test_framework_native_runner_accepts_intercell_small_suite() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_small",
        ]
    )

    assert args.suite == "intercell_small"


def test_framework_native_runner_accepts_intercell_brandimarte_probe_suite() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_brandimarte_probe",
        ]
    )

    assert args.suite == "intercell_brandimarte_probe"


def test_framework_native_runner_accepts_intercell_blocking_probe_suite() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_blocking_probe",
        ]
    )

    assert args.suite == "intercell_blocking_probe"


def test_framework_native_runner_accepts_intercell_wide_suite() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_wide",
        ]
    )

    assert args.suite == "intercell_wide"


def test_framework_native_runner_accepts_intercell_dynamics_extension_suite() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_dynamics_extension",
        ]
    )

    assert args.suite == "intercell_dynamics_extension"


def test_framework_native_runner_accepts_intercell_a3c9_probe_suite() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_a3c9_probe",
        ]
    )

    assert args.suite == "intercell_a3c9_probe"


def test_framework_native_runner_accepts_intercell_a5c12_harder_suite() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "langgraph",
            "--suite",
            "intercell_a5c12_harder",
        ]
    )

    assert args.suite == "intercell_a5c12_harder"


def test_framework_native_runner_accepts_intercell_a5c12_harder_candidates_suite() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "langgraph",
            "--suite",
            "intercell_a5c12_harder_candidates",
        ]
    )

    assert args.suite == "intercell_a5c12_harder_candidates"


def test_framework_native_runner_accepts_a3c9_family_suites() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    parsed_wide = script_module._parse_args(
        [
            "--framework",
            "langgraph",
            "--suite",
            "intercell_a3c9_wide",
        ]
    )
    parsed_dynamics = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_a3c9_dynamics",
        ]
    )
    parsed_nightmare = script_module._parse_args(
        [
            "--framework",
            "langgraph",
            "--suite",
            "intercell_a3c9_nightmare",
        ]
    )
    parsed_budget = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_a3c9_dynamics_budget_sweep",
        ]
    )

    assert parsed_wide.suite == "intercell_a3c9_wide"
    assert parsed_dynamics.suite == "intercell_a3c9_dynamics"
    assert parsed_nightmare.suite == "intercell_a3c9_nightmare"
    assert parsed_budget.suite == "intercell_a3c9_dynamics_budget_sweep"


def test_framework_native_runner_help_only_advertises_official_a3c9_public_surfaces() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    parser = script_module._parse_args

    try:
        parser(["--help"])
    except SystemExit:
        pass

    help_text = script_module._PUBLIC_SUITE_HELP
    assert "intercell_a3c9_wide" in help_text
    assert "intercell_a3c9_nightmare" in help_text
    assert "intercell_a3c9_dynamics" not in help_text
    assert "intercell_a3c9_dynamics_budget_sweep" not in help_text


def test_framework_native_runner_treats_a3c9_dynamics_as_deprecated_alias(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    captured: dict[str, object] = {}

    def fake_builder(
        repo_root: Path,
        *,
        numeric_profile_id: str | None = None,
        due_date_profile_id: str | None = None,
        authority_modes: tuple[str, ...] | None = None,
        seed_split: str = "test",
    ) -> list[object]:
        captured["repo_root"] = repo_root
        captured["numeric_profile_id"] = numeric_profile_id
        captured["due_date_profile_id"] = due_date_profile_id
        captured["authority_modes"] = authority_modes
        captured["seed_split"] = seed_split
        return []

    suite_builders = dict(script_module.SUITE_BUILDERS)
    suite_builders["intercell_a3c9_wide"] = fake_builder

    args = script_module._parse_args(
        [
            "--framework",
            "langgraph",
            "--suite",
            "intercell_a3c9_dynamics",
        ]
    )
    script_module._build_suite_from_args(args, suite_builders=suite_builders)

    stderr = capsys.readouterr().err
    assert "DEPRECATED" in stderr
    assert "intercell_a3c9_dynamics" in stderr
    assert "use intercell_a3c9_wide" in stderr
    assert captured["repo_root"] == PROJECT_ROOT


def test_framework_native_runner_accepts_intercell_congestion_args() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_small",
            "--intercell-max-in-flight",
            "2",
            "--intercell-inbound-buffer-cap",
            "1",
        ]
    )

    assert args.intercell_max_in_flight == 2
    assert args.intercell_inbound_buffer_cap == 1


def test_framework_native_runner_limits_suite_cases_and_seeds() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    cases = [
        type("Case", (), {"case_id": "case_a", "seeds": [1, 2, 3]})(),
        type("Case", (), {"case_id": "case_b", "seeds": [4, 5, 6]})(),
        type("Case", (), {"case_id": "case_c", "seeds": [7, 8, 9]})(),
    ]

    limited = script_module._limit_suite(
        cases,
        case_limit=2,
        seed_limit=1,
    )

    assert [case.case_id for case in limited] == ["case_a", "case_b"]
    assert [list(case.seeds) for case in limited] == [[1], [4]]


def test_framework_native_runner_rejects_existing_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    output_dir = tmp_path / "occupied_framework_run"
    output_dir.mkdir()
    (output_dir / "placeholder.txt").write_text("occupied", encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "framework": "agentscope",
            "suite": "small",
            "strict_framework": False,
            "output_dir": output_dir,
            "case_limit": None,
            "seed_limit": None,
            "llm_backend": "none",
            "llm_model": None,
            "llm_base_url": None,
            "llm_api_key": None,
            "numeric_profile": None,
            "due_date_profile": None,
        },
    )()

    monkeypatch.setattr(script_module, "_parse_args", lambda argv=None: args)
    monkeypatch.setattr(
        script_module,
        "get_framework_env_spec",
        lambda framework: type(
            "Spec",
            (),
            {
                "framework_key": str(framework),
                "framework_name": "AgentScope",
                "uv_project_environment": ".uv_envs/agentscope",
            },
        )(),
    )
    monkeypatch.setattr(script_module, "_build_suite_from_args", lambda parsed_args: [])
    monkeypatch.setattr(
        script_module,
        "_build_driver",
        lambda framework_key, *, strict_framework: object(),
    )
    monkeypatch.setattr(
        script_module,
        "run_experiment_suite",
        lambda **kwargs: pytest.fail("run_experiment_suite should not be called for an occupied output dir"),
    )

    with pytest.raises(SystemExit, match="already exists"):
        script_module.main()


def test_framework_native_runner_allows_precreated_empty_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    output_dir = tmp_path / "empty_framework_run"
    output_dir.mkdir()
    args = type(
        "Args",
        (),
        {
            "framework": "agentscope",
            "suite": "small",
            "strict_framework": False,
            "output_dir": output_dir,
            "case_limit": None,
            "seed_limit": None,
            "llm_backend": "none",
            "llm_model": None,
            "llm_base_url": None,
            "llm_api_key": None,
            "llm_timeout": 30.0,
            "require_zero_fallback": True,
            "strict_real_llm": False,
            "numeric_profile": None,
            "due_date_profile": None,
        },
    )()

    called: dict[str, object] = {}
    monkeypatch.setattr(script_module, "_parse_args", lambda argv=None: args)
    monkeypatch.setattr(
        script_module,
        "get_framework_env_spec",
        lambda framework: type(
            "Spec",
            (),
            {
                "framework_key": str(framework),
                "framework_name": "AgentScope",
                "uv_project_environment": ".uv_envs/agentscope",
            },
        )(),
    )
    monkeypatch.setattr(script_module, "_build_suite_from_args", lambda parsed_args: [])
    monkeypatch.setattr(
        script_module,
        "_build_driver",
        lambda framework_key, *, strict_framework: object(),
    )
    def _fake_run_experiment_suite(**kwargs: object) -> Path:
        called["output_dir"] = kwargs["output_dir"]
        called["require_zero_fallback"] = kwargs["require_zero_fallback"]
        return output_dir / "suite_summary.csv"

    monkeypatch.setattr(
        script_module,
        "run_experiment_suite",
        _fake_run_experiment_suite,
    )

    script_module.main()

    assert called["output_dir"] == output_dir
    assert called["require_zero_fallback"] is True


def test_framework_native_runner_strict_real_llm_requires_llm_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = type(
        "Args",
        (),
        {
            "framework": "agentscope",
            "suite": "small",
            "strict_framework": False,
            "output_dir": tmp_path / "strict_real_llm_missing_backend",
            "case_limit": None,
            "seed_limit": None,
            "llm_backend": "none",
            "llm_model": None,
            "llm_base_url": None,
            "llm_api_key": None,
            "llm_timeout": 30.0,
            "require_zero_fallback": False,
            "strict_real_llm": True,
            "numeric_profile": None,
            "due_date_profile": None,
        },
    )()

    monkeypatch.setattr(script_module, "_parse_args", lambda argv=None: args)
    monkeypatch.setattr(
        script_module,
        "get_framework_env_spec",
        lambda framework: type("Spec", (), {"framework_key": str(framework)})(),
    )
    monkeypatch.setattr(script_module, "_build_suite_from_args", lambda parsed_args: [])

    with pytest.raises(SystemExit, match="require --llm-backend=openai_compat"):
        script_module.main()


def test_framework_native_runner_response_backed_real_llm_requires_llm_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    args = type(
        "Args",
        (),
        {
            "framework": "agentscope",
            "suite": "small",
            "strict_framework": False,
            "output_dir": tmp_path / "response_backed_missing_backend",
            "case_limit": None,
            "seed_limit": None,
            "llm_backend": "none",
            "llm_model": None,
            "llm_base_url": None,
            "llm_api_key": None,
            "llm_timeout": 30.0,
            "require_zero_fallback": False,
            "strict_real_llm": False,
            "real_llm_response_backed": True,
            "numeric_profile": None,
            "due_date_profile": None,
        },
    )()

    monkeypatch.setattr(script_module, "_parse_args", lambda argv=None: args)
    monkeypatch.setattr(
        script_module,
        "get_framework_env_spec",
        lambda framework: type("Spec", (), {"framework_key": str(framework)})(),
    )
    monkeypatch.setattr(script_module, "_build_suite_from_args", lambda parsed_args: [])

    with pytest.raises(SystemExit, match="require --llm-backend=openai_compat"):
        script_module.main()


def test_framework_native_runner_strict_real_llm_validates_summary_after_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    output_dir = tmp_path / "strict_real_llm_run"
    summary_path = output_dir / "suite_summary.csv"
    args = type(
        "Args",
        (),
        {
            "framework": "agentscope",
            "suite": "intercell_a5c12_harder",
            "strict_framework": False,
            "output_dir": output_dir,
            "case_limit": None,
            "seed_limit": None,
            "llm_backend": "openai_compat",
            "llm_model": "qwen-test",
            "llm_base_url": "http://127.0.0.1:30000/v1",
            "llm_api_key": "EMPTY",
            "llm_timeout": 30.0,
            "require_zero_fallback": False,
            "strict_real_llm": True,
            "numeric_profile": None,
            "due_date_profile": None,
        },
    )()

    called: dict[str, object] = {}
    monkeypatch.setattr(script_module, "_parse_args", lambda argv=None: args)
    monkeypatch.setattr(
        script_module,
        "get_framework_env_spec",
        lambda framework: type(
            "Spec",
            (),
            {
                "framework_key": str(framework),
                "framework_name": "AgentScope",
                "uv_project_environment": ".uv_envs/agentscope",
            },
        )(),
    )
    monkeypatch.setattr(script_module, "_build_suite_from_args", lambda parsed_args: [])

    def _fake_build_llm_driver(framework_key: str, **kwargs: object) -> object:
        called["build_llm_driver"] = {
            "framework_key": framework_key,
            "strict_framework": kwargs.get("strict_framework"),
        }
        return type("Policy", (), {"name": "external_framework_agentscope_llm"})()

    monkeypatch.setattr(script_module, "_build_llm_driver", _fake_build_llm_driver)

    def _fake_run_experiment_suite(**kwargs: object) -> Path:
        called["require_zero_fallback"] = kwargs["require_zero_fallback"]
        return summary_path

    monkeypatch.setattr(script_module, "run_experiment_suite", _fake_run_experiment_suite)
    monkeypatch.setattr(
        script_module,
        "_preflight_openai_compat_endpoint",
        lambda **kwargs: called.setdefault("preflight", kwargs),
    )

    def _fake_validate(*, summary_csv: Path, framework: str, suite: str, uv_project_environment: str) -> None:
        called["validated"] = {
            "summary_csv": summary_csv,
            "framework": framework,
            "suite": suite,
            "uv_project_environment": uv_project_environment,
        }

    monkeypatch.setattr(script_module, "_validate_strict_real_llm_summary", _fake_validate)

    script_module.main()

    assert called["require_zero_fallback"] is True
    assert called["build_llm_driver"] == {
        "framework_key": "agentscope",
        "strict_framework": True,
    }
    assert called["preflight"] == {
        "base_url": "http://127.0.0.1:30000/v1",
        "api_key": "EMPTY",
        "timeout_seconds": 30.0,
    }
    assert called["validated"] == {
        "summary_csv": summary_path,
        "framework": "AgentScope",
        "suite": "intercell_a5c12_harder",
        "uv_project_environment": ".uv_envs/agentscope",
    }


def test_framework_native_runner_response_backed_validates_summary_after_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    output_dir = tmp_path / "response_backed_run"
    summary_path = output_dir / "suite_summary.csv"
    args = type(
        "Args",
        (),
        {
            "framework": "langgraph",
            "suite": "intercell_a3c9_wide",
            "strict_framework": False,
            "output_dir": output_dir,
            "case_limit": None,
            "seed_limit": None,
            "llm_backend": "openai_compat",
            "llm_model": "qwen-test",
            "llm_base_url": "http://127.0.0.1:30063/v1",
            "llm_api_key": "EMPTY",
            "llm_timeout": 30.0,
            "require_zero_fallback": False,
            "strict_real_llm": False,
            "real_llm_response_backed": True,
            "numeric_profile": None,
            "due_date_profile": None,
        },
    )()

    called: dict[str, object] = {}
    monkeypatch.setattr(script_module, "_parse_args", lambda argv=None: args)
    monkeypatch.setattr(
        script_module,
        "get_framework_env_spec",
        lambda framework: type(
            "Spec",
            (),
            {
                "framework_key": str(framework),
                "framework_name": "LangGraph",
                "uv_project_environment": ".uv_envs/langgraph",
            },
        )(),
    )
    monkeypatch.setattr(script_module, "_build_suite_from_args", lambda parsed_args: [])

    def _fake_build_llm_driver(framework_key: str, **kwargs: object) -> object:
        called["build_llm_driver"] = {
            "framework_key": framework_key,
            "strict_framework": kwargs.get("strict_framework"),
        }
        return type("Policy", (), {"name": "external_framework_langgraph_llm"})()

    monkeypatch.setattr(script_module, "_build_llm_driver", _fake_build_llm_driver)

    def _fake_run_experiment_suite(**kwargs: object) -> Path:
        called["require_zero_fallback"] = kwargs["require_zero_fallback"]
        return summary_path

    monkeypatch.setattr(script_module, "run_experiment_suite", _fake_run_experiment_suite)
    monkeypatch.setattr(
        script_module,
        "_preflight_openai_compat_endpoint",
        lambda **kwargs: called.setdefault("preflight", kwargs),
    )
    monkeypatch.setattr(
        script_module,
        "_validate_strict_real_llm_summary",
        lambda **kwargs: called.setdefault("strict_validated", True),
    )

    def _fake_validate_response_backed(
        *,
        summary_csv: Path,
        framework: str,
        suite: str,
        uv_project_environment: str,
    ) -> None:
        called["validated"] = {
            "summary_csv": summary_csv,
            "framework": framework,
            "suite": suite,
            "uv_project_environment": uv_project_environment,
        }

    monkeypatch.setattr(
        script_module,
        "_validate_response_backed_real_llm_summary",
        _fake_validate_response_backed,
    )

    script_module.main()

    assert called["require_zero_fallback"] is False
    assert called["build_llm_driver"] == {
        "framework_key": "langgraph",
        "strict_framework": True,
    }
    assert called["preflight"] == {
        "base_url": "http://127.0.0.1:30063/v1",
        "api_key": "EMPTY",
        "timeout_seconds": 30.0,
    }
    assert "strict_validated" not in called
    assert called["validated"] == {
        "summary_csv": summary_path,
        "framework": "LangGraph",
        "suite": "intercell_a3c9_wide",
        "uv_project_environment": ".uv_envs/langgraph",
    }


def test_framework_native_runner_strict_real_llm_preflights_endpoint_before_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    output_dir = tmp_path / "strict_real_llm_preflight_run"
    summary_path = output_dir / "suite_summary.csv"
    args = type(
        "Args",
        (),
        {
            "framework": "langgraph",
            "suite": "intercell_a5c12_harder",
            "strict_framework": False,
            "output_dir": output_dir,
            "case_limit": None,
            "seed_limit": None,
            "llm_backend": "openai_compat",
            "llm_model": "qwen-test",
            "llm_base_url": "http://127.0.0.1:30062/v1",
            "llm_api_key": "EMPTY",
            "llm_timeout": 12.5,
            "require_zero_fallback": False,
            "strict_real_llm": True,
            "numeric_profile": None,
            "due_date_profile": None,
        },
    )()

    called: dict[str, object] = {}
    monkeypatch.setattr(script_module, "_parse_args", lambda argv=None: args)
    monkeypatch.setattr(
        script_module,
        "get_framework_env_spec",
        lambda framework: type(
            "Spec",
            (),
            {
                "framework_key": str(framework),
                "framework_name": "LangGraph",
                "uv_project_environment": ".uv_envs/langgraph",
            },
        )(),
    )
    monkeypatch.setattr(script_module, "_build_suite_from_args", lambda parsed_args: [])
    monkeypatch.setattr(
        script_module,
        "_build_llm_driver",
        lambda framework_key, **kwargs: type(
            "Policy",
            (),
            {"name": "external_framework_langgraph_llm"},
        )(),
    )

    def _fake_preflight(*, base_url: str, api_key: str, timeout_seconds: float | None) -> None:
        called["preflight"] = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(script_module, "_preflight_openai_compat_endpoint", _fake_preflight)
    monkeypatch.setattr(
        script_module,
        "run_experiment_suite",
        lambda **kwargs: summary_path,
    )
    monkeypatch.setattr(
        script_module,
        "_validate_strict_real_llm_summary",
        lambda **kwargs: called.setdefault("validated", True),
    )

    script_module.main()

    assert called["preflight"] == {
        "base_url": "http://127.0.0.1:30062/v1",
        "api_key": "EMPTY",
        "timeout_seconds": 12.5,
    }
    assert called["validated"] is True


def test_framework_native_runner_strict_real_llm_aborts_when_preflight_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    output_dir = tmp_path / "strict_real_llm_preflight_fail"
    args = type(
        "Args",
        (),
        {
            "framework": "agentscope",
            "suite": "intercell_a5c12_harder",
            "strict_framework": False,
            "output_dir": output_dir,
            "case_limit": None,
            "seed_limit": None,
            "llm_backend": "openai_compat",
            "llm_model": "qwen-test",
            "llm_base_url": "http://127.0.0.1:39999/v1",
            "llm_api_key": "EMPTY",
            "llm_timeout": 5.0,
            "require_zero_fallback": False,
            "strict_real_llm": True,
            "numeric_profile": None,
            "due_date_profile": None,
        },
    )()

    monkeypatch.setattr(script_module, "_parse_args", lambda argv=None: args)
    monkeypatch.setattr(
        script_module,
        "get_framework_env_spec",
        lambda framework: type("Spec", (), {"framework_key": str(framework)})(),
    )
    monkeypatch.setattr(script_module, "_build_suite_from_args", lambda parsed_args: [])
    monkeypatch.setattr(
        script_module,
        "_build_llm_driver",
        lambda framework_key, **kwargs: type(
            "Policy",
            (),
            {"name": "external_framework_agentscope_llm"},
        )(),
    )
    monkeypatch.setattr(
        script_module,
        "_preflight_openai_compat_endpoint",
        lambda **kwargs: (_ for _ in ()).throw(SystemExit("preflight failed")),
    )

    def _unexpected_run(**kwargs: object) -> Path:
        raise AssertionError("run_experiment_suite should not be called when preflight fails")

    monkeypatch.setattr(script_module, "run_experiment_suite", _unexpected_run)

    with pytest.raises(SystemExit, match="preflight failed"):
        script_module.main()


def test_preflight_openai_compat_endpoint_accepts_standard_data_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    import examples.run_tree_runtime_v2_framework_native as script_module

    class _FakeResponse:
        status = 200

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def getcode(self) -> int:
            return 200

        def read(self) -> bytes:
            return json.dumps(
                {
                    "object": "list",
                    "data": [
                        {"id": "deepseek-chat", "object": "model"},
                    ],
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        script_module.urllib_request,
        "urlopen",
        lambda request, timeout: _FakeResponse(),
    )

    script_module._preflight_openai_compat_endpoint(
        base_url="https://api.deepseek.com/v1",
        api_key="EMPTY",
        timeout_seconds=12.5,
    )


def test_framework_native_runner_threads_intercell_args_into_intercell_builder() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    captured: dict[str, object] = {}

    def fake_builder(
        repo_root: Path,
        *,
        numeric_profile_id: str | None = None,
        due_date_profile_id: str | None = None,
        seed_split: str = "test",
        max_in_flight_jobs: int = 1,
        inbound_buffer_cap: int = 1,
    ) -> list[object]:
        captured["repo_root"] = repo_root
        captured["numeric_profile_id"] = numeric_profile_id
        captured["due_date_profile_id"] = due_date_profile_id
        captured["seed_split"] = seed_split
        captured["max_in_flight_jobs"] = max_in_flight_jobs
        captured["inbound_buffer_cap"] = inbound_buffer_cap
        return []

    suite_builders = dict(script_module.SUITE_BUILDERS)
    suite_builders["intercell_small"] = fake_builder

    args = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_small",
            "--intercell-max-in-flight",
            "3",
            "--intercell-inbound-buffer-cap",
            "2",
        ]
    )
    script_module._build_suite_from_args(args, suite_builders=suite_builders)

    assert captured["repo_root"] == PROJECT_ROOT
    assert captured["max_in_flight_jobs"] == 3
    assert captured["inbound_buffer_cap"] == 2


def test_framework_native_runner_threads_authority_modes_into_a3c9_builder() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    captured: dict[str, object] = {}

    def fake_builder(
        repo_root: Path,
        *,
        numeric_profile_id: str | None = None,
        due_date_profile_id: str | None = None,
        seed_split: str = "test",
        authority_modes: tuple[str, ...] | None = None,
    ) -> list[object]:
        captured["repo_root"] = repo_root
        captured["numeric_profile_id"] = numeric_profile_id
        captured["due_date_profile_id"] = due_date_profile_id
        captured["seed_split"] = seed_split
        captured["authority_modes"] = authority_modes
        return []

    suite_builders = dict(script_module.SUITE_BUILDERS)
    suite_builders["intercell_a3c9_wide"] = fake_builder

    args = script_module._parse_args(
        [
            "--framework",
            "agentscope",
            "--suite",
            "intercell_a3c9_wide",
            "--authority-mode",
            "centralized",
            "--authority-mode",
            "holonic_hybrid",
        ]
    )
    script_module._build_suite_from_args(args, suite_builders=suite_builders)

    assert captured["repo_root"] == PROJECT_ROOT
    assert captured["authority_modes"] == ("centralized", "holonic_hybrid")


def test_framework_native_runner_threads_authority_modes_into_a5c12_harder_builder() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    captured: dict[str, object] = {}

    def fake_builder(
        repo_root: Path,
        *,
        numeric_profile_id: str | None = None,
        due_date_profile_id: str | None = None,
        seed_split: str = "test",
        authority_modes: tuple[str, ...] | None = None,
    ) -> list[object]:
        captured["repo_root"] = repo_root
        captured["numeric_profile_id"] = numeric_profile_id
        captured["due_date_profile_id"] = due_date_profile_id
        captured["seed_split"] = seed_split
        captured["authority_modes"] = authority_modes
        return []

    suite_builders = dict(script_module.SUITE_BUILDERS)
    suite_builders["intercell_a5c12_harder"] = fake_builder

    args = script_module._parse_args(
        [
            "--framework",
            "langgraph",
            "--suite",
            "intercell_a5c12_harder",
            "--authority-mode",
            "centralized",
            "--authority-mode",
            "heterarchical_cnp",
        ]
    )
    script_module._build_suite_from_args(args, suite_builders=suite_builders)

    assert captured["repo_root"] == PROJECT_ROOT
    assert captured["authority_modes"] == ("centralized", "heterarchical_cnp")


def test_framework_native_runner_threads_authority_modes_into_a5c12_harder_candidate_builder() -> None:
    import examples.run_tree_runtime_v2_framework_native as script_module

    captured: dict[str, object] = {}

    def fake_builder(
        repo_root: Path,
        *,
        numeric_profile_id: str | None = None,
        due_date_profile_id: str | None = None,
        seed_split: str = "test",
        authority_modes: tuple[str, ...] | None = None,
    ) -> list[object]:
        captured["repo_root"] = repo_root
        captured["numeric_profile_id"] = numeric_profile_id
        captured["due_date_profile_id"] = due_date_profile_id
        captured["seed_split"] = seed_split
        captured["authority_modes"] = authority_modes
        return []

    suite_builders = dict(script_module.SUITE_BUILDERS)
    suite_builders["intercell_a5c12_harder_candidates"] = fake_builder

    args = script_module._parse_args(
        [
            "--framework",
            "langgraph",
            "--suite",
            "intercell_a5c12_harder_candidates",
            "--authority-mode",
            "hierarchical",
            "--authority-mode",
            "holonic_hybrid",
        ]
    )
    script_module._build_suite_from_args(args, suite_builders=suite_builders)

    assert captured["repo_root"] == PROJECT_ROOT
    assert captured["authority_modes"] == ("hierarchical", "holonic_hybrid")


def test_subprocess_mas_runner_threads_profile_args_into_suite_builder() -> None:
    import examples.run_tree_runtime_v2_subprocess_mas as script_module

    captured: dict[str, object] = {}

    def fake_builder(
        repo_root: Path,
        *,
        numeric_profile_id: str | None = None,
        due_date_profile_id: str | None = None,
        seed_split: str = "test",
    ) -> list[object]:
        captured["repo_root"] = repo_root
        captured["numeric_profile_id"] = numeric_profile_id
        captured["due_date_profile_id"] = due_date_profile_id
        captured["seed_split"] = seed_split
        return []

    suite_builders = dict(script_module.SUITE_BUILDERS)
    suite_builders["small"] = fake_builder

    args = script_module._parse_args(
        [
            "--suite",
            "small",
            "--numeric-profile",
            "us_ca_pge_e20_camx_summer_2026q1",
            "--due-date-profile",
            "twk_nominal_k3_relaxed_v1",
            "--controller-command",
            "python",
            "controller.py",
        ]
    )
    script_module._build_suite_from_args(args, suite_builders=suite_builders)

    assert captured["repo_root"] == PROJECT_ROOT
    assert captured["numeric_profile_id"] == "us_ca_pge_e20_camx_summer_2026q1"
    assert captured["due_date_profile_id"] == "twk_nominal_k3_relaxed_v1"


def test_tree_runtime_v2_behavior_cloned_family_example_script_imports_cleanly() -> None:
    script_path = PROJECT_ROOT / 'examples' / 'run_tree_runtime_v2_behavior_cloned_family.py'

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                "import runpy; "
                f"runpy.run_path(r'{script_path}', run_name='not_main')"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_run_tree_runtime_v2_case_writes_artifact_manifest_and_trace_metrics() -> None:
    from fms_bench import PlantCellConfig
    from fms_bench.v2.tree_runtime_env import TreeRuntimeEnv

    flex_root = PROJECT_ROOT / 'external' / 'benchmarks' / 'flexible jobshop' / 'instances'
    common = {
        'family_key': 'kacem',
        'instance_path': flex_root / 'KacemHammadiBorne2002' / 'kacem1.txt',
        'instance_kind': 'kacem',
        'wip_cap': 2,
    }
    case = HolonicExperimentCase(
        case_id='tree_runtime_v2_small__kacem_pair__R0',
        suite_tier='tree_runtime_v2_small',
        level='R0',
        benchmark_item_id='suite=tree_runtime_v2_small__regime=R0__topology=tree_a2_c2__cells=kacem_pair__split=test',
        cell_configs=[
            PlantCellConfig(cell_id=0, label='cell-a', processing_time_scale=0.9, **common),
            PlantCellConfig(cell_id=1, label='cell-b', processing_time_scale=1.1, **common),
        ],
        seeds=[13],
        env_name='tree_runtime_v2',
        env_cls=lambda cell_configs, config, seed: TreeRuntimeEnv(
            cell_configs,
            area_to_cells={0: [0], 1: [1]},
            config=config,
            seed=seed,
        ),
        regime_id='R0',
        topology_id='tree_a2_c2',
    )
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2'

    result = run_experiment_case(
        case=case,
        policies=[FlatTreePolicy()],
        output_dir=out_dir,
    )

    assert result.summary_csv.exists()
    assert result.artifact_manifest.exists()
    row = result.rows[0]
    assert row['policy_name'] == 'tree_flat_mask_first'
    assert float(row['activation_record_count_mean']) > 0.0
    assert float(row['settlement_event_count_mean']) > 0.0
    assert float(row['settled_contract_count_mean']) > 0.0
    assert float(row['open_contract_count_mean']) == 0.0
    assert float(row['completed_unsettled_contract_count_mean']) == 0.0

    with result.artifact_manifest.open('r', encoding='utf-8', newline='') as f:
        manifest_rows = list(csv.DictReader(f))
    assert len(manifest_rows) == 1
    assert Path(manifest_rows[0]['results_json']).exists()
    assert Path(manifest_rows[0]['episode_csv']).exists()
    assert Path(manifest_rows[0]['activation_trace_json']).exists()
    assert manifest_rows[0]['case_budget_provenance_class'] == 'benchmark_case_design'
    assert manifest_rows[0]['case_budget_scope'] == 'wip,energy,carbon'
    budget_snapshot_json = Path(manifest_rows[0]['case_budget_snapshot_json'])
    assert budget_snapshot_json.exists()
    budget_snapshot = __import__('json').loads(budget_snapshot_json.read_text(encoding='utf-8'))
    assert budget_snapshot['case_budget_provenance_class'] == 'benchmark_case_design'
    assert budget_snapshot['case_budget_scope'] == 'wip,energy,carbon'
    assert budget_snapshot['case_budget_source_ref'] == 'experiment_runner_tree_runtime_v2_case_cell_budgets_r20260327'
    assert len(budget_snapshot['cells']) == 2
    assert budget_snapshot['cells'][0]['label'] == 'cell-a'
    assert 'wip_cap' in budget_snapshot['cells'][0]
    assert 'energy_cap_kwh' in budget_snapshot['cells'][0]
    assert 'carbon_cap_kg' in budget_snapshot['cells'][0]
    assert 'processing_time_scale' in budget_snapshot['cells'][0]
    payload = __import__('json').loads(Path(manifest_rows[0]['results_json']).read_text(encoding='utf-8'))
    assert payload['episodes'][0]['state_trace']
    assert 'cell_summaries' in payload['episodes'][0]['state_trace'][0]


def test_run_tree_runtime_v2_case_accepts_external_session_driver() -> None:
    from fms_bench import PlantCellConfig
    from fms_bench.v2.external_controller_adapter import InProcessPolicyExternalDriver
    from fms_bench.v2.tree_runtime_env import TreeRuntimeEnv

    flex_root = PROJECT_ROOT / 'external' / 'benchmarks' / 'flexible jobshop' / 'instances'
    common = {
        'family_key': 'kacem',
        'instance_path': flex_root / 'KacemHammadiBorne2002' / 'kacem1.txt',
        'instance_kind': 'kacem',
        'wip_cap': 2,
    }
    case = HolonicExperimentCase(
        case_id='tree_runtime_v2_small__kacem_pair_external__R0',
        suite_tier='tree_runtime_v2_small',
        level='R0',
        benchmark_item_id='suite=tree_runtime_v2_small__regime=R0__topology=tree_a2_c2__cells=kacem_pair__split=test',
        cell_configs=[
            PlantCellConfig(cell_id=0, label='cell-a', processing_time_scale=0.9, **common),
            PlantCellConfig(cell_id=1, label='cell-b', processing_time_scale=1.1, **common),
        ],
        seeds=[13],
        env_name='tree_runtime_v2',
        env_cls=lambda cell_configs, config, seed: TreeRuntimeEnv(
            cell_configs,
            area_to_cells={0: [0], 1: [1]},
            config=config,
            seed=seed,
        ),
        regime_id='R0',
        topology_id='tree_a2_c2',
    )
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2_external'

    result = run_experiment_case(
        case=case,
        policies=[
            InProcessPolicyExternalDriver(
                FlatTreePolicy(),
                controller_name='external_tree_flat_mask_first',
            )
        ],
        output_dir=out_dir,
    )

    assert result.summary_csv.exists()
    assert result.artifact_manifest.exists()
    row = result.rows[0]
    assert row['policy_name'] == 'external_tree_flat_mask_first'
    assert row['controller_interface_mode'] == 'external_session'
    assert float(row['activation_record_count_mean']) > 0.0
    assert float(row['settlement_event_count_mean']) > 0.0
    with result.artifact_manifest.open('r', encoding='utf-8', newline='') as f:
        manifest_rows = list(csv.DictReader(f))
    assert len(manifest_rows) == 1
    assert manifest_rows[0]['controller_interface_mode'] == 'external_session'
    assert Path(manifest_rows[0]['settlement_trace_json']).exists()


def test_run_tree_runtime_v2_case_accepts_framework_backed_driver() -> None:
    from fms_bench import PlantCellConfig
    from fms_bench.v2.framework_drivers import AgentScopeTreeDriver
    from fms_bench.v2.tree_runtime_env import TreeRuntimeEnv

    flex_root = PROJECT_ROOT / 'external' / 'benchmarks' / 'flexible jobshop' / 'instances'
    common = {
        'family_key': 'kacem',
        'instance_path': flex_root / 'KacemHammadiBorne2002' / 'kacem1.txt',
        'instance_kind': 'kacem',
        'wip_cap': 2,
    }
    case = HolonicExperimentCase(
        case_id='tree_runtime_v2_small__kacem_pair_framework__R0',
        suite_tier='tree_runtime_v2_small',
        level='R0',
        benchmark_item_id='suite=tree_runtime_v2_small__regime=R0__topology=tree_a2_c2__cells=kacem_pair__split=test',
        cell_configs=[
            PlantCellConfig(cell_id=0, label='cell-a', processing_time_scale=0.9, **common),
            PlantCellConfig(cell_id=1, label='cell-b', processing_time_scale=1.1, **common),
        ],
        seeds=[13],
        env_name='tree_runtime_v2',
        env_cls=lambda cell_configs, config, seed: TreeRuntimeEnv(
            cell_configs,
            area_to_cells={0: [0], 1: [1]},
            config=config,
            seed=seed,
        ),
        regime_id='R0',
        topology_id='tree_a2_c2',
    )
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2_framework'

    result = run_experiment_case(
        case=case,
        policies=[AgentScopeTreeDriver()],
        output_dir=out_dir,
    )

    assert result.summary_csv.exists()
    assert result.artifact_manifest.exists()
    row = result.rows[0]
    assert row['policy_name'] == 'external_framework_agentscope'
    assert row['controller_interface_mode'] == 'external_session'
    assert float(row['activation_record_count_mean']) > 0.0
    assert float(row['settlement_event_count_mean']) > 0.0


def test_run_tree_runtime_v2_case_records_framework_backend_metadata_for_langgraph() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("langgraph")
    from fms_bench import PlantCellConfig
    from fms_bench.v2.framework_drivers import LangGraphTreeDriver
    from fms_bench.v2.tree_runtime_env import TreeRuntimeEnv

    flex_root = PROJECT_ROOT / 'external' / 'benchmarks' / 'flexible jobshop' / 'instances'
    common = {
        'family_key': 'kacem',
        'instance_path': flex_root / 'KacemHammadiBorne2002' / 'kacem1.txt',
        'instance_kind': 'kacem',
        'wip_cap': 2,
    }
    case = HolonicExperimentCase(
        case_id='tree_runtime_v2_small__kacem_pair_langgraph_native__R0',
        suite_tier='tree_runtime_v2_small',
        level='R0',
        benchmark_item_id='suite=tree_runtime_v2_small__regime=R0__topology=tree_a2_c2__cells=kacem_pair__split=test',
        cell_configs=[
            PlantCellConfig(cell_id=0, label='cell-a', processing_time_scale=0.9, **common),
            PlantCellConfig(cell_id=1, label='cell-b', processing_time_scale=1.1, **common),
        ],
        seeds=[13],
        env_name='tree_runtime_v2',
        env_cls=lambda cell_configs, config, seed: TreeRuntimeEnv(
            cell_configs,
            area_to_cells={0: [0], 1: [1]},
            config=config,
            seed=seed,
        ),
        regime_id='R0',
        topology_id='tree_a2_c2',
    )
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2_langgraph_native'

    result = run_experiment_case(
        case=case,
        policies=[LangGraphTreeDriver()],
        output_dir=out_dir,
    )

    row = result.rows[0]
    assert row['policy_name'] == 'external_framework_langgraph'
    assert row['framework_name'] == 'LangGraph'
    assert row['framework_backend_mode'] == 'native_framework'

    with result.artifact_manifest.open('r', encoding='utf-8', newline='') as f:
        manifest_rows = list(csv.DictReader(f))
    assert manifest_rows[0]['framework_name'] == 'LangGraph'
    assert manifest_rows[0]['framework_backend_mode'] == 'native_framework'


def test_run_tree_runtime_v2_case_records_llm_backend_metadata_for_langgraph_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fms_bench import PlantCellConfig
    from fms_bench.v2.framework_drivers import LangGraphLLMTreeDriver
    from fms_bench.v2.tree_runtime_env import TreeRuntimeEnv

    flex_root = PROJECT_ROOT / 'external' / 'benchmarks' / 'flexible jobshop' / 'instances'
    common = {
        'family_key': 'kacem',
        'instance_path': flex_root / 'KacemHammadiBorne2002' / 'kacem1.txt',
        'instance_kind': 'kacem',
        'wip_cap': 2,
    }
    case = HolonicExperimentCase(
        case_id='tree_runtime_v2_small__kacem_pair_langgraph_llm__R0',
        suite_tier='tree_runtime_v2_small',
        level='R0',
        benchmark_item_id='suite=tree_runtime_v2_small__regime=R0__topology=tree_a2_c2__cells=kacem_pair__split=test',
        cell_configs=[
            PlantCellConfig(cell_id=0, label='cell-a', processing_time_scale=0.9, **common),
            PlantCellConfig(cell_id=1, label='cell-b', processing_time_scale=1.1, **common),
        ],
        seeds=[13],
        env_name='tree_runtime_v2',
        env_cls=lambda cell_configs, config, seed: TreeRuntimeEnv(
            cell_configs,
            area_to_cells={0: [0], 1: [1]},
            config=config,
            seed=seed,
        ),
        regime_id='R0',
        topology_id='tree_a2_c2',
    )
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2_langgraph_llm'

    driver = LangGraphLLMTreeDriver(
        model_name='qwen-test',
        base_url='http://127.0.0.1:30000/v1',
        api_key='EMPTY',
    )
    monkeypatch.setattr(
        driver,
        '_select_openai_compat_action',
        lambda *, agent_id, agent_view, legal_actions: next(
            index for index, enabled in enumerate(agent_view['action_mask']) if int(enabled) > 0
        ),
    )

    result = run_experiment_case(
        case=case,
        policies=[driver],
        output_dir=out_dir,
    )

    row = result.rows[0]
    assert row['policy_name'] == 'external_framework_langgraph_llm'
    assert row['framework_name'] == 'LangGraph'
    assert row['framework_backend_mode'] in {'native_framework', 'contract_stub'}
    assert row['framework_llm_backend'] == 'openai_compat'
    assert row['framework_llm_model'] == 'qwen-test'
    assert row['framework_llm_base_url'] == 'http://127.0.0.1:30000/v1'

    with result.artifact_manifest.open('r', encoding='utf-8', newline='') as f:
        manifest_rows = list(csv.DictReader(f))
    assert manifest_rows[0]['framework_llm_backend'] == 'openai_compat'
    assert manifest_rows[0]['framework_llm_model'] == 'qwen-test'
    assert manifest_rows[0]['framework_llm_base_url'] == 'http://127.0.0.1:30000/v1'


def test_run_tree_runtime_v2_case_records_llm_decision_audit_for_langgraph_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fms_bench import PlantCellConfig
    from fms_bench.v2.framework_drivers import LangGraphLLMTreeDriver
    from fms_bench.v2.tree_runtime_env import TreeRuntimeEnv

    flex_root = PROJECT_ROOT / 'external' / 'benchmarks' / 'flexible jobshop' / 'instances'
    common = {
        'family_key': 'kacem',
        'instance_path': flex_root / 'KacemHammadiBorne2002' / 'kacem1.txt',
        'instance_kind': 'kacem',
        'wip_cap': 2,
    }
    case = HolonicExperimentCase(
        case_id='tree_runtime_v2_small__kacem_pair_langgraph_llm_audit__R0',
        suite_tier='tree_runtime_v2_small',
        level='R0',
        benchmark_item_id='suite=tree_runtime_v2_small__regime=R0__topology=tree_a2_c2__cells=kacem_pair__split=test',
        cell_configs=[
            PlantCellConfig(cell_id=0, label='cell-a', processing_time_scale=0.9, **common),
            PlantCellConfig(cell_id=1, label='cell-b', processing_time_scale=1.1, **common),
        ],
        seeds=[13],
        env_name='tree_runtime_v2',
        env_cls=lambda cell_configs, config, seed: TreeRuntimeEnv(
            cell_configs,
            area_to_cells={0: [0], 1: [1]},
            config=config,
            seed=seed,
        ),
        regime_id='R0',
        topology_id='tree_a2_c2',
    )
    out_dir = tmp_path / 'tree_runtime_v2_langgraph_llm_audit'

    driver = LangGraphLLMTreeDriver(
        model_name='qwen-test',
        base_url='http://127.0.0.1:30000/v1',
        api_key='EMPTY',
    )
    monkeypatch.setattr(
        driver,
        '_select_openai_compat_action',
        lambda *, agent_id, agent_view, legal_actions: None,
    )

    result = run_experiment_case(
        case=case,
        policies=[driver],
        output_dir=out_dir,
    )

    row = result.rows[0]
    assert float(row['framework_llm_decision_count_mean']) == 0.0
    assert float(row['framework_fallback_contract_decision_count_mean']) > 0.0
    assert float(row['framework_no_legal_action_count_mean']) == 0.0

    json_path = out_dir / case.case_id / 'external_framework_langgraph_llm.json'
    payload = json.loads(json_path.read_text(encoding='utf-8'))
    episode = payload['episodes'][0]
    assert episode['framework_llm_decision_count'] == 0
    assert episode['framework_fallback_contract_decision_count'] > 0
    assert episode['framework_no_legal_action_count'] == 0
    assert episode['framework_decision_source_trace']
    assert {
        entry['source'] for entry in episode['framework_decision_source_trace']
    } == {'fallback_contract'}


def test_run_tree_runtime_v2_case_records_framework_backend_metadata_for_agentscope() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("agentscope")
    from fms_bench import PlantCellConfig
    from fms_bench.v2.framework_drivers import AgentScopeTreeDriver
    from fms_bench.v2.tree_runtime_env import TreeRuntimeEnv

    flex_root = PROJECT_ROOT / 'external' / 'benchmarks' / 'flexible jobshop' / 'instances'
    common = {
        'family_key': 'kacem',
        'instance_path': flex_root / 'KacemHammadiBorne2002' / 'kacem1.txt',
        'instance_kind': 'kacem',
        'wip_cap': 2,
    }
    case = HolonicExperimentCase(
        case_id='tree_runtime_v2_small__kacem_pair_agentscope_native__R0',
        suite_tier='tree_runtime_v2_small',
        level='R0',
        benchmark_item_id='suite=tree_runtime_v2_small__regime=R0__topology=tree_a2_c2__cells=kacem_pair__split=test',
        cell_configs=[
            PlantCellConfig(cell_id=0, label='cell-a', processing_time_scale=0.9, **common),
            PlantCellConfig(cell_id=1, label='cell-b', processing_time_scale=1.1, **common),
        ],
        seeds=[13],
        env_name='tree_runtime_v2',
        env_cls=lambda cell_configs, config, seed: TreeRuntimeEnv(
            cell_configs,
            area_to_cells={0: [0], 1: [1]},
            config=config,
            seed=seed,
        ),
        regime_id='R0',
        topology_id='tree_a2_c2',
    )
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2_agentscope_native'

    result = run_experiment_case(
        case=case,
        policies=[AgentScopeTreeDriver()],
        output_dir=out_dir,
    )

    row = result.rows[0]
    assert row['policy_name'] == 'external_framework_agentscope'
    assert row['framework_name'] == 'AgentScope'
    assert row['framework_backend_mode'] == 'native_framework'
    assert row['framework_python_package'] == 'agentscope'
    assert row['framework_optional_dependency'] == 'framework_agentscope'
    assert row['framework_uv_project_environment'] == '.uv_envs/agentscope'

    with result.artifact_manifest.open('r', encoding='utf-8', newline='') as f:
        manifest_rows = list(csv.DictReader(f))
    assert manifest_rows[0]['framework_name'] == 'AgentScope'
    assert manifest_rows[0]['framework_backend_mode'] == 'native_framework'
    assert manifest_rows[0]['framework_python_package'] == 'agentscope'
    assert manifest_rows[0]['framework_optional_dependency'] == 'framework_agentscope'
    assert manifest_rows[0]['framework_uv_project_environment'] == '.uv_envs/agentscope'


def test_run_tree_runtime_v2_case_records_framework_backend_metadata_for_spade() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("spade")
    from fms_bench import PlantCellConfig
    from fms_bench.v2.framework_drivers import SPADETreeDriver
    from fms_bench.v2.tree_runtime_env import TreeRuntimeEnv

    flex_root = PROJECT_ROOT / 'external' / 'benchmarks' / 'flexible jobshop' / 'instances'
    common = {
        'family_key': 'kacem',
        'instance_path': flex_root / 'KacemHammadiBorne2002' / 'kacem1.txt',
        'instance_kind': 'kacem',
        'wip_cap': 2,
    }
    case = HolonicExperimentCase(
        case_id='tree_runtime_v2_small__kacem_pair_spade_native__R0',
        suite_tier='tree_runtime_v2_small',
        level='R0',
        benchmark_item_id='suite=tree_runtime_v2_small__regime=R0__topology=tree_a2_c2__cells=kacem_pair__split=test',
        cell_configs=[
            PlantCellConfig(cell_id=0, label='cell-a', processing_time_scale=0.9, **common),
            PlantCellConfig(cell_id=1, label='cell-b', processing_time_scale=1.1, **common),
        ],
        seeds=[13],
        env_name='tree_runtime_v2',
        env_cls=lambda cell_configs, config, seed: TreeRuntimeEnv(
            cell_configs,
            area_to_cells={0: [0], 1: [1]},
            config=config,
            seed=seed,
        ),
        regime_id='R0',
        topology_id='tree_a2_c2',
    )
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2_spade_native'

    result = run_experiment_case(
        case=case,
        policies=[SPADETreeDriver()],
        output_dir=out_dir,
    )

    row = result.rows[0]
    assert row['policy_name'] == 'external_framework_spade'
    assert row['framework_name'] == 'SPADE'
    assert row['framework_backend_mode'] == 'native_framework'

    with result.artifact_manifest.open('r', encoding='utf-8', newline='') as f:
        manifest_rows = list(csv.DictReader(f))
    assert manifest_rows[0]['framework_name'] == 'SPADE'
    assert manifest_rows[0]['framework_backend_mode'] == 'native_framework'


def test_run_tree_runtime_v2_case_records_framework_backend_metadata_for_google_adk() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("google.adk")
    from fms_bench import PlantCellConfig
    from fms_bench.v2.framework_drivers import GoogleADKTreeDriver
    from fms_bench.v2.tree_runtime_env import TreeRuntimeEnv

    flex_root = PROJECT_ROOT / 'external' / 'benchmarks' / 'flexible jobshop' / 'instances'
    common = {
        'family_key': 'kacem',
        'instance_path': flex_root / 'KacemHammadiBorne2002' / 'kacem1.txt',
        'instance_kind': 'kacem',
        'wip_cap': 2,
    }
    case = HolonicExperimentCase(
        case_id='tree_runtime_v2_small__kacem_pair_google_adk_native__R0',
        suite_tier='tree_runtime_v2_small',
        level='R0',
        benchmark_item_id='suite=tree_runtime_v2_small__regime=R0__topology=tree_a2_c2__cells=kacem_pair__split=test',
        cell_configs=[
            PlantCellConfig(cell_id=0, label='cell-a', processing_time_scale=0.9, **common),
            PlantCellConfig(cell_id=1, label='cell-b', processing_time_scale=1.1, **common),
        ],
        seeds=[13],
        env_name='tree_runtime_v2',
        env_cls=lambda cell_configs, config, seed: TreeRuntimeEnv(
            cell_configs,
            area_to_cells={0: [0], 1: [1]},
            config=config,
            seed=seed,
        ),
        regime_id='R0',
        topology_id='tree_a2_c2',
    )
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2_google_adk_native'

    result = run_experiment_case(
        case=case,
        policies=[GoogleADKTreeDriver()],
        output_dir=out_dir,
    )

    row = result.rows[0]
    assert row['policy_name'] == 'external_framework_google_adk'
    assert row['framework_name'] == 'Google ADK'
    assert row['framework_backend_mode'] == 'native_framework'

    with result.artifact_manifest.open('r', encoding='utf-8', newline='') as f:
        manifest_rows = list(csv.DictReader(f))
    assert manifest_rows[0]['framework_name'] == 'Google ADK'
    assert manifest_rows[0]['framework_backend_mode'] == 'native_framework'


def test_build_tree_runtime_v2_small_suite_contains_runtime_env_cases() -> None:
    suite = build_tree_runtime_v2_small_suite(PROJECT_ROOT)

    case = suite[0]
    assert case.env_name == 'tree_runtime_v2'
    assert case.regime_id in {'R0', 'R4'}
    assert 'suite=tree_runtime_v2_small' in case.benchmark_item_id
    assert 'topology=' in case.benchmark_item_id


def test_build_tree_runtime_v2_small_suite_exposes_nontrivial_area_topologies() -> None:
    suite = build_tree_runtime_v2_small_suite(PROJECT_ROOT)

    topologies = sorted({case.topology_id for case in suite})
    compositions = sorted({case.case_id.split('__')[1] for case in suite})

    assert topologies == ['tree_a2_c3', 'tree_a3_c3']
    assert compositions == ['fattahi_three_cell', 'kacem_three_cell']
    assert all(len(case.cell_configs) == 3 for case in suite)
    assert any(case.topology_id == 'tree_a2_c3' for case in suite)


def test_build_tree_runtime_v2_small_suite_uses_locked_test_seed_protocol() -> None:
    suite = build_tree_runtime_v2_small_suite(PROJECT_ROOT)

    with (PROJECT_ROOT / 'docs' / 'benchmark_seed_protocol_v1.json').open('r', encoding='utf-8-sig') as f:
        expected_test_seeds = __import__('json').load(f)['test_seeds']

    assert suite
    assert all(case.seeds == expected_test_seeds for case in suite)
    assert all('__split=test' in case.benchmark_item_id for case in suite)


def test_tree_runtime_v2_example_script_runs_as_cli_without_module_import_errors() -> None:
    script_path = PROJECT_ROOT / 'examples' / 'run_tree_runtime_v2_small.py'

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                "import runpy; "
                f"runpy.run_path(r'{script_path}', run_name='not_main')"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_tree_runtime_v2_pilot_example_script_runs_as_cli_without_module_import_errors() -> None:
    script_path = PROJECT_ROOT / 'examples' / 'run_tree_runtime_v2_pilot.py'

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                "import runpy; "
                f"runpy.run_path(r'{script_path}', run_name='not_main')"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_build_tree_runtime_v2_pilot_suite_contains_ordertrap_and_r2_cases() -> None:
    suite = build_tree_runtime_v2_pilot_suite(PROJECT_ROOT)

    topologies = sorted({case.topology_id for case in suite})
    compositions = sorted({case.case_id.split('__')[1] for case in suite})
    regimes = sorted({case.regime_id for case in suite})

    with (PROJECT_ROOT / 'docs' / 'benchmark_seed_protocol_v1.json').open('r', encoding='utf-8-sig') as f:
        expected_dev_seeds = __import__('json').load(f)['validation_seeds']

    assert topologies == ['tree_a2_c3', 'tree_a3_c3']
    assert compositions == ['brandimarte_ordertrap_three_cell', 'fattahi_three_cell', 'kacem_three_cell']
    assert regimes == ['R0', 'R2', 'R4']
    assert all(case.env_name == 'tree_runtime_v2' for case in suite)
    assert all(case.seeds == expected_dev_seeds for case in suite)
    assert all('__split=dev' in case.benchmark_item_id for case in suite)


def test_tree_runtime_v2_pilot_suite_contains_discriminative_ordertrap_case() -> None:
    suite = build_tree_runtime_v2_pilot_suite(PROJECT_ROOT)
    case = next(
        item
        for item in suite
        if item.case_id == 'tree_runtime_v2_pilot__brandimarte_ordertrap_three_cell__tree_a2_c3__R0'
    )
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2'

    result = run_experiment_case(
        case=case,
        policies=[FlatTreePolicy(), HierarchicalTreePolicy(), TreeContractNetPolicy()],
        output_dir=out_dir,
    )

    rows = {row['policy_name']: row for row in result.rows}
    assert float(rows['tree_flat_mask_first']['violation_magnitude_mean']) != float(rows['tree_contract_net_budgeted']['violation_magnitude_mean'])


def test_tree_runtime_v2_wide_example_script_runs_as_cli_without_module_import_errors() -> None:
    script_path = PROJECT_ROOT / 'examples' / 'run_tree_runtime_v2_wide.py'

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                "import runpy; "
                f"runpy.run_path(r'{script_path}', run_name='not_main')"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_tree_runtime_v2_ladder_example_script_runs_as_cli_without_module_import_errors() -> None:
    script_path = PROJECT_ROOT / 'examples' / 'run_tree_runtime_v2_ladder.py'

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                "import runpy; "
                f"runpy.run_path(r'{script_path}', run_name='not_main')"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_tree_runtime_v2_external_family_ladder_example_script_imports_cleanly() -> None:
    script_path = PROJECT_ROOT / 'examples' / 'run_tree_runtime_v2_external_family_ladder.py'

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                "import runpy; "
                f"runpy.run_path(r'{script_path}', run_name='not_main')"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_build_tree_runtime_v2_wide_suite_uses_locked_test_widened_matrix() -> None:
    suite = build_tree_runtime_v2_wide_suite(PROJECT_ROOT)

    topologies = sorted({case.topology_id for case in suite})
    compositions = sorted({case.case_id.split('__')[1] for case in suite})
    regimes = sorted({case.regime_id for case in suite})

    with (PROJECT_ROOT / 'docs' / 'benchmark_seed_protocol_v1.json').open('r', encoding='utf-8-sig') as f:
        expected_test_seeds = __import__('json').load(f)['test_seeds']

    assert topologies == ['tree_a2_c3', 'tree_a3_c3']
    assert compositions == ['brandimarte_ordertrap_three_cell', 'fattahi_three_cell', 'kacem_three_cell']
    assert regimes == ['R0', 'R2', 'R4']
    assert all(case.env_name == 'tree_runtime_v2' for case in suite)
    assert all(case.seeds == expected_test_seeds for case in suite)
    assert all(case.suite_tier == 'tree_runtime_v2_wide' for case in suite)
    assert all('__split=test' in case.benchmark_item_id for case in suite)


def test_build_tree_runtime_v2_ladder_suite_extends_beyond_three_cell_matrix() -> None:
    suite = build_tree_runtime_v2_ladder_suite(PROJECT_ROOT)

    topologies = sorted({case.topology_id for case in suite})
    compositions = sorted({case.case_id.split('__')[1] for case in suite})
    regimes = sorted({case.regime_id for case in suite})
    cell_counts = sorted({len(case.cell_configs) for case in suite})

    with (PROJECT_ROOT / 'docs' / 'benchmark_seed_protocol_v1.json').open('r', encoding='utf-8-sig') as f:
        expected_test_seeds = __import__('json').load(f)['test_seeds']

    assert topologies == [
        'tree_a2_c3',
        'tree_a2_c4',
        'tree_a2_c5',
        'tree_a3_c3',
        'tree_a3_c4',
        'tree_a3_c5',
    ]
    assert compositions == [
        'brandimarte_ordertrap_five_cell',
        'brandimarte_ordertrap_three_cell',
        'fattahi_four_cell',
        'fattahi_three_cell',
        'kacem_four_cell',
        'kacem_three_cell',
    ]
    assert regimes == ['R0', 'R2', 'R4']
    assert cell_counts == [3, 4, 5]
    assert all(case.env_name == 'tree_runtime_v2' for case in suite)
    assert all(case.seeds == expected_test_seeds for case in suite)
    assert all(case.suite_tier == 'tree_runtime_v2_ladder' for case in suite)


def test_build_tree_runtime_v2_intercell_small_suite_contains_first_locked_case() -> None:
    suite = build_tree_runtime_v2_intercell_small_suite(PROJECT_ROOT)

    assert len(suite) == 1
    case = suite[0]
    assert case.suite_tier == "tree_runtime_v2_intercell_small"
    assert case.regime_id == "R0"
    assert case.topology_id == "tree_a2_c3"
    assert case.env_name == "tree_runtime_v2_intercell"
    assert "suite=tree_runtime_v2_intercell_small" in case.benchmark_item_id
    assert "__cells=kacem_intercell_three_cell__" in case.benchmark_item_id
    assert len(case.cell_configs) == 3
    assert "__split=test" in case.benchmark_item_id


def test_build_tree_runtime_v2_intercell_small_suite_accepts_congestion_knobs() -> None:
    suite = build_tree_runtime_v2_intercell_small_suite(
        PROJECT_ROOT,
        max_in_flight_jobs=3,
        inbound_buffer_cap=2,
    )

    case = suite[0]
    assert all(cell.inbound_buffer_cap == 2 for cell in case.cell_configs)

    env = case.env_cls(case.cell_configs, experiment_runner_module.make_regime_config("R0"), 13)
    try:
        assert env._env.max_in_flight_jobs == 3
    finally:
        env.close()


def test_build_tree_runtime_v2_intercell_brandimarte_probe_suite_freezes_moderate_congestion_point() -> None:
    suite = build_tree_runtime_v2_intercell_brandimarte_probe_suite(PROJECT_ROOT)

    assert len(suite) == 1
    case = suite[0]
    assert case.suite_tier == "tree_runtime_v2_intercell_brandimarte_probe"
    assert case.regime_id == "R0"
    assert case.topology_id == "tree_a2_c3"
    assert case.env_name == "tree_runtime_v2_intercell"
    assert "__cells=brandimarte_ordertrap_three_cell__" in case.benchmark_item_id
    assert "__k=2__" in case.benchmark_item_id
    assert "__b=1__" in case.benchmark_item_id
    assert "__profile=moderate_congestion__" in case.benchmark_item_id
    assert "__c1=1.20__" in case.benchmark_item_id
    assert "__c2=1.05__" in case.benchmark_item_id
    assert "__split=test" in case.benchmark_item_id
    assert len(case.cell_configs) == 3
    assert case.cell_configs[0].inbound_buffer_cap == 1
    assert case.cell_configs[1].processing_time_scale == pytest.approx(1.2)
    assert case.cell_configs[2].processing_time_scale == pytest.approx(1.05)

    env = case.env_cls(case.cell_configs, experiment_runner_module.make_regime_config("R0"), 13)
    try:
        assert env._env.max_in_flight_jobs == 2
    finally:
        env.close()


def test_build_tree_runtime_v2_intercell_blocking_probe_suite_freezes_structural_blocking_point() -> None:
    suite = build_tree_runtime_v2_intercell_blocking_probe_suite(PROJECT_ROOT)

    assert len(suite) == 1
    case = suite[0]
    assert case.suite_tier == "tree_runtime_v2_intercell_blocking_probe"
    assert case.regime_id == "R0"
    assert case.topology_id == "tree_a2_c3"
    assert case.env_name == "tree_runtime_v2_intercell"
    assert "__cells=brandimarte_ordertrap_three_cell__" in case.benchmark_item_id
    assert "__profile=blocking_anchor__" in case.benchmark_item_id
    assert "__k=2__" in case.benchmark_item_id
    assert "__ib0=1__" in case.benchmark_item_id
    assert "__ib1=1__" in case.benchmark_item_id
    assert "__ib2=0__" in case.benchmark_item_id
    assert "__c0=0.65__" in case.benchmark_item_id
    assert "__c1=2.20__" in case.benchmark_item_id
    assert "__c2=1.00__" in case.benchmark_item_id
    assert len(case.cell_configs) == 3
    assert case.cell_configs[0].processing_time_scale == pytest.approx(0.65)
    assert case.cell_configs[1].processing_time_scale == pytest.approx(2.2)
    assert case.cell_configs[2].processing_time_scale == pytest.approx(1.0)
    assert [cell.inbound_buffer_cap for cell in case.cell_configs] == [1, 1, 0]

    env = case.env_cls(case.cell_configs, experiment_runner_module.make_regime_config("R0"), 13)
    try:
        assert env._env.max_in_flight_jobs == 2
    finally:
        env.close()


def test_build_tree_runtime_v2_intercell_wide_suite_exposes_complete_category_b_surface() -> None:
    suite = build_tree_runtime_v2_intercell_wide_suite(PROJECT_ROOT)

    assert len(suite) == 3
    assert all(case.suite_tier == "tree_runtime_v2_intercell_wide" for case in suite)
    assert all(case.env_name == "tree_runtime_v2_intercell" for case in suite)
    assert all(case.regime_id == "R0" for case in suite)
    assert all(case.topology_id == "tree_a2_c3" for case in suite)
    case_ids = [case.case_id for case in suite]
    assert case_ids == [
        "tree_runtime_v2_intercell_wide__kacem_intercell_three_cell__tree_a2_c3__R0__locked_anchor",
        "tree_runtime_v2_intercell_wide__brandimarte_ordertrap_three_cell__tree_a2_c3__R0__moderate_congestion",
        "tree_runtime_v2_intercell_wide__brandimarte_ordertrap_three_cell__tree_a2_c3__R0__stress_congestion",
    ]
    benchmark_item_ids = [case.benchmark_item_id for case in suite]
    assert "__profile=locked_anchor__" in benchmark_item_ids[0]
    assert "__profile=moderate_congestion__" in benchmark_item_ids[1]
    assert "__profile=stress_congestion__" in benchmark_item_ids[2]
    assert "__cells=kacem_intercell_three_cell__" in benchmark_item_ids[0]
    assert "__cells=brandimarte_ordertrap_three_cell__" in benchmark_item_ids[1]
    assert "__cells=brandimarte_ordertrap_three_cell__" in benchmark_item_ids[2]
    assert suite[1].cell_configs[1].processing_time_scale == pytest.approx(1.2)
    assert suite[1].cell_configs[2].processing_time_scale == pytest.approx(1.05)
    assert suite[2].cell_configs[1].processing_time_scale == pytest.approx(2.0)
    assert suite[2].cell_configs[2].processing_time_scale == pytest.approx(1.0)

    for case, expected_inflight in zip(suite, [1, 2, 2], strict=True):
        env = case.env_cls(case.cell_configs, experiment_runner_module.make_regime_config("R0"), 13)
        try:
            assert env._env.max_in_flight_jobs == expected_inflight
        finally:
            env.close()


def test_build_tree_runtime_v2_intercell_dynamics_extension_suite_surfaces_arrival_failure_and_setup_profiles() -> None:
    suite = build_tree_runtime_v2_intercell_dynamics_extension_suite(PROJECT_ROOT)

    assert len(suite) == 4
    assert {case.suite_tier for case in suite} == {
        "tree_runtime_v2_intercell_dynamics_extension"
    }
    assert all(case.env_name == "tree_runtime_v2_intercell" for case in suite)
    assert all(case.arrival_profile_id == "arrival_nhpp_random_batch_nominal_v1" for case in suite)
    assert all(case.setup_profile_id == "setup_family_change_nominal_v1" for case in suite)
    assert {case.failure_profile_id for case in suite} == {
        "failure_none_v1",
        "failure_exponential_nominal_v1",
        "failure_weibull_aging_nominal_v1",
        "failure_load_dependent_nominal_v1",
    }


def test_build_tree_runtime_v2_intercell_a3c9_probe_suite_surfaces_tree_a3_c9_case() -> None:
    suite = build_tree_runtime_v2_intercell_a3c9_probe_suite(PROJECT_ROOT)

    assert len(suite) == 1
    case = suite[0]
    assert case.suite_tier == "tree_runtime_v2_intercell_a3c9_probe"
    assert case.regime_id == "R0"
    assert case.topology_id == "tree_a3_c9"
    assert case.env_name == "tree_runtime_v2_intercell"
    assert "__cells=brandimarte_ordertrap_nine_cell__" in case.benchmark_item_id
    assert "__profile=moderate_expansion__" in case.benchmark_item_id
    assert "__k=3__" in case.benchmark_item_id
    assert "__bk=3__" in case.benchmark_item_id
    assert "__rg=0-1-3-4-6-7_1-2-4-5-7-8_0-2-3-5-6-8__" in case.benchmark_item_id
    assert len(case.cell_configs) == 9

    env = case.env_cls(case.cell_configs, experiment_runner_module.make_regime_config("R0"), 13)
    try:
        assert env._env.max_in_flight_jobs == 3
        assert env._env.area_ids == [0, 1, 2]
        assert tuple(env._env.route_grammar) == (
            (0, 1, 3, 4, 6, 7),
            (1, 2, 4, 5, 7, 8),
            (0, 2, 3, 5, 6, 8),
        )
    finally:
        env.close()


def test_build_tree_runtime_v2_intercell_a3c9_wide_suite_surfaces_main_release_profiles() -> None:
    suite = build_tree_runtime_v2_intercell_a3c9_wide_suite(PROJECT_ROOT)

    expected_authority_modes = {
        "centralized",
        "hierarchical",
        "heterarchical_cnp",
        "holonic_hybrid",
    }

    assert len(suite) == 12
    assert {case.suite_tier for case in suite} == {"tree_runtime_v2_intercell_a3c9_wide"}
    assert {case.topology_id for case in suite} == {"tree_a3_c9"}
    assert {case.arrival_profile_id for case in suite} == {"arrival_nhpp_random_batch_overlap_a3c9_v2"}
    assert {case.setup_profile_id for case in suite} == {"setup_family_change_nominal_v1"}
    assert {case.failure_profile_id for case in suite} == {
        "failure_exponential_nominal_v1",
        "failure_weibull_aging_nominal_v1",
        "failure_load_dependent_nominal_v1",
    }
    assert {case.official_surface_id for case in suite} == {"intercell_a3c9_wide"}
    assert {case.official_surface_role for case in suite} == {"official_standard"}
    assert {case.semantic_setup_enabled for case in suite} == {True}
    assert {case.semantic_buffer_wip_enabled for case in suite} == {True}
    assert {case.semantic_transport_enabled for case in suite} == {True}
    assert {case.benchmark_item_id.split("__profile=", 1)[1].split("__", 1)[0] for case in suite} == {
        "balanced_branching_anchor",
        "cross_area_tradeoff",
        "late_stage_reversal",
    }

    by_profile: dict[str, list[HolonicExperimentCase]] = {}
    by_mode: dict[str, HolonicExperimentCase] = {}
    for case in suite:
        profile_id = case.benchmark_item_id.split("__profile=", 1)[1].split("__", 1)[0]
        authority_mode = case.benchmark_item_id.split("__authority=", 1)[1].split("__", 1)[0]
        by_profile.setdefault(profile_id, []).append(case)
        by_mode.setdefault(authority_mode, case)
        assert case.case_id.endswith(f"__authority_{authority_mode}")
        assert f"__authority={authority_mode}" in case.benchmark_item_id

    assert set(by_profile) == {
        "balanced_branching_anchor",
        "cross_area_tradeoff",
        "late_stage_reversal",
    }
    assert all(
        {
            case.benchmark_item_id.split("__authority=", 1)[1].split("__", 1)[0]
            for case in profile_cases
        }
        == expected_authority_modes
        for profile_cases in by_profile.values()
    )

    late_stage_reversal = next(
        case
        for case in by_profile["late_stage_reversal"]
        if "__authority=holonic_hybrid" in case.benchmark_item_id
    )
    assert "__k=3" in late_stage_reversal.benchmark_item_id
    for token in ("__ib2=0", "__ib5=0", "__ib6=0", "__ib8=0"):
        assert token in late_stage_reversal.benchmark_item_id
    assert "__c0=1.80" in late_stage_reversal.benchmark_item_id
    assert "__c3=1.80" in late_stage_reversal.benchmark_item_id
    assert "__tm=1.30" in late_stage_reversal.benchmark_item_id

    for authority_mode, case in by_mode.items():
        env = case.env_cls(case.cell_configs, experiment_runner_module.make_regime_config("R0"), 13)
        try:
            assert env.authority_mode.mode_id == authority_mode
        finally:
            env.close()


def test_build_tree_runtime_v2_intercell_a3c9_dynamics_suite_surfaces_deprecated_alias_profiles() -> None:
    suite = build_tree_runtime_v2_intercell_a3c9_dynamics_suite(PROJECT_ROOT)

    expected_authority_modes = {
        "centralized",
        "hierarchical",
        "heterarchical_cnp",
        "holonic_hybrid",
    }

    assert len(suite) == 12
    assert {case.suite_tier for case in suite} == {"tree_runtime_v2_intercell_a3c9_dynamics"}
    assert len({case.case_id for case in suite}) == 12
    assert {case.topology_id for case in suite} == {"tree_a3_c9"}
    assert {case.official_surface_id for case in suite} == {"intercell_a3c9_wide"}
    assert {case.official_surface_role for case in suite} == {"deprecated_alias"}
    assert {case.arrival_profile_id for case in suite} == {"arrival_nhpp_random_batch_overlap_a3c9_v2"}
    assert {case.setup_profile_id for case in suite} == {"setup_family_change_nominal_v1"}
    assert {case.failure_profile_id for case in suite} == {
        "failure_exponential_nominal_v1",
        "failure_weibull_aging_nominal_v1",
        "failure_load_dependent_nominal_v1",
    }

    by_profile: dict[str, list[HolonicExperimentCase]] = {}
    by_mode: dict[str, HolonicExperimentCase] = {}
    for case in suite:
        profile_id = case.benchmark_item_id.split("__profile=", 1)[1].split("__", 1)[0]
        authority_mode = case.benchmark_item_id.split("__authority=", 1)[1].split("__", 1)[0]
        by_profile.setdefault(profile_id, []).append(case)
        by_mode.setdefault(authority_mode, case)
        assert case.case_id.endswith(f"__authority_{authority_mode}")
        assert f"__authority={authority_mode}" in case.benchmark_item_id

    assert set(by_profile) == {
        "arrival_setup_exponential",
        "arrival_setup_weibull",
        "arrival_setup_load_dependent",
    }
    assert all(
        {
            case.benchmark_item_id.split("__authority=", 1)[1].split("__", 1)[0]
            for case in profile_cases
        }
        == expected_authority_modes
        for profile_cases in by_profile.values()
    )

    exponential = next(
        case
        for case in by_profile["arrival_setup_exponential"]
        if "__authority=holonic_hybrid" in case.benchmark_item_id
    )
    assert "__k=4" in exponential.benchmark_item_id
    assert {case.benchmark_item_id.count("__bk=3") for case in suite} == {1}
    for token in ("__ib2=0", "__ib3=0", "__ib5=0", "__ib6=0", "__ib8=0"):
        assert token in exponential.benchmark_item_id
    assert "__c0=1.55" in exponential.benchmark_item_id
    assert "__arrival=arrival_nhpp_random_batch_overlap_a3c9_v2" in exponential.benchmark_item_id

    weibull = next(
        case
        for case in by_profile["arrival_setup_weibull"]
        if "__authority=holonic_hybrid" in case.benchmark_item_id
    )
    for token in ("__ib2=0", "__ib3=0", "__ib5=0", "__ib6=0", "__ib8=0"):
        assert token in weibull.benchmark_item_id
    assert "__c4=1.30" in weibull.benchmark_item_id
    assert "__c8=1.10" in weibull.benchmark_item_id
    assert "__arrival=arrival_nhpp_random_batch_overlap_a3c9_v2" in weibull.benchmark_item_id

    load = next(
        case
        for case in by_profile["arrival_setup_load_dependent"]
        if "__authority=holonic_hybrid" in case.benchmark_item_id
    )
    for token in ("__ib2=0", "__ib3=0", "__ib5=0", "__ib6=0", "__ib8=0"):
        assert token in load.benchmark_item_id
    assert "__c0=1.20" in load.benchmark_item_id
    assert "__c4=1.20" in load.benchmark_item_id
    assert "__c5=1.45" in load.benchmark_item_id
    assert "__arrival=arrival_nhpp_random_batch_overlap_a3c9_v2" in load.benchmark_item_id

    for authority_mode, case in by_mode.items():
        env = case.env_cls(case.cell_configs, experiment_runner_module.make_regime_config("R0"), 13)
        try:
            assert env.authority_mode.mode_id == authority_mode
        finally:
            env.close()


def test_case_config_allows_explicit_horizon_override_without_changing_baseline_dynamics() -> None:
    baseline_case = build_tree_runtime_v2_intercell_a3c9_dynamics_suite(PROJECT_ROOT)[0]
    override_case = replace(
        baseline_case,
        horizon_hours_override=375.0,
        budget_tier="short",
        budget_relative_to_nominal=0.75,
    )

    baseline_config = experiment_runner_module._case_config(baseline_case)
    override_config = experiment_runner_module._case_config(override_case)

    assert baseline_config.horizon_hours == 500
    assert override_config.horizon_hours == 375.0


def test_build_tree_runtime_v2_intercell_a3c9_dynamics_budget_sweep_suite_freezes_short_long_surface() -> None:
    suite = build_tree_runtime_v2_intercell_a3c9_dynamics_budget_sweep_suite(PROJECT_ROOT)

    assert len(suite) == 6
    assert {case.suite_tier for case in suite} == {"tree_runtime_v2_intercell_a3c9_dynamics_budget_sweep"}
    assert {case.topology_id for case in suite} == {"tree_a3_c9"}
    assert {case.official_surface_id for case in suite} == {"intercell_a3c9_wide"}
    assert {case.official_surface_role for case in suite} == {"appendix_budget_extension"}
    assert {case.arrival_profile_id for case in suite} == {"arrival_nhpp_random_batch_overlap_a3c9_v2"}
    assert {case.setup_profile_id for case in suite} == {"setup_family_change_nominal_v1"}
    assert {case.budget_tier for case in suite} == {"short", "long"}
    assert {case.budget_relative_to_nominal for case in suite} == {0.75, 1.5}

    by_budget = {
        budget_tier: [case for case in suite if case.budget_tier == budget_tier]
        for budget_tier in ("short", "long")
    }
    assert len(by_budget["short"]) == 3
    assert len(by_budget["long"]) == 3

    short_horizons = {case.horizon_hours_override for case in by_budget["short"]}
    long_horizons = {case.horizon_hours_override for case in by_budget["long"]}
    assert short_horizons == {375.0}
    assert long_horizons == {750.0}

    assert {
        case.benchmark_item_id.split("__profile=", 1)[1].split("__", 1)[0]
        for case in suite
    } == {
        "arrival_setup_exponential",
        "arrival_setup_weibull",
        "arrival_setup_load_dependent",
    }

    short_case_ids = {case.case_id for case in by_budget["short"]}
    long_case_ids = {case.case_id for case in by_budget["long"]}
    assert all("__budget_short" in case_id for case_id in short_case_ids)
    assert all("__budget_long" in case_id for case_id in long_case_ids)
    assert all("__budget=short" in case.benchmark_item_id for case in by_budget["short"])
    assert all("__budget=long" in case.benchmark_item_id for case in by_budget["long"])


def test_build_tree_runtime_v2_intercell_a3c9_nightmare_suite_surfaces_appendix_profiles() -> None:
    suite = build_tree_runtime_v2_intercell_a3c9_nightmare_suite(PROJECT_ROOT)

    expected_authority_modes = {
        "centralized",
        "hierarchical",
        "heterarchical_cnp",
        "holonic_hybrid",
    }

    assert len(suite) == 8
    assert {case.suite_tier for case in suite} == {"tree_runtime_v2_intercell_a3c9_nightmare"}
    assert {case.topology_id for case in suite} == {"tree_a3_c9"}
    assert {case.official_surface_id for case in suite} == {"intercell_a3c9_nightmare"}
    assert {case.official_surface_role for case in suite} == {"official_nightmare"}
    assert {
        case.benchmark_item_id.split("__profile=", 1)[1].split("__", 1)[0]
        for case in suite
    } == {
        "blocking_reroute_choke",
        "night_shift_breakdown_pressure",
    }

    by_profile: dict[str, list[HolonicExperimentCase]] = {}
    by_mode: dict[str, HolonicExperimentCase] = {}
    for case in suite:
        profile_id = case.benchmark_item_id.split("__profile=", 1)[1].split("__", 1)[0]
        authority_mode = case.benchmark_item_id.split("__authority=", 1)[1].split("__", 1)[0]
        by_profile.setdefault(profile_id, []).append(case)
        by_mode.setdefault(authority_mode, case)
        assert case.case_id.endswith(f"__authority_{authority_mode}")
        assert f"__authority={authority_mode}" in case.benchmark_item_id

    assert all(
        {
            case.benchmark_item_id.split("__authority=", 1)[1].split("__", 1)[0]
            for case in profile_cases
        }
        == expected_authority_modes
        for profile_cases in by_profile.values()
    )

    blocking = next(
        case
        for case in by_profile["blocking_reroute_choke"]
        if "__authority=holonic_hybrid" in case.benchmark_item_id
    )
    assert "__k=5" in blocking.benchmark_item_id
    for token in ("__ib2=0", "__ib4=0", "__ib5=0", "__ib7=0", "__ib8=0"):
        assert token in blocking.benchmark_item_id
    assert "__c1=1.55" in blocking.benchmark_item_id
    assert "__c4=1.95" in blocking.benchmark_item_id
    assert "__c7=1.70" in blocking.benchmark_item_id
    assert "__tm=1.45" in blocking.benchmark_item_id

    night = next(
        case
        for case in by_profile["night_shift_breakdown_pressure"]
        if "__authority=holonic_hybrid" in case.benchmark_item_id
    )
    assert "__k=5" in night.benchmark_item_id
    for token in ("__ib2=0", "__ib4=0", "__ib5=0", "__ib7=0", "__ib8=0"):
        assert token in night.benchmark_item_id
    assert "__c1=1.45" in night.benchmark_item_id
    assert "__c4=1.85" in night.benchmark_item_id
    assert "__c7=1.60" in night.benchmark_item_id
    assert "__tm=1.50" in night.benchmark_item_id

    for authority_mode, case in by_mode.items():
        env = case.env_cls(case.cell_configs, experiment_runner_module.make_regime_config("R0"), 13)
        try:
            assert env.authority_mode.mode_id == authority_mode
        finally:
            env.close()


def test_build_tree_runtime_v2_intercell_a5c12_harder_suite_surfaces_three_profiles() -> None:
    suite = experiment_runner_module.build_tree_runtime_v2_intercell_a5c12_harder_suite(PROJECT_ROOT)

    assert len(suite) == 12
    assert len({case.case_id for case in suite}) == 12
    assert {case.suite_tier for case in suite} == {"tree_runtime_v2_intercell_a5c12_harder"}
    assert {case.topology_id for case in suite} == {"tree_a5_c12"}
    assert {case.official_surface_id for case in suite} == {"intercell_a5c12_harder"}
    assert {case.official_surface_role for case in suite} == {"experimental_harder_surface"}
    assert {
        case.profile_id for case in suite
    } == {
        "branch_pressure_k4_tm115_bk4",
        "cluster_pull_k4_tm120_bk4",
        "late_commit_soft_k4_tm125_bk4",
    }
    assert {case.semantic_setup_enabled for case in suite} == {True}
    assert {case.semantic_buffer_wip_enabled for case in suite} == {True}
    assert {case.semantic_transport_enabled for case in suite} == {True}


def test_build_tree_runtime_v2_intercell_a5c12_harder_candidates_suite_exposes_soft_knob_grid() -> None:
    suite = build_tree_runtime_v2_intercell_a5c12_harder_candidates_suite(PROJECT_ROOT)

    assert len(suite) == 40
    assert {case.suite_tier for case in suite} == {
        "tree_runtime_v2_intercell_a5c12_harder_candidates"
    }
    assert all(case.topology_id == "tree_a5_c12" for case in suite)
    assert any("__k=5" in case.benchmark_item_id for case in suite)
    assert any("__k=6" in case.benchmark_item_id for case in suite)
    assert any("__bk=5" in case.benchmark_item_id for case in suite)
    assert any("__tm=1.30" in case.benchmark_item_id for case in suite)
    assert any("__ib11=0" in case.benchmark_item_id for case in suite)
    assert {case.authority_mode_id for case in suite} == {
        "centralized",
        "hierarchical",
        "heterarchical_cnp",
        "holonic_hybrid",
    }


def test_build_tree_runtime_v2_intercell_a5c12_harder_suite_is_subset_of_candidates() -> None:
    candidate_suite = build_tree_runtime_v2_intercell_a5c12_harder_candidates_suite(PROJECT_ROOT)
    final_suite = experiment_runner_module.build_tree_runtime_v2_intercell_a5c12_harder_suite(PROJECT_ROOT)

    candidate_keys = {(case.profile_id, case.authority_mode_id) for case in candidate_suite}
    final_keys = {(case.profile_id, case.authority_mode_id) for case in final_suite}

    assert final_suite
    assert len(final_suite) < len(candidate_suite)
    assert final_keys.issubset(candidate_keys)
    assert {case.suite_tier for case in final_suite} == {"tree_runtime_v2_intercell_a5c12_harder"}


def test_run_tree_runtime_v2_intercell_small_case_writes_runtime_artifacts() -> None:
    suite = build_tree_runtime_v2_intercell_small_suite(PROJECT_ROOT)
    case = replace(suite[0], seeds=[13])
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2_intercell'

    result = run_experiment_case(
        case=case,
        policies=[FlatTreePolicy()],
        output_dir=out_dir,
    )

    assert result.summary_csv.exists()
    assert result.artifact_manifest.exists()
    row = result.rows[0]
    assert row['policy_name'] == 'tree_flat_mask_first'
    assert row['topology_id'] == 'tree_a2_c3'
    assert float(row['activation_record_count_mean']) > 0.0
    assert float(row['settlement_event_count_mean']) > 0.0

    with result.artifact_manifest.open('r', encoding='utf-8', newline='') as f:
        manifest_rows = list(csv.DictReader(f))
    assert len(manifest_rows) == 1
    payload = __import__('json').loads(Path(manifest_rows[0]['results_json']).read_text(encoding='utf-8'))
    initial_state = payload['episodes'][0]['state_trace'][0]
    assert len(initial_state['cell_summaries']) == 3
    assert len(initial_state['area_summaries']) == 2


def test_run_tree_runtime_v2_intercell_blocking_probe_case_writes_nonzero_blocking_metric() -> None:
    suite = build_tree_runtime_v2_intercell_blocking_probe_suite(PROJECT_ROOT)
    case = replace(suite[0], seeds=[301])
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2_intercell_blocking_probe'

    result = run_experiment_case(
        case=case,
        policies=[FlatTreePolicy()],
        output_dir=out_dir,
    )

    assert result.summary_csv.exists()
    row = result.rows[0]
    assert row['policy_name'] == 'tree_flat_mask_first'
    assert float(row['blocking_time_mean']) > 0.0
    assert float(row['buffer_wait_time_mean']) > 0.0


def test_run_tree_runtime_v2_intercell_blocking_probe_case_auto_advances_idle_epochs_for_framework_driver() -> None:
    from fms_bench.v2.framework_drivers.langgraph_driver import LangGraphTreeDriver

    suite = build_tree_runtime_v2_intercell_blocking_probe_suite(PROJECT_ROOT)
    case = replace(suite[0], seeds=[301])
    out_dir = (
        PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2_intercell_blocking_probe_framework'
    )

    result = run_experiment_case(
        case=case,
        policies=[LangGraphTreeDriver(strict_framework=False)],
        output_dir=out_dir,
    )

    assert result.summary_csv.exists()
    row = result.rows[0]
    assert row['policy_name'] == 'external_framework_langgraph'
    assert float(row['blocking_time_mean']) > 0.0


def test_run_brandimarte_intercell_case_auto_advances_idle_blocked_native_rollout() -> None:
    from fms_bench.tree_builder import build_tree_topology

    cell_configs = experiment_runner_module._build_tree_shared_backlog_cells(
        PROJECT_ROOT,
        'brandimarte_ordertrap_three_cell',
        'R0',
    )
    cell_configs = [
        replace(
            cell,
            inbound_buffer_cap=2,
            ingress_transport_time_hours=1.0 if cell.cell_id == 0 else 1.5,
            ingress_transport_energy_kwh=0.2 if cell.cell_id == 0 else 0.3,
        )
        for cell in cell_configs
    ]
    topology = build_tree_topology(cell_configs, area_count=2)
    case = HolonicExperimentCase(
        case_id='tree_runtime_v2_intercell_brandimarte_probe_native__tree_a2_c3__R0',
        suite_tier='tree_runtime_v2_intercell_probe',
        level='R0',
        benchmark_item_id='suite=tree_runtime_v2_intercell_probe__cells=brandimarte__k=2__b=2',
        cell_configs=cell_configs,
        seeds=[301],
        env_name='tree_runtime_v2_intercell',
        env_cls=experiment_runner_module._intercell_tree_runtime_env_factory(
            topology.area_to_cells,
            max_in_flight_jobs=2,
        ),
        regime_id='R0',
        topology_id=topology.topology_id,
    )
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2_intercell_brandimarte_native'

    result = run_experiment_case(
        case=case,
        policies=[FlatTreePolicy()],
        output_dir=out_dir,
    )

    assert result.summary_csv.exists()
    row = result.rows[0]
    assert row['policy_name'] == 'tree_flat_mask_first'
    assert float(row['activation_record_count_mean']) > 0.0


def test_run_brandimarte_intercell_case_auto_advances_idle_blocked_external_rollout() -> None:
    from fms_bench.tree_builder import build_tree_topology
    from fms_bench.v2.external_controller_adapter import InProcessPolicyExternalDriver

    cell_configs = experiment_runner_module._build_tree_shared_backlog_cells(
        PROJECT_ROOT,
        'brandimarte_ordertrap_three_cell',
        'R0',
    )
    cell_configs = [
        replace(
            cell,
            inbound_buffer_cap=2,
            ingress_transport_time_hours=1.0 if cell.cell_id == 0 else 1.5,
            ingress_transport_energy_kwh=0.2 if cell.cell_id == 0 else 0.3,
        )
        for cell in cell_configs
    ]
    topology = build_tree_topology(cell_configs, area_count=2)
    case = HolonicExperimentCase(
        case_id='tree_runtime_v2_intercell_brandimarte_probe_external__tree_a2_c3__R0',
        suite_tier='tree_runtime_v2_intercell_probe',
        level='R0',
        benchmark_item_id='suite=tree_runtime_v2_intercell_probe__cells=brandimarte__k=2__b=2__external=1',
        cell_configs=cell_configs,
        seeds=[301],
        env_name='tree_runtime_v2_intercell',
        env_cls=experiment_runner_module._intercell_tree_runtime_env_factory(
            topology.area_to_cells,
            max_in_flight_jobs=2,
        ),
        regime_id='R0',
        topology_id=topology.topology_id,
    )
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2_intercell_brandimarte_external'

    result = run_experiment_case(
        case=case,
        policies=[
            InProcessPolicyExternalDriver(
                FlatTreePolicy(),
                controller_name='external_tree_flat_mask_first_brandimarte_probe',
            )
        ],
        output_dir=out_dir,
    )

    assert result.summary_csv.exists()
    row = result.rows[0]
    assert row['controller_interface_mode'] == 'external_session'
    assert float(row['activation_record_count_mean']) > 0.0


def test_run_tree_runtime_v2_ladder_case_supports_five_cell_runtime() -> None:
    suite = build_tree_runtime_v2_ladder_suite(PROJECT_ROOT)
    case = next(
        item
        for item in suite
        if item.case_id == 'tree_runtime_v2_ladder__brandimarte_ordertrap_five_cell__tree_a3_c5__R2'
    )
    case = replace(case, seeds=[13])
    out_dir = PROJECT_ROOT / 'tests' / '_tmp_experiment_runner_tree_runtime_v2_ladder'

    result = run_experiment_case(
        case=case,
        policies=[FlatTreePolicy()],
        output_dir=out_dir,
    )

    assert result.summary_csv.exists()
    assert result.artifact_manifest.exists()
    row = result.rows[0]
    assert row['policy_name'] == 'tree_flat_mask_first'
    assert row['topology_id'] == 'tree_a3_c5'
    assert float(row['activation_record_count_mean']) > 0.0
    assert float(row['settlement_event_count_mean']) > 0.0

    with result.artifact_manifest.open('r', encoding='utf-8', newline='') as f:
        manifest_rows = list(csv.DictReader(f))
    assert len(manifest_rows) == 1
    payload = __import__('json').loads(Path(manifest_rows[0]['results_json']).read_text(encoding='utf-8'))
    initial_state = payload['episodes'][0]['state_trace'][0]
    assert len(initial_state['cell_summaries']) == 5
    assert len(initial_state['area_summaries']) == 3


def test_build_tree_runtime_v2_intercell_harder_congestion_candidates_suite_exposes_controlled_grid() -> None:
    suite = build_tree_runtime_v2_intercell_harder_congestion_candidates_suite(PROJECT_ROOT)

    assert len(suite) == 16
    assert {case.suite_tier for case in suite} == {
        'tree_runtime_v2_intercell_harder_congestion_candidates'
    }
    assert all(case.env_name == 'tree_runtime_v2_intercell' for case in suite)
    assert all(case.regime_id == 'R0' for case in suite)
    assert all(case.topology_id == 'tree_a2_c3' for case in suite)
    assert any('__k=3' in case.benchmark_item_id for case in suite)
    assert any('__tm=1.50' in case.benchmark_item_id for case in suite)
    assert any('__ib1=0__ib2=0' in case.benchmark_item_id for case in suite)
    assert any(
        any(config.ingress_transport_time_hours > 1.5 for config in case.cell_configs)
        for case in suite
    )


def test_build_tree_runtime_v2_intercell_harder_congestion_suite_is_subset_of_candidates() -> None:
    candidate_suite = build_tree_runtime_v2_intercell_harder_congestion_candidates_suite(PROJECT_ROOT)
    final_suite = build_tree_runtime_v2_intercell_harder_congestion_suite(PROJECT_ROOT)

    candidate_profile_ids = {case.case_id.split("__")[-1] for case in candidate_suite}
    final_profile_ids = {case.case_id.split("__")[-1] for case in final_suite}

    assert final_suite
    assert len(final_suite) <= len(candidate_suite)
    assert final_profile_ids.issubset(candidate_profile_ids)
    assert {case.suite_tier for case in final_suite} == {
        'tree_runtime_v2_intercell_harder_congestion'
    }


def test_framework_native_runner_supports_intercell_harder_congestion_suites() -> None:
    import examples.run_tree_runtime_v2_framework_native as framework_native_runner

    parsed = framework_native_runner._parse_args(
        [
            '--framework',
            'langgraph',
            '--suite',
            'intercell_harder_congestion_candidates',
        ]
    )
    parsed_final = framework_native_runner._parse_args(
        [
            '--framework',
            'langgraph',
            '--suite',
            'intercell_harder_congestion',
        ]
    )
    parsed_dynamic = framework_native_runner._parse_args(
        [
            '--framework',
            'langgraph',
            '--suite',
            'intercell_dynamics_extension',
        ]
    )

    assert parsed.suite == 'intercell_harder_congestion_candidates'
    assert parsed_final.suite == 'intercell_harder_congestion'
    assert parsed_dynamic.suite == 'intercell_dynamics_extension'
