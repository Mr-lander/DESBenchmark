"""Machine-readable benchmark release manifest builder."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .authority_profiles import list_numeric_profiles
from .config import (
    ARRIVAL_PROFILE_METADATA_FIELDS,
    DUE_DATE_PROFILE_METADATA_FIELDS,
    NUMERIC_PROFILE_METADATA_FIELDS,
    SETUP_PROFILE_METADATA_FIELDS,
)
from .arrival_profiles import list_arrival_profiles
from .due_date_profiles import list_due_date_profiles
from .setup_profiles import list_setup_profiles
from .experiment_runner import (
    CASE_BUDGET_ARTIFACT_FILENAME,
    CASE_BUDGET_METADATA_FIELDS,
    DEFAULT_CASE_BUDGET_PROVENANCE_CLASS,
    DEFAULT_CASE_BUDGET_SCOPE,
    DEFAULT_CASE_BUDGET_SOURCE_REF,
)
from .holonic_evaluation import (
    EVALUATOR_SUMMARY_CONTRACT,
    LEGACY_SUMMARY_METRIC_CONTRACTS,
    STRUCTURED_VIOLATION_SUMMARY_FIELDS,
    SUITE_SUMMARY_SETTLEMENT_METRICS,
    SUITE_SUMMARY_STRUCTURED_VIOLATION_METRICS,
    TARDINESS_METRIC_CONTRACTS,
)
from .v2.controller_payloads import (
    PUBLIC_CONTROLLER_PAYLOAD_BASE_FIELDS,
    PUBLIC_CONTROLLER_PAYLOAD_ROLE_ALIAS_FIELDS,
)
from .v2.external_controller_adapter import (
    EXTERNAL_ACTION_VALIDATION_CONTRACT,
    PUBLIC_ARTIFACT_PAYLOAD_FIELDS,
    PUBLIC_DECISION_PAYLOAD_FIELDS,
    PUBLIC_FAIRNESS_SCOPE_CONTRACT,
)
from .v2.framework_drivers.base import (
    FRAMEWORK_ACTION_INPUT_CONTRACT,
    FRAMEWORK_PAYLOAD_AUDIT_STATUS,
)
from .v2.runtime_ledger import (
    REWARD_AUTHORITY_ID,
    SETTLEMENT_SEMANTICS_CONTRACT,
    SETTLEMENT_TRACE_CONTRACT,
)
from .v2.task_semantics import (
    IMPLEMENTED_PROPAGATION_ACTIONS,
    PUBLIC_WORK_ITEM_CONTRACT,
    RESERVED_PROPAGATION_ACTIONS,
    STABLE_PUBLIC_COMMITMENT_STATUSES,
)
from .v2.tree_runtime_env import TreeRuntimeEnv

RELEASE_MANIFEST_VALIDATION_CONTRACT = (
    "required_paths_summary_counts_and_composed_surface_consistency"
)
RELEASE_CLAIM_BOUNDARY_CONTRACT = (
    "reproducible_release_machinery_not_automatic_paper_strength"
)
HEADLINE_RELEASE_SURFACE_CONTRACT = (
    "wide_controller_frontier_checked_in_composed_test_surface"
)


def build_tree_runtime_release_manifest(
    *,
    output_path: str | Path,
    release_date: str,
    benchmark_line: str = "tree-runtime-v2",
    full_test_command: str = "pytest tests -q",
    last_verified_result: str = "286 passed",
) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    release_tag = f"{benchmark_line}-r{release_date.replace('-', '')}"
    suite_specs = [
        {
            "suite_id": "tree_runtime_v2_small",
            "surface_kind": "core_regression",
            "split_id": "test",
            "summary_path": "docs/sample_results/tree_runtime_v2_small_suite_summary_20260324.csv",
        },
        {
            "suite_id": "tree_runtime_v2_pilot",
            "surface_kind": "core_dev_matrix",
            "split_id": "dev",
            "summary_path": "docs/sample_results/tree_runtime_v2_pilot_suite_summary_20260324.csv",
        },
        {
            "suite_id": "tree_runtime_v2_wide",
            "surface_kind": "core_locked_test_matrix",
            "split_id": "test",
            "summary_path": "docs/sample_results/tree_runtime_v2_wide_suite_summary_20260324.csv",
        },
        {
            "suite_id": "tree_runtime_v2_ladder",
            "surface_kind": "difficulty_ladder",
            "split_id": "test",
            "summary_path": "docs/sample_results/tree_runtime_v2_ladder_suite_summary_20260325.csv",
        },
        {
            "suite_id": "tree_runtime_v2_pilot_controller_frontier",
            "surface_kind": "controller_frontier",
            "split_id": "dev",
            "summary_path": "docs/sample_results/tree_runtime_v2_pilot_controller_frontier_suite_summary_20260327.csv",
            "derived_from": [
                "docs/sample_results/tree_runtime_v2_pilot_external_family_suite_summary_20260326.csv",
                "docs/sample_results/tree_runtime_v2_pilot_framework_native_suite_summary_20260327.csv",
            ],
        },
        {
            "suite_id": "tree_runtime_v2_wide_framework_native",
            "surface_kind": "framework_native_extension",
            "split_id": "test",
            "summary_path": "docs/sample_results/tree_runtime_v2_wide_framework_native_suite_summary_20260327.csv",
        },
        {
            "suite_id": "tree_runtime_v2_wide_controller_frontier",
            "surface_kind": "controller_frontier",
            "split_id": "test",
            "summary_path": "docs/sample_results/tree_runtime_v2_wide_controller_frontier_suite_summary_20260327.csv",
            "derived_from": [
                "docs/sample_results/tree_runtime_v2_wide_external_family_suite_summary_20260325.csv",
                "docs/sample_results/tree_runtime_v2_wide_framework_native_suite_summary_20260327.csv",
            ],
        },
    ]
    required_checked_in_artifacts = [
        "docs/tree_runtime_system_map_v2.html",
        "docs/tree_runtime_reproduction_v1.md",
        "docs/tree_runtime_release_versioning_v1.md",
        "docs/tree_runtime_framework_envs_v1.md",
        "docs/tree_runtime_controller_frontier_snapshot_20260327.md",
        "docs/2026-03-27-controller_frontier_release_sufficiency_note.md",
        "docs/2026-03-27-release_comparison_narrative_freeze.md",
        "docs/2026-03-27-des_and_accounting_provenance_matrix.md",
        "docs/sample_results/tree_runtime_v2_pilot_external_family_suite_summary_20260326.csv",
        "docs/sample_results/tree_runtime_v2_wide_suite_summary_20260324.csv",
        "docs/sample_results/tree_runtime_v2_wide_external_family_suite_summary_20260325.csv",
        "docs/sample_results/tree_runtime_v2_ladder_suite_summary_20260325.csv",
        "docs/sample_results/tree_runtime_v2_wide_framework_native_suite_summary_20260327.csv",
        "docs/sample_results/tree_runtime_v2_pilot_controller_frontier_suite_summary_20260327.csv",
        "docs/sample_results/tree_runtime_v2_pilot_controller_frontier_publication_bundle_20260327/paper_report.html",
        "docs/sample_results/tree_runtime_v2_wide_controller_frontier_suite_summary_20260327.csv",
        "docs/sample_results/tree_runtime_v2_wide_controller_frontier_publication_bundle_20260327/paper_report.html",
    ]
    _require_paths_exist(repo_root, required_checked_in_artifacts)

    suite_surfaces = [
        _build_suite_surface(repo_root=repo_root, spec=spec)
        for spec in suite_specs
    ]
    payload = {
        "benchmark_line": benchmark_line,
        "release_tag": release_tag,
        "release_date": release_date,
        "suite_surfaces": suite_surfaces,
        "current_runtime_truth": _build_current_runtime_truth(suite_surfaces),
        "reproduction_paths": {
            "core": {
                "entrypoint": "docs/tree_runtime_reproduction_v1.md",
                "canonical_commands": [
                    "python examples/run_tree_runtime_v2_wide.py",
                    "python examples/run_tree_runtime_v2_external_family_wide.py",
                    "python examples/build_tree_runtime_paper_artifacts.py",
                ],
            },
            "framework_native_extension": {
                "entrypoint": "docs/tree_runtime_framework_envs_v1.md",
                "canonical_commands": [
                    "uv run --extra framework_agentscope python examples/run_tree_runtime_v2_framework_native.py --framework agentscope --suite small --strict-framework",
                    "python examples/build_tree_runtime_controller_frontier_artifacts.py",
                ],
            },
        },
        "release_surface_decision": {
            "headline_controller_frontier_suite_id": "tree_runtime_v2_wide_controller_frontier",
            "supporting_controller_frontier_suite_id": "tree_runtime_v2_pilot_controller_frontier",
            "monolithic_twelve_policy_rerun_required": False,
            "release_claim_boundary_contract": RELEASE_CLAIM_BOUNDARY_CONTRACT,
            "headline_release_surface_contract": HEADLINE_RELEASE_SURFACE_CONTRACT,
            "decision_note_path": "docs/2026-03-27-controller_frontier_release_sufficiency_note.md",
            "narrative_freeze_path": "docs/2026-03-27-release_comparison_narrative_freeze.md",
        },
        "verification": {
            "full_test_command": full_test_command,
            "last_verified_result": last_verified_result,
            "release_manifest_validation_contract": RELEASE_MANIFEST_VALIDATION_CONTRACT,
        },
        "required_checked_in_artifacts": required_checked_in_artifacts,
        "known_limitations": [
            "native framework evidence should be rerun in isolated uv environments and one framework per process for release-facing claims",
            "the checked-in pilot controller frontier remains supporting dev evidence and is a reproducible composed freeze rather than one monolithic release-grade serial twelve-policy rerun",
            "the checked-in wide controller frontier is the release-facing comparison surface for the current line, but it is still a reproducible composed freeze built from frozen external and native summaries",
            "the checked-in wide framework-native line includes one recovered SPADE subline rebuilt from completed case summaries after a late suite-tail teardown failure",
        ],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    validate_tree_runtime_release_manifest(manifest_path=output, repo_root=repo_root)
    return output


def validate_tree_runtime_release_manifest(
    *,
    manifest_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    repo_root_path = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))

    required_artifacts = list(payload.get("required_checked_in_artifacts", ()))
    _require_paths_exist(repo_root_path, required_artifacts)

    suite_surfaces = list(payload.get("suite_surfaces", ()))
    if not suite_surfaces:
        raise ValueError("release manifest must contain non-empty suite_surfaces")
    by_suite_id = {
        str(surface["suite_id"]): dict(surface)
        for surface in suite_surfaces
    }

    for surface in suite_surfaces:
        _validate_surface_snapshot(
            repo_root=repo_root_path,
            surface=dict(surface),
        )

    release_surface_decision = dict(payload.get("release_surface_decision", {}))
    headline_surface_id = str(release_surface_decision.get("headline_controller_frontier_suite_id", ""))
    supporting_surface_id = str(
        release_surface_decision.get("supporting_controller_frontier_suite_id", "")
    )
    if headline_surface_id not in by_suite_id:
        raise ValueError(f"headline release surface missing from manifest suite_surfaces: {headline_surface_id}")
    if supporting_surface_id not in by_suite_id:
        raise ValueError(
            f"supporting release surface missing from manifest suite_surfaces: {supporting_surface_id}"
        )

    headline_surface = by_suite_id[headline_surface_id]
    if str(headline_surface.get("surface_kind")) != "controller_frontier" or str(
        headline_surface.get("split_id")
    ) != "test":
        raise ValueError(
            "headline release surface must be a locked test controller_frontier surface"
        )

    supporting_surface = by_suite_id[supporting_surface_id]
    if str(supporting_surface.get("surface_kind")) != "controller_frontier" or str(
        supporting_surface.get("split_id")
    ) != "dev":
        raise ValueError(
            "supporting release surface must be a dev controller_frontier surface"
        )

    for surface in suite_surfaces:
        if str(surface.get("surface_kind")) != "controller_frontier":
            continue
        _validate_controller_frontier_composition(
            repo_root=repo_root_path,
            surface=dict(surface),
        )

    return {
        "validation_contract": RELEASE_MANIFEST_VALIDATION_CONTRACT,
        "headline_release_surface_id": headline_surface_id,
        "supporting_release_surface_id": supporting_surface_id,
        "checks": {
            "required_artifacts_exist": True,
            "suite_summary_counts_match": True,
            "suite_summary_schema_match": True,
            "headline_surface_declared_and_present": True,
            "headline_surface_is_locked_test_controller_frontier": True,
            "supporting_surface_is_dev_controller_frontier": True,
            "controller_frontier_composition_matches_derived_sources": True,
        },
    }


def _build_suite_surface(*, repo_root: Path, spec: dict[str, object]) -> dict[str, object]:
    _validate_suite_spec(spec)
    summary_rel = str(spec["summary_path"])
    summary_path = repo_root / summary_rel
    _require_paths_exist(repo_root, [summary_rel])
    rows = _load_rows(summary_path)
    surface = {
        "suite_id": spec["suite_id"],
        "surface_kind": spec["surface_kind"],
        "split_id": spec["split_id"],
        "summary_path": summary_rel,
        "summary_row_count": len(rows),
        "case_count": len({row["case_id"] for row in rows}),
        "controller_count": len({row["policy_name"] for row in rows}),
        "suite_tier": _stable_single_value(rows, "suite_tier"),
        "controller_interface_modes": sorted(
            {row.get("controller_interface_mode", "in_process") for row in rows}
        ),
        "summary_schema": _build_summary_schema(rows),
    }
    derived_from = spec.get("derived_from")
    if derived_from is not None:
        _require_paths_exist(repo_root, list(derived_from))
        surface["derived_from"] = list(derived_from)
    return surface


def _validate_suite_spec(spec: dict[str, object]) -> None:
    surface_kind = str(spec.get("surface_kind", ""))
    suite_id = str(spec.get("suite_id", ""))
    derived_from = spec.get("derived_from")

    if surface_kind == "controller_frontier":
        if not isinstance(derived_from, list) or len(derived_from) == 0:
            raise ValueError(
                f"{suite_id} controller_frontier suite spec must include non-empty derived_from"
            )
        return

    if surface_kind == "framework_native_extension":
        if derived_from is not None:
            raise ValueError(
                f"{suite_id} framework_native_extension suite spec must not include derived_from"
            )


def _validate_surface_snapshot(*, repo_root: Path, surface: dict[str, object]) -> None:
    summary_rel = str(surface["summary_path"])
    _require_paths_exist(repo_root, [summary_rel])
    rows = _load_rows(repo_root / summary_rel)

    expected_row_count = int(surface["summary_row_count"])
    actual_row_count = len(rows)
    if actual_row_count != expected_row_count:
        raise ValueError(
            f"suite surface row count mismatch for {surface['suite_id']}: {actual_row_count} != {expected_row_count}"
        )

    expected_case_count = int(surface["case_count"])
    actual_case_count = len({row["case_id"] for row in rows})
    if actual_case_count != expected_case_count:
        raise ValueError(
            f"suite surface case count mismatch for {surface['suite_id']}: {actual_case_count} != {expected_case_count}"
        )

    expected_controller_count = int(surface["controller_count"])
    actual_controller_count = len({row["policy_name"] for row in rows})
    if actual_controller_count != expected_controller_count:
        raise ValueError(
            f"suite surface controller count mismatch for {surface['suite_id']}: {actual_controller_count} != {expected_controller_count}"
        )

    expected_modes = sorted(str(mode) for mode in surface["controller_interface_modes"])
    actual_modes = sorted({row.get("controller_interface_mode", "in_process") for row in rows})
    if actual_modes != expected_modes:
        raise ValueError(
            f"suite surface controller interface mode mismatch for {surface['suite_id']}: "
            f"{actual_modes} != {expected_modes}"
        )

    expected_summary_schema = dict(surface["summary_schema"])
    actual_summary_schema = _build_summary_schema(rows)
    if actual_summary_schema != expected_summary_schema:
        raise ValueError(
            f"summary schema mismatch for {surface['suite_id']}: "
            f"{actual_summary_schema} != {expected_summary_schema}"
        )


def _validate_controller_frontier_composition(*, repo_root: Path, surface: dict[str, object]) -> None:
    derived_from = list(surface.get("derived_from", ()))
    if not derived_from:
        raise ValueError(f"controller frontier surface missing derived_from: {surface['suite_id']}")

    surface_rows = _load_rows(repo_root / str(surface["summary_path"]))
    derived_rows: list[dict[str, str]] = []
    for relative_path in derived_from:
        _require_paths_exist(repo_root, [str(relative_path)])
        derived_rows.extend(_load_rows(repo_root / str(relative_path)))

    surface_row_keys = {_controller_frontier_row_key(row) for row in surface_rows}
    derived_row_keys = {_controller_frontier_row_key(row) for row in derived_rows}
    surface_policy_names = {row["policy_name"] for row in surface_rows}
    derived_policy_names = {row["policy_name"] for row in derived_rows}
    surface_case_ids = {row["case_id"] for row in surface_rows}
    derived_case_ids = {row["case_id"] for row in derived_rows}

    if surface_row_keys != derived_row_keys:
        raise ValueError(
            f"controller frontier composition mismatch for {surface['suite_id']}: "
            "surface rows do not match the unique union of derived sources"
        )
    if surface_policy_names != derived_policy_names:
        raise ValueError(
            f"controller frontier composition mismatch for {surface['suite_id']}: "
            "surface policy roster does not match derived sources"
        )
    if surface_case_ids != derived_case_ids:
        raise ValueError(
            f"controller frontier composition mismatch for {surface['suite_id']}: "
            "surface case roster does not match derived sources"
        )


def _controller_frontier_row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("case_id", "")),
        str(row.get("policy_name", "")),
        str(row.get("controller_interface_mode", "")),
    )


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _build_summary_schema(rows: list[dict[str, str]]) -> dict[str, object]:
    columns = set(rows[0].keys()) if rows else set()
    return {
        "columns": sorted(columns),
        "has_case_budget_metadata": all(
            metric in columns for metric in CASE_BUDGET_METADATA_FIELDS
        ),
        "has_structured_violation_metrics": all(
            metric in columns for metric in SUITE_SUMMARY_STRUCTURED_VIOLATION_METRICS
        ),
        "has_settlement_closure_metrics": all(
            metric in columns for metric in SUITE_SUMMARY_SETTLEMENT_METRICS
        ),
        "has_execution_mode_metrics": all(
            metric in columns for metric in ("execution_mode", "supported_execution_modes")
        ),
    }


def _build_current_runtime_truth(
    suite_surfaces: list[dict[str, object]],
) -> dict[str, object]:
    headline_release_surface_id = "tree_runtime_v2_wide_controller_frontier"
    return {
        "reward_authority": REWARD_AUTHORITY_ID,
        "settlement_trace_contract": SETTLEMENT_TRACE_CONTRACT,
        "settlement_semantics_contract": SETTLEMENT_SEMANTICS_CONTRACT,
        "evaluator_summary_contract": EVALUATOR_SUMMARY_CONTRACT,
        "tardiness_metric_contracts": dict(TARDINESS_METRIC_CONTRACTS),
        "supported_execution_modes": TreeRuntimeEnv.declared_supported_execution_modes(),
        "decision_payload_fields": list(PUBLIC_DECISION_PAYLOAD_FIELDS),
        "artifact_payload_fields": list(PUBLIC_ARTIFACT_PAYLOAD_FIELDS),
        "controller_payload_base_fields": list(PUBLIC_CONTROLLER_PAYLOAD_BASE_FIELDS),
        "controller_payload_role_alias_fields": {
            role: list(fields)
            for role, fields in PUBLIC_CONTROLLER_PAYLOAD_ROLE_ALIAS_FIELDS.items()
        },
        "external_action_validation_contract": EXTERNAL_ACTION_VALIDATION_CONTRACT,
        "public_fairness_scope_contract": PUBLIC_FAIRNESS_SCOPE_CONTRACT,
        "framework_action_input_contract": FRAMEWORK_ACTION_INPUT_CONTRACT,
        "framework_payload_audit_status": FRAMEWORK_PAYLOAD_AUDIT_STATUS,
        "public_work_item_contract": PUBLIC_WORK_ITEM_CONTRACT,
        "implemented_propagation_actions": [
            action.value for action in IMPLEMENTED_PROPAGATION_ACTIONS
        ],
        "reserved_propagation_actions": [
            action.value for action in RESERVED_PROPAGATION_ACTIONS
        ],
        "stable_public_commitment_statuses": [
            status.value for status in STABLE_PUBLIC_COMMITMENT_STATUSES
        ],
        "runtime_state_kind_contract": TreeRuntimeEnv.RUNTIME_STATE_KIND_CONTRACT,
        "runtime_layer_contract": TreeRuntimeEnv.RUNTIME_LAYER_CONTRACT,
        "shared_des_runtime_boundary_contract": (
            TreeRuntimeEnv.SHARED_DES_RUNTIME_BOUNDARY_CONTRACT
        ),
        "runtime_agent_boundary_contract": TreeRuntimeEnv.RUNTIME_AGENT_BOUNDARY_CONTRACT,
        "public_epoch_semantics_contract": TreeRuntimeEnv.PUBLIC_EPOCH_SEMANTICS_CONTRACT,
        "public_role_mapping": TreeRuntimeEnv.benchmark_role_mapping(),
        "public_role_mapping_contract": TreeRuntimeEnv.PUBLIC_ROLE_MAPPING_CONTRACT,
        "literature_mapping_boundary_contract": TreeRuntimeEnv.LITERATURE_MAPPING_BOUNDARY_CONTRACT,
        "machine_power_model_contract": "benchmark_wide_busy_idle_scalar_model",
        "case_budget_metadata_fields": list(CASE_BUDGET_METADATA_FIELDS),
        "case_budget_metadata_defaults": {
            "case_budget_provenance_class": DEFAULT_CASE_BUDGET_PROVENANCE_CLASS,
            "case_budget_scope": DEFAULT_CASE_BUDGET_SCOPE,
            "case_budget_source_ref": DEFAULT_CASE_BUDGET_SOURCE_REF,
        },
        "case_budget_artifact_filename": CASE_BUDGET_ARTIFACT_FILENAME,
        "legacy_summary_metric_contracts": dict(LEGACY_SUMMARY_METRIC_CONTRACTS),
        "structured_violation_summary_fields": list(STRUCTURED_VIOLATION_SUMMARY_FIELDS),
        "suite_summary_structured_violation_metrics": list(
            SUITE_SUMMARY_STRUCTURED_VIOLATION_METRICS
        ),
        "suite_summary_settlement_metrics": list(SUITE_SUMMARY_SETTLEMENT_METRICS),
        "numeric_profile_metadata_fields": list(NUMERIC_PROFILE_METADATA_FIELDS),
        "available_numeric_profiles": list_numeric_profiles(),
        "due_date_profile_metadata_fields": list(DUE_DATE_PROFILE_METADATA_FIELDS),
        "available_due_date_profiles": list_due_date_profiles(),
        "setup_profile_metadata_fields": list(SETUP_PROFILE_METADATA_FIELDS),
        "available_setup_profiles": list_setup_profiles(),
        "arrival_profile_metadata_fields": list(ARRIVAL_PROFILE_METADATA_FIELDS),
        "available_arrival_profiles": list_arrival_profiles(),
        "checked_in_snapshot_alignment": {
            "release_snapshot_scope": "checked_in_sample_results_are_historical_release_surfaces",
            "headline_release_surface_id": headline_release_surface_id,
            "snapshot_surfaces_cover_current_structured_violation_metrics": all(
                bool(surface["summary_schema"]["has_structured_violation_metrics"])
                for surface in suite_surfaces
            ),
            "snapshot_surfaces_cover_current_settlement_closure_metrics": all(
                bool(surface["summary_schema"]["has_settlement_closure_metrics"])
                for surface in suite_surfaces
            ),
            "snapshot_surfaces_cover_current_execution_mode_metrics": all(
                bool(surface["summary_schema"]["has_execution_mode_metrics"])
                for surface in suite_surfaces
            ),
            "snapshot_surfaces_cover_current_case_budget_metadata": all(
                bool(surface["summary_schema"]["has_case_budget_metadata"])
                for surface in suite_surfaces
            ),
        },
    }


def _stable_single_value(rows: list[dict[str, str]], key: str) -> str:
    values = {row[key] for row in rows}
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def _require_paths_exist(repo_root: Path, relative_paths: list[str]) -> None:
    missing = [rel for rel in relative_paths if not (repo_root / rel).exists()]
    if missing:
        raise FileNotFoundError(f"missing required release-manifest paths: {', '.join(missing)}")

