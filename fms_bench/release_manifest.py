"""Machine-readable release manifest helpers for the DESBench source package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RELEASE_MANIFEST_VALIDATION_CONTRACT = "desbench_source_release_paths_and_suite_catalog"
RELEASE_CLAIM_BOUNDARY_CONTRACT = "source_code_and_smoke_tests_not_result_archive"
HEADLINE_RELEASE_SURFACE_CONTRACT = "a5c12_framework_backed_coordination_matrix"

OFFICIAL_A5C12_PROFILES = (
    "branch_pressure_k4_tm115_bk4",
    "cluster_pull_k4_tm120_bk4",
    "late_commit_soft_k4_tm125_bk4",
)
OFFICIAL_A5C12_AUTHORITY_MODES = (
    "centralized",
    "hierarchical",
    "heterarchical_cnp",
    "holonic_hybrid",
)
OFFICIAL_A5C12_FRAMEWORKS = (
    "agentscope",
    "langgraph",
)
OFFICIAL_A3C9_PROFILES = (
    "balanced_branching_anchor",
    "cross_area_tradeoff",
    "late_stage_reversal",
)
REQUIRED_SOURCE_RELEASE_PATHS = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    "docs/benchmark_overview.md",
    "docs/suites.md",
    "docs/a5c12_benchmark.md",
    "docs/a3c9_benchmark.md",
    "docs/paradigms.md",
    "docs/framework_adapters.md",
    "docs/extension_guide.md",
    "docs/reproducibility.md",
    "docs/benchmark_seed_protocol_v1.json",
    "examples/run_tree_runtime_v2_framework_native.py",
    "examples/run_case_shard.py",
    "examples/run_tree_runtime_v2_framework_case_shard.py",
    "fms_bench/experiment_runner.py",
    "fms_bench/v2/authority_modes.py",
    "fms_bench/v2/framework_drivers/base.py",
    "fms_bench/v2/framework_drivers/agentscope_driver.py",
    "fms_bench/v2/framework_drivers/langgraph_driver.py",
)


def build_tree_runtime_release_manifest(
    *,
    output_path: str | Path,
    release_date: str,
    benchmark_line: str = "a5c12-framework-backed",
    full_test_command: str = "pytest tests -q",
    last_verified_result: str = "not recorded in manifest",
) -> Path:
    """Write a DESBench source-release manifest.

    The historical function name is retained so older import paths keep working.
    The generated manifest describes this repository as a source package, not a
    bundled experiment-result archive.
    """

    repo_root = Path(__file__).resolve().parents[1]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "benchmark_name": "DESBench",
        "benchmark_line": benchmark_line,
        "release_tag": f"{benchmark_line}-r{release_date.replace('-', '')}",
        "release_date": str(release_date),
        "release_claim_boundary_contract": RELEASE_CLAIM_BOUNDARY_CONTRACT,
        "headline_release_surface_contract": HEADLINE_RELEASE_SURFACE_CONTRACT,
        "suite": {
            "suite_id": "intercell_a5c12_harder",
            "topology": "tree_a5_c12",
            "composition": "brandimarte_ordertrap_twelve_cell",
            "profiles": list(OFFICIAL_A5C12_PROFILES),
            "authority_modes": list(OFFICIAL_A5C12_AUTHORITY_MODES),
            "frameworks": list(OFFICIAL_A5C12_FRAMEWORKS),
            "case_count_per_model": (
                len(OFFICIAL_A5C12_PROFILES)
                * len(OFFICIAL_A5C12_AUTHORITY_MODES)
                * len(OFFICIAL_A5C12_FRAMEWORKS)
            ),
            "test_seed_count_per_case": 10,
        },
        "included_suites": {
            "intercell_a5c12_harder": {
                "suite_id": "intercell_a5c12_harder",
                "topology": "tree_a5_c12",
                "composition": "brandimarte_ordertrap_twelve_cell",
                "profiles": list(OFFICIAL_A5C12_PROFILES),
                "authority_modes": list(OFFICIAL_A5C12_AUTHORITY_MODES),
                "frameworks": list(OFFICIAL_A5C12_FRAMEWORKS),
                "case_count_per_model": (
                    len(OFFICIAL_A5C12_PROFILES)
                    * len(OFFICIAL_A5C12_AUTHORITY_MODES)
                    * len(OFFICIAL_A5C12_FRAMEWORKS)
                ),
                "role": "headline_release_surface",
            },
            "intercell_a3c9_wide": {
                "suite_id": "intercell_a3c9_wide",
                "topology": "tree_a3_c9",
                "composition": "brandimarte_ordertrap_nine_cell",
                "profiles": list(OFFICIAL_A3C9_PROFILES),
                "authority_modes": list(OFFICIAL_A5C12_AUTHORITY_MODES),
                "frameworks": list(OFFICIAL_A5C12_FRAMEWORKS),
                "case_count_per_model": (
                    len(OFFICIAL_A3C9_PROFILES)
                    * len(OFFICIAL_A5C12_AUTHORITY_MODES)
                    * len(OFFICIAL_A5C12_FRAMEWORKS)
                ),
                "role": "included_standard_suite",
            },
        },
        "public_entrypoints": {
            "smoke": "examples/run_tree_runtime_v2_framework_native.py",
            "case_shard": "examples/run_case_shard.py",
            "extension_guide": "docs/extension_guide.md",
        },
        "result_boundary": {
            "includes_result_csvs": False,
            "includes_raw_run_logs": False,
            "result_artifacts_expected_outside_repo": True,
        },
        "verification": {
            "full_test_command": str(full_test_command),
            "last_verified_result": str(last_verified_result),
            "release_manifest_validation_contract": RELEASE_MANIFEST_VALIDATION_CONTRACT,
        },
        "required_source_release_paths": list(REQUIRED_SOURCE_RELEASE_PATHS),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    validate_tree_runtime_release_manifest(manifest_path=output, repo_root=repo_root)
    return output


def validate_tree_runtime_release_manifest(
    *,
    manifest_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate that a generated manifest matches the checked-in source package."""

    repo_root_path = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))

    missing = [
        path
        for path in list(payload.get("required_source_release_paths", ()))
        if not (repo_root_path / str(path)).exists()
    ]
    if missing:
        raise ValueError("release manifest references missing paths: " + ", ".join(missing))

    suite = dict(payload.get("suite", {}))
    profiles = list(suite.get("profiles", ()))
    authority_modes = list(suite.get("authority_modes", ()))
    frameworks = list(suite.get("frameworks", ()))
    expected_case_count = len(profiles) * len(authority_modes) * len(frameworks)
    if int(suite.get("case_count_per_model", -1)) != expected_case_count:
        raise ValueError(
            "release manifest case_count_per_model does not match "
            "profiles x authority_modes x frameworks"
        )
    if str(suite.get("suite_id", "")) != "intercell_a5c12_harder":
        raise ValueError("release manifest suite_id must be intercell_a5c12_harder")
    included_suites = dict(payload.get("included_suites", {}))
    for suite_id, suite_payload in included_suites.items():
        included = dict(suite_payload)
        included_profiles = list(included.get("profiles", ()))
        included_modes = list(included.get("authority_modes", ()))
        included_frameworks = list(included.get("frameworks", ()))
        included_expected = len(included_profiles) * len(included_modes) * len(included_frameworks)
        if int(included.get("case_count_per_model", -1)) != included_expected:
            raise ValueError(
                f"included suite {suite_id} case_count_per_model does not match "
                "profiles x authority_modes x frameworks"
            )
    if bool(dict(payload.get("result_boundary", {})).get("includes_raw_run_logs", True)):
        raise ValueError("source release manifest must not include raw run logs")
    return payload
