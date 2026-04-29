from __future__ import annotations

import json


def test_a5c12_release_manifest_builds_and_validates(tmp_path) -> None:
    from fms_bench.release_manifest import (
        build_tree_runtime_release_manifest,
        validate_tree_runtime_release_manifest,
    )

    manifest_path = build_tree_runtime_release_manifest(
        output_path=tmp_path / "manifest.json",
        release_date="2026-04-29",
        full_test_command="python -m pytest tests/test_release_manifest.py -q",
        last_verified_result="1 passed",
    )

    payload = validate_tree_runtime_release_manifest(manifest_path=manifest_path)

    assert payload["release_claim_boundary_contract"] == "source_code_and_smoke_tests_not_result_archive"
    assert payload["benchmark_name"] == "DESBench"
    assert payload["suite"]["suite_id"] == "intercell_a5c12_harder"
    assert payload["suite"]["case_count_per_model"] == 24
    assert payload["included_suites"]["intercell_a3c9_wide"] == {
        "suite_id": "intercell_a3c9_wide",
        "topology": "tree_a3_c9",
        "composition": "brandimarte_ordertrap_nine_cell",
        "profiles": [
            "balanced_branching_anchor",
            "cross_area_tradeoff",
            "late_stage_reversal",
        ],
        "authority_modes": [
            "centralized",
            "hierarchical",
            "mediated_heterarchical_cnp",
            "holonic_hybrid",
        ],
        "frameworks": ["agentscope", "langgraph"],
        "case_count_per_model": 24,
        "role": "included_standard_suite",
    }
    assert "docs/a3c9_benchmark.md" in payload["required_source_release_paths"]
    assert payload["result_boundary"]["includes_result_csvs"] is False
    assert payload["result_boundary"]["includes_raw_run_logs"] is False
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == payload
