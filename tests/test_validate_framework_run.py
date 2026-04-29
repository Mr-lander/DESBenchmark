from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_case_result_json(
    summary_dir: Path,
    *,
    case_id: str,
    policy_name: str,
    failure_kinds: list[str],
) -> None:
    case_dir = summary_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "episodes": [
            {
                "framework_decision_failure_trace": [
                    {
                        "agent_id": "cell_1",
                        "attempt": index,
                        "failure_kind": failure_kind,
                        "legal_actions": [0],
                    }
                    for index, failure_kind in enumerate(failure_kinds)
                ]
            }
        ]
    }
    (case_dir / f"{policy_name}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def test_validate_framework_run_accepts_real_llm_zero_fallback(tmp_path: Path) -> None:
    summary_csv = tmp_path / "suite_summary.csv"
    _write_summary(
        summary_csv,
        [
            {
                "suite_id": "tree_runtime_v2_wide",
                "framework_name": "LangGraph",
                "controller_interface_mode": "external_session",
                "framework_backend_mode": "native_framework",
                "framework_uv_project_environment": ".uv_envs/langgraph",
                "framework_llm_backend": "openai_compat",
                "framework_llm_decision_count_mean": "24.0",
                "framework_fallback_contract_decision_count_mean": "0.0",
                "framework_no_legal_action_count_mean": "0.0",
                "case_id": "case_a",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validate_framework_run.py"),
            "--summary-csv",
            str(summary_csv),
            "--expected-framework",
            "LangGraph",
            "--expected-suite",
            "wide",
            "--expected-output-dir",
            str(tmp_path),
            "--expected-uv-project-environment",
            ".uv_envs/langgraph",
            "--expected-llm-backend",
            "openai_compat",
            "--require-real-llm",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"real_llm_validated": true' in result.stdout


def test_validate_framework_run_rejects_fallback_when_real_llm_required(tmp_path: Path) -> None:
    summary_csv = tmp_path / "suite_summary.csv"
    _write_summary(
        summary_csv,
        [
            {
                "suite_id": "tree_runtime_v2_wide",
                "framework_name": "AgentScope",
                "controller_interface_mode": "external_session",
                "framework_backend_mode": "native_framework",
                "framework_uv_project_environment": ".uv_envs/agentscope",
                "framework_llm_backend": "openai_compat",
                "framework_llm_decision_count_mean": "12.0",
                "framework_fallback_contract_decision_count_mean": "1.0",
                "framework_no_legal_action_count_mean": "0.0",
                "case_id": "case_b",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validate_framework_run.py"),
            "--summary-csv",
            str(summary_csv),
            "--expected-framework",
            "AgentScope",
            "--expected-suite",
            "wide",
            "--expected-output-dir",
            str(tmp_path),
            "--expected-uv-project-environment",
            ".uv_envs/agentscope",
            "--expected-llm-backend",
            "openai_compat",
            "--require-real-llm",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "framework_fallback_contract_decision_count_mean must be 0" in result.stderr or (
        "framework_fallback_contract_decision_count_mean must be 0" in result.stdout
    )


def test_validate_framework_run_rejects_contract_stub_when_real_llm_required(tmp_path: Path) -> None:
    summary_csv = tmp_path / "suite_summary.csv"
    _write_summary(
        summary_csv,
        [
            {
                "suite_id": "tree_runtime_v2_wide",
                "framework_name": "LangGraph",
                "controller_interface_mode": "external_session",
                "framework_backend_mode": "contract_stub",
                "framework_uv_project_environment": ".uv_envs/langgraph",
                "framework_llm_backend": "openai_compat",
                "framework_llm_decision_count_mean": "24.0",
                "framework_fallback_contract_decision_count_mean": "0.0",
                "framework_no_legal_action_count_mean": "0.0",
                "case_id": "case_c",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validate_framework_run.py"),
            "--summary-csv",
            str(summary_csv),
            "--expected-framework",
            "LangGraph",
            "--expected-suite",
            "wide",
            "--expected-output-dir",
            str(tmp_path),
            "--expected-uv-project-environment",
            ".uv_envs/langgraph",
            "--expected-llm-backend",
            "openai_compat",
            "--require-real-llm",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "framework_backend_mode must be native_framework" in result.stderr or (
        "framework_backend_mode must be native_framework" in result.stdout
    )


def test_validate_framework_run_accepts_response_backed_invalid_response_fallback(
    tmp_path: Path,
) -> None:
    summary_csv = tmp_path / "suite_summary.csv"
    row = {
        "suite_id": "tree_runtime_v2_intercell_a3c9_wide",
        "framework_name": "LangGraph",
        "controller_interface_mode": "external_session",
        "policy_name": "external_framework_langgraph_llm",
        "framework_backend_mode": "native_framework",
        "framework_uv_project_environment": ".uv_envs/langgraph",
        "framework_llm_backend": "openai_compat",
        "framework_llm_decision_count_mean": "149.0",
        "framework_fallback_contract_decision_count_mean": "1.0",
        "framework_no_legal_action_count_mean": "0.0",
        "case_id": "case_response_backed_ok",
    }
    _write_summary(summary_csv, [row])
    _write_case_result_json(
        tmp_path,
        case_id=row["case_id"],
        policy_name=row["policy_name"],
        failure_kinds=["invalid_response"],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validate_framework_run.py"),
            "--summary-csv",
            str(summary_csv),
            "--expected-framework",
            "LangGraph",
            "--expected-suite",
            "intercell_a3c9_wide",
            "--expected-output-dir",
            str(tmp_path),
            "--expected-uv-project-environment",
            ".uv_envs/langgraph",
            "--expected-llm-backend",
            "openai_compat",
            "--require-real-llm-response-backed",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"real_llm_response_backed_validated": true' in result.stdout
    assert '"invalid_response": 1' in result.stdout


def test_validate_framework_run_rejects_response_backed_empty_response(tmp_path: Path) -> None:
    summary_csv = tmp_path / "suite_summary.csv"
    row = {
        "suite_id": "tree_runtime_v2_intercell_a3c9_wide",
        "framework_name": "LangGraph",
        "controller_interface_mode": "external_session",
        "policy_name": "external_framework_langgraph_llm",
        "framework_backend_mode": "native_framework",
        "framework_uv_project_environment": ".uv_envs/langgraph",
        "framework_llm_backend": "openai_compat",
        "framework_llm_decision_count_mean": "149.0",
        "framework_fallback_contract_decision_count_mean": "1.0",
        "framework_no_legal_action_count_mean": "0.0",
        "case_id": "case_response_backed_empty",
    }
    _write_summary(summary_csv, [row])
    _write_case_result_json(
        tmp_path,
        case_id=row["case_id"],
        policy_name=row["policy_name"],
        failure_kinds=["empty_response"],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validate_framework_run.py"),
            "--summary-csv",
            str(summary_csv),
            "--expected-framework",
            "LangGraph",
            "--expected-suite",
            "intercell_a3c9_wide",
            "--expected-output-dir",
            str(tmp_path),
            "--expected-uv-project-environment",
            ".uv_envs/langgraph",
            "--expected-llm-backend",
            "openai_compat",
            "--require-real-llm-response-backed",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "only ['invalid_response'] are allowed" in result.stderr or (
        "only ['invalid_response'] are allowed" in result.stdout
    )


def test_validate_framework_run_rejects_response_backed_exception(tmp_path: Path) -> None:
    summary_csv = tmp_path / "suite_summary.csv"
    row = {
        "suite_id": "tree_runtime_v2_intercell_a3c9_wide",
        "framework_name": "AgentScope",
        "controller_interface_mode": "external_session",
        "policy_name": "external_framework_agentscope_llm",
        "framework_backend_mode": "native_framework",
        "framework_uv_project_environment": ".uv_envs/agentscope",
        "framework_llm_backend": "openai_compat",
        "framework_llm_decision_count_mean": "145.0",
        "framework_fallback_contract_decision_count_mean": "1.0",
        "framework_no_legal_action_count_mean": "0.0",
        "case_id": "case_response_backed_exception",
    }
    _write_summary(summary_csv, [row])
    _write_case_result_json(
        tmp_path,
        case_id=row["case_id"],
        policy_name=row["policy_name"],
        failure_kinds=["exception"],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validate_framework_run.py"),
            "--summary-csv",
            str(summary_csv),
            "--expected-framework",
            "AgentScope",
            "--expected-suite",
            "intercell_a3c9_wide",
            "--expected-output-dir",
            str(tmp_path),
            "--expected-uv-project-environment",
            ".uv_envs/agentscope",
            "--expected-llm-backend",
            "openai_compat",
            "--require-real-llm-response-backed",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "only ['invalid_response'] are allowed" in result.stderr or (
        "only ['invalid_response'] are allowed" in result.stdout
    )


def test_validate_framework_run_rejects_response_backed_no_legal_action(tmp_path: Path) -> None:
    summary_csv = tmp_path / "suite_summary.csv"
    row = {
        "suite_id": "tree_runtime_v2_intercell_a3c9_wide",
        "framework_name": "LangGraph",
        "controller_interface_mode": "external_session",
        "policy_name": "external_framework_langgraph_llm",
        "framework_backend_mode": "native_framework",
        "framework_uv_project_environment": ".uv_envs/langgraph",
        "framework_llm_backend": "openai_compat",
        "framework_llm_decision_count_mean": "149.0",
        "framework_fallback_contract_decision_count_mean": "0.0",
        "framework_no_legal_action_count_mean": "1.0",
        "case_id": "case_response_backed_no_legal",
    }
    _write_summary(summary_csv, [row])
    _write_case_result_json(
        tmp_path,
        case_id=row["case_id"],
        policy_name=row["policy_name"],
        failure_kinds=[],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validate_framework_run.py"),
            "--summary-csv",
            str(summary_csv),
            "--expected-framework",
            "LangGraph",
            "--expected-suite",
            "intercell_a3c9_wide",
            "--expected-output-dir",
            str(tmp_path),
            "--expected-uv-project-environment",
            ".uv_envs/langgraph",
            "--expected-llm-backend",
            "openai_compat",
            "--require-real-llm-response-backed",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "framework_no_legal_action_count_mean must be 0" in result.stderr or (
        "framework_no_legal_action_count_mean must be 0" in result.stdout
    )
