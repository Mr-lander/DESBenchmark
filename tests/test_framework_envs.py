from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_framework_env_registry_exposes_core_uv_profiles() -> None:
    from fms_bench.v2.framework_envs import FRAMEWORK_ENV_SPECS, get_framework_env_spec

    assert sorted(FRAMEWORK_ENV_SPECS) == [
        "ag2",
        "agentscope",
        "crewai",
        "google_adk",
        "langgraph",
        "spade",
    ]

    agentscope = get_framework_env_spec("agentscope")
    assert agentscope.framework_name == "AgentScope"
    assert agentscope.python_package == "agentscope"
    assert agentscope.optional_dependency == "framework_agentscope"
    assert agentscope.uv_project_environment == ".uv_envs/agentscope"

    assert agentscope.sync_environment() == {
        "UV_PROJECT_ENVIRONMENT": ".uv_envs/agentscope",
    }
    assert agentscope.recommended_uv_sync_args(include_test=True) == [
        "uv",
        "sync",
        "--extra",
        "framework_agentscope",
        "--extra",
        "test",
    ]


def test_framework_driver_runtime_info_reports_isolated_uv_profile() -> None:
    from fms_bench.v2.framework_drivers import CrewAITreeDriver

    driver = CrewAITreeDriver()

    assert driver.framework_runtime_info["framework_name"] == "CrewAI"
    assert driver.framework_runtime_info["python_package"] == "crewai"
    assert driver.framework_runtime_info["optional_dependency"] == "framework_crewai"
    assert driver.framework_runtime_info["uv_project_environment"] == ".uv_envs/crewai"
