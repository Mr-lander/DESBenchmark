from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from fms_bench import PlantCellConfig
from fms_bench.experiment_runner import HolonicExperimentCase, run_experiment_case
from fms_bench.v2.framework_drivers import AgentScopeLLMTreeDriver, LangGraphLLMTreeDriver
from fms_bench.v2.tree_runtime_env import TreeRuntimeEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_agentscope_llm_driver_exposes_decision_audit_payload() -> None:
    class _TextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Response:
        def __init__(self, text: str) -> None:
            self.content = [_TextBlock(text)]

    class _AsyncModel:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, messages: object) -> object:
            del messages
            self.calls += 1
            if self.calls == 1:
                return _Response('{"action": 1}')
            return _Response("not-json")

    driver = AgentScopeLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
        request_retries=0,
    )
    driver._llm_model = _AsyncModel()

    llm_action = asyncio.run(
        driver._select_llm_action_async(
            agent_id="plant",
            agent_view={"role": "plant", "action_mask": [0, 1]},
        )
    )
    fallback_action = asyncio.run(
        driver._select_llm_action_async(
            agent_id="plant",
            agent_view={"role": "plant", "action_mask": [1, 0]},
        )
    )

    assert llm_action == 1
    assert fallback_action == 0
    assert driver.framework_decision_audit_info["framework_llm_decision_count"] == 1
    assert driver.framework_decision_audit_info["framework_fallback_contract_decision_count"] == 1
    assert driver.framework_decision_audit_info["framework_no_legal_action_count"] == 0
    assert driver.framework_decision_audit_info["framework_decision_failure_trace"] == [
        {
            "agent_id": "plant",
            "attempt": 0,
            "failure_kind": "invalid_response",
            "legal_actions": [0],
            "raw_response_excerpt": "not-json",
            "raw_response_length": 8,
        }
    ]


def test_langgraph_llm_case_surfaces_decision_audit_in_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flex_root = PROJECT_ROOT / "external" / "benchmarks" / "flexible jobshop" / "instances"
    common = {
        "family_key": "kacem",
        "instance_path": flex_root / "KacemHammadiBorne2002" / "kacem1.txt",
        "instance_kind": "kacem",
        "wip_cap": 2,
    }
    case = HolonicExperimentCase(
        case_id="tree_runtime_v2_small__kacem_pair_langgraph_llm_audit__R0",
        suite_tier="tree_runtime_v2_small",
        level="R0",
        benchmark_item_id="suite=tree_runtime_v2_small__regime=R0__topology=tree_a2_c2__cells=kacem_pair__split=test",
        cell_configs=[
            PlantCellConfig(cell_id=0, label="cell-a", processing_time_scale=0.9, **common),
            PlantCellConfig(cell_id=1, label="cell-b", processing_time_scale=1.1, **common),
        ],
        seeds=[13],
        env_name="tree_runtime_v2",
        env_cls=lambda cell_configs, config, seed: TreeRuntimeEnv(
            cell_configs,
            area_to_cells={0: [0], 1: [1]},
            config=config,
            seed=seed,
        ),
        regime_id="R0",
        topology_id="tree_a2_c2",
    )
    driver = LangGraphLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
        request_retries=0,
    )

    class _ResponseMessage:
        content = "not-json"

    class _ResponseChoice:
        message = _ResponseMessage()

    class _Response:
        choices = [_ResponseChoice()]

    class _FakeCompletions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            return _Response()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(driver, "_llm_client", _FakeClient())

    result = run_experiment_case(
        case=case,
        policies=[driver],
        output_dir=tmp_path / "tree_runtime_v2_langgraph_llm_audit",
    )

    row = result.rows[0]
    assert float(row["framework_llm_decision_count_mean"]) == 0.0
    assert float(row["framework_fallback_contract_decision_count_mean"]) > 0.0
    assert float(row["framework_no_legal_action_count_mean"]) == 0.0

    json_path = tmp_path / "tree_runtime_v2_langgraph_llm_audit" / case.case_id / "external_framework_langgraph_llm.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    episode = payload["episodes"][0]
    assert episode["framework_llm_decision_count"] == 0
    assert episode["framework_fallback_contract_decision_count"] > 0
    assert episode["framework_no_legal_action_count"] == 0
    assert episode["framework_decision_source_trace"]
    assert episode["framework_decision_failure_trace"]
    assert episode["framework_decision_failure_trace"][0]["failure_kind"] == "invalid_response"
    assert {entry["source"] for entry in episode["framework_decision_source_trace"]} == {
        "fallback_contract"
    }


def test_run_experiment_case_rejects_fallback_when_zero_fallback_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flex_root = PROJECT_ROOT / "external" / "benchmarks" / "flexible jobshop" / "instances"
    common = {
        "family_key": "kacem",
        "instance_path": flex_root / "KacemHammadiBorne2002" / "kacem1.txt",
        "instance_kind": "kacem",
        "wip_cap": 2,
    }
    case = HolonicExperimentCase(
        case_id="tree_runtime_v2_small__kacem_pair_langgraph_llm_strict__R0",
        suite_tier="tree_runtime_v2_small",
        level="R0",
        benchmark_item_id="suite=tree_runtime_v2_small__regime=R0__topology=tree_a2_c2__cells=kacem_pair__split=test",
        cell_configs=[
            PlantCellConfig(cell_id=0, label="cell-a", processing_time_scale=0.9, **common),
            PlantCellConfig(cell_id=1, label="cell-b", processing_time_scale=1.1, **common),
        ],
        seeds=[13],
        env_name="tree_runtime_v2",
        env_cls=lambda cell_configs, config, seed: TreeRuntimeEnv(
            cell_configs,
            area_to_cells={0: [0], 1: [1]},
            config=config,
            seed=seed,
        ),
        regime_id="R0",
        topology_id="tree_a2_c2",
    )
    driver = LangGraphLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
    )
    monkeypatch.setattr(
        driver,
        "_select_openai_compat_action",
        lambda *, agent_id, agent_view, legal_actions: None,
    )

    with pytest.raises(RuntimeError, match="fallback_contract decisions are disallowed"):
        run_experiment_case(
            case=case,
            policies=[driver],
            output_dir=tmp_path / "tree_runtime_v2_langgraph_llm_strict",
            require_zero_fallback=True,
        )
