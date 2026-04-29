"""Tests for controller family adapters over the v2 runtime contract."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import BenchmarkConfig, PlantCellConfig
from fms_bench.v2.controller_api import TreeControllerAdapter
from fms_bench.v2.tree_runtime_env import TreeRuntimeEnv


FLEX_ROOT = PROJECT_ROOT / "external" / "benchmarks" / "flexible jobshop" / "instances"


def _make_config() -> BenchmarkConfig:
    config = BenchmarkConfig()
    config.horizon_hours = 200
    config.failure.mtbf_hours = 0.0
    config.failure.mttr_hours = 0.0
    return config


def _replicated_cell_configs() -> list[PlantCellConfig]:
    common = {
        "family_key": "kacem",
        "instance_path": FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
        "instance_kind": "kacem",
        "wip_cap": 2,
    }
    return [
        PlantCellConfig(
            cell_id=0,
            label="cell-fast",
            processing_time_scale=0.8,
            energy_cap_kwh=650.0,
            carbon_cap_kg=390.0,
            **common,
        ),
        PlantCellConfig(
            cell_id=1,
            label="cell-slow",
            processing_time_scale=1.4,
            energy_cap_kwh=900.0,
            carbon_cap_kg=540.0,
            **common,
        ),
    ]


def _ordertrap_cell_configs() -> list[PlantCellConfig]:
    common = {
        "family_key": "brandimarte",
        "instance_path": FLEX_ROOT / "Brandimarte1993" / "mk01.txt",
        "instance_kind": "brandimarte",
        "wip_cap": 2,
    }
    return [
        PlantCellConfig(
            cell_id=0,
            label="cell-slow-tight",
            processing_time_scale=1.38,
            energy_cap_kwh=220.0,
            carbon_cap_kg=132.0,
            **common,
        ),
        PlantCellConfig(
            cell_id=1,
            label="cell-fast-loose",
            processing_time_scale=0.74,
            energy_cap_kwh=1500.0,
            carbon_cap_kg=900.0,
            **common,
        ),
        PlantCellConfig(
            cell_id=2,
            label="cell-medium",
            processing_time_scale=1.0,
            energy_cap_kwh=780.0,
            carbon_cap_kg=468.0,
            **common,
        ),
    ]


class _ConstantPolicy:
    def __init__(self, action: int, name: str) -> None:
        self._action = int(action)
        self._name = name

    def select_action(self, observation: np.ndarray, info: dict[str, object]) -> int:
        del observation, info
        return self._action

    @property
    def name(self) -> str:
        return self._name


def test_role_policy_router_prefers_exact_agent_then_prefix() -> None:
    from fms_bench.v2.controller_families import RolePolicyRouter

    router = RolePolicyRouter(
        exact_policies={"plant": _ConstantPolicy(action=3, name="plant-policy")},
        prefix_policies={
            "area_": _ConstantPolicy(action=5, name="area-policy"),
            "cell_": _ConstantPolicy(action=7, name="cell-policy"),
        },
        controller_name="router",
    )

    observation = np.zeros(3, dtype=np.float32)

    assert router.select_action(observation, {"agent_id": "plant"}) == 3
    assert router.select_action(observation, {"agent_id": "area_2"}) == 5
    assert router.select_action(observation, {"agent_id": "cell_9"}) == 7
    assert router.name == "router"


def test_prompt_controller_policy_uses_prompt_builder_over_legal_agent_view() -> None:
    from fms_bench.v2.controller_families import PromptControllerPolicy

    calls: list[tuple[str, object, object]] = []

    def prompt_builder(observation: np.ndarray, info: dict[str, object]) -> str:
        calls.append(("build", info["agent_id"], len(info["visible_protocol_objects"])))
        return f"agent={info['agent_id']};obs={observation.shape[0]}"

    def prompt_runner(prompt: str, observation: np.ndarray, info: dict[str, object]) -> int:
        calls.append(("run", prompt, info["agent_id"]))
        mask = np.asarray(info["action_mask"], dtype=np.int32)
        valid = np.flatnonzero(mask > 0)
        return int(valid[0]) if len(valid) else 0

    env = TreeRuntimeEnv(
        _replicated_cell_configs(),
        area_to_cells={0: [0], 1: [1]},
        config=_make_config(),
        seed=13,
    )
    try:
        env.reset(seed=13)
        adapter = TreeControllerAdapter(env)
        policy = PromptControllerPolicy(
            prompt_builder=prompt_builder,
            prompt_runner=prompt_runner,
            controller_name="prompt-stub",
        )

        plant_actions = adapter.step_controller(policy)
        assert set(plant_actions) == {"plant"}
        adapter.step_dict(plant_actions)

        area_actions = adapter.step_controller(policy)
        assert set(area_actions) == {"area_0"}
        assert ("build", "plant", 0) in calls
        assert ("build", "area_0", 1) in calls
        assert any(kind == "run" and payload == "area_0" for kind, _, payload in calls)
        assert policy.name == "prompt-stub"
    finally:
        env.close()


def test_policy_network_controller_delegates_to_inference_backend() -> None:
    from fms_bench.v2.controller_families import PolicyNetworkController

    seen_agents: list[str] = []

    def inference_backend(observation: np.ndarray, info: dict[str, object]) -> int:
        del observation
        seen_agents.append(str(info["agent_id"]))
        mask = np.asarray(info["action_mask"], dtype=np.int32)
        valid = np.flatnonzero(mask > 0)
        return int(valid[0]) if len(valid) else 0

    env = TreeRuntimeEnv(
        _replicated_cell_configs(),
        area_to_cells={0: [0], 1: [1]},
        config=_make_config(),
        seed=13,
    )
    try:
        env.reset(seed=13)
        adapter = TreeControllerAdapter(env)
        policy = PolicyNetworkController(
            inference_fn=inference_backend,
            controller_name="network-stub",
        )

        actions = adapter.step_controller(policy)
        assert set(actions) == {"plant"}
        assert seen_agents == ["plant"]
        assert policy.name == "network-stub"
    finally:
        env.close()


def test_tree_controller_adapter_exposes_role_scoped_dispatch_state_for_active_area() -> None:
    env = TreeRuntimeEnv(
        _ordertrap_cell_configs(),
        area_to_cells={0: [0, 1], 1: [2]},
        config=_make_config(),
        seed=13,
    )
    try:
        env.reset(seed=13)
        env.step_dict({"plant": 0})
        adapter = TreeControllerAdapter(env)

        area_view = adapter.observe("area_0")

        assert area_view["valid_cells"] == [0, 1]
        assert sorted(area_view["visible_cell_summaries"]) == [0, 1]
        assert 2 not in area_view["visible_cell_summaries"]
    finally:
        env.close()


def test_v2_tree_policies_can_diverge_on_area_child_choice() -> None:
    from fms_bench import FlatTreePolicy, HierarchicalTreePolicy, TreeContractNetPolicy

    def choose_area_child(policy: object) -> int:
        env = TreeRuntimeEnv(
            _ordertrap_cell_configs(),
            area_to_cells={0: [0, 1], 1: [2]},
            config=_make_config(),
            seed=13,
        )
        try:
            env.reset(seed=13)
            env.step_dict({"plant": 0})
            adapter = TreeControllerAdapter(env)
            return int(adapter.step_controller(policy)["area_0"])
        finally:
            env.close()

    assert choose_area_child(FlatTreePolicy()) == 0
    assert choose_area_child(HierarchicalTreePolicy()) == 1
    assert choose_area_child(TreeContractNetPolicy()) == 1


def test_framework_driver_registry_exports_wave1_framework_drivers() -> None:
    from fms_bench import AgentScopeTreeDriver, LangGraphTreeDriver
    from fms_bench.v2.framework_drivers import (
        AgentScopeTreeDriver as V2AgentScopeTreeDriver,
        FrameworkBackedTreeDriver,
        LangGraphTreeDriver as V2LangGraphTreeDriver,
    )

    assert issubclass(V2AgentScopeTreeDriver, FrameworkBackedTreeDriver)
    assert issubclass(V2LangGraphTreeDriver, FrameworkBackedTreeDriver)
    assert AgentScopeTreeDriver is V2AgentScopeTreeDriver
    assert LangGraphTreeDriver is V2LangGraphTreeDriver


def test_framework_driver_registry_exports_llm_backed_wave1_variants() -> None:
    from fms_bench import AgentScopeLLMTreeDriver, LangGraphLLMTreeDriver
    from fms_bench.v2.framework_drivers import (
        AgentScopeLLMTreeDriver as V2AgentScopeLLMTreeDriver,
        FrameworkBackedTreeDriver,
        LangGraphLLMTreeDriver as V2LangGraphLLMTreeDriver,
    )

    assert issubclass(V2AgentScopeLLMTreeDriver, FrameworkBackedTreeDriver)
    assert issubclass(V2LangGraphLLMTreeDriver, FrameworkBackedTreeDriver)
    assert AgentScopeLLMTreeDriver is V2AgentScopeLLMTreeDriver
    assert LangGraphLLMTreeDriver is V2LangGraphLLMTreeDriver


def test_framework_driver_registry_exports_wave2_framework_drivers() -> None:
    from fms_bench import AG2TreeDriver, GoogleADKTreeDriver
    from fms_bench.v2.framework_drivers import (
        AG2TreeDriver as V2AG2TreeDriver,
        FrameworkBackedTreeDriver,
        GoogleADKTreeDriver as V2GoogleADKTreeDriver,
    )

    assert issubclass(V2AG2TreeDriver, FrameworkBackedTreeDriver)
    assert issubclass(V2GoogleADKTreeDriver, FrameworkBackedTreeDriver)
    assert AG2TreeDriver is V2AG2TreeDriver
    assert GoogleADKTreeDriver is V2GoogleADKTreeDriver


def test_framework_driver_registry_exports_wave3_framework_drivers() -> None:
    from fms_bench import CrewAITreeDriver, SPADETreeDriver
    from fms_bench.v2.framework_drivers import (
        CrewAITreeDriver as V2CrewAITreeDriver,
        FrameworkBackedTreeDriver,
        SPADETreeDriver as V2SPADETreeDriver,
    )

    assert issubclass(V2CrewAITreeDriver, FrameworkBackedTreeDriver)
    assert issubclass(V2SPADETreeDriver, FrameworkBackedTreeDriver)
    assert CrewAITreeDriver is V2CrewAITreeDriver
    assert SPADETreeDriver is V2SPADETreeDriver


def test_framework_driver_registry_exports_llm_backed_wave2_and_wave3_variants() -> None:
    from fms_bench import (
        AG2LLMTreeDriver,
        CrewAILLMTreeDriver,
        GoogleADKLLMTreeDriver,
        SPADELLMTreeDriver,
    )
    from fms_bench.v2.framework_drivers import (
        AG2LLMTreeDriver as V2AG2LLMTreeDriver,
        CrewAILLMTreeDriver as V2CrewAILLMTreeDriver,
        FrameworkBackedTreeDriver,
        GoogleADKLLMTreeDriver as V2GoogleADKLLMTreeDriver,
        SPADELLMTreeDriver as V2SPADELLMTreeDriver,
    )

    for driver_cls in (
        V2AG2LLMTreeDriver,
        V2CrewAILLMTreeDriver,
        V2GoogleADKLLMTreeDriver,
        V2SPADELLMTreeDriver,
    ):
        assert issubclass(driver_cls, FrameworkBackedTreeDriver)

    assert AG2LLMTreeDriver is V2AG2LLMTreeDriver
    assert CrewAILLMTreeDriver is V2CrewAILLMTreeDriver
    assert GoogleADKLLMTreeDriver is V2GoogleADKLLMTreeDriver
    assert SPADELLMTreeDriver is V2SPADELLMTreeDriver
