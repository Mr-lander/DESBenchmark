"""Tests for the external controller adapter over the v2 runtime."""

from __future__ import annotations

import inspect
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import textwrap
import types
from typing import Mapping

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench import BenchmarkConfig, PlantCellConfig
from fms_bench.v2.tree_runtime_env import TreeRuntimeEnv


FLEX_ROOT = PROJECT_ROOT / "external" / "benchmarks" / "flexible jobshop" / "instances"


def _make_config() -> BenchmarkConfig:
    config = BenchmarkConfig()
    config.horizon_hours = 200
    config.failure.mtbf_hours = 0.0
    config.failure.mttr_hours = 0.0
    return config


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


def _mask_first_actions(decision: dict[str, object]) -> dict[str, int]:
    actions: dict[str, int] = {}
    for agent_id, agent_view in dict(decision["active_agents"]).items():
        mask = list(agent_view["action_mask"])
        valid = [index for index, enabled in enumerate(mask) if enabled]
        actions[str(agent_id)] = valid[0] if valid else 0
    return actions


def _make_external_session():
    from fms_bench.v2.external_controller_adapter import ExternalControllerSession

    env = TreeRuntimeEnv(
        _ordertrap_cell_configs(),
        area_to_cells={0: [0, 1], 1: [2]},
        config=_make_config(),
        seed=13,
    )
    return env, ExternalControllerSession(env)


def _make_multi_cell_external_session():
    from fms_bench.v2.external_controller_adapter import ExternalControllerSession

    common = {
        "family_key": "kacem",
        "instance_path": FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
        "instance_kind": "kacem",
        "wip_cap": 1,
    }
    env = TreeRuntimeEnv(
        [
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
        ],
        area_to_cells={0: [0, 1]},
        config=_make_config(),
        seed=13,
    )
    return env, ExternalControllerSession(env)


def _make_holonic_accountability_session():
    from fms_bench.v2.external_controller_adapter import ExternalControllerSession

    common = {
        "family_key": "kacem",
        "instance_path": FLEX_ROOT / "KacemHammadiBorne2002" / "kacem1.txt",
        "instance_kind": "kacem",
        "wip_cap": 2,
    }
    env = TreeRuntimeEnv(
        [
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
            PlantCellConfig(
                cell_id=2,
                label="cell-medium",
                processing_time_scale=1.0,
                energy_cap_kwh=800.0,
                carbon_cap_kg=480.0,
                **common,
            ),
        ],
        area_to_cells={0: [0, 1], 1: [2]},
        config=_make_config(),
        seed=13,
        authority_mode="holonic_hybrid",
    )
    return env, ExternalControllerSession(env)


def _write_subprocess_controller(tmp_path: Path) -> Path:
    controller_path = tmp_path / "jsonl_controller.py"
    controller_path.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            import sys


            def _choose_action(agent_view, *, use_last):
                mask = list(agent_view["action_mask"])
                valid = [index for index, enabled in enumerate(mask) if enabled]
                if not valid:
                    return 0
                return valid[-1] if use_last else valid[0]


            parser = argparse.ArgumentParser()
            parser.add_argument("--mode", default="first_valid")
            args = parser.parse_args()

            decision_count = 0
            for raw_line in sys.stdin:
                line = raw_line.strip()
                if not line:
                    continue
                decision = json.loads(line)
                decision_count += 1
                use_last = args.mode == "last_valid"
                if args.mode == "first_decision_last_then_first":
                    use_last = decision_count == 1
                actions = {
                    str(agent_id): _choose_action(agent_view, use_last=use_last)
                    for agent_id, agent_view in dict(decision["active_agents"]).items()
                }
                sys.stdout.write(json.dumps(actions) + "\\n")
                sys.stdout.flush()
            """
        ),
        encoding="utf-8",
    )
    return controller_path


def _write_subprocess_controller_command(tmp_path: Path, *, mode: str) -> str:
    controller_dir = tmp_path / "subprocess controller"
    controller_dir.mkdir()
    controller_path = _write_subprocess_controller(controller_dir)
    return f'{sys.executable} "{controller_path}" --mode {mode}'


class _OpenAICompatTestServer:
    def __init__(
        self,
        *,
        response_formatter: callable | None = None,
    ) -> None:
        self.requests: list[dict[str, object]] = []
        self._response_formatter = response_formatter
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._build_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(body)
                outer.requests.append(payload)
                prompt = str(payload["messages"][-1]["content"])
                decision = json.loads(prompt)
                actions = {
                    str(agent_id): next(
                        (index for index, enabled in enumerate(dict(agent_view).get("action_mask", [])) if enabled),
                        0,
                    )
                    for agent_id, agent_view in dict(decision.get("active_agents", {})).items()
                }
                content = (
                    outer._response_formatter(decision, actions)
                    if outer._response_formatter is not None
                    else json.dumps(actions)
                )
                response = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": content,
                            },
                        }
                    ],
                }
                encoded = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                del format, args

        return _Handler


def test_external_controller_session_exports_json_safe_role_local_views() -> None:
    from fms_bench.v2.external_controller_adapter import ExternalControllerSession

    env = TreeRuntimeEnv(
        _ordertrap_cell_configs(),
        area_to_cells={0: [0, 1], 1: [2]},
        config=_make_config(),
        seed=13,
    )
    try:
        session = ExternalControllerSession(env)

        reset_payload = session.reset(seed=13)

        assert reset_payload["agents_to_act"] == ["plant"]
        plant_view = reset_payload["active_agents"]["plant"]
        assert plant_view["role"] == "plant"
        assert isinstance(plant_view["observation"], list)
        assert isinstance(plant_view["action_mask"], list)
        assert "simulator" not in plant_view
        json.dumps(reset_payload)

        transition = session.step({"plant": 0})
        area_view = transition["current_decision"]["active_agents"]["area_0"]

        assert area_view["role"] == "area"
        assert area_view["valid_cells"] == [0, 1]
        assert sorted(area_view["visible_cell_summaries"]) == ["0", "1"]
        assert area_view["visible_protocol_objects"][0]["kind"] == "assignment_request"
        json.dumps(transition)
    finally:
        env.close()


def test_external_controller_session_does_not_export_broad_info_to_controller_payload() -> None:
    env, session = _make_external_session()
    try:
        decision = session.reset(seed=3)

        assert "info" not in decision
        assert "artifact_info" not in decision
        assert set(decision) >= {
            "time",
            "terminated",
            "truncated",
            "agents_to_act",
            "active_agents",
            "execution_mode",
            "supported_execution_modes",
        }
    finally:
        env.close()


def test_external_controller_session_exposes_broad_debug_payload_only_via_explicit_debug_api() -> None:
    env, session = _make_external_session()
    try:
        session.reset(seed=3)

        debug_payload = session.current_artifact_payload()

        assert "info" in debug_payload
        assert "reward_components" in debug_payload["info"]
    finally:
        env.close()


def test_external_controller_session_decision_payload_stays_on_narrow_public_schema() -> None:
    env, session = _make_external_session()
    forbidden_keys = {
        "info",
        "artifact_info",
        "activation_trace",
        "protocol_trace",
        "world_event_trace",
        "settlement_trace",
        "reward_components",
        "current_decision_epoch",
    }
    expected_keys = {
        "time",
        "terminated",
        "truncated",
        "agents_to_act",
        "active_agents",
        "execution_mode",
        "supported_execution_modes",
    }
    try:
        decision = session.reset(seed=13)
        assert set(decision) == expected_keys
        assert set(decision).isdisjoint(forbidden_keys)
        assert decision["execution_mode"] == "single_agent"
        assert decision["supported_execution_modes"] == [
            "single_agent",
            "same_timestamp_multi_cell_local_regulation",
        ]

        for agent_view in dict(decision["active_agents"]).values():
            view_keys = {str(key) for key in dict(agent_view)}
            assert view_keys.isdisjoint(forbidden_keys)
            assert all(not key.endswith("_trace") for key in view_keys)

        transition = session.step({"plant": 0})
        next_decision = dict(transition["current_decision"])
        assert set(next_decision) == expected_keys
        assert set(next_decision).isdisjoint(forbidden_keys)

        for agent_view in dict(next_decision["active_agents"]).values():
            view_keys = {str(key) for key in dict(agent_view)}
            assert view_keys.isdisjoint(forbidden_keys)
            assert all(not key.endswith("_trace") for key in view_keys)
    finally:
        env.close()


def test_external_controller_session_agent_views_match_public_role_payload_contract() -> None:
    from fms_bench.v2.controller_payloads import (
        PUBLIC_CONTROLLER_PAYLOAD_BASE_FIELDS,
        PUBLIC_CONTROLLER_PAYLOAD_ROLE_ALIAS_FIELDS,
    )

    env, session = _make_external_session()
    try:
        decision = session.reset(seed=13)
        plant_view = decision["active_agents"]["plant"]
        assert set(plant_view) == set(PUBLIC_CONTROLLER_PAYLOAD_BASE_FIELDS).union(
            PUBLIC_CONTROLLER_PAYLOAD_ROLE_ALIAS_FIELDS["plant"]
        )

        area_decision = session.step({"plant": 0})["current_decision"]
        area_view = area_decision["active_agents"]["area_0"]
        assert set(area_view) == set(PUBLIC_CONTROLLER_PAYLOAD_BASE_FIELDS).union(
            PUBLIC_CONTROLLER_PAYLOAD_ROLE_ALIAS_FIELDS["area"]
        )

        cell_decision = session.step({"area_0": 0})["current_decision"]
        cell_view = cell_decision["active_agents"]["cell_0"]
        assert set(cell_view) == set(PUBLIC_CONTROLLER_PAYLOAD_BASE_FIELDS).union(
            PUBLIC_CONTROLLER_PAYLOAD_ROLE_ALIAS_FIELDS["cell"]
        )
    finally:
        env.close()


def test_external_controller_session_auto_resolves_exhausted_plant_accountability_epoch() -> None:
    env, session = _make_holonic_accountability_session()
    try:
        session.reset(seed=13)
        env.step_dict({"plant": 0})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_0": 1})
        env.step_dict({"area_0": 0})
        env.step_dict({"cell_1": 1})
        env.step_dict({"area_0": 0})
        env.step_dict({"plant": 1})
        env.step_dict({"area_1": 2})
        env.step_dict({"cell_2": 1})
        env.step_dict({"area_1": 0})

        assert env.agents_to_act() == ["plant"]
        assert env.current_decision_epoch.records_by_agent["plant"].decision_kind == "accountability_reroute"
        assert env.action_mask("plant").tolist() == [0, 0]

        decision = session.current_decision_payload()

        assert decision["agents_to_act"] == ["plant"]
        plant_view = decision["active_agents"]["plant"]
        assert plant_view["plant_decision_kind"] != "accountability_reroute"
        assert any(int(enabled) > 0 for enabled in plant_view["action_mask"])
    finally:
        env.close()


def test_framework_driver_modules_do_not_reference_artifact_debug_channels() -> None:
    framework_driver_paths = [
        PROJECT_ROOT / "fms_bench" / "v2" / "framework_drivers" / "ag2_driver.py",
        PROJECT_ROOT / "fms_bench" / "v2" / "framework_drivers" / "agentscope_driver.py",
        PROJECT_ROOT / "fms_bench" / "v2" / "framework_drivers" / "crewai_driver.py",
        PROJECT_ROOT / "fms_bench" / "v2" / "framework_drivers" / "google_adk_driver.py",
        PROJECT_ROOT / "fms_bench" / "v2" / "framework_drivers" / "langgraph_driver.py",
        PROJECT_ROOT / "fms_bench" / "v2" / "framework_drivers" / "spade_driver.py",
    ]
    forbidden_tokens = (
        "current_artifact_payload",
        "reward_components",
        "protocol_trace",
        "world_event_trace",
        "settlement_trace",
        "activation_trace",
    )

    for path in framework_driver_paths:
        text = path.read_text(encoding="utf-8")
        assert "agent_view" in text, path.name
        for token in forbidden_tokens:
            assert token not in text, (path.name, token)


def test_external_controller_session_rejects_inactive_and_mask_illegal_actions() -> None:
    from fms_bench.v2.external_controller_adapter import ExternalControllerSession

    env = TreeRuntimeEnv(
        _ordertrap_cell_configs(),
        area_to_cells={0: [0, 1], 1: [2]},
        config=_make_config(),
        seed=13,
    )
    try:
        session = ExternalControllerSession(env)
        reset_payload = session.reset(seed=13)
        plant_mask = list(reset_payload["active_agents"]["plant"]["action_mask"])

        with pytest.raises(KeyError):
            session.step({"area_0": 0})

        with pytest.raises(ValueError):
            session.step({"plant": len(plant_mask) + 5})

        transition = session.step([{"agent_id": "plant", "action": 0}])
        assert transition["current_decision"]["agents_to_act"] == ["area_0"]
    finally:
        env.close()


def test_external_controller_session_supports_multi_cell_same_timestamp_epoch() -> None:
    env, session = _make_multi_cell_external_session()
    try:
        session.reset(seed=13)
        session.step({"plant": 0})
        session.step({"area_0": 0})
        session.step({"cell_0": 0})
        session.step({"plant": 0})
        session.step({"area_0": 1})
        decision = session.step({"cell_1": 0})["current_decision"]

        assert decision["agents_to_act"] == ["cell_0", "cell_1"]
        assert decision["execution_mode"] == "same_timestamp_multi_cell_local_regulation"
        assert sorted(decision["active_agents"]) == ["cell_0", "cell_1"]

        transition = session.step({"cell_0": 0, "cell_1": 0})

        assert set(transition["submitted_actions"]) == {"cell_0", "cell_1"}
        assert transition["reward"] == transition["reward_components"]["total"]
        assert transition["current_decision"]["time"] > decision["time"]
    finally:
        env.close()


def test_external_controller_loop_supports_callable_and_object_style_drivers() -> None:
    from fms_bench.v2.external_controller_adapter import (
        ExternalControllerSession,
        run_external_controller_loop,
    )

    def network_style_driver(decision: dict[str, object]) -> dict[str, int]:
        return _mask_first_actions(decision)

    class PromptStyleDriver:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def select_actions(self, decision: dict[str, object]) -> dict[str, int]:
            for agent_id, agent_view in dict(decision["active_agents"]).items():
                self.prompts.append(f"{agent_id}:{agent_view['role']}")
            return _mask_first_actions(decision)

    class MasStyleCoordinator:
        def __init__(self) -> None:
            self.protocol_kinds: list[str] = []

        def select_actions(self, decision: dict[str, object]) -> dict[str, int]:
            for agent_view in dict(decision["active_agents"]).values():
                for obj in list(agent_view["visible_protocol_objects"]):
                    self.protocol_kinds.append(str(obj["kind"]))
            return _mask_first_actions(decision)

    drivers = [
        network_style_driver,
        PromptStyleDriver(),
        MasStyleCoordinator(),
    ]

    for driver in drivers:
        env = TreeRuntimeEnv(
            _ordertrap_cell_configs(),
            area_to_cells={0: [0, 1], 1: [2]},
            config=_make_config(),
            seed=13,
        )
        try:
            session = ExternalControllerSession(env)
            session.reset(seed=13)

            transitions = run_external_controller_loop(session, driver, max_steps=2)

            assert len(transitions) == 2
            assert transitions[0]["submitted_actions"] == {"plant": 0}
            assert transitions[0]["current_decision"]["agents_to_act"] == ["area_0"]
            assert transitions[1]["submitted_actions"] == {"area_0": 0}
            assert transitions[1]["current_decision"]["agents_to_act"] == ["cell_0"]
        finally:
            env.close()


def test_role_routed_external_driver_prefers_exact_agent_then_role_prefix() -> None:
    from fms_bench.v2.external_controller_adapter import RoleRoutedExternalDriver

    calls: list[str] = []

    def plant_driver(decision: dict[str, object], agent_view: dict[str, object]) -> int:
        del decision, agent_view
        calls.append("plant")
        return 0

    def area_driver(decision: dict[str, object], agent_view: dict[str, object]) -> int:
        del decision, agent_view
        calls.append("area")
        return 0

    def cell_driver(decision: dict[str, object], agent_view: dict[str, object]) -> int:
        del decision, agent_view
        calls.append("cell")
        return 0

    env = TreeRuntimeEnv(
        _ordertrap_cell_configs(),
        area_to_cells={0: [0, 1], 1: [2]},
        config=_make_config(),
        seed=13,
    )
    try:
        session = __import__(
            "fms_bench.v2.external_controller_adapter",
            fromlist=["ExternalControllerSession"],
        ).ExternalControllerSession(env)
        router = RoleRoutedExternalDriver(
            exact_drivers={"plant": plant_driver},
            role_drivers={"area": area_driver, "cell": cell_driver},
            controller_name="external_router",
        )

        session.reset(seed=13)
        session.step(router.select_actions(session.current_decision_payload()))
        session.step(router.select_actions(session.current_decision_payload()))
        session.step(router.select_actions(session.current_decision_payload()))

        assert calls == ["plant", "area", "cell"]
        assert router.name == "external_router"
    finally:
        env.close()


def test_external_tree_driver_family_can_diverge_on_area_child_choice() -> None:
    from fms_bench.v2.external_controller_adapter import ExternalControllerSession
    from fms_bench.v2.external_tree_drivers import (
        ExternalAdaptiveHolonTreeDriver,
        ExternalBlackboardTreeDriver,
        ExternalContractNetTreeDriver,
        ExternalFlatTreeDriver,
        ExternalHierarchicalTreeDriver,
    )

    def choose_area_child(driver: object) -> int:
        env = TreeRuntimeEnv(
            _ordertrap_cell_configs(),
            area_to_cells={0: [0, 1], 1: [2]},
            config=_make_config(),
            seed=13,
        )
        try:
            session = ExternalControllerSession(env)
            session.reset(seed=13)
            session.step({"plant": 0})
            decision = session.current_decision_payload()
            return int(driver.select_actions(decision)["area_0"])
        finally:
            env.close()

    assert choose_area_child(ExternalFlatTreeDriver()) == 0
    assert choose_area_child(ExternalHierarchicalTreeDriver()) == 1
    assert choose_area_child(ExternalContractNetTreeDriver()) == 1
    assert choose_area_child(ExternalAdaptiveHolonTreeDriver()) == 1
    assert choose_area_child(ExternalBlackboardTreeDriver()) == 1


def test_external_blackboard_tree_driver_uses_shared_memory_and_reset() -> None:
    from fms_bench.v2.external_tree_drivers import ExternalBlackboardTreeDriver

    driver = ExternalBlackboardTreeDriver(memory_penalty=5.0)
    decision = {
        "active_agents": {
            "plant": {
                "role": "plant",
                "action_mask": [1, 1],
                "valid_areas": [0, 1],
                "visible_area_summaries": {
                    "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                    "1": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                },
            },
            "area_0": {
                "role": "area",
                "action_mask": [1, 1],
                "valid_cells": [0, 1],
                "visible_cell_summaries": {
                    "0": {"processing_time_scale": 1.0, "active_jobs": 0.0, "busy_machines": 0.0, "backlog_jobs": 0.0},
                    "1": {"processing_time_scale": 1.0, "active_jobs": 0.0, "busy_machines": 0.0, "backlog_jobs": 0.0},
                },
            },
        }
    }

    first = driver.select_actions(decision)
    second = driver.select_actions(decision)
    driver.reset_episode()
    after_reset = driver.select_actions(decision)

    assert first == {"plant": 0, "area_0": 0}
    assert second == {"plant": 1, "area_0": 1}
    assert after_reset == {"plant": 0, "area_0": 0}
    assert driver.name == "external_tree_blackboard_balanced"


def test_external_adaptive_holon_tree_driver_switches_to_transient_mode_and_resets() -> None:
    from fms_bench.v2.external_tree_drivers import (
        ExternalAdaptiveHolonTreeDriver,
        ExternalHierarchicalTreeDriver,
    )

    driver = ExternalAdaptiveHolonTreeDriver(transient_cooldown_steps=3, disturbance_threshold=4.0)
    hierarchical = ExternalHierarchicalTreeDriver()
    decision = {
        "active_agents": {
            "plant": {
                "role": "plant",
                "action_mask": [1, 1],
                "valid_areas": [0, 1],
                "visible_area_summaries": {
                    "0": {
                        "active_jobs": 0.0,
                        "backlog_jobs": 0.0,
                        "busy_machines": 0.0,
                        "down_machines": 3.0,
                        "energy_kwh": 200.0,
                        "carbon_kg": 120.0,
                        "budget_cap_energy_kwh": 210.0,
                        "budget_cap_carbon_kg": 132.0,
                        "budget_cap_wip": 2.0,
                    },
                    "1": {
                        "active_jobs": 1.0,
                        "backlog_jobs": 1.0,
                        "busy_machines": 1.0,
                        "down_machines": 0.0,
                        "energy_kwh": 100.0,
                        "carbon_kg": 60.0,
                        "budget_cap_energy_kwh": 210.0,
                        "budget_cap_carbon_kg": 132.0,
                        "budget_cap_wip": 2.0,
                    },
                },
            }
        }
    }

    first = driver.select_actions(decision)
    second = driver.select_actions(decision)
    reset = driver.reset_episode()
    after_reset = driver.select_actions(decision)

    assert driver.name == "external_tree_adacor_adaptive"
    assert first == hierarchical.select_actions(decision)
    assert first == {"plant": 0}
    assert second == {"plant": 1}
    assert after_reset == {"plant": 0}
    assert reset is None


def test_external_adaptive_holon_tree_driver_matches_hierarchical_on_calm_area_view() -> None:
    from fms_bench.v2.external_tree_drivers import (
        ExternalAdaptiveHolonTreeDriver,
        ExternalHierarchicalTreeDriver,
    )

    decision = {
        "active_agents": {
            "plant": {
                "role": "plant",
                "action_mask": [1, 1],
                "valid_areas": [0, 1],
                "visible_area_summaries": {
                    "0": {
                        "cell_ids": [0, 1],
                        "active_jobs": 0.0,
                        "backlog_jobs": 0.0,
                        "energy_kwh": 0.0,
                        "carbon_kg": 0.0,
                    },
                    "1": {
                        "cell_ids": [2],
                        "active_jobs": 1.0,
                        "backlog_jobs": 0.0,
                        "energy_kwh": 0.0,
                        "carbon_kg": 0.0,
                    },
                },
            }
        }
    }

    adaptive = ExternalAdaptiveHolonTreeDriver(disturbance_threshold=4.0)
    hierarchical = ExternalHierarchicalTreeDriver()

    assert adaptive.select_actions(decision) == hierarchical.select_actions(decision) == {"plant": 0}
    assert adaptive._transient_cooldowns == {}


def test_external_supervisor_tree_driver_propagates_area_disturbance_memory_and_resets() -> None:
    from fms_bench.v2.external_tree_drivers import ExternalSupervisorReactiveTreeDriver

    driver = ExternalSupervisorReactiveTreeDriver(memory_decay=1.0, disturbance_weight=3.0)
    plant_decision = {
        "active_agents": {
            "plant": {
                "role": "plant",
                "action_mask": [1, 1],
                "valid_areas": [0, 1],
                "visible_area_summaries": {
                    "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                    "1": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                },
            }
        }
    }
    disturbed_area_decision = {
        "active_agents": {
            "area_0": {
                "role": "area",
                "action_mask": [1, 1],
                "valid_cells": [0, 1],
                "visible_cell_summaries": {
                    "0": {
                        "processing_time_scale": 1.0,
                        "active_jobs": 0.0,
                        "busy_machines": 0.0,
                        "backlog_jobs": 0.0,
                        "down_machines": 3.0,
                    },
                    "1": {
                        "processing_time_scale": 1.1,
                        "active_jobs": 0.0,
                        "busy_machines": 0.0,
                        "backlog_jobs": 0.0,
                        "down_machines": 0.0,
                    },
                },
            }
        }
    }

    first_plant = driver.select_actions(plant_decision)
    area_choice = driver.select_actions(disturbed_area_decision)
    second_plant = driver.select_actions(plant_decision)
    driver.reset_episode()
    after_reset = driver.select_actions(plant_decision)

    assert first_plant == {"plant": 0}
    assert area_choice == {"area_0": 1}
    assert second_plant == {"plant": 1}
    assert after_reset == {"plant": 0}
    assert driver.name == "external_tree_supervisor_reactive"


def test_external_supervisor_tree_driver_uses_public_dispatch_options_and_machine_state() -> None:
    from fms_bench.v2.external_tree_drivers import ExternalSupervisorReactiveTreeDriver

    driver = ExternalSupervisorReactiveTreeDriver(memory_decay=1.0, disturbance_weight=3.0)
    disturbed_decision = {
        "active_agents": {
            "cell_2": {
                "role": "cell",
                "cell_id": 2,
                "action_mask": [1, 1],
                "dispatch_options": [
                    {"machine_id": 5, "proc_time": 4.0},
                    {"machine_id": 6, "proc_time": 1.0},
                ],
                "visible_machine_summaries": {
                    "5": {"machine_id": 5, "status": "idle", "down_time": 0.0},
                    "6": {"machine_id": 6, "status": "idle", "down_time": 4.0},
                },
            }
        }
    }
    calm_decision = {
        "active_agents": {
            "cell_2": {
                "role": "cell",
                "cell_id": 2,
                "action_mask": [1, 1],
                "dispatch_options": [
                    {"machine_id": 5, "proc_time": 4.0},
                    {"machine_id": 6, "proc_time": 1.0},
                ],
                "visible_machine_summaries": {
                    "5": {"machine_id": 5, "status": "idle", "down_time": 0.0},
                    "6": {"machine_id": 6, "status": "idle", "down_time": 0.0},
                },
            }
        }
    }

    assert driver.select_actions(disturbed_decision) == {"cell_2": 0}
    driver.reset_episode()
    assert driver.select_actions(calm_decision) == {"cell_2": 1}


def test_external_session_rollout_does_not_emit_nonterminal_empty_decision_epochs() -> None:
    from fms_bench.v2.external_controller_adapter import ExternalControllerSession
    from fms_bench.v2.external_tree_drivers import ExternalSupervisorReactiveTreeDriver

    config = _make_config()
    config.failure.mtbf_hours = 30.0
    config.failure.mttr_hours = 3.0
    env = TreeRuntimeEnv(
        _ordertrap_cell_configs(),
        area_to_cells={0: [0], 1: [1], 2: [2]},
        config=config,
        seed=229,
    )
    try:
        session = ExternalControllerSession(env)
        driver = ExternalSupervisorReactiveTreeDriver()
        decision = session.reset(seed=229)

        for _ in range(200):
            if decision["terminated"] or decision["truncated"]:
                break
            assert decision["agents_to_act"], decision
            decision = session.step(driver.select_actions(decision))["current_decision"]
        else:
            pytest.fail("External rollout did not terminate within 200 decision epochs")
    finally:
        env.close()


def test_agentscope_tree_driver_uses_routing_memory_and_resets() -> None:
    from fms_bench.v2.framework_drivers import AgentScopeTreeDriver

    driver = AgentScopeTreeDriver(routing_penalty=5.0)
    decision = {
        "active_agents": {
            "plant": {
                "role": "plant",
                "action_mask": [1, 1],
                "valid_areas": [0, 1],
                "visible_area_summaries": {
                    "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                    "1": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                },
            },
            "area_0": {
                "role": "area",
                "action_mask": [1, 1],
                "valid_cells": [0, 1],
                "visible_cell_summaries": {
                    "0": {"processing_time_scale": 1.0, "active_jobs": 0.0, "busy_machines": 0.0, "backlog_jobs": 0.0},
                    "1": {"processing_time_scale": 1.0, "active_jobs": 0.0, "busy_machines": 0.0, "backlog_jobs": 0.0},
                },
            },
        }
    }

    first = driver.select_actions(decision)
    second = driver.select_actions(decision)
    driver.reset_episode()
    after_reset = driver.select_actions(decision)

    assert first == {"plant": 0, "area_0": 0}
    assert second == {"plant": 1, "area_0": 1}
    assert after_reset == {"plant": 0, "area_0": 0}
    assert driver.name == "external_framework_agentscope"


def _centralized_plant_cell_selection_agent_view() -> dict[str, object]:
    return {
        "role": "plant",
        "plant_decision_kind": "cell_selection",
        "action_mask": [1, 1, 0, 1, 1, 0, 1, 1, 0],
        "valid_areas": [0, 1, 2],
        "visible_area_summaries": {
            "0": {"active_jobs": 8.0, "backlog_jobs": 8.0, "energy_kwh": 40.0},
            "1": {"active_jobs": 7.0, "backlog_jobs": 7.0, "energy_kwh": 35.0},
            "2": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
        },
        "valid_execution_cells": [0, 1, 3, 4, 6, 7],
        "execution_cell_summaries": {
            "0": {"processing_time_scale": 1.5, "active_jobs": 0.0, "busy_machines": 0.0, "backlog_jobs": 0.0},
            "1": {"processing_time_scale": 1.4, "active_jobs": 0.0, "busy_machines": 0.0, "backlog_jobs": 0.0},
            "3": {"processing_time_scale": 1.3, "active_jobs": 0.0, "busy_machines": 0.0, "backlog_jobs": 0.0},
            "4": {"processing_time_scale": 1.2, "active_jobs": 0.0, "busy_machines": 0.0, "backlog_jobs": 0.0},
            "6": {"processing_time_scale": 1.1, "active_jobs": 0.0, "busy_machines": 0.0, "backlog_jobs": 0.0},
            "7": {"processing_time_scale": 0.8, "active_jobs": 0.0, "busy_machines": 0.0, "backlog_jobs": 0.0},
        },
    }


def test_agentscope_tree_driver_routes_centralized_plant_cell_selection_over_execution_cells() -> None:
    from fms_bench.v2.framework_drivers import AgentScopeTreeDriver

    driver = AgentScopeTreeDriver(routing_penalty=5.0)

    action = driver._select_contract_action(
        agent_id="plant",
        agent_view=_centralized_plant_cell_selection_agent_view(),
    )

    assert action == 7
    assert driver._handoff_counts["plant"][7] == 1


def test_framework_backed_tree_driver_does_not_claim_native_runtime_for_importable_stub() -> None:
    from fms_bench.v2.framework_drivers.base import FrameworkBackedTreeDriver

    class _ImportableStubDriver(FrameworkBackedTreeDriver):
        framework_name = "ImportableStub"
        python_package = "math"
        controller_name = "external_framework_importable_stub"

        def _select_framework_action(
            self,
            *,
            agent_id: str,
            agent_view: dict[str, object],
            decision: dict[str, object],
        ) -> int:
            del agent_id, agent_view, decision
            return 0

    driver = _ImportableStubDriver()

    assert driver.framework_runtime_info["available"] is True
    assert driver.framework_runtime_info["backend_mode"] == "contract_stub"


def test_framework_backed_tree_driver_does_not_forward_broad_decision_blob() -> None:
    from fms_bench.v2.framework_drivers.base import FrameworkBackedTreeDriver

    seen: list[dict[str, object]] = []

    class _CaptureDriver(FrameworkBackedTreeDriver):
        framework_name = "Capture"
        controller_name = "external_framework_capture"

        def _select_framework_action(
            self,
            *,
            agent_id: str,
            agent_view: Mapping[str, object],
        ) -> int:
            seen.append(
                {
                    "agent_id": agent_id,
                    "agent_view": dict(agent_view),
                }
            )
            return 0

    driver = _CaptureDriver()
    decision = {
        "time": 0.0,
        "agents_to_act": ["area_0"],
        "info": {"reward_components": {"total": 1.0}},
        "active_agents": {
            "area_0": {
                "agent_id": "area_0",
                "role": "area",
                "action_mask": [1, 1],
                "local_subtree_state": {"child_ids": [0, 1]},
            }
        },
    }

    assert driver.select_actions(decision) == {"area_0": 0}
    assert seen[0] == {
        "agent_id": "area_0",
        "agent_view": {
            "agent_id": "area_0",
            "role": "area",
            "action_mask": [1, 1],
            "local_subtree_state": {"child_ids": [0, 1]},
        },
    }


def test_framework_driver_contract_only_accepts_agent_local_payloads() -> None:
    from fms_bench.v2.framework_drivers import (
        AG2TreeDriver,
        AgentScopeTreeDriver,
        AgentScopeLLMTreeDriver,
        CrewAITreeDriver,
        GoogleADKTreeDriver,
        LangGraphTreeDriver,
        LangGraphLLMTreeDriver,
        SPADETreeDriver,
    )

    drivers = [
        AgentScopeTreeDriver(),
        AgentScopeLLMTreeDriver(
            model_name="qwen-test",
            base_url="http://127.0.0.1:30000/v1",
            api_key="EMPTY",
        ),
        LangGraphTreeDriver(),
        LangGraphLLMTreeDriver(
            model_name="qwen-test",
            base_url="http://127.0.0.1:30000/v1",
            api_key="EMPTY",
        ),
        AG2TreeDriver(),
        GoogleADKTreeDriver(),
        CrewAITreeDriver(),
        SPADETreeDriver(),
    ]

    for driver in drivers:
        framework_params = list(inspect.signature(driver._select_framework_action).parameters)
        assert framework_params == ["agent_id", "agent_view"]

        contract_params = list(inspect.signature(driver._select_contract_action).parameters)
        assert contract_params == ["agent_id", "agent_view"]

        native_selector = getattr(driver, "_select_native_action", None)
        if callable(native_selector):
            native_params = list(inspect.signature(native_selector).parameters)
            assert native_params == ["agent_id", "agent_view"]


def test_langgraph_tree_driver_does_not_push_decision_blob_into_native_state() -> None:
    from fms_bench.v2.framework_drivers import LangGraphTreeDriver

    captured_states: list[dict[str, object]] = []

    class _FakeCompiledGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, int]:
            captured_states.append(dict(state))
            return {"action": 0}

    driver = LangGraphTreeDriver()
    driver._compiled_graph = _FakeCompiledGraph()

    action = driver._select_framework_action(
        agent_id="area_0",
        agent_view={
            "agent_id": "area_0",
            "role": "area",
            "action_mask": [1, 1],
            "local_subtree_state": {"child_ids": [0, 1]},
            "feedback_view": {},
        },
    )

    assert action == 0
    assert "decision" not in captured_states[0]


def test_agentscope_tree_driver_builds_native_runtime_when_agentscope_available() -> None:
    pytest.importorskip("agentscope")
    from fms_bench.v2.framework_drivers import AgentScopeTreeDriver

    driver = AgentScopeTreeDriver()

    assert driver.framework_runtime_info["available"] is True
    assert driver.framework_runtime_info["backend_mode"] == "native_framework"
    assert driver._native_runtime is not None


def test_langgraph_tree_driver_propagates_state_memory_and_resets() -> None:
    from fms_bench.v2.framework_drivers import LangGraphTreeDriver

    driver = LangGraphTreeDriver(memory_decay=1.0, disturbance_weight=3.0)
    plant_decision = {
        "active_agents": {
            "plant": {
                "role": "plant",
                "action_mask": [1, 1],
                "valid_areas": [0, 1],
                "visible_area_summaries": {
                    "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                    "1": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                },
            }
        }
    }
    disturbed_area_decision = {
        "active_agents": {
            "area_0": {
                "role": "area",
                "action_mask": [1, 1],
                "valid_cells": [0, 1],
                "visible_cell_summaries": {
                    "0": {
                        "processing_time_scale": 1.0,
                        "active_jobs": 0.0,
                        "busy_machines": 0.0,
                        "backlog_jobs": 0.0,
                        "down_machines": 3.0,
                    },
                    "1": {
                        "processing_time_scale": 1.1,
                        "active_jobs": 0.0,
                        "busy_machines": 0.0,
                        "backlog_jobs": 0.0,
                        "down_machines": 0.0,
                    },
                },
            }
        }
    }

    first_plant = driver.select_actions(plant_decision)
    area_choice = driver.select_actions(disturbed_area_decision)
    second_plant = driver.select_actions(plant_decision)
    driver.reset_episode()
    after_reset = driver.select_actions(plant_decision)

    assert first_plant == {"plant": 0}
    assert area_choice == {"area_0": 1}
    assert second_plant == {"plant": 1}
    assert after_reset == {"plant": 0}
    assert driver.name == "external_framework_langgraph"


def test_langgraph_tree_driver_routes_centralized_plant_cell_selection_over_execution_cells() -> None:
    from fms_bench.v2.framework_drivers import LangGraphTreeDriver

    driver = LangGraphTreeDriver(memory_decay=1.0, disturbance_weight=3.0)

    action = driver._select_contract_action(
        agent_id="plant",
        agent_view=_centralized_plant_cell_selection_agent_view(),
    )

    assert action == 7


def test_langgraph_tree_driver_builds_native_state_graph_when_langgraph_available() -> None:
    pytest.importorskip("langgraph")
    from fms_bench.v2.framework_drivers import LangGraphTreeDriver

    driver = LangGraphTreeDriver()

    assert driver.framework_runtime_info["available"] is True
    assert driver.framework_runtime_info["backend_mode"] == "native_framework"
    assert driver._compiled_graph is not None
    assert type(driver._compiled_graph).__name__ == "CompiledStateGraph"


def test_langgraph_llm_tree_driver_reports_openai_compat_runtime_metadata() -> None:
    from fms_bench.v2.framework_drivers import LangGraphLLMTreeDriver

    driver = LangGraphLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
    )

    assert driver.name == "external_framework_langgraph_llm"
    assert driver.framework_runtime_info["framework_name"] == "LangGraph"
    assert driver.framework_runtime_info["llm_backend"] == "openai_compat"
    assert driver.framework_runtime_info["llm_model"] == "qwen-test"
    assert driver.framework_runtime_info["llm_base_url"] == "http://127.0.0.1:30000/v1"


def test_langgraph_llm_tree_driver_builds_openai_client_with_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    from fms_bench.v2.framework_drivers import LangGraphLLMTreeDriver

    captured: dict[str, object] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **_kwargs: None))

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))

    driver = LangGraphLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
        request_timeout=12.5,
    )

    assert driver.request_timeout == 12.5
    assert captured["timeout"] == 12.5
    assert captured["base_url"] == "http://127.0.0.1:30000/v1"


def test_agentscope_llm_tree_driver_reports_openai_compat_runtime_metadata() -> None:
    from fms_bench.v2.framework_drivers import AgentScopeLLMTreeDriver

    driver = AgentScopeLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
    )

    assert driver.name == "external_framework_agentscope_llm"
    assert driver.framework_runtime_info["framework_name"] == "AgentScope"
    assert driver.framework_runtime_info["llm_backend"] == "openai_compat"
    assert driver.framework_runtime_info["llm_model"] == "qwen-test"
    assert driver.framework_runtime_info["llm_base_url"] == "http://127.0.0.1:30000/v1"


def test_agentscope_llm_tree_driver_builds_model_with_shared_generation_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fms_bench.v2.framework_drivers import AgentScopeLLMTreeDriver

    captured: dict[str, object] = {}

    class _FakeOpenAIChatModel:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "agentscope.model",
        types.SimpleNamespace(OpenAIChatModel=_FakeOpenAIChatModel),
    )
    monkeypatch.setattr(AgentScopeLLMTreeDriver, "_detect_framework_availability", lambda self: False)

    driver = AgentScopeLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
        request_timeout=12.5,
    )
    driver._build_llm_model()

    assert captured["generate_kwargs"] == {
        "temperature": 0.0,
        "max_tokens": 512,
        "reasoning": {"effort": "none", "exclude": True},
    }
    assert captured["client_kwargs"] == {
        "base_url": "http://127.0.0.1:30000/v1",
        "timeout": 12.5,
    }


def test_agentscope_extract_chat_text_reads_dict_text_blocks() -> None:
    from fms_bench.v2.framework_drivers.agentscope_driver import _extract_chat_text

    class _Response:
        content = [
            {"type": "text", "text": '{"action": 0}'},
        ]

    assert _extract_chat_text(_Response()) == '{"action": 0}'


def test_agentscope_build_llm_messages_uses_compact_agent_payload() -> None:
    from fms_bench.v2.framework_drivers.agentscope_driver import _build_llm_messages
    from fms_bench.v2.framework_drivers.llm_prompting import (
        build_openai_compat_benchmark_user_prompt,
    )

    messages = _build_llm_messages(
        agent_id="plant",
        agent_view={"role": "plant", "action_mask": [0, 1]},
    )

    assert messages[-1]["content"] == build_openai_compat_benchmark_user_prompt(
        agent_id="plant",
        agent_view={"role": "plant", "action_mask": [0, 1]},
    )


def test_agentscope_llm_tree_driver_awaits_async_model_results() -> None:
    import asyncio

    from fms_bench.v2.framework_drivers import AgentScopeLLMTreeDriver

    class _TextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Response:
        def __init__(self, text: str) -> None:
            self.content = [_TextBlock(text)]

    class _AsyncModel:
        async def __call__(self, messages: object) -> object:
            del messages
            return _Response('{"action": 1}')

    driver = AgentScopeLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
    )
    driver._llm_model = _AsyncModel()

    action = asyncio.run(
        driver._select_llm_action_async(
            agent_id="plant",
            agent_view={
                "role": "plant",
                "action_mask": [0, 1],
            },
        )
    )

    assert action == 1


def test_agentscope_llm_tree_driver_uses_openai_compat_http_path_without_agentscope_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from fms_bench.v2.framework_drivers import AgentScopeLLMTreeDriver

    monkeypatch.setattr(AgentScopeLLMTreeDriver, "_detect_framework_availability", lambda self: False)

    driver = AgentScopeLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
        request_retries=0,
    )
    driver._llm_model = None

    async def _fake_request_openai_compat_completion_async(
        *,
        messages: list[dict[str, str]],
    ) -> str | None:
        assert messages
        return '{"action": 1}'

    driver._request_openai_compat_completion_async = _fake_request_openai_compat_completion_async  # type: ignore[method-assign]

    action = asyncio.run(
        driver._select_llm_action_async(
            agent_id="plant",
            agent_view={
                "role": "plant",
                "action_mask": [0, 1],
            },
        )
    )

    assert action == 1
    assert driver.framework_decision_audit_info["llm_decision_count"] == 1
    assert driver.framework_decision_audit_info["fallback_contract_decision_count"] == 0


def test_agentscope_llm_tree_driver_bypasses_native_pipeline_for_llm_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fms_bench.v2.framework_drivers import AgentScopeLLMTreeDriver

    monkeypatch.setattr(AgentScopeLLMTreeDriver, "_detect_framework_availability", lambda self: False)

    driver = AgentScopeLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
    )

    driver._native_runtime_enabled = lambda: True  # type: ignore[method-assign]

    def _unexpected_native_action(**kwargs: object) -> int:
        raise AssertionError(f"native pipeline should not be used: {kwargs}")

    def _fake_llm_action(**kwargs: object) -> int:
        del kwargs
        return 1

    driver._select_native_action = _unexpected_native_action  # type: ignore[method-assign]
    driver._select_llm_action = _fake_llm_action  # type: ignore[method-assign]

    action = driver._select_framework_action(
        agent_id="plant",
        agent_view={"role": "plant", "action_mask": [0, 1]},
    )

    assert action == 1


def test_agentscope_llm_tree_driver_retries_transient_model_failure_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from fms_bench.v2.framework_drivers import AgentScopeLLMTreeDriver

    class _TextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Response:
        def __init__(self, text: str) -> None:
            self.content = [{"type": "text", "text": text}]

    class _FlakyAsyncModel:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, messages: object) -> object:
            del messages
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient backend error")
            return _Response('{"action": 1}')

    monkeypatch.setattr(AgentScopeLLMTreeDriver, "_detect_framework_availability", lambda self: False)

    driver = AgentScopeLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
    )
    driver._llm_model = _FlakyAsyncModel()

    action = asyncio.run(
        driver._select_llm_action_async(
            agent_id="plant",
            agent_view={"role": "plant", "action_mask": [0, 1]},
        )
    )

    assert action == 1
    assert driver.framework_decision_audit_info["llm_decision_count"] == 1
    assert driver.framework_decision_audit_info["fallback_contract_decision_count"] == 0


def test_agentscope_llm_tree_driver_exponential_backoff_does_not_poison_recovered_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from fms_bench.v2.framework_drivers import AgentScopeLLMTreeDriver

    monkeypatch.setattr(AgentScopeLLMTreeDriver, "_detect_framework_availability", lambda self: False)

    sleeps: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        "fms_bench.v2.framework_drivers.agentscope_driver.asyncio.sleep",
        _fake_sleep,
    )

    driver = AgentScopeLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
        request_retries=3,
        retry_backoff_seconds=2.0,
        retry_backoff_max_seconds=5.0,
    )
    call_count = 0

    async def _fake_request_openai_compat_completion_async(
        *,
        messages: list[dict[str, str]],
    ) -> str | None:
        nonlocal call_count
        assert messages
        call_count += 1
        if call_count <= 2:
            raise TimeoutError("transient timeout")
        return '{"action": 1}'

    driver._request_openai_compat_completion_async = _fake_request_openai_compat_completion_async  # type: ignore[method-assign]

    action = asyncio.run(
        driver._select_llm_action_async(
            agent_id="plant",
            agent_view={"role": "plant", "action_mask": [0, 1]},
        )
    )

    assert action == 1
    assert sleeps == [2.0, 4.0]
    assert driver.framework_decision_audit_info["llm_decision_count"] == 1
    assert driver.framework_decision_audit_info["fallback_contract_decision_count"] == 0
    assert driver.framework_decision_audit_info["decision_failure_trace"] == []


def test_agentscope_llm_tree_driver_hard_times_out_hung_async_model_and_falls_back() -> None:
    import asyncio

    from fms_bench.v2.framework_drivers import AgentScopeLLMTreeDriver

    class _HungAsyncModel:
        async def __call__(self, messages: object) -> object:
            del messages
            await asyncio.Future()

    driver = AgentScopeLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
        request_timeout=0.05,
        request_retries=0,
    )
    driver._llm_model = _HungAsyncModel()

    action = asyncio.run(
        asyncio.wait_for(
            driver._select_llm_action_async(
                agent_id="plant",
                agent_view={"role": "plant", "action_mask": [1, 0]},
            ),
            timeout=0.3,
        )
    )

    assert action == 0
    assert driver.framework_decision_audit_info["llm_decision_count"] == 0
    assert driver.framework_decision_audit_info["fallback_contract_decision_count"] == 1
    assert driver.framework_decision_audit_info["decision_source_trace"] == [
        {"agent_id": "plant", "source": "fallback_contract", "action": 0, "legal_actions": [0]},
    ]


def test_agentscope_llm_tree_driver_records_llm_and_fallback_decision_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from fms_bench.v2.framework_drivers import AgentScopeLLMTreeDriver

    monkeypatch.setattr(AgentScopeLLMTreeDriver, "_detect_framework_availability", lambda self: False)

    driver = AgentScopeLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
    )

    call_count = 0

    async def _fake_request_openai_compat_completion_async(
        *,
        messages: list[dict[str, str]],
    ) -> str | None:
        nonlocal call_count
        assert messages
        call_count += 1
        if call_count == 1:
            return '{"action": 1}'
        return "not-json"

    driver._request_openai_compat_completion_async = _fake_request_openai_compat_completion_async  # type: ignore[method-assign]

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
    assert driver.framework_decision_audit_info["llm_decision_count"] == 1
    assert driver.framework_decision_audit_info["fallback_contract_decision_count"] == 1
    assert driver.framework_decision_audit_info["decision_source_trace"] == [
        {"agent_id": "plant", "source": "llm", "action": 1, "legal_actions": [1]},
        {"agent_id": "plant", "source": "fallback_contract", "action": 0, "legal_actions": [0]},
    ]


def test_langgraph_llm_tree_driver_retries_transient_openai_failure_before_fallback() -> None:
    from fms_bench.v2.framework_drivers import LangGraphLLMTreeDriver

    driver = LangGraphLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
    )

    call_count = 0

    def _fake_request_openai_compat_completion(
        *,
        messages: list[dict[str, str]],
    ) -> str | None:
        nonlocal call_count
        assert messages
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient backend error")
        return '{"action": 1}'

    driver._request_openai_compat_completion = _fake_request_openai_compat_completion  # type: ignore[method-assign]

    action = driver._select_llm_or_contract_action(
        agent_id="plant",
        agent_view={
            "role": "plant",
            "action_mask": [0, 1],
            "valid_areas": [0, 1],
            "visible_area_summaries": {
                "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                "1": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
            },
        },
    )

    assert action == 1
    assert driver.framework_decision_audit_info["llm_decision_count"] == 1
    assert driver.framework_decision_audit_info["fallback_contract_decision_count"] == 0


def test_langgraph_llm_tree_driver_exponential_backoff_does_not_poison_recovered_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fms_bench.v2.framework_drivers import LangGraphLLMTreeDriver

    sleeps: list[float] = []
    monkeypatch.setattr(
        "fms_bench.v2.framework_drivers.langgraph_driver.time.sleep",
        lambda delay: sleeps.append(delay),
    )

    driver = LangGraphLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
        request_retries=4,
        retry_backoff_seconds=2.0,
        retry_backoff_max_seconds=5.0,
    )

    call_count = 0

    def _fake_request_openai_compat_completion(
        *,
        messages: list[dict[str, str]],
    ) -> str | None:
        nonlocal call_count
        assert messages
        call_count += 1
        if call_count <= 3:
            raise TimeoutError("transient timeout")
        return '{"action": 1}'

    driver._request_openai_compat_completion = _fake_request_openai_compat_completion  # type: ignore[method-assign]

    action = driver._select_llm_or_contract_action(
        agent_id="plant",
        agent_view={
            "role": "plant",
            "action_mask": [0, 1],
            "valid_areas": [0, 1],
            "visible_area_summaries": {
                "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                "1": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
            },
        },
    )

    assert action == 1
    assert sleeps == [2.0, 4.0, 5.0]
    assert driver.framework_decision_audit_info["llm_decision_count"] == 1
    assert driver.framework_decision_audit_info["fallback_contract_decision_count"] == 0
    assert driver.framework_decision_audit_info["decision_failure_trace"] == []


def test_langgraph_llm_tree_driver_uses_openai_compat_http_path_without_openai_client() -> None:
    from fms_bench.v2.framework_drivers import LangGraphLLMTreeDriver

    driver = LangGraphLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
        request_retries=0,
    )
    driver._llm_client = None

    def _fake_request_openai_compat_completion(
        *,
        messages: list[dict[str, str]],
    ) -> str | None:
        assert messages
        return '{"action": 1}'

    driver._request_openai_compat_completion = _fake_request_openai_compat_completion  # type: ignore[method-assign]

    action = driver._select_llm_or_contract_action(
        agent_id="plant",
        agent_view={
            "role": "plant",
            "action_mask": [0, 1],
            "valid_areas": [0, 1],
            "visible_area_summaries": {
                "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                "1": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
            },
        },
    )

    assert action == 1
    assert driver.framework_decision_audit_info["llm_decision_count"] == 1
    assert driver.framework_decision_audit_info["fallback_contract_decision_count"] == 0


def test_langgraph_llm_tree_driver_rejects_illegal_llm_action_and_falls_back() -> None:
    from fms_bench.v2.framework_drivers import LangGraphLLMTreeDriver

    driver = LangGraphLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
        request_retries=0,
    )

    def _fake_request_openai_compat_completion(
        *,
        messages: list[dict[str, str]],
    ) -> str | None:
        assert messages
        return '{"action": 9}'

    driver._request_openai_compat_completion = _fake_request_openai_compat_completion  # type: ignore[method-assign]

    action = driver._select_llm_or_contract_action(
        agent_id="plant",
        agent_view={
            "role": "plant",
            "action_mask": [1, 0],
            "valid_areas": [0],
            "visible_area_summaries": {
                "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
            },
        },
    )

    assert action == 0
    assert driver.framework_decision_audit_info["llm_decision_count"] == 0
    assert driver.framework_decision_audit_info["fallback_contract_decision_count"] == 1
    assert driver.framework_decision_audit_info["decision_source_trace"] == [
        {"agent_id": "plant", "source": "fallback_contract", "action": 0, "legal_actions": [0]},
    ]


def test_langgraph_llm_tree_driver_records_llm_and_fallback_decision_sources() -> None:
    from fms_bench.v2.framework_drivers import LangGraphLLMTreeDriver

    driver = LangGraphLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
    )

    def _fake_select_openai_compat_action(
        *,
        agent_id: str,
        agent_view: Mapping[str, object],
        legal_actions: list[int],
    ) -> int | None:
        del agent_id, legal_actions
        if list(agent_view["action_mask"]) == [0, 1]:
            return 1
        return None

    driver._select_openai_compat_action = _fake_select_openai_compat_action  # type: ignore[method-assign]

    llm_action = driver._select_llm_or_contract_action(
        agent_id="plant",
        agent_view={
            "role": "plant",
            "action_mask": [0, 1],
            "valid_areas": [0, 1],
            "visible_area_summaries": {
                "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                "1": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
            },
        },
    )
    fallback_action = driver._select_llm_or_contract_action(
        agent_id="plant",
        agent_view={
            "role": "plant",
            "action_mask": [1, 0],
            "valid_areas": [0],
            "visible_area_summaries": {
                "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
            },
        },
    )

    assert llm_action == 1
    assert fallback_action == 0
    assert driver.framework_decision_audit_info["llm_decision_count"] == 1
    assert driver.framework_decision_audit_info["fallback_contract_decision_count"] == 1
    assert driver.framework_decision_audit_info["decision_source_trace"] == [
        {"agent_id": "plant", "source": "llm", "action": 1, "legal_actions": [1]},
        {"agent_id": "plant", "source": "fallback_contract", "action": 0, "legal_actions": [0]},
    ]


def test_ag2_tree_driver_propagates_context_memory_and_resets() -> None:
    from fms_bench.v2.framework_drivers import AG2TreeDriver

    driver = AG2TreeDriver(memory_decay=1.0, disturbance_weight=3.0, handoff_penalty=0.0)
    plant_decision = {
        "active_agents": {
            "plant": {
                "role": "plant",
                "action_mask": [1, 1],
                "valid_areas": [0, 1],
                "visible_area_summaries": {
                    "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                    "1": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                },
            }
        }
    }
    disturbed_area_decision = {
        "active_agents": {
            "area_0": {
                "role": "area",
                "action_mask": [1, 1],
                "valid_cells": [0, 1],
                "visible_cell_summaries": {
                    "0": {
                        "processing_time_scale": 1.0,
                        "active_jobs": 0.0,
                        "busy_machines": 0.0,
                        "backlog_jobs": 0.0,
                        "down_machines": 3.0,
                    },
                    "1": {
                        "processing_time_scale": 1.1,
                        "active_jobs": 0.0,
                        "busy_machines": 0.0,
                        "backlog_jobs": 0.0,
                        "down_machines": 0.0,
                    },
                },
            }
        }
    }

    first_plant = driver.select_actions(plant_decision)
    area_choice = driver.select_actions(disturbed_area_decision)
    second_plant = driver.select_actions(plant_decision)
    driver.reset_episode()
    after_reset = driver.select_actions(plant_decision)

    assert first_plant == {"plant": 0}
    assert area_choice == {"area_0": 1}
    assert second_plant == {"plant": 1}
    assert after_reset == {"plant": 0}
    assert driver.name == "external_framework_ag2"


def test_ag2_tree_driver_enables_native_runtime_when_framework_becomes_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fms_bench.v2.framework_drivers import AG2TreeDriver

    built_tags: list[str] = []

    def _fake_detect(self: AG2TreeDriver) -> bool:
        return True

    def _fake_build(self: AG2TreeDriver) -> dict[str, object]:
        tag = f"ag2-runtime-{len(built_tags)}"
        built_tags.append(tag)
        return {"runtime_tag": tag}

    monkeypatch.setattr(AG2TreeDriver, "_detect_framework_availability", _fake_detect)
    monkeypatch.setattr(AG2TreeDriver, "_build_native_runtime", _fake_build, raising=False)

    driver = AG2TreeDriver()

    assert driver.framework_runtime_info["available"] is True
    assert driver.framework_runtime_info["backend_mode"] == "native_framework"
    assert driver._native_runtime == {"runtime_tag": "ag2-runtime-0"}

    driver.reset_episode()

    assert driver.framework_runtime_info["backend_mode"] == "native_framework"
    assert driver._native_runtime == {"runtime_tag": "ag2-runtime-1"}
    assert built_tags == ["ag2-runtime-0", "ag2-runtime-1"]


def test_google_adk_tree_driver_uses_workflow_memory_and_resets() -> None:
    from fms_bench.v2.framework_drivers import GoogleADKTreeDriver

    driver = GoogleADKTreeDriver(loop_penalty=5.0)
    decision = {
        "active_agents": {
            "plant": {
                "role": "plant",
                "action_mask": [1, 1],
                "valid_areas": [0, 1],
                "visible_area_summaries": {
                    "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                    "1": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                },
            },
            "area_0": {
                "role": "area",
                "action_mask": [1, 1],
                "valid_cells": [0, 1],
                "visible_cell_summaries": {
                    "0": {"processing_time_scale": 1.0, "active_jobs": 0.0, "busy_machines": 0.0, "backlog_jobs": 0.0},
                    "1": {"processing_time_scale": 1.0, "active_jobs": 0.0, "busy_machines": 0.0, "backlog_jobs": 0.0},
                },
            },
        }
    }

    first = driver.select_actions(decision)
    second = driver.select_actions(decision)
    driver.reset_episode()
    after_reset = driver.select_actions(decision)

    assert first == {"plant": 0, "area_0": 0}
    assert second == {"plant": 1, "area_0": 1}
    assert after_reset == {"plant": 0, "area_0": 0}
    assert driver.name == "external_framework_google_adk"


def test_google_adk_tree_driver_builds_native_runtime_when_google_adk_available() -> None:
    pytest.importorskip("google.adk")
    from fms_bench.v2.framework_drivers import GoogleADKTreeDriver

    driver = GoogleADKTreeDriver()

    assert driver.framework_runtime_info["available"] is True
    assert driver.framework_runtime_info["backend_mode"] == "native_framework"
    assert driver._native_runtime is not None


def test_google_adk_tree_driver_rebuilds_unique_runtime_namespace_per_episode() -> None:
    pytest.importorskip("google.adk")
    from fms_bench.v2.framework_drivers import GoogleADKTreeDriver

    driver = GoogleADKTreeDriver()
    assert isinstance(driver._native_runtime, dict)
    first_app_name = str(driver._native_runtime["app_name"])
    first_user_id = str(driver._native_runtime["user_id"])
    first_runtime_tag = str(driver._native_runtime["runtime_tag"])

    driver.reset_episode()

    assert isinstance(driver._native_runtime, dict)
    assert str(driver._native_runtime["app_name"]) != first_app_name
    assert str(driver._native_runtime["user_id"]) != first_user_id
    assert str(driver._native_runtime["runtime_tag"]) != first_runtime_tag


def test_crewai_tree_driver_uses_manager_memory_and_resets() -> None:
    from fms_bench.v2.framework_drivers import CrewAITreeDriver

    driver = CrewAITreeDriver(manager_penalty=5.0)
    decision = {
        "active_agents": {
            "plant": {
                "role": "plant",
                "action_mask": [1, 1],
                "valid_areas": [0, 1],
                "visible_area_summaries": {
                    "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                    "1": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                },
            },
            "area_0": {
                "role": "area",
                "action_mask": [1, 1],
                "valid_cells": [0, 1],
                "visible_cell_summaries": {
                    "0": {"processing_time_scale": 1.0, "active_jobs": 0.0, "busy_machines": 0.0, "backlog_jobs": 0.0},
                    "1": {"processing_time_scale": 1.0, "active_jobs": 0.0, "busy_machines": 0.0, "backlog_jobs": 0.0},
                },
            },
        }
    }

    first = driver.select_actions(decision)
    second = driver.select_actions(decision)
    driver.reset_episode()
    after_reset = driver.select_actions(decision)

    assert first == {"plant": 0, "area_0": 0}
    assert second == {"plant": 1, "area_0": 1}
    assert after_reset == {"plant": 0, "area_0": 0}
    assert driver.name == "external_framework_crewai"


def test_crewai_tree_driver_enables_native_runtime_when_framework_becomes_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fms_bench.v2.framework_drivers import CrewAITreeDriver

    built_tags: list[str] = []

    def _fake_detect(self: CrewAITreeDriver) -> bool:
        return True

    def _fake_build(self: CrewAITreeDriver) -> dict[str, object]:
        tag = f"crewai-runtime-{len(built_tags)}"
        built_tags.append(tag)
        return {"runtime_tag": tag}

    monkeypatch.setattr(CrewAITreeDriver, "_detect_framework_availability", _fake_detect)
    monkeypatch.setattr(CrewAITreeDriver, "_build_native_runtime", _fake_build, raising=False)

    driver = CrewAITreeDriver()

    assert driver.framework_runtime_info["available"] is True
    assert driver.framework_runtime_info["backend_mode"] == "native_framework"
    assert driver._native_runtime == {"runtime_tag": "crewai-runtime-0"}

    driver.reset_episode()

    assert driver.framework_runtime_info["backend_mode"] == "native_framework"
    assert driver._native_runtime == {"runtime_tag": "crewai-runtime-1"}
    assert built_tags == ["crewai-runtime-0", "crewai-runtime-1"]


def test_crewai_tree_driver_suppresses_native_flow_console_noise(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fms_bench.v2.framework_drivers import CrewAITreeDriver

    class _NoisyFlow:
        def __init__(self, *, payload: Mapping[str, object]) -> None:
            self.payload = dict(payload)

        def kickoff(self) -> dict[str, int]:
            import sys

            print("crewai stdout noise")
            print("crewai stderr noise", file=sys.stderr)
            return {"action": 0}

    monkeypatch.setattr(CrewAITreeDriver, "_detect_framework_availability", lambda self: True)
    monkeypatch.setattr(
        CrewAITreeDriver,
        "_build_native_runtime",
        lambda self: {"runtime_tag": "crewai-runtime-test", "flow_class": _NoisyFlow},
        raising=False,
    )

    driver = CrewAITreeDriver()
    decision = {
        "active_agents": {
            "plant": {
                "role": "plant",
                "action_mask": [1, 0],
                "valid_areas": [0],
                "visible_area_summaries": {
                    "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                },
            }
        }
    }

    assert driver.select_actions(decision) == {"plant": 0}
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_spade_tree_driver_uses_message_memory_and_resets() -> None:
    from fms_bench.v2.framework_drivers import SPADETreeDriver

    driver = SPADETreeDriver(memory_decay=1.0, disturbance_weight=3.0)
    plant_decision = {
        "active_agents": {
            "plant": {
                "role": "plant",
                "action_mask": [1, 1],
                "valid_areas": [0, 1],
                "visible_area_summaries": {
                    "0": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                    "1": {"active_jobs": 0.0, "backlog_jobs": 0.0, "energy_kwh": 0.0},
                },
            }
        }
    }
    disturbed_area_decision = {
        "active_agents": {
            "area_0": {
                "role": "area",
                "action_mask": [1, 1],
                "valid_cells": [0, 1],
                "visible_cell_summaries": {
                    "0": {
                        "processing_time_scale": 1.0,
                        "active_jobs": 0.0,
                        "busy_machines": 0.0,
                        "backlog_jobs": 0.0,
                        "down_machines": 3.0,
                    },
                    "1": {
                        "processing_time_scale": 1.1,
                        "active_jobs": 0.0,
                        "busy_machines": 0.0,
                        "backlog_jobs": 0.0,
                        "down_machines": 0.0,
                    },
                },
            }
        }
    }

    first_plant = driver.select_actions(plant_decision)
    area_choice = driver.select_actions(disturbed_area_decision)
    second_plant = driver.select_actions(plant_decision)
    driver.reset_episode()
    after_reset = driver.select_actions(plant_decision)

    assert first_plant == {"plant": 0}
    assert area_choice == {"area_0": 1}
    assert second_plant == {"plant": 1}
    assert after_reset == {"plant": 0}
    assert driver.name == "external_framework_spade"


def test_spade_tree_driver_builds_native_runtime_when_spade_available() -> None:
    pytest.importorskip("spade")
    from fms_bench.v2.framework_drivers import SPADETreeDriver

    driver = SPADETreeDriver()

    assert driver.framework_runtime_info["available"] is True
    assert driver.framework_runtime_info["backend_mode"] == "native_framework"
    assert driver._native_runtime is not None


def test_spade_tree_driver_rebuilds_unique_agent_jids_per_episode() -> None:
    pytest.importorskip("spade")
    from fms_bench.v2.framework_drivers import SPADETreeDriver

    driver = SPADETreeDriver()
    first_agent = driver._ensure_native_agent("plant")
    first_jid = str(first_agent.jid)

    driver.reset_episode()

    second_agent = driver._ensure_native_agent("plant")
    second_jid = str(second_agent.jid)

    assert second_jid != first_jid


def test_framework_driver_reset_episode_closes_previous_native_runtime() -> None:
    from fms_bench.v2.framework_drivers.base import FrameworkBackedTreeDriver

    class _ClosableRuntime:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class _ClosableRuntimeDriver(FrameworkBackedTreeDriver):
        framework_name = "Closable Runtime"
        controller_name = "external_framework_closable_runtime"

        def __init__(self) -> None:
            self.created_runtimes: list[_ClosableRuntime] = []
            super().__init__()

        def _detect_framework_availability(self) -> bool:
            return True

        def reset_episode(self) -> None:
            super().reset_episode()
            runtime = _ClosableRuntime()
            self.created_runtimes.append(runtime)
            self._native_runtime = runtime

        def _select_framework_action(
            self,
            *,
            agent_id: str,
            agent_view: Mapping[str, object],
        ) -> int:
            del agent_id, agent_view
            return 0

    driver = _ClosableRuntimeDriver()
    first_runtime = driver.created_runtimes[0]

    driver.reset_episode()

    assert first_runtime.closed is True
    assert len(driver.created_runtimes) == 2
    assert driver.created_runtimes[1].closed is False


def test_framework_driver_reset_episode_closes_nested_native_runtime_members() -> None:
    from fms_bench.v2.framework_drivers.base import FrameworkBackedTreeDriver

    class _CloseRecorder:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class _StopRecorder:
        def __init__(self) -> None:
            self.stopped = False

        async def stop(self) -> None:
            self.stopped = True

    class _NestedRuntimeDriver(FrameworkBackedTreeDriver):
        framework_name = "Nested Runtime"
        controller_name = "external_framework_nested_runtime"

        def __init__(self) -> None:
            self.created_runtimes: list[dict[str, object]] = []
            super().__init__()

        def _detect_framework_availability(self) -> bool:
            return True

        def reset_episode(self) -> None:
            super().reset_episode()
            runtime = {
                "runner": _CloseRecorder(),
                "agents": {
                    "plant": _StopRecorder(),
                },
            }
            self.created_runtimes.append(runtime)
            self._native_runtime = runtime

        def _select_framework_action(
            self,
            *,
            agent_id: str,
            agent_view: Mapping[str, object],
        ) -> int:
            del agent_id, agent_view
            return 0

    driver = _NestedRuntimeDriver()
    first_runtime = driver.created_runtimes[0]
    first_runner = first_runtime["runner"]
    first_agent = first_runtime["agents"]["plant"]  # type: ignore[index]

    driver.reset_episode()

    assert isinstance(first_runner, _CloseRecorder)
    assert isinstance(first_agent, _StopRecorder)
    assert first_runner.closed is True
    assert first_agent.stopped is True


def test_subprocess_external_driver_runs_jsonl_controller_loop(tmp_path: Path) -> None:
    from fms_bench.v2.external_controller_adapter import (
        ExternalControllerSession,
        SubprocessExternalControllerDriver,
        run_external_controller_loop,
    )

    controller_path = _write_subprocess_controller(tmp_path)
    env = TreeRuntimeEnv(
        _ordertrap_cell_configs(),
        area_to_cells={0: [0, 1], 1: [2]},
        config=_make_config(),
        seed=13,
    )
    driver = SubprocessExternalControllerDriver(
        [sys.executable, str(controller_path), "--mode", "last_valid"],
        controller_name="external_subprocess_last_valid",
    )
    try:
        session = ExternalControllerSession(env)
        session.reset(seed=13)

        transitions = run_external_controller_loop(session, driver, max_steps=2)

        assert transitions[0]["submitted_actions"] == {"plant": 1}
        assert transitions[0]["current_decision"]["agents_to_act"] == ["area_1"]
        assert transitions[1]["submitted_actions"] == {"area_1": 2}
    finally:
        driver.close()
        env.close()


def test_subprocess_external_driver_reset_episode_restarts_fresh_state(
    tmp_path: Path,
) -> None:
    from fms_bench.v2.external_controller_adapter import SubprocessExternalControllerDriver

    controller_path = _write_subprocess_controller(tmp_path)
    env, session = _make_external_session()
    driver = SubprocessExternalControllerDriver(
        [sys.executable, str(controller_path), "--mode", "first_decision_last_then_first"],
        controller_name="external_subprocess_resettable",
    )
    try:
        first_decision = session.reset(seed=13)
        first_actions = driver.select_actions(first_decision)
        session.step(first_actions)

        driver.reset_episode()

        second_decision = session.reset(seed=13)
        second_actions = driver.select_actions(second_decision)

        assert first_actions == {"plant": 1}
        assert second_actions == {"plant": 1}
    finally:
        driver.close()
        env.close()


def test_subprocess_mas_runner_executes_small_suite_with_controller_command(
    tmp_path: Path,
) -> None:
    controller_path = _write_subprocess_controller(tmp_path)
    runner_path = PROJECT_ROOT / "examples" / "run_tree_runtime_v2_subprocess_mas.py"
    output_dir = tmp_path / "subprocess_runner_output"

    result = subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "--suite",
            "small",
            "--case-limit",
            "1",
            "--seed-limit",
            "1",
            "--output-dir",
            str(output_dir),
            "--controller-name",
            "external_subprocess_first_valid",
            "--controller-command",
            sys.executable,
            str(controller_path),
            "--mode",
            "first_valid",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary_path = Path(result.stdout.strip().splitlines()[-1].strip())
    assert summary_path.exists()


def test_subprocess_external_driver_runs_jsonl_controller_loop_from_controller_command_string(
    tmp_path: Path,
) -> None:
    from fms_bench.v2.external_controller_adapter import (
        ExternalControllerSession,
        SubprocessExternalControllerDriver,
        run_external_controller_loop,
    )

    controller_command = _write_subprocess_controller_command(
        tmp_path,
        mode="last_valid",
    )
    env = TreeRuntimeEnv(
        _ordertrap_cell_configs(),
        area_to_cells={0: [0, 1], 1: [2]},
        config=_make_config(),
        seed=13,
    )
    driver = SubprocessExternalControllerDriver(
        controller_command,
        controller_name="external_subprocess_last_valid_string_command",
    )
    try:
        session = ExternalControllerSession(env)
        session.reset(seed=13)

        transitions = run_external_controller_loop(session, driver, max_steps=2)

        assert transitions[0]["submitted_actions"] == {"plant": 1}
        assert transitions[0]["current_decision"]["agents_to_act"] == ["area_1"]
        assert transitions[1]["submitted_actions"] == {"area_1": 2}
    finally:
        driver.close()
        env.close()


def test_subprocess_external_driver_reset_episode_restarts_fresh_state_from_controller_command_string(
    tmp_path: Path,
) -> None:
    from fms_bench.v2.external_controller_adapter import SubprocessExternalControllerDriver

    controller_command = _write_subprocess_controller_command(
        tmp_path,
        mode="first_decision_last_then_first",
    )
    env, session = _make_external_session()
    driver = SubprocessExternalControllerDriver(
        controller_command,
        controller_name="external_subprocess_resettable_string_command",
    )
    try:
        first_decision = session.reset(seed=13)
        first_actions = driver.select_actions(first_decision)
        session.step(first_actions)

        driver.reset_episode()

        second_decision = session.reset(seed=13)
        second_actions = driver.select_actions(second_decision)

        assert first_actions == {"plant": 1}
        assert second_actions == {"plant": 1}
    finally:
        driver.close()
        env.close()


def test_subprocess_mas_runner_executes_small_suite_with_single_controller_command_string(
    tmp_path: Path,
) -> None:
    controller_command = _write_subprocess_controller_command(
        tmp_path,
        mode="first_valid",
    )
    runner_path = PROJECT_ROOT / "examples" / "run_tree_runtime_v2_subprocess_mas.py"
    output_dir = tmp_path / "subprocess_runner_string_command_output"

    result = subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "--suite",
            "small",
            "--case-limit",
            "1",
            "--seed-limit",
            "1",
            "--output-dir",
            str(output_dir),
            "--controller-name",
            "external_subprocess_first_valid",
            "--controller-command",
            controller_command,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary_path = Path(result.stdout.strip().splitlines()[-1].strip())
    assert summary_path.exists()


def test_example_mask_first_driver_works_with_minimal_decision_payload() -> None:
    from examples.run_tree_runtime_v2_external_adapter import mask_first_driver

    decision = {
        "active_agents": {
            "plant": {"action_mask": [0, 1, 1]},
            "area_0": {"action_mask": [0, 0, 1, 1]},
        }
    }

    actions = mask_first_driver(decision)

    assert actions == {"plant": 1, "area_0": 2}


def test_subprocess_mask_first_controller_works_with_minimal_decision_payload() -> None:
    controller_script = PROJECT_ROOT / "examples" / "subprocess_mask_first_controller.py"
    decision = {
        "active_agents": {
            "plant": {"action_mask": [0, 1, 1]},
            "area_0": {"action_mask": [0, 0, 1]},
        }
    }

    result = subprocess.run(
        [sys.executable, str(controller_script)],
        cwd=PROJECT_ROOT,
        input=json.dumps(decision) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    actions = json.loads(result.stdout.strip())
    assert actions == {"plant": 1, "area_0": 2}


def test_subprocess_openai_compat_controller_works_with_minimal_decision_payload() -> None:
    controller_script = PROJECT_ROOT / "examples" / "subprocess_openai_compat_controller.py"
    decision = {
        "active_agents": {
            "plant": {"action_mask": [0, 1, 1]},
            "area_0": {"action_mask": [0, 0, 1]},
        }
    }
    server = _OpenAICompatTestServer()
    server.start()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(controller_script),
                "--model",
                "dummy-model",
                "--base-url",
                server.base_url,
                "--api-key",
                "EMPTY",
            ],
            cwd=PROJECT_ROOT,
            input=json.dumps(decision) + "\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
    finally:
        server.close()

    assert result.returncode == 0, result.stderr
    actions = json.loads(result.stdout.strip())
    assert actions == {"plant": 1, "area_0": 2}
    assert server.requests
    assert server.requests[0]["model"] == "dummy-model"


def test_subprocess_openai_compat_controller_accepts_fenced_json_mapping() -> None:
    controller_script = PROJECT_ROOT / "examples" / "subprocess_openai_compat_controller.py"
    decision = {
        "active_agents": {
            "plant": {"action_mask": [0, 1, 1]},
            "area_0": {"action_mask": [0, 0, 1]},
        }
    }
    server = _OpenAICompatTestServer(
        response_formatter=lambda _decision, actions: f"```json\n{json.dumps(actions)}\n```",
    )
    server.start()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(controller_script),
                "--model",
                "dummy-model",
                "--base-url",
                server.base_url,
                "--api-key",
                "EMPTY",
            ],
            cwd=PROJECT_ROOT,
            input=json.dumps(decision) + "\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
    finally:
        server.close()

    assert result.returncode == 0, result.stderr
    actions = json.loads(result.stdout.strip())
    assert actions == {"plant": 1, "area_0": 2}


def test_subprocess_openai_compat_controller_falls_back_when_backend_unavailable() -> None:
    controller_script = PROJECT_ROOT / "examples" / "subprocess_openai_compat_controller.py"
    decision = {
        "active_agents": {
            "plant": {"action_mask": [1, 0, 1]},
        }
    }
    result = subprocess.run(
        [
            sys.executable,
            str(controller_script),
            "--model",
            "dummy-model",
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--api-key",
            "EMPTY",
            "--timeout",
            "0.2",
            "--on-invalid-response",
            "first_valid",
        ],
        cwd=PROJECT_ROOT,
        input=json.dumps(decision) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    actions = json.loads(result.stdout.strip())
    assert actions == {"plant": 0}


def test_subprocess_mas_runner_executes_small_suite_with_openai_compat_controller(
    tmp_path: Path,
) -> None:
    controller_script = PROJECT_ROOT / "examples" / "subprocess_openai_compat_controller.py"
    runner_path = PROJECT_ROOT / "examples" / "run_tree_runtime_v2_subprocess_mas.py"
    output_dir = tmp_path / "subprocess_openai_compat_runner_output"
    server = _OpenAICompatTestServer()
    server.start()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(runner_path),
                "--suite",
                "small",
                "--case-limit",
                "1",
                "--seed-limit",
                "1",
                "--output-dir",
                str(output_dir),
                "--controller-name",
                "external_subprocess_openai_compat",
                "--controller-command",
                sys.executable,
                str(controller_script),
                "--model",
                "dummy-model",
                "--base-url",
                server.base_url,
                "--api-key",
                "EMPTY",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        server.close()

    assert result.returncode == 0, result.stderr
    summary_path = Path(result.stdout.strip().splitlines()[-1].strip())
    assert summary_path.exists()
    assert server.requests
