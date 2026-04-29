from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sample_agent_view() -> dict[str, object]:
    return {
        "role": "plant",
        "plant_decision_kind": "backlog_selection",
        "action_mask": [1, 0, 1],
        "valid_backlog_ids": [11, 22],
        "visible_backlog_candidates": [
            {"job_id": 11, "stage_id": 0, "valid_areas": [0]},
            {"job_id": 99, "stage_id": 0, "valid_areas": [1]},
            {"job_id": 22, "stage_id": 1, "valid_areas": [0, 1]},
        ],
        "activation": {"functional_loop": "routing"},
        "feedback_view": {
            "progress": {"completed_jobs": 9},
            "completion": {"status_counts": {"accepted": 2}},
        },
        "parent_context": {"message": {"kind": "award"}},
        "observation": [1.0, 2.0, 3.0],
        "local_env_events": [{"type": "failure"}],
    }


def test_shared_openai_compat_prompt_helper_builds_expected_messages() -> None:
    from fms_bench.v2.framework_drivers.llm_prompting import (
        OPENAI_COMPAT_BENCHMARK_SYSTEM_PROMPT,
        build_openai_compat_benchmark_prompt_payload,
        build_openai_compat_benchmark_messages,
    )

    expected_payload = {
        "agent": "plant",
        "role": "plant",
        "kind": "backlog_selection",
        "legal_actions": [0, 2],
        "decision_factors": [
            "Prefer backlog choices that keep more downstream routing paths feasible.",
            "Account for recent reject, blocking, and overload signals in working_memory.",
            "Avoid choices that increase completion debt through weak late-stage routes.",
        ],
        "options": [
            {
                "action": 0,
                "job_id": 11,
                "stage_id": 0,
                "valid_areas": [0],
            },
            {
                "action": 2,
                "job_id": 22,
                "stage_id": 1,
                "valid_areas": [0, 1],
            },
        ],
        "context": {
            "feedback": {
                "progress": {"completed_jobs": 9},
                "completion_status_counts": {"accepted": 2},
            },
            "recent_event_types": ["failure"],
        },
    }

    assert build_openai_compat_benchmark_prompt_payload(
        agent_id="plant",
        agent_view=_sample_agent_view(),
    ) == expected_payload

    messages = build_openai_compat_benchmark_messages(
        agent_id="plant",
        agent_view=_sample_agent_view(),
    )

    assert messages == [
        {
            "role": "system",
            "content": OPENAI_COMPAT_BENCHMARK_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": '{"agent":"plant","role":"plant","kind":"backlog_selection","legal_actions":[0,2],"decision_factors":["Prefer backlog choices that keep more downstream routing paths feasible.","Account for recent reject, blocking, and overload signals in working_memory.","Avoid choices that increase completion debt through weak late-stage routes."],"options":[{"action":0,"job_id":11,"stage_id":0,"valid_areas":[0]},{"action":2,"job_id":22,"stage_id":1,"valid_areas":[0,1]}],"context":{"feedback":{"progress":{"completed_jobs":9},"completion_status_counts":{"accepted":2}},"recent_event_types":["failure"]}}',
        },
    ]


def test_openai_compat_system_prompt_uses_hard_json_only_contract() -> None:
    from fms_bench.v2.framework_drivers.llm_prompting import (
        OPENAI_COMPAT_BENCHMARK_SYSTEM_PROMPT,
    )

    assert "CRITICAL OUTPUT CONTRACT" in OPENAI_COMPAT_BENCHMARK_SYSTEM_PROMPT
    assert 'exactly one JSON object' in OPENAI_COMPAT_BENCHMARK_SYSTEM_PROMPT
    assert "legal_actions" in OPENAI_COMPAT_BENCHMARK_SYSTEM_PROMPT
    assert "decision_factors" in OPENAI_COMPAT_BENCHMARK_SYSTEM_PROMPT
    assert 'Valid schema: {"action": <index>}' in OPENAI_COMPAT_BENCHMARK_SYSTEM_PROMPT


def test_openai_compat_generation_budget_is_large_enough_for_json_only_replies() -> None:
    from fms_bench.v2.framework_drivers.llm_prompting import (
        OPENAI_COMPAT_BENCHMARK_MAX_TOKENS,
        build_openai_compat_benchmark_generation_kwargs,
    )

    assert OPENAI_COMPAT_BENCHMARK_MAX_TOKENS == 512
    assert build_openai_compat_benchmark_generation_kwargs() == {
        "temperature": 0.0,
        "max_tokens": 512,
    }
    assert build_openai_compat_benchmark_generation_kwargs(model_name="gpt-5.4") == {
        "temperature": 0.0,
        "max_tokens": 512,
    }
    assert build_openai_compat_benchmark_generation_kwargs(
        model_name="qwen/qwen3.5-plus-02-15",
    ) == {
        "temperature": 0.0,
        "max_tokens": 512,
        "reasoning": {
            "effort": "none",
            "exclude": True,
        },
    }


def test_langgraph_qwen_openai_compat_http_request_disables_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from fms_bench.v2.framework_drivers.langgraph_driver import LangGraphLLMTreeDriver

    captured_request_payload: dict[str, object] = {}

    class _FakeHTTPResponse:
        def __enter__(self) -> "_FakeHTTPResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"action": 0}',
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def _fake_urlopen(request, timeout=None):  # type: ignore[no-untyped-def]
        del timeout
        captured_request_payload.update(json.loads(request.data.decode("utf-8")))
        return _FakeHTTPResponse()

    monkeypatch.setattr(
        "fms_bench.v2.framework_drivers.langgraph_driver.urllib_request.urlopen",
        _fake_urlopen,
    )
    monkeypatch.setattr(
        LangGraphLLMTreeDriver,
        "_detect_framework_availability",
        lambda self: False,
    )

    driver = LangGraphLLMTreeDriver(
        model_name="qwen/qwen3.5-plus-02-15",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
        request_retries=0,
    )

    content = driver._request_openai_compat_completion(
        messages=[
            {
                "role": "system",
                "content": "Return JSON only.",
            },
            {
                "role": "user",
                "content": '{"action": 0}',
            },
        ],
    )

    assert content == '{"action": 0}'
    assert captured_request_payload["reasoning"] == {
        "effort": "none",
        "exclude": True,
    }


def test_prompt_payload_drops_heavy_controller_fields_and_shrinks_serialized_size() -> None:
    import json

    from fms_bench.v2.framework_drivers.llm_prompting import (
        build_openai_compat_benchmark_prompt_payload,
    )

    compact = build_openai_compat_benchmark_prompt_payload(
        agent_id="plant",
        agent_view=_sample_agent_view(),
    )

    serialized_full = json.dumps(
        {
            "agent_id": "plant",
            "agent_view": _sample_agent_view(),
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    serialized_compact = json.dumps(
        compact,
        ensure_ascii=True,
        separators=(",", ":"),
    )

    assert "activation" not in compact
    assert "feedback_view" not in compact
    assert "parent_context" not in compact
    assert compact["context"]["feedback"] == {
        "progress": {"completed_jobs": 9},
        "completion_status_counts": {"accepted": 2},
    }
    assert len(serialized_compact) - len(serialized_full) < 64


def test_cell_dispatch_prompt_payload_compacts_machine_state_per_legal_option() -> None:
    from fms_bench.v2.framework_drivers.llm_prompting import (
        build_openai_compat_benchmark_prompt_payload,
    )

    payload = build_openai_compat_benchmark_prompt_payload(
        agent_id="cell_3",
        agent_view={
            "role": "cell",
            "cell_id": 3,
            "action_mask": [0, 1, 1],
            "activation": {"functional_loop": "local_regulation"},
            "dispatch_options": [
                {"machine_id": 7, "proc_time": 9.0, "operation_id": "ignore-me"},
                {"machine_id": 8, "proc_time": 3.5, "operation_id": "op-a"},
                {"machine_id": 9, "proc_time": 4.0, "operation_id": "op-b"},
            ],
            "visible_machine_summaries": {
                "8": {"status": "busy", "remaining_time": 6.0, "down_time": 0.0, "energy_kwh": 12.0},
                "9": {"status": "idle", "remaining_time": 0.0, "down_time": 0.0, "energy_kwh": 10.0},
            },
        },
    )

    assert payload == {
        "agent": "cell_3",
        "role": "cell",
        "kind": "local_regulation",
        "legal_actions": [1, 2],
        "decision_factors": [
            "Compare proc_time with machine remaining/down time and current WIP pressure.",
            "Avoid dispatches that amplify blocking, setup burden, or repeated failure exposure.",
            "Prefer the machine that is feasible now and least likely to cause near-term regret.",
        ],
        "options": [
            {
                "action": 1,
                "machine_id": 8,
                "proc_time": 3.5,
                "machine": {
                    "status": "busy",
                    "remaining_time": 6.0,
                    "down_time": 0.0,
                },
            },
            {
                "action": 2,
                "machine_id": 9,
                "proc_time": 4.0,
                "machine": {
                    "status": "idle",
                    "remaining_time": 0.0,
                    "down_time": 0.0,
                },
            },
        ],
    }


def test_prompt_payload_compacts_working_memory_without_changing_legal_actions() -> None:
    from fms_bench.v2.framework_drivers.llm_prompting import (
        build_openai_compat_benchmark_prompt_payload,
    )

    agent_view = _sample_agent_view()
    agent_view["working_memory"] = {
        "recent_outcomes": [
            {
                "step": 4,
                "kind": "backlog_selection",
                "action": 2,
                "agent": "plant",
                "selected_target": "job:22",
                "observed_feedback": {
                    "progress": {"completed_jobs": 8},
                    "capacity": {"local_blocked": False, "wip_load": 3},
                    "completion_status_counts": {"accepted": 2},
                },
                "visible_commitments": [{"contract_id": "c-1", "status": "accepted"}],
                "event_types": ["repair"],
            }
        ],
        "open_commitments": [
            {"contract_id": "c-1", "status": "accepted", "job_id": 22},
        ],
        "rolled_summary": {
            "compressed_count": 7,
            "decision_counts_by_kind": {"backlog_selection": 7},
            "top_repeated_targets": [{"target": "job:22", "count": 5}],
            "mean_wip_load": 2.5,
        },
    }

    payload = build_openai_compat_benchmark_prompt_payload(
        agent_id="plant",
        agent_view=agent_view,
    )

    assert payload["legal_actions"] == [0, 2]
    assert payload["options"] == [
        {"action": 0, "job_id": 11, "stage_id": 0, "valid_areas": [0]},
        {"action": 2, "job_id": 22, "stage_id": 1, "valid_areas": [0, 1]},
    ]
    assert payload["context"]["working_memory"] == {
        "recent_outcomes": [
            {
                "step": 4,
                "kind": "backlog_selection",
                "action": 2,
                "agent": "plant",
                "selected_target": "job:22",
                "observed_feedback": {
                    "progress": {"completed_jobs": 8},
                    "capacity": {"local_blocked": False, "wip_load": 3},
                    "completion_status_counts": {"accepted": 2},
                },
                "event_types": ["repair"],
            }
        ],
        "open_commitments": [{"contract_id": "c-1", "status": "accepted", "job_id": 22}],
        "rolled_summary": {
            "compressed_count": 7,
            "decision_counts_by_kind": {"backlog_selection": 7},
            "mean_wip_load": 2.5,
            "top_repeated_targets": [{"target": "job:22", "count": 5}],
        },
    }


def test_langgraph_llm_runtime_uses_shared_prompt_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fms_bench.v2.framework_drivers.langgraph_driver import (
        LangGraphLLMTreeDriver,
        _build_llm_messages,
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        LangGraphLLMTreeDriver,
        "_detect_framework_availability",
        lambda self: False,
    )
    monkeypatch.setattr(
        LangGraphLLMTreeDriver,
        "_request_openai_compat_completion",
        lambda self, *, messages: captured.setdefault("messages", messages) and '{"action": 2}',
    )

    driver = LangGraphLLMTreeDriver(
        model_name="qwen-test",
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY",
        request_retries=0,
    )

    action = driver._select_llm_or_contract_action(
        agent_id="plant",
        agent_view=_sample_agent_view(),
    )

    assert action == 2
    assert captured["messages"] == _build_llm_messages(
        agent_id="plant",
        agent_view=_sample_agent_view(),
    )


def test_framework_base_injects_bounded_working_memory_between_decisions() -> None:
    from fms_bench.v2.framework_drivers.base import FrameworkBackedTreeDriver

    seen_agent_views: list[dict[str, object]] = []

    class _MemoryCaptureDriver(FrameworkBackedTreeDriver):
        framework_name = "capture"

        def _select_framework_action(self, *, agent_id: str, agent_view: dict[str, object]) -> int:
            del agent_id
            seen_agent_views.append(dict(agent_view))
            return 1

    driver = _MemoryCaptureDriver()
    decision = {
        "active_agents": {
            "cell_0": {
                "agent_id": "cell_0",
                "role": "cell",
                "cell_id": 0,
                "action_mask": [1, 1],
                "activation": {"functional_loop": "commitment_management"},
                "action_semantics": {
                    "action_type": "commitment_management",
                    "action_targets": [
                        {"action_id": 0, "action_kind": "accept"},
                        {"action_id": 1, "action_kind": "reject"},
                    ],
                },
                "feedback_view": {
                    "progress": {"active_jobs": 2},
                    "capacity_risk": {
                        "wip_load": 2,
                        "local_blocked": True,
                        "machine_health_summary": {
                            "busy_machines": 1,
                            "down_machines": 0,
                        },
                    },
                    "completion": {
                        "status_counts": {"accepted": 1},
                        "tracked_contracts": [
                            {
                                "contract_id": "c-1",
                                "job_id": 7,
                                "status": "accepted",
                                "completed_operations": 1,
                                "total_operations": 4,
                            }
                        ],
                    },
                },
                "parent_context": {
                    "current_contract": {
                        "contract_id": "c-1",
                        "job_id": 7,
                        "status": "accepted",
                        "completed_operations": 1,
                        "total_operations": 4,
                    }
                },
                "local_env_events": [{"type": "failure"}],
            }
        }
    }

    for _ in range(8):
        driver.select_actions(decision)

    last_prompt_view = seen_agent_views[-1]
    assert "working_memory" in last_prompt_view
    assert len(last_prompt_view["working_memory"]["recent_outcomes"]) == 6
    assert last_prompt_view["working_memory"]["open_commitments"] == [
        {
            "contract_id": "c-1",
            "job_id": 7,
            "status": "accepted",
            "completed_operations": 1,
            "total_operations": 4,
        }
    ]
    assert last_prompt_view["working_memory"]["rolled_summary"]["compressed_count"] == 1
    assert last_prompt_view["working_memory"]["rolled_summary"]["local_blocked_count"] == 1
    assert last_prompt_view["working_memory"]["recent_outcomes"][-1]["selected_target"] == (
        "commitment_management:reject"
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "fms_bench.v2.framework_drivers.ag2_driver",
        "fms_bench.v2.framework_drivers.agentscope_driver",
        "fms_bench.v2.framework_drivers.crewai_driver",
        "fms_bench.v2.framework_drivers.google_adk_driver",
        "fms_bench.v2.framework_drivers.langgraph_driver",
        "fms_bench.v2.framework_drivers.spade_driver",
    ],
)
def test_all_framework_llm_drivers_build_identical_prompt_messages(module_name: str) -> None:
    from fms_bench.v2.framework_drivers.llm_prompting import (
        build_openai_compat_benchmark_messages,
    )

    module = __import__(module_name, fromlist=["_build_llm_messages"])
    build_messages = getattr(module, "_build_llm_messages")

    assert build_messages(
        agent_id="plant",
        agent_view=_sample_agent_view(),
    ) == build_openai_compat_benchmark_messages(
        agent_id="plant",
        agent_view=_sample_agent_view(),
    )


def test_agentscope_and_langgraph_share_strict_json_action_contract() -> None:
    from fms_bench.v2.framework_drivers.agentscope_driver import _parse_llm_action_text
    from fms_bench.v2.framework_drivers.langgraph_driver import LangGraphLLMTreeDriver

    legal_actions = [0, 1]

    assert _parse_llm_action_text('{"action": 1}', legal_actions=legal_actions) == 1
    assert LangGraphLLMTreeDriver._parse_action_from_content('{"action": 1}') == 1

    assert _parse_llm_action_text("1", legal_actions=legal_actions) is None
    assert LangGraphLLMTreeDriver._parse_action_from_content("1") is None

    assert _parse_llm_action_text("best action is 1", legal_actions=legal_actions) is None
    assert LangGraphLLMTreeDriver._parse_action_from_content("best action is 1") is None
