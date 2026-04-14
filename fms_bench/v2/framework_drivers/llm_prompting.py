from __future__ import annotations

import json
import re
from typing import Mapping


OPENAI_COMPAT_BENCHMARK_SYSTEM_PROMPT = (
    "You are a benchmark controller agent. "
    "Select exactly one legal integer action index from action_mask. "
    "Use only the provided agent-local view. "
    'Return JSON only with one integer field: {"action": <index>}.'
)
OPENAI_COMPAT_BENCHMARK_MAX_TOKENS = 32


def build_openai_compat_benchmark_user_prompt(
    *,
    agent_id: str,
    agent_view: Mapping[str, object],
) -> str:
    payload = {
        "agent_id": str(agent_id),
        "agent_view": dict(agent_view),
    }
    return (
        "Choose one action index for this agent-local benchmark payload.\n"
        "Return JSON only.\n"
        f"{json.dumps(payload, ensure_ascii=True, separators=(',', ':'))}"
    )


def build_openai_compat_benchmark_messages(
    *,
    agent_id: str,
    agent_view: Mapping[str, object],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": OPENAI_COMPAT_BENCHMARK_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_openai_compat_benchmark_user_prompt(
                agent_id=agent_id,
                agent_view=agent_view,
            ),
        },
    ]


def build_openai_compat_benchmark_generation_kwargs(
    *,
    temperature: float = 0.0,
) -> dict[str, object]:
    return {
        "temperature": float(temperature),
        "max_tokens": OPENAI_COMPAT_BENCHMARK_MAX_TOKENS,
    }


def parse_openai_compat_benchmark_action_text(content: str) -> int | None:
    normalized = _strip_code_fence(content)
    if not normalized:
        return None
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    action = payload.get("action")
    if isinstance(action, int):
        return int(action)
    if isinstance(action, str) and action.strip().lstrip("-").isdigit():
        return int(action.strip())
    return None


def _strip_code_fence(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()
