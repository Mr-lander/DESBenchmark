from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Mapping
from urllib import error, request


def _parse_args(argv: list[str] | None = None) -> object:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible JSONL subprocess controller for the benchmark external-session path.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--on-invalid-response",
        choices=["strict", "first_valid"],
        default="strict",
    )
    return parser.parse_args(argv)


def _chat_completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _first_valid_actions(decision: Mapping[str, object]) -> dict[str, int]:
    actions: dict[str, int] = {}
    for agent_id, agent_view in dict(decision.get("active_agents", {})).items():
        mask = list(dict(agent_view).get("action_mask", []))
        selected = next((index for index, enabled in enumerate(mask) if enabled), 0)
        actions[str(agent_id)] = int(selected)
    return actions


def _legal_action_indices(agent_view: Mapping[str, object]) -> list[int]:
    mask = list(agent_view.get("action_mask", []))
    return [index for index, enabled in enumerate(mask) if bool(enabled)]


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _normalize_actions(
    payload: object,
    *,
    decision: Mapping[str, object],
) -> dict[str, int] | None:
    active_agents = {
        str(agent_id): dict(agent_view)
        for agent_id, agent_view in dict(decision.get("active_agents", {})).items()
    }
    current_agents = set(active_agents)
    if len(active_agents) == 1 and isinstance(payload, Mapping) and "action" in payload:
        agent_id = next(iter(active_agents))
        action = payload.get("action")
        try:
            parsed_action = int(action)
        except (TypeError, ValueError):
            return None
        return (
            {agent_id: parsed_action}
            if parsed_action in _legal_action_indices(active_agents[agent_id])
            else None
        )
    if isinstance(payload, Mapping):
        try:
            actions = {str(agent_id): int(action) for agent_id, action in payload.items()}
        except (TypeError, ValueError):
            return None
        if set(actions) != current_agents:
            return None
        for agent_id, action in actions.items():
            if action not in _legal_action_indices(active_agents[agent_id]):
                return None
        return actions
    return None


def _extract_content(response_payload: Mapping[str, object]) -> str | None:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None
    message = first.get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                text_parts.append(str(item["text"]))
        if text_parts:
            return "".join(text_parts)
    return None


def _request_actions(
    *,
    decision: Mapping[str, object],
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
) -> dict[str, int] | None:
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a benchmark controller. "
                    "Use only the provided decision payload. "
                    "Return compact JSON mapping each active agent id to one legal integer action."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(decision, ensure_ascii=True, sort_keys=True),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    req = request.Request(
        _chat_completions_url(base_url),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            response_payload = json.loads(resp.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    content = _extract_content(response_payload)
    if not isinstance(content, str) or not content.strip():
        return None
    normalized = _strip_code_fence(content)
    for candidate in (normalized, content.strip()):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        actions = _normalize_actions(parsed, decision=decision)
        if actions is not None:
            return actions
    return None


def main() -> None:
    args = _parse_args()
    api_key = str(args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY")
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        decision = json.loads(line)
        actions = _request_actions(
            decision=decision,
            model=str(args.model),
            base_url=str(args.base_url),
            api_key=api_key,
            timeout=float(args.timeout),
        )
        if actions is None:
            if str(args.on_invalid_response) != "first_valid":
                raise SystemExit("openai-compatible controller returned no valid action mapping")
            actions = _first_valid_actions(decision)
        sys.stdout.write(json.dumps(actions) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
