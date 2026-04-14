from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .controller_api import TreeControllerAdapter
from .tree_runtime_env import TreeRuntimeEnv


SerializedValue = (
    None
    | bool
    | int
    | float
    | str
    | list["SerializedValue"]
    | dict[str, "SerializedValue"]
)
ExternalActionPayload = Mapping[str, int] | Sequence[Mapping[str, object]]
ExternalAgentDecisionFn = Callable[[Mapping[str, object], Mapping[str, object]], int]
EXTERNAL_ACTION_VALIDATION_CONTRACT = "active_agent_set_and_action_mask_enforced"
PUBLIC_FAIRNESS_SCOPE_CONTRACT = "serialized_public_epochs_protocol_fairness"
PUBLIC_DECISION_PAYLOAD_FIELDS = (
    "time",
    "terminated",
    "truncated",
    "agents_to_act",
    "execution_mode",
    "supported_execution_modes",
    "active_agents",
)
PUBLIC_ARTIFACT_PAYLOAD_FIELDS = (
    "time",
    "info",
)


@runtime_checkable
class ExternalDecisionDriverLike(Protocol):
    @property
    def name(self) -> str: ...

    def select_actions(self, decision: Mapping[str, object]) -> ExternalActionPayload: ...


@dataclass(frozen=True)
class ExternalControllerDriver:
    """Named external-session driver over the benchmark decision payload."""

    decision_fn: Callable[[Mapping[str, object]], ExternalActionPayload]
    controller_name: str = "external_controller"
    uses_external_session: bool = True

    def select_actions(self, decision: Mapping[str, object]) -> ExternalActionPayload:
        return self.decision_fn(decision)

    @property
    def name(self) -> str:
        return self.controller_name


@dataclass(frozen=True)
class InProcessPolicyExternalDriver:
    """Run an in-process benchmark policy through the external session boundary."""

    policy: Any
    controller_name: str | None = None
    uses_external_session: bool = True

    def select_actions(self, decision: Mapping[str, object]) -> dict[str, int]:
        actions: dict[str, int] = {}
        for agent_id, agent_view in dict(decision["active_agents"]).items():
            info = dict(agent_view)
            observation = np.asarray(info.pop("observation", []), dtype=np.float32)
            action = _call_policy_like(self.policy, observation, info)
            if isinstance(action, Mapping):
                action = action.get(agent_id, 0)
            actions[str(agent_id)] = int(action)
        return actions

    @property
    def name(self) -> str:
        if self.controller_name is not None:
            return self.controller_name
        policy_name = getattr(self.policy, "name", getattr(self.policy, "__name__", "policy"))
        return f"external_{policy_name}"


@dataclass(frozen=True)
class RoleRoutedExternalDriver:
    """Route serialized agent views to role- or agent-specific external handlers."""

    exact_drivers: Mapping[str, ExternalAgentDecisionFn] | None = None
    role_drivers: Mapping[str, ExternalAgentDecisionFn] | None = None
    default_driver: ExternalAgentDecisionFn | None = None
    controller_name: str = "external_role_router"
    uses_external_session: bool = True

    def select_actions(self, decision: Mapping[str, object]) -> dict[str, int]:
        actions: dict[str, int] = {}
        for agent_id, agent_view in dict(decision["active_agents"]).items():
            driver = self._resolve_driver(str(agent_id), dict(agent_view))
            if driver is None:
                raise KeyError(f"No external driver configured for runtime agent: {agent_id}")
            actions[str(agent_id)] = int(driver(decision, dict(agent_view)))
        return actions

    @property
    def name(self) -> str:
        return self.controller_name

    def _resolve_driver(
        self,
        agent_id: str,
        agent_view: Mapping[str, object],
    ) -> ExternalAgentDecisionFn | None:
        exact_drivers = self.exact_drivers or {}
        exact = exact_drivers.get(agent_id)
        if exact is not None:
            return exact
        role = str(agent_view.get("role", _agent_role(agent_id)))
        role_drivers = self.role_drivers or {}
        role_driver = role_drivers.get(role)
        if role_driver is not None:
            return role_driver
        return self.default_driver


class SubprocessExternalControllerDriver:
    """Run an external controller in a separate process over one-line JSON messages."""

    uses_external_session = True

    def __init__(
        self,
        command: str | Sequence[str],
        *,
        controller_name: str = "external_subprocess_controller",
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self._command = command
        self._cwd = None if cwd is None else str(cwd)
        self._env = None if env is None else {str(key): str(value) for key, value in env.items()}
        self._encoding = str(encoding)
        self._controller_name = str(controller_name)
        self._process: subprocess.Popen[str] | None = None

    @property
    def name(self) -> str:
        return self._controller_name

    def select_actions(self, decision: Mapping[str, object]) -> ExternalActionPayload:
        process = self._ensure_process()
        stdin = process.stdin
        stdout = process.stdout
        if stdin is None or stdout is None:
            raise RuntimeError("Subprocess controller must be started with piped stdin/stdout")

        payload = json.dumps(_serialize_value(decision), ensure_ascii=True)
        try:
            stdin.write(payload + "\n")
            stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(self._format_process_error("stdin pipe closed")) from exc

        response = stdout.readline()
        if response == "":
            raise RuntimeError(self._format_process_error("stdout closed before returning actions"))

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{self.name} returned invalid JSON action payload: {response.strip()!r}"
            ) from exc

        if not isinstance(parsed, (Mapping, list, tuple)):
            raise TypeError(
                f"{self.name} must return a mapping or sequence of action records, got {type(parsed)!r}"
            )
        return parsed

    def reset_episode(self) -> None:
        self._stop_process()

    def close(self) -> None:
        self._stop_process()

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            self._process = subprocess.Popen(
                self._popen_command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                cwd=self._cwd,
                env=self._env,
                text=True,
                encoding=self._encoding,
                bufsize=1,
            )
            return self._process

        return_code = self._process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"{self.name} exited unexpectedly before action selection with return code {return_code}"
            )
        return self._process

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return

        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

        try:
            if process.stdout is not None:
                process.stdout.close()
        except OSError:
            pass

    def _popen_command(self) -> str | list[str]:
        if isinstance(self._command, str):
            return self._command
        return [str(part) for part in self._command]

    def _format_process_error(self, detail: str) -> str:
        process = self._process
        if process is None:
            return f"{self.name} {detail}"
        return_code = process.poll()
        if return_code is None:
            return f"{self.name} {detail}"
        return f"{self.name} {detail}; subprocess exited with return code {return_code}"


def run_external_controller_loop(
    session: "ExternalControllerSession",
    driver: ExternalDecisionDriverLike | Callable[[Mapping[str, object]], ExternalActionPayload],
    *,
    max_steps: int = 10_000,
) -> list[dict[str, SerializedValue]]:
    """Drive an external controller session until done or max_steps is reached."""

    _reset_external_driver_episode(driver)
    transitions: list[dict[str, SerializedValue]] = []
    for _ in range(int(max_steps)):
        decision = session.current_decision_payload()
        if not decision["agents_to_act"] or decision["terminated"] or decision["truncated"]:
            break
        actions = _call_external_driver(driver, decision)
        transition = session.step(actions)
        transitions.append(transition)
        if transition["terminated"] or transition["truncated"]:
            break
    return transitions


class ExternalControllerSession:
    """JSON-safe session wrapper for prompt/network/framework-managed controllers."""

    def __init__(self, env: TreeRuntimeEnv) -> None:
        self.env = env
        self.adapter = TreeControllerAdapter(env)
        self._last_info: Mapping[str, object] | None = None
        self._terminated = False
        self._truncated = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, SerializedValue]:
        _, info = self.env.reset(seed=seed, options=options)
        self._last_info = info
        self._terminated = False
        self._truncated = False
        return self.current_decision_payload()

    def current_decision_payload(self) -> dict[str, SerializedValue]:
        self._advance_idle_epochs()
        agents_to_act = list(self.env.agents_to_act())
        return {
            "time": float(self.env.simulator.time),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
            "agents_to_act": [_serialize_value(agent_id) for agent_id in agents_to_act],
            "execution_mode": _serialize_value(self.env.current_execution_mode(agents_to_act)),
            "supported_execution_modes": _serialize_value(self.env.supported_execution_modes()),
            "active_agents": {
                agent_id: self._agent_payload(agent_id)
                for agent_id in agents_to_act
            },
        }

    def current_artifact_payload(self) -> dict[str, SerializedValue]:
        self._advance_idle_epochs()
        info = self._last_info if self._last_info is not None else self.env.render()
        return {
            "time": float(self.env.simulator.time),
            "info": _serialize_value(info),
        }

    def step(
        self,
        actions: ExternalActionPayload,
    ) -> dict[str, SerializedValue]:
        normalized_actions = self._normalize_action_payload(actions)
        self._validate_action_dict(normalized_actions)
        result = self.env.step_dict(normalized_actions)
        self._last_info = result.info
        self._terminated = bool(result.terminated)
        self._truncated = bool(result.truncated)
        return {
            "submitted_actions": {
                agent_id: _serialize_value(action)
                for agent_id, action in normalized_actions.items()
            },
            "reward": float(result.reward),
            "terminated": bool(result.terminated),
            "truncated": bool(result.truncated),
            "reward_components": _serialize_value(result.info.get("reward_components", {})),
            "current_decision": self.current_decision_payload(),
        }

    def _advance_idle_epochs(self) -> None:
        while not self._terminated and not self._truncated and not self.env.agents_to_act():
            if not hasattr(self.env, "advance_idle_epoch"):
                break
            result = self.env.advance_idle_epoch()
            self._last_info = result.info
            self._terminated = bool(result.terminated)
            self._truncated = bool(result.truncated)

    def _agent_payload(self, agent_id: str) -> dict[str, SerializedValue]:
        agent_view = self.adapter.observe(agent_id)
        return {
            "role": _serialize_value(_agent_role(agent_id)),
            **{
                str(key): _serialize_value(value)
                for key, value in agent_view.items()
            },
        }

    def _validate_action_dict(self, actions: Mapping[str, int]) -> None:
        current_agents = self.env.agents_to_act()
        if set(actions) != set(current_agents):
            raise KeyError(
                f"External actions must match current active agents: {current_agents}"
            )

        for agent_id, action in actions.items():
            mask = np.asarray(self.adapter.action_mask(agent_id), dtype=np.int32)
            action_value = int(action)
            if action_value < 0 or action_value >= len(mask):
                raise ValueError(
                    f"Action {action_value} is out of bounds for {agent_id} with mask length {len(mask)}"
                )
            if mask[action_value] <= 0:
                raise ValueError(
                    f"Action {action_value} is masked out for {agent_id}: {mask.tolist()}"
                )

    @staticmethod
    def _normalize_action_payload(
        actions: ExternalActionPayload,
    ) -> dict[str, int]:
        if isinstance(actions, Mapping):
            return {
                str(agent_id): ExternalControllerSession._coerce_action_value(value)
                for agent_id, value in actions.items()
            }

        normalized: dict[str, int] = {}
        for item in actions:
            agent_id = str(item["agent_id"])
            if agent_id in normalized:
                raise KeyError(f"Duplicate external action for active agent: {agent_id}")
            normalized[agent_id] = ExternalControllerSession._coerce_action_value(
                item["action"]
            )
        return normalized

    @staticmethod
    def _coerce_action_value(value: object) -> int:
        if isinstance(value, Mapping):
            if "action" in value:
                return int(value["action"])
            if "action_id" in value:
                return int(value["action_id"])
        return int(value)


def _call_external_driver(
    driver: ExternalDecisionDriverLike | Callable[[Mapping[str, object]], ExternalActionPayload],
    decision: Mapping[str, object],
) -> ExternalActionPayload:
    if hasattr(driver, "select_actions"):
        return driver.select_actions(decision)
    return driver(decision)


def _reset_external_driver_episode(driver: object) -> None:
    reset_fn = getattr(driver, "reset_episode", None)
    if callable(reset_fn):
        reset_fn()


def _call_policy_like(
    policy: Any,
    observation: np.ndarray,
    info: Mapping[str, object],
) -> object:
    if hasattr(policy, "select_action"):
        return policy.select_action(observation, info)
    if hasattr(policy, "act"):
        return policy.act(observation, info)
    if callable(policy):
        return policy(observation, info)
    raise TypeError(f"Unsupported in-process policy wrapped as external driver: {type(policy)!r}")


def _agent_role(agent_id: str) -> str:
    if agent_id == "plant":
        return "plant"
    if agent_id.startswith("area_"):
        return "area"
    if agent_id.startswith("cell_"):
        return "cell"
    return "unknown"


def _serialize_value(value: object) -> SerializedValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, np.generic):
        return _serialize_value(value.item())

    if isinstance(value, np.ndarray):
        return [_serialize_value(item) for item in value.tolist()]

    if hasattr(value, "to_dict"):
        return _serialize_value(value.to_dict())

    if is_dataclass(value):
        return _serialize_value(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): _serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item) for item in value]

    return str(value)
