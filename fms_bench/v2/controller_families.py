from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np


ActionOutput = int | Mapping[str, int]


@runtime_checkable
class ControllerPolicyLike(Protocol):
    @property
    def name(self) -> str: ...

    def select_action(
        self,
        observation: np.ndarray,
        info: Mapping[str, object],
    ) -> ActionOutput: ...


def _call_policy(
    policy: ControllerPolicyLike | Callable[[np.ndarray, Mapping[str, object]], ActionOutput],
    observation: np.ndarray,
    info: Mapping[str, object],
) -> ActionOutput:
    if hasattr(policy, "select_action"):
        return policy.select_action(observation, info)
    return policy(observation, info)


@dataclass(frozen=True)
class PolicyNetworkController:
    """Thin wrapper for a role-conditioned policy backend."""

    inference_fn: Callable[[np.ndarray, Mapping[str, object]], ActionOutput]
    controller_name: str = "policy_network"

    def select_action(
        self,
        observation: np.ndarray,
        info: Mapping[str, object],
    ) -> ActionOutput:
        return self.inference_fn(observation, info)

    @property
    def name(self) -> str:
        return self.controller_name


@dataclass(frozen=True)
class PromptControllerPolicy:
    """Prompt-driven controller wrapper over the same benchmark action contract."""

    prompt_builder: Callable[[np.ndarray, Mapping[str, object]], str]
    prompt_runner: Callable[[str, np.ndarray, Mapping[str, object]], ActionOutput]
    controller_name: str = "prompt_controller"

    def select_action(
        self,
        observation: np.ndarray,
        info: Mapping[str, object],
    ) -> ActionOutput:
        prompt = self.prompt_builder(observation, info)
        return self.prompt_runner(prompt, observation, info)

    @property
    def name(self) -> str:
        return self.controller_name


@dataclass(frozen=True)
class RolePolicyRouter:
    """Route one legal agent slice to one role-specific policy realization."""

    exact_policies: Mapping[str, ControllerPolicyLike | Callable[[np.ndarray, Mapping[str, object]], ActionOutput]] = field(
        default_factory=dict
    )
    prefix_policies: Mapping[str, ControllerPolicyLike | Callable[[np.ndarray, Mapping[str, object]], ActionOutput]] = field(
        default_factory=dict
    )
    default_policy: ControllerPolicyLike | Callable[[np.ndarray, Mapping[str, object]], ActionOutput] | None = None
    controller_name: str = "role_policy_router"

    def select_action(
        self,
        observation: np.ndarray,
        info: Mapping[str, object],
    ) -> ActionOutput:
        agent_id = str(info.get("agent_id", ""))
        policy = self._resolve_policy(agent_id)
        if policy is None:
            raise KeyError(f"No routed policy is configured for runtime agent: {agent_id}")
        return _call_policy(policy, observation, info)

    @property
    def name(self) -> str:
        return self.controller_name

    def _resolve_policy(
        self,
        agent_id: str,
    ) -> ControllerPolicyLike | Callable[[np.ndarray, Mapping[str, object]], ActionOutput] | None:
        exact = self.exact_policies.get(agent_id)
        if exact is not None:
            return exact

        for prefix in sorted(self.prefix_policies, key=len, reverse=True):
            if agent_id.startswith(prefix):
                return self.prefix_policies[prefix]
        return self.default_policy
