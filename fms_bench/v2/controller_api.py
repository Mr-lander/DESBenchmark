from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .controller_payloads import build_controller_payload
from .tree_runtime_env import TreeRuntimeEnv


class TreeControllerAdapter:
    """Role-safe adapter for controller interaction with the v2 tree runtime."""

    def __init__(self, env: TreeRuntimeEnv) -> None:
        self.env = env

    def observe(self, agent_id: str) -> dict[str, object]:
        self._require_current_agent(agent_id)
        return build_controller_payload(self.env, agent_id)

    def action_mask(self, agent_id: str) -> np.ndarray:
        self._require_current_agent(agent_id)
        return self.env.action_mask(agent_id)

    def step_dict(self, actions: Mapping[str, int]):
        return self.env.step_dict(actions)

    def step_controller(self, policy: Any) -> dict[str, int]:
        current_agents = list(self.env.agents_to_act())
        execution_mode = self.env.current_execution_mode(current_agents)
        supported_execution_modes = self.env.supported_execution_modes()
        if len(current_agents) > 1 and hasattr(policy, "select_actions"):
            decision = {
                "time": float(self.env.simulator.time),
                "agents_to_act": current_agents,
                "execution_mode": execution_mode,
                "supported_execution_modes": supported_execution_modes,
                "active_agents": {
                    agent_id: self.observe(agent_id)
                    for agent_id in current_agents
                },
            }
            actions = policy.select_actions(decision)
            return {
                str(agent_id): int(actions[agent_id])
                for agent_id in current_agents
            }

        actions: dict[str, int] = {}
        for agent_id in current_agents:
            agent_view = self.observe(agent_id)
            policy_info = {
                key: value
                for key, value in agent_view.items()
                if key != "observation"
            }
            policy_info["action_mask"] = np.asarray(agent_view["action_mask"]).tolist()
            policy_info["agents_to_act"] = list(self.env.agents_to_act())
            policy_info["execution_mode"] = execution_mode
            policy_info["supported_execution_modes"] = list(supported_execution_modes)
            policy_info["visible_protocol_objects"] = list(
                agent_view.get("visible_protocol_objects", ())
            )
            action = policy.select_action(
                np.asarray(agent_view["observation"]),
                policy_info,
            )
            if isinstance(action, Mapping):
                action = action.get(agent_id, 0)
            actions[agent_id] = int(action)
        return actions

    def _require_current_agent(self, agent_id: str) -> None:
        if agent_id not in self.env.agents_to_act():
            raise KeyError(
                f"{agent_id} is not active in the current decision epoch: {self.env.agents_to_act()}"
            )
