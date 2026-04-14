from __future__ import annotations

from dataclasses import asdict, dataclass

from .framework_drivers.deterministic_kernel import DeterministicCoordinationKernel


@dataclass(frozen=True)
class CategoryBControllerFamily:
    family_id: str
    controller_name: str
    authority_mode: str
    paradigm: str
    routing_authority: str
    local_autonomy: str
    reroute_policy: str
    escalation_policy: str
    dispatch_kernel_contract: str = "shared_deterministic_coordination_kernel_v1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def build_external_driver(self) -> "CategoryBExternalDriver":
        return CategoryBExternalDriver(
            authority_mode=self.authority_mode,
            controller_name=self.controller_name,
        )


class CategoryBExternalDriver:
    """Shared-kernel external driver whose family difference is authority distribution only."""

    uses_external_session = True

    def __init__(
        self,
        *,
        authority_mode: str,
        controller_name: str,
        reject_pressure_threshold: float = 4.0,
    ) -> None:
        self.authority_mode = str(authority_mode)
        self.controller_name = str(controller_name)
        self.reject_pressure_threshold = float(reject_pressure_threshold)
        self.kernel = DeterministicCoordinationKernel()

    @property
    def name(self) -> str:
        return self.controller_name

    def select_actions(self, decision: dict[str, object]) -> dict[str, int]:
        active_agents = dict(decision.get("active_agents", {}))
        return {
            str(agent_id): int(self._select_agent_action(dict(agent_view)))
            for agent_id, agent_view in active_agents.items()
        }

    def _select_agent_action(self, agent_view: dict[str, object]) -> int:
        role = str(agent_view.get("role", "unknown"))
        if role == "plant":
            return self._select_plant_action(agent_view)
        if role == "area":
            return self._select_area_action(agent_view)
        if role == "cell":
            return self._select_cell_action(agent_view)
        return self._first_valid_action(agent_view)

    def _select_plant_action(self, agent_view: dict[str, object]) -> int:
        plant_decision_kind = str(agent_view.get("plant_decision_kind", "area_selection"))
        if plant_decision_kind == "backlog_selection":
            return int(self.kernel.select_backlog(agent_view))
        if plant_decision_kind == "cell_selection":
            return int(
                self.kernel.select_cell(
                    {
                        "valid_cells": list(agent_view.get("valid_execution_cells", ())),
                        "visible_cell_summaries": dict(
                            agent_view.get("execution_cell_summaries", {})
                        ),
                        "action_mask": list(agent_view.get("action_mask", ())),
                    }
                )
            )
        return int(self.kernel.select_area(agent_view))

    def _select_area_action(self, agent_view: dict[str, object]) -> int:
        action_semantics = dict(agent_view.get("action_semantics", {}))
        if str(action_semantics.get("action_type", "")) == "accountability_and_escalation":
            return 0
        return int(self.kernel.select_cell(agent_view))

    def _select_cell_action(self, agent_view: dict[str, object]) -> int:
        activation = dict(agent_view.get("activation", {}))
        if str(activation.get("functional_loop", "")) == "local_regulation":
            return int(self.kernel.select_dispatch(agent_view))

        action_targets = list(dict(agent_view.get("action_semantics", {})).get("action_targets", ()))
        action_kinds = [str(target.get("action_kind", "")) for target in action_targets]
        if "reject_contract" in action_kinds and self._should_reject(agent_view):
            return int(action_kinds.index("reject_contract"))
        return 0

    def _should_reject(self, agent_view: dict[str, object]) -> bool:
        capacity_risk = dict(dict(agent_view.get("feedback_view", {})).get("capacity_risk", {}))
        if bool(capacity_risk.get("local_blocked", False)):
            return True
        cell_summary = dict(agent_view.get("cell_summary", {}))
        return (
            self.kernel.cell_disturbance_pressure(cell_summary)
            >= self.reject_pressure_threshold
        )

    @staticmethod
    def _first_valid_action(agent_view: dict[str, object]) -> int:
        mask = list(agent_view.get("action_mask", ()))
        for index, enabled in enumerate(mask):
            if int(enabled) > 0:
                return int(index)
        return 0


_CATEGORY_B_CONTROLLER_FAMILIES: dict[str, CategoryBControllerFamily] = {
    "centralized_controller": CategoryBControllerFamily(
        family_id="centralized_controller",
        controller_name="category_b_centralized_controller",
        authority_mode="centralized",
        paradigm="Centralized",
        routing_authority="plant/global controller direct cell award",
        local_autonomy="cell executes only; no contract reject right",
        reroute_policy="plant reselects next execution cell directly",
        escalation_policy="plant remains final reroute authority",
    ),
    "hierarchical_controller": CategoryBControllerFamily(
        family_id="hierarchical_controller",
        controller_name="category_b_hierarchical_controller",
        authority_mode="hierarchical",
        paradigm="Hierarchical",
        routing_authority="fixed plant to area to cell chain",
        local_autonomy="cell accept only; area executes delegated child selection",
        reroute_policy="reject or failure goes upward; no area-local reroute",
        escalation_policy="plant revises routing after upward escalation",
    ),
    "mediated_heterarchical_cnp_controller": CategoryBControllerFamily(
        family_id="mediated_heterarchical_cnp_controller",
        controller_name="category_b_mediated_heterarchical_cnp_controller",
        authority_mode="mediated_heterarchical_cnp",
        paradigm="Mediated Heterarchical CNP",
        routing_authority="allocation awarded from mediated cell competition",
        local_autonomy="cell can bid and reject within tree-mediated routing",
        reroute_policy="mediated bid and award without area-local sibling reroute",
        escalation_policy="plant revises after mediated path cannot commit",
    ),
    "holonic_hybrid_controller": CategoryBControllerFamily(
        family_id="holonic_hybrid_controller",
        controller_name="category_b_holonic_hybrid_controller",
        authority_mode="holonic_hybrid",
        paradigm="Holonic / Hybrid",
        routing_authority="plant-area-cell with bounded cell reject",
        local_autonomy="cell accept/reject plus area-local reroute",
        reroute_policy="area reroutes locally before upward escalation",
        escalation_policy="plant fallback only after local reroute exhaustion",
    ),
}


def get_category_b_controller_family(family_id: str) -> CategoryBControllerFamily:
    try:
        return _CATEGORY_B_CONTROLLER_FAMILIES[str(family_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown Category B controller family: {family_id}") from exc


def list_category_b_controller_families() -> list[dict[str, object]]:
    return [family.to_dict() for family in _CATEGORY_B_CONTROLLER_FAMILIES.values()]


def build_category_b_external_driver(family_id: str) -> CategoryBExternalDriver:
    return get_category_b_controller_family(family_id).build_external_driver()
