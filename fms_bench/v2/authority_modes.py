from __future__ import annotations

from dataclasses import asdict, dataclass


_HETERARCHICAL_CNP_ALIAS = "mediated_heterarchical_cnp"


@dataclass(frozen=True)
class AuthorityMode:
    mode_id: str
    routing_authority_mode: str
    cell_contract_actions: str
    area_local_reroute_enabled: bool
    plant_reroute_required: bool
    mediated_bid_round_enabled: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_AUTHORITY_MODE_REGISTRY: dict[str, AuthorityMode] = {
    "centralized": AuthorityMode(
        mode_id="centralized",
        routing_authority_mode="centralized",
        cell_contract_actions="accept_only",
        area_local_reroute_enabled=False,
        plant_reroute_required=False,
        mediated_bid_round_enabled=False,
    ),
    "hierarchical": AuthorityMode(
        mode_id="hierarchical",
        routing_authority_mode="hierarchical",
        cell_contract_actions="accept_only",
        area_local_reroute_enabled=False,
        plant_reroute_required=True,
        mediated_bid_round_enabled=False,
    ),
    "heterarchical_cnp": AuthorityMode(
        mode_id="heterarchical_cnp",
        routing_authority_mode="mediated_cnp",
        cell_contract_actions="accept_reject",
        area_local_reroute_enabled=False,
        plant_reroute_required=False,
        mediated_bid_round_enabled=True,
    ),
    "holonic_hybrid": AuthorityMode(
        mode_id="holonic_hybrid",
        routing_authority_mode="holonic_hybrid",
        cell_contract_actions="accept_reject",
        area_local_reroute_enabled=True,
        plant_reroute_required=False,
        mediated_bid_round_enabled=False,
    ),
}


def get_authority_mode(mode_id: str) -> AuthorityMode:
    normalized_mode_id = str(mode_id)
    if normalized_mode_id == _HETERARCHICAL_CNP_ALIAS:
        normalized_mode_id = "heterarchical_cnp"
    try:
        return _AUTHORITY_MODE_REGISTRY[normalized_mode_id]
    except KeyError as exc:
        raise ValueError(f"Unknown authority mode: {mode_id}") from exc


def list_authority_modes() -> list[dict[str, object]]:
    return [mode.to_dict() for mode in _AUTHORITY_MODE_REGISTRY.values()]
