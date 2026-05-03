"""Minimal paradigm registration example for DESBench."""

from __future__ import annotations

from fms_bench.paradigms import AuthorityMode, register_authority_mode


register_authority_mode(
    AuthorityMode(
        mode_id="my_hybrid",
        routing_authority_mode="holonic_hybrid",
        cell_contract_actions="accept_reject",
        area_local_reroute_enabled=True,
        plant_reroute_required=False,
        mediated_bid_round_enabled=True,
    )
)
