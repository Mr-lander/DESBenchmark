"""Tests for explicit Category B authority-mode definitions."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_authority_mode_registry_exposes_four_required_modes() -> None:
    from fms_bench.v2.authority_modes import get_authority_mode, list_authority_modes

    assert {mode["mode_id"] for mode in list_authority_modes()} >= {
        "centralized",
        "hierarchical",
        "mediated_heterarchical_cnp",
        "holonic_hybrid",
    }

    assert get_authority_mode("centralized").routing_authority_mode == "centralized"
    assert get_authority_mode("hierarchical").routing_authority_mode == "hierarchical"
    assert (
        get_authority_mode("mediated_heterarchical_cnp").routing_authority_mode
        == "mediated_cnp"
    )
    assert get_authority_mode("holonic_hybrid").routing_authority_mode == "holonic_hybrid"


def test_hierarchical_and_holonic_modes_differ_on_reject_and_reroute_rights() -> None:
    from fms_bench.v2.authority_modes import get_authority_mode

    hierarchical = get_authority_mode("hierarchical")
    holonic = get_authority_mode("holonic_hybrid")

    assert hierarchical.cell_contract_actions == "accept_only"
    assert hierarchical.area_local_reroute_enabled is False
    assert hierarchical.plant_reroute_required is True
    assert hierarchical.mediated_bid_round_enabled is False

    assert holonic.cell_contract_actions == "accept_reject"
    assert holonic.area_local_reroute_enabled is True
    assert holonic.plant_reroute_required is False
    assert holonic.mediated_bid_round_enabled is False


def test_mediated_cnp_mode_enables_bid_round_without_claiming_true_peer_to_peer() -> None:
    from fms_bench.v2.authority_modes import get_authority_mode

    mode = get_authority_mode("mediated_heterarchical_cnp")

    assert mode.routing_authority_mode == "mediated_cnp"
    assert mode.cell_contract_actions == "accept_reject"
    assert mode.area_local_reroute_enabled is False
    assert mode.plant_reroute_required is False
    assert mode.mediated_bid_round_enabled is True


def test_authority_mode_lookup_rejects_unknown_mode() -> None:
    from fms_bench.v2.authority_modes import get_authority_mode

    try:
        get_authority_mode("unknown")
    except ValueError as exc:
        assert "Unknown authority mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown authority mode")
