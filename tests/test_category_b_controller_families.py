"""Tests for Category B controller-family registry over one authority contract."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fms_bench.experiment_runner import (
    build_tree_runtime_v2_intercell_a3c9_probe_suite,
    make_regime_config,
)
from fms_bench.v2.external_controller_adapter import ExternalControllerSession
from fms_bench.v2.intercell_tree_runtime_env import IntercellTreeRuntimeEnv


_CANONICAL_ROUTE_GRAMMAR = (
    (0, 1, 3, 4, 6, 7),
    (1, 2, 4, 5, 7, 8),
    (0, 2, 3, 5, 6, 8),
)


def _make_intercell_env(*, authority_mode: str) -> IntercellTreeRuntimeEnv:
    suite = build_tree_runtime_v2_intercell_a3c9_probe_suite(PROJECT_ROOT)
    case = suite[0]
    return IntercellTreeRuntimeEnv(
        case.cell_configs,
        area_to_cells={0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]},
        config=make_regime_config("R0"),
        seed=13,
        max_in_flight_jobs=3,
        route_grammar=_CANONICAL_ROUTE_GRAMMAR,
        backlog_top_k=3,
        route_grammar_contract="category_b_a3c9_probe_v1",
        authority_mode=authority_mode,
    )


def test_category_b_controller_family_registry_exposes_four_authority_bound_baselines() -> None:
    from fms_bench.v2.category_b_controller_families import (
        get_category_b_controller_family,
        list_category_b_controller_families,
    )

    families = {item["family_id"]: item for item in list_category_b_controller_families()}

    assert set(families) >= {
        "centralized_controller",
        "hierarchical_controller",
        "heterarchical_cnp_controller",
        "holonic_hybrid_controller",
    }
    assert families["centralized_controller"]["authority_mode"] == "centralized"
    assert families["hierarchical_controller"]["authority_mode"] == "hierarchical"
    assert families["heterarchical_cnp_controller"]["authority_mode"] == "heterarchical_cnp"
    assert families["holonic_hybrid_controller"]["authority_mode"] == "holonic_hybrid"

    shared_contract = families["centralized_controller"]["dispatch_kernel_contract"]
    assert shared_contract == "shared_deterministic_coordination_kernel_v1"
    assert families["hierarchical_controller"]["dispatch_kernel_contract"] == shared_contract
    assert families["heterarchical_cnp_controller"]["dispatch_kernel_contract"] == shared_contract
    assert families["holonic_hybrid_controller"]["dispatch_kernel_contract"] == shared_contract

    holonic = get_category_b_controller_family("holonic_hybrid_controller")
    assert holonic.routing_authority == "plant-area-cell with bounded cell reject"
    assert holonic.local_autonomy == "cell accept/reject plus area-local reroute"


def test_category_b_family_builders_return_external_session_drivers_with_frozen_names() -> None:
    from fms_bench.v2.category_b_controller_families import (
        build_category_b_external_driver,
        get_category_b_controller_family,
    )

    expected = {
        "centralized_controller": "category_b_centralized_controller",
        "hierarchical_controller": "category_b_hierarchical_controller",
        "heterarchical_cnp_controller": "category_b_heterarchical_cnp_controller",
        "holonic_hybrid_controller": "category_b_holonic_hybrid_controller",
    }

    for family_id, driver_name in expected.items():
        family = get_category_b_controller_family(family_id)
        driver = build_category_b_external_driver(family_id)

        assert family.controller_name == driver_name
        assert family.controller_name == driver.name
        assert family.authority_mode in driver.name
        assert driver.uses_external_session is True


def test_category_b_external_drivers_can_step_first_two_a3c9_intercell_epochs() -> None:
    from fms_bench.v2.category_b_controller_families import (
        build_category_b_external_driver,
        get_category_b_controller_family,
    )

    for family_id in (
        "centralized_controller",
        "hierarchical_controller",
        "heterarchical_cnp_controller",
        "holonic_hybrid_controller",
    ):
        family = get_category_b_controller_family(family_id)
        driver = build_category_b_external_driver(family_id)
        env = _make_intercell_env(authority_mode=family.authority_mode)
        try:
            session = ExternalControllerSession(env)
            decision0 = session.reset(seed=13)
            step0 = session.step(driver.select_actions(decision0))
            decision1 = step0["current_decision"]
            step1 = session.step(driver.select_actions(decision1))
            decision2 = step1["current_decision"]

            assert decision2["agents_to_act"]
            if family.authority_mode == "centralized":
                assert list(decision2["agents_to_act"])[0].startswith("cell_")
            else:
                assert list(decision2["agents_to_act"])[0].startswith("area_")
        finally:
            env.close()
