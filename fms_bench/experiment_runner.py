"""Experiment runners for public holonic benchmark tables."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence
import csv
import json

from .authority_profiles import apply_numeric_profile
from .arrival_profiles import apply_arrival_profile
from .config import BenchmarkConfig
from .due_date_profiles import apply_due_date_profile
from .failure_profiles import apply_failure_profile
from .holonic_evaluation import evaluate_holonic_policy
from .holonic_types import PlantCellConfig
from .plant_env import HolonicPlantEnv
from .shared_backlog_env import SharedBacklogPlantEnv
from .setup_profiles import apply_setup_profile
from .tree_builder import build_tree_topology
from .tree_shared_backlog_env import TreeSharedBacklogEnv


IMPLEMENTED_LEVELS = ('L0', 'L2', 'L4')
IMPLEMENTED_REGIMES = ('R0', 'R2', 'R4')
DEFAULT_CASE_BUDGET_PROVENANCE_CLASS = 'benchmark_case_design'
DEFAULT_CASE_BUDGET_SCOPE = 'wip,energy,carbon'
DEFAULT_CASE_BUDGET_SOURCE_REF = 'experiment_runner_tree_runtime_v2_case_cell_budgets_r20260327'
CASE_BUDGET_ARTIFACT_FILENAME = 'case_budget_snapshot.json'
CASE_BUDGET_METADATA_FIELDS = (
    'case_budget_provenance_class',
    'case_budget_scope',
    'case_budget_source_ref',
)
SEMANTIC_CASE_METADATA_FIELDS = (
    'profile_id',
    'authority_mode_id',
    'semantic_profile_id',
    'semantic_arrival_mode',
    'semantic_disturbance_family',
    'semantic_setup_enabled',
    'semantic_buffer_wip_enabled',
    'semantic_transport_enabled',
    'official_surface_id',
    'official_surface_role',
)


def load_benchmark_seed_split(repo_root: Path, split_name: str) -> list[int]:
    split_field_by_name = {
        'train': 'train_seeds',
        'dev': 'validation_seeds',
        'validation': 'validation_seeds',
        'test': 'test_seeds',
    }
    if split_name not in split_field_by_name:
        raise ValueError(f'Unsupported benchmark seed split: {split_name}')

    protocol_path = Path(repo_root) / 'docs' / 'benchmark_seed_protocol_v1.json'
    payload = json.loads(protocol_path.read_text(encoding='utf-8-sig'))
    return [int(seed) for seed in payload[split_field_by_name[split_name]]]


@dataclass(frozen=True)
class HolonicExperimentCase:
    case_id: str
    suite_tier: str
    level: str
    benchmark_item_id: str
    cell_configs: List[PlantCellConfig]
    seeds: List[int]
    env_name: str = 'holonic_plant'
    env_cls: Any = HolonicPlantEnv
    regime_id: str | None = None
    topology_id: str | None = None
    numeric_profile_id: str | None = None
    due_date_profile_id: str | None = None
    setup_profile_id: str | None = None
    arrival_profile_id: str | None = None
    failure_profile_id: str | None = None
    horizon_hours_override: float | None = None
    budget_tier: str | None = None
    budget_relative_to_nominal: float | None = None
    case_budget_provenance_class: str = DEFAULT_CASE_BUDGET_PROVENANCE_CLASS
    case_budget_scope: str = DEFAULT_CASE_BUDGET_SCOPE
    case_budget_source_ref: str = DEFAULT_CASE_BUDGET_SOURCE_REF
    profile_id: str | None = None
    authority_mode_id: str | None = None
    semantic_profile_id: str | None = None
    semantic_arrival_mode: str | None = None
    semantic_disturbance_family: str | None = None
    semantic_setup_enabled: bool = False
    semantic_buffer_wip_enabled: bool = False
    semantic_transport_enabled: bool = False
    official_surface_id: str | None = None
    official_surface_role: str | None = None


@dataclass(frozen=True)
class IntercellCongestionProfileSpec:
    profile_id: str
    max_in_flight_jobs: int
    inbound_buffer_cap_overrides: dict[int, int]
    processing_scale_overrides: dict[int, float]
    transport_time_multiplier: float = 1.0


A3C9_DYNAMICS_NOMINAL_HORIZON_HOURS = 500.0
A3C9_DYNAMICS_BUDGET_RATIOS: dict[str, float] = {
    'short': 0.75,
    'long': 1.50,
}
DEFAULT_INTERCELL_AUTHORITY_MODE = 'holonic_hybrid'
MAINLINE_CATEGORY_B_AUTHORITY_MODES = (
    'centralized',
    'hierarchical',
    'mediated_heterarchical_cnp',
    'holonic_hybrid',
)


@dataclass(frozen=True)
class ExperimentCaseRunResult:
    case_id: str
    summary_csv: Path
    artifact_manifest: Path
    rows: List[dict[str, Any]]


def make_level_config(
    level: str,
    *,
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> BenchmarkConfig:
    if level not in IMPLEMENTED_LEVELS:
        raise ValueError(f'Unsupported benchmark level for current runtime: {level}')

    config = BenchmarkConfig()
    config.horizon_hours = 500
    config.failure.mttr_hours = 0.0

    if level == 'L0':
        apply_failure_profile(config, "failure_none_v1")
    elif level == 'L2':
        apply_failure_profile(config, "failure_exponential_nominal_v1")
    elif level == 'L4':
        apply_failure_profile(config, "failure_none_v1")
    apply_due_date_profile(config, due_date_profile_id)
    return apply_numeric_profile(config, numeric_profile_id)


def make_regime_config(
    regime_id: str,
    *,
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> BenchmarkConfig:
    if regime_id not in IMPLEMENTED_REGIMES:
        raise ValueError(f'Unsupported benchmark regime for current runtime: {regime_id}')

    config = BenchmarkConfig()
    config.horizon_hours = 500
    config.failure.mttr_hours = 0.0

    if regime_id == 'R0':
        apply_failure_profile(config, "failure_none_v1")
    elif regime_id == 'R2':
        apply_failure_profile(config, "failure_exponential_nominal_v1")
    elif regime_id == 'R4':
        apply_failure_profile(config, "failure_none_v1")
    apply_due_date_profile(config, due_date_profile_id)
    return apply_numeric_profile(config, numeric_profile_id)


def _flex_root(repo_root: Path) -> Path:
    return repo_root / 'external' / 'benchmarks' / 'flexible jobshop' / 'instances'


def _build_case_cells(repo_root: Path, composition: str, level: str) -> List[PlantCellConfig]:
    root = _flex_root(repo_root)
    if composition == 'kacem+fattahi':
        if level == 'L4':
            return [
                PlantCellConfig(
                    cell_id=0,
                    family_key='kacem',
                    instance_path=root / 'KacemHammadiBorne2002' / 'kacem1.txt',
                    instance_kind='kacem',
                    label='cell-kacem',
                    wip_cap=1,
                    energy_cap_kwh=1500.0,
                    carbon_cap_kg=900.0,
                ),
                PlantCellConfig(
                    cell_id=1,
                    family_key='fattahi',
                    instance_path=root / 'FattahiMehrabadJolai2007' / 'fattahi1.txt',
                    instance_kind='fattahi',
                    label='cell-fattahi',
                    wip_cap=1,
                    energy_cap_kwh=1200.0,
                    carbon_cap_kg=750.0,
                ),
            ]
        return [
            PlantCellConfig(
                cell_id=0,
                family_key='kacem',
                instance_path=root / 'KacemHammadiBorne2002' / 'kacem1.txt',
                instance_kind='kacem',
                label='cell-kacem',
                wip_cap=1,
            ),
            PlantCellConfig(
                cell_id=1,
                family_key='fattahi',
                instance_path=root / 'FattahiMehrabadJolai2007' / 'fattahi1.txt',
                instance_kind='fattahi',
                label='cell-fattahi',
                wip_cap=1,
            ),
        ]

    if composition == 'brandimarte+hurink':
        if level == 'L4':
            return [
                PlantCellConfig(
                    cell_id=0,
                    family_key='brandimarte',
                    instance_path=root / 'Brandimarte1993' / 'mk01.txt',
                    instance_kind='brandimarte',
                    label='cell-mk01',
                    wip_cap=1,
                    energy_cap_kwh=2200.0,
                    carbon_cap_kg=1320.0,
                ),
                PlantCellConfig(
                    cell_id=1,
                    family_key='hurink',
                    instance_path=root / 'HurinkJurischThole1994' / 'edata' / 'la01.txt',
                    instance_kind='hurink',
                    label='cell-hurink-la01',
                    wip_cap=1,
                    energy_cap_kwh=1800.0,
                    carbon_cap_kg=1080.0,
                ),
            ]
        return [
            PlantCellConfig(
                cell_id=0,
                family_key='brandimarte',
                instance_path=root / 'Brandimarte1993' / 'mk01.txt',
                instance_kind='brandimarte',
                label='cell-mk01',
                wip_cap=1,
            ),
            PlantCellConfig(
                cell_id=1,
                family_key='hurink',
                instance_path=root / 'HurinkJurischThole1994' / 'edata' / 'la01.txt',
                instance_kind='hurink',
                label='cell-hurink-la01',
                wip_cap=1,
            ),
        ]

    raise ValueError(f'Unknown composition: {composition}')


def build_core_v1_small_suite(repo_root: Path) -> List[HolonicExperimentCase]:
    repo_root = Path(repo_root)
    suite_tier = 'core_v1_small'
    seeds = [13, 17]
    cases: List[HolonicExperimentCase] = []
    for composition in ('kacem+fattahi', 'brandimarte+hurink'):
        for level in IMPLEMENTED_LEVELS:
            benchmark_item_id = f'suite={suite_tier}__level={level}__cells={composition}__split=test'
            cases.append(
                HolonicExperimentCase(
                    case_id=f'{suite_tier}__{composition}__{level}',
                    suite_tier=suite_tier,
                    level=level,
                    benchmark_item_id=benchmark_item_id,
                    cell_configs=_build_case_cells(repo_root, composition, level),
                    seeds=list(seeds),
                )
            )
    return cases


def _build_shared_backlog_cells(repo_root: Path, composition: str, level: str) -> List[PlantCellConfig]:
    root = _flex_root(repo_root)

    if composition == 'kacem_fastslow':
        common = {
            'instance_path': root / 'KacemHammadiBorne2002' / 'kacem1.txt',
            'instance_kind': 'kacem',
            'family_key': 'kacem',
            'wip_cap': 2,
        }
        if level == 'L4':
            return [
                PlantCellConfig(cell_id=0, label='cell-kacem-fast', processing_time_scale=0.8, energy_cap_kwh=650.0, carbon_cap_kg=390.0, **common),
                PlantCellConfig(cell_id=1, label='cell-kacem-slow', processing_time_scale=1.4, energy_cap_kwh=900.0, carbon_cap_kg=540.0, **common),
            ]
        return [
            PlantCellConfig(cell_id=0, label='cell-kacem-fast', processing_time_scale=0.8, **common),
            PlantCellConfig(cell_id=1, label='cell-kacem-slow', processing_time_scale=1.4, **common),
        ]

    if composition == 'kacem_budgettrap':
        common = {
            'instance_path': root / 'KacemHammadiBorne2002' / 'kacem1.txt',
            'instance_kind': 'kacem',
            'family_key': 'kacem',
            'wip_cap': 3,
        }
        if level == 'L4':
            return [
                PlantCellConfig(cell_id=0, label='cell-kacem-fast-tight', processing_time_scale=0.8, energy_cap_kwh=250.0, carbon_cap_kg=150.0, **common),
                PlantCellConfig(cell_id=1, label='cell-kacem-slow-loose', processing_time_scale=1.2, energy_cap_kwh=900.0, carbon_cap_kg=540.0, **common),
            ]
        return [
            PlantCellConfig(cell_id=0, label='cell-kacem-fast-tight', processing_time_scale=0.8, **common),
            PlantCellConfig(cell_id=1, label='cell-kacem-slow-loose', processing_time_scale=1.2, **common),
        ]

    if composition == 'fattahi_fastslow':
        common = {
            'instance_path': root / 'FattahiMehrabadJolai2007' / 'fattahi1.txt',
            'instance_kind': 'fattahi',
            'family_key': 'fattahi',
            'wip_cap': 2,
        }
        if level == 'L4':
            return [
                PlantCellConfig(cell_id=0, label='cell-fattahi-fast', processing_time_scale=0.85, energy_cap_kwh=850.0, carbon_cap_kg=510.0, **common),
                PlantCellConfig(cell_id=1, label='cell-fattahi-slow', processing_time_scale=1.3, energy_cap_kwh=1050.0, carbon_cap_kg=630.0, **common),
            ]
        return [
            PlantCellConfig(cell_id=0, label='cell-fattahi-fast', processing_time_scale=0.85, **common),
            PlantCellConfig(cell_id=1, label='cell-fattahi-slow', processing_time_scale=1.3, **common),
        ]

    raise ValueError(f'Unknown shared-backlog composition: {composition}')


def build_shared_backlog_v1_small_suite(repo_root: Path) -> List[HolonicExperimentCase]:
    repo_root = Path(repo_root)
    suite_tier = 'shared_backlog_v1_small'
    seeds = [13, 17]
    cases: List[HolonicExperimentCase] = []
    for composition in ('kacem_fastslow', 'kacem_budgettrap', 'fattahi_fastslow'):
        for level in IMPLEMENTED_LEVELS:
            benchmark_item_id = f'suite={suite_tier}__level={level}__cells={composition}__split=test'
            cases.append(
                HolonicExperimentCase(
                    case_id=f'{suite_tier}__{composition}__{level}',
                    suite_tier=suite_tier,
                    level=level,
                    benchmark_item_id=benchmark_item_id,
                    cell_configs=_build_shared_backlog_cells(repo_root, composition, level),
                    seeds=list(seeds),
                    env_name='shared_backlog',
                    env_cls=SharedBacklogPlantEnv,
                )
            )
    return cases


def _build_tree_shared_backlog_cells(repo_root: Path, composition: str, regime_id: str) -> List[PlantCellConfig]:
    root = _flex_root(repo_root)

    if composition == 'kacem_three_cell':
        common = {
            'instance_path': root / 'KacemHammadiBorne2002' / 'kacem1.txt',
            'instance_kind': 'kacem',
            'family_key': 'kacem',
            'wip_cap': 2,
        }
        if regime_id == 'R4':
            return [
                PlantCellConfig(cell_id=0, label='cell-kacem-fast-tight', processing_time_scale=0.8, energy_cap_kwh=250.0, carbon_cap_kg=150.0, **common),
                PlantCellConfig(cell_id=1, label='cell-kacem-slow-loose', processing_time_scale=1.3, energy_cap_kwh=950.0, carbon_cap_kg=570.0, **common),
                PlantCellConfig(cell_id=2, label='cell-kacem-medium', processing_time_scale=1.0, energy_cap_kwh=700.0, carbon_cap_kg=420.0, **common),
            ]
        return [
            PlantCellConfig(cell_id=0, label='cell-kacem-fast', processing_time_scale=0.8, energy_cap_kwh=800.0, carbon_cap_kg=480.0, **common),
            PlantCellConfig(cell_id=1, label='cell-kacem-slow', processing_time_scale=1.3, energy_cap_kwh=950.0, carbon_cap_kg=570.0, **common),
            PlantCellConfig(cell_id=2, label='cell-kacem-medium', processing_time_scale=1.0, energy_cap_kwh=850.0, carbon_cap_kg=510.0, **common),
        ]

    if composition == 'fattahi_three_cell':
        common = {
            'instance_path': root / 'FattahiMehrabadJolai2007' / 'fattahi1.txt',
            'instance_kind': 'fattahi',
            'family_key': 'fattahi',
            'wip_cap': 2,
        }
        if regime_id == 'R4':
            return [
                PlantCellConfig(cell_id=0, label='cell-fattahi-fast-tight', processing_time_scale=0.85, energy_cap_kwh=320.0, carbon_cap_kg=192.0, **common),
                PlantCellConfig(cell_id=1, label='cell-fattahi-slow-loose', processing_time_scale=1.3, energy_cap_kwh=1050.0, carbon_cap_kg=630.0, **common),
                PlantCellConfig(cell_id=2, label='cell-fattahi-medium', processing_time_scale=1.0, energy_cap_kwh=820.0, carbon_cap_kg=492.0, **common),
            ]
        return [
            PlantCellConfig(cell_id=0, label='cell-fattahi-fast', processing_time_scale=0.85, energy_cap_kwh=900.0, carbon_cap_kg=540.0, **common),
            PlantCellConfig(cell_id=1, label='cell-fattahi-slow', processing_time_scale=1.3, energy_cap_kwh=1050.0, carbon_cap_kg=630.0, **common),
            PlantCellConfig(cell_id=2, label='cell-fattahi-medium', processing_time_scale=1.0, energy_cap_kwh=920.0, carbon_cap_kg=552.0, **common),
        ]

    if composition == 'kacem_four_cell':
        common = {
            'instance_path': root / 'KacemHammadiBorne2002' / 'kacem1.txt',
            'instance_kind': 'kacem',
            'family_key': 'kacem',
            'wip_cap': 2,
        }
        if regime_id == 'R4':
            return [
                PlantCellConfig(cell_id=0, label='cell-kacem-fast-tight', processing_time_scale=0.78, energy_cap_kwh=240.0, carbon_cap_kg=144.0, **common),
                PlantCellConfig(cell_id=1, label='cell-kacem-fast', processing_time_scale=0.9, energy_cap_kwh=720.0, carbon_cap_kg=432.0, **common),
                PlantCellConfig(cell_id=2, label='cell-kacem-medium', processing_time_scale=1.05, energy_cap_kwh=760.0, carbon_cap_kg=456.0, **common),
                PlantCellConfig(cell_id=3, label='cell-kacem-slow-loose', processing_time_scale=1.28, energy_cap_kwh=980.0, carbon_cap_kg=588.0, **common),
            ]
        return [
            PlantCellConfig(cell_id=0, label='cell-kacem-fast', processing_time_scale=0.78, energy_cap_kwh=820.0, carbon_cap_kg=492.0, **common),
            PlantCellConfig(cell_id=1, label='cell-kacem-fast-medium', processing_time_scale=0.9, energy_cap_kwh=840.0, carbon_cap_kg=504.0, **common),
            PlantCellConfig(cell_id=2, label='cell-kacem-medium', processing_time_scale=1.05, energy_cap_kwh=880.0, carbon_cap_kg=528.0, **common),
            PlantCellConfig(cell_id=3, label='cell-kacem-slow', processing_time_scale=1.28, energy_cap_kwh=980.0, carbon_cap_kg=588.0, **common),
        ]

    if composition == 'fattahi_four_cell':
        common = {
            'instance_path': root / 'FattahiMehrabadJolai2007' / 'fattahi1.txt',
            'instance_kind': 'fattahi',
            'family_key': 'fattahi',
            'wip_cap': 2,
        }
        if regime_id == 'R4':
            return [
                PlantCellConfig(cell_id=0, label='cell-fattahi-fast-tight', processing_time_scale=0.82, energy_cap_kwh=300.0, carbon_cap_kg=180.0, **common),
                PlantCellConfig(cell_id=1, label='cell-fattahi-fast', processing_time_scale=0.94, energy_cap_kwh=760.0, carbon_cap_kg=456.0, **common),
                PlantCellConfig(cell_id=2, label='cell-fattahi-medium', processing_time_scale=1.08, energy_cap_kwh=840.0, carbon_cap_kg=504.0, **common),
                PlantCellConfig(cell_id=3, label='cell-fattahi-slow-loose', processing_time_scale=1.28, energy_cap_kwh=1080.0, carbon_cap_kg=648.0, **common),
            ]
        return [
            PlantCellConfig(cell_id=0, label='cell-fattahi-fast', processing_time_scale=0.82, energy_cap_kwh=920.0, carbon_cap_kg=552.0, **common),
            PlantCellConfig(cell_id=1, label='cell-fattahi-fast-medium', processing_time_scale=0.94, energy_cap_kwh=950.0, carbon_cap_kg=570.0, **common),
            PlantCellConfig(cell_id=2, label='cell-fattahi-medium', processing_time_scale=1.08, energy_cap_kwh=980.0, carbon_cap_kg=588.0, **common),
            PlantCellConfig(cell_id=3, label='cell-fattahi-slow', processing_time_scale=1.28, energy_cap_kwh=1080.0, carbon_cap_kg=648.0, **common),
        ]

    if composition == 'brandimarte_ordertrap_three_cell':
        common = {
            'instance_path': root / 'Brandimarte1993' / 'mk01.txt',
            'instance_kind': 'brandimarte',
            'family_key': 'brandimarte',
            'wip_cap': 2,
        }
        if regime_id == 'R4':
            return [
                PlantCellConfig(cell_id=0, label='cell-brandimarte-slow-tight', processing_time_scale=1.38, energy_cap_kwh=180.0, carbon_cap_kg=108.0, **common),
                PlantCellConfig(cell_id=1, label='cell-brandimarte-fast-loose', processing_time_scale=0.74, energy_cap_kwh=1500.0, carbon_cap_kg=900.0, **common),
                PlantCellConfig(cell_id=2, label='cell-brandimarte-medium', processing_time_scale=1.0, energy_cap_kwh=620.0, carbon_cap_kg=372.0, **common),
            ]
        return [
            PlantCellConfig(cell_id=0, label='cell-brandimarte-slow-tight', processing_time_scale=1.38, energy_cap_kwh=220.0, carbon_cap_kg=132.0, **common),
            PlantCellConfig(cell_id=1, label='cell-brandimarte-fast-loose', processing_time_scale=0.74, energy_cap_kwh=1500.0, carbon_cap_kg=900.0, **common),
            PlantCellConfig(cell_id=2, label='cell-brandimarte-medium', processing_time_scale=1.0, energy_cap_kwh=780.0, carbon_cap_kg=468.0, **common),
        ]

    if composition == 'brandimarte_ordertrap_five_cell':
        common = {
            'instance_path': root / 'Brandimarte1993' / 'mk01.txt',
            'instance_kind': 'brandimarte',
            'family_key': 'brandimarte',
            'wip_cap': 2,
        }
        if regime_id == 'R4':
            return [
                PlantCellConfig(cell_id=0, label='cell-brandimarte-slow-tight', processing_time_scale=1.42, energy_cap_kwh=180.0, carbon_cap_kg=108.0, **common),
                PlantCellConfig(cell_id=1, label='cell-brandimarte-fast-loose', processing_time_scale=0.72, energy_cap_kwh=1500.0, carbon_cap_kg=900.0, **common),
                PlantCellConfig(cell_id=2, label='cell-brandimarte-medium', processing_time_scale=1.0, energy_cap_kwh=760.0, carbon_cap_kg=456.0, **common),
                PlantCellConfig(cell_id=3, label='cell-brandimarte-fast-lean', processing_time_scale=0.88, energy_cap_kwh=620.0, carbon_cap_kg=372.0, **common),
                PlantCellConfig(cell_id=4, label='cell-brandimarte-buffer-tight', processing_time_scale=1.18, energy_cap_kwh=260.0, carbon_cap_kg=156.0, **common),
            ]
        return [
            PlantCellConfig(cell_id=0, label='cell-brandimarte-slow-tight', processing_time_scale=1.42, energy_cap_kwh=240.0, carbon_cap_kg=144.0, **common),
            PlantCellConfig(cell_id=1, label='cell-brandimarte-fast-loose', processing_time_scale=0.72, energy_cap_kwh=1500.0, carbon_cap_kg=900.0, **common),
            PlantCellConfig(cell_id=2, label='cell-brandimarte-medium', processing_time_scale=1.0, energy_cap_kwh=780.0, carbon_cap_kg=468.0, **common),
            PlantCellConfig(cell_id=3, label='cell-brandimarte-fast-lean', processing_time_scale=0.88, energy_cap_kwh=720.0, carbon_cap_kg=432.0, **common),
            PlantCellConfig(cell_id=4, label='cell-brandimarte-buffer-tight', processing_time_scale=1.18, energy_cap_kwh=340.0, carbon_cap_kg=204.0, **common),
        ]

    if composition == 'brandimarte_ordertrap_nine_cell':
        common = {
            'instance_path': root / 'Brandimarte1993' / 'mk01.txt',
            'instance_kind': 'brandimarte',
            'family_key': 'brandimarte',
            'wip_cap': 2,
        }
        return [
            PlantCellConfig(cell_id=0, label='area0-front-fast', processing_time_scale=0.84, energy_cap_kwh=900.0, carbon_cap_kg=540.0, **common),
            PlantCellConfig(cell_id=1, label='area0-mid-balanced', processing_time_scale=1.02, energy_cap_kwh=760.0, carbon_cap_kg=456.0, **common),
            PlantCellConfig(cell_id=2, label='area0-tail-tight', processing_time_scale=1.20, energy_cap_kwh=520.0, carbon_cap_kg=312.0, **common),
            PlantCellConfig(cell_id=3, label='area1-front-balanced', processing_time_scale=0.92, energy_cap_kwh=860.0, carbon_cap_kg=516.0, **common),
            PlantCellConfig(cell_id=4, label='area1-mid-bottleneck', processing_time_scale=1.18, energy_cap_kwh=620.0, carbon_cap_kg=372.0, **common),
            PlantCellConfig(cell_id=5, label='area1-tail-tight', processing_time_scale=1.34, energy_cap_kwh=460.0, carbon_cap_kg=276.0, **common),
            PlantCellConfig(cell_id=6, label='area2-front-fast', processing_time_scale=0.80, energy_cap_kwh=940.0, carbon_cap_kg=564.0, **common),
            PlantCellConfig(cell_id=7, label='area2-mid-balanced', processing_time_scale=0.98, energy_cap_kwh=780.0, carbon_cap_kg=468.0, **common),
            PlantCellConfig(cell_id=8, label='area2-tail-tight', processing_time_scale=1.16, energy_cap_kwh=540.0, carbon_cap_kg=324.0, **common),
        ]

    raise ValueError(f'Unknown tree shared-backlog composition: {composition}')


def _tree_shared_backlog_env_factory(area_to_cells: dict[int, list[int]]) -> Callable[..., TreeSharedBacklogEnv]:
    def factory(cell_configs: List[PlantCellConfig], config: BenchmarkConfig, seed: int) -> TreeSharedBacklogEnv:
        return TreeSharedBacklogEnv(
            cell_configs,
            area_to_cells=area_to_cells,
            config=config,
            seed=seed,
        )

    return factory


def _tree_runtime_env_factory(area_to_cells: dict[int, list[int]]) -> Callable[..., Any]:
    def factory(cell_configs: List[PlantCellConfig], config: BenchmarkConfig, seed: int) -> Any:
        from .v2.tree_runtime_env import TreeRuntimeEnv

        return TreeRuntimeEnv(
            cell_configs,
            area_to_cells=area_to_cells,
            config=config,
            seed=seed,
        )

    return factory


def _intercell_tree_runtime_env_factory(
    area_to_cells: dict[int, list[int]],
    *,
    max_in_flight_jobs: int,
    route_grammar: tuple[tuple[int, ...], ...] | None = None,
    backlog_top_k: int = 1,
    route_grammar_contract: str | None = None,
    authority_mode: str = DEFAULT_INTERCELL_AUTHORITY_MODE,
) -> Callable[..., Any]:
    def factory(cell_configs: List[PlantCellConfig], config: BenchmarkConfig, seed: int) -> Any:
        from .v2.intercell_tree_runtime_env import IntercellTreeRuntimeEnv

        return IntercellTreeRuntimeEnv(
            cell_configs,
            area_to_cells=area_to_cells,
            config=config,
            seed=seed,
            max_in_flight_jobs=max_in_flight_jobs,
            route_grammar=route_grammar,
            backlog_top_k=backlog_top_k,
            route_grammar_contract=route_grammar_contract,
            authority_mode=authority_mode,
        )

    return factory


def build_tree_benchmark_suite(repo_root: Path) -> List[HolonicExperimentCase]:
    repo_root = Path(repo_root)
    suite_tier = 'tree_shared_backlog_v1_small'
    seeds = [13]
    cases: List[HolonicExperimentCase] = []
    for composition in ('kacem_three_cell', 'fattahi_three_cell'):
        for regime_id in IMPLEMENTED_REGIMES:
            cell_configs = _build_tree_shared_backlog_cells(repo_root, composition, regime_id)
            for area_count in (2, 3):
                topology = build_tree_topology(cell_configs, area_count=area_count)
                benchmark_item_id = (
                    f'suite={suite_tier}__regime={regime_id}__topology={topology.topology_id}'
                    f'__cells={composition}__split=test'
                )
                cases.append(
                    HolonicExperimentCase(
                        case_id=f'{suite_tier}__{composition}__{topology.topology_id}__{regime_id}',
                        suite_tier=suite_tier,
                        level=regime_id,
                        benchmark_item_id=benchmark_item_id,
                        cell_configs=cell_configs,
                        seeds=list(seeds),
                        env_name='tree_shared_backlog',
                        env_cls=_tree_shared_backlog_env_factory(topology.area_to_cells),
                        regime_id=regime_id,
                        topology_id=topology.topology_id,
                    )
                )
    return cases


def build_tree_runtime_v2_small_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> List[HolonicExperimentCase]:
    return _build_tree_runtime_v2_suite(
        repo_root=repo_root,
        suite_tier='tree_runtime_v2_small',
        seed_split=seed_split,
        compositions=('kacem_three_cell', 'fattahi_three_cell'),
        regimes=('R0', 'R4'),
        numeric_profile_id=numeric_profile_id,
        due_date_profile_id=due_date_profile_id,
    )


def build_tree_runtime_v2_pilot_suite(
    repo_root: Path,
    *,
    seed_split: str = 'dev',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> List[HolonicExperimentCase]:
    return _build_tree_runtime_v2_suite(
        repo_root=repo_root,
        suite_tier='tree_runtime_v2_pilot',
        seed_split=seed_split,
        compositions=(
            'kacem_three_cell',
            'fattahi_three_cell',
            'brandimarte_ordertrap_three_cell',
        ),
        regimes=IMPLEMENTED_REGIMES,
        numeric_profile_id=numeric_profile_id,
        due_date_profile_id=due_date_profile_id,
    )


def build_tree_runtime_v2_wide_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> List[HolonicExperimentCase]:
    return _build_tree_runtime_v2_suite(
        repo_root=repo_root,
        suite_tier='tree_runtime_v2_wide',
        seed_split=seed_split,
        compositions=(
            'kacem_three_cell',
            'fattahi_three_cell',
            'brandimarte_ordertrap_three_cell',
        ),
        regimes=IMPLEMENTED_REGIMES,
        numeric_profile_id=numeric_profile_id,
        due_date_profile_id=due_date_profile_id,
    )


def build_tree_runtime_v2_ladder_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> List[HolonicExperimentCase]:
    return _build_tree_runtime_v2_suite(
        repo_root=repo_root,
        suite_tier='tree_runtime_v2_ladder',
        seed_split=seed_split,
        compositions=(
            'kacem_three_cell',
            'kacem_four_cell',
            'fattahi_three_cell',
            'fattahi_four_cell',
            'brandimarte_ordertrap_three_cell',
            'brandimarte_ordertrap_five_cell',
        ),
        regimes=IMPLEMENTED_REGIMES,
        numeric_profile_id=numeric_profile_id,
        due_date_profile_id=due_date_profile_id,
    )


def build_tree_runtime_v2_intercell_small_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
    max_in_flight_jobs: int = 1,
    inbound_buffer_cap: int = 1,
) -> List[HolonicExperimentCase]:
    return [
        _build_intercell_case(
            repo_root=Path(repo_root),
            suite_tier='tree_runtime_v2_intercell_small',
            profile_id='locked_anchor',
            seed_split=seed_split,
            composition='kacem_intercell_three_cell',
            source_composition='kacem_three_cell',
            regime_id='R0',
            max_in_flight_jobs=max_in_flight_jobs,
            inbound_buffer_cap=inbound_buffer_cap,
            numeric_profile_id=numeric_profile_id,
            due_date_profile_id=due_date_profile_id,
        )
    ]


def build_tree_runtime_v2_intercell_brandimarte_probe_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> List[HolonicExperimentCase]:
    return [
        _build_intercell_case(
            repo_root=Path(repo_root),
            suite_tier='tree_runtime_v2_intercell_brandimarte_probe',
            profile_id='moderate_congestion',
            seed_split=seed_split,
            composition='brandimarte_ordertrap_three_cell',
            source_composition='brandimarte_ordertrap_three_cell',
            regime_id='R0',
            max_in_flight_jobs=2,
            inbound_buffer_cap=1,
            processing_scale_overrides={1: 1.2, 2: 1.05},
            numeric_profile_id=numeric_profile_id,
            due_date_profile_id=due_date_profile_id,
        )
    ]


def build_tree_runtime_v2_intercell_blocking_probe_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> List[HolonicExperimentCase]:
    return [
        _build_intercell_case(
            repo_root=Path(repo_root),
            suite_tier='tree_runtime_v2_intercell_blocking_probe',
            profile_id='blocking_anchor',
            seed_split=seed_split,
            composition='brandimarte_ordertrap_three_cell',
            source_composition='brandimarte_ordertrap_three_cell',
            regime_id='R0',
            max_in_flight_jobs=2,
            inbound_buffer_cap=1,
            inbound_buffer_cap_overrides={0: 1, 1: 1, 2: 0},
            processing_scale_overrides={0: 0.65, 1: 2.2, 2: 1.0},
            numeric_profile_id=numeric_profile_id,
            due_date_profile_id=due_date_profile_id,
        )
    ]


def build_tree_runtime_v2_intercell_wide_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> List[HolonicExperimentCase]:
    repo_root = Path(repo_root)
    return [
        _build_intercell_case(
            repo_root=repo_root,
            suite_tier='tree_runtime_v2_intercell_wide',
            profile_id='locked_anchor',
            seed_split=seed_split,
            composition='kacem_intercell_three_cell',
            source_composition='kacem_three_cell',
            regime_id='R0',
            max_in_flight_jobs=1,
            inbound_buffer_cap=1,
            numeric_profile_id=numeric_profile_id,
            due_date_profile_id=due_date_profile_id,
        ),
        _build_intercell_case(
            repo_root=repo_root,
            suite_tier='tree_runtime_v2_intercell_wide',
            profile_id='moderate_congestion',
            seed_split=seed_split,
            composition='brandimarte_ordertrap_three_cell',
            source_composition='brandimarte_ordertrap_three_cell',
            regime_id='R0',
            max_in_flight_jobs=2,
            inbound_buffer_cap=1,
            processing_scale_overrides={1: 1.2, 2: 1.05},
            numeric_profile_id=numeric_profile_id,
            due_date_profile_id=due_date_profile_id,
        ),
        _build_intercell_case(
            repo_root=repo_root,
            suite_tier='tree_runtime_v2_intercell_wide',
            profile_id='stress_congestion',
            seed_split=seed_split,
            composition='brandimarte_ordertrap_three_cell',
            source_composition='brandimarte_ordertrap_three_cell',
            regime_id='R0',
            max_in_flight_jobs=2,
            inbound_buffer_cap=1,
            processing_scale_overrides={1: 2.0, 2: 1.0},
            numeric_profile_id=numeric_profile_id,
            due_date_profile_id=due_date_profile_id,
        ),
    ]


def build_tree_runtime_v2_intercell_dynamics_extension_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> List[HolonicExperimentCase]:
    repo_root = Path(repo_root)
    shared_kwargs = {
        'repo_root': repo_root,
        'suite_tier': 'tree_runtime_v2_intercell_dynamics_extension',
        'seed_split': seed_split,
        'composition': 'brandimarte_ordertrap_three_cell',
        'source_composition': 'brandimarte_ordertrap_three_cell',
        'regime_id': 'R0',
        'max_in_flight_jobs': 2,
        'inbound_buffer_cap': 1,
        'processing_scale_overrides': {1: 1.2, 2: 1.05},
        'numeric_profile_id': numeric_profile_id,
        'due_date_profile_id': due_date_profile_id,
        'arrival_profile_id': 'arrival_nhpp_random_batch_nominal_v1',
        'setup_profile_id': 'setup_family_change_nominal_v1',
    }
    return [
        _build_intercell_case(
            **shared_kwargs,
            profile_id='arrival_setup_failure_none',
            failure_profile_id='failure_none_v1',
        ),
        _build_intercell_case(
            **shared_kwargs,
            profile_id='arrival_setup_failure_exponential',
            failure_profile_id='failure_exponential_nominal_v1',
        ),
        _build_intercell_case(
            **shared_kwargs,
            profile_id='arrival_setup_failure_weibull',
            failure_profile_id='failure_weibull_aging_nominal_v1',
        ),
        _build_intercell_case(
            **shared_kwargs,
            profile_id='arrival_setup_failure_load_dependent',
            failure_profile_id='failure_load_dependent_nominal_v1',
        ),
    ]


_A3C9_CANONICAL_ROUTE_GRAMMAR = (
    (0, 1, 3, 4, 6, 7),
    (1, 2, 4, 5, 7, 8),
    (0, 2, 3, 5, 6, 8),
)


def build_tree_runtime_v2_intercell_a3c9_probe_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> List[HolonicExperimentCase]:
    return [
        _build_intercell_case(
            repo_root=Path(repo_root),
            suite_tier='tree_runtime_v2_intercell_a3c9_probe',
            profile_id='moderate_expansion',
            seed_split=seed_split,
            composition='brandimarte_ordertrap_nine_cell',
            source_composition='brandimarte_ordertrap_nine_cell',
            regime_id='R0',
            max_in_flight_jobs=3,
            inbound_buffer_cap=1,
            processing_scale_overrides={4: 1.28, 5: 1.42, 8: 1.22},
            numeric_profile_id=numeric_profile_id,
            due_date_profile_id=due_date_profile_id,
            area_count=3,
            route_grammar=_A3C9_CANONICAL_ROUTE_GRAMMAR,
            backlog_top_k=3,
            route_grammar_contract='a3c9_canonical_non_degenerate_v1',
        )
    ]


def build_tree_runtime_v2_intercell_a3c9_wide_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
    authority_modes: Sequence[str] | None = None,
) -> List[HolonicExperimentCase]:
    repo_root = Path(repo_root)
    mode_ids = _resolve_intercell_authority_modes(
        authority_modes,
        default_modes=MAINLINE_CATEGORY_B_AUTHORITY_MODES,
    )
    shared_kwargs = {
        'repo_root': repo_root,
        'suite_tier': 'tree_runtime_v2_intercell_a3c9_wide',
        'seed_split': seed_split,
        'composition': 'brandimarte_ordertrap_nine_cell',
        'source_composition': 'brandimarte_ordertrap_nine_cell',
        'regime_id': 'R0',
        'area_count': 3,
        'route_grammar': _A3C9_CANONICAL_ROUTE_GRAMMAR,
        'route_grammar_contract': 'a3c9_canonical_non_degenerate_v1',
        'backlog_top_k': 3,
        'numeric_profile_id': numeric_profile_id,
        'due_date_profile_id': due_date_profile_id,
        'arrival_profile_id': 'arrival_nhpp_random_batch_overlap_a3c9_v2',
        'setup_profile_id': 'setup_family_change_nominal_v1',
    }
    profile_specs = (
        {
            'profile_id': 'balanced_branching_anchor',
            'max_in_flight_jobs': 3,
            'inbound_buffer_cap': 1,
            'processing_scale_overrides': {4: 1.15, 5: 1.10, 7: 1.05},
            'failure_profile_id': 'failure_exponential_nominal_v1',
        },
        {
            'profile_id': 'cross_area_tradeoff',
            'max_in_flight_jobs': 4,
            'inbound_buffer_cap': 1,
            'processing_scale_overrides': {1: 1.20, 4: 1.35, 7: 0.95},
            'transport_time_multiplier': 1.15,
            'failure_profile_id': 'failure_weibull_aging_nominal_v1',
        },
        {
            'profile_id': 'late_stage_reversal',
            'max_in_flight_jobs': 3,
            'inbound_buffer_cap': 1,
            'inbound_buffer_cap_overrides': {2: 0, 5: 0, 6: 0, 8: 0},
            'processing_scale_overrides': {0: 1.80, 1: 1.20, 3: 1.80, 4: 1.20, 7: 1.10},
            'transport_time_multiplier': 1.30,
            'failure_profile_id': 'failure_load_dependent_nominal_v1',
        },
    )
    return [
        _build_intercell_case(
            **shared_kwargs,
            **profile_spec,
            authority_mode=authority_mode,
            include_authority_tag=True,
        )
        for profile_spec in profile_specs
        for authority_mode in mode_ids
    ]


def build_tree_runtime_v2_intercell_a3c9_dynamics_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
    authority_modes: Sequence[str] | None = None,
) -> List[HolonicExperimentCase]:
    repo_root = Path(repo_root)
    mode_ids = _resolve_intercell_authority_modes(
        authority_modes,
        default_modes=MAINLINE_CATEGORY_B_AUTHORITY_MODES,
    )
    shared_kwargs = {
        'repo_root': repo_root,
        'suite_tier': 'tree_runtime_v2_intercell_a3c9_dynamics',
        'seed_split': seed_split,
        'composition': 'brandimarte_ordertrap_nine_cell',
        'source_composition': 'brandimarte_ordertrap_nine_cell',
        'regime_id': 'R0',
        'max_in_flight_jobs': 4,
        'inbound_buffer_cap': 1,
        'area_count': 3,
        'route_grammar': _A3C9_CANONICAL_ROUTE_GRAMMAR,
        'route_grammar_contract': 'a3c9_canonical_non_degenerate_v1',
        'backlog_top_k': 3,
        'numeric_profile_id': numeric_profile_id,
        'due_date_profile_id': due_date_profile_id,
        'arrival_profile_id': 'arrival_nhpp_random_batch_overlap_a3c9_v2',
        'setup_profile_id': 'setup_family_change_nominal_v1',
    }
    profile_specs = (
        {
            'profile_id': 'arrival_setup_exponential',
            'failure_profile_id': 'failure_exponential_nominal_v1',
            'inbound_buffer_cap_overrides': {2: 0, 3: 0, 5: 0, 6: 0, 8: 0},
            'processing_scale_overrides': {0: 1.55, 1: 1.10, 4: 1.20, 7: 1.05},
        },
        {
            'profile_id': 'arrival_setup_weibull',
            'failure_profile_id': 'failure_weibull_aging_nominal_v1',
            'inbound_buffer_cap_overrides': {2: 0, 3: 0, 5: 0, 6: 0, 8: 0},
            'processing_scale_overrides': {4: 1.30, 8: 1.10},
        },
        {
            'profile_id': 'arrival_setup_load_dependent',
            'failure_profile_id': 'failure_load_dependent_nominal_v1',
            'inbound_buffer_cap_overrides': {2: 0, 3: 0, 5: 0, 6: 0, 8: 0},
            'processing_scale_overrides': {0: 1.20, 4: 1.20, 5: 1.45, 8: 1.20},
        },
    )
    return [
        _build_intercell_case(
            **shared_kwargs,
            **profile_spec,
            authority_mode=authority_mode,
            include_authority_tag=True,
        )
        for profile_spec in profile_specs
        for authority_mode in mode_ids
    ]


def build_tree_runtime_v2_intercell_a3c9_dynamics_budget_sweep_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> List[HolonicExperimentCase]:
    base_suite = build_tree_runtime_v2_intercell_a3c9_dynamics_suite(
        repo_root,
        seed_split=seed_split,
        numeric_profile_id=numeric_profile_id,
        due_date_profile_id=due_date_profile_id,
        authority_modes=(DEFAULT_INTERCELL_AUTHORITY_MODE,),
    )
    sweep_suite: list[HolonicExperimentCase] = []
    for base_case in base_suite:
        for budget_tier, ratio in A3C9_DYNAMICS_BUDGET_RATIOS.items():
            horizon_hours = A3C9_DYNAMICS_NOMINAL_HORIZON_HOURS * float(ratio)
            sweep_suite.append(
                replace(
                    base_case,
                    case_id=(
                        base_case.case_id.replace(
                            'tree_runtime_v2_intercell_a3c9_dynamics',
                            'tree_runtime_v2_intercell_a3c9_dynamics_budget_sweep',
                        )
                        + f'__budget_{budget_tier}'
                    ),
                    suite_tier='tree_runtime_v2_intercell_a3c9_dynamics_budget_sweep',
                    benchmark_item_id=(
                        base_case.benchmark_item_id.replace(
                            'suite=tree_runtime_v2_intercell_a3c9_dynamics',
                            'suite=tree_runtime_v2_intercell_a3c9_dynamics_budget_sweep',
                        )
                        + f'__budget={budget_tier}__bh={horizon_hours:.1f}__br={ratio:.2f}'
                    ),
                    horizon_hours_override=horizon_hours,
                    budget_tier=budget_tier,
                    budget_relative_to_nominal=float(ratio),
                    official_surface_role='appendix_budget_extension',
                )
            )
    return sweep_suite


def build_tree_runtime_v2_intercell_a3c9_nightmare_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
    authority_modes: Sequence[str] | None = None,
) -> List[HolonicExperimentCase]:
    repo_root = Path(repo_root)
    mode_ids = _resolve_intercell_authority_modes(
        authority_modes,
        default_modes=MAINLINE_CATEGORY_B_AUTHORITY_MODES,
    )
    shared_kwargs = {
        'repo_root': repo_root,
        'suite_tier': 'tree_runtime_v2_intercell_a3c9_nightmare',
        'seed_split': seed_split,
        'composition': 'brandimarte_ordertrap_nine_cell',
        'source_composition': 'brandimarte_ordertrap_nine_cell',
        'regime_id': 'R0',
        'max_in_flight_jobs': 5,
        'inbound_buffer_cap': 1,
        'area_count': 3,
        'route_grammar': _A3C9_CANONICAL_ROUTE_GRAMMAR,
        'route_grammar_contract': 'a3c9_canonical_non_degenerate_v1',
        'backlog_top_k': 5,
        'numeric_profile_id': numeric_profile_id,
        'due_date_profile_id': due_date_profile_id,
        'arrival_profile_id': 'arrival_nhpp_random_batch_nominal_v1',
        'setup_profile_id': 'setup_family_change_nominal_v1',
    }
    profile_specs = (
        {
            'profile_id': 'blocking_reroute_choke',
            'inbound_buffer_cap_overrides': {2: 0, 4: 0, 5: 0, 7: 0, 8: 0},
            'processing_scale_overrides': {1: 1.55, 4: 1.95, 7: 1.70},
            'transport_time_multiplier': 1.45,
            'failure_profile_id': 'failure_weibull_aging_nominal_v1',
        },
        {
            'profile_id': 'night_shift_breakdown_pressure',
            'inbound_buffer_cap_overrides': {2: 0, 4: 0, 5: 0, 7: 0, 8: 0},
            'processing_scale_overrides': {1: 1.45, 4: 1.85, 7: 1.60},
            'transport_time_multiplier': 1.50,
            'failure_profile_id': 'failure_load_dependent_nominal_v1',
        },
    )
    return [
        _build_intercell_case(
            **shared_kwargs,
            **profile_spec,
            authority_mode=authority_mode,
            include_authority_tag=True,
        )
        for profile_spec in profile_specs
        for authority_mode in mode_ids
    ]


def _semantic_arrival_mode_from_profile(arrival_profile_id: str | None) -> str:
    if arrival_profile_id is None:
        return 'static_release'
    if 'nhpp_random_batch_overlap' in arrival_profile_id:
        return 'online_nhpp_random_batch_overlap'
    if 'nhpp_random_batch' in arrival_profile_id:
        return 'online_nhpp_random_batch'
    return arrival_profile_id


def _semantic_disturbance_family_from_profile(failure_profile_id: str | None) -> str:
    if failure_profile_id is None or failure_profile_id == 'failure_none_v1':
        return 'none'
    if 'load_dependent' in failure_profile_id:
        return 'load_dependent'
    if 'weibull' in failure_profile_id:
        return 'weibull_aging'
    if 'exponential' in failure_profile_id:
        return 'exponential'
    return failure_profile_id


def _official_surface_id_for_suite_tier(suite_tier: str) -> str:
    if suite_tier in {
        'tree_runtime_v2_intercell_a3c9_wide',
        'tree_runtime_v2_intercell_a3c9_dynamics',
        'tree_runtime_v2_intercell_a3c9_dynamics_budget_sweep',
    }:
        return 'intercell_a3c9_wide'
    if suite_tier == 'tree_runtime_v2_intercell_a3c9_nightmare':
        return 'intercell_a3c9_nightmare'
    if suite_tier.startswith('tree_runtime_v2_'):
        return suite_tier.removeprefix('tree_runtime_v2_')
    return suite_tier


def _official_surface_role_for_suite_tier(suite_tier: str) -> str:
    if suite_tier == 'tree_runtime_v2_intercell_a3c9_wide':
        return 'official_standard'
    if suite_tier == 'tree_runtime_v2_intercell_a3c9_nightmare':
        return 'official_nightmare'
    if suite_tier == 'tree_runtime_v2_intercell_a3c9_dynamics':
        return 'deprecated_alias'
    if suite_tier == 'tree_runtime_v2_intercell_a3c9_dynamics_budget_sweep':
        return 'appendix_budget_extension'
    if suite_tier == 'tree_runtime_v2_intercell_a3c9_probe':
        return 'internal_probe'
    return 'internal_extension'


def _intercell_harder_congestion_candidate_specs() -> tuple[IntercellCongestionProfileSpec, ...]:
    return (
        IntercellCongestionProfileSpec(
            profile_id='hc_k2_tail_mid_tm100',
            max_in_flight_jobs=2,
            inbound_buffer_cap_overrides={0: 1, 1: 1, 2: 0},
            processing_scale_overrides={1: 2.2, 2: 1.0},
            transport_time_multiplier=1.0,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k2_tail_mid_tm150',
            max_in_flight_jobs=2,
            inbound_buffer_cap_overrides={0: 1, 1: 1, 2: 0},
            processing_scale_overrides={1: 2.2, 2: 1.0},
            transport_time_multiplier=1.5,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k2_tail_extreme_tm100',
            max_in_flight_jobs=2,
            inbound_buffer_cap_overrides={0: 1, 1: 1, 2: 0},
            processing_scale_overrides={0: 0.75, 1: 2.6, 2: 1.15},
            transport_time_multiplier=1.0,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k2_tail_extreme_tm150',
            max_in_flight_jobs=2,
            inbound_buffer_cap_overrides={0: 1, 1: 1, 2: 0},
            processing_scale_overrides={0: 0.75, 1: 2.6, 2: 1.15},
            transport_time_multiplier=1.5,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k2_deep_mid_tm100',
            max_in_flight_jobs=2,
            inbound_buffer_cap_overrides={0: 1, 1: 0, 2: 0},
            processing_scale_overrides={1: 2.2, 2: 1.0},
            transport_time_multiplier=1.0,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k2_deep_mid_tm150',
            max_in_flight_jobs=2,
            inbound_buffer_cap_overrides={0: 1, 1: 0, 2: 0},
            processing_scale_overrides={1: 2.2, 2: 1.0},
            transport_time_multiplier=1.5,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k2_deep_extreme_tm100',
            max_in_flight_jobs=2,
            inbound_buffer_cap_overrides={0: 1, 1: 0, 2: 0},
            processing_scale_overrides={0: 0.75, 1: 2.6, 2: 1.15},
            transport_time_multiplier=1.0,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k2_deep_extreme_tm150',
            max_in_flight_jobs=2,
            inbound_buffer_cap_overrides={0: 1, 1: 0, 2: 0},
            processing_scale_overrides={0: 0.75, 1: 2.6, 2: 1.15},
            transport_time_multiplier=1.5,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k3_tail_mid_tm100',
            max_in_flight_jobs=3,
            inbound_buffer_cap_overrides={0: 1, 1: 1, 2: 0},
            processing_scale_overrides={1: 2.2, 2: 1.0},
            transport_time_multiplier=1.0,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k3_tail_mid_tm150',
            max_in_flight_jobs=3,
            inbound_buffer_cap_overrides={0: 1, 1: 1, 2: 0},
            processing_scale_overrides={1: 2.2, 2: 1.0},
            transport_time_multiplier=1.5,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k3_tail_extreme_tm100',
            max_in_flight_jobs=3,
            inbound_buffer_cap_overrides={0: 1, 1: 1, 2: 0},
            processing_scale_overrides={0: 0.75, 1: 2.6, 2: 1.15},
            transport_time_multiplier=1.0,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k3_tail_extreme_tm150',
            max_in_flight_jobs=3,
            inbound_buffer_cap_overrides={0: 1, 1: 1, 2: 0},
            processing_scale_overrides={0: 0.75, 1: 2.6, 2: 1.15},
            transport_time_multiplier=1.5,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k3_deep_mid_tm100',
            max_in_flight_jobs=3,
            inbound_buffer_cap_overrides={0: 1, 1: 0, 2: 0},
            processing_scale_overrides={1: 2.2, 2: 1.0},
            transport_time_multiplier=1.0,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k3_deep_mid_tm150',
            max_in_flight_jobs=3,
            inbound_buffer_cap_overrides={0: 1, 1: 0, 2: 0},
            processing_scale_overrides={1: 2.2, 2: 1.0},
            transport_time_multiplier=1.5,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k3_deep_extreme_tm100',
            max_in_flight_jobs=3,
            inbound_buffer_cap_overrides={0: 1, 1: 0, 2: 0},
            processing_scale_overrides={0: 0.75, 1: 2.6, 2: 1.15},
            transport_time_multiplier=1.0,
        ),
        IntercellCongestionProfileSpec(
            profile_id='hc_k3_deep_extreme_tm150',
            max_in_flight_jobs=3,
            inbound_buffer_cap_overrides={0: 1, 1: 0, 2: 0},
            processing_scale_overrides={0: 0.75, 1: 2.6, 2: 1.15},
            transport_time_multiplier=1.5,
        ),
    )


def build_tree_runtime_v2_intercell_harder_congestion_candidates_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> List[HolonicExperimentCase]:
    repo_root = Path(repo_root)
    return [
        _build_intercell_case(
            repo_root=repo_root,
            suite_tier='tree_runtime_v2_intercell_harder_congestion_candidates',
            profile_id=spec.profile_id,
            seed_split=seed_split,
            composition='brandimarte_ordertrap_three_cell',
            source_composition='brandimarte_ordertrap_three_cell',
            regime_id='R0',
            max_in_flight_jobs=spec.max_in_flight_jobs,
            inbound_buffer_cap=1,
            inbound_buffer_cap_overrides=spec.inbound_buffer_cap_overrides,
            processing_scale_overrides=spec.processing_scale_overrides,
            transport_time_multiplier=spec.transport_time_multiplier,
            numeric_profile_id=numeric_profile_id,
            due_date_profile_id=due_date_profile_id,
        )
        for spec in _intercell_harder_congestion_candidate_specs()
    ]


def build_tree_runtime_v2_intercell_harder_congestion_suite(
    repo_root: Path,
    *,
    seed_split: str = 'test',
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> List[HolonicExperimentCase]:
    selected_profile_ids = {
        'hc_k2_tail_extreme_tm100',
        'hc_k2_tail_mid_tm150',
        'hc_k3_tail_extreme_tm150',
    }
    repo_root = Path(repo_root)
    return [
        _build_intercell_case(
            repo_root=repo_root,
            suite_tier='tree_runtime_v2_intercell_harder_congestion',
            profile_id=spec.profile_id,
            seed_split=seed_split,
            composition='brandimarte_ordertrap_three_cell',
            source_composition='brandimarte_ordertrap_three_cell',
            regime_id='R0',
            max_in_flight_jobs=spec.max_in_flight_jobs,
            inbound_buffer_cap=1,
            inbound_buffer_cap_overrides=spec.inbound_buffer_cap_overrides,
            processing_scale_overrides=spec.processing_scale_overrides,
            transport_time_multiplier=spec.transport_time_multiplier,
            numeric_profile_id=numeric_profile_id,
            due_date_profile_id=due_date_profile_id,
        )
        for spec in _intercell_harder_congestion_candidate_specs()
        if spec.profile_id in selected_profile_ids
    ]


def _build_intercell_case(
    *,
    repo_root: Path,
    suite_tier: str,
    profile_id: str,
    seed_split: str,
    composition: str,
    source_composition: str,
    regime_id: str,
    max_in_flight_jobs: int,
    inbound_buffer_cap: int,
    inbound_buffer_cap_overrides: dict[int, int] | None = None,
    processing_scale_overrides: dict[int, float] | None = None,
    transport_time_multiplier: float = 1.0,
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
    setup_profile_id: str | None = None,
    arrival_profile_id: str | None = None,
    failure_profile_id: str | None = None,
    area_count: int = 2,
    route_grammar: tuple[tuple[int, ...], ...] | None = None,
    backlog_top_k: int = 1,
    route_grammar_contract: str | None = None,
    authority_mode: str = DEFAULT_INTERCELL_AUTHORITY_MODE,
    include_authority_tag: bool = False,
) -> HolonicExperimentCase:
    seeds = load_benchmark_seed_split(repo_root, seed_split)
    cell_configs = _build_tree_shared_backlog_cells(repo_root, source_composition, regime_id)
    overrides = processing_scale_overrides or {}
    inbound_overrides = inbound_buffer_cap_overrides or {}
    cell_configs = [
        replace(
            cell,
            processing_time_scale=float(overrides.get(cell.cell_id, cell.processing_time_scale)),
            inbound_buffer_cap=int(inbound_overrides.get(cell.cell_id, inbound_buffer_cap)),
            ingress_transport_time_hours=(1.0 if cell.cell_id == 0 else 1.5) * float(transport_time_multiplier),
            ingress_transport_energy_kwh=(0.2 if cell.cell_id == 0 else 0.3) * float(transport_time_multiplier),
        )
        for cell in cell_configs
    ]
    topology = build_tree_topology(cell_configs, area_count=area_count)

    case_id = f'{suite_tier}__{composition}__{topology.topology_id}__{regime_id}'
    benchmark_item_id = (
        f'suite={suite_tier}__regime={regime_id}__topology={topology.topology_id}'
        f'__cells={composition}__profile={profile_id}'
    )
    if max_in_flight_jobs != 1:
        benchmark_item_id += f'__k={max_in_flight_jobs}'
    if inbound_buffer_cap != 1:
        benchmark_item_id += f'__b={inbound_buffer_cap}'
    if inbound_overrides:
        for cell_id in sorted(inbound_overrides):
            benchmark_item_id += f'__ib{cell_id}={int(inbound_overrides[cell_id])}'
    elif composition.startswith('brandimarte'):
        benchmark_item_id += '__b=1'
    for cell_id in sorted(overrides):
        benchmark_item_id += f'__c{cell_id}={overrides[cell_id]:.2f}'
    if transport_time_multiplier != 1.0:
        benchmark_item_id += f'__tm={transport_time_multiplier:.2f}'
    if backlog_top_k != 1:
        benchmark_item_id += f'__bk={backlog_top_k}'
    if setup_profile_id is not None:
        benchmark_item_id += f'__setup={setup_profile_id}'
    if arrival_profile_id is not None:
        benchmark_item_id += f'__arrival={arrival_profile_id}'
    if failure_profile_id is not None:
        benchmark_item_id += f'__failure={failure_profile_id}'
    if route_grammar is not None:
        route_tag = "_".join("-".join(str(cell_id) for cell_id in stage_cells) for stage_cells in route_grammar)
        benchmark_item_id += f'__rg={route_tag}'
    if include_authority_tag:
        benchmark_item_id += f'__authority={authority_mode}'
    benchmark_item_id += f'__split={seed_split}'

    if suite_tier in {
        'tree_runtime_v2_intercell_a3c9_wide',
        'tree_runtime_v2_intercell_a3c9_dynamics',
        'tree_runtime_v2_intercell_a3c9_nightmare',
        'tree_runtime_v2_intercell_wide',
        'tree_runtime_v2_intercell_dynamics_extension',
        'tree_runtime_v2_intercell_harder_congestion_candidates',
        'tree_runtime_v2_intercell_harder_congestion',
    }:
        case_id += f'__{profile_id}'
    if include_authority_tag:
        case_id += f'__authority_{authority_mode}'

    semantic_arrival_mode = _semantic_arrival_mode_from_profile(arrival_profile_id)
    semantic_disturbance_family = _semantic_disturbance_family_from_profile(failure_profile_id)
    official_surface_id = _official_surface_id_for_suite_tier(suite_tier)
    official_surface_role = _official_surface_role_for_suite_tier(suite_tier)
    semantic_profile_id = (
        f'{official_surface_id}__{profile_id}'
        f'__arrival_{semantic_arrival_mode}'
        f'__disturbance_{semantic_disturbance_family}'
    )

    return HolonicExperimentCase(
        case_id=case_id,
        suite_tier=suite_tier,
        level=regime_id,
        benchmark_item_id=benchmark_item_id,
        cell_configs=cell_configs,
        seeds=list(seeds),
        env_name='tree_runtime_v2_intercell',
        env_cls=_intercell_tree_runtime_env_factory(
            topology.area_to_cells,
            max_in_flight_jobs=max_in_flight_jobs,
            route_grammar=route_grammar,
            backlog_top_k=backlog_top_k,
            route_grammar_contract=route_grammar_contract,
            authority_mode=authority_mode,
        ),
        regime_id=regime_id,
        topology_id=topology.topology_id,
        numeric_profile_id=numeric_profile_id,
        due_date_profile_id=due_date_profile_id,
        setup_profile_id=setup_profile_id,
        arrival_profile_id=arrival_profile_id,
        failure_profile_id=failure_profile_id,
        profile_id=profile_id,
        authority_mode_id=(authority_mode if include_authority_tag else None),
        semantic_profile_id=semantic_profile_id,
        semantic_arrival_mode=semantic_arrival_mode,
        semantic_disturbance_family=semantic_disturbance_family,
        semantic_setup_enabled=setup_profile_id is not None,
        semantic_buffer_wip_enabled=True,
        semantic_transport_enabled=True,
        official_surface_id=official_surface_id,
        official_surface_role=official_surface_role,
    )


def _resolve_intercell_authority_modes(
    authority_modes: Sequence[str] | None,
    *,
    default_modes: Sequence[str],
) -> tuple[str, ...]:
    from .v2.authority_modes import get_authority_mode

    requested = tuple(default_modes if authority_modes is None else authority_modes)
    resolved: list[str] = []
    for authority_mode in requested:
        mode_id = get_authority_mode(str(authority_mode)).mode_id
        if mode_id not in resolved:
            resolved.append(mode_id)
    if not resolved:
        raise ValueError('authority_modes must not be empty')
    return tuple(resolved)


def _tree_runtime_area_counts(cell_count: int) -> tuple[int, ...]:
    return tuple(area_count for area_count in (2, 3) if area_count <= cell_count)


def _build_tree_runtime_v2_suite(
    *,
    repo_root: Path,
    suite_tier: str,
    seed_split: str,
    compositions: Sequence[str],
    regimes: Sequence[str],
    numeric_profile_id: str | None = None,
    due_date_profile_id: str | None = None,
) -> List[HolonicExperimentCase]:
    repo_root = Path(repo_root)
    seeds = load_benchmark_seed_split(repo_root, seed_split)
    cases: List[HolonicExperimentCase] = []
    for composition in compositions:
        for regime_id in regimes:
            cell_configs = _build_tree_shared_backlog_cells(repo_root, composition, regime_id)
            for area_count in _tree_runtime_area_counts(len(cell_configs)):
                topology = build_tree_topology(cell_configs, area_count=area_count)
                benchmark_item_id = (
                    f'suite={suite_tier}__regime={regime_id}__topology={topology.topology_id}'
                    f'__cells={composition}__split={seed_split}'
                )
                cases.append(
                    HolonicExperimentCase(
                        case_id=f'{suite_tier}__{composition}__{topology.topology_id}__{regime_id}',
                        suite_tier=suite_tier,
                        level=regime_id,
                        benchmark_item_id=benchmark_item_id,
                        cell_configs=cell_configs,
                        seeds=list(seeds),
                        env_name='tree_runtime_v2',
                        env_cls=_tree_runtime_env_factory(topology.area_to_cells),
                        regime_id=regime_id,
                        topology_id=topology.topology_id,
                        numeric_profile_id=numeric_profile_id,
                        due_date_profile_id=due_date_profile_id,
                    )
                )
    return cases


def _case_config(case: HolonicExperimentCase) -> BenchmarkConfig:
    if case.regime_id is not None:
        config = make_regime_config(
            case.regime_id,
            numeric_profile_id=case.numeric_profile_id,
            due_date_profile_id=case.due_date_profile_id,
        )
    else:
        config = make_level_config(
            case.level,
            numeric_profile_id=case.numeric_profile_id,
            due_date_profile_id=case.due_date_profile_id,
        )
    if case.setup_profile_id is not None:
        apply_setup_profile(config, case.setup_profile_id)
    if case.arrival_profile_id is not None:
        apply_arrival_profile(config, case.arrival_profile_id)
    if case.failure_profile_id is not None:
        apply_failure_profile(config, case.failure_profile_id)
    if case.horizon_hours_override is not None:
        config.horizon_hours = float(case.horizon_hours_override)
    return config


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def _write_csv_rows(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return path

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _validate_unique_case_policy_rows(rows: Sequence[dict[str, Any]]) -> None:
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_pairs: list[str] = []
    for row in rows:
        key = (str(row.get('case_id', '')), str(row.get('policy_name', '')))
        if key in seen_pairs:
            duplicate_pairs.append(f'{key[0]} / {key[1]}')
            continue
        seen_pairs.add(key)
    if duplicate_pairs:
        raise ValueError(
            'duplicate case/policy rows: ' + ', '.join(sorted(duplicate_pairs))
        )


def merge_suite_summary_csvs(
    *,
    summary_csv_paths: Sequence[str | Path],
    output_csv: str | Path,
) -> Path:
    rows: list[dict[str, str]] = []
    for summary_csv in summary_csv_paths:
        summary_path = Path(summary_csv)
        rows.extend(_load_csv_rows(summary_path))
    _validate_unique_case_policy_rows(rows)
    return _write_csv_rows(Path(output_csv), rows)


def recover_experiment_suite_summary(
    *,
    output_dir: str | Path,
    cases: Iterable[HolonicExperimentCase] | None = None,
    policies: Sequence[Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_case_ids = [case.case_id for case in cases] if cases is not None else None
    expected_policy_names = [str(policy.name) for policy in policies] if policies is not None else None

    if expected_case_ids is None:
        case_ids = sorted(
            child.name
            for child in output_dir.iterdir()
            if child.is_dir() and (child / 'summary.csv').exists()
        )
        if not case_ids:
            raise ValueError(f'no case summaries found under: {output_dir}')
    else:
        case_ids = list(expected_case_ids)
        missing_case_ids = [
            case_id for case_id in case_ids if not (output_dir / case_id / 'summary.csv').exists()
        ]
        if missing_case_ids:
            raise ValueError(
                'missing case summaries: ' + ', '.join(missing_case_ids)
            )

    recovered_rows: list[dict[str, str]] = []
    for case_id in case_ids:
        summary_path = output_dir / case_id / 'summary.csv'
        if not summary_path.exists():
            continue
        case_rows = _load_csv_rows(summary_path)
        if not case_rows:
            raise ValueError(f'empty case summary: {summary_path}')

        actual_case_ids = {row.get('case_id', '') for row in case_rows}
        if actual_case_ids != {case_id}:
            raise ValueError(
                f'case summary carries mismatched case ids: {summary_path}'
            )

        if expected_policy_names is not None:
            actual_policy_names = {str(row.get('policy_name', '')) for row in case_rows}
            if actual_policy_names != set(expected_policy_names):
                raise ValueError(
                    'unexpected policy coverage in case summary: '
                    f'{summary_path} -> expected {sorted(expected_policy_names)} got {sorted(actual_policy_names)}'
                )
            case_rows.sort(
                key=lambda row: expected_policy_names.index(str(row.get('policy_name', '')))
            )

        recovered_rows.extend(case_rows)

    _validate_unique_case_policy_rows(recovered_rows)
    return _write_csv_rows(output_dir / 'suite_summary.csv', recovered_rows)


def run_experiment_case(
    case: HolonicExperimentCase,
    policies: Sequence[Any],
    output_dir: Path,
) -> ExperimentCaseRunResult:
    output_dir = Path(output_dir)
    case_dir = output_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    case_budget_snapshot_json = case_dir / CASE_BUDGET_ARTIFACT_FILENAME
    case_budget_snapshot_json.write_text(
        json.dumps(_build_case_budget_snapshot(case), indent=2),
        encoding='utf-8',
    )

    rows: List[dict[str, Any]] = []
    artifact_rows: List[dict[str, Any]] = []
    for policy in policies:
        results = evaluate_holonic_policy(
            policy=policy,
            cell_configs=case.cell_configs,
            config=_case_config(case),
            seeds=case.seeds,
            benchmark_item_id=case.benchmark_item_id,
            env_cls=case.env_cls,
        )
        base = case_dir / policy.name
        results_json = base.with_suffix('.json')
        episode_csv = base.with_suffix('.csv')
        activation_trace_json = base.with_name(f'{policy.name}_activation_trace.json')
        settlement_trace_json = base.with_name(f'{policy.name}_settlement_trace.json')
        protocol_trace_json = base.with_name(f'{policy.name}_protocol_trace.json')
        world_event_trace_json = base.with_name(f'{policy.name}_world_event_trace.json')
        results.save_json(results_json)
        results.save_episode_csv(episode_csv)
        _write_trace_artifact(
            path=activation_trace_json,
            results=results,
            trace_key='activation_trace',
        )
        _write_trace_artifact(
            path=settlement_trace_json,
            results=results,
            trace_key='settlement_trace',
        )
        _write_trace_artifact(
            path=protocol_trace_json,
            results=results,
            trace_key='protocol_trace',
        )
        _write_trace_artifact(
            path=world_event_trace_json,
            results=results,
            trace_key='world_event_trace',
        )
        summary = results.summary()
        summary['case_id'] = case.case_id
        summary['level'] = case.level
        summary['suite_tier'] = case.suite_tier
        summary['env_name'] = case.env_name
        summary['regime_id'] = case.regime_id
        summary['topology_id'] = case.topology_id
        summary.update(_semantic_case_metadata(case))
        summary.update(_case_budget_metadata(case))
        rows.append(summary)
        artifact_rows.append(
            {
                'case_id': case.case_id,
                'policy_name': policy.name,
                'controller_interface_mode': str(summary.get('controller_interface_mode', 'in_process')),
                'framework_name': str(summary.get('framework_name', '')),
                'framework_backend_mode': str(summary.get('framework_backend_mode', '')),
                'framework_python_package': str(summary.get('framework_python_package', '')),
                'framework_optional_dependency': str(summary.get('framework_optional_dependency', '')),
                'framework_uv_project_environment': str(summary.get('framework_uv_project_environment', '')),
                'framework_llm_backend': str(summary.get('framework_llm_backend', '')),
                'framework_llm_model': str(summary.get('framework_llm_model', '')),
                'framework_llm_base_url': str(summary.get('framework_llm_base_url', '')),
                **_artifact_case_metadata(case),
                'numeric_profile_id': str(summary.get('numeric_profile_id', '')),
                'numeric_profile_provenance_class': str(summary.get('numeric_profile_provenance_class', '')),
                'numeric_profile_projection_kind': str(summary.get('numeric_profile_projection_kind', '')),
                'numeric_profile_scope': str(summary.get('numeric_profile_scope', '')),
                'numeric_profile_source_ref': str(summary.get('numeric_profile_source_ref', '')),
                'benchmark_item_id': case.benchmark_item_id,
                'env_name': case.env_name,
                'results_json': str(results_json),
                'episode_csv': str(episode_csv),
                'activation_trace_json': str(activation_trace_json),
                'settlement_trace_json': str(settlement_trace_json),
                'protocol_trace_json': str(protocol_trace_json),
                'world_event_trace_json': str(world_event_trace_json),
                'case_budget_snapshot_json': str(case_budget_snapshot_json),
            }
        )

    summary_csv = case_dir / 'summary.csv'
    _write_csv_rows(summary_csv, rows)

    artifact_manifest = case_dir / 'artifacts.csv'
    _write_csv_rows(artifact_manifest, artifact_rows)

    return ExperimentCaseRunResult(
        case_id=case.case_id,
        summary_csv=summary_csv,
        artifact_manifest=artifact_manifest,
        rows=rows,
    )


def _write_trace_artifact(
    *,
    path: Path,
    results: Any,
    trace_key: str,
) -> None:
    payload = {
        'benchmark_item_id': results.benchmark_item_id,
        'policy_name': results.policy_name,
        'trace_key': trace_key,
        'episodes': [
            {
                'episode_id': episode.episode_id,
                'seed': episode.seed,
                'trace': episode.extra.get(trace_key, []),
            }
            for episode in results.episodes
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _semantic_case_metadata(case: HolonicExperimentCase) -> dict[str, Any]:
    return {field: getattr(case, field) for field in SEMANTIC_CASE_METADATA_FIELDS}


def _case_budget_metadata(case: HolonicExperimentCase) -> dict[str, str]:
    return {field: getattr(case, field) for field in CASE_BUDGET_METADATA_FIELDS}


def _artifact_case_metadata(case: HolonicExperimentCase) -> dict[str, str]:
    semantic_metadata = {
        key: str(value or '') if value is not None else ''
        for key, value in _semantic_case_metadata(case).items()
    }
    semantic_metadata['semantic_setup_enabled'] = str(case.semantic_setup_enabled)
    semantic_metadata['semantic_buffer_wip_enabled'] = str(case.semantic_buffer_wip_enabled)
    semantic_metadata['semantic_transport_enabled'] = str(case.semantic_transport_enabled)
    return semantic_metadata | _case_budget_metadata(case)


def _build_case_budget_snapshot(case: HolonicExperimentCase) -> dict[str, Any]:
    return {
        'case_id': case.case_id,
        'benchmark_item_id': case.benchmark_item_id,
        'suite_tier': case.suite_tier,
        'regime_id': case.regime_id,
        'topology_id': case.topology_id,
        **_semantic_case_metadata(case),
        **_case_budget_metadata(case),
        'cells': [
            {
                'cell_id': cell.cell_id,
                'label': cell.label,
                'family_key': cell.family_key,
                'instance_kind': cell.instance_kind,
                'wip_cap': cell.wip_cap,
                'energy_cap_kwh': cell.energy_cap_kwh,
                'carbon_cap_kg': cell.carbon_cap_kg,
                'processing_time_scale': cell.processing_time_scale,
            }
            for cell in case.cell_configs
        ],
    }


def run_experiment_suite(
    cases: Iterable[HolonicExperimentCase],
    policies: Sequence[Any],
    output_dir: Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict[str, Any]] = []
    summary_path = output_dir / 'suite_summary.csv'
    for case in cases:
        result = run_experiment_case(case=case, policies=policies, output_dir=output_dir)
        all_rows.extend(result.rows)
        _write_csv_rows(summary_path, all_rows)
    if not all_rows:
        _write_csv_rows(summary_path, all_rows)
    return summary_path
