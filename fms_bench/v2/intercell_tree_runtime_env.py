from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

from fms_bench.config import BenchmarkConfig
from fms_bench.holonic_types import PlantCellConfig
from fms_bench.tree_intercell_env import TreeIntercellEnv

from .area_runtime import AreaRuntime
from .authority_modes import get_authority_mode
from .cell_runtime import CellRuntime
from .event_interpreter import DecisionEpoch
from .runtime_ledger import RuntimeLedger
from .tree_runtime_env import TreeRuntimeEnv


class IntercellTreeRuntimeEnv(TreeRuntimeEnv):
    """Interpreter-driven runtime over the first minimal inter-cell env."""

    def __init__(
        self,
        cell_configs: Iterable[PlantCellConfig],
        area_to_cells: Mapping[int, Sequence[int]],
        config: Optional[BenchmarkConfig] = None,
        seed: Optional[int] = None,
        authority_mode: str = "holonic_hybrid",
        max_in_flight_jobs: int = 1,
        route_grammar: Optional[Sequence[Sequence[int]]] = None,
        backlog_top_k: int = 1,
        route_grammar_contract: str | None = None,
    ) -> None:
        self._env = TreeIntercellEnv(
            cell_configs=cell_configs,
            area_to_cells=area_to_cells,
            config=config,
            seed=seed,
            max_in_flight_jobs=max_in_flight_jobs,
            route_grammar=route_grammar,
            backlog_top_k=backlog_top_k,
            route_grammar_contract=route_grammar_contract,
        )
        self.authority_mode = get_authority_mode(authority_mode)
        self._area_runtimes = {
            f"area_{area_id}": AreaRuntime(
                area_id=f"area_{area_id}",
                child_ids=tuple(
                    f"cell_{cell_id}" for cell_id in self._env.area_to_cells[area_id]
                ),
                authority_mode=self.authority_mode.mode_id,
            )
            for area_id in self._env.area_ids
        }
        self._cell_runtimes = {
            f"cell_{cell_id}": CellRuntime(
                cell_id=f"cell_{cell_id}",
                authority_mode=self.authority_mode.mode_id,
            )
            for cell_id in sorted(self._env.assembly.cells)
        }
        self.current_decision_epoch = DecisionEpoch(
            activation_timestamp=0.0,
            records_by_agent={},
        )
        self.runtime_ledger = RuntimeLedger()
        self._pending_plant_action: int | None = None
        self._pending_area_id: int | None = None
        self._pending_cell_id: int | None = None
        self._pending_runtime_contract_id: str | None = None
