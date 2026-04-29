from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from fms_bench.config import BenchmarkConfig
from fms_bench.holonic_types import HolonicEvent, PlantCellConfig, TreeMessage
from fms_bench.plant_env import PlantStepResult
from fms_bench.tree_shared_backlog_env import TreeSharedBacklogEnv

from .area_runtime import AreaRuntime
from .authority_modes import AuthorityMode, get_authority_mode
from .cell_runtime import CellRuntime
from .event_interpreter import DecisionEpoch, interpret_decision_epoch
from .runtime_ledger import REWARD_AUTHORITY_ID, RuntimeLedger
from .task_semantics import PropagationAction
from .types import ActivationRecord


class TreeRuntimeEnv:
    """Interpreter-driven v2 runtime boundary over one shared DES world."""

    _MAX_PROTOCOL_LIVELOCK_STEPS = 64
    _SUPPORTED_EXECUTION_MODES = (
        "single_agent",
        "same_timestamp_multi_cell_local_regulation",
    )
    RUNTIME_LAYER_CONTRACT = (
        "manufacturing_operations_and_distributed_production_control_benchmark_runtime"
    )
    INDUSTRIAL_RUNTIME_SCOPE_CONTRACT = RUNTIME_LAYER_CONTRACT
    SHARED_DES_RUNTIME_BOUNDARY_CONTRACT = "one_shared_des_world_tree_runtime_boundary"
    SHARED_RUNTIME_BOUNDARY_CONTRACT = SHARED_DES_RUNTIME_BOUNDARY_CONTRACT
    RUNTIME_AGENT_BOUNDARY_CONTRACT = "plant_area_cell_are_public_runtime_agents"
    PUBLIC_EPOCH_SEMANTICS_CONTRACT = (
        "serialized_public_epochs_plus_limited_same_timestamp_multi_cell_local_regulation"
    )
    PUBLIC_EPOCH_SERIALIZATION_CONTRACT = PUBLIC_EPOCH_SEMANTICS_CONTRACT
    PUBLIC_ROLE_MAPPING_CONTRACT = "benchmark_facing_plant_area_cell_role_mapping"
    LITERATURE_MAPPING_BOUNDARY_CONTRACT = (
        "isa95_level3_prosa_adacor_inspired_not_direct_implementation"
    )
    RUNTIME_STATE_KIND_CONTRACT = (
        "decision_active_release_blocked_local_dispatch_active_deadlock"
    )
    _BENCHMARK_ROLE_MAPPING = {
        "plant": "operations_management_release_and_escalation_role",
        "area": "supervisor_or_sub_holon_coordination_role",
        "cell": "operational_unit_commitment_and_local_dispatch_role",
    }

    def __init__(
        self,
        cell_configs: Iterable[PlantCellConfig],
        area_to_cells: Mapping[int, Sequence[int]],
        config: Optional[BenchmarkConfig] = None,
        seed: Optional[int] = None,
        authority_mode: str = "holonic_hybrid",
    ) -> None:
        self.authority_mode: AuthorityMode = get_authority_mode(authority_mode)
        self._env = TreeSharedBacklogEnv(
            cell_configs=cell_configs,
            area_to_cells=area_to_cells,
            config=config,
            seed=seed,
        )
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
        self._pending_backlog_id: int | None = None
        self._pending_plant_action: int | None = None
        self._pending_area_id: int | None = None
        self._pending_cell_id: int | None = None
        self._pending_runtime_contract_id: str | None = None
        self._protocol_livelock_streak = 0

    @property
    def simulator(self) -> Any:
        return self._env.simulator

    @property
    def observation_space(self) -> Any:
        return self._env.observation_space

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self._env.reset(seed=seed, options=options)
        self.runtime_ledger = RuntimeLedger()
        self._clear_pending_protocol_state()
        self._reset_protocol_livelock_guard()
        self.current_decision_epoch = self._build_current_epoch()
        augmented_info = self._augment_info(
            info,
            activation_trace=self._serialize_records(
                self.current_decision_epoch.records_by_agent.values()
            ),
            protocol_trace=[],
            settlement_trace=[],
            reward_components={
                "delta_time": 0.0,
                "delta_tardiness": 0.0,
                "energy_penalty": 0.0,
                "carbon_penalty": 0.0,
                "invalid_action_penalty": 0.0,
                "deadlock_penalty": 0.0,
                "terminal_bonus": 0.0,
                "total": 0.0,
            },
        )
        return observation, augmented_info

    def observe(self, agent: str = "plant") -> np.ndarray:
        return self._env.observe(agent)

    def action_mask(self, agent: str = "plant") -> np.ndarray:
        record = self.current_decision_epoch.records_by_agent.get(agent)
        if (
            record is not None
        ):
            if agent == "plant" and record.functional_loop == "accountability_and_escalation":
                return self._plant_accountability_action_mask(record)
            if agent == "plant" and self._supports_backlog_selection():
                plant_decision_kind = self._plant_decision_kind(record)
                if plant_decision_kind == "backlog_selection":
                    return self._plant_backlog_action_mask(record)
                if plant_decision_kind == "cell_selection":
                    return self._plant_cell_selection_action_mask(record)
                if plant_decision_kind == "area_selection":
                    return self._plant_area_selection_action_mask(record)
            if agent.startswith("area_") and record.functional_loop == "accountability_and_escalation":
                return np.ones(1, dtype=np.int8)
            if agent.startswith("area_") and record.functional_loop == "commitment_management":
                area_id = int(agent.split("_", 1)[1])
                mask = np.zeros(len(self._env.assembly.cells), dtype=np.int8)
                for cell_id in self.controller_valid_cell_ids(area_id):
                    mask[int(cell_id)] = 1
                return mask
            if record.functional_loop == "commitment_management":
                if agent.startswith("cell_"):
                    legal_actions = self._cell_runtimes[agent].legal_actions(record)
                    return np.ones(len(legal_actions), dtype=np.int8)
            if record.functional_loop == "local_regulation" and agent.startswith("cell_"):
                dispatch_options = tuple(record.context_handles.get("dispatch_options", ()))
                return np.ones(len(dispatch_options), dtype=np.int8)
        return self._env.action_mask(agent)

    def controller_valid_area_ids(
        self,
        record: ActivationRecord | None = None,
    ) -> list[int]:
        if record is None:
            current = self.current_decision_epoch.records_by_agent.get("plant")
            if current is not None and current.agent_id == "plant":
                record = current
        if record is not None and record.functional_loop == "accountability_and_escalation":
            return self._valid_plant_accountability_area_ids(record)
        if self._supports_backlog_selection() and self._pending_backlog_id is not None:
            return [
                int(area_id)
                for area_id in self._env._valid_release_areas_for_job(self._pending_backlog_id)
            ]
        return [int(area_id) for area_id in self._env._valid_release_areas()]

    def controller_valid_cell_ids(self, area_id: int) -> list[int]:
        if self._supports_backlog_selection() and self._pending_backlog_id is not None:
            valid_cells = set(self._env._valid_release_cells_for_job(self._pending_backlog_id))
        else:
            valid_cells = set(self._env._valid_release_cells())
        return [
            int(cell_id)
            for cell_id in self._env.area_to_cells[int(area_id)]
            if int(cell_id) in valid_cells
        ]

    def controller_plant_state(self) -> dict[str, Any]:
        completed_assigned = len(
            self._env.simulator.completed_jobs.intersection(self._env.assigned_clone_jobs)
        )
        return {
            "released_jobs_total": len(self._env.assigned_clone_jobs),
            "completed_jobs_total": completed_assigned,
            "backlog_jobs_total": len(self._env.shared_backlog),
            "visible_backlog_jobs_total": len(self.controller_visible_backlog_candidates()),
            "valid_backlog_jobs_total": len(self.controller_valid_backlog_ids()),
            "dormant_jobs_total": len(self._env.simulator.jobs) - len(self._env.assigned_clone_jobs),
            "busy_machines_total": sum(
                1
                for machine in self._env.simulator.machines.values()
                if machine.status in {"busy", "setup"}
            ),
            "setup_machines_total": sum(
                1
                for machine in self._env.simulator.machines.values()
                if machine.status == "setup"
            ),
            "down_machines_total": sum(
                1
                for machine in self._env.simulator.machines.values()
                if machine.status == "down"
            ),
        }

    def controller_area_summary(self, area_id: int) -> dict[str, Any]:
        return dict(self._env._area_feedback(int(area_id)).to_dict())

    def controller_cell_summary(self, cell_id: int) -> dict[str, Any]:
        return dict(self._env._cell_feedback(int(cell_id)).to_dict())

    def controller_machine_summaries(
        self,
        *,
        cell_id: int | None = None,
    ) -> dict[int, dict[str, Any]]:
        summaries = {
            int(machine_id): dict(machine_summary)
            for machine_id, machine_summary in self._env._machine_summaries().items()
        }
        if cell_id is None:
            return summaries
        return {
            machine_id: machine_summary
            for machine_id, machine_summary in summaries.items()
            if int(machine_summary.get("cell_id", -1)) == int(cell_id)
        }

    def controller_visible_backlog_candidates(self) -> list[dict[str, Any]]:
        candidate_builder = getattr(self._env, "visible_backlog_candidates", None)
        if not callable(candidate_builder):
            return []
        return [dict(candidate) for candidate in candidate_builder()]

    def controller_valid_backlog_ids(self) -> list[int]:
        valid_builder = getattr(self._env, "valid_backlog_ids", None)
        if callable(valid_builder):
            return [int(job_id) for job_id in valid_builder()]
        candidates = self.controller_visible_backlog_candidates()
        return [
            int(candidate["job_id"])
            for candidate in candidates
            if list(candidate.get("valid_areas", ()))
        ]

    def _supports_backlog_selection(self) -> bool:
        return bool(getattr(self._env, "backlog_top_k", 1)) and int(
            getattr(self._env, "backlog_top_k", 1)
        ) > 1

    def _plant_decision_kind(self, record: ActivationRecord | None = None) -> str | None:
        current = record
        if current is None:
            current = self.current_decision_epoch.records_by_agent.get("plant")
        if current is None:
            return None
        if current.decision_kind is not None:
            return str(current.decision_kind)
        kind = dict(current.context_handles).get("plant_decision_kind")
        return None if kind is None else str(kind)

    def _selected_backlog_context(self, backlog_id: int | None = None) -> dict[str, Any]:
        selected_id = self._pending_backlog_id if backlog_id is None else int(backlog_id)
        if selected_id is None:
            return {
                "selected_backlog_id": None,
                "selected_backlog_stage_id": None,
                "selected_backlog_valid_areas": [],
                "selected_backlog_valid_cells_by_area": {},
            }
        visible_candidates = self.controller_visible_backlog_candidates()
        selected = next(
            (
                candidate
                for candidate in visible_candidates
                if int(candidate.get("job_id", -1)) == int(selected_id)
            ),
            None,
        )
        if selected is None:
            valid_areas = []
            valid_cells_by_area: dict[int, list[int]] = {}
        else:
            valid_areas = [int(area_id) for area_id in selected.get("valid_areas", ())]
            valid_cells_by_area = {
                int(area_id): [int(cell_id) for cell_id in cell_ids]
                for area_id, cell_ids in dict(selected.get("valid_cells_by_area", {})).items()
            }
        stage_id = None
        if selected_id in getattr(self._env.simulator, "jobs", {}):
            stage_id = int(self._env.simulator.jobs[selected_id].current_op_idx)
        return {
            "selected_backlog_id": None if selected_id is None else int(selected_id),
            "selected_backlog_stage_id": stage_id,
            "selected_backlog_valid_areas": valid_areas,
            "selected_backlog_valid_cells_by_area": valid_cells_by_area,
        }

    def _plant_backlog_action_mask(self, record: ActivationRecord) -> np.ndarray:
        visible_candidates = list(record.context_handles.get("visible_backlog_candidates", ()))
        mask = np.zeros(max(1, len(visible_candidates)), dtype=np.int8)
        valid_ids = {int(job_id) for job_id in record.context_handles.get("valid_backlog_ids", ())}
        for index, candidate in enumerate(visible_candidates):
            if int(candidate.get("job_id", -1)) in valid_ids:
                mask[index] = 1
        return mask

    def _plant_area_selection_action_mask(self, record: ActivationRecord) -> np.ndarray:
        valid_area_ids = [
            int(area_id)
            for area_id in record.context_handles.get(
                "selected_backlog_valid_areas",
                self.controller_valid_area_ids(record),
            )
        ]
        mask = np.zeros(max(1, len(self._env.area_ids)), dtype=np.int8)
        for area_id in valid_area_ids:
            mask[self._env.area_index[int(area_id)]] = 1
        return mask

    def _plant_cell_selection_action_mask(self, record: ActivationRecord) -> np.ndarray:
        valid_cell_ids = [
            int(cell_id)
            for cell_ids in dict(
                record.context_handles.get("selected_backlog_valid_cells_by_area", {})
            ).values()
            for cell_id in cell_ids
        ]
        mask = np.zeros(max(1, len(self._env.assembly.cells)), dtype=np.int8)
        for cell_id in valid_cell_ids:
            mask[int(cell_id)] = 1
        return mask

    def _resolve_backlog_action(self, record: ActivationRecord, action: int) -> tuple[int, int]:
        visible_candidates = list(record.context_handles.get("visible_backlog_candidates", ()))
        valid_ids = [int(job_id) for job_id in record.context_handles.get("valid_backlog_ids", ())]
        if not visible_candidates or not valid_ids:
            raise RuntimeError("Plant has no valid backlog candidate at the current decision epoch")
        valid_indices = [
            index
            for index, candidate in enumerate(visible_candidates)
            if int(candidate.get("job_id", -1)) in valid_ids
        ]
        if int(action) in valid_indices:
            selected_index = int(action)
        else:
            selected_index = int(valid_indices[0])
        selected_job_id = int(visible_candidates[selected_index]["job_id"])
        return selected_job_id, selected_index

    def agents_to_act(self) -> list[str]:
        return list(self.current_decision_epoch.agents_to_act)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        step = self.step_dict({"plant": int(action)})
        return (
            step.observation,
            step.reward,
            step.terminated,
            step.truncated,
            step.info,
        )

    def step_dict(self, action_dict: Mapping[str, int]) -> PlantStepResult:
        current_agents = self.agents_to_act()
        if not current_agents:
            raise RuntimeError("No controller action is expected at the current decision epoch")

        if set(action_dict) != set(current_agents):
            raise KeyError(
                f"action_dict keys must match current decision epoch agents: {current_agents}"
            )

        world_state_before = self._env.render()
        if len(current_agents) != 1:
            execution_mode = self.current_execution_mode(current_agents)
            if execution_mode == "same_timestamp_multi_cell_local_regulation":
                result = self._step_multi_cell_local_regulation(action_dict)
                return self._apply_protocol_livelock_guard(
                    world_state_before=world_state_before,
                    result=result,
                )
            raise NotImplementedError(
                "TreeRuntimeEnv does not support the current multi-agent execution mode: "
                f"{execution_mode}"
            )

        current_agent = current_agents[0]
        action_value = int(action_dict[current_agent])
        if current_agent == "plant":
            record = self.current_decision_epoch.records_by_agent[current_agent]
            if record.functional_loop == "accountability_and_escalation":
                result = self._step_plant_accountability(record, action_value)
            else:
                result = self._step_plant(action_value)
            return self._apply_protocol_livelock_guard(
                world_state_before=world_state_before,
                result=result,
            )
        if current_agent.startswith("area_"):
            result = self._step_area(current_agent, action_value)
            return self._apply_protocol_livelock_guard(
                world_state_before=world_state_before,
                result=result,
            )
        if current_agent.startswith("cell_"):
            result = self._step_cell(current_agent, action_value)
            return self._apply_protocol_livelock_guard(
                world_state_before=world_state_before,
                result=result,
            )
        raise ValueError(f"Unknown runtime agent: {current_agent}")

    def advance_idle_epoch(self) -> PlantStepResult:
        if self.agents_to_act():
            raise RuntimeError("Controller action is required at the current decision epoch")

        tardiness_before = self._compute_total_tardiness()
        transition = self._advance_world_until_decision(stop_on_dispatch=True)
        self._env._last_transition = transition

        terminated = self._env._is_done()
        truncated = False
        if not terminated:
            if self._env.simulator.time >= float(self._env.config.horizon_hours):
                truncated = True
            if bool(transition["deadlocked"]):
                truncated = True
            if self._should_truncate_for_no_progress(transition):
                truncated = True
        self._env._terminated = terminated
        self._env._truncated = truncated

        info = self._env.render()
        self._sync_runtime_contract_from_world(info)
        tardiness_after = self._compute_total_tardiness()
        reward_breakdown = self.runtime_ledger.finalize_interval(
            start_time=float(transition.get("time_start", info.get("time", 0.0))),
            end_time=float(transition.get("time_end", info.get("time", 0.0))),
            energy_kwh=float(info.get("energy_kwh_step", 0.0)),
            carbon_kg=float(info.get("carbon_kg_step", 0.0)),
            invalid_action=False,
            terminated=terminated,
            deadlocked=bool(info.get("deadlocked", False)),
            tardiness_before=tardiness_before,
            tardiness_after=tardiness_after,
            energy_overshoot_kwh=self._aggregate_energy_overshoot(info),
            carbon_overshoot_kg=self._aggregate_carbon_overshoot(info),
        )
        self.current_decision_epoch = self._build_current_epoch()
        self._clear_pending_protocol_state()

        augmented_info = self._augment_info(
            info,
            activation_trace=self._serialize_records(
                self.current_decision_epoch.records_by_agent.values()
            ),
            protocol_trace=[],
            world_event_trace=list(info.get("typed_events", [])),
            settlement_trace=self.runtime_ledger.settlement_trace(),
            reward_components=reward_breakdown.to_dict(),
        )
        next_observation_agent = self.agents_to_act()[0] if self.agents_to_act() else "plant"
        return PlantStepResult(
            observation=self._env.observe(next_observation_agent),
            reward=reward_breakdown.total,
            terminated=terminated,
            truncated=truncated,
            info=augmented_info,
        )

    def render(self) -> dict[str, Any]:
        return self._augment_info(
            self._env.render(),
            activation_trace=self._serialize_records(
                self.current_decision_epoch.records_by_agent.values()
            ),
        )

    def close(self) -> None:
        self._env.close()

    def _step_plant(self, action: int) -> PlantStepResult:
        record = self.current_decision_epoch.records_by_agent.get("plant")
        if self._supports_backlog_selection() and record is not None:
            plant_decision_kind = self._plant_decision_kind(record)
            if plant_decision_kind == "backlog_selection":
                return self._step_plant_backlog_selection(record, int(action))
            if plant_decision_kind == "cell_selection":
                return self._step_plant_cell_selection(record, int(action))
        selected_area_id, staged_plant_action = self._resolve_plant_action(action)
        if self.authority_mode.routing_authority_mode == "centralized":
            message = self._make_assignment_request(selected_area_id)
            return self._step_plant_direct_award(
                selected_area_id=selected_area_id,
                staged_plant_action=staged_plant_action,
                message=message,
            )
        message = self._make_assignment_request(selected_area_id)

        self._pending_plant_action = staged_plant_action
        self._pending_area_id = selected_area_id
        self._pending_cell_id = None
        self._pending_runtime_contract_id = None
        self.current_decision_epoch = self._annotate_epoch(
            interpret_decision_epoch(
                world_time=float(self._env.simulator.time),
                events=(),
                messages=(message,),
            )
        )

        info = self._augment_info(
            self._env.render(),
            activation_trace=self._serialize_records(
                self.current_decision_epoch.records_by_agent.values()
            ),
            protocol_trace=[message.to_dict()],
            world_event_trace=[],
            settlement_trace=self.runtime_ledger.settlement_trace(),
            reward_components={
                "delta_time": 0.0,
                "delta_tardiness": 0.0,
                "energy_penalty": 0.0,
                "carbon_penalty": 0.0,
                "invalid_action_penalty": 0.0,
                "deadlock_penalty": 0.0,
                "terminal_bonus": 0.0,
                "total": 0.0,
            },
        )
        info["selected_area_id"] = selected_area_id
        return PlantStepResult(
            observation=self._env.observe("plant"),
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
        )

    def _step_plant_backlog_selection(
        self,
        record: ActivationRecord,
        action: int,
    ) -> PlantStepResult:
        selected_backlog_id, staged_plant_action = self._resolve_backlog_action(record, action)
        self._pending_backlog_id = selected_backlog_id
        self._pending_plant_action = staged_plant_action
        self._pending_area_id = None
        self._pending_cell_id = None
        self._pending_runtime_contract_id = None
        if self.authority_mode.routing_authority_mode == "centralized":
            self.current_decision_epoch = self._build_plant_cell_selection_epoch(
                selected_backlog_id=selected_backlog_id
            )
        else:
            self.current_decision_epoch = self._build_plant_area_selection_epoch(
                selected_backlog_id=selected_backlog_id
            )
        info = self._augment_info(
            self._env.render(),
            activation_trace=self._serialize_records(
                self.current_decision_epoch.records_by_agent.values()
            ),
            protocol_trace=[],
            world_event_trace=[],
            settlement_trace=self.runtime_ledger.settlement_trace(),
            reward_components={
                "delta_time": 0.0,
                "delta_tardiness": 0.0,
                "energy_penalty": 0.0,
                "carbon_penalty": 0.0,
                "invalid_action_penalty": 0.0,
                "deadlock_penalty": 0.0,
                "terminal_bonus": 0.0,
                "total": 0.0,
            },
        )
        info.update(self._selected_backlog_context(selected_backlog_id))
        info["plant_decision_kind"] = (
            "cell_selection"
            if self.authority_mode.routing_authority_mode == "centralized"
            else "area_selection"
        )
        return PlantStepResult(
            observation=self._env.observe("plant"),
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
        )

    def _step_plant_cell_selection(
        self,
        record: ActivationRecord,
        action: int,
    ) -> PlantStepResult:
        valid_cell_ids = [
            int(cell_id)
            for cell_ids in dict(
                record.context_handles.get("selected_backlog_valid_cells_by_area", {})
            ).values()
            for cell_id in cell_ids
        ]
        if not valid_cell_ids:
            raise RuntimeError("Plant has no valid execution cell at the current decision epoch")
        selected_cell_id = int(action) if int(action) in valid_cell_ids else int(valid_cell_ids[0])
        selected_area_id = int(self._env.cell_to_area[selected_cell_id])
        return self._step_plant_direct_award(
            selected_area_id=selected_area_id,
            staged_plant_action=selected_cell_id,
            message=self._make_assignment_request(selected_area_id),
            selected_cell_id=selected_cell_id,
        )

    def _step_plant_accountability(
        self,
        record: ActivationRecord,
        action: int,
    ) -> PlantStepResult:
        valid_area_ids = self._valid_plant_accountability_area_ids(record)
        if not valid_area_ids:
            self._clear_pending_protocol_state()
            self.current_decision_epoch = self._build_current_epoch()
            info = self._augment_info(
                self._env.render(),
                activation_trace=self._serialize_records(
                    self.current_decision_epoch.records_by_agent.values()
                ),
                protocol_trace=[],
                world_event_trace=[],
                settlement_trace=self.runtime_ledger.settlement_trace(),
                reward_components={
                    "delta_time": 0.0,
                    "delta_tardiness": 0.0,
                    "energy_penalty": 0.0,
                    "carbon_penalty": 0.0,
                    "invalid_action_penalty": 0.0,
                    "deadlock_penalty": 0.0,
                    "terminal_bonus": 0.0,
                    "total": 0.0,
                },
            )
            info["accountability_reroute_exhausted"] = True
            return PlantStepResult(
                observation=self._env.observe("plant"),
                reward=0.0,
                terminated=False,
                truncated=False,
                info=info,
            )
        selected_area_id, staged_plant_action = self._resolve_plant_action(
            action,
            valid_area_ids=valid_area_ids,
        )
        message = self._make_revise_contract_request(selected_area_id, record)
        if self.authority_mode.routing_authority_mode == "centralized":
            return self._step_plant_direct_award(
                selected_area_id=selected_area_id,
                staged_plant_action=staged_plant_action,
                message=message,
            )

        self._pending_plant_action = staged_plant_action
        self._pending_area_id = selected_area_id
        self._pending_cell_id = None
        self._pending_runtime_contract_id = None
        self.current_decision_epoch = self._annotate_epoch(
            interpret_decision_epoch(
                world_time=float(self._env.simulator.time),
                events=(),
                messages=(message,),
            )
        )

        info = self._augment_info(
            self._env.render(),
            activation_trace=self._serialize_records(
                self.current_decision_epoch.records_by_agent.values()
            ),
            protocol_trace=[message.to_dict()],
            world_event_trace=[],
            settlement_trace=self.runtime_ledger.settlement_trace(),
            reward_components={
                "delta_time": 0.0,
                "delta_tardiness": 0.0,
                "energy_penalty": 0.0,
                "carbon_penalty": 0.0,
                "invalid_action_penalty": 0.0,
                "deadlock_penalty": 0.0,
                "terminal_bonus": 0.0,
                "total": 0.0,
            },
        )
        info["selected_area_id"] = selected_area_id
        return PlantStepResult(
            observation=self._env.observe("plant"),
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
        )

    def _step_plant_direct_award(
        self,
        *,
        selected_area_id: int,
        staged_plant_action: int,
        message: TreeMessage,
        selected_cell_id: int | None = None,
    ) -> PlantStepResult:
        self._pending_plant_action = staged_plant_action
        self._pending_area_id = selected_area_id
        self._pending_cell_id = None
        self._pending_runtime_contract_id = None

        area_agent = f"area_{selected_area_id}"
        area_epoch = self._annotate_epoch(
            interpret_decision_epoch(
                world_time=float(self._env.simulator.time),
                events=(),
                messages=(message,),
            )
        )
        area_record = area_epoch.records_by_agent[area_agent]
        if selected_cell_id is None:
            selected_child_id, selected_cell_id = self._resolve_area_action(area_agent, -1)
        else:
            selected_child_id = f"cell_{int(selected_cell_id)}"
        area_runtime = self._area_runtimes[area_agent]
        area_decision = area_runtime.handle_activation(
            self._with_preferred_child(area_record, selected_child_id)
        )
        if area_decision.issued_contract is not None:
            self.runtime_ledger.apply_protocol_object(
                area_decision.issued_contract,
                timestamp=float(self._env.simulator.time),
            )
            self._pending_runtime_contract_id = area_decision.issued_contract.contract_id
        self._pending_cell_id = selected_cell_id

        cell_epoch = self._annotate_epoch(
            interpret_decision_epoch(
                world_time=float(self._env.simulator.time),
                events=(),
                messages=area_decision.emitted_messages,
            )
        )
        cell_runtime = self._cell_runtimes[selected_child_id]
        cell_record = cell_epoch.records_by_agent[selected_child_id]
        accept_action = cell_runtime.legal_actions(cell_record)[0]
        cell_decision = cell_runtime.handle_activation(cell_record, action=accept_action)
        protocol_messages = tuple(cell_decision.emitted_messages)
        for protocol_message in protocol_messages:
            self.runtime_ledger.apply_protocol_object(
                protocol_message,
                timestamp=float(self._env.simulator.time),
            )

        contract = self._env._release_next_job(
            selected_cell_id,
            job_id=self._pending_backlog_id,
        )
        self._env._last_contract = contract
        self._pending_backlog_id = None
        self._env._last_transition = {
            "time_start": float(self._env.simulator.time),
            "time_end": float(self._env.simulator.time),
            "events": [],
            "event_records": [],
            "deadlocked": False,
            "energy_kwh_step": 0.0,
            "energy_cost_step": 0.0,
            "carbon_kg_step": 0.0,
        }
        info = self._env.render()
        self._sync_runtime_contract_from_world(info)
        self.current_decision_epoch = self._build_local_dispatch_epoch()
        if not self.current_decision_epoch.records_by_agent:
            if (
                self.authority_mode.routing_authority_mode == "centralized"
                and self._supports_backlog_selection()
                and hasattr(self._env, "_valid_release_cells")
            ):
                original_valid_release_cells = self._env._valid_release_cells
                try:
                    self._env._valid_release_cells = lambda: []
                    transition = self._advance_world_until_decision(stop_on_dispatch=True)
                    self._env._last_transition = transition
                finally:
                    self._env._valid_release_cells = original_valid_release_cells
                info = self._env.render()
                self._sync_runtime_contract_from_world(info)
                self.current_decision_epoch = self._build_local_dispatch_epoch()
            if not self.current_decision_epoch.records_by_agent:
                self.current_decision_epoch = self._build_current_epoch()

        protocol_trace = [message.to_dict()]
        protocol_trace.extend(item.to_dict() for item in area_decision.emitted_messages)
        protocol_trace.extend(item.to_dict() for item in protocol_messages)
        augmented_info = self._augment_info(
            info,
            activation_trace=self._serialize_records(
                self.current_decision_epoch.records_by_agent.values()
            ),
            protocol_trace=protocol_trace,
            world_event_trace=[],
            settlement_trace=self.runtime_ledger.settlement_trace(),
            reward_components={
                "delta_time": 0.0,
                "delta_tardiness": 0.0,
                "energy_penalty": 0.0,
                "carbon_penalty": 0.0,
                "invalid_action_penalty": 0.0,
                "deadlock_penalty": 0.0,
                "terminal_bonus": 0.0,
                "total": 0.0,
            },
        )
        augmented_info["selected_area_id"] = int(selected_area_id)
        augmented_info["selected_cell_id"] = int(selected_cell_id)
        next_observation_agent = self.agents_to_act()[0] if self.agents_to_act() else "plant"
        return PlantStepResult(
            observation=self._env.observe(next_observation_agent),
            reward=0.0,
            terminated=bool(self._env._terminated),
            truncated=bool(self._env._truncated),
            info=augmented_info,
        )

    def _step_area(self, area_agent: str, action: int) -> PlantStepResult:
        record = self.current_decision_epoch.records_by_agent[area_agent]
        if record.functional_loop == "accountability_and_escalation":
            return self._step_area_accountability(area_agent, record, int(action))
        if self._pending_area_id is None or area_agent != f"area_{self._pending_area_id}":
            raise RuntimeError("Area activation has no matching pending plant request")

        selected_child_id, selected_cell_id = self._resolve_area_action(area_agent, action)
        runtime = self._area_runtimes[area_agent]
        decision = runtime.handle_activation(
            self._with_preferred_child(record, selected_child_id)
        )
        if decision.issued_contract is not None:
            self.runtime_ledger.apply_protocol_object(
                decision.issued_contract,
                timestamp=float(self._env.simulator.time),
            )
            self._pending_runtime_contract_id = decision.issued_contract.contract_id
        self._pending_cell_id = selected_cell_id
        self.current_decision_epoch = self._annotate_epoch(
            interpret_decision_epoch(
                world_time=float(self._env.simulator.time),
                events=(),
                messages=decision.emitted_messages,
            )
        )

        protocol_trace = [message.to_dict() for message in decision.emitted_messages]
        if self.authority_mode.mediated_bid_round_enabled:
            contract_id = (
                decision.issued_contract.contract_id
                if decision.issued_contract is not None
                else f"{area_agent}::pending::{selected_child_id}"
            )
            protocol_trace = [
                TreeMessage(
                    sender=area_agent,
                    recipient=selected_child_id,
                    kind="cfp",
                    payload={
                        "contract_id": contract_id,
                        "selected_cell_id": int(selected_cell_id),
                    },
                ).to_dict(),
                TreeMessage(
                    sender=selected_child_id,
                    recipient=area_agent,
                    kind="bid",
                    payload={
                        "contract_id": contract_id,
                        "selected_cell_id": int(selected_cell_id),
                    },
                ).to_dict(),
                TreeMessage(
                    sender=area_agent,
                    recipient=selected_child_id,
                    kind="award",
                    payload={
                        "contract_id": contract_id,
                        "selected_cell_id": int(selected_cell_id),
                    },
                ).to_dict(),
                *protocol_trace,
            ]

        info = self._augment_info(
            self._env.render(),
            activation_trace=self._serialize_records(
                self.current_decision_epoch.records_by_agent.values()
            ),
            protocol_trace=protocol_trace,
            world_event_trace=[],
            settlement_trace=self.runtime_ledger.settlement_trace(),
            reward_components={
                "delta_time": 0.0,
                "delta_tardiness": 0.0,
                "energy_penalty": 0.0,
                "carbon_penalty": 0.0,
                "invalid_action_penalty": 0.0,
                "deadlock_penalty": 0.0,
                "terminal_bonus": 0.0,
                "total": 0.0,
            },
        )
        info["selected_area_id"] = self._pending_area_id
        info["selected_cell_id"] = selected_cell_id
        return PlantStepResult(
            observation=self._env.observe(area_agent),
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
        )

    def _step_area_accountability(
        self,
        area_agent: str,
        record: ActivationRecord,
        action: int,
    ) -> PlantStepResult:
        if int(action) != 0:
            self._env._invalid_actions += 1

        runtime = self._area_runtimes[area_agent]
        decision = runtime.handle_activation(record)
        if decision.issued_contract is not None:
            self.runtime_ledger.apply_protocol_object(
                decision.issued_contract,
                timestamp=float(self._env.simulator.time),
            )
            self._pending_runtime_contract_id = decision.issued_contract.contract_id
            if decision.issued_contract.recipient.startswith("cell_"):
                self._pending_cell_id = int(
                    decision.issued_contract.recipient.split("_", 1)[1]
                )
                self._pending_area_id = int(area_agent.split("_", 1)[1])
        for message in decision.emitted_messages:
            self.runtime_ledger.apply_protocol_object(
                message,
                timestamp=float(self._env.simulator.time),
            )
        if decision.propagation_action != PropagationAction.REROUTE:
            self._clear_pending_protocol_state()
        self.current_decision_epoch = self._annotate_epoch(
            interpret_decision_epoch(
                world_time=float(self._env.simulator.time),
                events=(),
                messages=decision.emitted_messages,
            )
        )

        info = self._augment_info(
            self._env.render(),
            activation_trace=self._serialize_records(
                self.current_decision_epoch.records_by_agent.values()
            ),
            protocol_trace=[message.to_dict() for message in decision.emitted_messages],
            world_event_trace=[],
            settlement_trace=self.runtime_ledger.settlement_trace(),
            reward_components={
                "delta_time": 0.0,
                "delta_tardiness": 0.0,
                "energy_penalty": 0.0,
                "carbon_penalty": 0.0,
                "invalid_action_penalty": 0.0,
                "deadlock_penalty": 0.0,
                "terminal_bonus": 0.0,
                "total": 0.0,
            },
        )
        return PlantStepResult(
            observation=self._env.observe(area_agent),
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
        )

    def _step_cell(self, cell_agent: str, action: int) -> PlantStepResult:
        record = self.current_decision_epoch.records_by_agent[cell_agent]
        if record.functional_loop == "commitment_management":
            return self._step_cell_commitment(cell_agent, record, int(action))
        if record.functional_loop == "local_regulation":
            return self._step_cell_local_regulation(cell_agent, record, int(action))
        raise ValueError(
            f"Unsupported cell functional loop at runtime boundary: {record.functional_loop}"
        )

    def _step_cell_commitment(
        self,
        cell_agent: str,
        record: ActivationRecord,
        action: int,
    ) -> PlantStepResult:
        if (
            self._pending_plant_action is None
            or self._pending_area_id is None
            or self._pending_cell_id is None
            or self._pending_runtime_contract_id is None
        ):
            raise RuntimeError("Cell activation has no staged plant/area decision to execute")

        runtime = self._cell_runtimes[cell_agent]
        legal_actions = runtime.legal_actions(record)
        if 0 <= int(action) < len(legal_actions):
            selected_action = legal_actions[int(action)]
        else:
            selected_action = legal_actions[0]
            self._env._invalid_actions += 1
        decision = runtime.handle_activation(record, action=selected_action)
        protocol_messages = tuple(
            self._enrich_cell_reject_message(record, message)
            for message in decision.emitted_messages
        )
        for message in protocol_messages:
            self.runtime_ledger.apply_protocol_object(
                message,
                timestamp=float(self._env.simulator.time),
            )

        if decision.action_kind == "accept_contract":
            contract = self._env._release_next_job(
                self._pending_cell_id,
                job_id=self._pending_backlog_id,
            )
            self._env._last_contract = contract
            self._pending_backlog_id = None
        else:
            self._env._last_contract = None
            self._pending_runtime_contract_id = None
        self._env._last_transition = {
            "time_start": float(self._env.simulator.time),
            "time_end": float(self._env.simulator.time),
            "events": [],
            "event_records": [],
            "deadlocked": False,
            "energy_kwh_step": 0.0,
            "energy_cost_step": 0.0,
            "carbon_kg_step": 0.0,
        }
        info = self._env.render()
        self._sync_runtime_contract_from_world(info)
        if decision.action_kind == "reject_contract":
            self.current_decision_epoch = self._annotate_epoch(
                interpret_decision_epoch(
                    world_time=float(self._env.simulator.time),
                    events=(),
                    messages=protocol_messages,
                )
            )
            augmented_info = self._augment_info(
                info,
                activation_trace=self._serialize_records(
                    self.current_decision_epoch.records_by_agent.values()
                ),
                protocol_trace=[message.to_dict() for message in protocol_messages],
                world_event_trace=[],
                settlement_trace=self.runtime_ledger.settlement_trace(),
                reward_components={
                    "delta_time": 0.0,
                    "delta_tardiness": 0.0,
                    "energy_penalty": 0.0,
                    "carbon_penalty": 0.0,
                    "invalid_action_penalty": 0.0,
                    "deadlock_penalty": 0.0,
                    "terminal_bonus": 0.0,
                    "total": 0.0,
                },
            )
            return PlantStepResult(
                observation=self._env.observe(cell_agent),
                reward=0.0,
                terminated=bool(self._env._terminated),
                truncated=bool(self._env._truncated),
                info=augmented_info,
            )
        self.current_decision_epoch = self._build_current_epoch()
        world_event_trace: list[dict[str, Any]] = []
        reward_components = {
            "delta_time": 0.0,
            "delta_tardiness": 0.0,
            "energy_penalty": 0.0,
            "carbon_penalty": 0.0,
            "invalid_action_penalty": 0.0,
            "deadlock_penalty": 0.0,
            "terminal_bonus": 0.0,
            "total": 0.0,
        }
        terminated = bool(self._env._terminated)
        truncated = bool(self._env._truncated)
        if not self.current_decision_epoch.agents_to_act and not (terminated or truncated):
            tardiness_before = self._compute_total_tardiness()
            transition = self._advance_world_until_decision(stop_on_dispatch=True)
            self._env._last_transition = transition
            terminated = self._env._is_done()
            truncated = False
            if not terminated:
                if self._env.simulator.time >= float(self._env.config.horizon_hours):
                    truncated = True
                if bool(transition["deadlocked"]):
                    truncated = True
                if self._should_truncate_for_no_progress(transition):
                    truncated = True
            self._env._terminated = terminated
            self._env._truncated = truncated
            info = self._env.render()
            self._sync_runtime_contract_from_world(info)
            tardiness_after = self._compute_total_tardiness()
            reward_components = self.runtime_ledger.finalize_interval(
                start_time=float(transition.get("time_start", info.get("time", 0.0))),
                end_time=float(transition.get("time_end", info.get("time", 0.0))),
                energy_kwh=float(info.get("energy_kwh_step", 0.0)),
                carbon_kg=float(info.get("carbon_kg_step", 0.0)),
                invalid_action=False,
                terminated=terminated,
                deadlocked=bool(info.get("deadlocked", False)),
                tardiness_before=tardiness_before,
                tardiness_after=tardiness_after,
                energy_overshoot_kwh=self._aggregate_energy_overshoot(info),
                carbon_overshoot_kg=self._aggregate_carbon_overshoot(info),
            ).to_dict()
            self.current_decision_epoch = self._build_current_epoch()
            world_event_trace = list(info.get("typed_events", []))
        self._clear_pending_protocol_state()

        augmented_info = self._augment_info(
            info,
            activation_trace=self._serialize_records(
                self.current_decision_epoch.records_by_agent.values()
            ),
            protocol_trace=[message.to_dict() for message in protocol_messages],
            world_event_trace=world_event_trace,
            settlement_trace=self.runtime_ledger.settlement_trace(),
            reward_components=reward_components,
        )
        return PlantStepResult(
            observation=self._env.observe(cell_agent),
            reward=float(reward_components["total"]),
            terminated=bool(self._env._terminated),
            truncated=bool(self._env._truncated),
            info=augmented_info,
        )

    def _step_cell_local_regulation(
        self,
        cell_agent: str,
        record: ActivationRecord,
        action: int,
    ) -> PlantStepResult:
        runtime = self._cell_runtimes[cell_agent]
        cell_id = int(cell_agent.split("_", 1)[1])
        dispatch_options = self._env._dispatch_options_for_cell(cell_id)
        if not dispatch_options:
            raise RuntimeError(f"{cell_agent} has no legal dispatch options at this epoch")

        invalid_action = False
        if 0 <= action < len(dispatch_options):
            selected_index = int(action)
        else:
            selected_index = 0
            invalid_action = True
            self._env._invalid_actions += 1
        selected_option = dispatch_options[selected_index]

        legal_actions = runtime.legal_actions(record)
        decision = runtime.handle_activation(record, action=legal_actions[selected_index])
        tardiness_before = self._compute_total_tardiness()
        self._env.dispatch_job_to_machine(selected_option.job_id, selected_option.machine_id)
        transition = self._advance_world_until_decision(stop_on_dispatch=True)
        self._env._last_transition = transition

        terminated = self._env._is_done()
        truncated = False
        if not terminated:
            if self._env.simulator.time >= float(self._env.config.horizon_hours):
                truncated = True
            if bool(transition["deadlocked"]):
                truncated = True
            if self._should_truncate_for_no_progress(transition):
                truncated = True
        self._env._terminated = terminated
        self._env._truncated = truncated

        info = self._env.render()
        self._sync_runtime_contract_from_world(info)
        tardiness_after = self._compute_total_tardiness()
        reward_breakdown = self.runtime_ledger.finalize_interval(
            start_time=float(transition.get("time_start", info.get("time", 0.0))),
            end_time=float(transition.get("time_end", info.get("time", 0.0))),
            energy_kwh=float(info.get("energy_kwh_step", 0.0)),
            carbon_kg=float(info.get("carbon_kg_step", 0.0)),
            invalid_action=invalid_action,
            terminated=terminated,
            deadlocked=bool(info.get("deadlocked", False)),
            tardiness_before=tardiness_before,
            tardiness_after=tardiness_after,
            energy_overshoot_kwh=self._aggregate_energy_overshoot(info),
            carbon_overshoot_kg=self._aggregate_carbon_overshoot(info),
        )
        self.current_decision_epoch = self._build_current_epoch()

        info["last_local_dispatch"] = self._env._dispatch_option_to_public_dict(selected_option)
        augmented_info = self._augment_info(
            info,
            activation_trace=self._serialize_records(
                self.current_decision_epoch.records_by_agent.values()
            ),
            protocol_trace=[message.to_dict() for message in decision.emitted_messages],
            world_event_trace=list(info.get("typed_events", [])),
            settlement_trace=self.runtime_ledger.settlement_trace(),
            reward_components=reward_breakdown.to_dict(),
        )
        return PlantStepResult(
            observation=self._env.observe(cell_agent),
            reward=reward_breakdown.total,
            terminated=terminated,
            truncated=truncated,
            info=augmented_info,
        )

    def _step_multi_cell_local_regulation(
        self,
        action_dict: Mapping[str, int],
    ) -> PlantStepResult:
        protocol_trace: list[dict[str, Any]] = []
        last_local_dispatches: list[dict[str, Any]] = []
        invalid_action = False

        for cell_agent in sorted(action_dict):
            record = self.current_decision_epoch.records_by_agent[cell_agent]
            runtime = self._cell_runtimes[cell_agent]
            dispatch_options = tuple(record.context_handles.get("dispatch_options", ()))
            if not dispatch_options:
                raise RuntimeError(f"{cell_agent} has no legal dispatch options at this epoch")

            selected_index = int(action_dict[cell_agent])
            if not 0 <= selected_index < len(dispatch_options):
                selected_index = 0
                invalid_action = True
                self._env._invalid_actions += 1

            legal_actions = runtime.legal_actions(record)
            decision = runtime.handle_activation(record, action=legal_actions[selected_index])
            protocol_trace.extend(message.to_dict() for message in decision.emitted_messages)

            selected_option = dict(dispatch_options[selected_index])
            self._env.dispatch_job_to_machine(
                int(selected_option["job_id"]),
                int(selected_option["machine_id"]),
            )
            last_local_dispatches.append(selected_option)

        tardiness_before = self._compute_total_tardiness()
        transition = self._advance_world_until_decision(stop_on_dispatch=True)
        self._env._last_transition = transition

        terminated = self._env._is_done()
        truncated = False
        if not terminated:
            if self._env.simulator.time >= float(self._env.config.horizon_hours):
                truncated = True
            if bool(transition["deadlocked"]):
                truncated = True
            if self._should_truncate_for_no_progress(transition):
                truncated = True
        self._env._terminated = terminated
        self._env._truncated = truncated

        info = self._env.render()
        self._sync_runtime_contract_from_world(info)
        tardiness_after = self._compute_total_tardiness()
        reward_breakdown = self.runtime_ledger.finalize_interval(
            start_time=float(transition.get("time_start", info.get("time", 0.0))),
            end_time=float(transition.get("time_end", info.get("time", 0.0))),
            energy_kwh=float(info.get("energy_kwh_step", 0.0)),
            carbon_kg=float(info.get("carbon_kg_step", 0.0)),
            invalid_action=invalid_action,
            terminated=terminated,
            deadlocked=bool(info.get("deadlocked", False)),
            tardiness_before=tardiness_before,
            tardiness_after=tardiness_after,
            energy_overshoot_kwh=self._aggregate_energy_overshoot(info),
            carbon_overshoot_kg=self._aggregate_carbon_overshoot(info),
        )
        self.current_decision_epoch = self._build_current_epoch()

        info["last_local_dispatches"] = [dict(item) for item in last_local_dispatches]
        augmented_info = self._augment_info(
            info,
            activation_trace=self._serialize_records(
                self.current_decision_epoch.records_by_agent.values()
            ),
            protocol_trace=protocol_trace,
            world_event_trace=list(info.get("typed_events", [])),
            settlement_trace=self.runtime_ledger.settlement_trace(),
            reward_components=reward_breakdown.to_dict(),
        )
        next_observation_agent = self.agents_to_act()[0] if self.agents_to_act() else "plant"
        return PlantStepResult(
            observation=self._env.observe(next_observation_agent),
            reward=reward_breakdown.total,
            terminated=terminated,
            truncated=truncated,
            info=augmented_info,
        )

    def _build_current_epoch(self) -> DecisionEpoch:
        if self._env._terminated or self._env._truncated:
            return DecisionEpoch(
                activation_timestamp=float(self._env.simulator.time),
                records_by_agent={},
            )

        if self._supports_backlog_selection():
            if self._pending_backlog_id is not None:
                return self._build_plant_area_selection_epoch(
                    selected_backlog_id=self._pending_backlog_id
                )
            if self.controller_valid_backlog_ids():
                return self._build_plant_backlog_selection_epoch()

        if self._env._valid_release_areas():
            event = HolonicEvent(
                time=float(self._env.simulator.time),
                kind="release_window_open",
                scope="plant",
                payload={"shared_backlog_remaining": len(self._env.shared_backlog)},
            )
            return self._annotate_epoch(
                interpret_decision_epoch(
                    world_time=float(self._env.simulator.time),
                    events=(event,),
                )
            )

        return self._build_local_dispatch_epoch()

    def _build_plant_backlog_selection_epoch(self) -> DecisionEpoch:
        visible_backlog_candidates = self.controller_visible_backlog_candidates()
        valid_backlog_ids = self.controller_valid_backlog_ids()
        record = ActivationRecord(
            agent_id="plant",
            activation_cause="world_event",
            functional_loop="commitment_management",
            activation_timestamp=float(self._env.simulator.time),
            decision_kind="backlog_selection",
            event_summary=(
                HolonicEvent(
                    time=float(self._env.simulator.time),
                    kind="release_backlog_selection_window",
                    scope="plant",
                    payload={
                        "visible_backlog_count": len(visible_backlog_candidates),
                        "valid_backlog_count": len(valid_backlog_ids),
                    },
                ),
            ),
            context_handles={
                "plant_decision_kind": "backlog_selection",
                "visible_backlog_candidates": visible_backlog_candidates,
                "valid_backlog_ids": valid_backlog_ids,
                "selected_backlog_id": None,
                "selected_backlog_stage_id": None,
                "selected_backlog_valid_areas": [],
                "selected_backlog_valid_cells_by_area": {},
            },
        )
        return DecisionEpoch(
            activation_timestamp=float(self._env.simulator.time),
            records_by_agent={"plant": record},
        )

    def _build_plant_area_selection_epoch(self, *, selected_backlog_id: int) -> DecisionEpoch:
        selected_context = self._selected_backlog_context(selected_backlog_id)
        record = ActivationRecord(
            agent_id="plant",
            activation_cause="staged_decision",
            functional_loop="commitment_management",
            activation_timestamp=float(self._env.simulator.time),
            decision_kind="area_selection",
            event_summary=(
                HolonicEvent(
                    time=float(self._env.simulator.time),
                    kind="release_area_selection_window",
                    scope="plant",
                    payload={
                        "selected_backlog_id": int(selected_backlog_id),
                        "valid_area_count": len(selected_context["selected_backlog_valid_areas"]),
                    },
                ),
            ),
            context_handles={
                "plant_decision_kind": "area_selection",
                "visible_backlog_candidates": self.controller_visible_backlog_candidates(),
                "valid_backlog_ids": self.controller_valid_backlog_ids(),
                **selected_context,
            },
        )
        return DecisionEpoch(
            activation_timestamp=float(self._env.simulator.time),
            records_by_agent={"plant": record},
        )

    def _build_plant_cell_selection_epoch(self, *, selected_backlog_id: int) -> DecisionEpoch:
        selected_context = self._selected_backlog_context(selected_backlog_id)
        valid_execution_cells = [
            int(cell_id)
            for cell_ids in dict(selected_context["selected_backlog_valid_cells_by_area"]).values()
            for cell_id in cell_ids
        ]
        record = ActivationRecord(
            agent_id="plant",
            activation_cause="staged_decision",
            functional_loop="commitment_management",
            activation_timestamp=float(self._env.simulator.time),
            decision_kind="cell_selection",
            event_summary=(
                HolonicEvent(
                    time=float(self._env.simulator.time),
                    kind="release_cell_selection_window",
                    scope="plant",
                    payload={
                        "selected_backlog_id": int(selected_backlog_id),
                        "valid_cell_count": len(valid_execution_cells),
                    },
                ),
            ),
            context_handles={
                "plant_decision_kind": "cell_selection",
                "visible_backlog_candidates": self.controller_visible_backlog_candidates(),
                "valid_backlog_ids": self.controller_valid_backlog_ids(),
                "valid_execution_cells": valid_execution_cells,
                **selected_context,
            },
        )
        return DecisionEpoch(
            activation_timestamp=float(self._env.simulator.time),
            records_by_agent={"plant": record},
        )

    def _build_local_dispatch_epoch(self) -> DecisionEpoch:
        records_by_agent: dict[str, ActivationRecord] = {}
        for cell_id in sorted(self._env.assembly.cells):
            dispatch_options = self._env._dispatch_options_for_cell(cell_id)
            if not dispatch_options:
                continue

            public_options = tuple(
                self._env._dispatch_option_to_public_dict(option)
                for option in dispatch_options
            )
            first_option = public_options[0]
            event = HolonicEvent(
                time=float(self._env.simulator.time),
                kind="machine_idle",
                scope=f"cell_{cell_id}",
                payload={
                    "machine_id": int(first_option["machine_id"]),
                    "dispatch_option_count": len(public_options),
                    "cell_id": int(cell_id),
                },
            )
            records_by_agent[f"cell_{cell_id}"] = ActivationRecord(
                agent_id=f"cell_{cell_id}",
                activation_cause="world_event",
                functional_loop="local_regulation",
                activation_timestamp=float(self._env.simulator.time),
                event_summary=(event,),
                context_handles={"dispatch_options": public_options},
            )

        if records_by_agent:
            return DecisionEpoch(
                activation_timestamp=float(self._env.simulator.time),
                records_by_agent=records_by_agent,
            )

        return DecisionEpoch(
            activation_timestamp=float(self._env.simulator.time),
            records_by_agent={},
        )

    def _annotate_epoch(self, epoch: DecisionEpoch) -> DecisionEpoch:
        records_by_agent: dict[str, ActivationRecord] = {}
        for agent_id, record in epoch.records_by_agent.items():
            if agent_id == "plant" and record.functional_loop == "accountability_and_escalation":
                message = dict(record.context_handles).get("message")
                payload = {} if message is None else dict(getattr(message, "payload", {}))
                selected_backlog_valid_areas = payload.get("selected_backlog_valid_areas")
                if selected_backlog_valid_areas is None:
                    selected_backlog_valid_areas = [
                        int(str(area_id).split("_", 1)[1])
                        for area_id in payload.get("attempted_area_ids", [])
                        if str(area_id).startswith("area_")
                    ]
                records_by_agent[agent_id] = ActivationRecord(
                    agent_id=record.agent_id,
                    activation_cause=record.activation_cause,
                    functional_loop=record.functional_loop,
                    activation_timestamp=record.activation_timestamp,
                    decision_kind="accountability_reroute",
                    event_summary=record.event_summary,
                    context_handles={
                        **dict(record.context_handles),
                        "plant_decision_kind": "accountability_reroute",
                        "selected_backlog_id": payload.get("shared_backlog_id"),
                        "selected_backlog_stage_id": payload.get("selected_backlog_stage_id"),
                        "selected_backlog_valid_areas": [
                            int(area_id) for area_id in selected_backlog_valid_areas
                        ],
                        "selected_backlog_valid_cells_by_area": dict(
                            payload.get("selected_backlog_valid_cells_by_area", {})
                        ),
                    },
                )
                continue
            if agent_id.startswith("area_") and record.functional_loop == "commitment_management":
                context_handles = dict(record.context_handles)
                message = context_handles.get("message")
                payload: dict[str, Any] = {}
                if isinstance(message, TreeMessage):
                    payload = dict(message.payload)
                elif isinstance(message, Mapping):
                    payload = dict(message.get("payload", {}))
                area_id = int(agent_id.split("_", 1)[1])
                valid_cells = [
                    int(cell_id)
                    for cell_id in context_handles.get("valid_cells", ())
                ]
                if not valid_cells:
                    valid_cells = [
                        int(cell_id)
                        for cell_id in payload.get("candidate_cell_ids", ())
                    ]
                if not valid_cells:
                    valid_cells_by_area = payload.get(
                        "selected_backlog_valid_cells_by_area",
                        {},
                    )
                    if isinstance(valid_cells_by_area, Mapping):
                        candidate_cells = valid_cells_by_area.get(
                            area_id,
                            valid_cells_by_area.get(str(area_id), ()),
                        )
                        valid_cells = [int(cell_id) for cell_id in candidate_cells]
                if not valid_cells:
                    valid_cells = self.controller_valid_cell_ids(area_id)
                records_by_agent[agent_id] = ActivationRecord(
                    agent_id=record.agent_id,
                    activation_cause=record.activation_cause,
                    functional_loop=record.functional_loop,
                    activation_timestamp=record.activation_timestamp,
                    decision_kind=record.decision_kind,
                    event_summary=record.event_summary,
                    context_handles={
                        **context_handles,
                        "valid_cells": list(valid_cells),
                        "valid_cell_count": len(valid_cells),
                    },
                )
                continue
            records_by_agent[agent_id] = record
        return DecisionEpoch(
            activation_timestamp=epoch.activation_timestamp,
            records_by_agent=records_by_agent,
        )

    @staticmethod
    def _transition_has_no_progress(transition: Mapping[str, Any]) -> bool:
        return (
            float(transition.get("time_end", 0.0)) == float(transition.get("time_start", 0.0))
            and not list(transition.get("events", ()))
        )

    def _advance_world_until_decision(
        self,
        *,
        stop_on_dispatch: bool,
    ) -> dict[str, Any]:
        return self._env._advance_until_decision(
            auto_dispatch=False,
            stop_on_dispatch=stop_on_dispatch,
        )

    def _should_truncate_for_no_progress(self, transition: Mapping[str, Any]) -> bool:
        if not self._transition_has_no_progress(transition):
            return False
        next_epoch = self._build_current_epoch()
        return not bool(next_epoch.agents_to_act)

    def _is_multi_cell_local_regulation_epoch(
        self,
        current_agents: Sequence[str],
    ) -> bool:
        return all(
            agent_id.startswith("cell_")
            and self.current_decision_epoch.records_by_agent[agent_id].functional_loop
            == "local_regulation"
            for agent_id in current_agents
        )

    def supported_execution_modes(self) -> list[str]:
        return list(self._SUPPORTED_EXECUTION_MODES)

    @classmethod
    def declared_supported_execution_modes(cls) -> list[str]:
        return list(cls._SUPPORTED_EXECUTION_MODES)

    @classmethod
    def benchmark_role_mapping(cls) -> dict[str, str]:
        return dict(cls._BENCHMARK_ROLE_MAPPING)

    @classmethod
    def declared_benchmark_role_mapping(cls) -> dict[str, str]:
        return cls.benchmark_role_mapping()

    def current_execution_mode(
        self,
        current_agents: Sequence[str] | None = None,
    ) -> str:
        agents = list(self.agents_to_act() if current_agents is None else current_agents)
        if len(agents) <= 1:
            return "single_agent"
        if self._is_multi_cell_local_regulation_epoch(agents):
            return "same_timestamp_multi_cell_local_regulation"
        return "unsupported_multi_agent"

    def _resolve_plant_action(
        self,
        action: int,
        *,
        valid_area_ids: Sequence[int] | None = None,
    ) -> tuple[int, int]:
        valid_areas = (
            list(valid_area_ids)
            if valid_area_ids is not None
            else list(self.controller_valid_area_ids())
        )
        if not valid_areas:
            raise RuntimeError("Plant has no valid release area at the current decision epoch")
        valid_area_indices = [self._env.area_index[area_id] for area_id in valid_areas]
        if action in valid_area_indices:
            return self._env.index_to_area[action], action
        fallback_area_id = valid_areas[0]
        return fallback_area_id, self._env.area_index[fallback_area_id]

    def _resolve_area_action(self, area_agent: str, action: int) -> tuple[str, int]:
        area_id = int(area_agent.split("_", 1)[1])
        valid_cells = list(self.controller_valid_cell_ids(area_id))
        if not valid_cells:
            raise RuntimeError(f"{area_agent} has no valid child holon to assign")
        if action in valid_cells:
            return f"cell_{action}", action
        return f"cell_{valid_cells[0]}", valid_cells[0]

    def _make_assignment_request(self, selected_area_id: int) -> TreeMessage:
        shared_backlog_id = int(
            self._pending_backlog_id
            if self._pending_backlog_id is not None
            else self._env.shared_backlog[0]
        )
        synthetic_contract_id = (
            f"req@{float(self._env.simulator.time):.3f}:{selected_area_id}:{shared_backlog_id}"
        )
        payload = {
            "contract_id": synthetic_contract_id,
            "work_id": f"shared_backlog_{shared_backlog_id}",
            "shared_backlog_id": shared_backlog_id,
            "area_id": selected_area_id,
            **self._selected_backlog_context(shared_backlog_id),
        }
        payload_builder = getattr(self._env, "build_assignment_request_payload", None)
        if callable(payload_builder):
            payload.update(dict(payload_builder(shared_backlog_id, selected_area_id)))
        return TreeMessage(
            sender="plant",
            recipient=f"area_{selected_area_id}",
            kind="assignment_request",
            payload=payload,
        )

    def _make_revise_contract_request(
        self,
        selected_area_id: int,
        record: ActivationRecord,
    ) -> TreeMessage:
        message = record.context_handles.get("message")
        if not isinstance(message, TreeMessage):
            raise ValueError("Plant accountability requires TreeMessage activation context")
        upstream_contract_id = str(
            message.payload.get(
                "upstream_contract_id",
                message.payload.get("contract_id", ""),
            )
        )
        work_id = str(message.payload.get("work_id", upstream_contract_id))
        shared_backlog_id = message.payload.get("shared_backlog_id")
        attempted_area_ids = [
            str(area_id)
            for area_id in message.payload.get("attempted_area_ids", [])
            if isinstance(area_id, str)
        ]
        selected_area_key = f"area_{selected_area_id}"
        if selected_area_key not in attempted_area_ids:
            attempted_area_ids.append(selected_area_key)
        payload: dict[str, Any] = {
            "contract_id": upstream_contract_id,
            "upstream_contract_id": upstream_contract_id,
            "work_id": work_id,
            "shared_backlog_id": shared_backlog_id,
            "area_id": selected_area_id,
            "attempted_area_ids": attempted_area_ids,
            "reroute_reason": "area_escalation",
            "escalation_status": message.payload.get("status"),
            **self._selected_backlog_context(
                None if shared_backlog_id is None else int(shared_backlog_id)
            ),
        }
        return TreeMessage(
            sender="plant",
            recipient=f"area_{selected_area_id}",
            kind="revise_contract",
            payload=payload,
        )

    @staticmethod
    def _with_preferred_child(record: ActivationRecord, child_id: str) -> ActivationRecord:
        message = record.context_handles.get("message")
        if not isinstance(message, TreeMessage):
            raise ValueError("Area activation record is missing its parent message")
        preferred_message = TreeMessage(
            sender=message.sender,
            recipient=message.recipient,
            kind=message.kind,
            payload={
                **dict(message.payload),
                "preferred_child": child_id,
            },
        )
        return ActivationRecord(
            agent_id=record.agent_id,
            activation_cause=record.activation_cause,
            functional_loop=record.functional_loop,
            activation_timestamp=record.activation_timestamp,
            event_summary=record.event_summary,
            context_handles={
                **dict(record.context_handles),
                "message": preferred_message,
            },
        )

    @staticmethod
    def _enrich_cell_reject_message(
        record: ActivationRecord,
        message: TreeMessage,
    ) -> TreeMessage:
        if message.kind != "reject_contract":
            return message
        parent_message = record.context_handles.get("message")
        if not isinstance(parent_message, TreeMessage):
            return message
        attempted_child_ids = [
            str(child_id)
            for child_id in parent_message.payload.get("attempted_child_ids", [])
            if isinstance(child_id, str)
        ]
        if not attempted_child_ids and str(parent_message.recipient).startswith("cell_"):
            attempted_child_ids = [str(parent_message.recipient)]
        if str(parent_message.recipient) not in attempted_child_ids:
            attempted_child_ids.append(str(parent_message.recipient))
        return TreeMessage(
            sender=message.sender,
            recipient=message.recipient,
            kind=message.kind,
            payload={
                **dict(message.payload),
                "upstream_contract_id": parent_message.payload.get(
                    "upstream_contract_id",
                    parent_message.payload.get("contract_id"),
                ),
                "work_id": parent_message.payload.get("work_id"),
                "attempted_child_ids": attempted_child_ids,
                "shared_backlog_id": parent_message.payload.get("shared_backlog_id"),
                "attempted_area_ids": list(
                    parent_message.payload.get("attempted_area_ids", [])
                ),
            },
        )

    def _valid_plant_accountability_area_ids(
        self,
        record: ActivationRecord,
    ) -> list[int]:
        message = record.context_handles.get("message")
        valid_areas = [int(area_id) for area_id in self._env._valid_release_areas()]
        if not isinstance(message, TreeMessage):
            return valid_areas
        attempted_area_ids = {
            str(area_id)
            for area_id in message.payload.get("attempted_area_ids", [])
            if isinstance(area_id, str)
        }
        attempted_area_ids.add(str(message.payload.get("area_id", "")))
        filtered = [
            area_id
            for area_id in valid_areas
            if f"area_{area_id}" not in attempted_area_ids
        ]
        return filtered

    def _plant_accountability_action_mask(
        self,
        record: ActivationRecord,
    ) -> np.ndarray:
        mask = np.zeros_like(self._env.action_mask("plant"))
        for area_id in self._valid_plant_accountability_area_ids(record):
            mask[self._env.area_index[int(area_id)]] = 1
        return mask

    def _clear_pending_protocol_state(self) -> None:
        self._pending_backlog_id = None
        self._pending_plant_action = None
        self._pending_area_id = None
        self._pending_cell_id = None
        self._pending_runtime_contract_id = None

    def _reset_protocol_livelock_guard(self) -> None:
        self._protocol_livelock_streak = 0

    @staticmethod
    def _protocol_livelock_fingerprint(info: Mapping[str, Any]) -> tuple[object, ...]:
        plant = dict(info.get("plant", {}))
        cell_summaries = dict(info.get("cell_summaries", {}))
        normalized_cells = []
        for raw_cell_id, raw_summary in sorted(
            cell_summaries.items(),
            key=lambda item: int(dict(item[1]).get("cell_id", item[0])),
        ):
            summary = dict(raw_summary)
            normalized_cells.append(
                (
                    int(summary.get("cell_id", raw_cell_id)),
                    int(summary.get("backlog_jobs", 0)),
                    int(summary.get("released_jobs", 0)),
                    int(summary.get("active_jobs", 0)),
                    int(summary.get("completed_jobs", 0)),
                    int(summary.get("ready_jobs", 0)),
                    int(summary.get("inbound_jobs", 0)),
                    int(summary.get("busy_machines", 0)),
                    int(summary.get("down_machines", 0)),
                )
            )
        return (
            float(info.get("time", 0.0)),
            int(plant.get("released_jobs_total", 0)),
            int(plant.get("completed_jobs_total", 0)),
            int(plant.get("backlog_jobs_total", 0)),
            int(plant.get("future_backlog_jobs_total", 0)),
            int(plant.get("blocked_jobs_total", 0)),
            int(plant.get("buffer_waiting_jobs_total", 0)),
            int(plant.get("inbound_jobs_total", 0)),
            int(plant.get("busy_machines_total", 0)),
            int(plant.get("setup_machines_total", 0)),
            int(plant.get("down_machines_total", 0)),
            int(plant.get("dormant_jobs_total", 0)),
            tuple(normalized_cells),
        )

    def _apply_protocol_livelock_guard(
        self,
        *,
        world_state_before: Mapping[str, Any],
        result: PlantStepResult,
    ) -> PlantStepResult:
        if result.terminated or result.truncated:
            self._reset_protocol_livelock_guard()
            return result

        before_fingerprint = self._protocol_livelock_fingerprint(world_state_before)
        after_fingerprint = self._protocol_livelock_fingerprint(result.info)
        if after_fingerprint == before_fingerprint:
            self._protocol_livelock_streak += 1
        else:
            self._reset_protocol_livelock_guard()
            return result

        if self._protocol_livelock_streak < self._MAX_PROTOCOL_LIVELOCK_STEPS:
            return result

        livelock_streak = int(self._protocol_livelock_streak)
        self._env._terminated = False
        self._env._truncated = True
        self.current_decision_epoch = DecisionEpoch(
            activation_timestamp=float(self._env.simulator.time),
            records_by_agent={},
        )

        result.terminated = False
        result.truncated = True
        result.info["protocol_livelock_detected"] = True
        result.info["protocol_livelock_streak"] = livelock_streak
        result.info["truncated_reason"] = "protocol_livelock"
        result.info["current_decision_epoch"] = self._serialize_epoch(self.current_decision_epoch)
        result.info["agents_to_act"] = []
        result.info["runtime_state_kind"] = self.runtime_state_kind(result.info)
        self._reset_protocol_livelock_guard()
        return result

    def _compute_total_tardiness(self) -> float:
        total = 0.0
        for job_id, completion_time in self._env.simulator.job_completion_times.items():
            due_time = self._env.simulator.jobs[job_id].due_time
            if due_time is None:
                continue
            total += max(0.0, float(completion_time) - float(due_time))
        return total

    def _sync_runtime_contract_from_world(self, info: Mapping[str, Any]) -> None:
        if self._pending_runtime_contract_id is not None:
            last_contract = info.get("last_contract")
            if isinstance(last_contract, Mapping):
                job_id = int(last_contract["job_id"])
                total_operations = len(self._env.simulator.jobs[job_id].operations)
                self.runtime_ledger.bind_job(
                    contract_id=self._pending_runtime_contract_id,
                    job_id=job_id,
                    total_operations=total_operations,
                )

        for job_id in self.runtime_ledger.bound_job_ids():
            total_operations = len(self._env.simulator.jobs[job_id].operations)
            completed = job_id in self._env.simulator.completed_jobs
            completed_operations = (
                total_operations
                if completed
                else int(self._env.simulator.jobs[job_id].current_op_idx)
            )
            self.runtime_ledger.record_physical_progress(
                job_id=job_id,
                completed_operations=completed_operations,
                completed=completed,
                timestamp=float(info.get("time", self._env.simulator.time)),
            )
            if completed:
                self.runtime_ledger.mark_job_settled(
                    job_id=job_id,
                    timestamp=float(info.get("time", self._env.simulator.time)),
                )

    @staticmethod
    def _aggregate_energy_overshoot(info: Mapping[str, Any]) -> float:
        total = 0.0
        for summary in dict(info.get("cell_summaries", {})).values():
            total += float(dict(summary).get("energy_overshoot_kwh", 0.0))
        return total

    @staticmethod
    def _aggregate_carbon_overshoot(info: Mapping[str, Any]) -> float:
        total = 0.0
        for summary in dict(info.get("cell_summaries", {})).values():
            total += float(dict(summary).get("carbon_overshoot_kg", 0.0))
        return total

    def _interpret_activation_trace(
        self,
        info: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        world_time = float(info.get("time", self._env.simulator.time))
        events = [
            HolonicEvent(
                time=float(item["time"]),
                kind=str(item["kind"]),
                scope=str(item["scope"]),
                payload=dict(item.get("payload", {})),
            )
            for item in info.get("typed_events", [])
            if self._is_runtime_agent_scope(str(item.get("scope", "")))
        ]
        messages = [
            TreeMessage(
                sender=str(item["sender"]),
                recipient=str(item["recipient"]),
                kind=str(item["kind"]),
                payload=dict(item.get("payload", {})),
            )
            for item in info.get("messages", [])
        ]
        epoch = interpret_decision_epoch(
            world_time=world_time,
            events=events,
            messages=messages,
        )
        return self._serialize_records(epoch.records_by_agent.values())

    def _augment_info(
        self,
        info: Mapping[str, Any],
        *,
        activation_trace: list[dict[str, Any]],
        protocol_trace: list[dict[str, Any]] | None = None,
        world_event_trace: list[dict[str, Any]] | None = None,
        settlement_trace: list[dict[str, Any]] | None = None,
        reward_components: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        augmented = dict(info)
        augmented["activation_trace"] = activation_trace
        augmented["protocol_trace"] = (
            list(info.get("messages", []))
            if protocol_trace is None
            else protocol_trace
        )
        augmented["world_event_trace"] = (
            list(info.get("typed_events", []))
            if world_event_trace is None
            else world_event_trace
        )
        augmented["settlement_trace"] = (
            self.runtime_ledger.settlement_trace()
            if settlement_trace is None
            else settlement_trace
        )
        augmented["settlement_summary"] = self.runtime_ledger.settlement_summary()
        augmented["reward_authority"] = REWARD_AUTHORITY_ID
        augmented["reward_components"] = (
            {"total": float(augmented.get("reward", 0.0))}
            if reward_components is None
            else dict(reward_components)
        )
        augmented["violation_summary"] = self.runtime_ledger.violation_summary()
        augmented["current_decision_epoch"] = self._serialize_epoch(self.current_decision_epoch)
        augmented["agents_to_act"] = self.agents_to_act()
        augmented["authority_mode"] = self.authority_mode.to_dict()
        if self._supports_backlog_selection():
            augmented["valid_backlog_ids"] = self.controller_valid_backlog_ids()
            augmented["visible_backlog_candidates"] = self.controller_visible_backlog_candidates()
            augmented.update(self._selected_backlog_context())
        current_plant_record = self.current_decision_epoch.records_by_agent.get("plant")
        if current_plant_record is not None:
            augmented["plant_decision_kind"] = self._plant_decision_kind(current_plant_record)
        augmented["runtime_state_kind"] = self.runtime_state_kind(augmented)
        return augmented

    def runtime_state_kind(self, info: Mapping[str, Any] | None = None) -> str:
        snapshot = {} if info is None else dict(info)
        if bool(snapshot.get("deadlocked", False)):
            return "deadlock"

        current_agents = list(snapshot.get("agents_to_act", self.agents_to_act()))
        if current_agents and all(
            agent_id.startswith("cell_")
            and self.current_decision_epoch.records_by_agent[agent_id].functional_loop
            == "local_regulation"
            for agent_id in current_agents
            if agent_id in self.current_decision_epoch.records_by_agent
        ):
            return "local_dispatch_active"

        if not current_agents and len(self._env.shared_backlog) > 0:
            return "release_blocked"

        return "decision_active"

    @staticmethod
    def _serialize_epoch(epoch: DecisionEpoch) -> dict[str, Any]:
        execution_mode = TreeRuntimeEnv._execution_mode_for_epoch(epoch)
        return {
            "activation_timestamp": epoch.activation_timestamp,
            "agents_to_act": list(epoch.agents_to_act),
            "execution_mode": execution_mode,
            "supported_execution_modes": list(TreeRuntimeEnv._SUPPORTED_EXECUTION_MODES),
            "records_by_agent": {
                agent_id: TreeRuntimeEnv._serialize_record(record)
                for agent_id, record in epoch.records_by_agent.items()
            },
        }

    @staticmethod
    def _serialize_records(records: Iterable[ActivationRecord]) -> list[dict[str, Any]]:
        return [TreeRuntimeEnv._serialize_record(record) for record in records]

    @staticmethod
    def _serialize_record(record: ActivationRecord) -> dict[str, Any]:
        return {
            "agent_id": record.agent_id,
            "activation_cause": record.activation_cause,
            "functional_loop": record.functional_loop,
            "plant_decision_kind": (
                None if record.agent_id != "plant" else record.decision_kind
            ),
            "activation_timestamp": record.activation_timestamp,
            "event_summary": [TreeRuntimeEnv._serialize_event(item) for item in record.event_summary],
            "context_handles": {
                key: TreeRuntimeEnv._serialize_event(value)
                for key, value in dict(record.context_handles).items()
            },
        }

    @staticmethod
    def _serialize_event(event: object) -> Any:
        if event is None or isinstance(event, (bool, int, float, str)):
            return event
        if hasattr(event, "to_dict"):
            return TreeRuntimeEnv._serialize_event(event.to_dict())
        if isinstance(event, Mapping):
            return {
                str(key): TreeRuntimeEnv._serialize_event(value)
                for key, value in event.items()
            }
        if isinstance(event, (list, tuple)):
            return [TreeRuntimeEnv._serialize_event(item) for item in event]
        return str(event)

    @staticmethod
    def _execution_mode_for_epoch(epoch: DecisionEpoch) -> str:
        agents = list(epoch.agents_to_act)
        if len(agents) <= 1:
            return "single_agent"
        if all(
            agent_id.startswith("cell_")
            and epoch.records_by_agent[agent_id].functional_loop == "local_regulation"
            for agent_id in agents
        ):
            return "same_timestamp_multi_cell_local_regulation"
        return "unsupported_multi_agent"

    @staticmethod
    def _is_runtime_agent_scope(scope: str) -> bool:
        return scope == "plant" or scope.startswith(("area_", "cell_"))
