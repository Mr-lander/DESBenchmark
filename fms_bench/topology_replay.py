"""Topology-aware dynamic replay artifacts derived from benchmark traces."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable, Mapping
import csv
import json


@dataclass(frozen=True)
class TopologyReplayArtifactPaths:
    artifacts_csv: Path
    replay_summary_json: Path
    topology_replay_html: Path


def build_topology_replay_artifacts(
    *,
    artifacts_csv: str | Path,
    output_dir: str | Path,
) -> TopologyReplayArtifactPaths:
    artifacts_path = Path(artifacts_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = _load_csv_rows(artifacts_path)
    case_id = _stable_single_value(manifest_rows, "case_id")
    base_dir = artifacts_path.parent

    policies = sorted({row["policy_name"] for row in manifest_rows})
    replay_by_policy: dict[str, dict[str, object]] = {}
    topology_nodes: dict[str, dict[str, object]] = {}
    topology_edges: dict[str, dict[str, object]] = {}

    for row in sorted(manifest_rows, key=lambda item: item["policy_name"]):
        policy_name = row["policy_name"]
        results_payload = _load_json_payload(base_dir, row.get("results_json", ""))
        state_trace = _extract_policy_state_trace(results_payload)
        activation_records = _load_trace_records(base_dir, row["activation_trace_json"])
        settlement_records = sorted(
            _load_trace_records(base_dir, row["settlement_trace_json"]),
            key=lambda item: (float(item.get("timestamp", 0.0)), str(item.get("contract_id", ""))),
        )
        world_records = sorted(
            _load_trace_records(base_dir, row["world_event_trace_json"]),
            key=lambda item: (float(item.get("time", 0.0)), str(item.get("kind", ""))),
        )

        message_events = _timestamped_message_events(activation_records)
        for node in _infer_nodes(activation_records, settlement_records):
            topology_nodes[node["id"]] = node
        for edge in _infer_edges(message_events, settlement_records):
            topology_edges[edge["id"]] = edge

        contract_index = _build_contract_index(
            message_events=message_events,
            settlement_records=settlement_records,
        )
        replay_by_policy[policy_name] = {
            "controller_interface_mode": row["controller_interface_mode"],
            "policy_summary": _extract_policy_summary(results_payload),
            "policy_episode": _extract_policy_episode(results_payload),
            "state_trace": state_trace,
            "contract_index": contract_index,
            "frames": _build_frames(
                activation_records=activation_records,
                message_events=message_events,
                settlement_records=settlement_records,
                world_records=world_records,
                state_trace=state_trace,
                contract_index=contract_index,
            ),
        }

    topology = {
        "nodes": sorted(topology_nodes.values(), key=_node_sort_key),
        "edges": sorted(topology_edges.values(), key=lambda item: item["id"]),
    }
    payload = {
        "case_id": case_id,
        "policies": policies,
        "topology": topology,
        "replay_by_policy": replay_by_policy,
    }

    replay_summary_json = out_dir / "replay_summary.json"
    topology_replay_html = out_dir / "topology_replay.html"
    replay_summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    topology_replay_html.write_text(
        _render_html_report(payload),
        encoding="utf-8",
    )

    return TopologyReplayArtifactPaths(
        artifacts_csv=artifacts_path,
        replay_summary_json=replay_summary_json,
        topology_replay_html=topology_replay_html,
    )


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    normalized = []
    for row in rows:
        item = dict(row)
        if not item.get("controller_interface_mode"):
            item["controller_interface_mode"] = "in_process"
        normalized.append(item)
    return normalized


def _stable_single_value(rows: Iterable[Mapping[str, str]], key: str) -> str:
    values = {row[key] for row in rows}
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def _resolve_trace_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_trace_records(base_dir: Path, value: str) -> list[dict[str, object]]:
    payload = _load_json_payload(base_dir, value)
    records: list[dict[str, object]] = []
    for episode in payload.get("episodes", []):
        records.extend(episode.get("trace", []))
    return records


def _load_json_payload(base_dir: Path, value: str) -> dict[str, object]:
    if not value:
        return {}
    return json.loads(_resolve_trace_path(base_dir, value).read_text(encoding="utf-8"))


def _extract_policy_summary(results_payload: Mapping[str, object]) -> dict[str, object]:
    summary = results_payload.get("summary", {})
    if not isinstance(summary, Mapping):
        return {}
    keys = (
        "num_episodes",
        "controller_interface_mode",
        "success_rate",
        "deadlock_rate",
        "truncation_rate",
        "makespan_mean",
        "tardiness_mean",
        "completed_jobs_mean",
        "throughput_mean",
        "energy_mean",
        "carbon_mean",
        "violation_count_mean",
        "violation_magnitude_mean",
        "energy_overshoot_magnitude_total_mean",
        "carbon_overshoot_magnitude_total_mean",
        "energy_overshoot_duration_mean",
        "carbon_overshoot_duration_mean",
        "energy_overshoot_frequency_mean",
        "carbon_overshoot_frequency_mean",
        "decision_steps_mean",
        "coordination_message_count_mean",
        "active_agent_slots_mean",
        "runtime_mean_seconds",
        "activation_record_count_mean",
        "settlement_event_count_mean",
    )
    return {key: summary[key] for key in keys if key in summary}


def _extract_policy_episode(results_payload: Mapping[str, object]) -> dict[str, object]:
    episodes = results_payload.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        return {}
    episode = episodes[0]
    if not isinstance(episode, Mapping):
        return {}
    keys = (
        "episode_id",
        "seed",
        "makespan",
        "total_tardiness",
        "completed_jobs",
        "throughput",
        "total_energy_kwh",
        "total_carbon_kg",
        "total_reward",
        "backlog_jobs_total",
        "down_machines_total",
        "activation_record_count",
        "settlement_event_count",
    )
    return {key: episode[key] for key in keys if key in episode}


def _extract_policy_state_trace(results_payload: Mapping[str, object]) -> list[dict[str, object]]:
    episodes = results_payload.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        return []
    episode = episodes[0]
    if not isinstance(episode, Mapping):
        return []
    state_trace = episode.get("state_trace", [])
    if not isinstance(state_trace, list):
        return []
    normalized: list[dict[str, object]] = []
    for item in state_trace:
        if isinstance(item, Mapping):
            normalized.append(dict(item))
    return normalized


def _timestamped_message_events(activation_records: list[dict[str, object]]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for record in activation_records:
        if str(record.get("activation_cause")) != "parent_message":
            continue
        context_handles = record.get("context_handles", {})
        if not isinstance(context_handles, Mapping):
            continue
        message = context_handles.get("message")
        if not isinstance(message, Mapping):
            continue
        sender = str(message.get("sender", ""))
        recipient = str(message.get("recipient", ""))
        if not sender or not recipient:
            continue
        events.append(
            {
                "timestamp": float(record.get("activation_timestamp", 0.0)),
                "sender": sender,
                "recipient": recipient,
                "kind": str(message.get("kind", "")),
                "scope": str(message.get("scope", "parent_child")),
                "payload": dict(message.get("payload", {})),
                "contract_id": str(message.get("payload", {}).get("contract_id", "")),
                "upstream_contract_id": str(message.get("payload", {}).get("upstream_contract_id", "")),
            }
        )
    return events


def _build_contract_index(
    *,
    message_events: list[dict[str, object]],
    settlement_records: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    contract_index: dict[str, dict[str, object]] = {}

    def ensure(contract_id: str) -> dict[str, object]:
        entry = contract_index.get(contract_id)
        if entry is None:
            entry = {
                "contract_id": contract_id,
                "sender": "",
                "recipient": "",
                "work_id": "",
                "job_id": None,
                "accepted_by": "",
                "latest_status": "pending",
                "completed_operations": 0,
                "total_operations": None,
                "upstream_contract_id": "",
                "schedule_risk": "n/a",
            }
            contract_index[contract_id] = entry
        return entry

    for item in message_events:
        contract_id = str(item.get("contract_id", ""))
        if not contract_id:
            continue
        entry = ensure(contract_id)
        entry["sender"] = str(item.get("sender", "")) or str(entry.get("sender", ""))
        entry["recipient"] = str(item.get("recipient", "")) or str(entry.get("recipient", ""))
        payload = item.get("payload", {})
        if isinstance(payload, Mapping):
            work_id = str(payload.get("work_id", ""))
            if work_id:
                entry["work_id"] = work_id
            upstream_contract_id = str(payload.get("upstream_contract_id", ""))
            if upstream_contract_id:
                entry["upstream_contract_id"] = upstream_contract_id
        if str(entry.get("latest_status", "")) in {"", "pending"}:
            entry["latest_status"] = "requested"

    for item in settlement_records:
        contract_id = str(item.get("contract_id", ""))
        if not contract_id:
            continue
        entry = ensure(contract_id)
        entry["sender"] = str(item.get("sender", "")) or str(entry.get("sender", ""))
        entry["recipient"] = str(item.get("recipient", "")) or str(entry.get("recipient", ""))
        entry["work_id"] = str(item.get("work_id", "")) or str(entry.get("work_id", ""))
        entry["accepted_by"] = str(item.get("accepted_by", "")) or str(entry.get("accepted_by", ""))
        entry["latest_status"] = str(item.get("status", "")) or str(entry.get("latest_status", ""))
        entry["job_id"] = item.get("job_id") if item.get("job_id") is not None else entry.get("job_id")
        entry["completed_operations"] = int(item.get("completed_operations", entry.get("completed_operations", 0)))
        total_operations = item.get("total_operations")
        if total_operations is not None:
            entry["total_operations"] = int(total_operations)

    return dict(sorted(contract_index.items()))


def _infer_nodes(
    activation_records: list[dict[str, object]],
    settlement_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    node_ids: set[str] = set()
    for record in activation_records:
        agent_id = str(record.get("agent_id", ""))
        if agent_id:
            node_ids.add(agent_id)
        context_handles = record.get("context_handles", {})
        if isinstance(context_handles, Mapping):
            message = context_handles.get("message")
            if isinstance(message, Mapping):
                sender = str(message.get("sender", ""))
                recipient = str(message.get("recipient", ""))
                if sender:
                    node_ids.add(sender)
                if recipient:
                    node_ids.add(recipient)

    for record in settlement_records:
        sender = str(record.get("sender", ""))
        recipient = str(record.get("recipient", ""))
        if sender:
            node_ids.add(sender)
        if recipient:
            node_ids.add(recipient)

    return [
        {
            "id": node_id,
            "role": _node_role(node_id),
            "label": _node_label(node_id),
        }
        for node_id in node_ids
    ]


def _infer_edges(
    message_events: list[dict[str, object]],
    settlement_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    edges: dict[str, dict[str, object]] = {}
    for item in message_events:
        edge_id = f"{item['sender']}->{item['recipient']}"
        edges.setdefault(
            edge_id,
            {
                "id": edge_id,
                "source": item["sender"],
                "target": item["recipient"],
                "kind": "message",
            },
        )
    for item in settlement_records:
        sender = str(item.get("sender", ""))
        recipient = str(item.get("recipient", ""))
        if not sender or not recipient:
            continue
        edge_id = f"{sender}->{recipient}"
        edges.setdefault(
            edge_id,
            {
                "id": edge_id,
                "source": sender,
                "target": recipient,
                "kind": "settlement",
            },
        )
    return list(edges.values())


def _build_frames(
    *,
    activation_records: list[dict[str, object]],
    message_events: list[dict[str, object]],
    settlement_records: list[dict[str, object]],
    world_records: list[dict[str, object]],
    state_trace: list[dict[str, object]],
    contract_index: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    settlement_idx = 0
    world_idx = 0
    state_idx = 0
    activation_counts = Counter()
    message_edge_counts = Counter()
    settlement_status_counts = Counter()
    world_event_kind_counts = Counter()
    completed_contracts = 0
    shared_backlog_remaining: int | None = None
    open_contract_ids: set[str] = set()
    contract_state_by_id: dict[str, dict[str, object]] = {}
    previous_focus_contract_id: str | None = None
    frames: list[dict[str, object]] = []

    for index, record in enumerate(activation_records):
        active_agent = str(record.get("agent_id", ""))
        timestamp = float(record.get("activation_timestamp", 0.0))
        activation_counts[active_agent] += 1
        for event in record.get("event_summary", []):
            if not isinstance(event, Mapping):
                continue
            payload = event.get("payload", {})
            if not isinstance(payload, Mapping):
                continue
            value = payload.get("shared_backlog_remaining")
            if isinstance(value, (int, float)):
                shared_backlog_remaining = int(value)

        frame_message_events: list[dict[str, object]] = []
        if str(record.get("activation_cause")) == "parent_message":
            context_handles = record.get("context_handles", {})
            if isinstance(context_handles, Mapping):
                message = context_handles.get("message")
                if isinstance(message, Mapping):
                    sender = str(message.get("sender", ""))
                    recipient = str(message.get("recipient", ""))
                    edge_id = f"{sender}->{recipient}"
                    message_edge_counts[edge_id] += 1
                    frame_message_events.append(
                        {
                            "timestamp": timestamp,
                            "sender": sender,
                            "recipient": recipient,
                            "kind": str(message.get("kind", "")),
                            "scope": str(message.get("scope", "parent_child")),
                            "payload": dict(message.get("payload", {})),
                            "contract_id": str(message.get("payload", {}).get("contract_id", "")),
                            "upstream_contract_id": str(message.get("payload", {}).get("upstream_contract_id", "")),
                            "edge_id": edge_id,
                        }
                    )

        frame_settlement_events: list[dict[str, object]] = []
        while settlement_idx < len(settlement_records) and float(settlement_records[settlement_idx].get("timestamp", 0.0)) <= timestamp:
            item = dict(settlement_records[settlement_idx])
            status = str(item.get("status", ""))
            contract_id = str(item.get("contract_id", ""))
            settlement_status_counts[status] += 1
            if status in {"completed", "settled"}:
                completed_contracts += 1
            if contract_id:
                contract_state_by_id[contract_id] = item
                if status in {"completed", "settled", "rejected", "cancelled", "failed"}:
                    open_contract_ids.discard(contract_id)
                else:
                    open_contract_ids.add(contract_id)
            item["edge_id"] = f"{item.get('sender', '')}->{item.get('recipient', '')}"
            frame_settlement_events.append(item)
            settlement_idx += 1

        frame_world_events: list[dict[str, object]] = []
        while world_idx < len(world_records) and float(world_records[world_idx].get("time", 0.0)) <= timestamp:
            item = dict(world_records[world_idx])
            kind = str(item.get("kind", ""))
            world_event_kind_counts[kind] += 1
            frame_world_events.append(item)
            world_idx += 1

        current_state_snapshot: dict[str, object] = {}
        while state_idx < len(state_trace):
            item = state_trace[state_idx]
            item_time = float(item.get("time", 0.0))
            if item_time > timestamp:
                break
            current_state_snapshot = copy.deepcopy(item)
            state_idx += 1

        highlighted_nodes = {active_agent}
        highlighted_edges = set()
        for item in frame_message_events:
            highlighted_nodes.add(item["sender"])
            highlighted_nodes.add(item["recipient"])
            highlighted_edges.add(item["edge_id"])
        for item in frame_settlement_events:
            sender = str(item.get("sender", ""))
            recipient = str(item.get("recipient", ""))
            if sender:
                highlighted_nodes.add(sender)
            if recipient:
                highlighted_nodes.add(recipient)
            if item["edge_id"] != "->":
                highlighted_edges.add(item["edge_id"])

        cumulative_snapshot = {
            "activation_counts_by_node": dict(sorted(activation_counts.items())),
            "message_edge_counts": dict(sorted(message_edge_counts.items())),
            "settlement_status_counts": dict(sorted(settlement_status_counts.items())),
            "world_event_kind_counts": dict(sorted(world_event_kind_counts.items())),
            "completed_contracts": completed_contracts,
            "open_contract_count": len(open_contract_ids),
            "shared_backlog_remaining": shared_backlog_remaining,
            "node_state_by_id": _node_state_by_id(
                activation_counts=activation_counts,
                contract_state_by_id=contract_state_by_id,
            ),
        }
        node_constraint_overlay = _node_constraint_overlay(
            state_snapshot=current_state_snapshot,
            cumulative=cumulative_snapshot,
        )
        focus_contract_id = _select_focus_contract_id(
            frame_message_events=frame_message_events,
            frame_settlement_events=frame_settlement_events,
            previous_focus_contract_id=previous_focus_contract_id,
        )
        contract_events = _build_contract_events_for_frame(
            focus_contract_id=focus_contract_id,
            frame_message_events=frame_message_events,
            frame_settlement_events=frame_settlement_events,
        )
        contract_focus = _build_contract_focus_snapshot(
            focus_contract_id=focus_contract_id,
            contract_index=contract_index,
            contract_state_by_id=contract_state_by_id,
            node_constraint_overlay=node_constraint_overlay,
        )
        if focus_contract_id:
            previous_focus_contract_id = focus_contract_id

        frames.append(
            {
                "index": index,
                "timestamp": timestamp,
                "active_agent": active_agent,
                "activation_cause": str(record.get("activation_cause", "")),
                "functional_loop": str(record.get("functional_loop", "")),
                "event_summary": list(record.get("event_summary", [])),
                "message_events": frame_message_events,
                "settlement_events": frame_settlement_events,
                "world_events": frame_world_events,
                "state_snapshot": current_state_snapshot,
                "highlighted_nodes": sorted(item for item in highlighted_nodes if item),
                "highlighted_edges": sorted(item for item in highlighted_edges if item),
                "cumulative": cumulative_snapshot,
                "focus_contract_id": focus_contract_id,
                "contract_events": contract_events,
                "contract_focus": contract_focus,
                "node_constraint_overlay": node_constraint_overlay,
                "graph_overlay_by_edge": _graph_overlay_by_edge(
                    frame_message_events=frame_message_events,
                    frame_settlement_events=frame_settlement_events,
                    contract_state_by_id=contract_state_by_id,
                    node_constraint_overlay=node_constraint_overlay,
                ),
                "graph_overlay_by_node": _graph_overlay_by_node(
                    state_snapshot=current_state_snapshot,
                    cumulative=cumulative_snapshot,
                ),
            }
        )

    return frames


def _select_focus_contract_id(
    *,
    frame_message_events: list[dict[str, object]],
    frame_settlement_events: list[dict[str, object]],
    previous_focus_contract_id: str | None,
) -> str | None:
    for item in reversed(frame_settlement_events):
        contract_id = str(item.get("contract_id", ""))
        if contract_id:
            return contract_id
    for item in reversed(frame_message_events):
        contract_id = str(item.get("contract_id", "")) or str(item.get("upstream_contract_id", ""))
        if contract_id:
            return contract_id
    return previous_focus_contract_id


def _build_contract_events_for_frame(
    *,
    focus_contract_id: str | None,
    frame_message_events: list[dict[str, object]],
    frame_settlement_events: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not focus_contract_id:
        return []

    events: list[dict[str, object]] = []
    for item in frame_message_events:
        contract_id = str(item.get("contract_id", "")) or str(item.get("upstream_contract_id", ""))
        if contract_id != focus_contract_id:
            continue
        events.append(
            {
                "event_kind": "message",
                "contract_id": focus_contract_id,
                "label": str(item.get("kind", "")),
                "sender": str(item.get("sender", "")),
                "recipient": str(item.get("recipient", "")),
                "edge_id": str(item.get("edge_id", "")),
            }
        )
    for item in frame_settlement_events:
        contract_id = str(item.get("contract_id", ""))
        if contract_id != focus_contract_id:
            continue
        events.append(
            {
                "event_kind": "settlement",
                "contract_id": focus_contract_id,
                "label": str(item.get("status", "")),
                "sender": str(item.get("sender", "")),
                "recipient": str(item.get("recipient", "")),
                "edge_id": str(item.get("edge_id", "")),
                "status": str(item.get("status", "")),
            }
        )
    return events


def _build_contract_focus_snapshot(
    *,
    focus_contract_id: str | None,
    contract_index: Mapping[str, Mapping[str, object]],
    contract_state_by_id: Mapping[str, Mapping[str, object]],
    node_constraint_overlay: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if not focus_contract_id:
        return {}

    final_entry = contract_index.get(focus_contract_id, {})
    current_entry = contract_state_by_id.get(focus_contract_id, {})
    sender = str(current_entry.get("sender", "")) or str(final_entry.get("sender", ""))
    recipient = str(current_entry.get("recipient", "")) or str(final_entry.get("recipient", ""))
    recipient_overlay = node_constraint_overlay.get(recipient, {}) if recipient else {}
    return {
        "contract_id": focus_contract_id,
        "sender": sender,
        "recipient": recipient,
        "work_id": str(current_entry.get("work_id", "")) or str(final_entry.get("work_id", "")),
        "job_id": current_entry.get("job_id", final_entry.get("job_id")),
        "accepted_by": str(current_entry.get("accepted_by", "")) or str(final_entry.get("accepted_by", "")),
        "status": str(current_entry.get("status", "")) or str(final_entry.get("latest_status", "")),
        "completed_operations": int(current_entry.get("completed_operations", final_entry.get("completed_operations", 0))),
        "total_operations": current_entry.get("total_operations", final_entry.get("total_operations")),
        "schedule_risk": str(final_entry.get("schedule_risk", "n/a")),
        "capacity_wip_pressure": str(recipient_overlay.get("capacity_wip_pressure", "n/a")),
    }


def _graph_overlay_by_edge(
    *,
    frame_message_events: list[dict[str, object]],
    frame_settlement_events: list[dict[str, object]],
    contract_state_by_id: Mapping[str, Mapping[str, object]],
    node_constraint_overlay: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    overlays: dict[str, dict[str, object]] = {}
    terminal_statuses = {"completed", "settled", "rejected", "cancelled", "failed"}
    active_statuses = {"active", "partially_completed"}

    def ensure(edge_id: str, *, sender: str, recipient: str) -> dict[str, object]:
        overlay = overlays.get(edge_id)
        if overlay is None:
            overlay = {
                "edge_id": edge_id,
                "sender": sender,
                "recipient": recipient,
                "open_contracts": 0,
                "active_contracts": 0,
                "completed_contracts": 0,
                "frame_message_count": 0,
                "frame_settlement_count": 0,
                "latest_label": "",
            }
            overlays[edge_id] = overlay
        return overlay

    for item in frame_message_events:
        sender = str(item.get("sender", ""))
        recipient = str(item.get("recipient", ""))
        edge_id = str(item.get("edge_id", "")) or f"{sender}->{recipient}"
        if not sender or not recipient:
            continue
        overlay = ensure(edge_id, sender=sender, recipient=recipient)
        overlay["frame_message_count"] = int(overlay["frame_message_count"]) + 1
        overlay["latest_label"] = _edge_display_label(item.get("kind", ""))

    for item in frame_settlement_events:
        sender = str(item.get("sender", ""))
        recipient = str(item.get("recipient", ""))
        edge_id = str(item.get("edge_id", "")) or f"{sender}->{recipient}"
        if not sender or not recipient:
            continue
        overlay = ensure(edge_id, sender=sender, recipient=recipient)
        overlay["frame_settlement_count"] = int(overlay["frame_settlement_count"]) + 1
        overlay["latest_label"] = _edge_display_label(item.get("status", ""))

    for item in contract_state_by_id.values():
        sender = str(item.get("sender", ""))
        recipient = str(item.get("recipient", ""))
        if not sender or not recipient:
            continue
        edge_id = f"{sender}->{recipient}"
        overlay = ensure(edge_id, sender=sender, recipient=recipient)
        status = str(item.get("status", ""))
        if status not in terminal_statuses:
            overlay["open_contracts"] = int(overlay["open_contracts"]) + 1
        if status in active_statuses:
            overlay["active_contracts"] = int(overlay["active_contracts"]) + 1
        if status in {"completed", "settled"}:
            overlay["completed_contracts"] = int(overlay["completed_contracts"]) + 1
        if not str(overlay.get("latest_label", "")):
            overlay["latest_label"] = _edge_display_label(status)

    for overlay in overlays.values():
        recipient = str(overlay.get("recipient", ""))
        recipient_constraints = node_constraint_overlay.get(recipient, {})
        if not isinstance(recipient_constraints, Mapping):
            recipient_constraints = {}
        pressure = str(recipient_constraints.get("capacity_wip_pressure", "n/a"))
        open_contracts = int(overlay.get("open_contracts", 0))
        active_contracts = int(overlay.get("active_contracts", 0))
        completed_contracts = int(overlay.get("completed_contracts", 0))
        frame_message_count = int(overlay.get("frame_message_count", 0))
        frame_settlement_count = int(overlay.get("frame_settlement_count", 0))
        latest_label = str(overlay.get("latest_label", "")) or "idle"
        status = "neutral"
        if active_contracts > 0 and pressure == "tight":
            status = "constrained"
        elif active_contracts > 0:
            status = "active"
        elif open_contracts > 0:
            status = "open"
        elif completed_contracts > 0:
            status = "complete"
        elif frame_message_count > 0 or frame_settlement_count > 0:
            status = "active"
        overlay["status"] = status
        overlay["headline"] = f"{latest_label} | open {open_contracts}"
        overlay["detail"] = (
            f"active {active_contracts} | done {completed_contracts}"
        )
        overlay["footer"] = f"WIP {pressure}"
        overlay["badge"] = f"{latest_label} | o{open_contracts}"

    return dict(sorted(overlays.items()))


def _graph_overlay_by_node(
    *,
    state_snapshot: Mapping[str, object],
    cumulative: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    overlays: dict[str, dict[str, object]] = {}
    activation_counts = cumulative.get("activation_counts_by_node", {})
    node_state_by_id = cumulative.get("node_state_by_id", {})
    if not isinstance(activation_counts, Mapping):
        activation_counts = {}
    if not isinstance(node_state_by_id, Mapping):
        node_state_by_id = {}

    plant_state = state_snapshot.get("plant", {})
    if isinstance(plant_state, Mapping):
        backlog = _overlay_display_int(plant_state.get("backlog_jobs_total"))
        completed = _overlay_display_int(plant_state.get("completed_jobs_total"))
        busy = _overlay_display_int(plant_state.get("busy_machines_total"))
        down = _overlay_display_int(plant_state.get("down_machines_total"))
        backlog_num = _overlay_number(plant_state.get("backlog_jobs_total"))
        busy_num = _overlay_number(plant_state.get("busy_machines_total"))
        down_num = _overlay_number(plant_state.get("down_machines_total"))
        status = "complete"
        if down_num > 0:
            status = "down"
        elif backlog_num > 0:
            status = "backlog"
        elif busy_num > 0:
            status = "busy"
        overlays["plant"] = {
            "status": status,
            "headline": f"backlog {backlog} | comp {completed}",
            "detail": f"busy {busy} | down {down}",
            "footer": f"act {int(activation_counts.get('plant', 0))}",
            "activation_count": int(activation_counts.get("plant", 0)),
        }

    area_summaries = state_snapshot.get("area_summaries", {})
    if isinstance(area_summaries, Mapping):
        for area_id, summary in area_summaries.items():
            if not isinstance(summary, Mapping):
                continue
            node_id = f"area_{area_id}"
            node_state = node_state_by_id.get(node_id, {})
            if not isinstance(node_state, Mapping):
                node_state = {}
            active = summary.get("active_jobs", node_state.get("active_contracts", 0))
            backlog = summary.get("backlog_jobs", node_state.get("open_contracts", 0))
            energy = summary.get("energy_kwh")
            carbon = summary.get("carbon_kg")
            active_num = _overlay_number(active)
            backlog_num = _overlay_number(backlog)
            status = "idle"
            if active_num > 0:
                status = "active"
            elif backlog_num > 0:
                status = "backlog"
            overlays[node_id] = {
                "status": status,
                "headline": (
                    f"jobs {_overlay_display_int(active)} | "
                    f"backlog {_overlay_display_int(backlog)}"
                ),
                "detail": (
                    f"E {_overlay_display_float(energy)} | "
                    f"C {_overlay_display_float(carbon)}"
                ),
                "footer": f"act {int(activation_counts.get(node_id, 0))}",
                "activation_count": int(activation_counts.get(node_id, 0)),
            }

    cell_summaries = state_snapshot.get("cell_summaries", {})
    if isinstance(cell_summaries, Mapping):
        for cell_id, summary in cell_summaries.items():
            if not isinstance(summary, Mapping):
                continue
            node_id = f"cell_{cell_id}"
            node_state = node_state_by_id.get(node_id, {})
            if not isinstance(node_state, Mapping):
                node_state = {}
            active = summary.get("active_jobs", node_state.get("active_contracts", 0))
            ready = summary.get("ready_jobs")
            busy = summary.get("busy_machines")
            down = summary.get("down_machines")
            busy_num = _overlay_number(busy)
            down_num = _overlay_number(down)
            ready_num = _overlay_number(ready)
            active_num = _overlay_number(active)
            status = "idle"
            if down_num > 0:
                status = "down"
            elif busy_num > 0:
                status = "busy"
            elif ready_num > 0:
                status = "ready"
            elif active_num > 0:
                status = "active"
            overlays[node_id] = {
                "status": status,
                "headline": (
                    f"jobs {_overlay_display_int(active)} | "
                    f"ready {_overlay_display_int(ready)}"
                ),
                "detail": (
                    f"busy {_overlay_display_int(busy)} | "
                    f"down {_overlay_display_int(down)}"
                ),
                "footer": (
                    f"E {_overlay_display_float(summary.get('energy_kwh'))} | "
                    f"C {_overlay_display_float(summary.get('carbon_kg'))}"
                ),
                "activation_count": int(activation_counts.get(node_id, 0)),
            }

    return dict(sorted(overlays.items()))


def _node_constraint_overlay(
    *,
    state_snapshot: Mapping[str, object],
    cumulative: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    overlays: dict[str, dict[str, object]] = {}
    node_state_by_id = cumulative.get("node_state_by_id", {})
    if not isinstance(node_state_by_id, Mapping):
        node_state_by_id = {}

    plant_state = state_snapshot.get("plant", {})
    if isinstance(plant_state, Mapping):
        backlog_jobs = plant_state.get("backlog_jobs_total")
        overlays["plant"] = {
            "open_contracts": int(node_state_by_id.get("plant", {}).get("open_contracts", 0)),
            "active_contracts": int(node_state_by_id.get("plant", {}).get("active_contracts", 0)),
            "backlog_jobs": _overlay_display_int(backlog_jobs),
            "capacity_wip_pressure": "elevated" if _overlay_number(backlog_jobs) > 0 else "open",
            "schedule_risk": "n/a",
        }

    area_summaries = state_snapshot.get("area_summaries", {})
    if isinstance(area_summaries, Mapping):
        for area_id, summary in area_summaries.items():
            if not isinstance(summary, Mapping):
                continue
            node_id = f"area_{area_id}"
            active_jobs = summary.get("active_jobs")
            backlog_jobs = summary.get("backlog_jobs")
            pressure = "open"
            if _overlay_number(backlog_jobs) > 0 and _overlay_number(active_jobs) > 0:
                pressure = "elevated"
            elif _overlay_number(backlog_jobs) > 0:
                pressure = "queued"
            overlays[node_id] = {
                "open_contracts": int(node_state_by_id.get(node_id, {}).get("open_contracts", 0)),
                "active_contracts": int(node_state_by_id.get(node_id, {}).get("active_contracts", 0)),
                "active_jobs": _overlay_display_int(active_jobs),
                "backlog_jobs": _overlay_display_int(backlog_jobs),
                "capacity_wip_pressure": pressure,
                "schedule_risk": "n/a",
            }

    cell_summaries = state_snapshot.get("cell_summaries", {})
    if isinstance(cell_summaries, Mapping):
        for cell_id, summary in cell_summaries.items():
            if not isinstance(summary, Mapping):
                continue
            node_id = f"cell_{cell_id}"
            active_jobs = summary.get("active_jobs")
            ready_jobs = summary.get("ready_jobs")
            remaining_wip = summary.get("budget_remaining_wip")
            cap_wip = summary.get("budget_cap_wip")
            pressure = "open"
            if isinstance(remaining_wip, (int, float)) and float(remaining_wip) <= 0 and _overlay_number(active_jobs) > 0:
                pressure = "tight"
            elif (
                isinstance(cap_wip, (int, float))
                and _overlay_number(active_jobs) >= float(cap_wip)
                and float(cap_wip) > 0
            ):
                pressure = "tight"
            elif _overlay_number(ready_jobs) > 0 or _overlay_number(active_jobs) > 0:
                pressure = "elevated"
            overlays[node_id] = {
                "open_contracts": int(node_state_by_id.get(node_id, {}).get("open_contracts", 0)),
                "active_contracts": int(node_state_by_id.get(node_id, {}).get("active_contracts", 0)),
                "active_jobs": _overlay_display_int(active_jobs),
                "ready_jobs": _overlay_display_int(ready_jobs),
                "budget_remaining_wip": _overlay_display_int(remaining_wip),
                "budget_cap_wip": _overlay_display_int(cap_wip),
                "capacity_wip_pressure": pressure,
                "schedule_risk": "n/a",
            }

    return dict(sorted(overlays.items()))


def _overlay_number(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _overlay_display_int(value: object) -> str:
    if isinstance(value, bool):
        return "n/a"
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.1f}"
    return "n/a"


def _overlay_display_float(value: object) -> str:
    if isinstance(value, bool):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}"
    return "n/a"


def _edge_display_label(value: object) -> str:
    text = str(value or "")
    if not text:
        return "idle"
    display = {
        "assignment_request": "request",
        "assignment_contract": "contract",
        "accept_contract": "accept",
        "partially_completed": "partial",
    }.get(text, text)
    return display.replace("_", " ")


def _node_state_by_id(
    *,
    activation_counts: Mapping[str, int],
    contract_state_by_id: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    node_state: dict[str, dict[str, object]] = {}
    terminal_statuses = {"completed", "settled", "rejected", "cancelled", "failed"}
    active_statuses = {"active", "partially_completed"}

    def ensure(node_id: str) -> dict[str, object]:
        state = node_state.get(node_id)
        if state is None:
            state = {
                "activation_count": int(activation_counts.get(node_id, 0)),
                "contracts_seen": 0,
                "open_contracts": 0,
                "active_contracts": 0,
                "completed_contracts": 0,
                "completed_operations": 0.0,
                "total_operations": 0.0,
                "progress_ratio": None,
            }
            node_state[node_id] = state
        return state

    for node_id, count in activation_counts.items():
        ensure(node_id)["activation_count"] = int(count)

    for item in contract_state_by_id.values():
        participants = {
            str(item.get("sender", "")),
            str(item.get("recipient", "")),
        }
        status = str(item.get("status", ""))
        completed_operations = item.get("completed_operations", 0.0)
        total_operations = item.get("total_operations", 0.0)
        completed_value = float(completed_operations) if isinstance(completed_operations, (int, float)) else 0.0
        total_value = float(total_operations) if isinstance(total_operations, (int, float)) else 0.0
        for node_id in participants:
            if not node_id:
                continue
            state = ensure(node_id)
            state["contracts_seen"] = int(state["contracts_seen"]) + 1
            if status not in terminal_statuses:
                state["open_contracts"] = int(state["open_contracts"]) + 1
            if status in active_statuses:
                state["active_contracts"] = int(state["active_contracts"]) + 1
            if status in {"completed", "settled"}:
                state["completed_contracts"] = int(state["completed_contracts"]) + 1
            state["completed_operations"] = float(state["completed_operations"]) + completed_value
            state["total_operations"] = float(state["total_operations"]) + total_value

    for state in node_state.values():
        total_value = float(state["total_operations"])
        completed_value = float(state["completed_operations"])
        if total_value > 0:
            state["progress_ratio"] = min(1.0, completed_value / total_value)
        else:
            state["progress_ratio"] = None
        if state["completed_operations"].is_integer():
            state["completed_operations"] = int(state["completed_operations"])
        if state["total_operations"].is_integer():
            state["total_operations"] = int(state["total_operations"])

    return dict(sorted(node_state.items()))


def _node_role(node_id: str) -> str:
    if node_id == "plant":
        return "plant"
    if node_id.startswith("area_"):
        return "area"
    if node_id.startswith("cell_"):
        return "cell"
    return "other"


def _node_label(node_id: str) -> str:
    if node_id == "plant":
        return "Plant"
    if node_id.startswith("area_"):
        return f"Area {node_id.split('_', 1)[1]}"
    if node_id.startswith("cell_"):
        return f"Cell {node_id.split('_', 1)[1]}"
    return node_id


def _node_sort_key(node: Mapping[str, object]) -> tuple[int, int, str]:
    node_id = str(node["id"])
    role = str(node["role"])
    role_rank = {"plant": 0, "area": 1, "cell": 2}.get(role, 9)
    suffix = 0
    if "_" in node_id:
        try:
            suffix = int(node_id.split("_", 1)[1])
        except ValueError:
            suffix = 0
    return (role_rank, suffix, node_id)


def _render_html_report(payload: Mapping[str, object]) -> str:
    case_id = escape(str(payload["case_id"]))
    embedded_payload = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Topology Replay</title>
  <style>
    :root {{
      --bg: #efe7da;
      --panel: rgba(255, 252, 247, 0.94);
      --line: rgba(24, 29, 35, 0.12);
      --ink: #181d23;
      --muted: #5f6670;
      --warm: #bb5a2b;
      --cool: #1c6d7a;
      --accent: #234f98;
      --shadow: 0 22px 60px rgba(24, 29, 35, 0.12);
      --sans: "Aptos Display", "Segoe UI Variable Display", sans-serif;
      --mono: "Cascadia Code", Consolas, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: var(--sans);
      background:
        radial-gradient(circle at top left, rgba(187, 90, 43, 0.14), transparent 26%),
        radial-gradient(circle at top right, rgba(28, 109, 122, 0.14), transparent 24%),
        linear-gradient(180deg, var(--bg) 0%, #faf6ef 100%);
    }}
    .page {{
      width: min(1440px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 26px;
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 24px; }}
    .kicker {{
      color: var(--warm);
      text-transform: uppercase;
      letter-spacing: 0.09em;
      font-size: 12px;
    }}
    h1, h2, h3 {{ margin: 0; line-height: 1.08; }}
    h1 {{ font-size: clamp(34px, 5vw, 58px); margin-top: 10px; }}
    p {{ color: var(--muted); line-height: 1.7; }}
    code {{ font-family: var(--mono); color: var(--accent); }}
    .controls {{
      display: grid;
      grid-template-columns: 1.1fr 1.1fr auto auto auto 1fr;
      gap: 12px;
      margin-top: 18px;
      align-items: center;
    }}
    select, button, input[type="range"] {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.92);
      color: var(--ink);
      padding: 10px 12px;
      font: inherit;
    }}
    button {{
      cursor: pointer;
      font-weight: 700;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.7fr 1fr;
      gap: 18px;
      margin-top: 20px;
    }}
    .panel {{
      padding: 18px;
    }}
    .viewer {{
      min-height: 620px;
      position: relative;
      overflow: hidden;
    }}
    svg {{
      width: 100%;
      height: 100%;
      min-height: 560px;
      display: block;
    }}
    .edge {{
      stroke: rgba(35, 79, 152, 0.22);
      stroke-width: 4;
      transition: all 0.18s ease;
    }}
    .edge[data-state="open"] {{
      stroke: rgba(35, 79, 152, 0.48);
    }}
    .edge[data-state="active"] {{
      stroke: rgba(28, 109, 122, 0.62);
    }}
    .edge[data-state="constrained"] {{
      stroke: rgba(187, 90, 43, 0.76);
    }}
    .edge[data-state="complete"] {{
      stroke: rgba(24, 29, 35, 0.28);
      stroke-dasharray: 10 6;
    }}
    .edge.active {{
      stroke: var(--warm);
      stroke-width: 8;
    }}
    .edge.focus {{
      stroke: var(--accent);
      stroke-width: 7;
    }}
    .edge-count {{
      fill: var(--muted);
      font-size: 12px;
      font-family: var(--mono);
      text-anchor: middle;
    }}
    .edge-count[data-state="active"] {{
      fill: var(--cool);
    }}
    .edge-count[data-state="constrained"] {{
      fill: var(--warm);
      font-weight: 700;
    }}
    .edge-count[data-state="complete"] {{
      fill: rgba(24, 29, 35, 0.72);
    }}
    .node circle {{
      fill: rgba(255, 255, 255, 0.96);
      stroke: rgba(35, 79, 152, 0.18);
      stroke-width: 3;
      transition: all 0.18s ease;
    }}
    .node[data-state="backlog"] circle {{
      fill: rgba(35, 79, 152, 0.08);
      stroke: rgba(35, 79, 152, 0.46);
    }}
    .node[data-state="active"] circle {{
      fill: rgba(28, 109, 122, 0.08);
      stroke: rgba(28, 109, 122, 0.46);
    }}
    .node[data-state="busy"] circle {{
      fill: rgba(28, 109, 122, 0.12);
      stroke: rgba(28, 109, 122, 0.62);
    }}
    .node[data-state="ready"] circle {{
      fill: rgba(35, 79, 152, 0.12);
      stroke: rgba(35, 79, 152, 0.58);
    }}
    .node[data-state="down"] circle {{
      fill: rgba(187, 90, 43, 0.16);
      stroke: rgba(187, 90, 43, 0.88);
    }}
    .node[data-state="complete"] circle {{
      fill: rgba(255, 255, 255, 0.98);
      stroke: rgba(24, 29, 35, 0.24);
    }}
    .node.active circle {{
      fill: rgba(187, 90, 43, 0.12);
      stroke: var(--warm);
      stroke-width: 5;
      filter: drop-shadow(0 0 18px rgba(187, 90, 43, 0.28));
    }}
    .node text {{
      text-anchor: middle;
      fill: var(--ink);
    }}
    .node .label {{
      font-size: 15px;
      font-weight: 700;
    }}
    .node .meta {{
      font-size: 12px;
      fill: var(--muted);
      font-family: var(--mono);
    }}
    .cards {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .mini-cards {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 10px;
    }}
    .mini-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.86);
    }}
    .mini-card strong {{
      display: block;
      font-size: 22px;
      margin-top: 6px;
    }}
    .legend {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 10px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.82);
      color: var(--muted);
    }}
    .legend-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 2px solid rgba(24, 29, 35, 0.2);
      flex: 0 0 auto;
    }}
    .legend-swatch.backlog {{
      background: rgba(35, 79, 152, 0.12);
      border-color: rgba(35, 79, 152, 0.58);
    }}
    .legend-swatch.busy {{
      background: rgba(28, 109, 122, 0.14);
      border-color: rgba(28, 109, 122, 0.62);
    }}
    .legend-swatch.down {{
      background: rgba(187, 90, 43, 0.18);
      border-color: rgba(187, 90, 43, 0.88);
    }}
    .legend-swatch.complete {{
      background: rgba(255, 255, 255, 0.98);
      border-color: rgba(24, 29, 35, 0.24);
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.88);
    }}
    .card strong {{
      display: block;
      font-size: 26px;
      margin-top: 8px;
    }}
    .section {{
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.86);
    }}
    .list {{
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }}
    .event {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.82);
    }}
    .event strong {{
      color: var(--cool);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .empty {{
      color: var(--muted);
      font-style: italic;
    }}
    @media (max-width: 1120px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="kicker">Topology Replay</div>
      <h1>{case_id}</h1>
      <p>This replay viewer is generated only from benchmark trace artifacts. It replays controller activation, propagation, settlement, and world events over the topology graph without direct simulator access.</p>
      <div class="controls">
        <select id="policy-select"></select>
        <select id="contract-select"></select>
        <button id="prev-frame" type="button">Prev</button>
        <button id="play-pause" type="button">Play</button>
        <button id="next-frame" type="button">Next</button>
        <input id="frame-slider" type="range" min="0" max="0" value="0" />
      </div>
    </section>
    <section class="layout">
      <section class="panel viewer">
        <svg id="topology-svg" viewBox="0 0 960 560" preserveAspectRatio="xMidYMid meet"></svg>
      </section>
      <section class="panel">
        <div class="cards">
          <article class="card"><div class="kicker">Frame</div><strong id="frame-index">0 / 0</strong></article>
          <article class="card"><div class="kicker">Time</div><strong id="frame-time">0.000</strong></article>
          <article class="card"><div class="kicker">Active Agent</div><strong id="active-agent">plant</strong></article>
          <article class="card"><div class="kicker">Completed Contracts</div><strong id="completed-contracts">0</strong></article>
          <article class="card"><div class="kicker">Open Contracts</div><strong id="open-contracts">0</strong></article>
          <article class="card"><div class="kicker">Shared Backlog</div><strong id="shared-backlog">n/a</strong></article>
        </div>
        <div class="section">
          <div class="kicker">Policy Aggregate</div>
          <div id="policy-summary" class="mini-cards"></div>
        </div>
        <div class="section">
          <div class="kicker">Contract Lifecycle</div>
          <div id="contract-lifecycle" class="list"></div>
        </div>
        <div class="section">
          <div class="kicker">Overlay Legend</div>
          <div class="legend">
            <div class="legend-item"><span class="legend-swatch backlog"></span><span>backlog or release pressure</span></div>
            <div class="legend-item"><span class="legend-swatch busy"></span><span>busy or active processing</span></div>
            <div class="legend-item"><span class="legend-swatch down"></span><span>resource down or blocked</span></div>
            <div class="legend-item"><span class="legend-swatch complete"></span><span>idle or backlog cleared</span></div>
          </div>
        </div>
        <div class="section">
          <div class="kicker">Runtime Snapshot</div>
          <div id="runtime-snapshot" class="list"></div>
        </div>
        <div class="section">
          <div class="kicker">Machine State</div>
          <div id="machine-list" class="list"></div>
        </div>
        <div class="section">
          <div class="kicker">Activation</div>
          <div id="activation-summary" class="list"></div>
        </div>
        <div class="section">
          <div class="kicker">Messages</div>
          <div id="message-list" class="list"></div>
        </div>
        <div class="section">
          <div class="kicker">Settlement</div>
          <div id="settlement-list" class="list"></div>
        </div>
        <div class="section">
          <div class="kicker">World Events</div>
          <div id="world-list" class="list"></div>
        </div>
      </section>
    </section>
    <script id="replay-data" type="application/json">{escape(embedded_payload)}</script>
    <script>
      const payload = JSON.parse(document.getElementById("replay-data").textContent);
      const policySelect = document.getElementById("policy-select");
      const contractSelect = document.getElementById("contract-select");
      const slider = document.getElementById("frame-slider");
      const playPause = document.getElementById("play-pause");
      const prevButton = document.getElementById("prev-frame");
      const nextButton = document.getElementById("next-frame");
      const svg = document.getElementById("topology-svg");
      const frameIndexLabel = document.getElementById("frame-index");
      const frameTimeLabel = document.getElementById("frame-time");
      const activeAgentLabel = document.getElementById("active-agent");
      const completedContractsLabel = document.getElementById("completed-contracts");
      const openContractsLabel = document.getElementById("open-contracts");
      const sharedBacklogLabel = document.getElementById("shared-backlog");
      const policySummary = document.getElementById("policy-summary");
      const contractLifecycle = document.getElementById("contract-lifecycle");
      const runtimeSnapshot = document.getElementById("runtime-snapshot");
      const machineList = document.getElementById("machine-list");
      const activationSummary = document.getElementById("activation-summary");
      const messageList = document.getElementById("message-list");
      const settlementList = document.getElementById("settlement-list");
      const worldList = document.getElementById("world-list");

      let currentPolicy = payload.policies[0];
      let selectedContractId = "__auto__";
      let currentFrameIndex = 0;
      let playHandle = null;
      const nodeElements = new Map();
      const edgeElements = new Map();
      const edgeCountElements = new Map();
      const topology = payload.topology;

      function roleBuckets(nodes) {{
        return {{
          plant: nodes.filter((item) => item.role === "plant"),
          area: nodes.filter((item) => item.role === "area"),
          cell: nodes.filter((item) => item.role === "cell"),
          other: nodes.filter((item) => !["plant", "area", "cell"].includes(item.role)),
        }};
      }}

      function roleY(role) {{
        if (role === "plant") return 96;
        if (role === "area") return 260;
        if (role === "cell") return 430;
        return 510;
      }}

      function positionsFor(nodes) {{
        const width = 960;
        const buckets = roleBuckets(nodes);
        const positions = new Map();
        for (const [role, items] of Object.entries(buckets)) {{
          const y = roleY(role);
          const count = Math.max(1, items.length);
          items.forEach((item, index) => {{
            const x = count === 1 ? width / 2 : 120 + index * ((width - 240) / (count - 1));
            positions.set(item.id, {{ x, y }});
          }});
        }}
        return positions;
      }}

      const positions = positionsFor(topology.nodes);

      function createSvgElement(name, attrs) {{
        const element = document.createElementNS("http://www.w3.org/2000/svg", name);
        for (const [key, value] of Object.entries(attrs)) {{
          element.setAttribute(key, String(value));
        }}
        return element;
      }}

      function buildGraph() {{
        svg.innerHTML = "";

        topology.edges.forEach((edge) => {{
          const source = positions.get(edge.source);
          const target = positions.get(edge.target);
          if (!source || !target) return;
          const line = createSvgElement("line", {{
            x1: source.x,
            y1: source.y,
            x2: target.x,
            y2: target.y,
            class: "edge",
          }});
          svg.appendChild(line);
          edgeElements.set(edge.id, line);

          const count = createSvgElement("text", {{
            x: (source.x + target.x) / 2,
            y: (source.y + target.y) / 2 - 8,
            class: "edge-count",
          }});
          count.textContent = "0";
          svg.appendChild(count);
          edgeCountElements.set(edge.id, count);
        }});

        topology.nodes.forEach((node) => {{
          const pos = positions.get(node.id);
          if (!pos) return;
          const group = createSvgElement("g", {{
            class: "node",
            transform: `translate(${{pos.x}} ${{pos.y}})`,
            "data-state": "neutral",
          }});
          group.appendChild(createSvgElement("circle", {{ r: 56 }}));

          const label = createSvgElement("text", {{ y: -16, class: "label" }});
          label.textContent = node.label;
          group.appendChild(label);

          const metaTop = createSvgElement("text", {{ y: 4, class: "meta" }});
          metaTop.textContent = "act: 0";
          group.appendChild(metaTop);

          const metaMiddle = createSvgElement("text", {{ y: 20, class: "meta" }});
          metaMiddle.textContent = "load: 0";
          group.appendChild(metaMiddle);

          const metaBottom = createSvgElement("text", {{ y: 36, class: "meta" }});
          metaBottom.textContent = "load: 0";
          group.appendChild(metaBottom);

          svg.appendChild(group);
          nodeElements.set(node.id, {{ group, metaTop, metaMiddle, metaBottom }});
        }});
      }}

      function frameData() {{
        return payload.replay_by_policy[currentPolicy];
      }}

      function frames() {{
        return frameData().frames;
      }}

      function currentContractIndex() {{
        return frameData().contract_index || {{}};
      }}

      function populateContractSelector() {{
        contractSelect.innerHTML = "";
        const autoOption = document.createElement("option");
        autoOption.value = "__auto__";
        autoOption.textContent = "Contract Focus";
        contractSelect.appendChild(autoOption);

        Object.keys(currentContractIndex()).sort().forEach((contractId) => {{
          const option = document.createElement("option");
          option.value = contractId;
          option.textContent = `Focused Contract: ${{contractId}}`;
          contractSelect.appendChild(option);
        }});

        if (!currentContractIndex()[selectedContractId]) {{
          selectedContractId = "__auto__";
        }}
        contractSelect.value = selectedContractId;
      }}

      function stopPlayback() {{
        if (playHandle !== null) {{
          window.clearInterval(playHandle);
          playHandle = null;
        }}
        playPause.textContent = "Play";
      }}

      function startPlayback() {{
        stopPlayback();
        playHandle = window.setInterval(() => {{
          const next = currentFrameIndex + 1;
          if (next >= frames().length) {{
            stopPlayback();
            return;
          }}
          setFrame(next);
        }}, 900);
        playPause.textContent = "Pause";
      }}

      function setFrame(index) {{
        const maxIndex = Math.max(0, frames().length - 1);
        currentFrameIndex = Math.max(0, Math.min(index, maxIndex));
        slider.value = String(currentFrameIndex);
        renderFrame();
      }}

      function renderList(container, items, renderer, emptyText = "No events on this frame.") {{
        container.innerHTML = "";
        if (!items.length) {{
          const empty = document.createElement("div");
          empty.className = "empty";
          empty.textContent = emptyText;
          container.appendChild(empty);
          return;
        }}
        items.forEach((item) => {{
          const block = document.createElement("article");
          block.className = "event";
          block.innerHTML = renderer(item);
          container.appendChild(block);
        }});
      }}

      function renderFrame() {{
        const frame = frames()[currentFrameIndex];
        const replay = frameData();
        const stateSnapshot = frame.state_snapshot || {{}};
        const focusContractId =
          selectedContractId === "__auto__"
            ? frame.focus_contract_id
            : selectedContractId;
        const contractFocus =
          selectedContractId === "__auto__"
            ? (frame.contract_focus || null)
            : (focusContractId ? {{
                ...(currentContractIndex()[focusContractId] || {{}}),
                contract_id: focusContractId,
                capacity_wip_pressure:
                  frame.node_constraint_overlay?.[currentContractIndex()[focusContractId]?.recipient]?.capacity_wip_pressure
                  ?? "n/a",
              }} : null);
        const contractEvents = focusContractId
          ? [
              ...(frame.contract_events || []).filter((item) => item.contract_id === focusContractId),
            ]
          : [];
        frameIndexLabel.textContent = `${{currentFrameIndex + 1}} / ${{frames().length}}`;
        frameTimeLabel.textContent = Number(frame.timestamp).toFixed(3);
        activeAgentLabel.textContent = frame.active_agent;
        completedContractsLabel.textContent = String(frame.cumulative.completed_contracts);
        openContractsLabel.textContent = String(frame.cumulative.open_contract_count || 0);
        sharedBacklogLabel.textContent =
          frame.cumulative.shared_backlog_remaining === null || frame.cumulative.shared_backlog_remaining === undefined
            ? "n/a"
            : String(frame.cumulative.shared_backlog_remaining);

        const aggregateCards = [
          ["Makespan", replay.policy_summary?.makespan_mean],
          ["Tardiness", replay.policy_summary?.tardiness_mean],
          ["Energy", replay.policy_summary?.energy_mean],
          ["Carbon", replay.policy_summary?.carbon_mean],
          ["Episode Reward", replay.policy_episode?.total_reward],
          ["Down Machines", replay.policy_episode?.down_machines_total],
        ];
        policySummary.innerHTML = "";
        aggregateCards.forEach(([label, value]) => {{
          const card = document.createElement("article");
          card.className = "mini-card";
          const numericValue = typeof value === "number" ? Number(value).toFixed(3) : (value ?? "n/a");
          card.innerHTML = `<div class="kicker">${{label}}</div><strong>${{numericValue}}</strong>`;
          policySummary.appendChild(card);
        }});

        nodeElements.forEach((item, nodeId) => {{
          item.group.classList.toggle("active", frame.highlighted_nodes.includes(nodeId));
          const isFocusNode =
            contractFocus
            && [contractFocus.sender, contractFocus.recipient, contractFocus.accepted_by].includes(nodeId);
          item.group.classList.toggle("contract-focus", Boolean(isFocusNode));
          const overlay = (frame.graph_overlay_by_node || {{}})[nodeId] || null;
          item.group.setAttribute("data-state", overlay?.status || "neutral");
          if (overlay) {{
            item.metaTop.textContent = overlay.headline || "n/a";
            item.metaMiddle.textContent = overlay.detail || "n/a";
            item.metaBottom.textContent = overlay.footer || "";
            return;
          }}
          const nodeState = frame.cumulative.node_state_by_id[nodeId] || {{}};
          const count = frame.cumulative.activation_counts_by_node[nodeId] || 0;
          if (nodeId === "plant") {{
            const plantState = stateSnapshot.plant || {{}};
            item.metaTop.textContent = `act: ${{count}} | backlog: ${{plantState.backlog_jobs_total ?? "n/a"}}`;
            item.metaMiddle.textContent = `busy: ${{plantState.busy_machines_total ?? "n/a"}} | down: ${{plantState.down_machines_total ?? "n/a"}}`;
            item.metaBottom.textContent = "";
            return;
          }}

          if (nodeId.startsWith("area_")) {{
            const areaState = (stateSnapshot.area_summaries || {{}})[nodeId.split("_", 2)[1]] || {{}};
            const jobs = areaState.active_jobs ?? nodeState.active_contracts ?? 0;
            const energy = typeof areaState.energy_kwh === "number" ? areaState.energy_kwh.toFixed(1) : "n/a";
            item.metaTop.textContent = `act: ${{count}} | jobs: ${{jobs}}`;
            item.metaMiddle.textContent = `energy: ${{energy}}`;
            item.metaBottom.textContent = "";
            return;
          }}

          if (nodeId.startsWith("cell_")) {{
            const cellState = (stateSnapshot.cell_summaries || {{}})[nodeId.split("_", 2)[1]] || {{}};
            const jobs = cellState.active_jobs ?? nodeState.active_contracts ?? 0;
            const busy = cellState.busy_machines ?? "n/a";
            const down = cellState.down_machines ?? "n/a";
            item.metaTop.textContent = `act: ${{count}} | jobs: ${{jobs}}`;
            item.metaMiddle.textContent = `busy: ${{busy}} | down: ${{down}}`;
            item.metaBottom.textContent = "";
            return;
          }}

          const open = nodeState.open_contracts || 0;
          const progress = typeof nodeState.progress_ratio === "number"
            ? `${{Math.round(nodeState.progress_ratio * 100)}}%`
            : "n/a";
          item.metaTop.textContent = `act: ${{count}} | open: ${{open}}`;
          item.metaMiddle.textContent = `load: ${{nodeState.active_contracts || 0}} | prog: ${{progress}}`;
          item.metaBottom.textContent = "";
        }});
        edgeElements.forEach((line, edgeId) => {{
          line.classList.toggle("active", frame.highlighted_edges.includes(edgeId));
          line.classList.toggle(
            "focus",
            Boolean(contractEvents.find((item) => item.edge_id === edgeId)),
          );
          const overlay = (frame.graph_overlay_by_edge || {{}})[edgeId] || null;
          line.setAttribute("data-state", overlay?.status || "neutral");
          const count = frame.cumulative.message_edge_counts[edgeId] || 0;
          const text = edgeCountElements.get(edgeId);
          if (text) {{
            text.setAttribute("data-state", overlay?.status || "neutral");
            text.textContent = overlay?.badge || String(count);
          }}
        }});

        renderList(
          contractLifecycle,
          contractFocus ? [contractFocus] : [],
          (item) => `
            <strong>${{item.contract_id || "Focused Contract"}}</strong>
            <div>route: <code>${{item.sender || "n/a"}}</code> -> <code>${{item.recipient || "n/a"}}</code></div>
            <div>status: <code>${{item.status || item.latest_status || "n/a"}}</code> | accepted_by: <code>${{item.accepted_by || "n/a"}}</code></div>
            <div>work/job: <code>${{item.work_id || "n/a"}}</code> / <code>${{item.job_id ?? "n/a"}}</code></div>
            <div>ops: <code>${{item.completed_operations ?? 0}}</code> / <code>${{item.total_operations ?? "n/a"}}</code></div>
            <div>Schedule Risk: <code>${{item.schedule_risk || "n/a"}}</code></div>
            <div>Capacity / WIP: <code>${{item.capacity_wip_pressure || "n/a"}}</code></div>
          `,
          "No focused contract on this frame.",
        );

        const runtimeItems = [];
        if (stateSnapshot.plant) {{
          runtimeItems.push({{
            title: "plant",
            lines: [
              `backlog: ${{stateSnapshot.plant.backlog_jobs_total ?? "n/a"}}`,
              `busy/down: ${{stateSnapshot.plant.busy_machines_total ?? "n/a"}} / ${{stateSnapshot.plant.down_machines_total ?? "n/a"}}`,
            ],
          }});
        }}
        Object.entries(stateSnapshot.area_summaries || {{}}).forEach(([areaId, summary]) => {{
          runtimeItems.push({{
            title: `area_${{areaId}}`,
            lines: [
              `jobs: ${{summary.active_jobs ?? "n/a"}}`,
              `energy/carbon: ${{summary.energy_kwh ?? "n/a"}} / ${{summary.carbon_kg ?? "n/a"}}`,
            ],
          }});
        }});
        Object.entries(stateSnapshot.cell_summaries || {{}}).forEach(([cellId, summary]) => {{
          runtimeItems.push({{
            title: `cell_${{cellId}}`,
            lines: [
              `jobs: ${{summary.active_jobs ?? "n/a"}} | ready: ${{summary.ready_jobs ?? "n/a"}}`,
              `busy/down: ${{summary.busy_machines ?? "n/a"}} / ${{summary.down_machines ?? "n/a"}}`,
            ],
          }});
        }});
        renderList(
          runtimeSnapshot,
          runtimeItems,
          (item) => `
            <strong>${{item.title}}</strong>
            <div>${{item.lines[0]}}</div>
            <div>${{item.lines[1]}}</div>
          `,
          "No public runtime snapshot.",
        );
        const machineItems = Object.values(stateSnapshot.machine_summaries || {{}})
          .filter((item) => item && typeof item === "object")
          .sort((left, right) => Number(left.machine_id ?? 0) - Number(right.machine_id ?? 0));
        renderList(
          machineList,
          machineItems,
          (item) => `
            <strong>machine_${{item.machine_id ?? "?"}}</strong>
            <div>cell/status: ${{item.cell_id ?? "n/a"}} / <code>${{item.status ?? "n/a"}}</code></div>
            <div>job/remaining: ${{item.current_job ?? "idle"}} / ${{typeof item.remaining_time === "number" ? Number(item.remaining_time).toFixed(3) : "n/a"}}</div>
          `,
          "No public machine state.",
        );
        renderList(
          activationSummary,
          [{{
            active_agent: frame.active_agent,
            activation_cause: frame.activation_cause,
            functional_loop: frame.functional_loop,
            event_summary: frame.event_summary,
          }}],
          (item) => `
            <strong>${{item.active_agent}}</strong>
            <div>cause: <code>${{item.activation_cause}}</code></div>
            <div>loop: <code>${{item.functional_loop}}</code></div>
            <div>events: <code>${{JSON.stringify(item.event_summary)}}</code></div>
          `,
        );
        renderList(
          messageList,
          focusContractId
            ? frame.message_events.filter((item) => {{
                const contractId = item.contract_id || item.upstream_contract_id;
                return contractId === focusContractId;
              }})
            : frame.message_events,
          (item) => `
            <strong>${{item.kind}}</strong>
            <div><code>${{item.sender}}</code> -> <code>${{item.recipient}}</code></div>
            <div>contract: <code>${{item.contract_id || item.upstream_contract_id || "n/a"}}</code></div>
          `,
        );
        renderList(
          settlementList,
          focusContractId
            ? frame.settlement_events.filter((item) => item.contract_id === focusContractId)
            : frame.settlement_events,
          (item) => `
            <strong>${{item.status}}</strong>
            <div><code>${{item.sender}}</code> -> <code>${{item.recipient}}</code></div>
            <div>contract: <code>${{item.contract_id}}</code></div>
          `,
        );
        renderList(
          worldList,
          frame.world_events,
          (item) => `
            <strong>${{item.kind}}</strong>
            <div>scope: <code>${{item.scope}}</code></div>
            <div>time: <code>${{Number(item.time).toFixed(3)}}</code></div>
          `,
        );
      }}

      payload.policies.forEach((policy) => {{
        const option = document.createElement("option");
        option.value = policy;
        option.textContent = policy;
        policySelect.appendChild(option);
      }});

      policySelect.addEventListener("change", () => {{
        currentPolicy = policySelect.value;
        selectedContractId = "__auto__";
        stopPlayback();
        populateContractSelector();
        slider.max = String(Math.max(0, frames().length - 1));
        setFrame(0);
      }});
      contractSelect.addEventListener("change", () => {{
        selectedContractId = contractSelect.value;
        stopPlayback();
        renderFrame();
      }});
      slider.addEventListener("input", () => {{
        stopPlayback();
        setFrame(Number(slider.value));
      }});
      prevButton.addEventListener("click", () => {{
        stopPlayback();
        setFrame(currentFrameIndex - 1);
      }});
      nextButton.addEventListener("click", () => {{
        stopPlayback();
        setFrame(currentFrameIndex + 1);
      }});
      playPause.addEventListener("click", () => {{
        if (playHandle === null) {{
          startPlayback();
        }} else {{
          stopPlayback();
        }}
      }});

      buildGraph();
      populateContractSelector();
      slider.max = String(Math.max(0, frames().length - 1));
      setFrame(0);
    </script>
  </main>
</body>
</html>
"""
