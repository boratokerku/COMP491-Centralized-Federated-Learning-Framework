"""Machine-readable run summary (`summary.json`).

Pure function of `RunData`: the same run always gives the same summary, and nothing here
reads files, the clock or the network.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from cfl import __version__
from cfl.coordinator.state_store import MetricRecord, ParticipantStatus
from cfl.reporting.data import RunData

SUMMARY_VERSION = 1

# planning/interfaces.md section 4
COORDINATOR_TIMING = (
    "load_ms",
    "aggregation_ms",
    "evaluation_ms",
    "write_ms",
    "waiting_ms",
    "round_duration_ms",
)
COORDINATOR_COUNTS = (
    "updates_received",
    "updates_used",
    "updates_rejected",
    "total_samples",
    "bytes_read",
    "model_bytes",
)
INCIDENT_EVENTS = (
    "CLIENT_DROPOUT",
    "UPDATE_REJECTED",
    "UPDATE_EXCLUDED",
    "UPDATE_RETRY_ACKNOWLEDGED",
    "ROUND_FAILED",
    "COORDINATOR_RESUMED",
    "RUN_FAILED",
)


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _coordinator_by_round(metrics: list[MetricRecord]) -> dict[int, dict[str, float]]:
    """Per-round values written by the coordinator (no client id, no message type)."""
    out: dict[int, dict[str, float]] = defaultdict(dict)
    for m in metrics:
        if m.round_id is not None and m.client_id is None and m.message_type is None:
            out[m.round_id][m.metric_name] = m.value
    return out


def _unit(metrics: list[MetricRecord], name: str) -> str:
    return next((m.unit for m in metrics if m.metric_name == name), "")


def build_summary(data: RunData) -> dict[str, Any]:
    run, metrics = data.run, data.metrics
    coordinator = _coordinator_by_round(metrics)

    rounds = []
    for r in data.rounds:
        parts = data.participants.get(r.round_id, [])
        values = coordinator.get(r.round_id, {})
        rounds.append(
            {
                "round_id": r.round_id,
                "state": r.state.value,
                "failure_reason": r.failure_reason,
                "opened_at": iso(r.opened_at),
                "closed_at": iso(r.closed_at),
                "deadline_at": iso(r.deadline_at),
                "min_participants": r.min_participants,
                "input_model_version": r.input_model_version,
                "output_model_version": r.output_model_version,
                "output_model_sha256": r.output_model_sha256,
                "participants": [
                    {
                        "client_id": p.client_id,
                        "status": p.status.value,
                        "reason_code": p.reason_code,
                    }
                    for p in sorted(parts, key=lambda p: p.client_id)
                ],
                "global_metrics": {k: v for k, v in values.items() if k.startswith("global_")},
                "timing_ms": {k: values[k] for k in COORDINATOR_TIMING if k in values},
                "counts": {k: values[k] for k in COORDINATOR_COUNTS if k in values},
            }
        )

    clients: dict[str, dict[str, Any]] = {}
    for r in data.rounds:
        for p in data.participants.get(r.round_id, []):
            c = clients.setdefault(
                p.client_id,
                {
                    "rounds_selected": 0,
                    "updates_accepted": 0,
                    "dropouts": 0,
                    "rejected": 0,
                    "samples_per_round": {},
                    "metrics": {},
                },
            )
            c["rounds_selected"] += 1
            c["updates_accepted"] += p.status is ParticipantStatus.UPDATE_ACCEPTED
            c["dropouts"] += p.status is ParticipantStatus.DROPOUT
            c["rejected"] += p.status is ParticipantStatus.REJECTED
        for u in data.updates.get(r.round_id, []):
            c = clients.setdefault(u.client_id, {"samples_per_round": {}, "metrics": {}})
            c["samples_per_round"][str(r.round_id)] = u.num_samples
    for c in clients.values():
        c["total_samples"] = sum(c["samples_per_round"].values())
    per_client: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for m in metrics:
        if m.client_id is not None and m.message_type is None:
            per_client[m.client_id][m.metric_name].append(m.value)
    for client_id, named in per_client.items():
        clients.setdefault(client_id, {"samples_per_round": {}, "metrics": {}, "total_samples": 0})
        clients[client_id]["metrics"] = {name: stats(v) for name, v in sorted(named.items())}

    grouped: dict[str, list[float]] = defaultdict(list)
    for m in metrics:
        if m.client_id is None and m.message_type is None:
            grouped[m.metric_name].append(m.value)
        elif m.client_id is not None and m.message_type is None:
            grouped[f"all_clients.{m.metric_name}"].append(m.value)
    metric_summaries = {
        name: {"unit": _unit(metrics, name.removeprefix("all_clients.")), **stats(v)}
        for name, v in sorted(grouped.items())
    }

    communication: dict[str, dict[str, Any]] = {}
    by_message: dict[tuple[str, str], list[float]] = defaultdict(list)
    for m in metrics:
        if m.message_type is not None:
            by_message[(m.message_type, m.metric_name)].append(m.value)
    for (message_type, name), v in sorted(by_message.items()):
        communication.setdefault(message_type, {})[name] = {
            "unit": _unit(metrics, name),
            **stats(v),
        }

    bytes_by_message: dict[str, float] = defaultdict(float)
    for m in metrics:
        if m.metric_name == "message_bytes" and m.message_type is not None:
            bytes_by_message[m.message_type] += m.value

    failed = [r for r in data.rounds if r.state.value == "FAILED"]
    done = [r for r in data.rounds if r.state.value == "DONE"]
    event_counts: dict[str, int] = defaultdict(int)
    for e in data.events:
        event_counts[e.event_type] += 1
    incidents = [
        {
            "event_type": e.event_type,
            "round_id": e.round_id,
            "client_id": e.client_id,
            "occurred_at": iso(e.created_at),
            "details": e.payload,
        }
        for e in data.events
        if e.event_type in INCIDENT_EVENTS
    ]

    duration = (
        (run.completed_at - run.started_at).total_seconds()
        if run.started_at and run.completed_at
        else None
    )
    return {
        "summary_version": SUMMARY_VERSION,
        "run": {
            "run_id": run.run_id,
            "name": run.name,
            "status": run.status.value,
            "failure_reason": run.failure_reason,
            "strategy": run.strategy,
            "task": run.task_name,
            "seed": run.seed,
            "created_at": iso(run.created_at),
            "started_at": iso(run.started_at),
            "completed_at": iso(run.completed_at),
            "duration_s": duration,
        },
        "versions": {
            "software": __version__,
            "protocol": run.protocol_version,
            "manifest": run.manifest_version,
            "manifest_sha256": run.manifest_sha256,
            "preprocessing": run.preprocessing_version,
            "model_architecture": run.model_architecture,
            "model_architecture_version": run.model_architecture_version,
            "model_fingerprint": run.model_fingerprint,
        },
        "config": run.config,
        "config_sha256": run.config_sha256,
        "rounds_summary": {
            "attempted": len(data.rounds),
            "completed": len(done),
            "failed": len(failed),
            "failures": [{"round_id": r.round_id, "reason": r.failure_reason} for r in failed],
        },
        "rounds": rounds,
        "clients": dict(sorted(clients.items())),
        "metric_summaries": metric_summaries,
        "communication": communication,
        "communication_totals": {
            "bytes": sum(bytes_by_message.values()),
            "bytes_by_message_type": dict(sorted(bytes_by_message.items())),
        },
        "reliability": {
            "dropouts": event_counts.get("CLIENT_DROPOUT", 0),
            "retries": sum(m.value for m in metrics if m.metric_name == "client_retry_count"),
            "retried_uploads": event_counts.get("UPDATE_RETRY_ACKNOWLEDGED", 0),
            "rejected_updates": event_counts.get("UPDATE_REJECTED", 0),
            "excluded_updates": event_counts.get("UPDATE_EXCLUDED", 0),
            "failed_rounds": len(failed),
            "coordinator_resumes": event_counts.get("COORDINATOR_RESUMED", 0),
        },
        "event_counts": dict(sorted(event_counts.items())),
        "incidents": incidents,
        "final_model": {
            "version": run.model_version,
            "path": run.final_model_path,
            "sha256": run.final_model_sha256,
            "format": run.final_model_format,
            "bytes": run.final_model_bytes,
        },
    }
