"""PNG plots of the principal ML and system metrics. A plot with no data is skipped."""

from __future__ import annotations

import io
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # no display needed; also the only backend that is safe in containers
import matplotlib.pyplot as plt  # noqa: E402

from cfl.reporting.data import RunData  # noqa: E402

GLOBAL = (("global_f1", "F1"), ("global_roc_auc", "ROC-AUC"), ("global_pr_auc", "PR-AUC"))
TIMING = (
    ("waiting_ms", "waiting for clients"),
    ("load_ms", "load"),
    ("aggregation_ms", "aggregation"),
    ("evaluation_ms", "evaluation"),
    ("write_ms", "write"),
)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _series(data: RunData, name: str, *, client: str | None = None) -> dict[int, float]:
    return {
        m.round_id: m.value
        for m in data.metrics
        if m.metric_name == name
        and m.round_id is not None
        and m.client_id == client
        and m.message_type is None
    }


def _global_metrics(data: RunData) -> bytes | None:
    fig, ax = plt.subplots(figsize=(7, 4))
    drawn = False
    for name, label in GLOBAL:
        series = _series(data, name)
        if series:
            ax.plot(sorted(series), [series[r] for r in sorted(series)], marker="o", label=label)
            drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set(xlabel="Round", ylabel="Score", title="Global model on the held-out test set")
    ax.set_xticks(sorted({r.round_id for r in data.rounds}))
    ax.grid(alpha=0.3)
    ax.legend()
    return _png(fig)


def _client_before_after(data: RunData) -> bytes | None:
    clients = sorted({m.client_id for m in data.metrics if m.client_id})
    if not clients:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    drawn = False
    for i, client in enumerate(clients):
        color = f"C{i % 10}"
        for suffix, style in (("before", "--"), ("after", "-")):
            series = _series(data, f"client_val_f1_{suffix}", client=client)
            if series:
                ax.plot(
                    sorted(series),
                    [series[r] for r in sorted(series)],
                    style,
                    color=color,
                    marker="o",
                    label=f"{client} {suffix}",
                )
                drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set(
        xlabel="Round",
        ylabel="Validation F1",
        title="Per client: before (dashed) and after (solid) local training",
    )
    ax.set_xticks(sorted({r.round_id for r in data.rounds}))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    return _png(fig)


def _round_time(data: RunData) -> bytes | None:
    rounds = sorted({r.round_id for r in data.rounds})
    parts = [(label, _series(data, name)) for name, label in TIMING]
    parts = [(label, s) for label, s in parts if s]
    if not rounds or not parts:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = [0.0] * len(rounds)
    for label, series in parts:
        heights = [series.get(r, 0.0) for r in rounds]
        ax.bar([str(r) for r in rounds], heights, bottom=bottom, label=label)
        bottom = [b + h for b, h in zip(bottom, heights, strict=True)]
    ax.set(xlabel="Round", ylabel="Milliseconds", title="Round duration breakdown")
    ax.legend(fontsize=8)
    return _png(fig)


def _transfer_bytes(data: RunData) -> bytes | None:
    model = _series(data, "model_bytes")
    uploaded: dict[int, int] = defaultdict(int)
    for round_id, updates in data.updates.items():
        for u in updates:
            uploaded[round_id] += u.payload_bytes
    rounds = sorted(set(model) | set(uploaded))
    if not rounds:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.4
    xs = list(range(len(rounds)))
    ax.bar(
        [x - width / 2 for x in xs],
        [model.get(r, 0) / 1e6 for r in rounds],
        width,
        label="global model (per client download)",
    )
    ax.bar(
        [x + width / 2 for x in xs],
        [uploaded.get(r, 0) / 1e6 for r in rounds],
        width,
        label="client updates received (sum)",
    )
    ax.set_xticks(xs, [str(r) for r in rounds])
    ax.set(xlabel="Round", ylabel="MB", title="Transfer sizes")
    ax.legend(fontsize=8)
    return _png(fig)


def render_plots(data: RunData) -> dict[str, bytes]:
    plots = {
        "global_metrics.png": _global_metrics(data),
        "client_before_after.png": _client_before_after(data),
        "round_time_breakdown.png": _round_time(data),
        "transfer_bytes.png": _transfer_bytes(data),
    }
    return {name: png for name, png in plots.items() if png is not None}
