"""Write the end-of-run report into the model store under `<run_id>/report/`."""

from __future__ import annotations

import csv
import io
import json

from cfl.coordinator.state_store import StateStore
from cfl.core.store import ModelStore
from cfl.reporting.data import RunData, load_run_data
from cfl.reporting.markdown import render_markdown
from cfl.reporting.plots import render_plots
from cfl.reporting.summary import build_summary

CSV_COLUMNS = (
    "run_id",
    "round_id",
    "client_id",
    "metric_name",
    "value",
    "unit",
    "message_type",
    "recorded_at",
)


def metrics_csv(data: RunData) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for m in sorted(
        data.metrics,
        key=lambda m: (m.round_id or 0, m.metric_name, m.client_id or "", m.recorded_at),
    ):
        writer.writerow(
            [
                m.run_id,
                m.round_id if m.round_id is not None else "",
                m.client_id or "",
                m.metric_name,
                repr(m.value),
                m.unit,
                m.message_type or "",
                m.recorded_at.isoformat(),
            ]
        )
    return buf.getvalue()


def generate_report(state: StateStore, models: ModelStore, run_id: str) -> list[str]:
    """Write summary.json, report.md, metrics.csv and plots/*.png; return the store keys.

    Safe to repeat: every file is replaced atomically, so the website never reads a
    half-written report. Raises KeyError if the run does not exist.
    """
    data = load_run_data(state, run_id)
    summary = build_summary(data)
    plots = render_plots(data)
    base = f"{run_id}/report"

    files: dict[str, bytes] = {
        f"{base}/summary.json": (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
        f"{base}/report.md": (render_markdown(summary, sorted(plots)) + "\n").encode(),
        f"{base}/metrics.csv": metrics_csv(data).encode(),
    }
    files.update({f"{base}/plots/{name}": png for name, png in plots.items()})
    for key, content in files.items():
        models.put(key, content)
    return list(files)
