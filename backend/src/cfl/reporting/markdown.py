"""Human-readable report (`report.md`), rendered from the summary dictionary only."""

from __future__ import annotations

import json
from typing import Any

GLOBAL_COLUMNS = ("global_f1", "global_roc_auc", "global_pr_auc", "global_accuracy", "global_loss")


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return f"{int(value):,}"
    return f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines + [""]


def _stat_rows(summaries: dict[str, dict[str, Any]], names: list[str]) -> list[list[str]]:
    return [
        [
            f"`{n}`",
            s["unit"] or "-",
            _num(s["count"]),
            _num(s["min"]),
            _num(s["max"]),
            _num(s["mean"]),
        ]
        for n in names
        if (s := summaries.get(n))
    ]


def render_markdown(summary: dict[str, Any], plots: list[str]) -> str:
    run, versions = summary["run"], summary["versions"]
    rs = summary["rounds_summary"]
    out = [f"# Run report: {run['name']}", ""]
    out += _table(
        ["Field", "Value"],
        [
            ["Run ID", f"`{run['run_id']}`"],
            ["Status", run["status"]],
            ["Strategy", run["strategy"]],
            ["Task", run["task"]],
            ["Seed", str(run["seed"])],
            ["Created", run["created_at"] or "-"],
            ["Started", run["started_at"] or "-"],
            ["Completed", run["completed_at"] or "-"],
            ["Duration", f"{run['duration_s']:.1f} s" if run["duration_s"] is not None else "-"],
        ],
    )
    if run["failure_reason"]:
        out += [f"**Run failure reason:** {run['failure_reason']}", ""]

    out += ["## Versions", ""]
    out += _table(
        ["Component", "Version"], [[k.replace("_", " "), f"`{v}`"] for k, v in versions.items()]
    )

    out += ["## Configuration", "", f"SHA-256: `{summary['config_sha256']}`", "", "```json"]
    out += [json.dumps(summary["config"], indent=2, sort_keys=True), "```", ""]

    out += ["## Rounds", ""]
    out += [
        f"Attempted: **{rs['attempted']}**, completed: **{rs['completed']}**, "
        f"failed: **{rs['failed']}**.",
        "",
    ]
    if rs["failures"]:
        out += _table(
            ["Round", "Reason"], [[str(f["round_id"]), f["reason"] or "-"] for f in rs["failures"]]
        )
    rows = []
    for r in summary["rounds"]:
        used = sum(p["status"] == "UPDATE_ACCEPTED" for p in r["participants"])
        lost = [p["client_id"] for p in r["participants"] if p["status"] in ("DROPOUT", "REJECTED")]
        rows.append(
            [
                str(r["round_id"]),
                r["state"],
                f"{used}/{len(r['participants'])}",
                ", ".join(lost) or "-",
                _num(r["timing_ms"].get("round_duration_ms")),
                f"v{r['output_model_version']}" if r["output_model_version"] is not None else "-",
            ]
        )
    out += _table(
        ["Round", "State", "Updates used", "Dropout / rejected", "Duration (ms)", "Model"], rows
    )

    out += ["## Clients", ""]
    rows = [
        [
            f"`{cid}`",
            _num(c.get("rounds_selected", 0)),
            _num(c.get("updates_accepted", 0)),
            _num(c.get("dropouts", 0)),
            _num(c.get("rejected", 0)),
            _num(c.get("total_samples", 0)),
            ", ".join(f"{k}: {_num(v)}" for k, v in c["samples_per_round"].items()) or "-",
        ]
        for cid, c in summary["clients"].items()
    ]
    out += _table(
        [
            "Client",
            "Selected",
            "Accepted",
            "Dropouts",
            "Rejected",
            "Samples (sum over rounds)",
            "Samples per round",
        ],
        rows,
    )

    out += ["## Global metrics per round", ""]
    cols = [c for c in GLOBAL_COLUMNS if any(c in r["global_metrics"] for r in summary["rounds"])]
    rows = [
        [str(r["round_id"])] + [_num(r["global_metrics"].get(c)) for c in cols]
        for r in summary["rounds"]
        if r["global_metrics"]
    ]
    out += _table(["Round", *cols], rows) if rows else ["No global metrics recorded.", ""]

    ms = summary["metric_summaries"]
    header = ["Metric", "Unit", "Count", "Min", "Max", "Mean"]
    out += ["## Metric summaries", "", "### Global model (held-out test set)", ""]
    out += _table(header, _stat_rows(ms, [n for n in ms if n.startswith("global_")]))
    out += ["### Clients (all clients and rounds)", ""]
    out += _table(header, _stat_rows(ms, [n for n in ms if n.startswith("all_clients.")]))
    out += ["### Coordinator timing and sizes", ""]
    coordinator = [n for n in ms if not n.startswith(("global_", "all_clients."))]
    out += _table(header, _stat_rows(ms, coordinator))

    out += ["## Communication", ""]
    if summary["communication"]:
        rows = [
            [
                f"`{mt}`",
                f"`{n}`",
                s["unit"] or "-",
                _num(s["count"]),
                _num(s["min"]),
                _num(s["max"]),
                _num(s["mean"]),
            ]
            for mt, named in summary["communication"].items()
            for n, s in named.items()
        ]
        out += _table(["Message type", *header], rows)
    else:
        out += [
            "No per-message gateway metrics (`transfer_ms`, `message_bytes`, `rtt_ms`, "
            "`retry_count`) were recorded for this run, for example because it ran as a "
            "local simulation.",
            "",
        ]

    out += ["## Events", ""]
    counts = summary["event_counts"]
    out += _table(["Event type", "Count"], [[f"`{k}`", _num(v)] for k, v in counts.items()])
    if summary["incidents"]:
        out += ["### Dropouts, rejections and failures", ""]
        out += _table(
            ["Time", "Event", "Round", "Client", "Details"],
            [
                [
                    i["occurred_at"] or "-",
                    f"`{i['event_type']}`",
                    str(i["round_id"] or "-"),
                    i["client_id"] or "-",
                    json.dumps(i["details"], sort_keys=True) if i["details"] else "-",
                ]
                for i in summary["incidents"]
            ],
        )
    else:
        out += ["No dropouts, rejections or failures were recorded.", ""]

    fm = summary["final_model"]
    out += ["## Final model", ""]
    out += _table(
        ["Field", "Value"],
        [
            ["Version", f"v{fm['version']}"],
            ["Location", f"`{fm['path']}`" if fm["path"] else "-"],
            ["SHA-256", f"`{fm['sha256']}`" if fm["sha256"] else "-"],
            ["Format", fm["format"] or "-"],
            ["Size (bytes)", _num(fm["bytes"])],
        ],
    )

    if plots:
        out += ["## Plots", ""]
        out += [f"![{name}](plots/{name})" for name in plots]
        out += [""]
    return "\n".join(out)
