import csv
import io
import json
from datetime import UTC, datetime, timedelta

import pytest

from cfl.coordinator.state import RoundState
from cfl.coordinator.state_store import (
    EventRecord,
    InMemoryStateStore,
    MetricRecord,
    ParticipantRecord,
    ParticipantStatus,
    RoundRecord,
    RunRecord,
    RunStatus,
    StoredUpdate,
)
from cfl.core.store import FileModelStore, InMemoryModelStore
from cfl.reporting.cli import main as report_main
from cfl.reporting.data import load_run_data
from cfl.reporting.generate import generate_report, metrics_csv
from cfl.reporting.summary import build_summary

RUN_ID = "11111111-1111-4111-8111-111111111111"
T0 = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
CLIENTS = ["bank-a", "bank-b", "bank-c"]
SHA = "a" * 64


def _round(n: int, state: RoundState, **kw) -> RoundRecord:
    return RoundRecord(
        run_id=RUN_ID,
        round_id=n,
        state=state,
        input_model_version=n - 1,
        input_model_path=f"{RUN_ID}/models/{n - 1}.safetensors",
        input_model_sha256=SHA,
        min_participants=2,
        deadline_at=T0 + timedelta(minutes=n * 10),
        opened_at=T0 + timedelta(minutes=n * 10 - 5),
        closed_at=T0 + timedelta(minutes=n * 10),
        **kw,
    )


def _metric(name, value, unit, round_id=None, client=None, message_type=None):
    return MetricRecord(
        run_id=RUN_ID,
        metric_name=name,
        value=value,
        unit=unit,
        recorded_at=T0,
        round_id=round_id,
        client_id=client,
        message_type=message_type,
    )


@pytest.fixture
def state() -> InMemoryStateStore:
    """Round 1: all three clients. Round 2: bank-c drops out. Round 3: fails (no quorum)."""
    store = InMemoryStateStore()
    store.create_run(
        RunRecord(
            run_id=RUN_ID,
            name="report-test",
            status=RunStatus.COMPLETED,
            strategy="fedavg",
            task_name="tabular",
            seed=42,
            config={"run": {"seed": 42}, "rounds": {"num_rounds": 3}},
            config_sha256=SHA,
            protocol_version="1.0",
            manifest_version="1.0.0",
            manifest_sha256=SHA,
            preprocessing_version="pre-1",
            model_architecture="mlp",
            model_architecture_version="1",
            model_fingerprint=SHA,
            model_version=2,
            model_path=f"{RUN_ID}/models/2.safetensors",
            model_sha256=SHA,
            created_at=T0,
            started_at=T0,
            completed_at=T0 + timedelta(seconds=90),
            final_model_path=f"{RUN_ID}/final/model.onnx",
            final_model_sha256="b" * 64,
            final_model_format="onnx",
            final_model_bytes=4096,
        ),
        EventRecord("RUN_STARTED", T0, run_id=RUN_ID),
    )
    plan = [
        (1, RoundState.DONE, CLIENTS, None),
        (2, RoundState.DONE, CLIENTS, "bank-c"),
        (3, RoundState.FAILED, CLIENTS, None),
    ]
    for n, round_state, selected, dropped in plan:
        extra = (
            {"failure_reason": "quorum not reached"}
            if round_state is RoundState.FAILED
            else {
                "output_model_version": n,
                "output_model_path": f"{RUN_ID}/models/{n}.safetensors",
                "output_model_sha256": SHA,
            }
        )
        participants = []
        for c in selected:
            if round_state is RoundState.FAILED or c == dropped:
                status = ParticipantStatus.DROPOUT
            else:
                status = ParticipantStatus.UPDATE_ACCEPTED
            participants.append(
                ParticipantRecord(RUN_ID, n, c, status, selected_at=T0, finished_at=T0)
            )
        store.create_round(
            _round(n, round_state, **extra),
            participants,
            EventRecord("ROUND_OPENED", T0, run_id=RUN_ID, round_id=n),
        )
        if round_state is RoundState.DONE:
            for c in selected:
                if c == dropped:
                    store.add_event(
                        EventRecord(
                            "CLIENT_DROPOUT",
                            T0,
                            run_id=RUN_ID,
                            round_id=n,
                            client_id=c,
                            payload={"reason_code": "deadline_passed"},
                        )
                    )
                    continue
                store.accept_update(
                    StoredUpdate(
                        update_id=f"{n}-{c}",
                        run_id=RUN_ID,
                        round_id=n,
                        client_id=c,
                        model_version=n - 1,
                        num_samples=100 + 10 * CLIENTS.index(c),
                        weights_path=f"{RUN_ID}/updates/{n}/{c}.safetensors",
                        checksum_sha256=SHA,
                        payload_bytes=2000,
                        received_at=T0,
                    ),
                    EventRecord("UPDATE_ACCEPTED", T0, run_id=RUN_ID, round_id=n, client_id=c),
                )
                store.add_metrics(
                    [
                        _metric("client_val_f1_before", 0.4 + 0.1 * n, "value", n, c),
                        _metric("client_val_f1_after", 0.5 + 0.1 * n, "value", n, c),
                        _metric("client_train_ms", 1000.0 * n, "ms", n, c),
                    ]
                )
            store.add_metrics(
                [
                    _metric("global_f1", 0.5 + 0.1 * n, "value", n),
                    _metric("global_roc_auc", 0.8 + 0.05 * n, "value", n),
                    _metric("global_pr_auc", 0.6 + 0.05 * n, "value", n),
                    _metric("waiting_ms", 300.0, "ms", n),
                    _metric("aggregation_ms", 20.0, "ms", n),
                    _metric("round_duration_ms", 400.0 + n, "ms", n),
                    _metric("model_bytes", 1000.0, "bytes", n),
                ]
            )
        else:
            store.add_event(
                EventRecord(
                    "ROUND_FAILED",
                    T0,
                    run_id=RUN_ID,
                    round_id=n,
                    payload={"reason": "quorum not reached"},
                )
            )
    store.add_event(EventRecord("RUN_COMPLETED", T0, run_id=RUN_ID))
    return store


def test_summary_counts_rounds_and_failure_reasons(state):
    summary = build_summary(load_run_data(state, RUN_ID))
    assert summary["rounds_summary"] == {
        "attempted": 3,
        "completed": 2,
        "failed": 1,
        "failures": [{"round_id": 3, "reason": "quorum not reached"}],
    }
    assert summary["run"]["duration_s"] == 90
    assert summary["versions"]["protocol"] == "1.0"
    assert summary["versions"]["manifest"] == "1.0.0"
    assert summary["versions"]["preprocessing"] == "pre-1"
    assert summary["config"]["rounds"]["num_rounds"] == 3
    assert summary["final_model"]["sha256"] == "b" * 64


def test_summary_participation_dropout_and_samples(state):
    summary = build_summary(load_run_data(state, RUN_ID))
    round2 = summary["rounds"][1]
    statuses = {p["client_id"]: p["status"] for p in round2["participants"]}
    assert statuses == {
        "bank-a": "UPDATE_ACCEPTED",
        "bank-b": "UPDATE_ACCEPTED",
        "bank-c": "DROPOUT",
    }
    assert summary["clients"]["bank-c"]["dropouts"] == 2  # round 2 and the failed round 3
    assert summary["clients"]["bank-a"]["samples_per_round"] == {"1": 100, "2": 100}
    assert summary["clients"]["bank-b"]["total_samples"] == 220
    assert [i["event_type"] for i in summary["incidents"]] == ["CLIENT_DROPOUT", "ROUND_FAILED"]


def test_metric_summaries_have_min_max_mean(state):
    summary = build_summary(load_run_data(state, RUN_ID))
    f1 = summary["metric_summaries"]["global_f1"]
    assert (f1["count"], f1["min"], f1["max"]) == (2, pytest.approx(0.6), pytest.approx(0.7))
    assert f1["mean"] == pytest.approx(0.65)
    client_f1 = summary["metric_summaries"]["all_clients.client_val_f1_after"]
    assert client_f1["count"] == 5  # 3 clients in round 1, 2 in round 2
    assert summary["clients"]["bank-a"]["metrics"]["client_train_ms"]["max"] == 2000.0


def test_communication_summary_groups_by_message_type(state):
    state.add_metrics(
        [
            _metric("transfer_ms", 10.0, "ms", 1, "bank-a", "model_download"),
            _metric("transfer_ms", 30.0, "ms", 2, "bank-a", "model_download"),
            _metric("message_bytes", 500.0, "bytes", 1, "bank-a", "update_upload"),
        ]
    )
    comm = build_summary(load_run_data(state, RUN_ID))["communication"]
    assert comm["model_download"]["transfer_ms"]["mean"] == 20.0
    assert comm["update_upload"]["message_bytes"]["max"] == 500.0


def test_summary_is_deterministic(state):
    data = load_run_data(state, RUN_ID)
    assert json.dumps(build_summary(data), sort_keys=True) == json.dumps(
        build_summary(data), sort_keys=True
    )


def test_generate_report_writes_all_files(state, tmp_path):
    keys = generate_report(state, FileModelStore(tmp_path), RUN_ID)
    report_dir = tmp_path / RUN_ID / "report"
    for name in ("summary.json", "report.md", "metrics.csv"):
        assert (report_dir / name).is_file()
    plots = sorted(p.name for p in (report_dir / "plots").glob("*.png"))
    assert plots == [
        "client_before_after.png",
        "global_metrics.png",
        "round_time_breakdown.png",
        "transfer_bytes.png",
    ]
    for png in (report_dir / "plots").glob("*.png"):
        assert png.read_bytes().startswith(b"\x89PNG")
    assert len(keys) == 3 + len(plots)
    assert json.loads((report_dir / "summary.json").read_text())["run"]["run_id"] == RUN_ID


def test_markdown_explains_the_run_without_the_console(state):
    models = InMemoryModelStore()
    generate_report(state, models, RUN_ID)
    text = models.get(f"{RUN_ID}/report/report.md").decode()
    for expected in (
        RUN_ID,
        "pre-1",
        "1.0.0",
        "Attempted: **3**, completed: **2**, failed: **1**",
        "quorum not reached",
        "CLIENT_DROPOUT",
        "bank-c",
        "b" * 64,
        "![global_metrics.png](plots/global_metrics.png)",
    ):
        assert expected in text
    assert "No per-message gateway metrics" in text


def test_metrics_csv_has_every_row(state):
    data = load_run_data(state, RUN_ID)
    rows = list(csv.DictReader(io.StringIO(metrics_csv(data))))
    assert len(rows) == len(data.metrics)
    assert rows[0].keys() >= {"run_id", "round_id", "client_id", "metric_name", "value", "unit"}
    assert {r["run_id"] for r in rows} == {RUN_ID}


def test_report_can_be_regenerated(state):
    models = InMemoryModelStore()
    first = generate_report(state, models, RUN_ID)
    summary = models.get(f"{RUN_ID}/report/summary.json")
    assert generate_report(state, models, RUN_ID) == first
    assert models.get(f"{RUN_ID}/report/summary.json") == summary


def test_run_without_rounds_still_reports():
    store = InMemoryStateStore()
    run = RunRecord(
        run_id=RUN_ID,
        name="empty",
        status=RunStatus.PENDING,
        strategy="fedavg",
        task_name="text",
        seed=1,
        config={},
        config_sha256=SHA,
        protocol_version="1.0",
        manifest_version="1.0.0",
        manifest_sha256=SHA,
        preprocessing_version="p",
        model_architecture="m",
        model_architecture_version="1",
        model_fingerprint=SHA,
        model_version=0,
        model_path="x",
        model_sha256=SHA,
        created_at=T0,
    )
    store.create_run(run, EventRecord("RUN_STARTED", T0, run_id=RUN_ID))
    models = InMemoryModelStore()
    generate_report(store, models, RUN_ID)
    summary = json.loads(models.get(f"{RUN_ID}/report/summary.json"))
    assert summary["rounds_summary"]["attempted"] == 0
    assert not models.exists(f"{RUN_ID}/report/plots/global_metrics.png")


def test_unknown_run_raises(state):
    with pytest.raises(KeyError):
        generate_report(state, InMemoryModelStore(), "22222222-2222-4222-8222-222222222222")


def test_cli_needs_a_models_dir(monkeypatch):
    monkeypatch.delenv("CFL_MODELS_DIR", raising=False)
    with pytest.raises(SystemExit) as exc:
        report_main([RUN_ID])
    assert exc.value.code == 2
