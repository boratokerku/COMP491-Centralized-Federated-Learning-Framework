import json

import httpx
import pytest
from pydantic import ValidationError

from cfl.client.http import HttpTransport
from cfl.protocol.messages import TransferReport, UpdateAck, UpdateMetadata


def transport(handler) -> HttpTransport:
    client = httpx.Client(base_url="https://gateway.test", transport=httpx.MockTransport(handler))
    return HttpTransport("https://gateway.test", "token", client=client, sleep=lambda _: None)


def metadata() -> UpdateMetadata:
    return UpdateMetadata(
        run_id="run",
        round_id=3,
        client_id="c",
        update_id="u",
        model_version=2,
        num_samples=1,
        checksum_sha256="x",
    )


def test_report_carries_the_time_of_the_last_upload():
    reports = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/transfer":
            reports.append(json.loads(request.content))
            return httpx.Response(204)
        return httpx.Response(200, json=UpdateAck(accepted=True).model_dump())

    t = transport(handler)
    t.upload_update(metadata(), b"weights")
    t.report_transfer("run", 3, "update_upload")

    assert reports == [
        {
            "run_id": "run",
            "round_id": 3,
            "message_type": "update_upload",
            "duration_ms": t.stats.last_seconds["update_upload"] * 1000,
            "message_bytes": t.stats.last_bytes["update_upload"],
        }
    ]


def test_control_messages_report_their_size():
    reports = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/transfer":
            reports.append(json.loads(request.content))
            return httpx.Response(204)
        return httpx.Response(204)

    t = transport(handler)
    assert t.poll_task("c") is None
    t.report_transfer("run", 3, "task")

    assert reports[0]["message_type"] == "task"
    assert reports[0]["message_bytes"] == 0  # an empty answer to an empty request


def test_nothing_is_reported_before_a_transfer():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    transport(handler).report_transfer("run", 3, "update_upload")


def test_a_failed_attempt_does_not_count_as_the_transfer_time():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=UpdateAck(accepted=True).model_dump())

    t = transport(handler)
    t.upload_update(metadata(), b"weights")

    assert len(calls) == 2
    assert t.stats.seconds["update_upload"] >= t.stats.last_seconds["update_upload"]


def test_an_unreachable_gateway_does_not_disturb_the_client():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/transfer":
            raise httpx.ConnectError("gone")
        return httpx.Response(200, json=UpdateAck(accepted=True).model_dump())

    t = transport(handler)
    t.upload_update(metadata(), b"weights")
    t.report_transfer("run", 3, "update_upload")


@pytest.mark.parametrize(
    "fields",
    [
        {"message_type": "update_upload", "duration_ms": -1},
        {"message_type": "heartbeat", "duration_ms": 1},
    ],
)
def test_invalid_reports_are_rejected(fields):
    with pytest.raises(ValidationError):
        TransferReport(run_id="run", **fields)
