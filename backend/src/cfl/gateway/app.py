"""Gateway app: the only entry point for clients (planning/api-spec.md, section 3).

TLS is terminated by Caddy in front of it (deploy/). Every client endpoint requires a bearer
token (cfl.gateway.auth); all endpoints except /register additionally require a registered
client. The round rules are not implemented here: task assignment and the final artifact come
from the Coordinator's read-only methods, update acceptance from UpdateIntake, conformance from
cfl.conformance.checker, so simulation and the real gateway behave the same.

`app` connects to the database (CFL_DATABASE_URL) and the model directory (CFL_MODELS_DIR,
default /models) on first use, so importing this module needs neither.
"""

from __future__ import annotations

import fnmatch
import json
import os
import time
import uuid
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import Engine, and_, exists, select, text

from cfl import __version__
from cfl.conformance.checker import check_report
from cfl.coordinator.aggregator import global_model_key
from cfl.coordinator.coordinator import Coordinator, manifest_key
from cfl.coordinator.intake import IntakeReason, UpdateIntake
from cfl.coordinator.state_store import EventRecord, MetricRecord, RunRecord, RunStatus
from cfl.core.artifact import ARTIFACT, MANIFEST, MODEL_ONNX, MODEL_SAFETENSORS
from cfl.core.serialization import sha256
from cfl.core.store import FileModelStore, ModelStore
from cfl.db import schema as t
from cfl.db.state_store import PostgresStateStore, engine_from_env
from cfl.gateway.auth import CurrentClient, GatewayError, error_response
from cfl.gateway.clients import ClientRecord, ClientRegistry, ClientStatus
from cfl.protocol import PROTOCOL_VERSION
from cfl.protocol.messages import (
    ConformanceReportRequest,
    ConformanceReportResponse,
    ManifestSummary,
    RegisterRequest,
    RegisterResponse,
    TaskAssignment,
    TransferReport,
    UpdateAck,
    UpdateMetadata,
)

CONTROL_MESSAGES = frozenset({"register", "conformance", "task"})

# HTTP status for each intake decision (planning/api-spec.md, section 2.4).
INTAKE_STATUS = {
    None: 200,
    IntakeReason.DUPLICATE_IDEMPOTENT: 200,
    IntakeReason.INVALID_UPDATE_ID: 422,
    IntakeReason.CHECKSUM_MISMATCH: 422,
    IntakeReason.NOT_SELECTED: 403,
    IntakeReason.UNKNOWN_ROUND: 404,
    IntakeReason.STALE_ROUND: 409,
    IntakeReason.DEADLINE_PASSED: 409,
    IntakeReason.MODEL_VERSION_MISMATCH: 409,
    IntakeReason.DUPLICATE_UPDATE: 409,
}
FINAL_FILES = {
    "onnx": MODEL_ONNX,
    "safetensors": MODEL_SAFETENSORS,
    "artifact": ARTIFACT,
    "manifest": MANIFEST,
}


def _now() -> datetime:
    return datetime.now(UTC)


def _version_allowed(client_version: str) -> bool:
    """CFL_ALLOWED_CLIENT_VERSIONS: comma-separated patterns, e.g.
    "cfl-client/0.1.*,independent-client/*". Unset or empty: every version is accepted."""
    patterns = [p.strip() for p in os.environ.get("CFL_ALLOWED_CLIENT_VERSIONS", "").split(",")]
    patterns = [p for p in patterns if p]
    return not patterns or any(fnmatch.fnmatchcase(client_version, p) for p in patterns)


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def registered(client: CurrentClient) -> ClientRecord:
    if client.status is not ClientStatus.REGISTERED:
        raise GatewayError(403, "not_registered", "call POST /register first")
    return client


# Module level, so FastAPI can resolve the (string) annotation on the routes below.
RegisteredClient = Annotated[ClientRecord, Depends(registered)]


def create_app(
    engine: Engine | None = None, models: ModelStore | None = None, clock=_now
) -> FastAPI:
    app = FastAPI(title="CFL Gateway", version=__version__)

    @cache
    def db() -> Engine:
        return engine if engine is not None else engine_from_env()

    @cache
    def model_store() -> ModelStore:
        if models is not None:
            return models
        return FileModelStore(Path(os.environ.get("CFL_MODELS_DIR", "/models")))

    @cache
    def registry() -> ClientRegistry:
        return ClientRegistry(db(), clock)

    @cache
    def state() -> PostgresStateStore:
        return PostgresStateStore(db())

    @cache
    def coordinator() -> Coordinator:  # read-only use: the gateway never calls step()
        return Coordinator(state(), model_store(), clock)

    @cache
    def intake() -> UpdateIntake:
        return UpdateIntake(state(), model_store(), clock)

    app.state.db = db
    app.state.registry = registry

    @app.exception_handler(GatewayError)
    async def _gateway_error(request: Request, exc: GatewayError):
        return error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(request: Request, exc: RequestValidationError):
        details = {"errors": json.loads(json.dumps(exc.errors(), default=str))}
        return error_response(GatewayError(422, "invalid_request", "malformed request", details))

    def same_client(client: ClientRecord, claimed: str) -> None:
        if claimed != client.client_id:
            raise GatewayError(
                403,
                "identity_mismatch",
                "client_id does not match the token",
                {"token_client_id": client.client_id, "claimed_client_id": claimed},
            )

    def active_manifests() -> dict[str, tuple[bytes, RunRecord]]:
        """Manifests pinned by running runs, by id ("<dataset>@<version>")."""
        found: dict[str, tuple[bytes, RunRecord]] = {}
        for run in state().running_runs():
            data = model_store().get(manifest_key(run.run_id))
            if sha256(data) != run.manifest_sha256:
                raise GatewayError(500, "integrity_error", "stored manifest is corrupt")
            found.setdefault(run.manifest_version, (data, run))
        return found

    def get_run(run_id: str) -> RunRecord:
        if not _is_uuid(run_id):
            raise GatewayError(404, "unknown_run", f"no run {run_id}")
        try:
            return state().get_run(run_id)
        except KeyError:
            raise GatewayError(404, "unknown_run", f"no run {run_id}") from None

    def file_response(
        data: bytes, filename: str, checksum: str, headers: dict[str, str]
    ) -> Response:
        if sha256(data) != checksum:
            raise GatewayError(500, "integrity_error", f"{filename} does not match its checksum")
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Checksum-SHA256": checksum,
                **headers,
            },
        )

    def record_transfer(
        client_id: str,
        run_id: str,
        message_type: str,
        size: int,
        started: float,
        round_id: int | None = None,
        event_type: str | None = None,
    ) -> None:
        """Communication metrics (planning/interfaces.md, section 4) and, for downloads, an
        event. Accepted uploads already get their UPDATE_ACCEPTED event from the intake.
        `gateway_ms` is time inside the gateway; the network time is in `transfer_ms`, which
        the client reports through POST /transfer."""
        now, gateway_ms = clock(), (time.perf_counter() - started) * 1000
        if event_type is not None:
            payload = {"message_type": message_type, "bytes": size}
            state().add_event(EventRecord(event_type, now, run_id, round_id, client_id, payload))
        state().add_metrics(
            [
                MetricRecord(
                    run_id,
                    "message_bytes",
                    float(size),
                    "bytes",
                    now,
                    round_id,
                    client_id,
                    message_type,
                ),
                MetricRecord(
                    run_id,
                    "gateway_ms",
                    gateway_ms,
                    "ms",
                    now,
                    round_id,
                    client_id,
                    message_type,
                ),
            ]
        )

    # -- health -------------------------------------------------------------------------------
    @app.get("/health")
    def health() -> dict:
        try:
            with db().connect() as conn:
                conn.execute(text("SELECT 1"))
            database = "ok"
        except Exception:  # unconfigured or unreachable; health must still answer
            database = "unavailable"
        return {
            "status": "ok",
            "version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "database": database,
        }

    # -- 3.1 registration ---------------------------------------------------------------------
    @app.post("/register", response_model=RegisterResponse)
    def register(body: RegisterRequest, client: CurrentClient) -> RegisterResponse:
        same_client(client, body.client_id)
        if body.protocol_version != PROTOCOL_VERSION:
            raise GatewayError(
                422,
                "protocol_version_mismatch",
                f"the server speaks protocol {PROTOCOL_VERSION}",
                {"expected": PROTOCOL_VERSION, "received": body.protocol_version},
            )
        if not _version_allowed(body.client_version):
            raise GatewayError(
                403,
                "version_not_approved",
                f"client version {body.client_version} is not approved",
                {"client_version": body.client_version},
            )
        registry().register(client.client_id, body.client_version, body.protocol_version)
        manifests = list(active_manifests())
        return RegisterResponse(
            accepted=True,
            client_id=client.client_id,
            status=ClientStatus.REGISTERED.value,
            manifest_version=manifests[0] if len(manifests) == 1 else None,
            message="registered; fetch GET /conformance and submit a conformance report",
        )

    # -- 3.2 / 3.3 conformance ----------------------------------------------------------------
    @app.get("/conformance", response_model=list[ManifestSummary])
    def list_manifests(client: RegisteredClient) -> list[ManifestSummary]:
        return [
            ManifestSummary(
                manifest_version=mid,
                dataset=json.loads(data)["dataset"]["name"],
                manifest_sha256=run.manifest_sha256,
            )
            for mid, (data, run) in active_manifests().items()
        ]

    @app.get("/conformance/{manifest_version}")
    def get_manifest(manifest_version: str, client: RegisteredClient) -> Response:
        found = active_manifests().get(manifest_version)
        if found is None:
            raise GatewayError(404, "unknown_manifest", f"no active manifest {manifest_version}")
        data, run = found
        return Response(
            content=data,
            media_type="application/json",
            headers={
                "ETag": f'"{run.manifest_sha256}"',
                "X-Manifest-Checksum-SHA256": run.manifest_sha256,
            },
        )

    @app.post("/conformance/report", response_model=ConformanceReportResponse)
    def conformance_report(body: ConformanceReportRequest, client: RegisteredClient):
        same_client(client, body.client_id)
        found = active_manifests().get(body.manifest_version)
        if found is None:
            result_code, message, details, passed = (
                "manifest_version_mismatch",
                f"no active manifest {body.manifest_version}",
                {"active": list(active_manifests())},
                False,
            )
        else:
            data, run = found
            result = check_report(json.loads(data), run.manifest_sha256, body)
            result_code, message, details, passed = (
                result.reason_code,
                result.message,
                result.details,
                result.passed,
            )
        report_id = str(uuid.uuid4())
        details = json.loads(json.dumps(details, default=str))
        checked_at = registry().record_conformance(
            report_id=report_id,
            client_id=client.client_id,
            manifest_version=body.manifest_version,
            client_version=body.client_version,
            protocol_version=body.protocol_version,
            passed=passed,
            reason_code=result_code,
            payload={
                "report": body.model_dump(mode="json"),
                "server": {"reason_code": result_code, "details": details},
            },
        )
        response = ConformanceReportResponse(
            status="PASSED" if passed else "FAILED",
            client_id=client.client_id,
            manifest_version=body.manifest_version,
            report_id=report_id,
            reason_code=result_code,
            message="eligible for round selection" if passed else message,
            details=details,
            checked_at=checked_at,
        )
        return JSONResponse(
            status_code=200 if passed else 422, content=response.model_dump(mode="json")
        )

    # -- 3.4 task polling ---------------------------------------------------------------------
    @app.get("/task", response_model=TaskAssignment, responses={204: {"description": "no task"}})
    def task(client: RegisteredClient):
        assignment = coordinator().active_task(client.client_id)
        return assignment if assignment is not None else Response(status_code=204)

    # -- 3.7 final model (declared before 3.5 so "final" is not parsed as a version) ----------
    @app.get("/model/{run_id}/final")
    def final_model(run_id: str, client: RegisteredClient, format: str = "safetensors"):
        if format not in FINAL_FILES:
            raise GatewayError(
                422, "invalid_format", f"format must be one of {sorted(FINAL_FILES)}"
            )
        run = get_run(run_id)
        if run.status is RunStatus.RUNNING:
            raise GatewayError(409, "run_in_progress", "the run has not finished yet")
        files = coordinator().final_artifact(run_id)
        if files is None:
            raise GatewayError(404, "no_final_model", f"run {run_id} has no final model")
        name = FINAL_FILES[format]
        descriptor = json.loads(files[ARTIFACT])
        if name == MODEL_ONNX:
            checksum = run.final_model_sha256  # the server-published value from the database
        elif name == ARTIFACT:
            checksum = sha256(files[ARTIFACT])
        else:
            checksum = descriptor["files"][name]["sha256"]
        start = time.perf_counter()
        response = file_response(
            files[name],
            f"final_model_{run_id}.{name.split('.', 1)[1]}" if name.startswith("model") else name,
            checksum,
            {
                "X-Manifest-Version": run.manifest_version,
                "X-Model-Version": str(run.model_version),
                "X-Run-ID": run_id,
                "X-Final-Model-SHA256": run.final_model_sha256,
            },
        )
        record_transfer(
            client.client_id,
            run_id,
            "final_model_download",
            len(files[name]),
            start,
            event_type="FINAL_MODEL_DOWNLOADED",
        )
        return response

    # -- 3.5 model download -------------------------------------------------------------------
    @app.get("/model/{run_id}/{version}")
    def download_model(run_id: str, version: int, client: RegisteredClient):
        run = get_run(run_id)
        p, r = t.round_participants, t.rounds
        with db().connect() as conn:
            participant = conn.execute(
                select(exists().where(p.c.run_id == run_id, p.c.client_id == client.client_id))
            ).scalar()
            checksum = conn.execute(
                select(r.c.input_model_sha256)
                .where(r.c.run_id == run_id, r.c.input_model_version == version)
                .union(
                    select(r.c.output_model_sha256).where(
                        and_(r.c.run_id == run_id, r.c.output_model_version == version)
                    )
                )
            ).scalar()
        if not participant:
            raise GatewayError(403, "not_participant", "this client was not selected in the run")
        if checksum is None and version == run.model_version:
            checksum = run.model_sha256
        if checksum is None:
            raise GatewayError(404, "unknown_model_version", f"no model version {version}")
        start = time.perf_counter()
        data = model_store().get(global_model_key(run_id, version))
        response = file_response(
            data,
            f"global_model_v{version}.safetensors",
            checksum,
            {"X-Model-Version": str(version), "X-Run-ID": run_id},
        )
        record_transfer(
            client.client_id,
            run_id,
            "model_download",
            len(data),
            start,
            event_type="MODEL_DOWNLOADED",
        )
        return response

    # -- 3.8 transfer report (addition) ----------------------------------------------------------
    @app.post("/transfer", status_code=204)
    def report_transfer(body: TransferReport, client: RegisteredClient):
        get_run(body.run_id)
        p = t.round_participants
        with db().connect() as conn:
            participant = conn.execute(
                select(exists().where(p.c.run_id == body.run_id, p.c.client_id == client.client_id))
            ).scalar()
        if not participant:
            raise GatewayError(403, "not_participant", "this client was not selected in the run")
        now = clock()
        scope = (now, body.round_id, client.client_id, body.message_type)
        rows = [MetricRecord(body.run_id, "transfer_ms", body.duration_ms, "ms", *scope)]
        # The gateway records the size of the file transfers itself.
        if body.message_type in CONTROL_MESSAGES and body.message_bytes is not None:
            size = float(body.message_bytes)
            rows.append(MetricRecord(body.run_id, "message_bytes", size, "bytes", *scope))
        state().add_metrics(rows)

    # -- 3.6 update upload --------------------------------------------------------------------
    @app.post("/update", response_model=UpdateAck)
    def upload_update(
        client: RegisteredClient,
        metadata: Annotated[str, Form()],
        weights: Annotated[UploadFile, File()],
    ):
        try:
            meta = UpdateMetadata.model_validate_json(metadata)
        except ValidationError as exc:
            raise GatewayError(
                422,
                "invalid_metadata",
                "metadata is not a valid UpdateMetadata",
                {"errors": json.loads(exc.json())},
            ) from None
        same_client(client, meta.client_id)
        start = time.perf_counter()
        data = weights.file.read()
        ack = intake().submit(meta, data)
        if ack.accepted and ack.reason is None:
            record_transfer(
                client.client_id,
                meta.run_id,
                "update_upload",
                len(data),
                start,
                round_id=meta.round_id,
            )
        body = ack.model_copy(
            update={
                "update_id": meta.update_id,
                "client_id": client.client_id,
                "round_id": meta.round_id,
            }
        )
        status = INTAKE_STATUS.get(IntakeReason(ack.reason) if ack.reason else None, 500)
        return JSONResponse(status_code=status, content=body.model_dump(mode="json"))

    return app


app = create_app()
