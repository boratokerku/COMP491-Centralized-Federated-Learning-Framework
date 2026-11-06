"""HTTPS implementation of the client Transport (planning/api-spec.md, section 3).

Network failures are expected between independent sites, so every request is retried with
exponential backoff and full jitter when the failure is transient: connection errors, timeouts,
429 and 502/503/504 (e.g. Caddy while the gateway restarts). Answers that will not change on a
retry (401, 403, 404, 409, 422) are returned or raised at once. Retrying POST /update is safe:
the metadata keeps its update_id, so the gateway counts the update once even if the first
attempt was accepted and only its response was lost.

Every downloaded file is checked against the gateway's X-Checksum-SHA256 header; a corrupted
transfer is downloaded again and fails with ChecksumError if it stays corrupt.
"""

from __future__ import annotations

import random
import ssl
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from cfl.core.artifact import ARTIFACT, MANIFEST, MODEL_ONNX, MODEL_SAFETENSORS
from cfl.core.serialization import sha256
from cfl.core.transport import Transport
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

RETRY_STATUSES = frozenset({429, 502, 503, 504})
FINAL_FORMATS = {"onnx": MODEL_ONNX, "artifact": ARTIFACT, "manifest": MANIFEST}
FINAL_FORMATS_OPTIONAL = {"safetensors": MODEL_SAFETENSORS}


class GatewayHTTPError(RuntimeError):
    """A non-retryable error answer in the gateway's error schema."""

    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}


class GatewayUnavailable(RuntimeError):
    """The gateway could not be reached (or kept failing) after all retries."""


class ChecksumError(RuntimeError):
    """A download did not match the checksum the gateway published, even after retrying."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0

    def delay(self, retry: int, rng: random.Random) -> float:
        """Full jitter: uniform in [0, min(max, base * 2^retry)], so clients that failed at
        the same moment do not all come back at the same moment."""
        return rng.uniform(0, min(self.max_delay_s, self.base_delay_s * 2**retry))


@dataclass
class TransferStats:
    """Per message type: requests, retries, bytes and time, for the client's metrics."""

    requests: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    retries: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    bytes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    last_seconds: dict[str, float] = field(default_factory=dict)  # last successful request
    last_bytes: dict[str, int] = field(default_factory=dict)  # its request + response bodies


def _tls_failure(exc: BaseException) -> bool:
    """Certificate problems do not go away by retrying: report them at once."""
    while exc is not None:
        if isinstance(exc, ssl.SSLCertVerificationError):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _error(response: httpx.Response) -> GatewayHTTPError:
    try:
        error = response.json()["error"]
        return GatewayHTTPError(
            response.status_code, error["code"], error["message"], error.get("details")
        )
    except (ValueError, KeyError, TypeError):
        return GatewayHTTPError(response.status_code, "http_error", response.text[:200])


class HttpTransport(Transport):
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        verify: str | bool = True,
        timeout: httpx.Timeout | None = None,
        retry: RetryPolicy | None = None,
        client: httpx.Client | None = None,
        user_agent: str = "cfl-client",
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ):
        """`verify` is True (system CAs, e.g. Let's Encrypt) or the path of a CA bundle for a
        server with its own CA. `client` replaces the HTTP client, e.g. in tests."""
        if not base_url.startswith("https://"):
            raise ValueError("the gateway must be reached over https://")
        self.retry = retry or RetryPolicy()
        self.sleep = sleep
        self.rng = rng or random.Random()
        self.stats = TransferStats()
        headers = {"Authorization": f"Bearer {token}", "User-Agent": user_agent}
        if client is None:
            client = httpx.Client(
                base_url=base_url.rstrip("/"),
                verify=verify,
                timeout=timeout or httpx.Timeout(10.0, read=120.0, write=120.0),
                headers=headers,
            )
        else:
            client.headers.update(headers)
        self.client = client

    def close(self) -> None:
        self.client.close()

    # -- core request loop ----------------------------------------------------------------------
    def _request(self, message_type: str, method: str, url: str, **kwargs) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(self.retry.max_attempts):
            if attempt:
                self.stats.retries[message_type] += 1
                self.sleep(self.retry.delay(attempt - 1, self.rng))
            self.stats.requests[message_type] += 1
            start = time.perf_counter()
            try:
                response = self.client.request(method, url, **kwargs)
            except httpx.TransportError as exc:  # connect/read/write errors and timeouts
                if _tls_failure(exc):
                    raise GatewayUnavailable(
                        f"TLS certificate of {self.client.base_url} is not trusted: {exc} "
                        "(for a server with its own CA, pass its root certificate as verify=)"
                    ) from exc
                last = exc
                continue
            finally:
                elapsed = time.perf_counter() - start
                self.stats.seconds[message_type] += elapsed
            if response.status_code in RETRY_STATUSES:
                last = GatewayHTTPError(response.status_code, "unavailable", "gateway busy")
                continue
            self.stats.bytes[message_type] += len(response.content)
            self.stats.last_seconds[message_type] = elapsed
            sent = int(response.request.headers.get("content-length", 0))
            self.stats.last_bytes[message_type] = sent + len(response.content)
            return response
        raise GatewayUnavailable(
            f"{method} {url} failed after {self.retry.max_attempts} attempts: {last}"
        ) from last

    def _download(self, message_type: str, url: str, **kwargs) -> httpx.Response:
        """GET a file and verify X-Checksum-SHA256; a corrupt transfer is retried."""
        for attempt in range(self.retry.max_attempts):
            if attempt:
                self.stats.retries[message_type] += 1
                self.sleep(self.retry.delay(attempt - 1, self.rng))
            response = self._request(message_type, "GET", url, **kwargs)
            if response.status_code != 200:
                return response
            expected = response.headers.get("x-checksum-sha256")
            if expected is None:
                raise ChecksumError(f"{url}: the gateway sent no checksum")
            if sha256(response.content) == expected:
                return response
        raise ChecksumError(f"{url}: download does not match its checksum")

    # -- Transport interface ----------------------------------------------------------------------
    def register(self, request: RegisterRequest) -> RegisterResponse:
        response = self._request("register", "POST", "/register", json=request.model_dump())
        if response.status_code != 200:
            raise _error(response)
        return RegisterResponse.model_validate(response.json())

    def poll_task(self, client_id: str) -> TaskAssignment | None:
        """`client_id` is implied by the token; kept for the Transport interface."""
        response = self._request("task", "GET", "/task")
        if response.status_code == 204:
            return None
        if response.status_code != 200:
            raise _error(response)
        return TaskAssignment.model_validate(response.json())

    def download_model(self, run_id: str, model_version: int) -> bytes:
        response = self._download("model_download", f"/model/{run_id}/{model_version}")
        if response.status_code != 200:
            raise _error(response)
        if response.headers.get("x-model-version") != str(model_version):
            raise ChecksumError(f"asked for v{model_version}, got {response.headers}")
        return response.content

    def upload_update(self, metadata: UpdateMetadata, weights: bytes) -> UpdateAck:
        response = self._request(
            "update_upload",
            "POST",
            "/update",
            data={"metadata": metadata.model_dump_json()},
            files={"weights": (f"{metadata.update_id}.safetensors", weights)},
        )
        try:
            return UpdateAck.model_validate(response.json())
        except ValueError:  # the error schema: the update was never considered
            raise _error(response) from None

    def report_transfer(self, run_id: str, round_id: int | None, message_type: str) -> None:
        """Tell the gateway how long the last transfer of this type took. This is only a
        measurement, so a failure here must not disturb training."""
        seconds = self.stats.last_seconds.get(message_type)
        if seconds is None:
            return
        report = TransferReport(
            run_id=run_id,
            round_id=round_id,
            message_type=message_type,
            duration_ms=seconds * 1000,
            message_bytes=self.stats.last_bytes.get(message_type),
        )
        try:
            self._request("transfer", "POST", "/transfer", json=report.model_dump())
        except GatewayUnavailable:
            pass

    def download_final_artifact(self, run_id: str) -> tuple[dict[str, bytes], str] | None:
        """All final artifact files and the server-published SHA-256 of model.onnx, or None
        while the run is still in progress."""
        files: dict[str, bytes] = {}
        onnx_sha = None
        for fmt, name in {**FINAL_FORMATS, **FINAL_FORMATS_OPTIONAL}.items():
            response = self._download(
                "final_model_download", f"/model/{run_id}/final", params={"format": fmt}
            )
            if response.status_code == 409:
                return None
            if response.status_code != 200:
                raise _error(response)
            files[name] = response.content
            self.report_transfer(run_id, None, "final_model_download")
            onnx_sha = response.headers.get("x-final-model-sha256", onnx_sha)
        if onnx_sha is None:
            raise ChecksumError("the gateway did not publish the final model checksum")
        return files, onnx_sha

    # -- conformance (not part of the in-memory simulation) -------------------------------------
    def list_manifests(self) -> list[ManifestSummary]:
        response = self._request("conformance", "GET", "/conformance")
        if response.status_code != 200:
            raise _error(response)
        return [ManifestSummary.model_validate(m) for m in response.json()]

    def get_manifest(self, manifest_version: str) -> tuple[bytes, str]:
        """The manifest bytes and their verified SHA-256 (quoted in the report)."""
        response = self._request("conformance", "GET", f"/conformance/{manifest_version}")
        if response.status_code != 200:
            raise _error(response)
        expected = response.headers.get("x-manifest-checksum-sha256")
        if expected is None or sha256(response.content) != expected:
            raise ChecksumError(f"manifest {manifest_version} does not match its checksum")
        return response.content, expected

    def submit_report(self, report: ConformanceReportRequest) -> ConformanceReportResponse:
        """Passed (200) and failed (422) reports both return the gateway's decision."""
        response = self._request(
            "conformance", "POST", "/conformance/report", json=report.model_dump(mode="json")
        )
        if response.status_code in (200, 422):
            try:
                return ConformanceReportResponse.model_validate(response.json())
            except ValueError:
                pass  # a 422 in the error schema (malformed report)
        raise _error(response)

    def ping(self) -> float:
        """Round-trip time to the gateway in milliseconds (GET /health, no authentication)."""
        start = time.perf_counter()
        response = self._request("health", "GET", "/health")
        if response.status_code != 200:
            raise _error(response)
        return (time.perf_counter() - start) * 1000
