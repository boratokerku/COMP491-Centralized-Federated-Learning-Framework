"""Tests for Faz 5: Secured Channel (mTLS settings, HttpTransport, Gateway CN binding, certs)."""

from __future__ import annotations

import os
import ssl
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from cryptography import x509
from fastapi.testclient import TestClient

from cfl.client.http import GatewayUnavailable, HttpTransport, RetryPolicy
from cfl.client.settings import SettingsError, load_settings
from cfl.coordinator.state_store import InMemoryStateStore
from cfl.core.store import InMemoryModelStore
from cfl.gateway.app import create_app
from cfl.gateway.auth import _extract_cert_cn
from cfl.gateway.clients import AuthError, ClientRecord, ClientStatus

REPO_ROOT = Path(__file__).resolve().parents[2]
MTLS_SCRIPT = REPO_ROOT / "deploy" / "scripts" / "mtls_certs.sh"


class _StubRegistry:
    def __init__(self, client_ids: list[str]):
        now = datetime.now(UTC)
        self._tokens = {
            f"tok-{cid}": ClientRecord(
                client_id=cid,
                status=ClientStatus.REGISTERED,
                client_version="cfl-client/0.1.0",
                protocol_version="0.1",
                created_at=now,
                registered_at=now,
                last_seen_at=now,
            )
            for cid in client_ids
        }

    def authenticate(self, token: str) -> ClientRecord:
        rec = self._tokens.get(token)
        if rec is None:
            raise AuthError(401, "invalid_token", "unknown token")
        return rec


class TestClientSettingsMTLS:
    def test_mtls_pair_from_yaml_resolves_relative_paths(self, tmp_path):
        cfg = tmp_path / "client.yaml"
        cfg.write_text(
            "server_url: https://fl.localhost\n"
            "client_id: bank-a\n"
            "task: tabular\n"
            "client_cert: certs/bank-a.crt\n"
            "client_key: certs/bank-a.key\n"
        )
        s = load_settings(config_file=cfg, env={})
        assert s.client_cert == tmp_path / "certs" / "bank-a.crt"
        assert s.client_key == tmp_path / "certs" / "bank-a.key"
        assert s.cert == (str(s.client_cert), str(s.client_key))

    def test_mtls_partial_config_rejected(self):
        with pytest.raises(SettingsError, match="client_cert and client_key"):
            load_settings(env={"CFL_CLIENT_CERT": "/tmp/a.crt"})
        with pytest.raises(SettingsError, match="client_cert and client_key"):
            load_settings(env={"CFL_CLIENT_KEY": "/tmp/a.key"})

    def test_no_mtls_gives_none_cert(self):
        s = load_settings(env={})
        assert s.cert is None


class TestHttpTransportMTLS:
    def test_ssl_handshake_error_fails_fast_without_retries(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("handshake failed") from ssl.SSLError("CERTIFICATE_REQUIRED")

        client = httpx.Client(
            base_url="https://gateway.test", transport=httpx.MockTransport(handler)
        )
        t = HttpTransport(
            "https://gateway.test",
            "tok",
            client=client,
            retry=RetryPolicy(max_attempts=5),
            sleep=lambda _: None,
        )
        with pytest.raises(GatewayUnavailable, match="TLS certificate"):
            t.ping()
        assert calls == 1  # failed immediately, no wasted retries


class TestGatewayMTLSBinding:
    def test_extract_cert_cn(self):
        assert _extract_cert_cn("CN=bank-a,O=CFL-Client") == "bank-a"
        assert _extract_cert_cn("O=CFL-Client, CN=sms-b") == "sms-b"
        assert _extract_cert_cn("O=OnlyOrg") is None
        assert _extract_cert_cn("") is None

    def test_gateway_enforces_mtls_cn_when_enabled(self, monkeypatch):
        app = create_app(
            models=InMemoryModelStore(),
            state_store=InMemoryStateStore(),
            client_registry=_StubRegistry(["bank-a"]),
        )
        http = TestClient(app)
        hdr = {"Authorization": "Bearer tok-bank-a"}

        # Default (CFL_REQUIRE_MTLS unset): bearer token alone succeeds (204 no task).
        monkeypatch.delenv("CFL_REQUIRE_MTLS", raising=False)
        assert http.get("/task", headers=hdr).status_code == 204

        # Enable strict mTLS CN binding.
        monkeypatch.setenv("CFL_REQUIRE_MTLS", "true")

        # Missing cert subject header -> 403 missing_client_cert
        r_missing = http.get("/task", headers=hdr)
        assert r_missing.status_code == 403
        assert r_missing.json()["error"]["code"] == "missing_client_cert"

        # Wrong cert CN -> 403 client_cert_mismatch
        r_wrong = http.get("/task", headers={**hdr, "X-Client-Cert-Subject": "CN=bank-b,O=CFL"})
        assert r_wrong.status_code == 403
        assert r_wrong.json()["error"]["code"] == "client_cert_mismatch"

        # Matching cert CN -> 204 allowed
        r_ok = http.get("/task", headers={**hdr, "X-Client-Cert-Subject": "CN=bank-a,O=CFL-Client"})
        assert r_ok.status_code == 204


class TestMTLSCertScript:
    def test_script_generates_valid_ca_and_client_cert(self, tmp_path):
        env = {**os.environ, "CFL_MTLS_DIR": str(tmp_path / "mtls")}
        subprocess.run(["sh", str(MTLS_SCRIPT), "init"], env=env, check=True, capture_output=True)
        subprocess.run(
            ["sh", str(MTLS_SCRIPT), "client", "bank-a"], env=env, check=True, capture_output=True
        )

        ca_cert = x509.load_pem_x509_certificate((tmp_path / "mtls" / "ca.crt").read_bytes())
        client_cert = x509.load_pem_x509_certificate(
            (tmp_path / "mtls" / "clients" / "bank-a" / "bank-a.crt").read_bytes()
        )
        cn = client_cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
        assert cn == "bank-a"
        assert client_cert.issuer == ca_cert.subject
        eku = client_cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        assert x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH in eku
