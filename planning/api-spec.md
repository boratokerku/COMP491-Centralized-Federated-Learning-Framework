# CFL Gateway REST API & Message Specification

**Document Version:** 1.1.0 (aligned with the implementation, 2026-11-04)  
**Author / maintainer:** Enes Tunç (etunc22@ku.edu.tr)  
**Related Work Packages:** WP1 (T1.3 Interface Specification), WP3 (T3.1 Secure API Gateway)  
**Deliverable Association:** D1.3 (API and Message Specification)  
**Implementation:** [`backend/src/cfl/gateway/app.py`](../backend/src/cfl/gateway/app.py) (client API), [`backend/src/cfl/protocol/messages.py`](../backend/src/cfl/protocol/messages.py) (message models), [`backend/src/cfl/conformance/checker.py`](../backend/src/cfl/conformance/checker.py) (conformance validation)  
**Related Planning Documents:**
- [`db-schema.md`](db-schema.md) — Shared PostgreSQL 16 persistence contract
- [`conformance-manifest.md`](conformance-manifest.md) — Client conformance & semantic validation contract
- [`interfaces.md`](interfaces.md) — Team decisions (sections 6, 7 and 9 concern this API)
- Sequence Diagrams: [`01_registration_conformance.puml`](../diagrams/sequence/01_registration_conformance.puml), [`02_normal_round.puml`](../diagrams/sequence/02_normal_round.puml), [`03_dropout_reconnection.puml`](../diagrams/sequence/03_dropout_reconnection.puml), [`04_crash_resume.puml`](../diagrams/sequence/04_crash_resume.puml)

**Changes in 1.1.0:** manifest identity `"<dataset>@<version>"`; new `GET /conformance`; conformance report fields for categories, tokenizer, label and forward tests, and check 13 `forward_output_mismatch`; registration required before every other endpoint; model downloads limited to participants; final model `format` values `onnx`, `safetensors`, `artifact`, `manifest`; idempotent retries acknowledged before stale checks; SHA-256 token hashes; protocol version `0.1`; public port 443; the web read-only API (section 4) is served by the website, not by the gateway.

---

## 1. Executive Summary & Architecture Context

The Centralized Federated Learning (CFL) Gateway is the sole network entry point for federated client nodes. Implemented in **FastAPI** and served behind a **Caddy** reverse proxy with TLS termination, the Gateway provides a language-agnostic, secure REST API.

The Gateway is decoupled from the Coordinator process:
- **Clients always initiate HTTP requests** (`POST /register`, `GET /task`, `GET /model/...`, `POST /update`, `GET /model/.../final`). This client-pull / client-push paradigm eliminates inbound firewall and NAT traversal constraints on client institutions.
- **The Gateway and Coordinator communicate exclusively through PostgreSQL and the model storage volume.** The Gateway records client registration and conformance, accepts updates, serves models, and writes audit events and communication metrics. The Coordinator orchestrates the round state machine (`OPEN` → `COLLECTING` → `AGGREGATING` → `DONE` / `FAILED`), selects participants and aggregates updates without ever talking to clients.
- **The Gateway remains online during coordinator crashes.** Client polling and update submission continue; updates accepted during the outage count when the coordinator resumes.
- **The round rules exist once.** The Gateway uses the Coordinator's read-only methods for task assignment and the final artifact, and the same update intake rules as the local simulation (`cfl.coordinator.intake`), so simulation and real deployments behave identically.

```
  Internet, port 443
 clients  ──► fl.<ip>.sslip.io ──┐                        ┌──► Gateway (FastAPI) ──┐
 browsers ──► <ip>.sslip.io ─────┴──► Caddy (TLS) ────────┼──► Website (Next.js) ──┤ read-only role
                                                          └──► Grafana ────────────┤
                                                                                   v
                                                         PostgreSQL 16 + model storage volume
                                                                                   ^
                                                          Coordinator process ─────┘
```

---

## 2. Common Protocols, Conventions & Error Handling

### 2.1 Transport & Base URL
- **Protocol:** HTTPS only. Caddy terminates TLS (TLS 1.2 and 1.3) with a Let's Encrypt certificate; plain HTTP is only redirected to HTTPS, never served.
- **Base URL:** `https://fl.<server-ip>.sslip.io/` on port **443**. Institutional networks often block other outbound ports, so no other port is used. (The gateway process itself listens on 8000 inside the compose network only.)
- **Serialization Formats:**
  - Control & metadata plane: `application/json` (UTF-8)
  - Model weights distribution: `application/octet-stream` (binary `safetensors` or ONNX)
  - Update upload plane: `multipart/form-data` (JSON metadata + binary `safetensors`)
- **Request size:** uploads up to 64 MB (enforced by Caddy).

### 2.2 Authentication & Authorization
All client endpoints (except `/health`) require an HTTP Bearer token in the `Authorization` header:
```http
Authorization: Bearer <client token>
```
- **Token issuance:** an administrator approves a client and receives its token once: `cfl-admin add-client <client_id>` (`--token-file <file>` writes it to a mode-600 file instead of printing it). The token is delivered out of band and stored by the site as a Docker secret. Tokens are never baked into container images, committed to Git or stored by the server. Further commands: `cfl-admin rotate-token <id>`, `disable <id>`, `enable <id>`, `list`.
- **Token format and storage:** `cfl_` followed by 43 URL-safe characters (256 random bits). `clients.token_hash` stores its SHA-256. A slow salted hash (Argon2/bcrypt) protects guessable passwords; these tokens cannot be guessed, and an unsalted hash lets the Gateway find the client with one indexed lookup (GitHub stores its API tokens the same way).
- **Identity binding:** the client identity comes from the token. A request whose body claims another `client_id` is rejected with `403 identity_mismatch`.
- **Registration first:** every endpoint except `POST /register` requires a client with status `REGISTERED` (`403 not_registered`).
- **Activity tracking:** each authenticated request updates `clients.last_seen_at`.

### 2.3 Identifiers & Versions
- `update_id` is a client-generated **UUIDv4**, reused on retries (`422 invalid_update_id` otherwise).
- `run_id` and `report_id` are UUIDv4 strings. `round_id` starts at 1; `model_version` starts at 0 (the initial model) and increases by one per successful round.
- **Manifest identity:** `"<dataset>@<version>"`, e.g. `bank_marketing@1.0.0` or `sms_spam@1.0.0`. The bare version is ambiguous (both manifests are `1.0.0`), and passing one manifest must not make a client eligible for a run of the other task.
- **Protocol version:** `0.1` (`cfl.protocol.PROTOCOL_VERSION`).

### 2.4 Unified Status Code Matrix

| HTTP Status Code | Meaning | Error codes |
|:---|:---|:---|
| **200 OK** | Request succeeded (also an idempotent update retry). | — |
| **204 No Content** | Polling succeeded, but there is no task for this client now. | — |
| **401 Unauthorized** | Missing, malformed or unknown Bearer token (with `WWW-Authenticate: Bearer`). | `missing_token`, `invalid_token` |
| **403 Forbidden** | Authenticated but not allowed. | `client_disabled`, `not_registered`, `identity_mismatch`, `version_not_approved`, `not_participant`, `not_selected` |
| **404 Not Found** | Unknown run, round, manifest or model version. | `unknown_run`, `unknown_round`, `unknown_manifest`, `unknown_model_version`, `no_final_model` |
| **409 Conflict** | State conflict: duplicate or stale update, or run still in progress. | `duplicate_update`, `stale_round`, `deadline_passed`, `model_version_mismatch`, `run_in_progress` |
| **422 Unprocessable Entity** | Malformed request, protocol mismatch, failed conformance or integrity check. | `invalid_request`, `invalid_metadata`, `invalid_format`, `invalid_update_id`, `protocol_version_mismatch`, `checksum_mismatch`, conformance reason codes |
| **500 Internal Error** | A stored file does not match its recorded checksum, or another server fault. | `integrity_error` |

### 2.5 Standard Error Response Schema
All error responses (`4xx` and `5xx`) use one JSON shape (`ErrorResponse`):
```json
{
  "error": {
    "code": "identity_mismatch",
    "message": "client_id does not match the token",
    "details": {"token_client_id": "bank-a", "claimed_client_id": "bank-b"},
    "timestamp": "2026-11-04T15:16:59Z"
  }
}
```
Two endpoints answer with their own body instead because the outcome is a result, not an error: a failed conformance report (`422`, `ConformanceReportResponse`, section 3.3) and a rejected update (`403`/`404`/`409`/`422`, `UpdateAck`, section 3.6).

---

## 3. Client Protocol Endpoints (FL Core Lifecycle)

```
+----------------------------------------------------------------------------------------------------+
|                                CLIENT PROTOCOL LIFECYCLE SUMMARY                                   |
+----------------------------------------------------------------------------------------------------+
| 1. POST /register                  -> Client registration, protocol and version check              |
| 2. GET  /conformance               -> List the manifests of the running runs                       |
| 3. GET  /conformance/{manifest}    -> Fetch the manifest and its synthetic reference cases         |
| 4. POST /conformance/report        -> Submit the local validation report (re-checked by server)    |
| 5. GET  /task                      -> Poll for an active round assignment (200 / 204)              |
| 6. GET  /model/{run_id}/{version}  -> Download global model weights (.safetensors + SHA-256)       |
| 7. POST /update                    -> Upload local update weights & metrics (idempotent)           |
| 8. GET  /model/{run_id}/final      -> Download the final verified model (ONNX / safetensors)       |
+----------------------------------------------------------------------------------------------------+
```

---

### 3.1 `POST /register`
Registers the client for training runs. Managed-code client images and independent implementations announce their identity, version and protocol.

- **Sequence Diagram Reference:** `01_registration_conformance.puml`
- **Authentication:** Bearer token (required); the client may have status `APPROVED` or `REGISTERED` (re-registration updates the version).

#### Request Body (`RegisterRequest`)
```json
{
  "client_id": "bank-a",
  "client_version": "cfl-client/0.1.0",
  "protocol_version": "0.1"
}
```
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `client_id` | `string` | Yes | Must equal the token's client. |
| `client_version` | `string` | Yes | Managed image version (`cfl-client/<version>`, optionally with `@sha256:<digest>`) or independent client release. |
| `protocol_version` | `string` | No (default `0.1`) | Must equal the server's protocol version. |

The administrator can restrict accepted versions with `CFL_ALLOWED_CLIENT_VERSIONS` (comma-separated patterns, e.g. `cfl-client/0.1.*,independent-client/*`); unset accepts every version.

#### Response (`RegisterResponse`)
- **`200 OK`:**
```json
{
  "accepted": true,
  "client_id": "bank-a",
  "status": "REGISTERED",
  "manifest_version": "bank_marketing@1.0.0",
  "protocol_version": "0.1",
  "reason": null,
  "message": "registered; fetch GET /conformance and submit a conformance report"
}
```
  `manifest_version` is set when exactly one manifest is active; otherwise it is `null` and the client uses `GET /conformance`.
- **`401`:** `missing_token`, `invalid_token`. **`403`:** `client_disabled`, `identity_mismatch`, `version_not_approved`. **`422`:** `protocol_version_mismatch`.

#### Database Operations
- `clients`: `status = 'REGISTERED'`, `client_version`, `protocol_version`, `registered_at = now()`, `last_seen_at = now()`.
- `events`: `CLIENT_REGISTERED` (`message_type = 'register'`).

---

### 3.2 `GET /conformance` and `GET /conformance/{manifest_version}`
`GET /conformance` lists the manifests pinned by the running runs, so a client picks the one for its dataset:
```json
[
  {"manifest_version": "bank_marketing@1.0.0", "dataset": "bank_marketing", "manifest_sha256": "<64 hex>"},
  {"manifest_version": "sms_spam@1.0.0", "dataset": "sms_spam", "manifest_sha256": "<64 hex>"}
]
```

`GET /conformance/{manifest_version}` serves that manifest exactly as the run pinned it (`<run_id>/manifest.json`).

- **`200 OK`:** the manifest JSON, conforming to [`conformance-manifest.schema.json`](conformance-manifest.schema.json). Generated examples: [`examples/bank_marketing_manifest.json`](examples/bank_marketing_manifest.json), [`examples/sms_spam_manifest.json`](examples/sms_spam_manifest.json).
  - `ETag: "<sha256>"` and `X-Manifest-Checksum-SHA256: <sha256>` — the client verifies the body against it and quotes it in its report.
- **`404 unknown_manifest`:** no running run pins this manifest.

A manifest contains the model contract (architecture, exact parameter names/shapes/dtypes, fingerprint), the raw input schema (features in order with types, units, fixed scaling statistics and category lists; or the hashing tokenizer for text), the label definition, synthetic `reference_tests` (raw input → expected model input), `forward_reference_tests` (model input → expected logits under the deterministic reference weights `reference_sin_v1`) and the tolerances. See [`conformance-manifest.md`](conformance-manifest.md).

---

### 3.3 `POST /conformance/report`
The client runs its own pre-processing on the manifest's synthetic raw inputs and its own forward pass on the forward reference inputs, declares what it implements, and posts the report. No private records are shared. The Gateway re-checks everything; `client_passed` is informational only.

- **Sequence Diagram Reference:** `01_registration_conformance.puml`
- **Reference implementation (client side):** `cfl.conformance.report.build_report`

#### Request Body (`ConformanceReportRequest`)
```json
{
  "manifest_version": "bank_marketing@1.0.0",
  "manifest_sha256": "<64 hex, from X-Manifest-Checksum-SHA256>",
  "client_id": "bank-a",
  "client_version": "cfl-client/0.1.0",
  "protocol_version": "0.1",
  "declared_model": {
    "architecture_id": "bank_mlp",
    "architecture_version": "1.0.0",
    "fingerprint": "<64 hex, optional>",
    "parameters": [
      {"name": "fc1.weight", "shape": [64, 65], "dtype": "float32"},
      {"name": "fc1.bias", "shape": [64], "dtype": "float32"},
      {"name": "fc2.weight", "shape": [32, 64], "dtype": "float32"},
      {"name": "fc2.bias", "shape": [32], "dtype": "float32"},
      {"name": "fc3.weight", "shape": [2, 32], "dtype": "float32"},
      {"name": "fc3.bias", "shape": [2], "dtype": "float32"}
    ]
  },
  "declared_input": {
    "feature_names": ["age", "campaign", "pdays", "...", "poutcome"],
    "feature_types": ["numeric", "numeric", "numeric", "...", "categorical"],
    "feature_units": ["years", "contacts", "days", "...", "category"],
    "preprocessing_version": "1.0",
    "categories": {"job": ["admin.", "blue-collar", "..."], "poutcome": ["failure", "nonexistent", "success"]},
    "tokenizer": null
  },
  "declared_label": {"column": "y", "positive_value": "yes", "negative_value": "no", "encoded_values": [0, 1]},
  "client_passed": true,
  "checks": [{"code": "conformant", "passed": true}],
  "reference_outputs": [
    {"test_id": "bank-marketing-reference-001", "output_features": [0.0, 0.0, 0.0, 1.0, "..."]}
  ],
  "forward_outputs": [
    {"test_id": "bank-marketing-forward-001", "logits": [-0.030145511, -0.065617085]}
  ],
  "client_failure_codes": []
}
```
- `feature_units[i]` is the numeric unit, `"category"` for a categorical and `"text"` for a text feature.
- `categories` lists every categorical feature's vocabulary in manifest order. Text tasks send the manifest's `preprocessing.tokenizer` object as `tokenizer`.
- Reports include every manifest reference and forward test exactly once.

#### Deterministic Validation Pipeline
The Gateway checks in this order and reports the first failure:
1. `protocol_version_mismatch`
2. `manifest_version_mismatch` (identity or SHA-256 differ from the active manifest)
3. `model_version_mismatch` (architecture id/version)
4. `model_shape_mismatch` (parameter names, shapes, dtypes, fingerprint)
5. `feature_schema_mismatch` (feature count, presence, types)
6. `feature_order_mismatch`
7. `unit_mismatch` (e.g. days vs. months)
8. `encoding_mismatch` (category lists and their order; tokenizer)
9. `preprocessing_version_mismatch`
10. `label_semantics_mismatch` (label column, positive/negative meaning)
11. `label_range_mismatch` (encoded label values)
12. `reference_output_mismatch` (pre-processing outputs, `validation.reference_atol`)
13. `forward_output_mismatch` (logits under the reference weights, `validation.forward_atol`; catches transposed or reordered weights)
14. `invalid_report` (where found: missing/unknown test ids, wrong lengths, non-finite numbers, inconsistent declarations)

#### Response (`ConformanceReportResponse`)
- **`200 OK` (passed):**
```json
{
  "status": "PASSED",
  "client_id": "bank-a",
  "manifest_version": "bank_marketing@1.0.0",
  "report_id": "550e8400-e29b-41d4-a716-446655440000",
  "reason_code": null,
  "message": "eligible for round selection",
  "details": {},
  "checked_at": "2026-11-04T15:20:00Z"
}
```
- **`422` (failed):**
```json
{
  "status": "FAILED",
  "client_id": "bank-wrong",
  "manifest_version": "bank_marketing@1.0.0",
  "report_id": "550e8400-e29b-41d4-a716-446655440001",
  "reason_code": "feature_order_mismatch",
  "message": "declared feature order does not match the manifest ordering",
  "details": {"index": 0, "expected": "age", "received": "campaign"},
  "checked_at": "2026-11-04T15:20:01Z"
}
```
- **`401` / `403`:** token invalid, client disabled or not registered, `identity_mismatch`.

#### Database Operations
- `conformance_reports`: one row per attempt (`report_id`, `client_id`, `manifest_version`, `client_version`, `protocol_version`, `status`, `reason_code`, `report_payload` = the report plus the server's decision and details, `checked_at`).
- `events`: `CONFORMANCE_EVALUATED` (`status`, `reason_code`).
- The coordinator selects only clients whose **latest** report for the run's manifest identity is `PASSED` and whose status is not `DISABLED`.

---

### 3.4 `GET /task`
Task polling (e.g. every 2–5 seconds while a client is idle).

- **Sequence Diagram Reference:** `02_normal_round.puml`, `03_dropout_reconnection.puml`
- **Rules:** the client gets a task when a run is `RUNNING`, its latest round is `COLLECTING`, and the client is a `SELECTED` participant of that round that has not delivered its update yet.

#### Response
- **`200 OK` (`TaskAssignment`):**
```json
{
  "run_id": "d3b07384-d113-4966-bfbe-08f321cd4943",
  "round_id": 1,
  "model_version": 0,
  "deadline": "2026-11-04T15:21:30Z",
  "config": {
    "strategy": "fedavg",
    "local_epochs": 1,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "proximal_mu": 0.0,
    "class_weights": [1.0, 7.9]
  }
}
```
| Field | Description |
|:---|:---|
| `model_version` | Global model version to download and train on. |
| `deadline` | UTC; updates received later are rejected as stale (decided with the server's clock). |
| `config` | Round hyperparameters, fixed for the federation (`interfaces.md`, section 3). `proximal_mu > 0` means FedProx. |

- **`204 No Content`:** not selected, no collecting round, or the update for this round was already accepted.
- **`401` / `403`:** token invalid, client disabled or not registered.

---

### 3.5 `GET /model/{run_id}/{version}`
Downloads a global model version.

- **Sequence Diagram Reference:** `02_normal_round.puml`, `03_dropout_reconnection.puml`
- **Access:** clients selected in at least one round of the run.

#### Response
- **`200 OK`:** binary safetensors.
  - `Content-Disposition: attachment; filename="global_model_v{version}.safetensors"`
  - `X-Checksum-SHA256: <sha256 recorded in the database>`
  - `X-Model-Version`, `X-Run-ID`
- **`403 not_participant`**, **`404`** `unknown_run` / `unknown_model_version`, **`500 integrity_error`** (stored file does not match its recorded checksum; never served).

#### Client-Side Action
Compute the SHA-256 of the body and compare it with `X-Checksum-SHA256` before loading the parameters.

---

### 3.6 `POST /update`
Uploads the local update after training.

- **Sequence Diagram Reference:** `02_normal_round.puml`, `03_dropout_reconnection.puml`
- **Content type:** `multipart/form-data` with two fields:
  1. `metadata`: JSON string conforming to `UpdateMetadata`
  2. `weights`: the `.safetensors` file

```json
{
  "run_id": "d3b07384-d113-4966-bfbe-08f321cd4943",
  "round_id": 1,
  "client_id": "bank-a",
  "update_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "model_version": 0,
  "num_samples": 5766,
  "checksum_sha256": "<64 hex of the weights file>",
  "metrics": {
    "train_loss": 0.43,
    "train_s": 0.35,
    "val_f1_before": 0.01,
    "val_f1_after": 0.42,
    "val_roc_auc_after": 0.79
  }
}
```
| Field | Description |
|:---|:---|
| `client_id` | Must equal the token's client (`403 identity_mismatch`). |
| `update_id` | UUIDv4 idempotency key; a retry reuses it. |
| `model_version` | Must equal the round's input model version. |
| `num_samples` | Local training samples; the FedAvg weight. The coordinator rejects implausible values (`rounds.max_num_samples`). |
| `metrics` | Names and units: `interfaces.md`, section 4 (`train_loss`, `train_s`, `val_<metric>_before`, `val_<metric>_after`). |

#### Decision order
1. `update_id` not a UUID → **`422 invalid_update_id`**
2. `update_id` already accepted for this client and round → **`200`**, `reason: "duplicate_idempotent"`, counted once. This comes before every state check so that a client whose response was lost never sees its accepted update reported as stale after the round moved on.
3. SHA-256 of `weights` ≠ `checksum_sha256` → **`422 checksum_mismatch`**
4. Unknown run or round → **`404 unknown_round`**
5. Client not selected for the round → **`403 not_selected`**
6. Round not `COLLECTING` → **`409 stale_round`**; deadline passed → **`409 deadline_passed`**
7. `model_version` ≠ round input version → **`409 model_version_mismatch`**
8. Another update already accepted for this client and round → **`409 duplicate_update`**
9. Otherwise: the weights file is written atomically, then in one transaction the `updates` row is inserted (`UNIQUE (run_id, round_id, client_id)`), the participant set to `UPDATE_ACCEPTED`, and an `UPDATE_ACCEPTED` event written → **`200`**.

Rejections write an `UPDATE_REJECTED` event with the reason; acknowledged retries write `UPDATE_RETRY_ACKNOWLEDGED`.

#### Response (`UpdateAck`)
```json
{"accepted": true, "update_id": "9b1deb4d-...", "client_id": "bank-a", "round_id": 1, "reason": null}
```
```json
{"accepted": false, "update_id": "6f2c...", "client_id": "bank-a", "round_id": 1, "reason": "duplicate_update"}
```
A malformed `metadata` field (`422 invalid_metadata`) and a `client_id` other than the token's (`403 identity_mismatch`) return the error schema instead, because the update was never considered.

---

### 3.7 `GET /model/{run_id}/final`
Downloads the final model artifact of a completed run.

- **Sequence Diagram Reference:** `02_normal_round.puml` (final model delivery)
- **Access:** registered clients.
- **Query parameter `format`:** `safetensors` (default), `onnx`, `artifact` (`artifact.json`: versions, file checksums, reference predictions) or `manifest`.

#### Response
- **`200 OK`:** the requested file.
  - `X-Checksum-SHA256`: SHA-256 of this file
  - `X-Final-Model-SHA256`: SHA-256 of `model.onnx` as recorded in `runs.final_model_sha256`
  - `X-Manifest-Version`, `X-Model-Version`, `X-Run-ID`
- **`404`** `unknown_run` / `no_final_model`, **`409 run_in_progress`**, **`422 invalid_format`**.

#### Client-Side Action
Download `onnx` and `artifact` (and optionally `manifest`, `safetensors`), then call `cfl.core.artifact.verify_artifact(files, X-Final-Model-SHA256)`: it checks every checksum and reproduces the reference predictions with ONNX Runtime. Pre-processing (scaling, one-hot encoding or tokenization) is not part of the ONNX graph; apply it as described in `artifact.json` → `preprocessing`.

---

## 4. Web Application Read-Only Endpoints (`/api/v1/...`)

The website (owner: Hilal) implements these endpoints itself, as Next.js route handlers on `https://<server-ip>.sslip.io`; they are **not** part of the gateway, so the internet-facing client API stays minimal (`interfaces.md`, section 6). They read PostgreSQL through the read-only role `cfl_web` (SELECT only, no access to `clients.token_hash`) and report files from the model volume, mounted read-only.

Authentication: website session (username and password of the internal area, httpOnly cookie). Client tokens are never accepted here.

```
+----------------------------------------------------------------------------------------------------+
|                                WEB DASHBOARD ENDPOINTS SUMMARY                                     |
+----------------------------------------------------------------------------------------------------+
| GET /api/v1/runs                       -> List all runs (status, strategy, model version)          |
| GET /api/v1/runs/{run_id}              -> Run metadata, configuration & progress                   |
| GET /api/v1/runs/{run_id}/rounds       -> Rounds of a run (states, timings, quorum)                |
| GET /api/v1/runs/{run_id}/rounds/{id}  -> Round view (participants & per-client status)            |
| GET /api/v1/clients                    -> Clients & latest conformance status                      |
| GET /api/v1/clients/{client_id}        -> Client history & conformance records                     |
| GET /api/v1/runs/{run_id}/rounds/{id}/updates -> Update inbox of a round                           |
| GET /api/v1/events                     -> Filterable audit event log                               |
| GET /api/v1/runs/{run_id}/reports      -> Run report artifacts (JSON, Markdown, CSV, PNG plots)    |
+----------------------------------------------------------------------------------------------------+
```

### 4.1 Runs Endpoints

#### `GET /api/v1/runs`
- **Query Parameters:** `status` (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `ABORTED`), `limit` (default 50), `offset` (default 0).
- **Response `200 OK`:**
```json
{
  "total": 1,
  "runs": [
    {
      "run_id": "d3b07384-d113-4966-bfbe-08f321cd4943",
      "name": "demo-tabular-fedavg",
      "status": "RUNNING",
      "strategy": "fedavg",
      "task_name": "tabular",
      "seed": 42,
      "model_version": 3,
      "manifest_version": "bank_marketing@1.0.0",
      "started_at": "2026-11-04T15:00:00Z",
      "completed_at": null,
      "created_at": "2026-11-04T15:00:00Z"
    }
  ]
}
```

#### `GET /api/v1/runs/{run_id}`
Full run: immutable `config`, `config_sha256`, versions (protocol, manifest, preprocessing, model architecture and fingerprint), current model pointer, final model pointer (`final_model_path`, `final_model_sha256`, `final_model_format`, `final_model_bytes`), timestamps and `failure_reason`.

### 4.2 Rounds Endpoints

#### `GET /api/v1/runs/{run_id}/rounds`
```json
{
  "run_id": "d3b07384-d113-4966-bfbe-08f321cd4943",
  "rounds": [
    {
      "round_id": 1,
      "state": "DONE",
      "input_model_version": 0,
      "output_model_version": 1,
      "min_participants": 2,
      "selected_count": 4,
      "accepted_count": 3,
      "opened_at": "2026-11-04T15:00:00Z",
      "deadline_at": "2026-11-04T15:01:30Z",
      "closed_at": "2026-11-04T15:01:30Z",
      "failure_reason": null
    }
  ]
}
```

#### `GET /api/v1/runs/{run_id}/rounds/{round_id}`
The round plus its participants with status (`SELECTED`, `UPDATE_ACCEPTED`, `DROPOUT`, `REJECTED`), `reason_code` (e.g. `deadline_no_update`) and `finished_at`, and the round's metrics (`global_*`, durations).

### 4.3 Clients & Conformance Endpoints

#### `GET /api/v1/clients`
```json
{
  "clients": [
    {
      "client_id": "bank-a",
      "status": "REGISTERED",
      "client_version": "cfl-client/0.1.0",
      "protocol_version": "0.1",
      "last_seen_at": "2026-11-04T15:16:59Z",
      "registered_at": "2026-11-04T15:00:00Z",
      "latest_conformance": {
        "manifest_version": "bank_marketing@1.0.0",
        "status": "PASSED",
        "reason_code": null,
        "checked_at": "2026-11-04T15:00:05Z"
      }
    }
  ]
}
```

#### `GET /api/v1/clients/{client_id}`
Conformance attempts (with the server's reason code and details), participation history per round, and the client's reported metrics.

### 4.4 Updates Inbox

#### `GET /api/v1/runs/{run_id}/rounds/{round_id}/updates`
Accepted updates of the round (`update_id`, `client_id`, `model_version`, `num_samples`, `checksum_sha256`, `payload_bytes`, `received_at`, `metrics`), plus rejected uploads from the `UPDATE_REJECTED` events with their reasons.

### 4.5 Audit Event Log

#### `GET /api/v1/events`
- **Query Parameters:** `run_id`, `round_id`, `client_id`, `event_type`, `status`, `limit` (default 100), `offset`.
- **Event types:**

| Area | Event types |
|:---|:---|
| Clients | `CLIENT_APPROVED`, `CLIENT_TOKEN_ROTATED`, `CLIENT_DISABLED`, `CLIENT_ENABLED`, `CLIENT_REGISTERED`, `CONFORMANCE_EVALUATED` |
| Runs | `RUN_STARTED`, `COORDINATOR_RESUMED`, `RUN_COMPLETED`, `RUN_FAILED` |
| Rounds | `ROUND_OPENED`, `ROUND_COLLECTING`, `ROUND_AGGREGATING`, `ROUND_DONE`, `ROUND_FAILED`, `CLIENT_DROPOUT` |
| Updates | `UPDATE_ACCEPTED`, `UPDATE_REJECTED`, `UPDATE_RETRY_ACKNOWLEDGED`, `UPDATE_EXCLUDED` (rejected by the aggregator) |
| Transfers | `MODEL_DOWNLOADED`, `FINAL_MODEL_DOWNLOADED` |

```json
{
  "total": 1,
  "events": [
    {
      "event_id": 1042,
      "run_id": "d3b07384-d113-4966-bfbe-08f321cd4943",
      "round_id": 2,
      "client_id": "bank-c",
      "event_type": "UPDATE_REJECTED",
      "message_type": "update_upload",
      "status": null,
      "reason_code": "deadline_passed",
      "occurred_at": "2026-11-04T15:11:05Z",
      "duration_ms": null,
      "message_bytes": 26112,
      "details": {"update_id": "6f2c...", "message_type": "update_upload", "bytes": 26112, "reason_code": "deadline_passed"}
    }
  ]
}
```

### 4.6 Reports & Metrics Exports
The end-of-run report (owner: Hilal) is written to the model volume; these endpoints serve it.

- `GET /api/v1/runs/{run_id}/reports` — links to the files below
- `GET /api/v1/runs/{run_id}/reports/summary` — machine-readable summary JSON (configuration, versions, participation per round, metric summaries)
- `GET /api/v1/runs/{run_id}/reports/markdown` — human-readable report
- `GET /api/v1/runs/{run_id}/reports/metrics.csv` — every `metrics` row of the run
- `GET /api/v1/runs/{run_id}/reports/plots/{plot_name}` — generated PNG plots

### 4.7 Gateway Health Check

#### `GET /health` (gateway, no authentication)
```json
{"status": "ok", "version": "0.1.0", "protocol_version": "0.1", "database": "ok"}
```
`database` is `"unavailable"` when PostgreSQL cannot be reached; the endpoint itself still answers. Clients also use it to measure round-trip time (`rtt_ms`).

---

## 5. Sequence Diagram Traceability Matrix

| Sequence Diagram | Message / Arrow Description | HTTP Endpoint | Method | Expected Status Codes |
|:---|:---|:---|:---|:---|
| **01 (Registration)** | Client registers with version & token | `/register` | `POST` | `200`, `401`, `403`, `422` |
| **01 (Registration)** | Client lists and downloads the manifest | `/conformance`, `/conformance/{manifest_version}` | `GET` | `200`, `401`, `403`, `404` |
| **01 (Registration)** | Client submits its conformance report | `/conformance/report` | `POST` | `200` (passed), `422` (failed), `401`, `403` |
| **02 (Normal Round)** | Client polls for its task | `/task` | `GET` | `200` (assigned), `204` (no task), `401`, `403` |
| **02 (Normal Round)** | Client downloads the global model | `/model/{run_id}/{version}` | `GET` | `200`, `401`, `403`, `404` |
| **02 (Normal Round)** | Client uploads its update | `/update` | `POST` | `200`, `403`, `404`, `409`, `422` |
| **02 (Normal Round)** | Client downloads the final model | `/model/{run_id}/final` | `GET` | `200`, `401`, `403`, `404`, `409`, `422` |
| **03 (Dropout/Retry)** | Retried update after a lost response | `/update` (same `update_id`) | `POST` | `200` (`duplicate_idempotent`, counted once) |
| **03 (Dropout/Retry)** | Late update after the deadline | `/update` | `POST` | `409` (`deadline_passed` / `stale_round`) |
| **03 (Dropout/Retry)** | Reconnecting client polls a later round | `/task` | `GET` | `200` (assigned to t+1), `204` |
| **04 (Crash/Resume)** | Client polls/uploads while the coordinator restarts | `/task`, `/update` | `GET`/`POST` | Unchanged; the gateway works on PostgreSQL directly |

---

## 6. Implementation Notes

1. **Message models:** [`backend/src/cfl/protocol/messages.py`](../backend/src/cfl/protocol/messages.py).
2. **Gateway:** [`backend/src/cfl/gateway/app.py`](../backend/src/cfl/gateway/app.py) (routes), [`auth.py`](../backend/src/cfl/gateway/auth.py) (Bearer authentication, error schema), [`clients.py`](../backend/src/cfl/gateway/clients.py) (tokens, registration, conformance records), [`admin.py`](../backend/src/cfl/gateway/admin.py) (`cfl-admin`).
3. **Rules shared with the simulation:** update intake [`coordinator/intake.py`](../backend/src/cfl/coordinator/intake.py); task assignment and final artifact [`coordinator/coordinator.py`](../backend/src/cfl/coordinator/coordinator.py); conformance [`conformance/checker.py`](../backend/src/cfl/conformance/checker.py).
4. **Deployment:** [`deploy/README.md`](../deploy/README.md). Gateway settings: `CFL_DATABASE_URL`, `CFL_MODELS_DIR` (default `/models`), `CFL_ALLOWED_CLIENT_VERSIONS`.
5. **Automated tests** for every status code and the retry scenarios are still to be written (`backend/tests/test_gateway.py`).
