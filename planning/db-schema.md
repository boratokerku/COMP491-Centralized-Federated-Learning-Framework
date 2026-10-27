# PostgreSQL Database Schema

**Status:** design proposal for review with the team  
**Database:** PostgreSQL 16  
**Purpose:** shared persistence for the gateway, coordinator, and report generator.

The gateway and coordinator are separate processes. PostgreSQL is the source of truth for client approval, conformance, run/round state, selected clients, accepted updates, events, and metrics. Model and update tensors remain in file storage; the database stores their paths, checksums, versions, and sizes. Raw client data and bearer tokens are never stored here.

## Conventions

- IDs for clients are stable opaque text identifiers; run, update, and conformance report IDs are UUIDs.
- Timestamps use `TIMESTAMPTZ` and are written in UTC.
- Flexible run configuration, event details, metric dimensions, and conformance payloads use `JSONB`; fields required for filtering or integrity remain ordinary typed columns.
- `sha256` values are lowercase 64-character hex strings. File paths are relative to the configured model-storage root, not arbitrary client-provided paths.
- Each accepted upload is unique per `(run_id, round_id, client_id)`. A retry with the same `update_id` is idempotent; a different update ID for that same client and round is rejected.
- State changes and the corresponding event should be committed in one database transaction. Files are written atomically before metadata points to them; after a crash, the coordinator verifies file existence and checksum before resuming.

## Tables

### `clients`

Approved client identities and registration state. The raw bearer token is issued out of band and is never persisted.

| Column | PostgreSQL type | Rules / meaning |
|---|---|---|
| `client_id` | `TEXT` | **PK**; stable client identifier |
| `token_hash` | `TEXT` | **UNIQUE NOT NULL**; password-hash of bearer token, never the token |
| `status` | `TEXT` | **NOT NULL**, `CHECK` in `('APPROVED','REGISTERED','DISABLED')` |
| `client_version` | `TEXT` | nullable until registration; managed image tag/digest or independent-client release |
| `protocol_version` | `TEXT` | nullable until registration |
| `created_at` | `TIMESTAMPTZ` | **NOT NULL DEFAULT now()** |
| `registered_at` | `TIMESTAMPTZ` | nullable |
| `last_seen_at` | `TIMESTAMPTZ` | nullable; updated on authenticated requests |

### `runs`

One configured experiment. Stores the current model pointer for restart recovery and the final deliverable pointer for reporting.

| Column | PostgreSQL type | Rules / meaning |
|---|---|---|
| `run_id` | `UUID` | **PK** |
| `name` | `TEXT` | **NOT NULL** |
| `status` | `TEXT` | **NOT NULL**, `CHECK` in `('PENDING','RUNNING','COMPLETED','FAILED','ABORTED')` |
| `strategy` | `TEXT` | **NOT NULL**, e.g. `fedavg` or `fedprox` |
| `task_name` | `TEXT` | **NOT NULL**, e.g. `bank_marketing` |
| `seed` | `BIGINT` | **NOT NULL** |
| `config` | `JSONB` | **NOT NULL**; immutable effective run configuration |
| `protocol_version` | `TEXT` | **NOT NULL** |
| `model_version` | `INTEGER` | **NOT NULL DEFAULT 0 CHECK (model_version >= 0)`; latest committed global model |
| `model_path` | `TEXT` | nullable until initial model is saved; relative path in file storage |
| `model_sha256` | `CHAR(64)` | nullable until initial model is saved |
| `manifest_version` | `TEXT` | **NOT NULL**; conformance contract pinned for the run |
| `preprocessing_version` | `TEXT` | **NOT NULL** |
| `model_architecture` | `TEXT` | **NOT NULL** |
| `model_architecture_version` | `TEXT` | **NOT NULL** |
| `started_at` | `TIMESTAMPTZ` | nullable until run starts |
| `completed_at` | `TIMESTAMPTZ` | nullable |
| `failure_reason` | `TEXT` | nullable; terminal run failure reason |
| `final_model_path` | `TEXT` | nullable until final artifact is generated |
| `final_model_sha256` | `CHAR(64)` | nullable until final artifact is generated |
| `final_model_format` | `TEXT` | nullable; e.g. `onnx` or `safetensors` |
| `final_model_bytes` | `BIGINT` | nullable; final artifact size in bytes |
| `created_at` | `TIMESTAMPTZ` | **NOT NULL DEFAULT now()** |

### `rounds`

One attempted round within a run. Failed rounds are retained with their reason so the report can reconstruct skipped/failed work.

| Column | PostgreSQL type | Rules / meaning |
|---|---|---|
| `run_id` | `UUID` | **PK part**, **FK** → `runs.run_id` |
| `round_id` | `INTEGER` | **PK part**, **CHECK (round_id > 0)** |
| `state` | `TEXT` | **NOT NULL**, `CHECK` in `('OPEN','COLLECTING','AGGREGATING','DONE','FAILED')` |
| `input_model_version` | `INTEGER` | **NOT NULL**; global model version distributed for this round |
| `input_model_path` | `TEXT` | **NOT NULL**; immutable file path used to resume/reconstruct |
| `input_model_sha256` | `CHAR(64)` | **NOT NULL** |
| `output_model_version` | `INTEGER` | nullable until aggregation commits; normally input version + 1 |
| `output_model_path` | `TEXT` | nullable until aggregation commits |
| `output_model_sha256` | `CHAR(64)` | nullable until aggregation commits |
| `min_participants` | `INTEGER` | **NOT NULL CHECK (min_participants > 0)** |
| `deadline_at` | `TIMESTAMPTZ` | **NOT NULL** |
| `opened_at` | `TIMESTAMPTZ` | **NOT NULL** |
| `closed_at` | `TIMESTAMPTZ` | nullable; when terminal state is reached |
| `failure_reason` | `TEXT` | nullable; required when `state='FAILED'` |
| `state_version` | `BIGINT` | **NOT NULL DEFAULT 0**; incremented on each transition to detect stale writes |
| `created_at` | `TIMESTAMPTZ` | **NOT NULL DEFAULT now()** |

**Primary key:** `(run_id, round_id)`. State transition and event insertion are transactional. The coordinator may resume a non-terminal round using the persisted state, deadline, selected clients, and model path/checksum.

### `round_participants` (supporting association table)

One row per client selected for a round. This makes the selected-client list, participation, and dropouts queryable without encoding IDs in a JSON array.

| Column | PostgreSQL type | Rules / meaning |
|---|---|---|
| `run_id` | `UUID` | **PK part**, **FK** → `rounds(run_id, round_id)` |
| `round_id` | `INTEGER` | **PK part**, **FK** → `rounds(run_id, round_id)` |
| `client_id` | `TEXT` | **PK part**, **FK** → `clients.client_id` |
| `status` | `TEXT` | **NOT NULL**, `CHECK` in `('SELECTED','UPDATE_ACCEPTED','DROPOUT','REJECTED')` |
| `selected_at` | `TIMESTAMPTZ` | **NOT NULL DEFAULT now()** |
| `finished_at` | `TIMESTAMPTZ` | nullable |
| `reason_code` | `TEXT` | nullable; e.g. `deadline_no_update` |

**Primary key:** `(run_id, round_id, client_id)`. `updates` references this key, so an unselected client cannot contribute an update to that round.

### `updates`

Metadata for accepted weight uploads. Tensor bytes are stored in file storage, never in this table.

| Column | PostgreSQL type | Rules / meaning |
|---|---|---|
| `update_id` | `UUID` | **PK**; client-generated idempotency key, reused on retry |
| `run_id` | `UUID` | **NOT NULL**; part of round FK |
| `round_id` | `INTEGER` | **NOT NULL**; part of round FK |
| `client_id` | `TEXT` | **NOT NULL**; part of selected participant FK |
| `model_version` | `INTEGER` | **NOT NULL**; must equal round `input_model_version` |
| `num_samples` | `INTEGER` | **NOT NULL CHECK (num_samples > 0)**; aggregation weight |
| `weights_path` | `TEXT` | **NOT NULL**; relative path in file storage |
| `checksum_sha256` | `CHAR(64)` | **NOT NULL**; verified before acceptance |
| `payload_bytes` | `BIGINT` | **NOT NULL CHECK (payload_bytes >= 0)** |
| `metrics` | `JSONB` | **NOT NULL DEFAULT '{}'**; local before/after metrics and training metadata |
| `received_at` | `TIMESTAMPTZ` | **NOT NULL DEFAULT now()** |
| `upload_duration_ms` | `DOUBLE PRECISION` | nullable; server-observed transfer duration |

**Constraints:** `UNIQUE (run_id, round_id, client_id)`; **FK** `(run_id, round_id, client_id)` → `round_participants`; `CHECK` checksum matches 64 lowercase hex characters. `update_id` is globally unique as the primary key. A same-ID retry returns the prior acknowledgement; a different ID for the same client/round returns duplicate conflict.

### `conformance_reports`

Append-only record of each client check attempt. The manifest itself is a version-controlled JSON artifact distributed by the gateway; the run pins its version.

| Column | PostgreSQL type | Rules / meaning |
|---|---|---|
| `report_id` | `UUID` | **PK** |
| `client_id` | `TEXT` | **NOT NULL FK** → `clients.client_id` |
| `manifest_version` | `TEXT` | **NOT NULL** |
| `client_version` | `TEXT` | **NOT NULL** |
| `protocol_version` | `TEXT` | **NOT NULL** |
| `status` | `TEXT` | **NOT NULL**, `CHECK` in `('PASSED','FAILED')` |
| `reason_code` | `TEXT` | nullable on pass; stable code on failure |
| `report_payload` | `JSONB` | **NOT NULL**; submitted metadata and outputs for synthetic references only |
| `checked_at` | `TIMESTAMPTZ` | **NOT NULL DEFAULT now()** |

Add an index on `(client_id, manifest_version, checked_at DESC)`; the coordinator selects only the latest `PASSED` report matching the run's pinned manifest version. A changed dataset, feature contract, preprocessing, label semantics, or model contract requires a new manifest version.

### `events`

Append-only audit trail for registration, conformance, protocol messages, round transitions, retries, rejections, dropouts, and coordinator recovery.

| Column | PostgreSQL type | Rules / meaning |
|---|---|---|
| `event_id` | `BIGINT GENERATED ALWAYS AS IDENTITY` | **PK** |
| `run_id` | `UUID` | nullable **FK** → `runs.run_id` |
| `round_id` | `INTEGER` | nullable; paired with `run_id` in **FK** → `rounds(run_id, round_id)` |
| `client_id` | `TEXT` | nullable **FK** → `clients.client_id` |
| `event_type` | `TEXT` | **NOT NULL**; stable event name, e.g. `UPDATE_ACCEPTED`, `COORDINATOR_RESUMED` |
| `message_type` | `TEXT` | nullable; `register`, `task`, `model_download`, `update_upload`, `final_model_download` |
| `status` | `TEXT` | nullable; outcome such as `accepted`, `rejected`, `failed` |
| `reason_code` | `TEXT` | nullable stable rejection/failure code |
| `occurred_at` | `TIMESTAMPTZ` | **NOT NULL DEFAULT now()** |
| `duration_ms` | `DOUBLE PRECISION` | nullable; duration for this operation |
| `message_bytes` | `BIGINT` | nullable **CHECK (message_bytes >= 0)** |
| `details` | `JSONB` | **NOT NULL DEFAULT '{}'**; non-sensitive diagnostic context |

Use indexes on `(run_id, occurred_at)`, `(run_id, round_id, occurred_at)`, and `(client_id, occurred_at)`. Do not put bearer tokens, raw data, or weight contents in `details`.

### `metrics`

Normalized numeric measurements used by reports and Grafana. Keep one metric observation per row; dimensions identify run/round/client and `metric_name` identifies the measurement.

| Column | PostgreSQL type | Rules / meaning |
|---|---|---|
| `metric_id` | `BIGINT GENERATED ALWAYS AS IDENTITY` | **PK** |
| `run_id` | `UUID` | **NOT NULL FK** → `runs.run_id` |
| `round_id` | `INTEGER` | nullable; paired **FK** → `rounds(run_id, round_id)` |
| `client_id` | `TEXT` | nullable **FK** → `clients.client_id` |
| `metric_name` | `TEXT` | **NOT NULL**; e.g. `global_f1`, `client_auc`, `training_ms`, `wait_ms`, `transfer_ms`, `message_bytes`, `retry_count` |
| `value` | `DOUBLE PRECISION` | **NOT NULL**; numeric value |
| `unit` | `TEXT` | **NOT NULL**; e.g. `ratio`, `ms`, `bytes`, `count` |
| `message_type` | `TEXT` | nullable dimension for communication measurements |
| `dimensions` | `JSONB` | **NOT NULL DEFAULT '{}'**; optional dimensions such as split name |
| `recorded_at` | `TIMESTAMPTZ` | **NOT NULL DEFAULT now()** |

Use indexes on `(run_id, metric_name, recorded_at)` and `(run_id, round_id, metric_name)`. Reports calculate min/max/mean from these rows for system measurements; preserve per-round ML measurements as individual rows.

## Relationship overview

- A run has many rounds; a round has many selected participants.
- A client may be selected for many rounds and submit at most one accepted update per round.
- A client may have many conformance attempts; only a pass for the run's exact manifest version permits selection.
- Events and metrics may refer to a run, round, and/or client, and are indexed for report reconstruction.
- Update and model binary files live outside PostgreSQL. A row is not considered committed until its referenced file exists and its SHA-256 has been verified.

## Sequence-diagram database read/write map

This inventory follows all database arrows in `diagrams/sequence/01_registration_conformance.puml` through `04_crash_resume.puml`. Reads/writes below are the intended database effects; file reads/writes are called out separately because binaries are not database rows.

### 01 — Registration and conformance

| Diagram action | Database operation |
|---|---|
| Admin approves a client | `INSERT clients(client_id, token_hash, status='APPROVED', ...)`; persist only a one-way token hash. |
| Gateway verifies registration | `SELECT clients` by `client_id`, compare supplied token against `token_hash`, check status and managed client version; update `last_seen_at` and registration fields on success. |
| Invalid token / unapproved client | `INSERT events` with registration rejection reason when the client identity is known; do not persist the submitted token. |
| Managed-code version is not approved | Read `clients.client_version`/approved release policy; write a rejection `events` row. The allow-list may be represented by the client's approved version in `clients` for this prototype. |
| Client asks for a manifest | No database write is needed: gateway reads the immutable, versioned manifest JSON and synthetic inputs from the checked-in manifest artifact. The requested version must match the active run's `runs.manifest_version`. |
| Gateway accepts/rejects conformance report | `INSERT conformance_reports` with client, manifest/client/protocol versions, pass/fail, stable reason code, and synthetic-test report payload; also `INSERT events`. |
| Coordinator eligibility check | `SELECT` latest passing `conformance_reports` for each registered client where `manifest_version = runs.manifest_version`. |

### 02 — Normal round

| Diagram action | Database operation |
|---|---|
| Coordinator opens a round | Read the `runs` row and eligible clients (`clients` + matching `conformance_reports`); `INSERT rounds` with state, input model version/path/checksum, deadline and quorum; `INSERT round_participants` for the seeded selection; write a `ROUND_OPENED` event. |
| Client polls for a task | Gateway reads active `rounds` and the client's `round_participants` row. Return no task if none is assigned. |
| Client downloads the model | Gateway reads `runs`/`rounds` model version, path, checksum metadata; file storage serves the bytes. Update `clients.last_seen_at`; record transfer event/size/duration as available. |
| Gateway validates upload | Read round state/deadline/model version, selected participant, and existing `updates` by `update_id` and `(run_id,round_id,client_id)`. |
| Same `update_id` retry | Read existing accepted `updates` row and return its prior acknowledgement; do not insert or count it again. Optionally record retry event. |
| Different ID duplicate or stale/closed round | Do not insert an accepted update; write a rejection `events` row with `duplicate_update`, `stale_round`, `deadline_passed`, or `model_version_mismatch`. |
| Valid update accepted | After checksum verification and atomic file write, transactionally `INSERT updates`, set participant status to `UPDATE_ACCEPTED`, and `INSERT UPDATE_ACCEPTED event` with bytes/duration. |
| Coordinator checks progress | Read `round_participants` and count accepted `updates`; inspect `rounds.deadline_at` and `min_participants`. |
| Coordinator aggregates | Read accepted `updates` metadata and weight files; write next global model file and checksum; transactionally set `rounds` output model fields/state to `DONE`, update `runs` current model version/path/checksum, and write metrics and event. If quorum is missed, set `FAILED` plus reason and event without advancing the run model pointer. |
| End-of-run reporting | Read `runs`, `rounds`, `round_participants`, `updates`, `conformance_reports`, `events`, and `metrics`; write final artifact path/checksum/format/size to `runs`. Report/plot files go to file storage. |

### 03 — Dropout and reconnection

| Diagram action | Database operation |
|---|---|
| Initial upload accepted, response lost, same ID retried | First request inserts one `updates` row and marks participant accepted. Retry reads that row by `update_id`; no second accepted row or count is created. |
| Deadline reached with no update from selected client | Set `round_participants.status='DROPOUT'`, `reason_code='deadline_no_update'`, and `finished_at`; write a dropout event. Read quorum count and transition round to `DONE` or `FAILED` with the reason recorded. |
| Late update for closed round | Read terminal round/deadline; reject and write a stale-update event. No `updates` row is inserted. |
| Client reconnects for next round | Update `clients.last_seen_at`; query its new `round_participants` assignment and current model metadata. On accepted upload, insert the next-round `updates` row as usual. |

### 04 — Coordinator crash and resume

| Diagram action | Database operation |
|---|---|
| Persist each state transition | Update `rounds.state`, `state_version`, timestamps and event in the same transaction. Store current model version/path/checksum on `runs` only after output artifact verification. |
| Round 5 aggregation and round 6 creation | Persist round 5 output model metadata and `DONE`; update `runs` current model pointer; insert round 6 with its input model version/path/checksum, deadline and state. Use separate short transactions with each completed artifact durably written first. |
| Gateway keeps accepting during coordinator outage | It continues using persisted `rounds` and `round_participants`; accepted `updates` and events are committed normally. |
| Coordinator restart | Query `runs.status='RUNNING'` and its latest non-terminal `rounds` row; read accepted updates/participants and deadline. Read the model path/checksum from the row and verify the file before continuing. |
| Resume before deadline | Continue the same round using existing participant and update rows; do not recreate the round or select a new client set. |
| Resume after deadline | Apply the same quorum rule; persist `DONE` or `FAILED`, reason, metrics and event. |
| Log coordinator resumed | `INSERT events(event_type='COORDINATOR_RESUMED', run_id, round_id, ...)`. |

## Report-field coverage

| Required report information (scope §3.1.F) | Source |
|---|---|
| Run ID, timestamps, configuration, strategy, seed, protocol/model/preprocessing/manifest versions | `runs` |
| Dataset allocation summary and sample counts per client | `round_participants`, accepted `updates.num_samples`, `runs.config`; never raw records |
| Participants per round, dropouts, failures, skipped/failed round reasons | `rounds`, `round_participants`, `events` |
| Global/per-client ML metrics over time | `metrics` (`global_*` / `client_*`) and `updates.metrics` |
| Communication latency, transfer sizes, retry/dropout counts | `events` and `metrics`, with `message_type`, duration, bytes, reason |
| Training, aggregation, waiting, total round durations; min/max/mean summaries | `metrics` at client/round scope; report aggregates values by metric and unit |
| Attempted/completed rounds and failure reasons | `rounds.state`, `rounds.failure_reason`, `events` |
| Final model location, format, version, checksum and size | `runs.final_model_*` fields |

## Indexes and retention notes

In addition to primary/unique constraints, create indexes for active-run lookup (`runs(status)`), latest round lookup (`rounds(run_id, round_id DESC)`), pending deadline scans (`rounds(state, deadline_at)`), client update history (`updates(client_id, received_at DESC)`), and the event/metric indexes listed above. Keep run artifacts and audit records for reproducibility; any retention/deletion policy must be decided with the advisors before real organization data is used.

## ER diagram

See [`../diagrams/database/er_diagram.puml`](../diagrams/database/er_diagram.puml) and its rendered PNG. `round_participants` is an additional association table required to persist the selected-client set and dropout status explicitly.
