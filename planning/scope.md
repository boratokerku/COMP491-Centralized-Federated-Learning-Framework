# T1.1: Requirements and Scope Decision Document

**Project:** COMP 491 Capstone – Centralized Federated Learning Framework  
**Document Version:** Draft  
**Date:** 26 October 2026  
**Status:** Draft – pending advisor review  
**Related Work Packages:** WP1 (Design, Plan, and Data Preparation)  
**Deliverable Association:** D1.4 (Scope Decision and Work Plan)

---

## 1. Executive Summary & Context

Cross-silo Federated Learning (FL) enables distributed organizations (such as banks, medical institutions, or telecom providers) to collaboratively train machine learning models without transferring their private raw data to a central location.

While academic research often abstracts away real-world system complexities through single-machine simulations, real-world cross-silo FL deployments face major distributed systems challenges:
1. **Client Incompatibility:** Independently operated clients may have differing feature ordering, measurement units, categorical encodings, or label semantics.
2. **Network & Client Instability:** Heterogeneous network speeds, slow clients (stragglers), temporary dropouts, duplicate or stale update submissions, and unannounced disconnections.
3. **Coordinator Failures:** Server-side crashes requiring reliable resumption from persisted state rather than restarting from round zero.
4. **Observability & Traceability:** Requirements for comprehensive operational logging, latency tracking, and reproducible end-of-run reporting.

This document establishes the **requirements, feature boundaries (mandatory vs. optional), web application structure, and incremental build order** for the framework. Work package (WP), deliverable (D) and milestone (M) numbers refer to the project proposal.

---

## 2. Review of Capstone Objectives

Based on the official Capstone Project Description and the project proposal (advisors: Prof. Mehmet Emre Gürsoy, Prof. Barış Akgün; this document has not yet been reviewed by them):
- **Core Focus:** A **distributed systems, robustness, compatibility, and observability project**, not a study of new federated learning algorithms.
- **Cross-Silo Setting:** A small number (e.g., 3–10) of independently operated client organizations participating in centrally coordinated training over real networks (Internet).
- **Data Modalities:** The advisors confirmed this point only (the rest of this document is still pending their review): tabular banking data is mandatory. Exactly one additional modality is demonstrated in the same framework: **image or text**, as in the proposal (open decision, Section 6, B2). This narrows the Capstone Project Description, which lists tabular, image and text.
- **Fail-Safe Client Onboarding:** Incompatible clients must be detected and rejected before they can corrupt the global model.

---

## 3. Scope Specification: Mandatory vs. Optional Features

```
+-----------------------------------------------------------------------------------+
|                           CFL FRAMEWORK FEATURE SCOPE                             |
+-----------------------------------------------------------------------------------+
|  MANDATORY CORE                                  OPTIONAL EXTENSIONS (If Time)    |
|  * PyTorch Neural Network Pipeline               * Non-IID Partitioning           |
|  * FedAvg & FedProx (Proximal mu Loss)             (label / feature skew)         |
|  * Round State Machine (5 states)                * Update Clipping + Gaussian     |
|  * PostgreSQL 16 State Persistence & Resume        Noise (no formal DP guarantee) |
|  * Client Registration & Authentication          * Interactive web extras         |
|  * Pre-Round Conformance Manifest Check            (see Section 4.3)              |
|  * Managed-Code (Docker) & Independent Clients                                    |
|  * Real Internet Multi-Machine Deployment                                         |
|  * Automated JSON/CSV/Markdown Reports & Plots                                     |
|  * Final Model Delivery (versioned, checksummed)                                  |
|  * Simulation & Distributed Modes                                                |
+-----------------------------------------------------------------------------------+
```

### 3.1. Mandatory Features (Core Deliverables)

#### A. Core Machine Learning & Aggregation
1. **Neural Network Framework:** Built on PyTorch: an MLP for tabular data, and a model suited to the second modality (e.g., a small CNN for images or an embedding-based model for text).
2. **Federated Optimization Algorithms:**
   - **FedAvg (Federated Averaging):** Sample-weighted averaging of local model weights.
   - **FedProx:** Addition of a proximal term (with a coefficient μ) to the client local objective, keeping local models close to the global model to address system and data heterogeneity.
3. **Dual Execution Modes:**
   - **Simulation Mode:** Multiple clients run on a single machine for debugging and benchmarking. Clients communicate through an in-memory transport, and coordinator state is kept in an in-memory store behind the same storage interface used in distributed mode.
   - **Real Distributed Mode:** The coordinator and clients run on genuinely separate machines (university servers and team members' machines) and communicate over the Internet.
   - The two modes share the same training and orchestration abstractions where practical.
4. **Data Distribution:** A fully randomized (IID) partition of each dataset across clients, generated with a fixed seed.
5. **Configuration & Reproducibility:** Every run is defined by a YAML configuration file validated with Pydantic, including all random seeds, so any reported result can be rerun.

#### B. Coordination & Orchestration Layer
1. **Process Architecture:** The gateway and the coordinator are separate processes that share PostgreSQL and file storage; PostgreSQL is the single source of truth for run and round state.
2. **Round State Machine:** Explicit transitions through `OPEN` → `COLLECTING` → `AGGREGATING` → `DONE` (or `FAILED` from any non-terminal state) with deterministic transition guards.
3. **Round Lifecycle:** Client registration, round initiation, model/configuration distribution, local training, update collection, aggregation and evaluation, with no manual step between rounds.
4. **Partial Participation:** Each round selects a random subset of the available registered clients (configurable fraction, fixed seed).
5. **Participation Quorum & Timeouts:** Configurable minimum number of participating clients and per-round deadline. At the deadline the round is aggregated if the number of received updates meets the minimum; otherwise it is marked `FAILED` with a recorded reason.
6. **Straggler & Dropout Handling:** Updates arriving after the deadline are rejected as stale; clients that drop out before or during a round do not stall the run.
7. **Retries & Reconnection:** Clients retry temporary connection failures with exponential backoff and jitter, up to a configurable number of attempts; a retried upload reuses the same `update_id`. A dropped client can rejoin in a later round.
8. **Idempotency & Deduplication:** Duplicate and stale updates are rejected using a client-generated UUIDv4 `update_id` and a `UNIQUE(run_id, round_id, client_id)` constraint, plus round and model-version checks.
9. **Coordinator Crash Recovery:** Run, round and model state are persisted to PostgreSQL 16 (accessed through SQLAlchemy 2.0, schema managed with Alembic migrations) after every transition, so a restarted coordinator resumes from the last persisted state instead of round zero. Because the gateway is a separate process, updates keep arriving during a coordinator outage; an interrupted round continues if its deadline has not passed, otherwise the normal deadline rule is applied to it.

#### C. Client Onboarding & Conformance Contract
1. **Managed-Code Mode:** The approved client code is distributed as a versioned Docker image (`cfl-client`) through a private container registry (GitHub Container Registry), or as an image archive (`docker save`) for clients without registry access; no client needs access to a public Git repository. The client's version (image tag and digest) is verified at registration.
2. **Independent-Implementation Mode:** An organization may run its own implementation against the documented, language-agnostic REST interface. The implementation (pre-processing, training code, language, libraries) is free, but the model architecture and parameter names/shapes must match the run's model, since aggregation averages parameters. Such clients must submit a **Conformance Manifest** (JSON, validated against a schema) and pass its checks before joining a run.
3. **Pre-Training Conformance Validation:**
   - Model architecture/version and parameter tensor shapes.
   - Input/output shapes, feature names, exact column ordering and data types.
   - Pre-processing and categorical encoding versions.
   - Feature semantics and units (e.g., days vs. months).
   - Label semantics and expected value ranges.
   - Reference test cases: the client's pre-processing output on synthetic reference inputs must match the expected outputs. No raw client data is shared.

   The client runs the conformance checker locally and submits a pass/fail report that includes its outputs on the reference inputs; before accepting the client, the gateway re-checks those outputs against the expected outputs and the manifest metadata against the run contract. The coordinator only selects clients with a passing report for the current manifest version.
4. **Deterministic Disqualification:** A client failing any check is rejected with a clear reason, and the result is recorded in PostgreSQL.
5. **Demonstration:** The independent mode is demonstrated with a separately written minimal client that does not use our client package: one conforming variant and several mismatching variants (wrong feature order, unit, label range, pre-processing version, model shape).

#### D. Communication & Security
1. **Transport Layer:** A REST API implemented with FastAPI, served over HTTPS. TLS is terminated at a Caddy reverse proxy in front of the gateway; plain HTTP is refused. The certificate source is an open decision (Section 6, S2).
2. **Client-Initiated Architecture:** Clients always initiate requests (`POST /register`, `GET /task`, `GET /model`, `POST /update`), avoiding firewall/NAT traversal issues. Clients poll `GET /task` at a configurable interval.
3. **Client Registration & Authentication:** An administrator adds clients to the approved list with a CLI command. Each approved client receives a per-client token (stored hashed) that is required on every request. Unregistered clients are rejected.
4. **Weights & Integrity:** Model weights are exchanged in the `safetensors` format (never pickle) with SHA-256 checksums on every distributed and uploaded artifact. Weight files are stored on the server's file system; PostgreSQL stores their paths and metadata.

#### E. Evaluation & Metrics
1. **Machine-Learning Metrics:** Accuracy, F1 and AUC of the global model across rounds; per-client performance across rounds; each client's local performance before and after local training. The global model is evaluated both on a server-side held-out test set and by each client on its local test split (federated evaluation).
2. **Comparisons:** FedAvg vs. FedProx; federated vs. centralized and local-only baselines.
3. **Communication Metrics:** Round-trip time between coordinator and clients (measured as round trips and one-sided durations, since machine clocks are not synchronized); transfer time and size per message type (register, model download, update upload); total bytes transmitted; retry, dropout and failed-round counts.
4. **Computation & Timing Metrics:** Per-round client training duration, server aggregation/evaluation duration, end-to-end round duration, and waiting vs. computing time.
5. All metrics are reported with minimum, maximum and mean summaries.
6. All events and metrics are recorded in PostgreSQL tables; no separate metrics system is used.

#### F. End-of-Training Report & Experiment Log
Every completed run automatically produces:
1. A machine-readable log (`run_{id}_summary.json` and CSV exports).
2. A human-readable Markdown report (`run_{id}_report.md`) containing:
   - Run ID, timestamps, configuration, and software/protocol/model/pre-processing/manifest versions.
   - Dataset allocation summary and sample counts per client.
   - Participating clients per round, dropout and failure events.
   - Global and per-client ML metrics over time.
   - Communication and timing metrics with min/max/mean summaries.
   - Number of rounds attempted and completed, with the reason for every failed or skipped round.
   - Final global model location and checksum.
3. Plots of the principal ML and system metrics (global metrics vs. round, per-client performance, communication overhead, round duration breakdown).

The report alone must be sufficient to reconstruct what happened in a run without inspecting console output.

#### G. Final Model Delivery
1. The final global model is packaged as a versioned artifact with checksum, manifest version and run ID.
2. ONNX export, verified against the PyTorch model on the reference test cases.
3. A gateway endpoint lets registered clients download the model and verify its checksum; unregistered clients are refused.

#### H. Testing
Automated tests for aggregation, FedProx, partial participation and reproducibility, plus a failure-case suite: slow client, dropout before and during a round, temporary connection failure with successful retry, duplicate update, stale update, reconnection in a later round, coordinator restart, and conformance mismatches.

---

### 3.2. Optional Features (If Time Permits)

1. **Non-IID Partitioning Experiments:**
   - Label skew (e.g., highly imbalanced clients or clients missing classes).
   - Feature-distribution skew between client splits.
   - Exact construction to be decided with advisor input.
2. **Privacy Extension:** Client-side clipping and noise addition on the model update, measuring the effect on model quality. The exact mechanism is decided only if this extension is taken up. No formal differential-privacy guarantee or privacy accounting is claimed.
3. **Web Application Extras:** see Section 4.3.

---

## 4. Web Application Specification & Page Architecture

The web application is built with **Next.js (React)**. It has a public project site and an authenticated internal area. The whole Next.js application is hosted next to the gateway, behind the same reverse proxy. The internal area reads data through read-only gateway API endpoints, not directly from PostgreSQL, and refreshes by polling.

Grafana dashboards read metrics directly from PostgreSQL and are the main tool for time-series metric charts; dashboards are kept as code in the repository (Grafana provisioning). The web application displays the information Grafana cannot present well: run and round state, client and conformance details, update records, events and reports.

```
                           +------------------------+
                           |  Next.js Frontend App  |
                           +-----------+------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
         [ Public Site ]                         [ Authenticated Area ]
         * / (Overview)                          * /app/rounds (Round Viewer)
         * /docs (API & Conformance)             * /app/clients (Clients & Conformance)
         * /results (Results)                    * /app/updates (Update Inbox)
         * /team (About)                         * /app/logs (Event Log)
                                                 * /app/reports (Reports & Plots)
```

### 4.1. Public Area (Unauthenticated) – Mandatory

Online from week 4 (M6.1) and updated weekly until the end of the semester.

| Route | Page Name | Content |
| :--- | :--- | :--- |
| `/` | **Project Home** | Project summary, cross-silo FL explanation, weekly progress updates. |
| `/docs` | **Documentation** | Setup and run instructions, REST API reference, Conformance Manifest specification, client integration guide. |
| `/results` | **Results** | FedAvg vs. FedProx and federated vs. centralized/local-only results for both modalities (updated manually from the run reports); poster and demo video when available. |
| `/team` | **Team & About** | Team members, advisors, Koç University, link to the GitHub repository. |

### 4.2. Authenticated Area (`/app`) – Mandatory

Access is protected by a single admin login; the users are the team.

| Route | View Name | Content |
| :--- | :--- | :--- |
| `/app/rounds` | **Round Viewer** | Current run and round state, participating clients, received vs. minimum required updates, deadline, stragglers. |
| `/app/clients` | **Clients & Conformance** | Registered clients, last seen / online status, client version, conformance status (passed, or failed with reason and manifest version). |
| `/app/updates` | **Update Inbox** | Updates in the current round: client ID, timestamp, sample count, SHA-256 checksum, accepted/rejected with reason. |
| `/app/logs` | **Event Log** | Filterable event list from PostgreSQL (registrations, timeouts, rejections, aggregations, coordinator restarts). |
| `/app/reports` | **Reports & Plots** | List of completed runs, generated plots, Markdown/CSV/JSON report downloads. |

### 4.3. Optional Web Extras

- Control actions (Start Run, Abort): require write endpoints with separate admin authorization.
- Grafana panels embedded in the web application as iframes, with Grafana behind the same reverse proxy using its auth-proxy mode (never anonymous access on the Internet).

---

## 5. Build Order & Implementation Roadmap

Design-first, incremental layers, aligned with the proposal's work packages and Gantt chart (12 weeks, final deadline 15 January 2027). Each layer is tested before dependent layers build on it.

| Weeks | WP | Layer | Key outputs | Milestones |
| :--- | :--- | :--- | :--- | :--- |
| 1–2 | WP1 | Design, specification & data | Architecture, state machine, sequence and deployment diagrams; scope (this document); API spec, DB schema, conformance manifest spec; banking dataset (UCI Bank Marketing as fallback) cleaned with IID partition; web site plan; repo and task tracking | M1.1–M1.3 |
| 3–5 | WP2 | FL core in simulation | Transport interface + in-memory implementation; tabular MLP; FedAvg, FedProx, partial participation; automated tests; centralized/local-only/federated baselines; model tuning | M2.1–M2.3 |
| 3–4 | WP3 | Secure API gateway | FastAPI gateway behind Caddy/TLS; PostgreSQL tables + Alembic migrations; approved-list registration and per-client tokens; update intake with checksum; HTTPS transport | M3.1–M3.3 |
| 4–6 | WP4 | Orchestration & fault tolerance | Round state machine persisted in PostgreSQL; quorum, timeouts, retries; stale/duplicate handling; dropout, reconnection, coordinator resume; failure-case tests; first multi-machine Internet run | M4.1–M4.4 (mid-demo) |
| 5–7 | WP5 | Client compatibility & conformance | Manifest format; client-side checker with reference test cases; gateway/coordinator enforcement; managed-code Docker client with version check; mismatch tests | M5.1–M5.3 |
| 4–12 | WP6 | Observability & web application | Public site online (week 4) and maintained weekly; event and metric logging; authenticated internal area; end-of-training report and plots (week 6); Grafana on PostgreSQL | M6.1–M6.4 |
| 5–10 | WP7 | Experiments & evaluation | Second-modality pipeline and manifest; experiment design; runs on separate machines incl. failure scenarios; analysis tables/plots; reproducibility package | M7.1–M7.3 |
| 6–10 | WP8 | Final model delivery (+ optional privacy) | Versioned, checksummed artifact; ONNX export verified against PyTorch; download endpoint and client-side verification; (optional) clipping + noise | M8.1, (M8.2) |
| 8–12 | WP9 | Poster, demo video & final report | Final report, poster, demo video, clean repository with run instructions | M9.1–M9.3 |

---

## 6. Open Decisions

Decisions that are still open. When one is closed, write it into the relevant section above and remove it from this table.

| ID | Decision needed | Options | Recommendation | Depends on |
| :--- | :--- | :--- | :--- | :--- |
| A1 | Where the coordinator, gateway and web application run, and how clients reach them over the Internet | (a) University server with a public address (b) University server reachable only via VPN (c) Cloud VM with a public IP (d) A team member's machine with port forwarding | Ask university IT for (a). Needed for the public site (week 4) and the mid-demo (M4.4). | University IT |
| A15 | Use of university cluster nodes as client machines | (a) Cluster nodes run some of the clients (b) Only team members' machines | (a) if the advisors confirm access. Check first: outbound HTTPS (port 443) from compute nodes, whether Docker is allowed (clusters often only allow Apptainer/Singularity, which can run our Docker image), and job time limits. A first trial runs on team members' machines. | Advisors |
| S2 | TLS certificate source | (a) Let's Encrypt on a hostname (free: `sslip.io`, a `ku.edu.tr` subdomain, or a GitHub Student Pack `.tech`/`.me` domain); nothing to distribute; needs a public IP (b) Own CA via Caddy `tls internal`; works without a public IP; clients receive our root certificate | Public IP → (a), otherwise (b). | A1 |
| B1 | Tabular dataset | (a) Banking dataset from the advisors (b) UCI Bank Marketing (c) UCI Default of Credit Card Clients | (a) if it arrives with a clear license; development starts on (b). | Advisors |
| B2 | Second modality and dataset | (a) Image: MedMNIST (e.g., BloodMNIST or DermaMNIST, 28×28; avoid multi-label ChestMNIST) (b) Text: UCI SMS Spam Collection | (a): easy download, trains on CPU, no shared-vocabulary problem across clients. Decide before WP7 (T7.1). | Team |
