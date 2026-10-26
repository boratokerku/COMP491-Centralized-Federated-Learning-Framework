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
- **Data Modalities:** The advisors confirmed this point only (the rest of this document is still pending their review): tabular banking data is mandatory. Exactly one additional modality is demonstrated in the same framework: **image (MedMNIST, primary candidate) or text (UCI SMS Spam Collection)**; the choice is finalized in WP7 (T7.1). This narrows the Capstone Project Description, which lists tabular, image and text.
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
|  * Simulation & Distributed Mode Parity                                           |
+-----------------------------------------------------------------------------------+
```

### 3.1. Mandatory Features (Core Deliverables)

#### A. Core Machine Learning & Aggregation
1. **Neural Network Framework:** Built on PyTorch: an MLP for tabular data and a small CNN for images (or an embedding-based model for text, if text is chosen as the second modality).
2. **Federated Optimization Algorithms:**
   - **FedAvg (Federated Averaging):** Sample-weighted averaging of local model weights.
   - **FedProx:** Addition of a proximal term (with a coefficient μ) to the client local objective, keeping local models close to the global model to address system and data heterogeneity.
3. **Dual Execution Modes:**
   - **Simulation Mode:** In-memory transport executing multiple virtual clients on a single machine for debugging, benchmarking and continuous integration.
   - **Real Distributed Mode:** Execution across separate machines (university servers, cloud VMs, team members' machines) over the Internet, using the same training and orchestration abstractions as simulation mode.
4. **Data Distribution:** A fully randomized (IID) partition of each dataset across clients, generated with a fixed seed.

#### B. Coordination & Orchestration Layer
1. **Round State Machine:** Explicit transitions through `OPEN` → `COLLECTING` → `AGGREGATING` → `DONE` (or `FAILED` from any non-terminal state) with deterministic transition guards.
2. **Round Lifecycle:** Client registration, round initiation, model/configuration distribution, local training, update collection, aggregation and evaluation, with no manual step between rounds.
3. **Partial Participation:** A configurable subset of registered clients is selected for each round.
4. **Participation Quorum & Timeouts:** Configurable minimum number of participating clients and per-round deadline; at the deadline the round is aggregated with the clients that arrived or marked `FAILED` with a recorded reason.
5. **Straggler & Dropout Handling:** Late updates are discarded after the deadline; clients that drop out before or during a round do not stall the run.
6. **Retries & Reconnection:** Clients retry temporary connection failures with exponential backoff; a dropped client can rejoin in a later round.
7. **Idempotency & Deduplication:** Duplicate and stale updates are rejected using a client-generated UUIDv4 `update_id` and a `UNIQUE(run_id, round_id, client_id)` constraint, plus round and model-version checks.
8. **Coordinator Crash Recovery:** Run, round and model state are persisted to PostgreSQL after every transition, so a restarted coordinator resumes from the last completed round instead of round zero.

#### C. Client Onboarding & Conformance Contract
1. **Managed-Code Mode:** Versioned Docker images (`cfl-client`) or release bundles that do not require access to a public Git repository; the client's version (image tag/digest) is verified at registration.
2. **Independent-Implementation Mode:** A documented, language-agnostic REST interface; such clients must submit a **Conformance Manifest** and pass its checks before joining a run.
3. **Pre-Training Conformance Validation:**
   - Model architecture/version and parameter tensor shapes.
   - Input/output shapes, feature names, exact column ordering and data types.
   - Pre-processing and categorical encoding versions.
   - Feature semantics and units (e.g., days vs. months).
   - Label semantics and expected value ranges.
   - Reference test cases: the client's pre-processing output on synthetic inputs must match the expected reference outputs. No raw client data is shared.
4. **Deterministic Disqualification:** A client failing any check is rejected with a clear reason, and the result is recorded in PostgreSQL.

#### D. Communication & Security
1. **Transport Layer:** HTTPS over TLS, terminated at a Caddy reverse proxy; plain HTTP is refused.
2. **Client-Initiated Architecture:** Clients always initiate requests (`POST /register`, `GET /task`, `GET /model`, `POST /update`), avoiding firewall/NAT traversal issues.
3. **Client Registration & Authentication:** Only clients on an administrator-approved list can register; each client receives a per-client token (stored hashed) that is required on every request. Unregistered clients are rejected.
4. **Integrity Verification:** SHA-256 checksums on all distributed and uploaded model weight artifacts (`.safetensors`). Weights are never exchanged with pickle.

#### E. Evaluation & Metrics
1. **Machine-Learning Metrics:** Accuracy, F1 and AUC of the global model across rounds; per-client performance across rounds; each client's local performance before and after local training.
2. **Comparisons:** FedAvg vs. FedProx; federated vs. centralized and local-only baselines.
3. **Communication Metrics:** Round-trip time between coordinator and clients; transfer time and size per message type (register, model download, update upload); total bytes transmitted; retry, dropout and failed-round counts.
4. **Computation & Timing Metrics:** Per-round client training duration, server aggregation/evaluation duration, end-to-end round duration, and waiting vs. computing time.
5. All metrics are reported with minimum, maximum and mean summaries.

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

The web application is built with **Next.js (React)**. It has a public project site and an authenticated internal area. The internal area reads data through read-only gateway API endpoints, not directly from PostgreSQL. Grafana reads round metrics directly from PostgreSQL.

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
| `/results` | **Results** | FedAvg vs. FedProx and federated vs. centralized/local-only results for both modalities; poster and demo video when available. |
| `/team` | **Team & About** | Team members, advisors, Koç University, link to the GitHub repository. |

### 4.2. Authenticated Area (`/app`) – Mandatory

| Route | View Name | Content |
| :--- | :--- | :--- |
| `/app/rounds` | **Round Viewer** | Current run and round state, participating clients, received vs. minimum required updates, deadline, stragglers. |
| `/app/clients` | **Clients & Conformance** | Registered clients, last seen / online status, client version, conformance status (passed, or failed with reason and manifest version). |
| `/app/updates` | **Update Inbox** | Updates in the current round: client ID, timestamp, sample count, SHA-256 checksum, accepted/rejected with reason. |
| `/app/logs` | **Event Log** | Filterable event list from PostgreSQL (registrations, timeouts, rejections, aggregations, coordinator restarts). |
| `/app/reports` | **Reports & Plots** | List of completed runs, generated plots, Markdown/CSV/JSON report downloads. |

### 4.3. Optional Web Extras

- Control actions (Start Run, Abort): require write endpoints with separate admin authorization.
- Interactive charts (Recharts/Chart.js) in place of the generated static plots.
- Embedded Grafana panels.

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
## 6. Architecture, Tech Stack & Workflow Decisions

Implementation decisions behind the requirements above. Open decisions are closed by the team during WP1 (weeks 1–2).  
**How to use:** For each open item, discuss the options, write the chosen option in **Decision**, set **Status** to `Decided` with the date, and add a one-line reason. If a decision changes later, do not delete it: mark it `Superseded` and add the new one.

`Recommendation` is a starting point for discussion, not a decision.

---

### 6.0 Already decided

| ID | Topic | Decision | Source |
| :--- | :--- | :--- | :--- |
| D0.1 | ML framework | PyTorch | Proposal |
| D0.2 | Algorithms | FedAvg and FedProx | Capstone description, proposal |
| D0.3 | Gateway framework | FastAPI | Proposal |
| D0.4 | Database | PostgreSQL | Proposal |
| D0.5 | Web application | Next.js | Proposal |
| D0.6 | Metrics dashboards | Grafana reading from PostgreSQL | Proposal |
| D0.7 | Data modalities | Tabular (mandatory) + one of image or text | Advisors, Oct 2026 |
| D0.8 | Grafana in web app | Optional: Grafana panels embedded as iframes; the web app shows what Grafana cannot present well | Team, 26 Oct 2026 |
| D0.9 | Transport | Client-initiated HTTPS; clients always open the connection | Proposal |
| D0.10 | Task tracking | Every task, coding and non-coding (research, meetings, docs, data, slides), is a GitHub Issue with title, short description, acceptance criteria / checklist and assignee(s); labels and milestone optional | Course (GitHub Issues guide) |
| D0.11 | Development flow | Issue → branch for that issue → commits → PR → review → merge → issue closes | Course |
| D0.12 | Commit messages | Conventional Commits: `type(scope): description` (feat, fix, docs, refactor, test, chore) | Course |
| D0.13 | PR ↔ issue link | Every PR references its issue with a closing keyword (`Closes #ID`) | Course |
| D0.14 | Evaluation of contributions | Instructors review issues, PRs, commits and review comments, so each member's work must be visible in GitHub history | Course |

---

### 6.1 Architecture — decide this week

These block the API spec, DB schema and diagrams (D1.2, D1.3).

| ID | Decision needed | Options | Recommendation | Decision | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A1 | Where does the coordinator + gateway run, and how do clients reach it over the Internet? | (a) University server with a public address (b) University server via VPN (c) Cloud VM with a public IP (d) A team member's machine with port forwarding | Ask IT about (a) this week; keep (c) as fallback. Needed for the mid-demo (M4.4). | | Open |
| A2 | How do the gateway and coordinator interact? | (a) Separate processes sharing PostgreSQL + file storage (b) One process (coordinator runs inside the gateway) (c) Separate processes with direct calls | (a): PostgreSQL is the single source of truth, so a restarted coordinator resumes by reading it. | Separate processes sharing PostgreSQL + file storage | Decided |
| A3 | Where is coordinator state kept in simulation mode? | (a) PostgreSQL as in distributed mode (b) SQLite (c) In-memory store behind the same interface | (b) or (c) behind a storage interface, so tests and CI do not need PostgreSQL. | In-memory store behind a storage interface | Decided |
| A4 | Where is the global model evaluated? | (a) Server-side held-out test set (b) Federated evaluation: clients evaluate the global model on their local test split and report metrics (c) Both | (c): (a) for experiments and baselines, (b) for per-client metrics required by the description. | Both: server-side test set + federated evaluation | Decided |
| A5 | API style | (a) REST + JSON metadata, binary weights (b) gRPC | (a): simpler to debug; model sizes are small. | REST | Decided |
| A6 | How do clients get tasks? | (a) Short polling (b) Long polling (c) WebSocket/SSE | (a), with a configurable interval; (b) only if latency becomes a problem. | Short polling, configurable interval | Decided |
| A7 | Weight serialization | (a) safetensors (b) `torch.save` (pickle) (c) NumPy `.npz` | (a): pickle from untrusted clients allows code execution. | safetensors | Decided |
| A8 | Integrity check | (a) SHA-256 checksum (b) None | (a) | SHA-256 | Decided |
| A9 | Weight storage | (a) Local file system, path in DB (b) Object store (MinIO/S3) (c) Inside PostgreSQL | (a) | Local file system, path in DB | Decided |
| A10 | Duplicate / stale update rule | `update_id` format (UUIDv4?), `UNIQUE(run_id, round_id, client_id)`, reject updates for a closed round or old model version | Both `update_id` and the UNIQUE constraint; retries reuse the same `update_id`. | UUIDv4 `update_id` + `UNIQUE(run_id, round_id, client_id)` | Decided |
| A11 | Round deadline rule | At the deadline: aggregate if received ≥ minimum, else `FAILED` | As written; late updates after the deadline are rejected as stale. | Aggregate if received ≥ minimum, else FAILED; late updates rejected | Decided |
| A12 | Client selection per round | (a) Random subset with seed (b) All available clients (c) Round-robin | (a), configurable fraction; (b) as a special case. | Seeded random subset (configurable fraction) | Decided |
| A13 | Retry policy | Max attempts, backoff (fixed / exponential), jitter | Exponential backoff with jitter, max attempts in config. | Exponential backoff + jitter, max attempts in config | Decided |
| A14 | Latency measurement | (a) RTT and durations measured on one side (b) One-way latency (needs clock sync) | (a): machine clocks are not synchronized. | RTT and one-sided durations | Decided |

### 6.2 Security

| ID | Decision needed | Options | Recommendation | Decision | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S1 | TLS termination | (a) Caddy reverse proxy (b) nginx (c) Uvicorn with certificates directly | (a): automatic certificates, short config. | Caddy | Decided |
| S2 | Certificates | (a) Let's Encrypt on a hostname (free: `sslip.io`, a `ku.edu.tr` subdomain, or a Student Pack `.tech`/`.me` domain); nothing to distribute, needs a public IP (b) Own CA via Caddy `tls internal`; works without a public IP, clients get our root certificate | Depends on A1: public IP → (a), otherwise (b). | To be determined (after A1) | Open |
| S3 | Client authentication | (a) Per-client token issued at admin approval (b) Mutual TLS (c) Shared API key | (a); tokens stored hashed. (b) only if time permits. | Per-client token, stored hashed | Decided |
| S4 | Client approval flow | How an admin adds a client to the approved list (CLI, admin endpoint, DB seed) | Admin CLI command. | Admin CLI command | Decided |

### 6.3 Data & database

| ID | Decision needed | Options | Recommendation | Decision | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| B1 | Tabular dataset | (a) Banking dataset from the advisors (b) UCI Bank Marketing (c) UCI Default of Credit Card Clients | (a) if it arrives with a clear license; start development on (b). | | Open |
| B2 | Second modality | (a) Image: MedMNIST (b) Text: UCI SMS Spam | (a): easy download, trains on CPU, no shared-vocabulary problem. Deadline: WP7 (T7.1). | Image (MedMNIST) | Decided |
| B3 | Image dataset (if B2 = image) | DermaMNIST, BloodMNIST, PathMNIST (avoid ChestMNIST: multi-label) | BloodMNIST or DermaMNIST, 28×28. | BloodMNIST | Decided |
| B4 | DB access layer | (a) SQLAlchemy 2.0 + Alembic migrations (b) psycopg with raw SQL | (a): schema in one place, migrations for a 4-person team. | SQLAlchemy 2.0 + Alembic | Decided |
| B5 | PostgreSQL version | 15 / 16 / 17 | 16 | 16 | Decided |
| B6 | Run configuration | (a) YAML + Pydantic validation (b) Hydra (c) JSON | (a) | YAML + Pydantic | Decided |

### 6.4 Client compatibility

| ID | Decision needed | Options | Recommendation | Decision | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| C1 | Managed-code distribution | (a) Docker image in a private registry (GHCR) (b) Docker image as `docker save` tarball (c) Python wheel / release bundle | (a), with (b) for clients without registry access. Version = image tag + digest. | Docker image in private GHCR + `docker save` tarball | Decided |
| C2 | Manifest format | (a) JSON (b) YAML | (a), validated with a Pydantic model / JSON Schema. | JSON | Decided |
| C3 | Where conformance checks run | (a) Client runs the checker locally and submits a report (b) Server sends reference inputs, client returns outputs, server compares (c) Both | (b) for reference test cases (the client cannot fake the comparison), (a) for metadata checks. | Metadata: client report; reference tests: server-side comparison | Decided |
| C4 | Independent-client demonstration | How we show a non-managed client | A separately written minimal client (not using our package); one conforming and several mismatching variants (T5.5). | Separately written minimal client, conforming + mismatching variants | Decided |

### 6.5 Web application & observability

| ID | Decision needed | Options | Recommendation | Decision | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | Internal area login | (a) Single admin account (b) NextAuth with team accounts (c) Gateway-issued token | (a) or (c); the users are only the team. | Single admin account | Decided |
| W2 | Live updates | (a) Polling every few seconds (b) Server-Sent Events (c) WebSocket | (a) | Polling | Decided |
| W3 | Hosting | (a) Whole Next.js app next to the gateway (b) Public site on Vercel, internal area next to the gateway | (a) is simpler; (b) if the public site must stay up independently. | Whole Next.js app next to the gateway | Decided |
| W4 | Public results page data | (a) Updated manually from reports (b) Pulled automatically from reports | (a) | Updated manually | Decided |
| W5 | Grafana embedding access (if done) | (a) Auth proxy behind the same reverse proxy (b) Anonymous read-only, internal network only | (a) if exposed to the Internet; never anonymous on the Internet. | Auth proxy | Decided |
| W6 | Dashboards as code | (a) Grafana provisioning, JSON in `deploy/grafana/` (b) Created manually in the UI | (a) | Grafana provisioning, JSON in repo | Decided |
| W7 | Event/metric logging | (a) Events and metrics tables in PostgreSQL (b) Additional Prometheus | (a) | PostgreSQL tables | Decided |
