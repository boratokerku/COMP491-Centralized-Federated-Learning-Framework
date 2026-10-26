# T1.1: Requirements and Scope Decision Document

**Project:** COMP 491 Capstone – Centralized Federated Learning Framework  
**Document Version:** 1.0  
**Date:** Fall 2026  
**Status:** Approved  
**Related Work Packages:** WP1 (Design, Plan, and Data Preparation)  
**Deliverable Association:** D1.4 (Scope Decision and Work Plan)

---

## 1. Executive Summary & Context

Cross-silo Federated Learning (FL) enables distributed organizations (such as banks, medical institutions, or telecom providers) to collaboratively train machine learning models without transferring their private raw data to a central location. 

While academic research often abstracts away real-world system complexities through synchronous single-machine simulations, real-world cross-silo FL deployments face major distributed systems challenges:
1. **Client Incompatibility:** Independently operated clients may have differing feature ordering, measurement units, categorical encodings, or label semantics.
2. **Network & Client Instability:** Heterogeneous network speeds, slow clients (stragglers), temporary dropouts, duplicate update submissions, and unannounced disconnections.
3. **Coordinator Failures:** Server-side crashes requiring reliable resumption from persisted state rather than restarting from round zero.
4. **Observability & Traceability:** Requirements for comprehensive, audit-ready operational logging, latency tracking, and reproducible end-of-run reporting.

This document formally establishes the **requirements, feature boundaries (mandatory vs. optional), web application structure, and incremental build order** for the Centralized Federated Learning Framework.

---

## 2. Review of Capstone Objectives & Advisor Directives

Based on the official Capstone Project Description and advisory input from Prof. Mehmet Emre Gürsoy and Prof. Barış Akgün:
- **Core Focus:** The project is explicitly prioritized as a **distributed systems, robustness, compatibility, and observability project**, rather than a novel machine learning algorithm design study.
- **Cross-Silo Setting:** A small number (e.g., 3–10) of independently operated client organizations participating in centrally coordinated training across real networks (Internet/WAN).
- **Multi-Modal Verification:** While banking tabular data is the primary operational benchmark, the framework architecture must demonstrate multi-modality by supporting image and text data.
- **Fail-Safe Client Onboarding:** Incompatible clients must be intercepted and rejected before they can corrupt the global model.

---

## 3. Scope Specification: Mandatory vs. Optional Features

```
+-----------------------------------------------------------------------------------+
|                           CFL FRAMEWORK FEATURE SCOPE                             |
+-----------------------------------------------------------------------------------+
|  MANDATORY CORE (Baseline)                       OPTIONAL EXTENSIONS (If Time)    |
|  * PyTorch Neural Network Pipeline               * Non-IID Dirichlet Partitioning |
|  * FedAvg & FedProx (Proximal mu Loss)           * Client-Side DP-SGD / Noise     |
|  * 5-Phase Round Lifecycle Engine                * Byzantine-Robust Aggregation   |
|  * PostgreSQL 16 State Persistence               * Dynamic Client Clustering      |
|  * Pre-Round Conformance Manifest Check          * Secure Aggregation (SecAgg)    |
|  * Managed-Code (Docker) & Independent Clients   * Homomorphic Encryption         |
|  * Real Internet WAN Multi-Machine Deployment    * Web Push Notification Alerts   |
|  * Automated Markdown/CSV/Plot Reports           * Split Learning                 |
|  * In-Memory Simulation & WAN Mode Parity        * Model Quantization (INT8)      |
+-----------------------------------------------------------------------------------+
```

### 3.1. Mandatory Features (Core Deliverables)

#### A. Core Machine Learning & Aggregation
1. **Neural Network Framework:** Built on PyTorch, supporting feedforward neural networks (MLPs) for tabular data, CNNs for medical images, and RNN/Transformer or Linear Embeddings for text classification.
2. **Federated Optimization Algorithms:**
   - **FedAvg (Federated Averaging):** Sample-weighted coordinate averaging of local model weights.
   - **FedProx:** Addition of a proximal regularization term $\frac{\mu}{2} \|w - w^t\|^2$ to the client local objective to address system and data heterogeneity.
3. **Dual Execution Modes:**
   - **Simulation Mode:** Fast in-memory transport engine executing multiple virtual clients on a single machine for rapid testing and continuous integration.
   - **Real Distributed Mode:** Real-world execution across separate physical machines, cloud VMs, and student laptops over the Internet with strict architectural parity.

#### B. Coordination & Orchestration Layer
1. **5-State Round State Machine:** Explicit transitions through `OPEN` $\to$ `COLLECTING` $\to$ `AGGREGATING` $\to$ `DONE` (or `FAILED`) with deterministic transition guards.
2. **Participation Quorum & Timeouts:** Configurable minimum client threshold ($K_{\min}$) and per-round deadlines ($T_{\text{round}}$).
3. **Straggler & Dropout Handling:** Graceful handling of slow clients (discarding late updates after timeout) and client disconnections without stalling the global run.
4. **Idempotency & Deduplication:** Prevention of duplicate or stale update processing using unique client run tokens and UUIDv4 `update_id` keys.
5. **Coordinator Crash Recovery:** Continuous state synchronization to PostgreSQL 16, allowing a crashed coordinator to recover the last completed round and model state upon restart.

#### C. Client Onboarding & Conformance Contract
1. **Managed-Code Mode:** Pre-packaged Docker images (`cfl-client`) and release bundles with cryptographic version validation.
2. **Independent-Implementation Mode:** A public, language-agnostic REST interface requiring clients to submit a **Manifest of Conformance** prior to joining.
3. **Pre-Training Conformance Validation:**
   - Model architecture and parameter tensor compatibility.
   - Input/output shape matching, feature naming, and exact column ordering.
   - Data types (`dtypes`) and categorical encoding schema versions.
   - Unit verification (e.g., duration in seconds vs. months).
   - Deterministic synthetic test inference verification (comparing client preprocessed outputs against expected reference vectors).
4. **Deterministic Disqualification:** Immediate rejection and structured audit logging of any client failing conformance checks.

#### D. Communication & Security
1. **Transport Layer:** HTTPS over TLS 1.3 terminated at a central Caddy reverse proxy.
2. **Client-Initiated Architecture:** Clients always initiate outbound HTTP requests (`POST /register`, `GET /task`, `GET /model`, `POST /update`), avoiding firewall/NAT traversal issues.
3. **Integrity Verification:** End-to-end SHA-256 checksums on all distributed and uploaded model weight artifacts (`.safetensors`).

#### E. Observability & Reporting
1. **Telemetry & Metric Collection:** Detailed logging of Round-Trip Time (RTT), upload/download latency, payload sizes (bytes), client training duration, and server aggregation duration.
2. **Automated End-of-Run Report:**
   - Machine-readable artifact (`run_{id}_summary.json`).
   - Human-readable Markdown summary report (`run_{id}_report.md`).
   - Automated generation of evaluation plots: Global Loss/Accuracy vs. Round, Client Performance Variance, Communication Overhead, and Round Duration Breakdown.

---

### 3.2. Optional Features (Extensions & Research Enhancements)

The following features are designated as optional extensions and will be implemented only if core milestones are completed ahead of schedule:

1. **Non-IID Partitioning Experiments:**
   - Label skew using Dirichlet distribution ($\text{Dir}(\alpha)$ with varying degrees of concentration $\alpha \in [0.1, 1.0]$).
   - Feature distribution skew (e.g., regional or demographic variances between client splits).
2. **Differential Privacy (DP) Extension:**
   - Client-side gradient clipping ($\ell_2$-norm bound $C$) and Gaussian noise addition ($w \leftarrow w + \mathcal{N}(0, \sigma^2 C^2)$) to empirically quantify the privacy-utility tradeoff.
3. **Byzantine-Robust Aggregation Rules:**
   - Multi-Krum, Coordinate-wise Median, or Trimmed Mean to filter out adversarial poisoned updates.
4. **Advanced Cryptographic Protocols:**
   - Secure Multi-Party Aggregation (SecAgg) or Homomorphic Encryption (HE) (excluded from core scope as the project focuses on systems engineering rather than cryptography).
5. **Dynamic Client Tiering:**
   - Automated clustering of clients into fast and slow tiers based on historical training duration.

---

## 4. Web Application Specification & Page Architecture

The web application is built using **Next.js (React)** with a polished, modern user interface, clear separation between public documentation and authenticated operational monitoring.

```
                           +------------------------+
                           |  Next.js Frontend App  |
                           +-----------+------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
         [ Public Portal ]                       [ Authenticated Area ]
         * / (Overview & Mission)                * /app/dashboard (Live Overview)
         * /docs (API & Conformance)             * /app/rounds (Round Viewer)
         * /benchmarks (Results)                 * /app/clients (Registry & Conformance)
         * /team (About)                         * /app/updates (Update Inbox)
                                                 * /app/reports (Reports & Plots)
                                                 * /app/logs (Audit Trail)
                                                 * /app/telemetry (Grafana Embed)
```

### 4.1. Public Area (Unauthenticated)

| Route | Page Name | Primary Objective & Content |
| :--- | :--- | :--- |
| `/` | **Project Home** | High-level project summary, architectural value proposition, cross-silo FL explanation, dynamic system stats badge. |
| `/docs` | **Documentation & API** | Interactive documentation of REST API endpoints, Conformance Manifest specification, Docker setup instructions, and custom client integration guide. |
| `/benchmarks` | **Results & Benchmarks** | Public experiment showcase displaying comparison charts of FedAvg vs. FedProx, baseline centralized vs. federated accuracy, and multi-modal results. |
| `/team` | **Team & About** | Project background, contributors, academic advisors, institution details (Koç University), and links to GitHub repository. |

### 4.2. Authenticated Management Area (`/app`)

| Route | View Name | Key Components & Functionality |
| :--- | :--- | :--- |
| `/app/dashboard` | **Control Center** | Active training run status badge, active round progress card, live quorum gauge ($K/K_{\min}$), quick actions (Start Run, Abort, Download Model). |
| `/app/rounds` | **Round Viewer** | Visual interactive state machine (`OPEN` $\to$ `COLLECTING` $\to$ `AGGREGATING` $\to$ `DONE`), per-client progress checklist, round timer countdown, and straggler warning list. |
| `/app/clients` | **Client Registry & Conformance Hub** | Table of all registered clients, connection status (Online/Offline/Straggler), IP/Device info, conformance validation badge, and inspectable JSON failure trace for rejected clients. |
| `/app/updates` | **Update Inbox** | Detailed inspection table for received model weights in the active round: client ID, update timestamp, sample count weight ($n_k$), SHA-256 payload hash, and validation status. |
| `/app/reports` | **Reports & Plot Gallery** | Historical run browser, interactive Chart.js/Recharts loss/accuracy curves, communication overhead histograms, and 1-click Markdown/CSV/JSON report downloads. |
| `/app/logs` | **System Audit Trail** | Real-time searchable and filterable event stream from PostgreSQL (client registration, timeout fires, aggregation completions, crash recovery events). |
| `/app/telemetry` | **Grafana Live Telemetry** | Embedded Grafana dashboard displaying container resource utilization, network I/O, latency distributions, and memory footprints. |

---

## 5. Build Order & Implementation Roadmap

The implementation follows a **design-first, incremental layered methodology**, ensuring that each layer is tested and validated before building dependent components.

```
+-----------------------------------------------------------------------------------------+
|                                    BUILD ORDER PHASES                                   |
+-----------------------------------------------------------------------------------------+
|  LAYER 1: Design, Specifications & Data Preparation (Weeks 1 - 2) [WP1]                 |
|  * PlantUML Architecture, State Machine, Sequence, Deployment Diagrams                 |
|  * Scope Decision & Web Site Architecture Specification                                |
|  * REST API OpenAPI Contract, PostgreSQL DDL Schema, Conformance Manifest Specification |
|  * UCI Bank Marketing Data Cleaning, Preprocessing & Randomized IID Partitioner         |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
|  LAYER 2: FL Core Engine & Local Simulation Mode (Weeks 3 - 5) [WP2]                   |
|  * PyTorch Model Architecture (MLP for Tabular Banking)                                 |
|  * FedAvg & FedProx Local Training Modules with Loss Regularization                     |
|  * In-Memory Coordinator & Simulation Harness for Multi-Client Rounds                   |
|  * Baseline Benchmarking: Centralized vs. Local-Only vs. Federated                      |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
|  LAYER 3: Network Gateway, State Engine & Conformance Validator (Weeks 6 - 8) [WP3]     |
|  * PostgreSQL 16 Database Setup & Alembic Migrations                                    |
|  * FastAPI REST Gateway with TLS 1.3 & Asynchronous Route Handlers                      |
|  * Coordinator Round State Machine Engine with Quorum, Deadlines & Straggler Logic       |
|  * Conformance Engine: Dtype, Shape, Column Order & Synthetic Vector Validation         |
|  * State Persistence & Crash Recovery Mechanism (Resuming interrupted runs)             |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
|  LAYER 4: Multi-Node WAN Deployment & Robustness Testing (Weeks 9 - 11) [WP4]           |
|  * Docker Containerization (`cfl-coordinator`, `cfl-gateway`, `cfl-client`, `caddy`)   |
|  * Real Internet WAN Deployment across student laptops, university servers & cloud VMs  |
|  * Fault Injection Suite (client delay, sudden dropout, duplicate updates, coordinator kill)|
|  * Automated End-of-Run Markdown & JSON Report Generator with Matplotlib Plots          |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
|  LAYER 5: Presentation, Observability & Multi-Modal Experiments (Weeks 12 - 14) [WP5]   |
|  * Next.js Web Application (Public Documentation & Authenticated Control Dashboard)     |
|  * Grafana Telemetry Dashboard Integration via Prometheus/PostgreSQL                    |
|  * Multi-Modal Dataset Evaluations (MedMNIST Image & UCI SMS Spam Text)                 |
|  * (Optional) Non-IID Dirichlet Skew & Differential Privacy Experiments                 |
|  * Final Project Poster, Demo Video & Capstone Technical Report                         |
+-----------------------------------------------------------------------------------------+
```

---

## 6. Acceptance Criteria for T1.1 Sign-Off

- [x] Clear demarcation between mandatory core features and optional research extensions.
- [x] Explicit definition of cross-silo real-world distributed assumptions (no raw data leakage, client-side firewalls, independent client conformance).
- [x] Formal definition of public and authenticated web application page hierarchy.
- [x] Ordered, 5-layer incremental build sequence aligned with project weeks and course deliverables.
- [x] Full alignment with advisor directives and Capstone guidelines.
