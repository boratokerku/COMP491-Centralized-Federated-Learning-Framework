# COMP491 – Centralized Federated Learning Framework

A cross-silo federated learning framework: a central coordinator trains PyTorch models across
independently operated clients without collecting their raw data. Supports FedAvg and FedProx,
a local simulation mode and a real distributed mode over HTTPS.

## Repository layout

```
backend/
  src/cfl/
    core/          # Strategy (FedAvg/FedProx), Transport interface, TaskSpec, serialization
    protocol/      # Pydantic message schemas + protocol version
    gateway/       # FastAPI gateway (only entry point for clients)
    coordinator/   # round state machine (state lives in PostgreSQL)
    client/        # client runtime + conformance checker
    tasks/         # one subpackage per modality: tabular/, image/
    reporting/     # end-of-training report, exports, plots
  configs/         # run configs (YAML)
  tests/
frontend/          # Next.js web app (public site + internal area)
deploy/            # docker-compose (PostgreSQL, Grafana; later gateway, coordinator, Caddy)
diagrams/          # architecture, sequence, state-machine, flowcharts, deployment
documents/         # project description and proposal
data/              # local datasets (gitignored)
```

## Getting started

Requirements: Python 3.11+, Docker.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Run the gateway locally:

```bash
uvicorn cfl.gateway.app:app --reload
```

Start PostgreSQL and Grafana:

```bash
cp deploy/.env.example deploy/.env   # then set the passwords
docker compose -f deploy/docker-compose.yml up -d
```

## Conventions

- Model weights are exchanged as `safetensors` with a SHA-256 checksum. Never unpickle data
  received over the network.
- Raw data never goes into the repository (`data/` is gitignored).
- Lint with `ruff check .`, test with `pytest` (both run in CI).
