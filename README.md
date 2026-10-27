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
    tasks/         # one subpackage per modality: tabular/ + second modality (image or text, open)
    reporting/     # end-of-training report, exports, plots
  configs/         # run configs (YAML)
  tests/
frontend/          # Next.js web app (public site + internal area)
deploy/            # docker-compose (PostgreSQL, Grafana; later gateway, coordinator, Caddy)
diagrams/          # PlantUML sources + rendered PNG/SVG (architecture, sequence, state machine, deployment)
planning/          # scope, requirements and open decisions
.github/           # CI workflow, issue and PR templates
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

## Design specifications

- [PostgreSQL schema and sequence-diagram DB mapping](planning/db-schema.md)
- [Client conformance contract](planning/conformance-manifest.md), [JSON Schema](planning/conformance-manifest.schema.json), and [Bank Marketing example](planning/examples/bank_marketing_manifest.json)
- [ER diagram source](diagrams/database/er_diagram.puml) and rendered [PNG](diagrams/database/er_diagram.png)

## Workflow

1. Open an issue for every task, including non-coding work (research, meetings, documentation,
   data, slides). Use the **Task** template: description, acceptance criteria, assignee.
2. Create a branch for the issue (from the issue page: *Development → Create a branch*).
3. Commit with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
   `feat(gateway): add update upload endpoint`, `docs(readme): ...`, `fix(...)`, `test(...)`.
4. Open a PR that references the issue (`Closes #ID`); another member reviews it.
5. Merge after review and green CI; the issue closes automatically.

Scope and team decisions are in [planning/scope.md](planning/scope.md) (decisions: section 7).

## Conventions

- Model weights are exchanged as `safetensors` with a SHA-256 checksum. Never unpickle data
  received over the network.
- Raw data never goes into the repository (`data/` is gitignored).
- Lint with `ruff check .`, test with `pytest` (both run in CI).
