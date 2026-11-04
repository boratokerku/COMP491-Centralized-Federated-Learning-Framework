"""Run the coordinator process for distributed mode; same as `python -m cfl.coordinator`.

Usage (from the repository root, after `alembic upgrade head`):
    export CFL_DATABASE_URL=postgresql+psycopg://cfl:<password>@127.0.0.1:5432/cfl
    python backend/scripts/run_coordinator.py --models /data/models
    python backend/scripts/run_coordinator.py --models /data/models --start backend/configs/x.yaml
"""

from __future__ import annotations

import sys

from cfl.coordinator.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
