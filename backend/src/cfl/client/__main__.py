"""Managed-code client entrypoint (the cfl-client image): `python -m cfl.client <command>`.

    check  verify this site's setup: token, partition files, task, state directory, and that
           the gateway is reachable over HTTPS with a matching protocol version
    run    take part in training runs (the client runtime; see planning/interfaces.md)

Configuration comes from the environment, so the same image runs at every site:

    CFL_SERVER_URL      gateway base URL, e.g. https://fl.203-0-113-7.sslip.io
    CFL_CLIENT_ID       this site's approved client id
    CFL_TASK            tabular | text
    CFL_TASK_SPEC       optional module:Class of the site's TaskSpec implementation
    CFL_TOKEN_FILE      file with the bearer token (a Docker secret), default /run/secrets/cfl_token
    CFL_PARTITION_DIR   this site's partition (train.csv, val.csv, test.csv), mounted read-only
    CFL_STATE_DIR       writable directory for pending updates and downloaded models
    CFL_CA_BUNDLE       optional CA certificate to trust, for a server with its own CA
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import httpx

from cfl.protocol import PROTOCOL_VERSION
from cfl.tasks.registry import get_task

SPLITS = ("train", "val", "test")


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if not value:
        raise SystemExit(f"error: set {name}")
    return value


def check() -> int:
    problems: list[str] = []

    def ok(message: str) -> None:
        print(f"  ok    {message}")

    def fail(message: str) -> None:
        print(f"  FAIL  {message}")
        problems.append(message)

    client_id = _env("CFL_CLIENT_ID")
    print(f"cfl-client {os.environ.get('CFL_CLIENT_VERSION', 'dev')} as {client_id}")

    token_file = Path(_env("CFL_TOKEN_FILE", "/run/secrets/cfl_token"))
    try:
        token = token_file.read_text().strip()
        ok(f"token file {token_file}") if token else fail(f"token file {token_file} is empty")
    except OSError as exc:
        fail(f"token file {token_file}: {exc.strerror}")

    try:
        task = get_task(_env("CFL_TASK"))
        ok(f"task {task.name} ({task.dataset})")
    except ValueError as exc:
        fail(str(exc))
        task = None

    spec_path = os.environ.get("CFL_TASK_SPEC")
    if spec_path:
        module, _, cls = spec_path.partition(":")
        try:
            getattr(importlib.import_module(module), cls)
            ok(f"TaskSpec {spec_path}")
        except (ImportError, AttributeError) as exc:
            fail(f"TaskSpec {spec_path}: {exc}")

    partition = Path(_env("CFL_PARTITION_DIR", "/data"))
    for split in SPLITS:
        path = partition / f"{split}.csv"
        if not path.is_file():
            fail(f"missing {path}")
        elif task is not None:
            try:
                inputs, labels = task.load_split(path)
                ok(f"{path}: {len(labels)} rows, input {tuple(inputs.shape[1:])}")
            except Exception as exc:  # report any parsing problem instead of a traceback
                fail(f"{path}: {exc}")

    state = Path(_env("CFL_STATE_DIR", "/state"))
    probe = state / ".write-test"
    try:
        state.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok")
        probe.unlink()
        ok(f"state directory {state} is writable")
    except OSError as exc:
        fail(f"state directory {state}: {exc.strerror}")

    server = _env("CFL_SERVER_URL").rstrip("/")
    if not server.startswith("https://"):
        fail(f"CFL_SERVER_URL must use https:// (got {server})")
    else:
        verify: str | bool = os.environ.get("CFL_CA_BUNDLE") or True
        try:
            response = httpx.get(f"{server}/health", timeout=10, verify=verify)
            response.raise_for_status()
            remote = response.json().get("protocol_version")
            if remote == PROTOCOL_VERSION:
                ok(f"gateway {server} reachable, protocol {remote}")
            else:
                fail(f"gateway speaks protocol {remote}, this client {PROTOCOL_VERSION}")
        except httpx.HTTPError as exc:
            fail(f"gateway {server}: {exc}")

    print("ready" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


def run() -> int:
    print(
        "error: the client runtime (polling, training, uploads) is not implemented yet; "
        "`check` verifies the setup in the meantime",
        file=sys.stderr,
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cfl.client",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=["check", "run"])
    args = parser.parse_args(argv)
    return check() if args.command == "check" else run()


if __name__ == "__main__":
    sys.exit(main())
