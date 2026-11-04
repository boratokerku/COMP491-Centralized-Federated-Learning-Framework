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
    CFL_POLL_INTERVAL   seconds between task polls when idle (default 3)
    CFL_SEED            seed for local training (default 0); results are reproducible per seed

`run --exit-after-final` stops once the final model of a run this site trained in has been
downloaded and verified (useful for demos and tests); without it the client keeps serving.
"""

from __future__ import annotations

import argparse
import importlib
import os
import signal
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


def _load_task_spec(task_name: str):
    """CFL_TASK_SPEC=module:Class (constructed without arguments), else the reference spec."""
    from cfl.client.reference import ReferenceTaskSpec

    spec_path = os.environ.get("CFL_TASK_SPEC")
    if not spec_path:
        return ReferenceTaskSpec(task_name)
    module, _, cls = spec_path.partition(":")
    return getattr(importlib.import_module(module), cls)()


def run(exit_after_final: bool = False) -> int:
    from cfl.client.http import GatewayHTTPError, HttpTransport
    from cfl.client.runtime import ClientRuntime, ConformanceFailed

    # As PID 1 in a container, SIGTERM is ignored unless handled. Stopping is always safe:
    # a pending update stays on disk and is resent after the restart.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    task_name = _env("CFL_TASK")
    token = Path(_env("CFL_TOKEN_FILE", "/run/secrets/cfl_token")).read_text().strip()
    version = os.environ.get("CFL_CLIENT_VERSION", "cfl-client/dev")
    transport = HttpTransport(
        _env("CFL_SERVER_URL"),
        token,
        verify=os.environ.get("CFL_CA_BUNDLE") or True,
        user_agent=version,
    )
    runtime = ClientRuntime(
        transport=transport,
        task=_load_task_spec(task_name),
        client_id=_env("CFL_CLIENT_ID"),
        client_version=version,
        dataset=get_task(task_name).dataset,
        partition_dir=Path(_env("CFL_PARTITION_DIR", "/data")),
        state_dir=Path(_env("CFL_STATE_DIR", "/state")),
        poll_interval_s=float(os.environ.get("CFL_POLL_INTERVAL", "3")),
        seed=int(os.environ.get("CFL_SEED", "0")),
    )
    try:
        runtime.run(exit_after_final=exit_after_final)
    except ConformanceFailed as exc:
        print(f"error: conformance check failed: {exc}", file=sys.stderr)
        print(f"details: {exc.details}", file=sys.stderr)
        return 1
    except GatewayHTTPError as exc:
        print(f"error: the gateway refused this client: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("stopped; start again to continue")
    finally:
        transport.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cfl.client",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=["check", "run"])
    parser.add_argument("--exit-after-final", action="store_true", help="with run: see above")
    args = parser.parse_args(argv)
    return check() if args.command == "check" else run(args.exit_after_final)


if __name__ == "__main__":
    sys.exit(main())
