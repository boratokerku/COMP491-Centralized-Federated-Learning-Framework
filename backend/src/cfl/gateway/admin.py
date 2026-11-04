"""Client administration: `cfl-admin <command>` (database from CFL_DATABASE_URL).

    cfl-admin add-client bank-a [--token-file bank-a.token]   approve a client, print its token
    cfl-admin rotate-token bank-a [--token-file ...]           replace a leaked token
    cfl-admin disable bank-a | enable bank-a                   block / unblock a client
    cfl-admin list                                             approved clients and their state

A token is shown exactly once; only its hash is stored. Hand it to the site out of band and
let the site put it into its Docker secret file (deploy/client/token.txt). With --token-file it
is written to a file readable only by the current user instead of being printed.

In the containers: docker compose -f deploy/docker-compose.yml run --rm gateway cfl-admin list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cfl.db.state_store import engine_from_env
from cfl.gateway.clients import ClientRegistry


def _deliver(token: str, client_id: str, token_file: Path | None) -> None:
    if token_file is None:
        print(f"token for {client_id} (shown once, store it now):\n{token}")
        return
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    print(f"token for {client_id} written to {token_file} (mode 600)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cfl-admin", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("add-client", "rotate-token"):
        p = sub.add_parser(name)
        p.add_argument("client_id")
        p.add_argument("--token-file", type=Path, default=None, help="write the token here")
    for name in ("disable", "enable"):
        sub.add_parser(name).add_argument("client_id")
    sub.add_parser("list")
    args = parser.parse_args(argv)

    registry = ClientRegistry(engine_from_env())
    try:
        if args.command == "add-client":
            _deliver(registry.add_client(args.client_id), args.client_id, args.token_file)
        elif args.command == "rotate-token":
            _deliver(registry.rotate_token(args.client_id), args.client_id, args.token_file)
        elif args.command == "disable":
            registry.disable(args.client_id)
            print(f"{args.client_id} disabled")
        elif args.command == "enable":
            registry.enable(args.client_id)
            print(f"{args.client_id} enabled")
        else:
            clients = registry.list()
            print(f"{'client':<20} {'status':<11} {'version':<28} last seen")
            for c in clients:
                seen = c.last_seen_at.isoformat(timespec="seconds") if c.last_seen_at else "-"
                version = c.client_version or "-"
                print(f"{c.client_id:<20} {c.status.value:<11} {version:<28} {seen}")
    except KeyError as exc:
        print(f"error: unknown client {exc}", file=sys.stderr)
        return 1
    except (ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
