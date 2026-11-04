"""Client settings: command-line options, environment variables and a config file.

Precedence, highest first: command-line option, environment variable (CFL_*), config file
(--config, YAML), default. The container image sets its defaults through the environment;
a site running the client directly with Python usually writes a config file
(deploy/client/client.yaml.example).

The bearer token is never a setting itself, only the file that holds it: values on a command
line end up in shell history and in `ps`, and environment variables are visible to every
process of the user.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


@dataclass(frozen=True)
class Option:
    flag: str
    env: str
    help: str
    type: type = str


OPTIONS: dict[str, Option] = {
    "server_url": Option("--server-url", "CFL_SERVER_URL", "gateway base URL (https://...)"),
    "client_id": Option("--client-id", "CFL_CLIENT_ID", "this site's approved client id"),
    "task": Option("--task", "CFL_TASK", "tabular | text"),
    "task_spec": Option("--task-spec", "CFL_TASK_SPEC", "module:Class of the site's TaskSpec"),
    "token_file": Option("--token-file", "CFL_TOKEN_FILE", "file holding the bearer token"),
    "partition_dir": Option(
        "--partition-dir", "CFL_PARTITION_DIR", "folder with train.csv, val.csv, test.csv"
    ),
    "state_dir": Option(
        "--state-dir", "CFL_STATE_DIR", "writable folder for pending updates and final models"
    ),
    "ca_bundle": Option("--ca-bundle", "CFL_CA_BUNDLE", "CA certificate of a self-signed server"),
    "poll_interval": Option(
        "--poll-interval", "CFL_POLL_INTERVAL", "seconds between polls when idle", float
    ),
    "seed": Option("--seed", "CFL_SEED", "seed for local training", int),
    "client_version": Option(
        "--client-version", "CFL_CLIENT_VERSION", "version announced at registration"
    ),
}
PATH_FIELDS = ("token_file", "partition_dir", "state_dir", "ca_bundle")


class SettingsError(ValueError):
    pass


class ClientSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server_url: str | None = None
    client_id: str | None = None
    task: str | None = None
    task_spec: str | None = None
    token_file: Path = Path("/run/secrets/cfl_token")
    partition_dir: Path = Path("/data")
    state_dir: Path = Path("/state")
    ca_bundle: Path | None = None
    poll_interval: float = Field(default=3.0, gt=0)
    seed: int = 0
    client_version: str = "cfl-client/dev"

    @field_validator("server_url")
    @classmethod
    def _https(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("https://"):
            raise ValueError("must start with https://")
        return value.rstrip("/") if value else value

    @field_validator("task")
    @classmethod
    def _known_task(cls, value: str | None) -> str | None:
        if value is not None and value not in ("tabular", "text"):
            raise ValueError("must be tabular or text")
        return value

    @property
    def verify(self) -> str | bool:
        """httpx `verify`: the CA bundle when given, otherwise the system CAs."""
        return str(self.ca_bundle) if self.ca_bundle else True

    def require(self, *names: str) -> None:
        missing = [n for n in names if getattr(self, n) in (None, "")]
        if missing:
            hints = "; ".join(
                f"{n}: {OPTIONS[n].flag}, {OPTIONS[n].env} or `{n}:` in the config file"
                for n in missing
            )
            raise SettingsError(f"missing setting(s): {hints}")

    def describe(self) -> str:
        """For logs: every setting, never the token itself."""
        return ", ".join(f"{k}={v}" for k, v in self.model_dump().items() if v is not None)


def load_settings(
    config_file: Path | None = None,
    cli: Mapping[str, Any] | None = None,
    env: Mapping[str, str] = os.environ,
) -> ClientSettings:
    values: dict[str, Any] = {}
    if config_file is not None:
        try:
            raw = yaml.safe_load(Path(config_file).read_text()) or {}
        except OSError as exc:
            raise SettingsError(f"cannot read config file {config_file}: {exc.strerror}") from exc
        if not isinstance(raw, dict):
            raise SettingsError(f"{config_file} must contain a mapping of settings")
        if "token" in raw:
            raise SettingsError("put the token into a file and set token_file, not token")
        base = Path(config_file).resolve().parent
        for name in PATH_FIELDS:  # relative paths in the file are relative to the file
            if raw.get(name) and not Path(raw[name]).is_absolute():
                raw[name] = str(base / raw[name])
        values.update(raw)
    for name, option in OPTIONS.items():
        if env.get(option.env):
            values[name] = env[option.env]
    for name, value in (cli or {}).items():
        if value is not None:
            values[name] = value
    try:
        return ClientSettings.model_validate(values)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'config'}: {e['msg']}" for e in exc.errors()
        )
        raise SettingsError(f"invalid settings: {problems}") from None
