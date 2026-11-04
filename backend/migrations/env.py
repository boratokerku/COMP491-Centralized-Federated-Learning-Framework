"""Alembic environment: migrates the database named by CFL_DATABASE_URL to cfl.db.schema."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from cfl.db.schema import metadata

if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name)


def _url() -> str:
    url = os.environ.get("CFL_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "set CFL_DATABASE_URL, e.g. postgresql+psycopg://cfl:<password>@127.0.0.1:5432/cfl"
        )
    return url


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
