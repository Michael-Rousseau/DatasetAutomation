import json
import sqlite3
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import NewType

from dataset_automation.storage.schema import SCHEMA_SQL, SCHEMA_VERSION

RunId = NewType("RunId", int)


class SchemaVersionError(RuntimeError):
    pass


def open_database(path: Path) -> sqlite3.Connection:
    """Open (and create on first use) the shared database at `path`."""
    connection = sqlite3.connect(path)
    # SQLite leaves foreign keys off by default, per connection.
    connection.execute("PRAGMA foreign_keys = ON")
    # WAL lets several processes read while one module writes.
    connection.execute("PRAGMA journal_mode = WAL")
    _ensure_schema(connection)
    return connection


def start_run(
    connection: sqlite3.Connection,
    module: str,
    parameters: Mapping[str, object],
    code_version: str | None = None,
) -> RunId:
    cursor = connection.execute(
        "INSERT INTO runs (module, code_version, parameters, started_at) VALUES (?, ?, ?, ?)",
        (
            module,
            code_version or current_code_version(),
            json.dumps(parameters, sort_keys=True),
            datetime.now(UTC).isoformat(),
        ),
    )
    connection.commit()
    assert cursor.lastrowid is not None
    return RunId(cursor.lastrowid)


def current_code_version() -> str:
    """Git commit of the working tree (suffixed `-dirty` if modified), else the package version."""
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return version("dataset-automation")


def _ensure_schema(connection: sqlite3.Connection) -> None:
    found_version = connection.execute("PRAGMA user_version").fetchone()[0]
    if found_version == SCHEMA_VERSION:
        return
    if found_version != 0:
        raise SchemaVersionError(
            f"database has schema v{found_version}, this code expects v{SCHEMA_VERSION}"
        )
    connection.executescript(SCHEMA_SQL)
