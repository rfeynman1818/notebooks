from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence, Any
import pandas as pd
import pymssql

DEFAULT_DB_LIST_SQL = """
SELECT
    name AS database_name,
    database_id,
    state_desc,
    recovery_model_desc,
    create_date
FROM sys.databases
ORDER BY name;
"""

@dataclass(frozen=True)
class SqlServerConfig:
    server: str
    user: str
    password: str
    database: str = "master"
    charset: str = "UTF-8"

@contextmanager
def connect(
    cfg: SqlServerConfig,
    *,
    as_dict: bool = True,
    verbose: bool = True,
) -> Iterator[pymssql.Cursor]:
    conn = None
    try:
        conn = pymssql.connect(
            server=cfg.server,
            user=cfg.user,
            password=cfg.password,
            database=cfg.database,
            charset=cfg.charset,
        )
        if verbose:
            print("✅ Connection successful")
        cur = conn.cursor(as_dict=as_dict)
        yield cur
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

def list_databases_df(
    cfg: SqlServerConfig,
    *,
    sql: str = DEFAULT_DB_LIST_SQL,
    column_order: Optional[Sequence[str]] = (
        "database_name",
        "database_id",
        "state_desc",
        "recovery_model_desc",
        "create_date",
    ),
    verbose: bool = True,
) -> pd.DataFrame:
    with connect(cfg, as_dict=True, verbose=verbose) as cur:
        cur.execute(sql)
        rows = cur.fetchall() or []

    if not rows:
        if verbose:
            print("\n⚠️  No databases were returned – you may not have the VIEW ANY DATABASE permission.")
        return pd.DataFrame(columns=column_order or [])

    df = pd.DataFrame(rows)

    if column_order is not None:
        cols = [c for c in column_order if c in df.columns]
        rest = [c for c in df.columns if c not in cols]
        df = df[cols + rest]

    return df
