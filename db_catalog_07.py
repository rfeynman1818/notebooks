# ===============================================================================
# FILE: catalog_db.py
# ===============================================================================

from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple
import pandas as pd
import pymssql
import re, uuid

# -------------------------------------------------------------------------------
# SQL CONSTANTS
# -------------------------------------------------------------------------------

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

DEFAULT_OBJECTS_SQL = """
SELECT
    s.name AS schema_name,
    o.name AS object_name,
    CASE o.type
        WHEN 'U' THEN 'TABLE'
        WHEN 'V' THEN 'VIEW'
    END AS object_type,
    o.create_date
FROM sys.objects o
JOIN sys.schemas s ON o.schema_id = s.schema_id
WHERE o.type IN ('U','V')
ORDER BY s.name, o.name;
"""

DEFAULT_VIEWS_SQL = """
SELECT
    s.name AS schema_name,
    v.name AS view_name,
    v.create_date
FROM sys.views v
JOIN sys.schemas s ON v.schema_id = s.schema_id
ORDER BY s.name, v.name;
"""

# -------------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------------

@dataclass(frozen=True)
class SqlServerConfig:
    server: str
    user: str
    password: str
    database: str = "master"
    charset: str = "UTF-8"

# -------------------------------------------------------------------------------
# INTERNAL HELPERS (avoid duplication)
# -------------------------------------------------------------------------------

def _rows_to_df(rows, column_order: Optional[Sequence[str]]):
    if not rows:
        return pd.DataFrame(columns=column_order or [])
    df = pd.DataFrame(rows)
    if column_order is not None:
        cols = [c for c in column_order if c in df.columns]
        rest = [c for c in df.columns if c not in cols]
        df = df[cols + rest]
    return df

def _warn(msg: str, *, verbose: bool):
    if verbose:
        print(msg)

def _uid8() -> str:
    return uuid.uuid4().hex[:8]

def _safe_token(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"

def _base_name(schema: str, table: str) -> str:
    return f"{_safe_token(schema)}_{_safe_token(table)}"

# -------------------------------------------------------------------------------
# CONNECTION MANAGEMENT
# -------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------
# CATALOG: DATABASES
# -------------------------------------------------------------------------------

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
        _warn("\n⚠️  No databases were returned – you may not have the VIEW ANY DATABASE permission.", verbose=verbose)
    return _rows_to_df(rows, column_order)

# -------------------------------------------------------------------------------
# CATALOG: TABLES & VIEWS (together)
# -------------------------------------------------------------------------------

def list_tables_and_views_df(
    cfg: SqlServerConfig,
    *,
    sql: str = DEFAULT_OBJECTS_SQL,
    column_order: Optional[Sequence[str]] = (
        "schema_name",
        "object_name",
        "object_type",
        "create_date",
    ),
    verbose: bool = True,
) -> pd.DataFrame:
    with connect(cfg, as_dict=True, verbose=verbose) as cur:
        cur.execute(sql)
        rows = cur.fetchall() or []
    if not rows:
        _warn("\n⚠️  No tables or views were returned – check permissions on this database.", verbose=verbose)
    return _rows_to_df(rows, column_order)

# -------------------------------------------------------------------------------
# CATALOG: VIEWS (only)
# -------------------------------------------------------------------------------

def list_views_df(
    cfg: SqlServerConfig,
    *,
    sql: str = DEFAULT_VIEWS_SQL,
    column_order: Optional[Sequence[str]] = (
        "schema_name",
        "view_name",
        "create_date",
    ),
    verbose: bool = True,
) -> pd.DataFrame:
    with connect(cfg, as_dict=True, verbose=verbose) as cur:
        cur.execute(sql)
        rows = cur.fetchall() or []
    if not rows:
        _warn("\n⚠️  No views were returned – check permissions on this database.", verbose=verbose)
    return _rows_to_df(rows, column_order)

# -------------------------------------------------------------------------------
# DATA LOADING: TABLE OR VIEW -> DataFrame
# -------------------------------------------------------------------------------

def load_table_df(
    cfg: SqlServerConfig,
    *,
    schema: str,
    table: str,
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    col_sql = ", ".join(f"[{c}]" for c in columns) if columns else "*"
    if limit is None:
        sql = f"SELECT {col_sql} FROM [{schema}].[{table}]"
        if where:
            sql += f" WHERE {where}"
    else:
        sql = f"SELECT TOP ({limit}) {col_sql} FROM [{schema}].[{table}]"
        if where:
            sql += f" WHERE {where}"

    with connect(cfg, as_dict=True, verbose=verbose) as cur:
        cur.execute(sql)
        rows = cur.fetchall() or []

    if not rows:
        _warn(f"\n⚠️  No rows returned from {schema}.{table}.", verbose=verbose)
        return pd.DataFrame(columns=columns or [])

    return pd.DataFrame(rows)

# -------------------------------------------------------------------------------
# DATA LOADING: CHUNKED (large tables/views)
# -------------------------------------------------------------------------------

def load_table_chunks(
    cfg: SqlServerConfig,
    *,
    schema: str,
    table: str,
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    order_by: Optional[Sequence[str]] = None,
    chunksize: int = 100_000,
    verbose: bool = True,
):
    if chunksize <= 0:
        raise ValueError("chunksize must be > 0")

    col_sql = ", ".join(f"[{c}]" for c in columns) if columns else "*"
    base = f"SELECT {col_sql} FROM [{schema}].[{table}]"
    if where:
        base += f" WHERE {where}"

    if order_by and len(order_by) > 0:
        order_sql = ", ".join(f"[{c}]" for c in order_by)
    else:
        order_sql = "(SELECT NULL)"

    offset = 0
    first = True
    while True:
        sql = f"{base} ORDER BY {order_sql} OFFSET {offset} ROWS FETCH NEXT {chunksize} ROWS ONLY"
        with connect(cfg, as_dict=True, verbose=(verbose and first)) as cur:
            cur.execute(sql)
            rows = cur.fetchall() or []
        first = False

        if not rows:
            if offset == 0:
                _warn(f"\n⚠️  No rows returned from {schema}.{table}.", verbose=verbose)
            break

        yield pd.DataFrame(rows)
        offset += chunksize

def load_table_df_chunked(
    cfg: SqlServerConfig,
    *,
    schema: str,
    table: str,
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    order_by: Optional[Sequence[str]] = None,
    chunksize: int = 100_000,
    max_chunks: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    dfs = []
    n = 0
    for chunk_df in load_table_chunks(
        cfg,
        schema=schema,
        table=table,
        columns=columns,
        where=where,
        order_by=order_by,
        chunksize=chunksize,
        verbose=verbose,
    ):
        dfs.append(chunk_df)
        n += 1
        if max_chunks is not None and n >= max_chunks:
            break
    if not dfs:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(dfs, ignore_index=True)

# -------------------------------------------------------------------------------
# EXPORT: TABLE/VIEW -> CSV WITH UNIQUE NAME
# -------------------------------------------------------------------------------

def export_table_csv_unique(
    cfg: SqlServerConfig,
    *,
    schema: str,
    table: str,
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    limit: Optional[int] = None,
    chunked: bool = False,
    chunksize: int = 100_000,
    order_by: Optional[Sequence[str]] = None,
    max_chunks: Optional[int] = None,
    out_dir: str | Path = ".",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, str, Path]:
    uid = _uid8()
    base = _base_name(schema, table)
    df_name = f"{base}_{uid}"

    if chunked:
        df = load_table_df_chunked(
            cfg,
            schema=schema,
            table=table,
            columns=columns,
            where=where,
            order_by=order_by,
            chunksize=chunksize,
            max_chunks=max_chunks,
            verbose=verbose,
        )
    else:
        df = load_table_df(
            cfg,
            schema=schema,
            table=table,
            columns=columns,
            where=where,
            limit=limit,
            verbose=verbose,
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / f"{base}_{uid}.csv"
    df.to_csv(csv_path, index=False)

    if verbose:
        print(f"✅ DataFrame name: {df_name}")
        print(f"✅ CSV written: {csv_path}")
        print(df.head())

    return df, df_name, csv_path

def export_table_csv_unique_into_globals(
    cfg: SqlServerConfig,
    *,
    schema: str,
    table: str,
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    limit: Optional[int] = None,
    chunked: bool = False,
    chunksize: int = 100_000,
    order_by: Optional[Sequence[str]] = None,
    max_chunks: Optional[int] = None,
    out_dir: str | Path = ".",
    verbose: bool = True,
    target_globals: Optional[dict] = None,
) -> Tuple[pd.DataFrame, str, Path]:
    df, df_name, csv_path = export_table_csv_unique(
        cfg,
        schema=schema,
        table=table,
        columns=columns,
        where=where,
        limit=limit,
        chunked=chunked,
        chunksize=chunksize,
        order_by=order_by,
        max_chunks=max_chunks,
        out_dir=out_dir,
        verbose=verbose,
    )
    g = globals() if target_globals is None else target_globals
    g[df_name] = df
    return df, df_name, csv_path
