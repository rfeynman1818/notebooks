from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple
import re, uuid
import pandas as pd

from db_catalog import SqlServerConfig, load_table_df, load_table_df_chunked

def _uid8() -> str:
    return uuid.uuid4().hex[:8]

def _safe_token(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"

def _base_name(schema: str, table: str) -> str:
    return f"{_safe_token(schema)}_{_safe_token(table)}"

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

# Optional helper if you still want the “unique dataframe variable” injected like your snapshot:
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
