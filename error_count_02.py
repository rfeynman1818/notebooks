# ===============================================================================
# FILE: reports_design_error_count.py
# ===============================================================================

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any
import re, uuid
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------------
# IMPORT DB UTILITIES
# -------------------------------------------------------------------------------

try:
    from db_catalog import SqlServerConfig, connect, load_table_df, load_table_df_chunked
except Exception:
    from catalog_db import SqlServerConfig, connect, load_table_df, load_table_df_chunked

# -------------------------------------------------------------------------------
# SQL CONSTANTS
# -------------------------------------------------------------------------------

DEFAULT_SOURCE_SQL = """
SELECT
    CHG_ACTVTY_NUM,
    RSN_FOR_CHG_CODE_MJR,
    ACTL_PCKG_RLS_DATE,
    PRGRM_NAME,
    PMM_Program_ID,
    Fiscal_Release_Month,
    CHG_ACTVTY_CRT_DATE
FROM dbo.FISCAL_MONTH_CT_REPORTING
"""

# -------------------------------------------------------------------------------
# INTERNAL HELPERS
# -------------------------------------------------------------------------------

def _uid8() -> str:
    return uuid.uuid4().hex[:8]

def _safe_token(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"

def _warn(msg: str, *, verbose: bool):
    if verbose:
        print(msg)

def export_df_csv_unique(
    df: pd.DataFrame,
    *,
    base_name: str,
    out_dir: str | Path = ".",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, str, Path]:
    uid = _uid8()
    base = _safe_token(base_name)
    df_name = f"{base}_{uid}"

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / f"{base}_{uid}.csv"
    df.to_csv(csv_path, index=False)

    if verbose:
        print(f"✅ DataFrame name: {df_name}")
        print(f"✅ CSV written: {csv_path}")
        print(df.head())

    return df, df_name, csv_path

def inject_df_into_globals(
    df: pd.DataFrame,
    *,
    name: str,
    target_globals: Optional[dict] = None,
) -> str:
    g = globals() if target_globals is None else target_globals
    g[name] = df
    return name

# -------------------------------------------------------------------------------
# GENERIC QUERY RUNNERS
# -------------------------------------------------------------------------------

def run_query_df(
    cfg: SqlServerConfig,
    *,
    sql: str,
    params: Optional[Sequence[Any]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    with connect(cfg, as_dict=True, verbose=verbose) as cur:
        if params is None:
            cur.execute(sql)
        else:
            cur.execute(sql, params)
        rows = cur.fetchall() or []
    if not rows:
        _warn("\n⚠️  Query returned 0 rows.", verbose=verbose)
        return pd.DataFrame()
    return pd.DataFrame(rows)

@dataclass(frozen=True)
class TableLoadSpec:
    schema: str
    table: str
    columns: Optional[Sequence[str]] = None
    where: Optional[str] = None
    limit: Optional[int] = None
    chunked: bool = False
    chunksize: int = 100_000
    order_by: Optional[Sequence[str]] = None
    max_chunks: Optional[int] = None

def load_spec_df(cfg: SqlServerConfig, spec: TableLoadSpec, *, verbose: bool = True) -> pd.DataFrame:
    if spec.chunked:
        return load_table_df_chunked(
            cfg,
            schema=spec.schema,
            table=spec.table,
            columns=spec.columns,
            where=spec.where,
            order_by=spec.order_by,
            chunksize=spec.chunksize,
            max_chunks=spec.max_chunks,
            verbose=verbose,
        )
    return load_table_df(
        cfg,
        schema=spec.schema,
        table=spec.table,
        columns=spec.columns,
        where=spec.where,
        limit=spec.limit,
        verbose=verbose,
    )

# -------------------------------------------------------------------------------
# REPORT: DESIGN ERROR COUNT
# -------------------------------------------------------------------------------

def generate_design_error_count_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Design Error Count:
    
    Count records where:
    - RSN_FOR_CHG_CODE_MJR = "02 - Design Error"
    - ACTL_PCKG_RLS_DATE IS NOT NULL
    
    Grouped by PRGRM_NAME, PMM_Program_ID, and Fiscal_Release_Month
    """
    required = [
        "CHG_ACTVTY_NUM",
        "RSN_FOR_CHG_CODE_MJR",
        "ACTL_PCKG_RLS_DATE",
        "PRGRM_NAME",
        "PMM_Program_ID",
        "Fiscal_Release_Month",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter for Design Error records with a released package date
    design_error_mask = (
        (df["RSN_FOR_CHG_CODE_MJR"] == "02 - Design Error")
        & df["ACTL_PCKG_RLS_DATE"].notna()
    )

    # Aggregate by program and fiscal month
    agg = (
        df.assign(
            design_error_cnt=design_error_mask.astype(int),
        )
        .groupby(["PRGRM_NAME", "PMM_Program_ID", "Fiscal_Release_Month"], as_index=False)
        .agg(
            design_error_count=("design_error_cnt", "sum"),
        )
    )

    # Select and order columns for final report
    report = agg[
        [
            "PRGRM_NAME",
            "PMM_Program_ID",
            "Fiscal_Release_Month",
            "design_error_count",
        ]
    ]

    return report

# -------------------------------------------------------------------------------
# REPORT RUNNERS
# -------------------------------------------------------------------------------

def run_design_error_count_report_from_query(
    cfg: SqlServerConfig,
    *,
    sql: str = DEFAULT_SOURCE_SQL,
    params: Optional[Sequence[Any]] = None,
    out_dir: str | Path = ".",
    export_csv: bool = True,
    inject_globals: bool = False,
    target_globals: Optional[dict] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Optional[str], Optional[Path]]:
    """
    Run the Design Error Count report from a SQL query.
    
    Returns:
        Tuple of (report_df, df_name, csv_path)
    """
    src_df = run_query_df(cfg, sql=sql, params=params, verbose=verbose)
    report_df = generate_design_error_count_report(src_df)

    df_name = None
    csv_path = None

    if export_csv:
        report_df, df_name, csv_path = export_df_csv_unique(
            report_df,
            base_name="design_error_count",
            out_dir=out_dir,
            verbose=verbose,
        )

    if inject_globals:
        if df_name is None:
            df_name = f"design_error_count_{_uid8()}"
        inject_df_into_globals(report_df, name=df_name, target_globals=target_globals)

    return report_df, df_name, csv_path

def run_design_error_count_report_from_table(
    cfg: SqlServerConfig,
    *,
    spec: TableLoadSpec,
    out_dir: str | Path = ".",
    export_csv: bool = True,
    inject_globals: bool = False,
    target_globals: Optional[dict] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Optional[str], Optional[Path]]:
    """
    Run the Design Error Count report from a table specification.
    
    Returns:
        Tuple of (report_df, df_name, csv_path)
    """
    src_df = load_spec_df(cfg, spec, verbose=verbose)
    report_df = generate_design_error_count_report(src_df)

    df_name = None
    csv_path = None

    if export_csv:
        report_df, df_name, csv_path = export_df_csv_unique(
            report_df,
            base_name="design_error_count",
            out_dir=out_dir,
            verbose=verbose,
        )

    if inject_globals:
        if df_name is None:
            df_name = f"design_error_count_{_uid8()}"
        inject_df_into_globals(report_df, name=df_name, target_globals=target_globals)

    return report_df, df_name, csv_path

# -------------------------------------------------------------------------------
# OPTIONAL: FILTER HELPERS
# -------------------------------------------------------------------------------

def make_month_filter_sql(base_sql: str, *, month: str) -> Tuple[str, Sequence[Any]]:
    """Add a filter for a specific fiscal release month."""
    sql = base_sql.strip().rstrip(";") + " WHERE Fiscal_Release_Month = %s"
    return sql, (month,)

def make_program_filter_sql(base_sql: str, *, program: str) -> Tuple[str, Sequence[Any]]:
    """Add a filter for a specific program name."""
    sql = base_sql.strip().rstrip(";") + " WHERE PRGRM_NAME = %s"
    return sql, (program,)

def make_date_range_filter_sql(
    base_sql: str, 
    *, 
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[str, Sequence[Any]]:
    """Add a filter for a date range on ACTL_PCKG_RLS_DATE."""
    params = []
    conditions = []
    
    if start_date:
        conditions.append("ACTL_PCKG_RLS_DATE >= %s")
        params.append(start_date)
    
    if end_date:
        conditions.append("ACTL_PCKG_RLS_DATE <= %s")
        params.append(end_date)
    
    if conditions:
        sql = base_sql.strip().rstrip(";") + " WHERE " + " AND ".join(conditions)
        return sql, tuple(params)
    
    return base_sql, tuple()
