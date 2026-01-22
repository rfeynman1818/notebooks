# ===============================================================================
# FILE: reports_on_time.py
# ===============================================================================

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any
import re, uuid
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------------
# IMPORT DB UTILITIES (works whether you named it db_catalog.py or catalog_db.py)
# -------------------------------------------------------------------------------

try:
    from db_catalog import SqlServerConfig, connect, load_table_df, load_table_df_chunked
except Exception:
    from catalog_db import SqlServerConfig, connect, load_table_df, load_table_df_chunked

# -------------------------------------------------------------------------------
# SQL CONSTANTS (edit/tweak these per report / future prompts)
# -------------------------------------------------------------------------------

DEFAULT_SOURCE_SQL = """
SELECT
    Final_Planned_Unplanned,
    ACTL_PCKG_RLS_DATE,
    FORECAST_NEED_DATE,
    PMM_Program_Name,
    FM_REPORTING_MONTH
FROM dbo.FISCAL_MONTH_CT_REPORTING
"""

# -------------------------------------------------------------------------------
# INTERNAL HELPERS (same style / philosophy as db_catalog.py)
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

def _format_pct(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: "" if pd.isna(x) else f"{x:0.2f}%")

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
# GENERIC QUERY RUNNERS (reusable for future SQL prompts)
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
# REPORT: ON-TIME TO FORECAST
# -------------------------------------------------------------------------------

def generate_on_time_report(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Final_Planned_Unplanned",
        "ACTL_PCKG_RLS_DATE",
        "FORECAST_NEED_DATE",
        "PMM_Program_Name",
        "FM_REPORTING_MONTH",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    valid_mask = (
        (df["Final_Planned_Unplanned"] == "Discrete")
        & df["ACTL_PCKG_RLS_DATE"].notna()
        & df["FORECAST_NEED_DATE"].notna()
    )

    on_time_mask = valid_mask & (df["ACTL_PCKG_RLS_DATE"] <= df["FORECAST_NEED_DATE"])

    agg = (
        df.assign(
            total_cnt=valid_mask.astype(int),
            ontime_cnt=on_time_mask.astype(int),
        )
        .groupby(["PMM_Program_Name", "FM_REPORTING_MONTH"], as_index=False)
        .agg(
            total_cnt=("total_cnt", "sum"),
            ontime_cnt=("ontime_cnt", "sum"),
        )
    )

    agg["% On-Time to Forecast"] = np.where(
        agg["total_cnt"] == 0,
        np.nan,
        (agg["ontime_cnt"] / agg["total_cnt"]) * 100.0,
    )

    agg["% On-Time to Forecast (formatted)"] = _format_pct(agg["% On-Time to Forecast"])

    report = agg[
        [
            "PMM_Program_Name",
            "FM_REPORTING_MONTH",
            "% On-Time to Forecast",
            "% On-Time to Forecast (formatted)",
        ]
    ]

    return report

# -------------------------------------------------------------------------------
# REPORT RUNNERS (DB -> DF -> REPORT -> optional CSV + optional globals injection)
# -------------------------------------------------------------------------------

def run_on_time_report_from_query(
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
    src_df = run_query_df(cfg, sql=sql, params=params, verbose=verbose)
    report_df = generate_on_time_report(src_df)

    df_name = None
    csv_path = None

    if export_csv:
        report_df, df_name, csv_path = export_df_csv_unique(
            report_df,
            base_name="pmm_on_time_forecast",
            out_dir=out_dir,
            verbose=verbose,
        )

    if inject_globals:
        if df_name is None:
            df_name = f"pmm_on_time_forecast_{_uid8()}"
        inject_df_into_globals(report_df, name=df_name, target_globals=target_globals)

    return report_df, df_name, csv_path

def run_on_time_report_from_table(
    cfg: SqlServerConfig,
    *,
    spec: TableLoadSpec,
    out_dir: str | Path = ".",
    export_csv: bool = True,
    inject_globals: bool = False,
    target_globals: Optional[dict] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Optional[str], Optional[Path]]:
    src_df = load_spec_df(cfg, spec, verbose=verbose)
    report_df = generate_on_time_report(src_df)

    df_name = None
    csv_path = None

    if export_csv:
        report_df, df_name, csv_path = export_df_csv_unique(
            report_df,
            base_name="pmm_on_time_forecast",
            out_dir=out_dir,
            verbose=verbose,
        )

    if inject_globals:
        if df_name is None:
            df_name = f"pmm_on_time_forecast_{_uid8()}"
        inject_df_into_globals(report_df, name=df_name, target_globals=target_globals)

    return report_df, df_name, csv_path

# -------------------------------------------------------------------------------
# OPTIONAL: PRESETS / TEMPLATES YOU CAN COPY & TWEAK FOR NEW QUERIES
# -------------------------------------------------------------------------------

def make_month_filter_sql(base_sql: str, *, month: str) -> Tuple[str, Sequence[Any]]:
    sql = base_sql.strip().rstrip(";") + " WHERE FM_REPORTING_MONTH = %s"
    return sql, (month,)

def make_program_filter_sql(base_sql: str, *, program: str) -> Tuple[str, Sequence[Any]]:
    sql = base_sql.strip().rstrip(";") + " WHERE PMM_Program_Name = %s"
    return sql, (program,)
