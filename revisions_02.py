# ===============================================================================
# FILE: reports_preventable_revisions_mapped.py
# VERSION: Uses custom mapping for RSN_FOR_CHG_CODE_MJR values
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
# MAPPING UTILITIES
# -------------------------------------------------------------------------------

def load_mapping_from_csv(csv_path: str | Path) -> Dict[str, str]:
    """
    Load mapping from a CSV file with columns: rsn_value, category
    
    Example CSV:
    rsn_value,category
    "01 - Design Error",Preventable
    "02 - Manufacturing Error",Non-Preventable
    "03 - Field Issue",Exclude
    """
    df = pd.read_csv(csv_path)
    mapping = dict(zip(df['rsn_value'], df['category']))
    return mapping

def create_default_mapping() -> Dict[str, str]:
    """
    Default mapping - REPLACE THIS WITH YOUR ACTUAL VALUES
    
    Returns a dictionary mapping RSN_FOR_CHG_CODE_MJR values to:
    - 'Preventable'
    - 'Non-Preventable'
    - 'Exclude'
    """
    return {
        # Example - replace with your actual 24 values
        "Value1": "Preventable",
        "Value2": "Non-Preventable",
        "Value3": "Exclude",
        # Add all 24 mappings here...
    }

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
# REPORT: PREVENTABLE REVISIONS (WITH MAPPING)
# -------------------------------------------------------------------------------

def generate_preventable_revisions_report(
    df: pd.DataFrame,
    *,
    rsn_mapping: Dict[str, str],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calculate Preventable Revisions percentage using a custom mapping.
    
    % Preventable = (Count mapped to 'Preventable') / 
                    (Count mapped to 'Preventable' + Count mapped to 'Non-Preventable')
    
    Records mapped to 'Exclude' are excluded from calculation.
    
    Args:
        df: Source dataframe
        rsn_mapping: Dictionary mapping RSN_FOR_CHG_CODE_MJR values to 
                     'Preventable', 'Non-Preventable', or 'Exclude'
        verbose: Print diagnostic information
    
    Grouped by PRGRM_NAME, PMM_Program_ID, and Fiscal_Release_Month
    """
    required = [
        "CHG_ACTVTY_NUM",
        "RSN_FOR_CHG_CODE_MJR",
        "PRGRM_NAME",
        "PMM_Program_ID",
        "Fiscal_Release_Month",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create a copy and map values
    df_mapped = df.copy()
    df_mapped["_category"] = df_mapped["RSN_FOR_CHG_CODE_MJR"].map(rsn_mapping)
    
    if verbose:
        print("\n📊 Preventable Revisions Mapping Summary:")
        print(f"   Total records in dataset: {len(df_mapped)}")
        
        # Show mapping results
        category_counts = df_mapped["_category"].value_counts(dropna=False)
        print(f"\n   Mapping results:")
        for category, count in category_counts.items():
            if pd.isna(category):
                print(f"     ⚠️  Unmapped (NULL): {count} records")
            else:
                print(f"     - {category}: {count} records")
        
        # Show unmapped values
        unmapped = df_mapped[df_mapped["_category"].isna()]["RSN_FOR_CHG_CODE_MJR"].unique()
        if len(unmapped) > 0:
            print(f"\n   ⚠️  Warning: {len(unmapped)} unique unmapped values found:")
            for val in unmapped[:10]:  # Show first 10
                print(f"      - '{val}'")
            if len(unmapped) > 10:
                print(f"      ... and {len(unmapped) - 10} more")
            print("\n   💡 Add these values to your mapping dictionary!")
        
        # Show values mapped to each category
        print(f"\n   Values mapped to each category:")
        for category in ['Preventable', 'Non-Preventable', 'Exclude']:
            values = [k for k, v in rsn_mapping.items() if v == category]
            print(f"     {category}: {len(values)} values")
    
    # Filter: Keep only Preventable and Non-Preventable (exclude 'Exclude' and unmapped)
    valid_mask = df_mapped["_category"].isin(["Preventable", "Non-Preventable"])
    
    # Identify preventable records
    preventable_mask = valid_mask & (df_mapped["_category"] == "Preventable")
    
    if verbose:
        total_valid = valid_mask.sum()
        total_preventable = preventable_mask.sum()
        total_non_preventable = valid_mask.sum() - total_preventable
        
        print(f"\n   After filtering:")
        print(f"     Valid records (Preventable + Non-Preventable): {total_valid}")
        print(f"     Preventable records: {total_preventable}")
        print(f"     Non-Preventable records: {total_non_preventable}")
        
        if total_valid > 0:
            overall_pct = (total_preventable / total_valid) * 100
            print(f"     Overall % Preventable: {overall_pct:.2f}%")
        
        if total_valid == 0:
            print("\n   ⚠️  WARNING: No valid records after mapping!")
            print("   Check that your mapping includes both 'Preventable' and 'Non-Preventable' values.")

    # Aggregate by program and fiscal month
    agg = (
        df_mapped.assign(
            total_cnt=valid_mask.astype(int),
            preventable_cnt=preventable_mask.astype(int),
        )
        .groupby(["PRGRM_NAME", "PMM_Program_ID", "Fiscal_Release_Month"], as_index=False)
        .agg(
            total_cnt=("total_cnt", "sum"),
            preventable_cnt=("preventable_cnt", "sum"),
        )
    )

    # Calculate percentage
    agg["% Preventable Revisions"] = np.where(
        agg["total_cnt"] == 0,
        np.nan,
        (agg["preventable_cnt"] / agg["total_cnt"]) * 100.0,
    )

    agg["% Preventable Revisions (formatted)"] = _format_pct(agg["% Preventable Revisions"])

    # Select and order columns for final report
    report = agg[
        [
            "PRGRM_NAME",
            "PMM_Program_ID",
            "Fiscal_Release_Month",
            "total_cnt",
            "preventable_cnt",
            "% Preventable Revisions",
            "% Preventable Revisions (formatted)",
        ]
    ]

    return report

# -------------------------------------------------------------------------------
# REPORT RUNNERS
# -------------------------------------------------------------------------------

def run_preventable_revisions_report_from_query(
    cfg: SqlServerConfig,
    *,
    rsn_mapping: Dict[str, str],
    sql: str = DEFAULT_SOURCE_SQL,
    params: Optional[Sequence[Any]] = None,
    out_dir: str | Path = ".",
    export_csv: bool = True,
    inject_globals: bool = False,
    target_globals: Optional[dict] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Optional[str], Optional[Path]]:
    """
    Run the Preventable Revisions report from a SQL query with custom mapping.
    
    Args:
        rsn_mapping: Dictionary mapping RSN_FOR_CHG_CODE_MJR values to 
                     'Preventable', 'Non-Preventable', or 'Exclude'
    
    Returns:
        Tuple of (report_df, df_name, csv_path)
    """
    src_df = run_query_df(cfg, sql=sql, params=params, verbose=verbose)
    report_df = generate_preventable_revisions_report(
        src_df,
        rsn_mapping=rsn_mapping,
        verbose=verbose,
    )

    df_name = None
    csv_path = None

    if export_csv:
        report_df, df_name, csv_path = export_df_csv_unique(
            report_df,
            base_name="preventable_revisions",
            out_dir=out_dir,
            verbose=verbose,
        )

    if inject_globals:
        if df_name is None:
            df_name = f"preventable_revisions_{_uid8()}"
        inject_df_into_globals(report_df, name=df_name, target_globals=target_globals)

    return report_df, df_name, csv_path

def run_preventable_revisions_report_from_table(
    cfg: SqlServerConfig,
    *,
    rsn_mapping: Dict[str, str],
    spec: TableLoadSpec,
    out_dir: str | Path = ".",
    export_csv: bool = True,
    inject_globals: bool = False,
    target_globals: Optional[dict] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Optional[str], Optional[Path]]:
    """
    Run the Preventable Revisions report from a table specification.
    
    Returns:
        Tuple of (report_df, df_name, csv_path)
    """
    src_df = load_spec_df(cfg, spec, verbose=verbose)
    report_df = generate_preventable_revisions_report(
        src_df,
        rsn_mapping=rsn_mapping,
        verbose=verbose,
    )

    df_name = None
    csv_path = None

    if export_csv:
        report_df, df_name, csv_path = export_df_csv_unique(
            report_df,
            base_name="preventable_revisions",
            out_dir=out_dir,
            verbose=verbose,
        )

    if inject_globals:
        if df_name is None:
            df_name = f"preventable_revisions_{_uid8()}"
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
