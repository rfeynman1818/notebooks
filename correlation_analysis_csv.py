# ===============================================================================
# CSV-BASED CORRELATION ANALYSIS MODULE
# ===============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ===============================================================================
# CSV LOADING AND VALIDATION
# ===============================================================================

def validate_csv_columns(df: pd.DataFrame, required_cols: List[str], csv_name: str) -> bool:
    """
    Validate that CSV has required columns.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        csv_name: Name of CSV file for error messages
        
    Returns:
        True if valid, raises ValueError if not
    """
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        raise ValueError(
            f"❌ {csv_name} is missing required columns: {missing}\n"
            f"   Required: {required_cols}\n"
            f"   Found: {list(df.columns)}"
        )
    
    return True

def load_merged_metrics_csv(
    csv_path: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load a pre-merged metrics CSV file.
    
    Expected columns:
    - PRGRM_NAME
    - PMM_Program_ID
    - FM_REPORTING_MONTH
    - SPI, CEI, BEI (existing metrics)
    - % On-Time to Forecast
    - % Preventable Revisions
    - design_error_count
    - planned_ct_releases
    - ct_releases
    
    Args:
        csv_path: Path to CSV file
        verbose: Print validation info
        
    Returns:
        Validated DataFrame
    """
    if verbose:
        print(f"📂 Loading merged metrics from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    required_cols = [
        'PRGRM_NAME',
        'PMM_Program_ID', 
        'FM_REPORTING_MONTH',
        'SPI',
        'CEI',
        'BEI',
        '% On-Time to Forecast',
        '% Preventable Revisions',
        'design_error_count',
        'planned_ct_releases',
        'ct_releases'
    ]
    
    validate_csv_columns(df, required_cols, Path(csv_path).name)
    
    if verbose:
        print(f"✅ Loaded {len(df)} records")
        print(f"   Programs: {df['PRGRM_NAME'].nunique()}")
        print(f"   Date range: {df['FM_REPORTING_MONTH'].min()} to {df['FM_REPORTING_MONTH'].max()}")
        
        # Show data completeness
        print(f"\n📊 Data Completeness:")
        for col in required_cols:
            if col not in ['PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH']:
                non_null = df[col].notna().sum()
                pct = (non_null / len(df)) * 100
                print(f"   {col}: {non_null}/{len(df)} ({pct:.1f}%)")
    
    return df

def load_separate_csvs(
    existing_metrics_csv: str,
    ontime_csv: str,
    preventable_csv: str,
    design_error_csv: str,
    planned_ct_csv: str,
    ct_releases_csv: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load separate CSV files and merge them.
    
    Args:
        existing_metrics_csv: CSV with SPI, CEI, BEI
        ontime_csv: On-Time to Forecast report CSV
        preventable_csv: Preventable Revisions report CSV
        design_error_csv: Design Error Count report CSV
        planned_ct_csv: Planned CT Releases report CSV
        ct_releases_csv: CT Releases report CSV
        verbose: Print progress
        
    Returns:
        Merged DataFrame
    """
    if verbose:
        print("="*80)
        print("LOADING SEPARATE CSV FILES")
        print("="*80)
    
    # Load existing metrics
    if verbose:
        print("\n1️⃣  Loading existing metrics (SPI, CEI, BEI)...")
    existing_df = pd.read_csv(existing_metrics_csv)
    validate_csv_columns(existing_df, ['PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH', 'SPI', 'CEI', 'BEI'], 
                        Path(existing_metrics_csv).name)
    if verbose:
        print(f"   ✅ {len(existing_df)} records")
    
    # Load on-time
    if verbose:
        print("\n2️⃣  Loading On-Time to Forecast...")
    ontime_df = pd.read_csv(ontime_csv)
    # Handle potential column name variations
    if 'PMM_Program_Name' in ontime_df.columns:
        ontime_df = ontime_df.rename(columns={'PMM_Program_Name': 'PRGRM_NAME'})
    validate_csv_columns(ontime_df, ['PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH', '% On-Time to Forecast'],
                        Path(ontime_csv).name)
    if verbose:
        print(f"   ✅ {len(ontime_df)} records")
    
    # Load preventable revisions
    if verbose:
        print("\n3️⃣  Loading Preventable Revisions...")
    preventable_df = pd.read_csv(preventable_csv)
    validate_csv_columns(preventable_df, ['PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH', '% Preventable Revisions'],
                        Path(preventable_csv).name)
    if verbose:
        print(f"   ✅ {len(preventable_df)} records")
    
    # Load design error count
    if verbose:
        print("\n4️⃣  Loading Design Error Count...")
    design_df = pd.read_csv(design_error_csv)
    validate_csv_columns(design_df, ['PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH', 'design_error_count'],
                        Path(design_error_csv).name)
    if verbose:
        print(f"   ✅ {len(design_df)} records")
    
    # Load planned CT releases
    if verbose:
        print("\n5️⃣  Loading Planned CT Releases...")
    planned_df = pd.read_csv(planned_ct_csv)
    validate_csv_columns(planned_df, ['PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH', 'planned_ct_releases'],
                        Path(planned_ct_csv).name)
    if verbose:
        print(f"   ✅ {len(planned_df)} records")
    
    # Load CT releases
    if verbose:
        print("\n6️⃣  Loading CT Releases...")
    ct_df = pd.read_csv(ct_releases_csv)
    validate_csv_columns(ct_df, ['PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH', 'ct_releases'],
                        Path(ct_releases_csv).name)
    if verbose:
        print(f"   ✅ {len(ct_df)} records")
    
    # Merge all dataframes
    if verbose:
        print("\n🔗 Merging all CSVs...")
    
    merge_keys = ['PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH']
    
    merged = existing_df.copy()
    merged = merged.merge(ontime_df[merge_keys + ['% On-Time to Forecast']], on=merge_keys, how='left')
    merged = merged.merge(preventable_df[merge_keys + ['% Preventable Revisions']], on=merge_keys, how='left')
    merged = merged.merge(design_df[merge_keys + ['design_error_count']], on=merge_keys, how='left')
    merged = merged.merge(planned_df[merge_keys + ['planned_ct_releases']], on=merge_keys, how='left')
    merged = merged.merge(ct_df[merge_keys + ['ct_releases']], on=merge_keys, how='left')
    
    if verbose:
        print(f"   ✅ Merged dataset: {len(merged)} records")
        print(f"   Programs: {merged['PRGRM_NAME'].nunique()}")
        print(f"   Date range: {merged['FM_REPORTING_MONTH'].min()} to {merged['FM_REPORTING_MONTH'].max()}")
    
    return merged

def add_derived_metrics_csv(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add derived metrics to CSV data.
    
    Same as add_derived_metrics but for CSV input.
    """
    if verbose:
        print("\n" + "="*80)
        print("CALCULATING DERIVED METRICS")
        print("="*80)
    
    df = df.copy()
    
    # % Planned
    df['pct_planned'] = np.where(
        df['ct_releases'] > 0,
        (df['planned_ct_releases'] / df['ct_releases']) * 100,
        np.nan
    )
    
    # Unplanned CT count
    df['unplanned_ct_releases'] = df['ct_releases'] - df['planned_ct_releases'].fillna(0)
    
    # Design Error Rate
    df['design_error_rate'] = np.where(
        df['ct_releases'] > 0,
        (df['design_error_count'] / df['ct_releases']) * 100,
        np.nan
    )
    
    if verbose:
        print("✅ Added derived metrics:")
        print("   - pct_planned: % of releases that were planned")
        print("   - unplanned_ct_releases: Count of unplanned releases")
        print("   - design_error_rate: Design errors as % of total releases")
    
    return df

# ===============================================================================
# CSV EXPORT UTILITIES
# ===============================================================================

def export_reports_to_csv(
    cfg,
    rsn_mapping: Dict[str, str],
    output_dir: str = ".",
    verbose: bool = True
) -> Dict[str, str]:
    """
    Export all reports to CSV files for later analysis.
    
    This is a helper function to create the CSV files from database
    that can then be uploaded for CSV-based analysis.
    
    Args:
        cfg: Database configuration
        rsn_mapping: RSN mapping for preventable revisions
        output_dir: Directory to save CSV files
        verbose: Print progress
        
    Returns:
        Dictionary mapping report names to file paths
    """
    from reports_on_time import run_on_time_report_from_query
    from reports_preventable_revisions_mapped import run_preventable_revisions_report_from_query
    from reports_design_error_count import run_design_error_count_report_from_query
    from reports_planned_ct_releases import run_planned_ct_releases_report_from_query
    from reports_ct_releases import run_ct_releases_report_from_query
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = {}
    
    if verbose:
        print("="*80)
        print("EXPORTING REPORTS TO CSV")
        print("="*80)
    
    # On-Time
    if verbose:
        print("\n1️⃣  Exporting On-Time to Forecast...")
    ontime_df, _, _ = run_on_time_report_from_query(cfg, verbose=False, export_csv=False)
    ontime_path = output_dir / "ontime_forecast.csv"
    ontime_df.to_csv(ontime_path, index=False)
    files['ontime'] = str(ontime_path)
    if verbose:
        print(f"   ✅ Saved: {ontime_path}")
    
    # Preventable Revisions
    if verbose:
        print("\n2️⃣  Exporting Preventable Revisions...")
    preventable_df, _, _ = run_preventable_revisions_report_from_query(
        cfg, rsn_mapping=rsn_mapping, verbose=False, export_csv=False
    )
    preventable_path = output_dir / "preventable_revisions.csv"
    preventable_df.to_csv(preventable_path, index=False)
    files['preventable'] = str(preventable_path)
    if verbose:
        print(f"   ✅ Saved: {preventable_path}")
    
    # Design Error Count
    if verbose:
        print("\n3️⃣  Exporting Design Error Count...")
    design_df, _, _ = run_design_error_count_report_from_query(cfg, verbose=False, export_csv=False)
    design_path = output_dir / "design_error_count.csv"
    design_df.to_csv(design_path, index=False)
    files['design_error'] = str(design_path)
    if verbose:
        print(f"   ✅ Saved: {design_path}")
    
    # Planned CT Releases
    if verbose:
        print("\n4️⃣  Exporting Planned CT Releases...")
    planned_df, _, _ = run_planned_ct_releases_report_from_query(cfg, verbose=False, export_csv=False)
    planned_path = output_dir / "planned_ct_releases.csv"
    planned_df.to_csv(planned_path, index=False)
    files['planned_ct'] = str(planned_path)
    if verbose:
        print(f"   ✅ Saved: {planned_path}")
    
    # CT Releases
    if verbose:
        print("\n5️⃣  Exporting CT Releases...")
    ct_df, _, _ = run_ct_releases_report_from_query(cfg, verbose=False, export_csv=False)
    ct_path = output_dir / "ct_releases.csv"
    ct_df.to_csv(ct_path, index=False)
    files['ct_releases'] = str(ct_path)
    if verbose:
        print(f"   ✅ Saved: {ct_path}")
    
    if verbose:
        print("\n" + "="*80)
        print("✅ All reports exported!")
        print("="*80)
        print("\nYou can now use these CSV files for correlation analysis without")
        print("needing to re-run the database queries.")
    
    return files

def create_csv_template(output_dir: str = "."):
    """
    Create template CSV files showing the expected format.
    
    Args:
        output_dir: Directory to save templates
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Template for existing metrics
    existing_template = pd.DataFrame({
        'PRGRM_NAME': ['Program A', 'Program A', 'Program B'],
        'PMM_Program_ID': ['P001', 'P001', 'P002'],
        'FM_REPORTING_MONTH': ['2024-01', '2024-02', '2024-01'],
        'SPI': [1.05, 0.98, 1.12],
        'CEI': [0.95, 1.02, 0.88],
        'BEI': [1.00, 1.05, 0.95]
    })
    existing_template.to_csv(output_dir / "TEMPLATE_existing_metrics.csv", index=False)
    
    # Template for merged metrics
    merged_template = pd.DataFrame({
        'PRGRM_NAME': ['Program A', 'Program A'],
        'PMM_Program_ID': ['P001', 'P001'],
        'FM_REPORTING_MONTH': ['2024-01', '2024-02'],
        'SPI': [1.05, 0.98],
        'CEI': [0.95, 1.02],
        'BEI': [1.00, 1.05],
        '% On-Time to Forecast': [85.5, 78.3],
        '% Preventable Revisions': [32.1, 28.9],
        'design_error_count': [5, 7],
        'planned_ct_releases': [45, 52],
        'ct_releases': [50, 60]
    })
    merged_template.to_csv(output_dir / "TEMPLATE_merged_metrics.csv", index=False)
    
    print(f"✅ Template CSV files created in: {output_dir}")
    print(f"   - TEMPLATE_existing_metrics.csv")
    print(f"   - TEMPLATE_merged_metrics.csv")
    print("\nUse these as examples for the correct format.")

print("✅ CSV-based correlation analysis module loaded!")
