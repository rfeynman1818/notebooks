import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# ===============================================================================
# DATA CLEANING UTILITIES
# ===============================================================================

def clean_percentage_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean percentage columns and convert to decimal scale (0.0 - 1.0).
    
    Converts strings like:
    - '100.00%' → 1.00
    - '85.5%' → 0.855
    - '10%' → 0.10
    - 100 → 1.00 (if already numeric but >1)
    
    Args:
        df: DataFrame with potential percentage string columns
        verbose: Print cleaning info
        
    Returns:
        DataFrame with cleaned decimal columns (0-1 scale)
    """
    df = df.copy()
    
    # Columns that might be percentages
    potential_pct_cols = [col for col in df.columns if '%' in str(col) or 'pct_' in str(col).lower()]
    
    cleaned_cols = []
    
    for col in potential_pct_cols:
        if col in df.columns:
            # Check if column contains string percentages or large numbers
            sample = df[col].dropna().head(5)
            if len(sample) > 0:
                # Check if it's a string with %
                has_percent_symbol = any('%' in str(val) for val in sample)
                # Check if values are > 1 (suggesting 0-100 scale instead of 0-1)
                has_large_values = any(pd.notna(val) and isinstance(val, (int, float)) and val > 1 for val in sample)
                
                if has_percent_symbol:
                    # Clean string percentages: "100%" -> 1.0
                    df[col] = df[col].astype(str).str.replace('%', '').str.strip()
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100  # Divide by 100!
                    cleaned_cols.append((col, 'string %'))
                elif has_large_values:
                    # Convert 0-100 scale to 0-1 scale
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100  # Divide by 100!
                    cleaned_cols.append((col, 'numeric 0-100'))
    
    if verbose and cleaned_cols:
        print(f"\n🧹 Cleaned {len(cleaned_cols)} percentage columns to decimal scale (0-1):")
        for col, source_type in cleaned_cols:
            print(f"   - {col} (from {source_type})")
    
    return df

def clean_numeric_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean all numeric columns by removing commas and converting to proper numeric types.
    
    Handles:
    - '1,000' → 1000
    - '1,234.56' → 1234.56
    - Strings with spaces
    
    Does NOT divide by 100 (that's only for percentage columns)
    
    Args:
        df: DataFrame to clean
        verbose: Print info
        
    Returns:
        DataFrame with cleaned columns
    """
    df = df.copy()
    
    # Try to convert all object columns to numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            # Skip known text columns
            if col in ['PMM_Program_Name', 'Fm_Reporting_Month', 'Program Name', 'Project Status Date']:
                continue
            
            # Skip percentage columns (handled by clean_percentage_columns)
            if '%' in str(col) or 'pct_' in str(col).lower():
                continue
            
            # Try cleaning and converting
            try:
                # Remove commas and extra spaces
                cleaned = df[col].astype(str).str.replace(',', '').str.strip()
                # Try converting to numeric
                numeric = pd.to_numeric(cleaned, errors='coerce')
                
                # Only replace if we successfully converted some values
                if numeric.notna().sum() > 0:
                    df[col] = numeric
                    if verbose:
                        print(f"   ✓ Converted {col} to numeric")
            except:
                pass  # Leave as-is if conversion fails
    
    return df

# ===============================================================================
# PROGRAM NAME MAPPING
# ===============================================================================

def load_program_mapping(
    mapping_source: Union[str, Dict[str, str], pd.DataFrame],
    verbose: bool = True
) -> Dict[str, str]:
    """Load program name mapping from various sources."""
    if verbose:
        print("="*80)
        print("LOADING PROGRAM NAME MAPPING")
        print("="*80)
    
    if isinstance(mapping_source, dict):
        mapping = mapping_source
    elif isinstance(mapping_source, pd.DataFrame):
        if 'CEI_BEI_Program_Name' in mapping_source.columns and 'Metrics_Program_Name' in mapping_source.columns:
            mapping = dict(zip(
                mapping_source['CEI_BEI_Program_Name'],
                mapping_source['Metrics_Program_Name']
            ))
        else:
            mapping = dict(zip(mapping_source.iloc[:, 0], mapping_source.iloc[:, 1]))
    elif isinstance(mapping_source, str):
        df = pd.read_csv(mapping_source)
        if 'CEI_BEI_Program_Name' in df.columns and 'Metrics_Program_Name' in df.columns:
            mapping = dict(zip(df['CEI_BEI_Program_Name'], df['Metrics_Program_Name']))
        else:
            mapping = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    else:
        raise ValueError("mapping_source must be dict, DataFrame, or path to CSV")
    
    if verbose:
        print(f"✅ Loaded mapping for {len(mapping)} programs")
        print("\nSample mappings:")
        for i, (cei_name, metrics_name) in enumerate(list(mapping.items())[:5]):
            print(f"   {cei_name} → {metrics_name}")
        if len(mapping) > 5:
            print(f"   ... and {len(mapping) - 5} more")
    
    return mapping

def apply_program_mapping(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    source_col: str = 'Program Name',
    target_col: str = 'PMM_Program_Name',
    verbose: bool = True
) -> pd.DataFrame:
    """Apply program name mapping to a dataframe."""
    df = df.copy()
    df[target_col] = df[source_col].map(mapping)
    
    unmapped = df[df[target_col].isna()][source_col].unique()
    if verbose and len(unmapped) > 0:
        print(f"\n⚠️  Warning: {len(unmapped)} programs not in mapping:")
        for prog in unmapped[:10]:
            print(f"   - {prog}")
        if len(unmapped) > 10:
            print(f"   ... and {len(unmapped) - 10} more")
    
    return df

# ===============================================================================
# CEI DATA LOADING
# ===============================================================================

def load_cei_data(
    csv_path: str,
    program_mapping: Optional[Union[str, Dict[str, str], pd.DataFrame]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Load CEI data from CSV with automatic data cleaning."""
    if verbose:
        print("\n" + "="*80)
        print("LOADING CEI DATA")
        print("="*80)
        print(f"📂 Loading from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Clean numeric columns first
    df = clean_numeric_columns(df, verbose=False)
    
    # Check for required columns
    required = ['Program Name', 'Project Status Date', 'Hit (CEI)', 'Miss (CEI)', 'CEI Calc']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"❌ CEI CSV missing columns: {missing}\nFound: {list(df.columns)}")
    
    # Standardize column names
    df = df.rename(columns={
        'Project Status Date': 'Fm_Reporting_Month',
        'Hit (CEI)': 'cei_hit',
        'Miss (CEI)': 'cei_miss',
        'CEI Calc': 'CEI'
    })
    
    # Clean CEI column to decimal scale (0-1)
    df = clean_percentage_columns(df, verbose=verbose)
    
    # Apply program name mapping if provided
    if program_mapping is not None:
        if not isinstance(program_mapping, dict):
            program_mapping = load_program_mapping(program_mapping, verbose=verbose)
        df = apply_program_mapping(df, program_mapping, 
                                   source_col='Program Name',
                                   target_col='PMM_Program_Name',
                                   verbose=verbose)
    else:
        df = df.rename(columns={'Program Name': 'PMM_Program_Name'})
    
    if verbose:
        print(f"\n✅ Loaded {len(df)} CEI records")
        print(f"   Programs: {df['PMM_Program_Name'].nunique()}")
        print(f"   Date range: {df['Fm_Reporting_Month'].min()} to {df['Fm_Reporting_Month'].max()}")
        if 'CEI' in df.columns:
            cei_sample = df['CEI'].dropna().head(3)
            print(f"   CEI sample values (decimal): {cei_sample.tolist()}")
    
    return df

# ===============================================================================
# BEI DATA LOADING
# ===============================================================================

def load_bei_data(
    csv_path: str,
    program_mapping: Optional[Union[str, Dict[str, str], pd.DataFrame]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Load BEI data from CSV with automatic data cleaning."""
    if verbose:
        print("\n" + "="*80)
        print("LOADING BEI DATA")
        print("="*80)
        print(f"📂 Loading from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Clean numeric columns first
    df = clean_numeric_columns(df, verbose=False)
    
    # Check for required columns
    required = ['Program Name', 'Project Status Date', 'BEI Numerator', 'BEI Denominator', 'BEI (Calc)']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"❌ BEI CSV missing columns: {missing}\nFound: {list(df.columns)}")
    
    # Standardize column names
    df = df.rename(columns={
        'Project Status Date': 'Fm_Reporting_Month',
        'BEI Numerator': 'bei_numerator',
        'BEI Denominator': 'bei_denominator',
        'BEI (Calc)': 'BEI'
    })
    
    # Note: BEI is typically already a ratio (not percentage), so no scaling needed
    # But if it's > 1 for most values, it might need scaling
    # We'll leave BEI as-is unless it's explicitly formatted as percentage
    
    # Apply program name mapping if provided
    if program_mapping is not None:
        if not isinstance(program_mapping, dict):
            program_mapping = load_program_mapping(program_mapping, verbose=verbose)
        df = apply_program_mapping(df, program_mapping,
                                   source_col='Program Name',
                                   target_col='PMM_Program_Name',
                                   verbose=verbose)
    else:
        df = df.rename(columns={'Program Name': 'PMM_Program_Name'})
    
    if verbose:
        print(f"\n✅ Loaded {len(df)} BEI records")
        print(f"   Programs: {df['PMM_Program_Name'].nunique()}")
        print(f"   Date range: {df['Fm_Reporting_Month'].min()} to {df['Fm_Reporting_Month'].max()}")
        if 'BEI' in df.columns:
            bei_sample = df['BEI'].dropna().head(3)
            print(f"   BEI sample values: {bei_sample.tolist()}")
    
    return df

# ===============================================================================
# ENGINEERING METRICS LOADING
# ===============================================================================

def load_engineering_metrics(
    ontime_csv: str,
    preventable_csv: str,
    design_error_csv: str,
    planned_ct_csv: str,
    ct_releases_csv: str,
    verbose: bool = True
) -> pd.DataFrame:
    """Load engineering metrics with automatic percentage cleaning to decimal scale."""
    if verbose:
        print("\n" + "="*80)
        print("LOADING ENGINEERING METRICS")
        print("="*80)
    
    # Load On-Time
    if verbose:
        print("\n1️⃣  Loading On-Time to Forecast...")
    ontime_df = pd.read_csv(ontime_csv)
    ontime_df = clean_percentage_columns(ontime_df, verbose=verbose)  # Convert to 0-1 scale!
    ontime_df = ontime_df.rename(columns={
        'PMM Program Name': 'PMM_Program_Name',
        'Fm Reporting Month': 'Fm_Reporting_Month'
    })
    if verbose:
        print(f"   ✅ {len(ontime_df)} records")
        if '% On-Time to Forecast' in ontime_df.columns:
            sample = ontime_df['% On-Time to Forecast'].dropna().head(3)
            print(f"   Sample values (decimal): {sample.tolist()}")
    
    # Load Preventable Revisions
    if verbose:
        print("\n2️⃣  Loading Preventable Revisions...")
    preventable_df = pd.read_csv(preventable_csv)
    preventable_df = clean_percentage_columns(preventable_df, verbose=verbose)  # Convert to 0-1 scale!
    preventable_df = preventable_df.rename(columns={
        'PMM Program Name': 'PMM_Program_Name',
        'Fm Reporting Month': 'Fm_Reporting_Month',
        'Preventable % of Total Revisions': 'pct_preventable_revisions'
    })
    if verbose:
        print(f"   ✅ {len(preventable_df)} records")
        if 'pct_preventable_revisions' in preventable_df.columns:
            sample = preventable_df['pct_preventable_revisions'].dropna().head(3)
            print(f"   Sample values (decimal): {sample.tolist()}")
    
    # Load Design Error Count
    if verbose:
        print("\n3️⃣  Loading Design Error Count...")
    design_df = pd.read_csv(design_error_csv)
    design_df = clean_numeric_columns(design_df, verbose=False)
    design_df = design_df.rename(columns={
        'PMM Program Name': 'PMM_Program_Name',
        'Fm Reporting Month': 'Fm_Reporting_Month',
        'RFC 2 Released CT Count': 'design_error_count'
    })
    if verbose:
        print(f"   ✅ {len(design_df)} records")
    
    # Load Planned CT Releases
    if verbose:
        print("\n4️⃣  Loading Planned CT Releases...")
    planned_df = pd.read_csv(planned_ct_csv)
    planned_df = clean_numeric_columns(planned_df, verbose=False)
    planned_df = planned_df.rename(columns={
        'PMM Program Name': 'PMM_Program_Name',
        'Fm Reporting Month': 'Fm_Reporting_Month',
        'Planned Released CTs': 'planned_ct_releases'
    })
    if verbose:
        print(f"   ✅ {len(planned_df)} records")
    
    # Load CT Releases
    if verbose:
        print("\n5️⃣  Loading CT Releases...")
    ct_df = pd.read_csv(ct_releases_csv)
    ct_df = clean_numeric_columns(ct_df, verbose=False)
    ct_df = ct_df.rename(columns={
        'PMM Program Name': 'PMM_Program_Name',
        'Fm Reporting Month': 'Fm_Reporting_Month',
        'Released CT Count': 'ct_releases'
    })
    if verbose:
        print(f"   ✅ {len(ct_df)} records")
    
    # Merge all engineering metrics
    if verbose:
        print("\n🔗 Merging engineering metrics...")
    
    merge_keys = ['PMM_Program_Name', 'Fm_Reporting_Month']
    
    metrics = ontime_df.copy()
    metrics = metrics.merge(preventable_df[merge_keys + ['pct_preventable_revisions']], on=merge_keys, how='outer')
    metrics = metrics.merge(design_df[merge_keys + ['design_error_count']], on=merge_keys, how='outer')
    metrics = metrics.merge(planned_df[merge_keys + ['planned_ct_releases']], on=merge_keys, how='outer')
    metrics = metrics.merge(ct_df[merge_keys + ['ct_releases']], on=merge_keys, how='outer')
    
    if verbose:
        print(f"   ✅ Merged {len(metrics)} engineering metric records")
    
    return metrics

# ===============================================================================
# COMPLETE DATA LOADING
# ===============================================================================

def load_all_data(
    cei_csv: str,
    bei_csv: str,
    ontime_csv: str,
    preventable_csv: str,
    design_error_csv: str,
    planned_ct_csv: str,
    ct_releases_csv: str,
    program_mapping: Optional[Union[str, Dict[str, str], pd.DataFrame]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Load CEI, BEI, and all engineering metrics separately, then merge."""
    if verbose:
        print("="*80)
        print("LOADING ALL DATA")
        print("Percentages will be converted to decimal scale (0.0 - 1.0)")
        print("="*80)
    
    # Load CEI data
    cei_df = load_cei_data(cei_csv, program_mapping, verbose)
    
    # Load BEI data
    bei_df = load_bei_data(bei_csv, program_mapping, verbose)
    
    # Load engineering metrics
    metrics_df = load_engineering_metrics(
        ontime_csv, preventable_csv, design_error_csv,
        planned_ct_csv, ct_releases_csv, verbose
    )
    
    # Merge all data
    if verbose:
        print("\n" + "="*80)
        print("MERGING ALL DATA")
        print("="*80)
    
    merge_keys = ['PMM_Program_Name', 'Fm_Reporting_Month']
    
    # Start with CEI
    merged = cei_df.copy()
    
    # Merge BEI
    merged = merged.merge(
        bei_df[merge_keys + ['bei_numerator', 'bei_denominator', 'BEI']],
        on=merge_keys,
        how='outer'
    )
    
    # Merge engineering metrics
    merged = merged.merge(metrics_df, on=merge_keys, how='outer')
    
    # Final cleaning pass for any remaining percentage columns
    merged = clean_percentage_columns(merged, verbose=verbose)
    
    if verbose:
        print(f"\n✅ Merged dataset: {len(merged)} records")
        print(f"   Programs: {merged['PMM_Program_Name'].nunique()}")
        print(f"   Date range: {merged['Fm_Reporting_Month'].min()} to {merged['Fm_Reporting_Month'].max()}")
        print("\n📊 Percentage columns are now on decimal scale (0.0 - 1.0)")
    
    return merged

# ===============================================================================
# DERIVED METRICS (All as decimals 0-1 scale)
# ===============================================================================

def add_derived_metrics(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add derived metrics (as decimals on 0-1 scale).
    
    All percentage metrics will be 0.0 - 1.0:
    - pct_planned: 0.0 = 0%, 1.0 = 100%
    - design_error_rate: 0.0 = 0%, 1.0 = 100%
    - cei_success_rate: 0.0 = 0%, 1.0 = 100%
    """
    if verbose:
        print("\n" + "="*80)
        print("CALCULATING DERIVED METRICS (decimal scale 0-1)")
        print("="*80)
    
    df = df.copy()
    
    # CT Release metrics (as decimals)
    df['pct_planned'] = np.where(
        df['ct_releases'] > 0,
        df['planned_ct_releases'] / df['ct_releases'],  # No × 100!
        np.nan
    )
    
    df['unplanned_ct_releases'] = df['ct_releases'] - df['planned_ct_releases'].fillna(0)
    
    df['design_error_rate'] = np.where(
        df['ct_releases'] > 0,
        df['design_error_count'] / df['ct_releases'],  # No × 100!
        np.nan
    )
    
    # CEI derived metrics (as decimals)
    if 'cei_hit' in df.columns and 'cei_miss' in df.columns:
        df['cei_total'] = df['cei_hit'] + df['cei_miss']
        
        df['cei_success_rate'] = np.where(
            df['cei_total'] > 0,
            df['cei_hit'] / df['cei_total'],  # No × 100!
            np.nan
        )
        
        if df['CEI'].isna().any():
            df['CEI'] = np.where(
                df['cei_total'] > 0,
                df['cei_hit'] / df['cei_total'],  # No × 100!
                np.nan
            )
            if verbose:
                print("   ℹ️  Recalculated CEI from hit/miss components (as decimal)")
    
    # BEI validation/recalculation (already a ratio)
    if 'bei_numerator' in df.columns and 'bei_denominator' in df.columns:
        if df['BEI'].isna().any():
            df['BEI'] = np.where(
                df['bei_denominator'] > 0,
                df['bei_numerator'] / df['bei_denominator'],
                np.nan
            )
            if verbose:
                print("   ℹ️  Recalculated BEI from numerator/denominator")
    
    if verbose:
        print("✅ Added derived metrics (all as decimals 0-1):")
        print("   - pct_planned (0.9 = 90% planned)")
        print("   - unplanned_ct_releases (count)")
        print("   - design_error_rate (0.1 = 10% error rate)")
        if 'cei_hit' in df.columns:
            print("   - cei_total (count)")
            print("   - cei_success_rate (0.85 = 85% success)")
    
    return df

# ===============================================================================
# CSV TEMPLATES
# ===============================================================================

def create_csv_templates(output_dir: str = "."):
    """Create template CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CEI template
    cei_template = pd.DataFrame({
        'Program Name': ['Program Alpha', 'Program Alpha', 'Program Beta'],
        'Project Status Date': ['2024-01', '2024-02', '2024-01'],
        'Hit (CEI)': [45, 52, 38],
        'Miss (CEI)': [5, 8, 12],
        'CEI Calc': [90.0, 86.7, 76.0]  # Will be converted to 0.90, 0.867, 0.76
    })
    cei_template.to_csv(output_dir / "TEMPLATE_cei_data.csv", index=False)
    
    # BEI template
    bei_template = pd.DataFrame({
        'Program Name': ['Program Alpha', 'Program Alpha', 'Program Beta'],
        'Project Status Date': ['2024-01', '2024-02', '2024-01'],
        'BEI Numerator': [950, 980, 890],
        'BEI Denominator': [1000, 1000, 1000],
        'BEI (Calc)': [0.95, 0.98, 0.89]
    })
    bei_template.to_csv(output_dir / "TEMPLATE_bei_data.csv", index=False)
    
    # Program mapping template
    mapping_template = pd.DataFrame({
        'CEI_BEI_Program_Name': ['Program Alpha', 'Program Beta', 'Program Gamma'],
        'Metrics_Program_Name': ['ALPHA_PROGRAM', 'BETA_PROGRAM', 'GAMMA_PROGRAM']
    })
    mapping_template.to_csv(output_dir / "TEMPLATE_program_mapping.csv", index=False)
    
    print(f"✅ Template CSV files created in: {output_dir}")

print("✅ CSV correlation analysis module loaded!")
print("   ✓ Percentages converted to decimal scale (0.0 - 1.0)")
print("   ✓ Example: 100% → 1.0, 85.5% → 0.855, 0% → 0.0")
