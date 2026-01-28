# ===============================================================================
# CSV-BASED CORRELATION ANALYSIS MODULE
# CEI and BEI Processed Separately + Actual Column Names from Screenshots
# ===============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# ===============================================================================
# PROGRAM NAME MAPPING
# ===============================================================================

def load_program_mapping(
    mapping_source: Union[str, Dict[str, str], pd.DataFrame],
    verbose: bool = True
) -> Dict[str, str]:
    """
    Load program name mapping from various sources.
    
    Maps CEI/BEI "Program Name" to Metrics "PMM Program Name"
    """
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
    """
    Load CEI data from CSV.
    
    Expected columns:
    - Program Name
    - Project Status Date
    - Hit (CEI)
    - Miss (CEI)
    - CEI Calc
    
    Returns DataFrame with standardized column names:
    - PMM_Program_Name
    - Fm_Reporting_Month
    - cei_hit
    - cei_miss
    - CEI
    """
    if verbose:
        print("\n" + "="*80)
        print("LOADING CEI DATA")
        print("="*80)
        print(f"📂 Loading from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
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
    
    return df

# ===============================================================================
# BEI DATA LOADING
# ===============================================================================

def load_bei_data(
    csv_path: str,
    program_mapping: Optional[Union[str, Dict[str, str], pd.DataFrame]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load BEI data from CSV.
    
    Expected columns:
    - Program Name
    - Project Status Date
    - BEI Numerator
    - BEI Denominator
    - BEI (Calc)
    
    Returns DataFrame with standardized column names:
    - PMM_Program_Name
    - Fm_Reporting_Month
    - bei_numerator
    - bei_denominator
    - BEI
    """
    if verbose:
        print("\n" + "="*80)
        print("LOADING BEI DATA")
        print("="*80)
        print(f"📂 Loading from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
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
    
    return df

# ===============================================================================
# ENGINEERING METRICS LOADING (With Actual Column Names from Screenshots)
# ===============================================================================

def load_engineering_metrics(
    ontime_csv: str,
    preventable_csv: str,
    design_error_csv: str,
    planned_ct_csv: str,
    ct_releases_csv: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load engineering metrics from CSV files with ACTUAL column names from screenshots.
    
    Expected columns:
    - On-Time: "PMM Program Name", "Fm Reporting Month", "% On-Time to Forecast"
    - Preventable: "PMM Program Name", "Fm Reporting Month", "Preventable % of Total Revisions"
    - Design Error: "PMM Program Name", "Fm Reporting Month", "RFC 2 Released CT Count"
    - Planned CT: "PMM Program Name", "Fm Reporting Month", "Planned Released CTs"
    - CT Releases: "PMM Program Name", "Fm Reporting Month", "Released CT Count"
    """
    if verbose:
        print("\n" + "="*80)
        print("LOADING ENGINEERING METRICS")
        print("="*80)
    
    # Load On-Time
    if verbose:
        print("\n1️⃣  Loading On-Time to Forecast...")
    ontime_df = pd.read_csv(ontime_csv)
    ontime_df = ontime_df.rename(columns={
        'PMM Program Name': 'PMM_Program_Name',
        'Fm Reporting Month': 'Fm_Reporting_Month'
    })
    required = ['PMM_Program_Name', 'Fm_Reporting_Month', '% On-Time to Forecast']
    missing = [col for col in required if col not in ontime_df.columns]
    if missing:
        raise ValueError(f"❌ On-Time CSV missing: {missing}\nFound: {list(ontime_df.columns)}")
    if verbose:
        print(f"   ✅ {len(ontime_df)} records")
    
    # Load Preventable Revisions
    if verbose:
        print("\n2️⃣  Loading Preventable Revisions...")
    preventable_df = pd.read_csv(preventable_csv)
    preventable_df = preventable_df.rename(columns={
        'PMM Program Name': 'PMM_Program_Name',
        'Fm Reporting Month': 'Fm_Reporting_Month',
        'Preventable % of Total Revisions': 'pct_preventable_revisions'
    })
    required = ['PMM_Program_Name', 'Fm_Reporting_Month', 'pct_preventable_revisions']
    missing = [col for col in required if col not in preventable_df.columns]
    if missing:
        raise ValueError(f"❌ Preventable CSV missing: {missing}\nFound: {list(preventable_df.columns)}")
    if verbose:
        print(f"   ✅ {len(preventable_df)} records")
    
    # Load Design Error Count
    if verbose:
        print("\n3️⃣  Loading Design Error Count...")
    design_df = pd.read_csv(design_error_csv)
    design_df = design_df.rename(columns={
        'PMM Program Name': 'PMM_Program_Name',
        'Fm Reporting Month': 'Fm_Reporting_Month',
        'RFC 2 Released CT Count': 'design_error_count'
    })
    required = ['PMM_Program_Name', 'Fm_Reporting_Month', 'design_error_count']
    missing = [col for col in required if col not in design_df.columns]
    if missing:
        raise ValueError(f"❌ Design Error CSV missing: {missing}\nFound: {list(design_df.columns)}")
    if verbose:
        print(f"   ✅ {len(design_df)} records")
    
    # Load Planned CT Releases
    if verbose:
        print("\n4️⃣  Loading Planned CT Releases...")
    planned_df = pd.read_csv(planned_ct_csv)
    planned_df = planned_df.rename(columns={
        'PMM Program Name': 'PMM_Program_Name',
        'Fm Reporting Month': 'Fm_Reporting_Month',
        'Planned Released CTs': 'planned_ct_releases'
    })
    required = ['PMM_Program_Name', 'Fm_Reporting_Month', 'planned_ct_releases']
    missing = [col for col in required if col not in planned_df.columns]
    if missing:
        raise ValueError(f"❌ Planned CT CSV missing: {missing}\nFound: {list(planned_df.columns)}")
    if verbose:
        print(f"   ✅ {len(planned_df)} records")
    
    # Load CT Releases
    if verbose:
        print("\n5️⃣  Loading CT Releases...")
    ct_df = pd.read_csv(ct_releases_csv)
    ct_df = ct_df.rename(columns={
        'PMM Program Name': 'PMM_Program_Name',
        'Fm Reporting Month': 'Fm_Reporting_Month',
        'Released CT Count': 'ct_releases'
    })
    required = ['PMM_Program_Name', 'Fm_Reporting_Month', 'ct_releases']
    missing = [col for col in required if col not in ct_df.columns]
    if missing:
        raise ValueError(f"❌ CT Releases CSV missing: {missing}\nFound: {list(ct_df.columns)}")
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
# COMPLETE DATA LOADING (CEI + BEI + Engineering Metrics)
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
    """
    Load CEI, BEI, and all engineering metrics separately, then merge.
    
    Args:
        cei_csv: CEI data (separate file)
        bei_csv: BEI data (separate file)
        ontime_csv: On-Time to Forecast
        preventable_csv: Preventable Revisions
        design_error_csv: Design Error Count
        planned_ct_csv: Planned CT Releases
        ct_releases_csv: CT Releases
        program_mapping: Mapping from CEI/BEI program names to metrics program names
        verbose: Print progress
        
    Returns:
        Merged DataFrame with all data
    """
    if verbose:
        print("="*80)
        print("LOADING ALL DATA (CEI, BEI, ENGINEERING METRICS)")
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
    
    if verbose:
        print(f"\n✅ Merged dataset: {len(merged)} records")
        print(f"   Programs: {merged['PMM_Program_Name'].nunique()}")
        print(f"   Date range: {merged['Fm_Reporting_Month'].min()} to {merged['Fm_Reporting_Month'].max()}")
        
        # Show merge statistics
        cei_programs = set(cei_df['PMM_Program_Name'].dropna().unique())
        bei_programs = set(bei_df['PMM_Program_Name'].dropna().unique())
        metrics_programs = set(metrics_df['PMM_Program_Name'].dropna().unique())
        
        common_all = cei_programs & bei_programs & metrics_programs
        
        print(f"\n📊 Program Overlap:")
        print(f"   Common to all (CEI + BEI + Metrics): {len(common_all)}")
        print(f"   Only in CEI: {len(cei_programs - bei_programs - metrics_programs)}")
        print(f"   Only in BEI: {len(bei_programs - cei_programs - metrics_programs)}")
        print(f"   Only in Metrics: {len(metrics_programs - cei_programs - bei_programs)}")
    
    return merged

# ===============================================================================
# DERIVED METRICS
# ===============================================================================

def add_derived_metrics(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add derived metrics.
    
    Calculates:
    - pct_planned: % of releases that were planned
    - unplanned_ct_releases: Count of unplanned releases
    - design_error_rate: Design errors as % of total releases
    - cei_total: Total CEI attempts (hit + miss)
    - cei_success_rate: Hit rate for CEI
    """
    if verbose:
        print("\n" + "="*80)
        print("CALCULATING DERIVED METRICS")
        print("="*80)
    
    df = df.copy()
    
    # CT Release metrics
    df['pct_planned'] = np.where(
        df['ct_releases'] > 0,
        (df['planned_ct_releases'] / df['ct_releases']) * 100,
        np.nan
    )
    
    df['unplanned_ct_releases'] = df['ct_releases'] - df['planned_ct_releases'].fillna(0)
    
    df['design_error_rate'] = np.where(
        df['ct_releases'] > 0,
        (df['design_error_count'] / df['ct_releases']) * 100,
        np.nan
    )
    
    # CEI derived metrics (if CEI columns present)
    if 'cei_hit' in df.columns and 'cei_miss' in df.columns:
        df['cei_total'] = df['cei_hit'] + df['cei_miss']
        
        df['cei_success_rate'] = np.where(
            df['cei_total'] > 0,
            (df['cei_hit'] / df['cei_total']) * 100,
            np.nan
        )
        
        # Validate/recalculate CEI if needed
        if df['CEI'].isna().any():
            df['CEI'] = np.where(
                df['cei_total'] > 0,
                (df['cei_hit'] / df['cei_total']) * 100,
                np.nan
            )
            if verbose:
                print("   ℹ️  Recalculated CEI from hit/miss components")
    
    # BEI validation/recalculation (if BEI columns present)
    if 'bei_numerator' in df.columns and 'bei_denominator' in df.columns:
        if df['BEI'].isna().any():
            df['BEI'] = np.where(
                df['bei_denominator'] > 0,
                (df['bei_numerator'] / df['bei_denominator']),
                np.nan
            )
            if verbose:
                print("   ℹ️  Recalculated BEI from numerator/denominator")
    
    if verbose:
        print("✅ Added derived metrics:")
        print("   - pct_planned: % of releases that were planned")
        print("   - unplanned_ct_releases: Count of unplanned releases")
        print("   - design_error_rate: Design errors as % of total releases")
        if 'cei_hit' in df.columns:
            print("   - cei_total: Total CEI attempts (hit + miss)")
            print("   - cei_success_rate: Hit rate for CEI")
    
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
        'CEI Calc': [90.0, 86.7, 76.0]
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
    print(f"   - TEMPLATE_cei_data.csv")
    print(f"   - TEMPLATE_bei_data.csv")
    print(f"   - TEMPLATE_program_mapping.csv")

print("✅ CSV correlation analysis module loaded!")
print("   ✓ CEI and BEI processed separately")
print("   ✓ Actual column names from screenshots")
