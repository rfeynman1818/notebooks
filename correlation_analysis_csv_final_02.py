# ===============================================================================
# CSV-BASED CORRELATION ANALYSIS MODULE (With Program Name Mapping)
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
    
    Maps CEI/BEI "Program Name" to Metrics "PRGRM_NAME"
    
    Args:
        mapping_source: Can be:
            - Path to CSV file with columns: CEI_BEI_Program_Name, Metrics_Program_Name
            - Dictionary: {cei_bei_name: metrics_name}
            - DataFrame with columns: CEI_BEI_Program_Name, Metrics_Program_Name
        verbose: Print mapping info
        
    Returns:
        Dictionary mapping CEI/BEI names to Metrics names
    """
    if verbose:
        print("="*80)
        print("LOADING PROGRAM NAME MAPPING")
        print("="*80)
    
    # Handle different input types
    if isinstance(mapping_source, dict):
        mapping = mapping_source
    elif isinstance(mapping_source, pd.DataFrame):
        if 'CEI_BEI_Program_Name' in mapping_source.columns and 'Metrics_Program_Name' in mapping_source.columns:
            mapping = dict(zip(
                mapping_source['CEI_BEI_Program_Name'],
                mapping_source['Metrics_Program_Name']
            ))
        else:
            # Try first two columns
            mapping = dict(zip(
                mapping_source.iloc[:, 0],
                mapping_source.iloc[:, 1]
            ))
    elif isinstance(mapping_source, str):
        # Load from CSV
        df = pd.read_csv(mapping_source)
        if 'CEI_BEI_Program_Name' in df.columns and 'Metrics_Program_Name' in df.columns:
            mapping = dict(zip(df['CEI_BEI_Program_Name'], df['Metrics_Program_Name']))
        else:
            # Try first two columns
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
    target_col: str = 'PRGRM_NAME',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply program name mapping to a dataframe.
    
    Args:
        df: DataFrame with source program names
        mapping: Dictionary mapping source names to target names
        source_col: Column with original program names
        target_col: Column name to create with mapped names
        verbose: Print mapping statistics
        
    Returns:
        DataFrame with new target column
    """
    df = df.copy()
    
    # Map the names
    df[target_col] = df[source_col].map(mapping)
    
    # Check for unmapped programs
    unmapped = df[df[target_col].isna()][source_col].unique()
    
    if verbose and len(unmapped) > 0:
        print(f"\n⚠️  Warning: {len(unmapped)} programs not in mapping:")
        for prog in unmapped[:10]:
            print(f"   - {prog}")
        if len(unmapped) > 10:
            print(f"   ... and {len(unmapped) - 10} more")
    
    return df

def create_mapping_template(output_path: str = "program_name_mapping_template.csv"):
    """
    Create a template CSV file for program name mapping.
    
    Args:
        output_path: Where to save the template
    """
    template = pd.DataFrame({
        'CEI_BEI_Program_Name': [
            'Program Alpha',
            'Program Beta',
            'Program Gamma'
        ],
        'Metrics_Program_Name': [
            'ALPHA_PROGRAM',
            'BETA_PROGRAM',
            'GAMMA_PROGRAM'
        ]
    })
    
    template.to_csv(output_path, index=False)
    print(f"✅ Created mapping template: {output_path}")
    print("\nFill in your actual program names:")
    print("  - Column 1 (CEI_BEI_Program_Name): Names as they appear in CEI/BEI data")
    print("  - Column 2 (Metrics_Program_Name): Names as they appear in engineering metrics")

# ===============================================================================
# CSV LOADING AND VALIDATION
# ===============================================================================

def validate_csv_columns(df: pd.DataFrame, required_cols: List[str], csv_name: str) -> bool:
    """Validate that CSV has required columns."""
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        raise ValueError(
            f"❌ {csv_name} is missing required columns: {missing}\n"
            f"   Required: {required_cols}\n"
            f"   Found: {list(df.columns)}"
        )
    
    return True

def load_cei_bei_data(
    csv_path: str,
    program_mapping: Optional[Union[str, Dict[str, str], pd.DataFrame]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load CEI/BEI data from CSV.
    
    Expected columns:
    - Program Name
    - Project Status Date
    - Hit (CEI)
    - Miss (CEI)
    - CEI Calc
    - BEI Numerator
    - BEI Denominator
    - BEI (Calc)
    
    Args:
        csv_path: Path to CEI/BEI CSV file
        program_mapping: Optional mapping from CEI/BEI program names to metrics names
        verbose: Print progress
        
    Returns:
        DataFrame with standardized column names
    """
    if verbose:
        print("\n" + "="*80)
        print("LOADING CEI/BEI DATA")
        print("="*80)
        print(f"📂 Loading from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check for required columns
    required = ['Program Name', 'Project Status Date', 
                'Hit (CEI)', 'Miss (CEI)', 'CEI Calc',
                'BEI Numerator', 'BEI Denominator', 'BEI (Calc)']
    validate_csv_columns(df, required, Path(csv_path).name)
    
    # Standardize column names
    df = df.rename(columns={
        'Project Status Date': 'FM_REPORTING_MONTH',
        'Hit (CEI)': 'cei_hit',
        'Miss (CEI)': 'cei_miss',
        'CEI Calc': 'CEI',
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
                                   target_col='PRGRM_NAME',
                                   verbose=verbose)
    else:
        # No mapping - just rename the column
        df = df.rename(columns={'Program Name': 'PRGRM_NAME'})
    
    if verbose:
        print(f"\n✅ Loaded {len(df)} CEI/BEI records")
        print(f"   Programs: {df['PRGRM_NAME'].nunique()}")
        print(f"   Date range: {df['FM_REPORTING_MONTH'].min()} to {df['FM_REPORTING_MONTH'].max()}")
    
    return df

def load_separate_csvs(
    cei_bei_csv: str,
    ontime_csv: str,
    preventable_csv: str,
    design_error_csv: str,
    planned_ct_csv: str,
    ct_releases_csv: str,
    program_mapping: Optional[Union[str, Dict[str, str], pd.DataFrame]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load separate CSV files and merge them.
    
    Args:
        cei_bei_csv: CSV with CEI and BEI data (no PMM_Program_ID required)
        ontime_csv: On-Time to Forecast report CSV
        preventable_csv: Preventable Revisions report CSV
        design_error_csv: Design Error Count report CSV
        planned_ct_csv: Planned CT Releases report CSV
        ct_releases_csv: CT Releases report CSV
        program_mapping: Mapping from CEI/BEI program names to metrics names
        verbose: Print progress
        
    Returns:
        Merged DataFrame
    """
    if verbose:
        print("="*80)
        print("LOADING SEPARATE CSV FILES")
        print("="*80)
    
    # Load CEI/BEI data (with program mapping if provided)
    cei_bei_df = load_cei_bei_data(cei_bei_csv, program_mapping, verbose)
    
    # Load engineering metrics
    if verbose:
        print("\n" + "="*80)
        print("LOADING ENGINEERING METRICS")
        print("="*80)
    
    # On-time
    if verbose:
        print("\n1️⃣  Loading On-Time to Forecast...")
    ontime_df = pd.read_csv(ontime_csv)
    if 'PMM_Program_Name' in ontime_df.columns:
        ontime_df = ontime_df.rename(columns={'PMM_Program_Name': 'PRGRM_NAME'})
    # PMM_Program_ID is optional
    required_ontime = ['PRGRM_NAME', 'FM_REPORTING_MONTH', '% On-Time to Forecast']
    validate_csv_columns(ontime_df, required_ontime, Path(ontime_csv).name)
    if verbose:
        print(f"   ✅ {len(ontime_df)} records")
    
    # Preventable revisions
    if verbose:
        print("\n2️⃣  Loading Preventable Revisions...")
    preventable_df = pd.read_csv(preventable_csv)
    required_prev = ['PRGRM_NAME', 'FM_REPORTING_MONTH', '% Preventable Revisions']
    validate_csv_columns(preventable_df, required_prev, Path(preventable_csv).name)
    if verbose:
        print(f"   ✅ {len(preventable_df)} records")
    
    # Design error count
    if verbose:
        print("\n3️⃣  Loading Design Error Count...")
    design_df = pd.read_csv(design_error_csv)
    required_design = ['PRGRM_NAME', 'FM_REPORTING_MONTH', 'design_error_count']
    validate_csv_columns(design_df, required_design, Path(design_error_csv).name)
    if verbose:
        print(f"   ✅ {len(design_df)} records")
    
    # Planned CT releases
    if verbose:
        print("\n4️⃣  Loading Planned CT Releases...")
    planned_df = pd.read_csv(planned_ct_csv)
    required_planned = ['PRGRM_NAME', 'FM_REPORTING_MONTH', 'planned_ct_releases']
    validate_csv_columns(planned_df, required_planned, Path(planned_ct_csv).name)
    if verbose:
        print(f"   ✅ {len(planned_df)} records")
    
    # CT releases
    if verbose:
        print("\n5️⃣  Loading CT Releases...")
    ct_df = pd.read_csv(ct_releases_csv)
    required_ct = ['PRGRM_NAME', 'FM_REPORTING_MONTH', 'ct_releases']
    validate_csv_columns(ct_df, required_ct, Path(ct_releases_csv).name)
    if verbose:
        print(f"   ✅ {len(ct_df)} records")
    
    # Merge all dataframes
    if verbose:
        print("\n" + "="*80)
        print("MERGING ALL DATA")
        print("="*80)
    
    # Use only PRGRM_NAME and FM_REPORTING_MONTH as merge keys
    merge_keys = ['PRGRM_NAME', 'FM_REPORTING_MONTH']
    
    # Start with CEI/BEI
    merged = cei_bei_df.copy()
    
    # Merge engineering metrics
    merged = merged.merge(
        ontime_df[merge_keys + ['% On-Time to Forecast']], 
        on=merge_keys, 
        how='outer'
    )
    merged = merged.merge(
        preventable_df[merge_keys + ['% Preventable Revisions']], 
        on=merge_keys, 
        how='outer'
    )
    merged = merged.merge(
        design_df[merge_keys + ['design_error_count']], 
        on=merge_keys, 
        how='outer'
    )
    merged = merged.merge(
        planned_df[merge_keys + ['planned_ct_releases']], 
        on=merge_keys, 
        how='outer'
    )
    merged = merged.merge(
        ct_df[merge_keys + ['ct_releases']], 
        on=merge_keys, 
        how='outer'
    )
    
    if verbose:
        print(f"\n✅ Merged dataset: {len(merged)} records")
        print(f"   Programs: {merged['PRGRM_NAME'].nunique()}")
        print(f"   Date range: {merged['FM_REPORTING_MONTH'].min()} to {merged['FM_REPORTING_MONTH'].max()}")
        
        # Show merge statistics
        cei_bei_programs = set(cei_bei_df['PRGRM_NAME'].dropna().unique())
        metrics_programs = set(ontime_df['PRGRM_NAME'].dropna().unique())
        
        common_programs = cei_bei_programs & metrics_programs
        only_cei_bei = cei_bei_programs - metrics_programs
        only_metrics = metrics_programs - cei_bei_programs
        
        print(f"\n📊 Program Overlap:")
        print(f"   Common programs: {len(common_programs)}")
        if len(only_cei_bei) > 0:
            print(f"   ⚠️  Only in CEI/BEI: {len(only_cei_bei)}")
        if len(only_metrics) > 0:
            print(f"   ⚠️  Only in Metrics: {len(only_metrics)}")
    
    return merged

def add_derived_metrics_csv(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add derived metrics to CSV data.
    
    Calculates:
    - pct_planned: % of releases that were planned
    - unplanned_ct_releases: Count of unplanned releases
    - design_error_rate: Design errors as % of total releases
    - cei_total: Total CEI attempts (hit + miss)
    - cei_success_rate: Hit rate for CEI
    
    Also validates/recalculates CEI and BEI if needed.
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
    
    # CEI derived metrics
    if 'cei_hit' in df.columns and 'cei_miss' in df.columns:
        df['cei_total'] = df['cei_hit'] + df['cei_miss']
        
        df['cei_success_rate'] = np.where(
            df['cei_total'] > 0,
            (df['cei_hit'] / df['cei_total']) * 100,
            np.nan
        )
        
        # Validate/recalculate CEI if needed
        if df['CEI'].isna().any() or 'CEI' not in df.columns:
            df['CEI'] = np.where(
                df['cei_total'] > 0,
                (df['cei_hit'] / df['cei_total']) * 100,
                np.nan
            )
            if verbose:
                print("   ℹ️  Recalculated CEI from hit/miss components")
    
    # BEI validation/recalculation
    if 'bei_numerator' in df.columns and 'bei_denominator' in df.columns:
        if df['BEI'].isna().any() or 'BEI' not in df.columns:
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

def create_csv_template(output_dir: str = "."):
    """Create template CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CEI/BEI template (no PMM_Program_ID)
    cei_bei_template = pd.DataFrame({
        'Program Name': ['Program Alpha', 'Program Alpha', 'Program Beta'],
        'Project Status Date': ['2024-01', '2024-02', '2024-01'],
        'Hit (CEI)': [45, 52, 38],
        'Miss (CEI)': [5, 8, 12],
        'CEI Calc': [90.0, 86.7, 76.0],
        'BEI Numerator': [950, 980, 890],
        'BEI Denominator': [1000, 1000, 1000],
        'BEI (Calc)': [0.95, 0.98, 0.89]
    })
    cei_bei_template.to_csv(output_dir / "TEMPLATE_cei_bei_data.csv", index=False)
    
    # Program mapping template
    create_mapping_template(str(output_dir / "TEMPLATE_program_mapping.csv"))
    
    print(f"\n✅ Template CSV files created in: {output_dir}")
    print(f"   - TEMPLATE_cei_bei_data.csv (CEI/BEI format)")
    print(f"   - TEMPLATE_program_mapping.csv (Program name mapping)")

print("✅ CSV correlation analysis module loaded (with program name mapping)!")
