# ===============================================================================
# CSV-BASED CORRELATION ANALYSIS MODULE
# With Percentages as Decimals (0.0 - 1.0 scale)
# ===============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# ===============================================================================
# DATE NORMALIZATION
# ===============================================================================

def normalize_date_column(df: pd.DataFrame, date_col: str = 'Fm_Reporting_Month', verbose: bool = False) -> pd.DataFrame:
    """
    Normalize date column to standard YYYY-MM format for consistent comparison.
    
    Handles various input formats:
    - "1/26/2025", "01/26/2025" (MM/DD/YYYY)
    - "2025-01-26" (YYYY-MM-DD)
    - "Jan 2025", "January 2025" (Month Year)
    - "2025-01" (YYYY-MM)
    - Excel date numbers
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column to normalize
        verbose: Print diagnostic info
        
    Returns:
        DataFrame with normalized date column
    """
    if date_col not in df.columns:
        return df
    
    df = df.copy()
    original_values = df[date_col].copy()
    
    if verbose:
        print(f"\n📅 Normalizing dates in '{date_col}'...")
        print(f"   Original format samples: {df[date_col].dropna().head(3).tolist()}")
    
    # Try to parse dates with pandas (handles most formats automatically)
    try:
        # Attempt 1: Let pandas infer the format
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Convert to YYYY-MM format (year-month only, ignore day)
        df[date_col] = df[date_col].dt.to_period('M').astype(str)
        
        # Count successful conversions
        success_count = df[date_col].notna().sum()
        total_count = original_values.notna().sum()
        
        if verbose:
            print(f"   Standardized format: YYYY-MM (e.g., '2025-01')")
            print(f"   Successfully parsed: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
            print(f"   Normalized samples: {df[date_col].dropna().head(3).tolist()}")
        
        # Warn if some dates failed to parse
        if success_count < total_count:
            failed_count = total_count - success_count
            if verbose:
                print(f"   ⚠️  Warning: {failed_count} dates could not be parsed")
                failed_samples = original_values[df[date_col].isna() & original_values.notna()].head(3)
                if len(failed_samples) > 0:
                    print(f"   Failed samples: {failed_samples.tolist()}")
        
    except Exception as e:
        if verbose:
            print(f"   ⚠️  Date parsing failed: {e}")
            print(f"   Keeping original values")
        df[date_col] = original_values
    
    return df

# ===============================================================================
# DATA QUALITY VALIDATION
# ===============================================================================

def detect_placeholder_zeros(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None, 
                            threshold: float = 0.5, verbose: bool = True) -> Dict[str, any]:
    """
    Detect columns where 0.0 might be a placeholder for missing data.
    
    Args:
        df: DataFrame to check
        numeric_cols: Specific numeric columns to check (if None, checks all numeric)
        threshold: If >threshold of values are 0.0, flag as suspicious
        verbose: Print findings
        
    Returns:
        Dictionary with suspicious columns and statistics
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    suspicious = {}
    
    for col in numeric_cols:
        if col in df.columns:
            total = len(df[col])
            zero_count = (df[col] == 0.0).sum()
            zero_pct = zero_count / total if total > 0 else 0
            
            if zero_pct > threshold:
                suspicious[col] = {
                    'zero_count': zero_count,
                    'total': total,
                    'zero_pct': zero_pct * 100,
                    'also_null': df[col].isna().sum()
                }
    
    if verbose and suspicious:
        print(f"\n⚠️  Found {len(suspicious)} columns with >{threshold*100:.0f}% zeros (possible placeholders):")
        for col, stats in suspicious.items():
            print(f"   {col}:")
            print(f"      - {stats['zero_count']}/{stats['total']} values are 0.0 ({stats['zero_pct']:.1f}%)")
            print(f"      - {stats['also_null']} are NULL")
            print(f"      → Recommendation: Check if 0.0 means 'missing' for this column")
    
    return suspicious

def check_missing_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Comprehensive missing data analysis.
    
    Returns DataFrame with:
    - Column name
    - Missing count
    - Missing percentage
    - Data type
    - Recommendation
    """
    missing_info = []
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        total = len(df)
        missing_pct = (missing_count / total * 100) if total > 0 else 0
        dtype = str(df[col].dtype)
        
        # Recommendation based on missing percentage
        if missing_pct == 0:
            recommendation = "✅ Complete"
        elif missing_pct < 5:
            recommendation = "✅ Acceptable (<5%)"
        elif missing_pct < 20:
            recommendation = "⚠️  Moderate (5-20%)"
        elif missing_pct < 50:
            recommendation = "⚠️  High (20-50%)"
        else:
            recommendation = "❌ Critical (>50%) - Consider excluding"
        
        missing_info.append({
            'Column': col,
            'Missing_Count': missing_count,
            'Total_Count': total,
            'Missing_Pct': missing_pct,
            'Data_Type': dtype,
            'Recommendation': recommendation
        })
    
    missing_df = pd.DataFrame(missing_info)
    missing_df = missing_df.sort_values('Missing_Pct', ascending=False)
    
    if verbose:
        print("\n" + "="*80)
        print("MISSING DATA ANALYSIS")
        print("="*80)
        
        # Show only columns with missing data
        has_missing = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(has_missing) == 0:
            print("\n✅ No missing data found! All columns complete.")
        else:
            print(f"\n📊 {len(has_missing)} columns have missing data:")
            print()
            for _, row in has_missing.iterrows():
                print(f"   {row['Column']}:")
                print(f"      Missing: {row['Missing_Count']}/{row['Total_Count']} ({row['Missing_Pct']:.1f}%)")
                print(f"      {row['Recommendation']}")
        
        # Summary statistics
        critical = len(missing_df[missing_df['Missing_Pct'] > 50])
        high = len(missing_df[(missing_df['Missing_Pct'] > 20) & (missing_df['Missing_Pct'] <= 50)])
        moderate = len(missing_df[(missing_df['Missing_Pct'] > 5) & (missing_df['Missing_Pct'] <= 20)])
        
        if critical > 0 or high > 0:
            print(f"\n⚠️  Data Quality Summary:")
            if critical > 0:
                print(f"   ❌ {critical} columns with >50% missing (consider excluding)")
            if high > 0:
                print(f"   ⚠️  {high} columns with 20-50% missing (use with caution)")
            if moderate > 0:
                print(f"   ⚠️  {moderate} columns with 5-20% missing (acceptable)")
    
    return missing_df

def detect_duplicates(df: pd.DataFrame, 
                     subset: Optional[List[str]] = None,
                     keep: str = 'first',
                     verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect duplicate rows.
    
    Args:
        df: DataFrame to check
        subset: Columns to check for duplicates (None = all columns)
        keep: Which duplicates to mark ('first', 'last', False)
        verbose: Print findings
        
    Returns:
        (duplicate_rows, deduplicated_df)
    """
    # Find duplicates
    if subset is None:
        duplicates_mask = df.duplicated(keep=keep)
    else:
        duplicates_mask = df.duplicated(subset=subset, keep=keep)
    
    duplicate_rows = df[duplicates_mask].copy()
    
    if verbose:
        print("\n" + "="*80)
        print("DUPLICATE ROW DETECTION")
        print("="*80)
        
        if len(duplicate_rows) == 0:
            print("\n✅ No duplicate rows found!")
        else:
            print(f"\n⚠️  Found {len(duplicate_rows)} duplicate rows")
            
            if subset:
                print(f"   Based on columns: {subset}")
            else:
                print(f"   Based on: ALL columns")
            
            # Show example duplicates
            if len(duplicate_rows) > 0:
                print(f"\n   Example duplicate records:")
                # Get first few duplicates with their originals
                for idx in duplicate_rows.head(3).index:
                    if subset:
                        matching = df[df[subset].eq(df.loc[idx, subset]).all(axis=1)]
                    else:
                        matching = df[df.eq(df.loc[idx]).all(axis=1)]
                    
                    print(f"\n   Duplicate group (rows {matching.index.tolist()}):")
                    if subset:
                        for col in subset:
                            print(f"      {col}: {df.loc[idx, col]}")
                    else:
                        print(f"      Row {idx}: {len(matching)} identical copies")
    
    # Create deduplicated version
    deduplicated_df = df[~duplicates_mask].copy()
    
    return duplicate_rows, deduplicated_df

def validate_data_types(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Validate and report data types for all columns.
    
    Returns DataFrame with column types and validation status.
    """
    type_info = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_values = df[col].dropna().head(3).tolist()
        
        # Classify type
        if 'int' in dtype or 'float' in dtype:
            category = "Numeric"
            status = "✅ Correct"
        elif 'object' in dtype:
            # Check if it should be numeric
            if col in ['PMM_Program_Name', 'Fm_Reporting_Month']:
                category = "Identifier/Date"
                status = "✅ Correct"
            else:
                # Check if values look numeric
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    try:
                        pd.to_numeric(non_null.head(100), errors='raise')
                        category = "Text (should be Numeric?)"
                        status = "⚠️  Check if numeric"
                    except:
                        category = "Text"
                        status = "✅ Correct"
                else:
                    category = "Text (empty)"
                    status = "⚠️  All NULL"
        elif 'datetime' in dtype or 'period' in dtype:
            category = "Date/Time"
            status = "✅ Correct"
        else:
            category = "Other"
            status = "⚠️  Unknown type"
        
        type_info.append({
            'Column': col,
            'Data_Type': dtype,
            'Category': category,
            'Status': status,
            'Sample_Values': str(sample_values)[:50]
        })
    
    type_df = pd.DataFrame(type_info)
    
    if verbose:
        print("\n" + "="*80)
        print("DATA TYPE VALIDATION")
        print("="*80)
        
        # Group by category
        print("\n📊 Column Types Summary:")
        for category in type_df['Category'].unique():
            cols = type_df[type_df['Category'] == category]
            print(f"\n   {category}: {len(cols)} columns")
            for _, row in cols.iterrows():
                print(f"      {row['Column']}: {row['Data_Type']} {row['Status']}")
        
        # Warnings
        issues = type_df[type_df['Status'].str.contains('⚠️')]
        if len(issues) > 0:
            print(f"\n⚠️  {len(issues)} potential type issues:")
            for _, row in issues.iterrows():
                print(f"   {row['Column']}: {row['Status']}")
    
    return type_df

def generate_data_quality_report(df: pd.DataFrame, 
                                 check_duplicates_on: Optional[List[str]] = None,
                                 placeholder_zero_threshold: float = 0.5,
                                 verbose: bool = True) -> Dict[str, any]:
    """
    Comprehensive data quality report combining all checks.
    
    Args:
        df: DataFrame to validate
        check_duplicates_on: Columns to check for duplicates (None = all columns)
        placeholder_zero_threshold: Threshold for flagging columns with many zeros
        verbose: Print detailed report
        
    Returns:
        Dictionary with all quality check results
    """
    if verbose:
        print("\n" + "="*80)
        print("COMPREHENSIVE DATA QUALITY VALIDATION")
        print("="*80)
        print(f"\n📊 Dataset: {len(df)} rows × {len(df.columns)} columns")
    
    results = {}
    
    # 1. Missing data analysis
    results['missing_data'] = check_missing_data(df, verbose=verbose)
    
    # 2. Duplicate detection
    if check_duplicates_on is None:
        # Default: check on Program + Month
        if 'PMM_Program_Name' in df.columns and 'Fm_Reporting_Month' in df.columns:
            check_duplicates_on = ['PMM_Program_Name', 'Fm_Reporting_Month']
    
    duplicate_rows, deduplicated = detect_duplicates(df, subset=check_duplicates_on, verbose=verbose)
    results['duplicates'] = {
        'duplicate_rows': duplicate_rows,
        'deduplicated_df': deduplicated,
        'duplicate_count': len(duplicate_rows)
    }
    
    # 3. Data type validation
    results['data_types'] = validate_data_types(df, verbose=verbose)
    
    # 4. Placeholder zeros check
    results['placeholder_zeros'] = detect_placeholder_zeros(
        df, 
        threshold=placeholder_zero_threshold,
        verbose=verbose
    )
    
    # 5. Summary recommendations
    if verbose:
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        critical_missing = len(results['missing_data'][results['missing_data']['Missing_Pct'] > 50])
        has_duplicates = len(duplicate_rows) > 0
        has_placeholder_zeros = len(results['placeholder_zeros']) > 0
        
        if critical_missing == 0 and not has_duplicates and not has_placeholder_zeros:
            print("\n✅ Data quality is GOOD!")
            print("   No critical issues found.")
        else:
            print("\n⚠️  Action items:")
            
            if critical_missing > 0:
                print(f"\n   1. CRITICAL MISSING DATA:")
                print(f"      {critical_missing} columns have >50% missing data")
                print(f"      → Recommendation: Exclude these from analysis")
                crit_cols = results['missing_data'][results['missing_data']['Missing_Pct'] > 50]['Column'].tolist()
                print(f"      → Columns: {crit_cols}")
            
            if has_duplicates:
                print(f"\n   2. DUPLICATE ROWS:")
                print(f"      {len(duplicate_rows)} duplicate rows found")
                print(f"      → Recommendation: Review and remove if errors")
                print(f"      → Use: deduplicated_df = results['duplicates']['deduplicated_df']")
            
            if has_placeholder_zeros:
                print(f"\n   3. PLACEHOLDER ZEROS:")
                print(f"      {len(results['placeholder_zeros'])} columns with many 0.0 values")
                print(f"      → Recommendation: Verify if 0.0 means 'missing'")
                print(f"      → Columns: {list(results['placeholder_zeros'].keys())}")
    
    return results

def clean_data_quality_issues(df: pd.DataFrame,
                              remove_duplicates: bool = True,
                              duplicate_subset: Optional[List[str]] = None,
                              exclude_high_missing_cols: bool = True,
                              missing_threshold: float = 0.5,
                              convert_placeholder_zeros: bool = False,
                              placeholder_zero_cols: Optional[List[str]] = None,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Automatically clean data quality issues.
    
    Args:
        df: DataFrame to clean
        remove_duplicates: Remove duplicate rows
        duplicate_subset: Columns to check for duplicates
        exclude_high_missing_cols: Exclude columns with >threshold missing
        missing_threshold: Threshold for excluding columns (0.5 = 50%)
        convert_placeholder_zeros: Convert 0.0 to NaN in specified columns
        placeholder_zero_cols: Columns where 0.0 should be NaN
        verbose: Print actions taken
        
    Returns:
        Cleaned DataFrame
    """
    if verbose:
        print("\n" + "="*80)
        print("AUTOMATIC DATA CLEANING")
        print("="*80)
        print(f"\n📊 Starting with: {len(df)} rows × {len(df.columns)} columns")
    
    df_cleaned = df.copy()
    actions_taken = []
    
    # 1. Remove duplicates
    if remove_duplicates:
        if duplicate_subset is None and 'PMM_Program_Name' in df.columns and 'Fm_Reporting_Month' in df.columns:
            duplicate_subset = ['PMM_Program_Name', 'Fm_Reporting_Month']
        
        before_len = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates(subset=duplicate_subset, keep='first')
        removed = before_len - len(df_cleaned)
        
        if removed > 0:
            actions_taken.append(f"Removed {removed} duplicate rows")
            if verbose:
                print(f"\n✅ Removed {removed} duplicate rows")
    
    # 2. Exclude high-missing columns
    if exclude_high_missing_cols:
        missing_info = check_missing_data(df_cleaned, verbose=False)
        high_missing_cols = missing_info[missing_info['Missing_Pct'] > missing_threshold * 100]['Column'].tolist()
        
        if high_missing_cols:
            df_cleaned = df_cleaned.drop(columns=high_missing_cols)
            actions_taken.append(f"Excluded {len(high_missing_cols)} columns with >{missing_threshold*100:.0f}% missing")
            if verbose:
                print(f"\n✅ Excluded {len(high_missing_cols)} columns with >{missing_threshold*100:.0f}% missing:")
                for col in high_missing_cols:
                    print(f"   - {col}")
    
    # 3. Convert placeholder zeros to NaN
    if convert_placeholder_zeros and placeholder_zero_cols:
        for col in placeholder_zero_cols:
            if col in df_cleaned.columns:
                zero_count = (df_cleaned[col] == 0.0).sum()
                df_cleaned.loc[df_cleaned[col] == 0.0, col] = np.nan
                actions_taken.append(f"Converted {zero_count} zeros to NaN in {col}")
                if verbose:
                    print(f"\n✅ Converted {zero_count} placeholder zeros to NaN in '{col}'")
    
    if verbose:
        print(f"\n📊 Final dataset: {len(df_cleaned)} rows × {len(df_cleaned.columns)} columns")
        
        if actions_taken:
            print(f"\n📋 Summary of actions:")
            for action in actions_taken:
                print(f"   ✓ {action}")
        else:
            print(f"\nℹ️  No cleaning actions needed")
    
    return df_cleaned

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
    
    # Normalize date format for consistent comparison
    df = normalize_date_column(df, 'Fm_Reporting_Month', verbose=verbose)
    
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
    
    # Normalize date format for consistent comparison
    df = normalize_date_column(df, 'Fm_Reporting_Month', verbose=verbose)
    
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
    ontime_csv: Optional[str] = None,
    preventable_csv: Optional[str] = None,
    design_error_csv: Optional[str] = None,
    planned_ct_csv: Optional[str] = None,
    ct_releases_csv: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Load engineering metrics with automatic percentage cleaning to decimal scale."""
    if verbose:
        print("\n" + "="*80)
        print("LOADING ENGINEERING METRICS")
        print("="*80)
    
    dfs_to_merge = []
    merge_keys = ['PMM_Program_Name', 'Fm_Reporting_Month']
    
    # Load On-Time
    if ontime_csv:
        if verbose:
            print("\n1️⃣  Loading On-Time to Forecast...")
        try:
            ontime_df = pd.read_csv(ontime_csv)
            ontime_df = clean_percentage_columns(ontime_df, verbose=verbose)
            ontime_df = ontime_df.rename(columns={
                'PMM Program Name': 'PMM_Program_Name',
                'Fm Reporting Month': 'Fm_Reporting_Month'
            })
            ontime_df = normalize_date_column(ontime_df, 'Fm_Reporting_Month', verbose=verbose)
            dfs_to_merge.append(ontime_df)
            if verbose:
                print(f"   ✅ {len(ontime_df)} records")
                if '% On-Time to Forecast' in ontime_df.columns:
                    sample = ontime_df['% On-Time to Forecast'].dropna().head(3)
                    print(f"   Sample values (decimal): {sample.tolist()}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Could not load: {e}")
    
    # Load Preventable Revisions
    if preventable_csv:
        if verbose:
            print("\n2️⃣  Loading Preventable Revisions...")
        try:
            preventable_df = pd.read_csv(preventable_csv)
            preventable_df = clean_percentage_columns(preventable_df, verbose=verbose)
            preventable_df = preventable_df.rename(columns={
                'PMM Program Name': 'PMM_Program_Name',
                'Fm Reporting Month': 'Fm_Reporting_Month',
                'Preventable % of Total Revisions': 'pct_preventable_revisions'
            })
            preventable_df = normalize_date_column(preventable_df, 'Fm_Reporting_Month', verbose=verbose)
            dfs_to_merge.append(preventable_df)
            if verbose:
                print(f"   ✅ {len(preventable_df)} records")
                if 'pct_preventable_revisions' in preventable_df.columns:
                    sample = preventable_df['pct_preventable_revisions'].dropna().head(3)
                    print(f"   Sample values (decimal): {sample.tolist()}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Could not load: {e}")
    
    # Load Design Error Count
    if design_error_csv:
        if verbose:
            print("\n3️⃣  Loading Design Error Count...")
        try:
            design_df = pd.read_csv(design_error_csv)
            design_df = clean_numeric_columns(design_df, verbose=False)
            design_df = design_df.rename(columns={
                'PMM Program Name': 'PMM_Program_Name',
                'Fm Reporting Month': 'Fm_Reporting_Month',
                'RFC 2 Released CT Count': 'design_error_count'
            })
            design_df = normalize_date_column(design_df, 'Fm_Reporting_Month', verbose=verbose)
            dfs_to_merge.append(design_df)
            if verbose:
                print(f"   ✅ {len(design_df)} records")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Could not load: {e}")
    
    # Load Planned CT Releases
    if planned_ct_csv:
        if verbose:
            print("\n4️⃣  Loading Planned CT Releases...")
        try:
            planned_df = pd.read_csv(planned_ct_csv)
            planned_df = clean_numeric_columns(planned_df, verbose=False)
            planned_df = planned_df.rename(columns={
                'PMM Program Name': 'PMM_Program_Name',
                'Fm Reporting Month': 'Fm_Reporting_Month',
                'Planned Released CTs': 'planned_ct_releases'
            })
            planned_df = normalize_date_column(planned_df, 'Fm_Reporting_Month', verbose=verbose)
            dfs_to_merge.append(planned_df)
            if verbose:
                print(f"   ✅ {len(planned_df)} records")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Could not load: {e}")
    
    # Load CT Releases
    if ct_releases_csv:
        if verbose:
            print("\n5️⃣  Loading CT Releases...")
        try:
            ct_df = pd.read_csv(ct_releases_csv)
            ct_df = clean_numeric_columns(ct_df, verbose=False)
            ct_df = clean_numeric_columns(ct_df, verbose=False)
            ct_df = ct_df.rename(columns={
                'PMM Program Name': 'PMM_Program_Name',
                'Fm Reporting Month': 'Fm_Reporting_Month',
                'Released CT Count': 'ct_releases'
            })
            ct_df = normalize_date_column(ct_df, 'Fm_Reporting_Month', verbose=verbose)
            dfs_to_merge.append(ct_df)
            if verbose:
                print(f"   ✅ {len(ct_df)} records")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Could not load: {e}")
    
    # If no files loaded, return empty dataframe with correct columns
    if not dfs_to_merge:
        if verbose:
            print("\n⚠️  No engineering metrics loaded")
        return pd.DataFrame(columns=merge_keys)
    
    # Merge all engineering metrics
    if verbose:
        print("\n🔗 Merging engineering metrics...")
    
    metrics = dfs_to_merge[0].copy()
    for df in dfs_to_merge[1:]:
        metrics = metrics.merge(df, on=merge_keys, how='outer')
    
    if verbose:
        print(f"   ✅ Merged {len(metrics)} engineering metric records")
    
    return metrics

# ===============================================================================
# COMPLETE DATA LOADING
# ===============================================================================

def load_all_data(
    cei_csv: Optional[str] = None,
    bei_csv: Optional[str] = None,
    ontime_csv: str = None,
    preventable_csv: str = None,
    design_error_csv: str = None,
    planned_ct_csv: str = None,
    ct_releases_csv: str = None,
    program_mapping: Optional[Union[str, Dict[str, str], pd.DataFrame]] = None,
    filter_by_target_dates: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load CEI, BEI, and all engineering metrics separately, then merge.
    
    Args:
        cei_csv: Path to CEI CSV (optional - can analyze BEI only)
        bei_csv: Path to BEI CSV (optional - can analyze CEI only)
        ontime_csv: Path to On-Time CSV (optional)
        preventable_csv: Path to Preventable Revisions CSV (optional)
        design_error_csv: Path to Design Error CSV (optional)
        planned_ct_csv: Path to Planned CT CSV (optional)
        ct_releases_csv: Path to CT Releases CSV (optional)
        program_mapping: Program name mapping (optional)
        filter_by_target_dates: If True, filter engineering metrics to CEI/BEI dates only.
                               If False, include all dates (may have more missing data).
        verbose: Print progress
        
    Returns:
        Merged DataFrame
        
    Note: At least ONE of cei_csv or bei_csv must be provided (or both)
    """
    if verbose:
        print("="*80)
        print("LOADING ALL DATA")
        print("Percentages will be converted to decimal scale (0.0 - 1.0)")
        print("="*80)
    
    # Validate: need at least one target metric
    if cei_csv is None and bei_csv is None:
        raise ValueError("Must provide at least one target: cei_csv or bei_csv (or both)")
    
    # Load CEI data if provided
    cei_df = None
    if cei_csv is not None:
        cei_df = load_cei_data(cei_csv, program_mapping, verbose)
    else:
        if verbose:
            print("\n⚠️  CEI data not provided - analyzing BEI only")
    
    # Load BEI data if provided
    bei_df = None
    if bei_csv is not None:
        bei_df = load_bei_data(bei_csv, program_mapping, verbose)
    else:
        if verbose:
            print("\n⚠️  BEI data not provided - analyzing CEI only")
    
    # Load engineering metrics (if any provided)
    metrics_df = None
    has_metrics = any([ontime_csv, preventable_csv, design_error_csv, planned_ct_csv, ct_releases_csv])
    
    if has_metrics:
        metrics_df = load_engineering_metrics(
            ontime_csv,
            preventable_csv,
            design_error_csv,
            planned_ct_csv,
            ct_releases_csv,
            verbose
        )
    else:
        if verbose:
            print("\n⚠️  No engineering metrics provided")
    
    # Merge all data
    if verbose:
        print("\n" + "="*80)
        print("MERGING ALL DATA")
        print("="*80)
    
    merge_keys = ['PMM_Program_Name', 'Fm_Reporting_Month']
    
    # Determine the base dataframe (start with whichever target exists)
    if cei_df is not None:
        merged = cei_df.copy()
        base_name = "CEI"
    elif bei_df is not None:
        merged = bei_df.copy()
        base_name = "BEI"
    else:
        # Should never happen due to validation above, but just in case
        raise ValueError("No target data loaded!")
    
    if verbose:
        print(f"\n📍 Using {base_name} as base (anchor for date range)")
    
    # Merge the other target if it exists
    if cei_df is not None and bei_df is not None:
        # Both exist - merge BEI into CEI
        merged = merged.merge(
            bei_df[merge_keys + ['bei_numerator', 'bei_denominator', 'BEI']],
            on=merge_keys,
            how='left'
        )
        if verbose:
            print(f"   ✅ Merged BEI with CEI")
    elif bei_df is not None and cei_df is not None:
        # Both exist (already handled above, but keep for clarity)
        pass
    elif cei_df is not None and bei_df is None:
        # Only CEI
        if verbose:
            print(f"   ℹ️  No BEI data to merge")
    elif bei_df is not None and cei_df is None:
        # Only BEI - but we need to add CEI columns as None
        merged['cei_hit'] = None
        merged['cei_miss'] = None
        merged['CEI'] = None
        if verbose:
            print(f"   ℹ️  No CEI data to merge")
    
    # Get the date range from target data (CEI/BEI)
    target_dates = set(merged['Fm_Reporting_Month'].dropna().unique())
    
    if verbose:
        print(f"\n📅 Target ({base_name}) Date Range:")
        print(f"   Start: {merged['Fm_Reporting_Month'].min()}")
        print(f"   End: {merged['Fm_Reporting_Month'].max()}")
        print(f"   Total months: {len(target_dates)}")
        print(f"   Sample dates: {list(target_dates)[:5]}")
    
    # Filter and merge engineering metrics if provided
    if metrics_df is not None:
        if filter_by_target_dates:
            # Filter engineering metrics to only target dates
            # Dates are already normalized to YYYY-MM format, so direct comparison works
            if verbose:
                metrics_sample_dates = metrics_df['Fm_Reporting_Month'].dropna().unique()[:5]
                print(f"\n🔍 Date Filtering:")
                print(f"   Target dates (normalized): {list(target_dates)[:3]}")
                print(f"   Engineering dates (normalized): {list(metrics_sample_dates)[:3]}")
            
            # Filter engineering metrics to only target dates
            metrics_df_filtered = metrics_df[metrics_df['Fm_Reporting_Month'].isin(target_dates)].copy()
            
            if verbose:
                print(f"\n🔍 Filtering engineering metrics to target date range:")
                print(f"   Before filter: {len(metrics_df)} records")
                print(f"   After filter: {len(metrics_df_filtered)} records")
                print(f"   Filtered out: {len(metrics_df) - len(metrics_df_filtered)} records")
            
            # If still 0 records, provide diagnostic help
            if len(metrics_df_filtered) == 0 and verbose:
                print(f"\n⚠️  WARNING: No date matches found even after normalization!")
                print(f"   This is unexpected. Please check your data.")
                print(f"\n   💡 WORKAROUND: Disable date filtering:")
                print(f"      load_all_data(..., filter_by_target_dates=False)")
            
            # Merge engineering metrics (only dates that match target)
            merged = merged.merge(metrics_df_filtered, on=merge_keys, how='left')
        else:
            # No filtering - use outer merge (original behavior)
            if verbose:
                print(f"\n🔍 Merging engineering metrics WITHOUT date filtering:")
                print(f"   Engineering metrics: {len(metrics_df)} records")
                print(f"   Note: Using outer merge (may have more missing data)")
            merged = merged.merge(metrics_df, on=merge_keys, how='outer')
    
    # Final cleaning pass for any remaining percentage columns
    merged = clean_percentage_columns(merged, verbose=False)
    
    if verbose:
        print(f"\n✅ Merged dataset: {len(merged)} records")
        print(f"   Programs: {merged['PMM_Program_Name'].nunique()}")
        print(f"   Date range: {merged['Fm_Reporting_Month'].min()} to {merged['Fm_Reporting_Month'].max()}")
        
        # Data completeness for key metrics
        print(f"\n📊 Data Completeness:")
        if 'CEI' in merged.columns:
            print(f"   CEI: {merged['CEI'].notna().sum()}/{len(merged)} ({merged['CEI'].notna().sum()/len(merged)*100:.1f}%)")
        if 'BEI' in merged.columns:
            print(f"   BEI: {merged['BEI'].notna().sum()}/{len(merged)} ({merged['BEI'].notna().sum()/len(merged)*100:.1f}%)")
        if '% On-Time to Forecast' in merged.columns:
            print(f"   % On-Time: {merged['% On-Time to Forecast'].notna().sum()}/{len(merged)} ({merged['% On-Time to Forecast'].notna().sum()/len(merged)*100:.1f}%)")
        
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
