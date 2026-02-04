import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catalog_db import SqlServerConfig
from scipy.stats import pearsonr, spearmanr
from typing import Tuple, Optional

# Import all report generators
from reports_on_time import run_on_time_report_from_query
from reports_preventable_revisions_mapped import run_preventable_revisions_report_from_query
from reports_design_error_count import run_design_error_count_report_from_query
from reports_planned_ct_releases import run_planned_ct_releases_report_from_query
from reports_ct_releases import run_ct_releases_report_from_query

# ===============================================================================
# STEP 1: LOAD ALL REPORTS
# ===============================================================================

def load_all_reports(
    cfg: SqlServerConfig,
    rsn_mapping: dict,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load all 5 reports and merge them into a single dataframe.
    
    Args:
        cfg: Database configuration
        rsn_mapping: Mapping for preventable revisions
        verbose: Print progress
        
    Returns:
        Merged dataframe with all metrics
    """
    if verbose:
        print("="*80)
        print("LOADING ALL REPORTS")
        print("="*80)
    
    # 1. On-Time to Forecast
    if verbose:
        print("\n📊 Loading On-Time to Forecast...")
    on_time_df, _, _ = run_on_time_report_from_query(
        cfg, 
        export_csv=False, 
        verbose=False
    )
    # Keep only needed columns, rename for clarity
    on_time_df = on_time_df[[
        'PMM_Program_Name', 'PMM_Program_ID', 'FM_REPORTING_MONTH', 
        '% On-Time to Forecast'
    ]].rename(columns={'PMM_Program_Name': 'PRGRM_NAME'})
    
    if verbose:
        print(f"   ✅ Loaded {len(on_time_df)} records")
    
    # 2. Preventable Revisions
    if verbose:
        print("\n📊 Loading Preventable Revisions...")
    prev_rev_df, _, _ = run_preventable_revisions_report_from_query(
        cfg,
        rsn_mapping=rsn_mapping,
        export_csv=False,
        verbose=False
    )
    # Keep only needed columns
    prev_rev_df = prev_rev_df[[
        'PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH',
        '% Preventable Revisions'
    ]]
    
    if verbose:
        print(f"   ✅ Loaded {len(prev_rev_df)} records")
    
    # 3. Design Error Count
    if verbose:
        print("\n📊 Loading Design Error Count...")
    design_error_df, _, _ = run_design_error_count_report_from_query(
        cfg,
        export_csv=False,
        verbose=False
    )
    # Keep only needed columns
    design_error_df = design_error_df[[
        'PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH',
        'design_error_count'
    ]]
    
    if verbose:
        print(f"   ✅ Loaded {len(design_error_df)} records")
    
    # 4. Planned CT Releases
    if verbose:
        print("\n📊 Loading Planned CT Releases...")
    planned_ct_df, _, _ = run_planned_ct_releases_report_from_query(
        cfg,
        export_csv=False,
        verbose=False
    )
    # Keep only needed columns
    planned_ct_df = planned_ct_df[[
        'PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH',
        'planned_ct_releases'
    ]]
    
    if verbose:
        print(f"   ✅ Loaded {len(planned_ct_df)} records")
    
    # 5. Total CT Releases
    if verbose:
        print("\n📊 Loading Total CT Releases...")
    total_ct_df, _, _ = run_ct_releases_report_from_query(
        cfg,
        export_csv=False,
        verbose=False
    )
    # Keep only needed columns
    total_ct_df = total_ct_df[[
        'PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH',
        'ct_releases'
    ]]
    
    if verbose:
        print(f"   ✅ Loaded {len(total_ct_df)} records")
    
    # Merge all dataframes
    if verbose:
        print("\n🔗 Merging all reports...")
    
    merge_keys = ['PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH']
    
    # Start with on-time as base
    merged = on_time_df.copy()
    
    # Merge preventable revisions
    merged = merged.merge(prev_rev_df, on=merge_keys, how='outer')
    
    # Merge design errors
    merged = merged.merge(design_error_df, on=merge_keys, how='outer')
    
    # Merge planned CT
    merged = merged.merge(planned_ct_df, on=merge_keys, how='outer')
    
    # Merge total CT
    merged = merged.merge(total_ct_df, on=merge_keys, how='outer')
    
    if verbose:
        print(f"   ✅ Merged dataset: {len(merged)} records")
        print(f"   ✅ Date range: {merged['FM_REPORTING_MONTH'].min()} to {merged['FM_REPORTING_MONTH'].max()}")
        print(f"   ✅ Programs: {merged['PRGRM_NAME'].nunique()}")
    
    return merged

# ===============================================================================
# STEP 2: ADD DERIVED METRICS
# ===============================================================================

def add_derived_metrics(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add calculated metrics derived from the base data.
    
    Derived metrics:
    - % Planned: (Planned CT / Total CT) × 100
    - Unplanned CT: Total CT - Planned CT
    - Design Error Rate: (Design Errors / Total CT) × 100
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
# STEP 3: ADD EXISTING METRICS (SPI, CEI, BEI)
# ===============================================================================

def add_existing_metrics(
    df: pd.DataFrame,
    existing_metrics_df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Merge existing engineering metrics (SPI, CEI, BEI) into the dataset.
    
    Args:
        df: Main dataframe with all reports
        existing_metrics_df: Dataframe with columns:
            - PRGRM_NAME or PMM_Program_ID
            - FM_REPORTING_MONTH
            - SPI (Schedule Performance Index)
            - CEI (Cost Efficiency Index) 
            - BEI (Budget Efficiency Index)
        verbose: Print progress
        
    Returns:
        Merged dataframe
    """
    if verbose:
        print("\n" + "="*80)
        print("ADDING EXISTING METRICS (SPI, CEI, BEI)")
        print("="*80)
    
    # Determine merge keys based on what's available
    if 'PMM_Program_ID' in existing_metrics_df.columns:
        merge_keys = ['PMM_Program_ID', 'FM_REPORTING_MONTH']
    else:
        merge_keys = ['PRGRM_NAME', 'FM_REPORTING_MONTH']
    
    merged = df.merge(
        existing_metrics_df,
        on=merge_keys,
        how='left'
    )
    
    if verbose:
        print(f"✅ Merged {len(merged)} records")
        
        # Check for missing data
        for metric in ['SPI', 'CEI', 'BEI']:
            if metric in merged.columns:
                missing = merged[metric].isna().sum()
                pct_missing = (missing / len(merged)) * 100
                print(f"   - {metric}: {missing} missing ({pct_missing:.1f}%)")
    
    return merged

# ===============================================================================
# STEP 4: CALCULATE CORRELATION MATRIX
# ===============================================================================

def calculate_correlations(
    df: pd.DataFrame,
    method: str = 'pearson',
    min_observations: int = 10,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate correlation matrix and p-values.
    
    Args:
        df: Dataframe with metrics
        method: 'pearson' or 'spearman'
        min_observations: Minimum number of valid pairs needed
        verbose: Print progress
        
    Returns:
        Tuple of (correlation_matrix, p_values_matrix)
    """
    if verbose:
        print("\n" + "="*80)
        print(f"CALCULATING {method.upper()} CORRELATIONS")
        print("="*80)
    
    # Select numeric columns for correlation
    metric_cols = [
        '% On-Time to Forecast',
        'pct_preventable_revisions',  # Changed from '% Preventable Revisions'
        'design_error_count',
        'planned_ct_releases',
        'ct_releases',
        'pct_planned',
        'unplanned_ct_releases',
        'design_error_rate'
    ]
    
    # Add CEI/BEI components if available
    if 'cei_hit' in df.columns:
        metric_cols.extend(['cei_hit', 'cei_miss', 'cei_total', 'cei_success_rate'])
    if 'bei_numerator' in df.columns:
        metric_cols.extend(['bei_numerator', 'bei_denominator'])
    
    # Add CEI, BEI if available
    if 'CEI' in df.columns:
        metric_cols.append('CEI')
    if 'BEI' in df.columns:
        metric_cols.append('BEI')
    
    # Keep only available columns
    metric_cols = [col for col in metric_cols if col in df.columns]
    
    if len(metric_cols) == 0:
        print("❌ ERROR: No valid metric columns found in data!")
        print(f"   Available columns: {list(df.columns)}")
        return pd.DataFrame(), pd.DataFrame()
    
    if len(metric_cols) < 2:
        print(f"❌ ERROR: Need at least 2 metrics for correlation!")
        print(f"   Found only: {metric_cols}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Create subset with only metric columns
    df_metrics = df[metric_cols].copy()
    
    if verbose:
        print(f"\nAnalyzing {len(metric_cols)} metrics:")
        for col in metric_cols:
            n_valid = df_metrics[col].notna().sum()
            print(f"   - {col}: {n_valid} valid observations")
    
    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = df_metrics.corr(method='pearson')
    else:
        corr_matrix = df_metrics.corr(method='spearman')
    
    # Calculate p-values
    p_values = pd.DataFrame(
        np.zeros_like(corr_matrix),
        index=corr_matrix.index,
        columns=corr_matrix.columns
    )
    
    for i, col1 in enumerate(metric_cols):
        for j, col2 in enumerate(metric_cols):
            if i != j:
                # Get valid pairs
                mask = df_metrics[[col1, col2]].notna().all(axis=1)
                valid_data = df_metrics.loc[mask, [col1, col2]]
                
                if len(valid_data) >= min_observations:
                    if method == 'pearson':
                        _, p_val = pearsonr(valid_data[col1], valid_data[col2])
                    else:
                        _, p_val = spearmanr(valid_data[col1], valid_data[col2])
                    p_values.iloc[i, j] = p_val
                else:
                    p_values.iloc[i, j] = np.nan
    
    if verbose:
        print("\n✅ Correlation matrix calculated")
        print(f"   Shape: {corr_matrix.shape}")
        
        # Identify strong correlations
        print("\n🔍 Strong correlations (|r| > 0.7, p < 0.05):")
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Upper triangle only
                    r = corr_matrix.iloc[i, j]
                    p = p_values.iloc[i, j]
                    if abs(r) > 0.7 and p < 0.05:
                        print(f"   {col1} ↔ {col2}: r={r:.3f}, p={p:.4f}")
    
    return corr_matrix, p_values

# ===============================================================================
# STEP 5: VISUALIZATION - CORRELATION MATRIX
# ===============================================================================

def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    p_values: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None
):
    """
    Create a heatmap of the correlation matrix with significance indicators.
    
    Args:
        corr_matrix: Correlation matrix
        p_values: P-values matrix
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    if verbose:
        print("\n" + "="*80)
        print("CREATING CORRELATION MATRIX VISUALIZATION")
        print("="*80)
    
    # Check if correlation matrix is empty
    if len(corr_matrix) == 0:
        print("❌ ERROR: Correlation matrix is empty!")
        print("   Make sure your data has numeric columns.")
        return None
    
    if len(corr_matrix) < 2:
        print("❌ ERROR: Need at least 2 metrics to create correlation matrix!")
        print(f"   Found only: {list(corr_matrix.columns)}")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient (r)'},
        ax=ax
    )
    
    # Add significance stars - with proper indexing for masked heatmap
    text_index = 0
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            if i > j:  # Lower triangle only (not masked)
                try:
                    p_val = p_values.iloc[i, j]
                    if not np.isnan(p_val):
                        if p_val < 0.001:
                            sig = '***'
                        elif p_val < 0.01:
                            sig = '**'
                        elif p_val < 0.05:
                            sig = '*'
                        else:
                            sig = ''
                        
                        if sig and text_index < len(ax.texts):
                            # Add star to existing annotation
                            text = ax.texts[text_index]
                            current_text = text.get_text()
                            text.set_text(f"{current_text}\n{sig}")
                    
                    text_index += 1
                except (IndexError, AttributeError):
                    # Skip if text annotation doesn't exist
                    pass
    
    plt.title('Correlation Matrix: Engineering Metrics\n* p<0.05, ** p<0.01, *** p<0.001', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")
    
    plt.show()
    
    return fig

# ===============================================================================
# STEP 6: VISUALIZATION - SCATTERPLOT MATRIX
# ===============================================================================

def plot_scatterplot_matrix(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 16),
    save_path: Optional[str] = None
):
    """
    Create a scatterplot matrix (pair plot) of all metrics.
    
    Args:
        df: Dataframe with metrics
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    print("\n" + "="*80)
    print("CREATING SCATTERPLOT MATRIX")
    print("="*80)
    
    # Select numeric columns
    metric_cols = [
        '% On-Time to Forecast',
        '% Preventable Revisions',
        'design_error_count',
        'planned_ct_releases',
        'ct_releases',
        'pct_planned',
        'design_error_rate'
    ]
    
    # Add SPI, CEI, BEI if available
    if 'SPI' in df.columns:
        metric_cols.append('SPI')
    if 'CEI' in df.columns:
        metric_cols.append('CEI')
    if 'BEI' in df.columns:
        metric_cols.append('BEI')
    
    # Keep only available columns
    metric_cols = [col for col in metric_cols if col in df.columns]
    
    # Create subset
    df_plot = df[metric_cols].copy()
    
    # Create pair plot
    g = sns.PairGrid(df_plot, diag_sharey=False, height=2.5)
    
    # Upper triangle: scatter with regression line
    g.map_upper(sns.scatterplot, alpha=0.5, s=20)
    g.map_upper(sns.regplot, scatter=False, color='red', line_kws={'linewidth': 1})
    
    # Diagonal: histogram
    g.map_diag(sns.histplot, kde=True, color='steelblue')
    
    # Lower triangle: scatter with density
    g.map_lower(sns.kdeplot, cmap='Blues', fill=True, alpha=0.5)
    g.map_lower(sns.scatterplot, alpha=0.3, s=10)
    
    # Add title
    g.fig.suptitle('Scatterplot Matrix: Engineering Metrics', 
                   fontsize=16, fontweight='bold', y=1.001)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")
    
    plt.show()
    
    return g

# ===============================================================================
# STEP 7: COMBINED VISUALIZATION
# ===============================================================================

def create_combined_plot(
    corr_matrix: pd.DataFrame,
    p_values: pd.DataFrame,
    df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Create a single figure with correlation matrix on top and key scatterplots below.
    
    Args:
        corr_matrix: Correlation matrix
        p_values: P-values matrix
        df: Dataframe with metrics
        save_path: Path to save figure (optional)
    """
    print("\n" + "="*80)
    print("CREATING COMBINED VISUALIZATION")
    print("="*80)
    
    # Create figure with gridspec
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Top: Correlation matrix
    ax_corr = fig.add_subplot(gs[0])
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient (r)'},
        ax=ax_corr
    )
    
    ax_corr.set_title('Correlation Matrix\n* p<0.05, ** p<0.01, *** p<0.001', 
                      fontsize=14, fontweight='bold', pad=10)
    ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, ha='right')
    
    # Bottom: Key scatterplots
    gs_scatter = gs[1].subgridspec(2, 3, wspace=0.3, hspace=0.3)
    
    # Define key relationships to plot
    relationships = [
        ('% On-Time to Forecast', 'SPI', 'On-Time vs SPI'),
        ('% Preventable Revisions', 'design_error_rate', 'Preventable vs Design Error Rate'),
        ('pct_planned', '% On-Time to Forecast', '% Planned vs On-Time'),
        ('design_error_count', 'ct_releases', 'Design Errors vs Total Releases'),
        ('planned_ct_releases', 'ct_releases', 'Planned vs Total Releases'),
        ('CEI', 'BEI', 'CEI vs BEI'),
    ]
    
    for idx, (x_col, y_col, title) in enumerate(relationships):
        if x_col in df.columns and y_col in df.columns:
            ax = fig.add_subplot(gs_scatter[idx // 3, idx % 3])
            
            # Remove NaN
            plot_data = df[[x_col, y_col]].dropna()
            
            if len(plot_data) > 0:
                # Scatter
                ax.scatter(plot_data[x_col], plot_data[y_col], alpha=0.5, s=30)
                
                # Regression line
                z = np.polyfit(plot_data[x_col], plot_data[y_col], 1)
                p = np.poly1d(z)
                ax.plot(plot_data[x_col], p(plot_data[x_col]), "r-", linewidth=2)
                
                # Calculate correlation
                r = plot_data.corr().iloc[0, 1]
                ax.text(0.05, 0.95, f'r = {r:.3f}', 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel(x_col, fontsize=9)
                ax.set_ylabel(y_col, fontsize=9)
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
    
    plt.suptitle('Engineering Metrics Correlation Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")
    
    plt.show()
    
    return fig

# ===============================================================================
# STEP 8: SUMMARY STATISTICS
# ===============================================================================

def print_summary_statistics(df: pd.DataFrame, corr_matrix: pd.DataFrame):
    """
    Print summary statistics and key findings.
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Select metric columns
    metric_cols = [col for col in df.columns if col not in 
                   ['PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH']]
    
    # Descriptive stats
    print("\nDescriptive Statistics:")
    print(df[metric_cols].describe().round(2))
    
    # Find strongest correlations
    print("\n" + "="*80)
    print("STRONGEST CORRELATIONS (|r| > 0.5)")
    print("="*80)
    
    correlations = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.5 and not np.isnan(r):
                    correlations.append((col1, col2, r))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for col1, col2, r in correlations:
        direction = "positive" if r > 0 else "negative"
        strength = "very strong" if abs(r) > 0.8 else "strong"
        print(f"\n{col1}")
        print(f"  ↔ {col2}")
        print(f"  r = {r:.3f} ({strength} {direction})")

print("\n✅ Correlation analysis module loaded successfully!")
