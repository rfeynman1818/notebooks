# ===============================================================================
# JUPYTER NOTEBOOK: Engineering Metrics Correlation Analysis (CSV-ENABLED)
# ===============================================================================
# 
# This notebook supports TWO ways to load data:
# 
# OPTION A: Upload CSV files (faster, no database needed)
# OPTION B: Query database directly (generates fresh data)
#
# Choose the option that works best for you!
# ===============================================================================

# ==============================================================================
# CELL 1: Setup and Imports
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Import CSV-based analysis functions
from correlation_analysis_csv import (
    load_merged_metrics_csv,
    load_separate_csvs,
    add_derived_metrics_csv,
    export_reports_to_csv,
    create_csv_template
)

# Import standard correlation analysis functions
from correlation_analysis import (
    calculate_correlations,
    plot_correlation_matrix,
    plot_scatterplot_matrix,
    create_combined_plot,
    print_summary_statistics
)

print("✅ All imports successful!")
print("\n" + "="*80)
print("CHOOSE YOUR DATA LOADING METHOD")
print("="*80)
print("\n📂 OPTION A: Load from CSV files (recommended if you have data)")
print("   → Skip database queries, work with pre-exported data")
print("   → Faster iteration, easier sharing")
print("   → Jump to CELL 3A")
print("\n🗄️  OPTION B: Query database directly")
print("   → Generate fresh data from database")
print("   → Ensures most up-to-date information")
print("   → Jump to CELL 3B")
print("\n💡 TIP: Run CELL 2 first to prepare/check your CSV files")
print("="*80)

# ==============================================================================
# CELL 2: Prepare CSV Files (Run this ONCE if using CSV option)
# ==============================================================================

# Choose ONE of these actions:

# ────────────────────────────────────────────────────────────────────────────
# ACTION 1: Create template CSV files (if you need examples)
# ────────────────────────────────────────────────────────────────────────────

# Uncomment to create templates:
# create_csv_template(output_dir="csv_templates")
# print("✅ Template CSV files created in ./csv_templates/")
# print("   Use these as examples for the correct format")

# ────────────────────────────────────────────────────────────────────────────
# ACTION 2: Export reports from database to CSV (one-time export)
# ────────────────────────────────────────────────────────────────────────────

# If you want to export fresh data from database to CSV files:

# from catalog_db import SqlServerConfig
# 
# solve_cfg = SqlServerConfig(
#     server="your_server",
#     user="your_user",
#     password="your_password",
#     database="your_database"
# )
# 
# # Define your RSN mapping
# rsn_mapping = {
#     "Value1": "Preventable",
#     "Value2": "Non-Preventable",
#     # ... add all your values
# }
# 
# # Export all reports to CSV
# exported_files = export_reports_to_csv(
#     cfg=solve_cfg,
#     rsn_mapping=rsn_mapping,
#     output_dir="exported_reports",
#     verbose=True
# )
# 
# print("\n✅ Reports exported! You can now use these CSV files.")

# ────────────────────────────────────────────────────────────────────────────
# ACTION 3: Check your CSV files
# ────────────────────────────────────────────────────────────────────────────

# Uncomment to check if your CSV files are valid:

# print("Checking CSV files...")
# 
# # Check individual files
# files_to_check = {
#     'Existing Metrics': 'path/to/existing_metrics.csv',
#     'On-Time': 'path/to/ontime.csv',
#     'Preventable': 'path/to/preventable.csv',
#     'Design Error': 'path/to/design_error.csv',
#     'Planned CT': 'path/to/planned_ct.csv',
#     'CT Releases': 'path/to/ct_releases.csv',
# }
# 
# for name, path in files_to_check.items():
#     try:
#         df = pd.read_csv(path)
#         print(f"✅ {name}: {len(df)} records, {len(df.columns)} columns")
#         print(f"   Columns: {list(df.columns)[:5]}...")
#     except FileNotFoundError:
#         print(f"❌ {name}: File not found at {path}")
#     except Exception as e:
#         print(f"⚠️  {name}: Error - {e}")

print("💡 Uncomment the action you want to perform above")

# ==============================================================================
# CELL 3A: OPTION A - Load Data from CSV Files
# ==============================================================================
# Use this if you have CSV files ready
# Skip to CELL 3B if you want to query database instead

print("="*80)
print("OPTION A: Loading Data from CSV Files")
print("="*80)

# ────────────────────────────────────────────────────────────────────────────
# METHOD 1: Load a single merged CSV file (easiest)
# ────────────────────────────────────────────────────────────────────────────

# If you have ONE CSV with all metrics already merged:

# complete_data = load_merged_metrics_csv(
#     csv_path="path/to/merged_metrics.csv",
#     verbose=True
# )

# ────────────────────────────────────────────────────────────────────────────
# METHOD 2: Load separate CSV files and merge them
# ────────────────────────────────────────────────────────────────────────────

# If you have separate CSV files for each report:

# complete_data = load_separate_csvs(
#     existing_metrics_csv="csv_data/existing_metrics.csv",
#     ontime_csv="csv_data/ontime_forecast.csv",
#     preventable_csv="csv_data/preventable_revisions.csv",
#     design_error_csv="csv_data/design_error_count.csv",
#     planned_ct_csv="csv_data/planned_ct_releases.csv",
#     ct_releases_csv="csv_data/ct_releases.csv",
#     verbose=True
# )

# ────────────────────────────────────────────────────────────────────────────
# Add derived metrics
# ────────────────────────────────────────────────────────────────────────────

# complete_data = add_derived_metrics_csv(complete_data, verbose=True)

# print("\n✅ Data loaded from CSV!")
# print(f"   Total records: {len(complete_data)}")
# print(f"   Programs: {complete_data['PRGRM_NAME'].nunique()}")
# display(complete_data.head(10))

# # Save for backup
# complete_data.to_csv('complete_metrics_from_csv.csv', index=False)
# print("💾 Saved to: complete_metrics_from_csv.csv")

print("💡 Uncomment the method you want to use above")
print("   Then skip to CELL 4 for correlation analysis")

# ==============================================================================
# CELL 3B: OPTION B - Query Database Directly
# ==============================================================================
# Use this if you want to pull fresh data from database
# Skip to CELL 4 if you used CELL 3A instead

print("="*80)
print("OPTION B: Querying Database")
print("="*80)

# from catalog_db import SqlServerConfig
# from correlation_analysis import (
#     load_all_reports,
#     add_derived_metrics,
#     add_existing_metrics
# )
# 
# # Database configuration
# solve_cfg = SqlServerConfig(
#     server="your_server",
#     user="your_user",
#     password="your_password",
#     database="your_database"
# )
# 
# # RSN mapping for Preventable Revisions
# rsn_mapping = {
#     # TODO: Add your 24 actual values
#     "Value1": "Preventable",
#     "Value2": "Non-Preventable",
#     "Value3": "Exclude",
#     # ... add all values
# }
# 
# # Load all reports from database
# print("Loading reports from database...")
# merged_data = load_all_reports(
#     cfg=solve_cfg,
#     rsn_mapping=rsn_mapping,
#     verbose=True
# )
# 
# # Add derived metrics
# merged_data = add_derived_metrics(merged_data, verbose=True)
# 
# # Load existing metrics (SPI, CEI, BEI)
# # METHOD 1: From database table
# from catalog_db import load_table_df
# existing_metrics = load_table_df(
#     solve_cfg,
#     schema="dbo",
#     table="YOUR_METRICS_TABLE",
#     columns=["PMM_Program_ID", "FM_REPORTING_MONTH", "SPI", "CEI", "BEI"]
# )
# 
# # METHOD 2: From CSV
# # existing_metrics = pd.read_csv("existing_metrics.csv")
# 
# # Combine with existing metrics
# complete_data = add_existing_metrics(
#     df=merged_data,
#     existing_metrics_df=existing_metrics,
#     verbose=True
# )
# 
# print("\n✅ Data loaded from database!")
# print(f"   Total records: {len(complete_data)}")
# print(f"   Programs: {complete_data['PRGRM_NAME'].nunique()}")
# display(complete_data.head(10))
# 
# # Save for future use
# complete_data.to_csv('complete_metrics_from_db.csv', index=False)
# print("💾 Saved to: complete_metrics_from_db.csv")

print("💡 Uncomment the code above to load from database")

# ==============================================================================
# CELL 4: Data Quality Check
# ==============================================================================

# Verify you have data loaded (from either CELL 3A or 3B)

try:
    print("="*80)
    print("DATA QUALITY CHECK")
    print("="*80)
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   Records: {len(complete_data)}")
    print(f"   Programs: {complete_data['PRGRM_NAME'].nunique()}")
    print(f"   Date range: {complete_data['FM_REPORTING_MONTH'].min()} to {complete_data['FM_REPORTING_MONTH'].max()}")
    
    # Check for required columns
    required_metrics = [
        'SPI', 'CEI', 'BEI',
        '% On-Time to Forecast',
        '% Preventable Revisions',
        'design_error_count',
        'planned_ct_releases',
        'ct_releases',
        'pct_planned',
        'unplanned_ct_releases',
        'design_error_rate'
    ]
    
    print("\n📊 Data Completeness:")
    for metric in required_metrics:
        if metric in complete_data.columns:
            non_null = complete_data[metric].notna().sum()
            pct = (non_null / len(complete_data)) * 100
            status = "✅" if pct > 80 else "⚠️" if pct > 50 else "❌"
            print(f"   {status} {metric}: {non_null}/{len(complete_data)} ({pct:.1f}%)")
        else:
            print(f"   ❌ {metric}: MISSING")
    
    # Preview data
    print("\n📋 Data Preview:")
    display(complete_data[['PRGRM_NAME', 'FM_REPORTING_MONTH', 'SPI', '% On-Time to Forecast', 
                           'design_error_count', 'ct_releases']].head(10))
    
except NameError:
    print("❌ No data loaded yet!")
    print("   Please run CELL 3A (CSV) or CELL 3B (Database) first")

# ==============================================================================
# CELL 5: Calculate Correlations
# ==============================================================================

print("="*80)
print("CALCULATING CORRELATIONS")
print("="*80)

# Calculate correlation matrix and p-values
corr_matrix, p_values = calculate_correlations(
    df=complete_data,
    method='pearson',  # or 'spearman' for non-linear
    min_observations=10,
    verbose=True
)

# Save correlation results
corr_matrix.to_csv('correlation_matrix.csv')
p_values.to_csv('p_values_matrix.csv')
print("\n💾 Saved:")
print("   - correlation_matrix.csv")
print("   - p_values_matrix.csv")

# ==============================================================================
# CELL 6: Visualize Correlation Matrix
# ==============================================================================

print("="*80)
print("CREATING CORRELATION MATRIX VISUALIZATION")
print("="*80)

fig_corr = plot_correlation_matrix(
    corr_matrix=corr_matrix,
    p_values=p_values,
    figsize=(14, 12),
    save_path='correlation_matrix.png'
)

print("\n📊 Interpretation Guide:")
print("   - Blue = Positive correlation (move together)")
print("   - Red = Negative correlation (move opposite)")
print("   - Intensity = Strength of relationship")
print("   - Stars = Significance (* p<0.05, ** p<0.01, *** p<0.001)")

# ==============================================================================
# CELL 7: Create Scatterplot Matrix
# ==============================================================================

print("="*80)
print("CREATING SCATTERPLOT MATRIX")
print("="*80)

fig_scatter = plot_scatterplot_matrix(
    df=complete_data,
    figsize=(16, 16),
    save_path='scatterplot_matrix.png'
)

print("\n📈 What to look for:")
print("   - Linear patterns → Valid Pearson correlation")
print("   - Curved patterns → Non-linear relationship")
print("   - Clusters → Different groups/regimes")
print("   - Outliers → Unusual data points to investigate")

# ==============================================================================
# CELL 8: Create Combined Visualization
# ==============================================================================

print("="*80)
print("CREATING COMBINED VISUALIZATION")
print("="*80)

fig_combined = create_combined_plot(
    corr_matrix=corr_matrix,
    p_values=p_values,
    df=complete_data,
    save_path='combined_analysis.png'
)

print("\n💡 This shows both correlation matrix and key scatterplots together")

# ==============================================================================
# CELL 9: Summary Statistics
# ==============================================================================

print_summary_statistics(
    df=complete_data,
    corr_matrix=corr_matrix
)

# ==============================================================================
# CELL 10: Deep Dive - Existing Metrics vs New Metrics
# ==============================================================================

print("="*80)
print("ANALYZING: Existing Metrics (SPI, CEI, BEI) vs New Metrics")
print("="*80)

# Focus on correlations between existing and new metrics
existing_metrics = ['SPI', 'CEI', 'BEI']
new_metrics = [
    '% On-Time to Forecast',
    '% Preventable Revisions',
    'design_error_rate',
    'pct_planned'
]

print("\n📊 Correlation Summary:")
print("="*80)

for existing in existing_metrics:
    print(f"\n🎯 {existing} correlations:")
    print("-"*80)
    
    correlations = []
    for new in new_metrics:
        if existing in corr_matrix.index and new in corr_matrix.columns:
            r = corr_matrix.loc[existing, new]
            p = p_values.loc[existing, new]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            
            # Interpretation
            if abs(r) > 0.7:
                strength = "Very Strong"
            elif abs(r) > 0.5:
                strength = "Strong"
            elif abs(r) > 0.3:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            direction = "positive" if r > 0 else "negative"
            
            correlations.append({
                'New Metric': new,
                'r': f"{r:.3f}",
                'Sig': sig,
                'Interpretation': f"{strength} {direction}"
            })
    
    corr_df = pd.DataFrame(correlations)
    display(corr_df)

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("\n✅ Look for:")
print("   - Strong correlations (|r| > 0.7): Key predictive relationships")
print("   - Significant relationships (p < 0.05): Not due to chance")
print("   - Negative correlations: Inverse relationships (one ↑, other ↓)")

# ==============================================================================
# CELL 11: Specific Relationship Analysis
# ==============================================================================

# Analyze a specific relationship in detail
# Example: SPI vs % On-Time to Forecast

print("="*80)
print("DETAILED ANALYSIS: SPI vs % On-Time to Forecast")
print("="*80)

# Filter valid data
plot_data = complete_data[['SPI', '% On-Time to Forecast']].dropna()

if len(plot_data) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter with regression
    axes[0].scatter(plot_data['SPI'], plot_data['% On-Time to Forecast'], 
                   alpha=0.5, s=50, color='steelblue')
    
    # Add regression line
    z = np.polyfit(plot_data['SPI'], plot_data['% On-Time to Forecast'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(plot_data['SPI'].min(), plot_data['SPI'].max(), 100)
    axes[0].plot(x_line, p(x_line), "r-", linewidth=2, label='Regression')
    
    # Correlation
    r = plot_data.corr().iloc[0, 1]
    axes[0].text(0.05, 0.95, f'r = {r:.3f}\nn = {len(plot_data)}',
                transform=axes[0].transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    axes[0].set_xlabel('SPI', fontsize=12)
    axes[0].set_ylabel('% On-Time to Forecast', fontsize=12)
    axes[0].set_title('Relationship: SPI vs On-Time %', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Density plot
    axes[1].hexbin(plot_data['SPI'], plot_data['% On-Time to Forecast'],
                  gridsize=20, cmap='Blues', alpha=0.8)
    axes[1].set_xlabel('SPI', fontsize=12)
    axes[1].set_ylabel('% On-Time to Forecast', fontsize=12)
    axes[1].set_title('Density Plot', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('spi_vs_ontime_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Correlation: {r:.3f}")
    print(f"   Sample size: {len(plot_data)}")
    
    if abs(r) > 0.7:
        print(f"\n✅ STRONG RELATIONSHIP!")
        if r > 0:
            print("   → Higher SPI → Higher On-Time %")
            print("   → These metrics move together closely")
        else:
            print("   → Higher SPI → Lower On-Time %")
            print("   → Inverse relationship")
    elif abs(r) > 0.5:
        print(f"\n⚠️  MODERATE RELATIONSHIP")
        print("   → Some predictive value, but other factors also matter")
    else:
        print(f"\n❌ WEAK RELATIONSHIP")
        print("   → These metrics are mostly independent")
else:
    print("⚠️  Not enough data for this analysis")

# ==============================================================================
# CELL 12: Export All Results
# ==============================================================================

print("="*80)
print("EXPORTING RESULTS")
print("="*80)

# Create results summary
results = {
    'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_records': len(complete_data),
    'n_programs': complete_data['PRGRM_NAME'].nunique(),
    'n_months': complete_data['FM_REPORTING_MONTH'].nunique(),
    'date_range': f"{complete_data['FM_REPORTING_MONTH'].min()} to {complete_data['FM_REPORTING_MONTH'].max()}",
    'data_source': 'CSV or Database (check above)',
}

# Save summary
with open('analysis_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("ENGINEERING METRICS CORRELATION ANALYSIS - SUMMARY\n")
    f.write("="*80 + "\n\n")
    for key, value in results.items():
        f.write(f"{key}: {value}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("TOP CORRELATIONS\n")
    f.write("="*80 + "\n\n")
    
    # Find top 10 correlations
    top_corr = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            r = corr_matrix.iloc[i, j]
            p = p_values.iloc[i, j]
            if not np.isnan(r):
                top_corr.append((col1, col2, r, p))
    
    top_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for i, (col1, col2, r, p) in enumerate(top_corr[:10], 1):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        f.write(f"{i}. {col1} ↔ {col2}\n")
        f.write(f"   r = {r:.3f} {sig} (p = {p:.4f})\n\n")

print("✅ Results exported!")
print("\n📁 Generated files:")
print("   • complete_metrics_from_csv.csv (or _from_db.csv)")
print("   • correlation_matrix.csv")
print("   • p_values_matrix.csv")
print("   • correlation_matrix.png")
print("   • scatterplot_matrix.png")
print("   • combined_analysis.png")
print("   • spi_vs_ontime_detailed.png")
print("   • analysis_summary.txt")

print("\n🎉 Analysis complete!")

# ==============================================================================
# CELL 13: Next Steps and Recommendations
# ==============================================================================

print("="*80)
print("NEXT STEPS AND RECOMMENDATIONS")
print("="*80)

print("""
Based on your correlation analysis, consider:

1️⃣  VALIDATE STRONG CORRELATIONS
   → Discuss surprising relationships with subject matter experts
   → Verify data quality for unexpected correlations
   → Consider potential confounding variables

2️⃣  IDENTIFY LEADING INDICATORS
   → Strong correlations suggest predictive relationships
   → Use new metrics to forecast existing targets (SPI, CEI, BEI)
   → Set up early warning systems

3️⃣  STREAMLINE METRICS
   → Very high correlations (r > 0.9) suggest redundancy
   → Consider using one metric instead of multiple similar ones
   → Focus effort on unique/independent metrics

4️⃣  DESIGN INTERVENTIONS
   → Target metrics with strong negative correlations
   → Example: If Design Errors ↔ SPI = -0.8, focus on design quality
   → Prioritize modifiable factors with high impact

5️⃣  MONITOR OVER TIME
   → Track how correlations change month-to-month
   → Alert if relationships break down (process changes?)
   → Use time-lagged correlations to find causal patterns

6️⃣  BUILD PREDICTIVE MODELS
   → Use strong correlations for forecasting
   → Develop regression models for key targets
   → Create composite metrics if appropriate

📊 Share your findings with stakeholders!
   Use the visualizations to communicate insights clearly.
""")

print("="*80)
