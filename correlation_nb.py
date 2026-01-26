# ==============================================================================
# CELL 1: Setup
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

from catalog_db import SqlServerConfig

solve_cfg = SqlServerConfig(
    server="your_server",
    user="your_user",
    password="your_password",
    database="your_database"
)

from correlation_analysis import (
    load_all_reports,
    add_derived_metrics,
    add_existing_metrics,
    calculate_correlations,
    plot_correlation_matrix,
    plot_scatterplot_matrix,
    create_combined_plot,
    print_summary_statistics
)

print("✅ Setup complete!")

# ==============================================================================
# CELL 2: Define RSN Mapping
# ==============================================================================

# REPLACE WITH YOUR ACTUAL 24 VALUES
rsn_mapping = {
    # Preventable
    "Value1": "Preventable",
    "Value2": "Preventable",
    
    # Non-Preventable
    "Value10": "Non-Preventable",
    "Value11": "Non-Preventable",
    
    # Exclude
    "Value20": "Exclude",
    
    # ... add all 24 values
}

print(f"✅ Mapping defined: {len(rsn_mapping)} values")

# ==============================================================================
# CELL 3: Load Existing Metrics (SPI, CEI, BEI)
# ==============================================================================

# Option A: From CSV
# existing_metrics = pd.read_csv("existing_metrics.csv")

# Option B: From database
from catalog_db import load_table_df
existing_metrics = load_table_df(
    solve_cfg,
    schema="dbo",
    table="YOUR_METRICS_TABLE",  # CHANGE THIS
    columns=["PMM_Program_ID", "FM_REPORTING_MONTH", "SPI", "CEI", "BEI"]
)

# Option C: Generate sample data (for testing only)
# from catalog_db import connect
# sql = """
# SELECT DISTINCT PMM_Program_ID, PRGRM_NAME, FM_REPORTING_MONTH
# FROM dbo.FISCAL_MONTH_CT_REPORTING
# WHERE PMM_Program_ID IS NOT NULL
# """
# with connect(solve_cfg, as_dict=True, verbose=False) as cur:
#     cur.execute(sql)
#     rows = cur.fetchall()
# existing_metrics = pd.DataFrame(rows)
# np.random.seed(42)
# existing_metrics['SPI'] = np.random.uniform(0.8, 1.2, len(existing_metrics))
# existing_metrics['CEI'] = np.random.uniform(0.85, 1.15, len(existing_metrics))
# existing_metrics['BEI'] = np.random.uniform(0.9, 1.1, len(existing_metrics))

print(f"✅ Loaded {len(existing_metrics)} records")
display(existing_metrics.head())

# ==============================================================================
# CELL 4: Load All Reports
# ==============================================================================

print("Loading all reports...")

merged_data = load_all_reports(
    cfg=solve_cfg,
    rsn_mapping=rsn_mapping,
    verbose=True
)

print(f"\n✅ Loaded {len(merged_data)} records")
display(merged_data.head())

# ==============================================================================
# CELL 5: Add Derived Metrics
# ==============================================================================

merged_data = add_derived_metrics(merged_data, verbose=True)

print("\nMetrics available:")
for col in merged_data.columns:
    if col not in ['PRGRM_NAME', 'PMM_Program_ID', 'FM_REPORTING_MONTH']:
        print(f"   - {col}")

# ==============================================================================
# CELL 6: Combine with Existing Metrics
# ==============================================================================

complete_data = add_existing_metrics(
    df=merged_data,
    existing_metrics_df=existing_metrics,
    verbose=True
)

print(f"\n✅ Complete dataset: {len(complete_data)} records")
display(complete_data.head(10))

# Save
complete_data.to_csv('complete_metrics_dataset.csv', index=False)
print("💾 Saved to: complete_metrics_dataset.csv")

# ==============================================================================
# CELL 7: Calculate Correlations
# ==============================================================================

corr_matrix, p_values = calculate_correlations(
    df=complete_data,
    method='pearson',
    min_observations=10,
    verbose=True
)

# Save
corr_matrix.to_csv('correlation_matrix.csv')
p_values.to_csv('p_values_matrix.csv')
print("\n💾 Saved correlation matrices")

# ==============================================================================
# CELL 8: Visualize Correlation Matrix
# ==============================================================================

fig_corr = plot_correlation_matrix(
    corr_matrix=corr_matrix,
    p_values=p_values,
    figsize=(14, 12),
    save_path='correlation_matrix.png'
)

# ==============================================================================
# CELL 9: Visualize Scatterplot Matrix
# ==============================================================================

fig_scatter = plot_scatterplot_matrix(
    df=complete_data,
    figsize=(16, 16),
    save_path='scatterplot_matrix.png'
)

# ==============================================================================
# CELL 10: Create Combined Visualization
# ==============================================================================

fig_combined = create_combined_plot(
    corr_matrix=corr_matrix,
    p_values=p_values,
    df=complete_data,
    save_path='combined_analysis.png'
)

# ==============================================================================
# CELL 11: Summary Statistics
# ==============================================================================

print_summary_statistics(
    df=complete_data,
    corr_matrix=corr_matrix
)

# ==============================================================================
# CELL 12: Deep Dive - Specific Relationships
# ==============================================================================

# Example: Analyze On-Time vs SPI in detail
plot_data = complete_data[['% On-Time to Forecast', 'SPI']].dropna()

if len(plot_data) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter with regression
    axes[0].scatter(plot_data['SPI'], plot_data['% On-Time to Forecast'], alpha=0.5)
    sns.regplot(x='SPI', y='% On-Time to Forecast', data=plot_data, 
                ax=axes[0], scatter=False, color='red')
    axes[0].set_title('On-Time Performance vs SPI')
    axes[0].set_xlabel('SPI (Schedule Performance Index)')
    axes[0].set_ylabel('% On-Time to Forecast')
    axes[0].grid(True, alpha=0.3)
    
    r = plot_data.corr().iloc[0, 1]
    axes[0].text(0.05, 0.95, f'r = {r:.3f}', transform=axes[0].transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Density plot
    axes[1].scatter(plot_data['SPI'], plot_data['% On-Time to Forecast'], 
                   alpha=0.2, s=20)
    sns.kdeplot(x='SPI', y='% On-Time to Forecast', data=plot_data, 
               ax=axes[1], levels=5, color='blue', linewidths=2)
    axes[1].set_title('Density: On-Time vs SPI')
    
    plt.tight_layout()
    plt.savefig('ontime_vs_spi.png', dpi=300)
    plt.show()
    
    print(f"Correlation: {r:.3f}")
    print(f"Sample size: {len(plot_data)}")
else:
    print("⚠️  Not enough data for this analysis")

# ==============================================================================
# CELL 13: Key Findings Summary
# ==============================================================================

# Create findings report
findings = []

print("="*80)
print("KEY FINDINGS")
print("="*80)

# Find strongest correlations
for i, col1 in enumerate(corr_matrix.columns):
    for j, col2 in enumerate(corr_matrix.columns):
        if i < j:
            r = corr_matrix.iloc[i, j]
            p = p_values.iloc[i, j]
            if abs(r) > 0.5 and p < 0.05:
                direction = "positive" if r > 0 else "negative"
                strength = "very strong" if abs(r) > 0.7 else "strong"
                findings.append({
                    'Metric 1': col1,
                    'Metric 2': col2,
                    'Correlation': r,
                    'P-value': p,
                    'Description': f"{strength} {direction}"
                })

findings_df = pd.DataFrame(findings).sort_values('Correlation', 
                                                  key=abs, 
                                                  ascending=False)

print("\nTop correlations:")
display(findings_df)

# Save findings
findings_df.to_csv('key_findings.csv', index=False)
print("\n💾 Saved to: key_findings.csv")

# ==============================================================================
# CELL 14: Export Everything
# ==============================================================================

print("\n" + "="*80)
print("FILES GENERATED")
print("="*80)

files = [
    'complete_metrics_dataset.csv',
    'correlation_matrix.csv',
    'p_values_matrix.csv',
    'correlation_matrix.png',
    'scatterplot_matrix.png',
    'combined_analysis.png',
    'ontime_vs_spi.png',
    'key_findings.csv'
]

for f in files:
    print(f"   ✅ {f}")

print("\n🎉 Analysis complete!")
