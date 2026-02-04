# ==============================================================================
# CELL 1: Setup and Imports
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Import decimal module (converts percentages to 0-1 scale)
from correlation_analysis_csv_decimal import (
    load_all_data,
    add_derived_metrics,
    create_csv_templates,
    load_program_mapping
)

from correlation_analysis import (
    calculate_correlations,
    plot_correlation_matrix,
    plot_scatterplot_matrix,
    print_summary_statistics
)

print("✅ All imports successful!")
print("\n" + "="*80)
print("CEI/BEI CORRELATION ANALYSIS - FINAL VERSION")
print("Percentages converted to decimal scale (0-1): 100% = 1.0, 85.5% = 0.855")
print("="*80)
print("\nColumn names used (from your screenshots):")
print("  CEI/BEI:")
print("    - 'Program Name', 'Project Status Date'")
print("    - CEI: 'Hit (CEI)', 'Miss (CEI)', 'CEI Calc'")
print("    - BEI: 'BEI Numerator', 'BEI Denominator', 'BEI (Calc)'")
print("  Engineering Metrics:")
print("    - 'PMM Program Name', 'Fm Reporting Month'")
print("    - '% On-Time to Forecast'")
print("    - 'Preventable % of Total Revisions'")
print("    - 'RFC 2 Released CT Count'")
print("    - 'Planned Released CTs'")
print("    - 'Released CT Count'")
print("="*80)

# ==============================================================================
# CELL 2: Define Program Name Mapping (if needed)
# ==============================================================================

# If CEI/BEI program names differ from engineering metrics program names,
# define mapping here:

# OPTION 1: Use CSV file
program_mapping = "program_name_mapping.csv"

# OPTION 2: Use Python dictionary
# program_mapping = {
#     "Program Alpha (in CEI/BEI)": "ALPHA_PROGRAM",
#     "Program Beta (in CEI/BEI)": "BETA_PROGRAM",
#     "Program Gamma (in CEI/BEI)": "GAMMA_PROGRAM",
# }

# OPTION 3: No mapping needed (if names already match)
# program_mapping = None

print("💡 Program mapping configured")

# ==============================================================================
# CELL 3: Load All Data
# ==============================================================================

print("="*80)
print("LOADING ALL DATA")
print("="*80)

complete_data = load_all_data(
    cei_csv="cei_data.csv",                      # ← UPDATE THIS PATH
    bei_csv="bei_data.csv",                      # ← UPDATE THIS PATH
    ontime_csv="ontime_forecast.csv",            # ← UPDATE THIS PATH
    preventable_csv="preventable_revisions.csv", # ← UPDATE THIS PATH
    design_error_csv="design_error_count.csv",   # ← UPDATE THIS PATH
    planned_ct_csv="planned_ct_releases.csv",    # ← UPDATE THIS PATH
    ct_releases_csv="ct_releases.csv",           # ← UPDATE THIS PATH
    program_mapping=program_mapping,             # ← Your mapping
    verbose=True
)

# Add derived metrics
complete_data = add_derived_metrics(complete_data, verbose=True)

print("\n✅ Data loaded successfully!")
print(f"   Total records: {len(complete_data)}")
print(f"   Programs: {complete_data['PMM_Program_Name'].nunique()}")
print(f"   Date range: {complete_data['Fm_Reporting_Month'].min()} to {complete_data['Fm_Reporting_Month'].max()}")

display(complete_data.head(10))

# Save for backup
complete_data.to_csv('complete_metrics_cei_bei.csv', index=False)
print("\n💾 Saved to: complete_metrics_cei_bei.csv")

# ==============================================================================
# CELL 4: Data Quality Check
# ==============================================================================

print("="*80)
print("DATA QUALITY CHECK")
print("="*80)

metrics_to_check = [
    'CEI', 'cei_hit', 'cei_miss', 'cei_total', 'cei_success_rate',
    'BEI', 'bei_numerator', 'bei_denominator',
    '% On-Time to Forecast', 'pct_preventable_revisions',
    'design_error_count', 'planned_ct_releases', 'ct_releases',
    'pct_planned', 'unplanned_ct_releases', 'design_error_rate'
]

print(f"\n✅ Data loaded: {len(complete_data)} records")
print(f"   Programs: {complete_data['PMM_Program_Name'].nunique()}")

print("\n📊 Data Completeness:")
for metric in metrics_to_check:
    if metric in complete_data.columns:
        non_null = complete_data[metric].notna().sum()
        pct = (non_null / len(complete_data)) * 100
        status = "✅" if pct > 80 else "⚠️" if pct > 50 else "❌"
        print(f"   {status} {metric}: {non_null}/{len(complete_data)} ({pct:.1f}%)")

# Preview data
print("\n📋 Data Preview:")
preview_cols = ['PMM_Program_Name', 'Fm_Reporting_Month', 'CEI', 'BEI', 
                '% On-Time to Forecast', 'design_error_count', 'ct_releases']
display(complete_data[preview_cols].head(10))

# ==============================================================================
# CELL 5: Calculate Correlations
# ==============================================================================

print("="*80)
print("CALCULATING CORRELATIONS")
print("="*80)

corr_matrix, p_values = calculate_correlations(
    df=complete_data,
    method='pearson',
    min_observations=10,
    verbose=True
)

# Save results
corr_matrix.to_csv('correlation_matrix.csv')
p_values.to_csv('p_values_matrix.csv')
print("\n💾 Saved:")
print("   - correlation_matrix.csv")
print("   - p_values_matrix.csv")

# ==============================================================================
# CELL 6: Visualize Correlation Matrix
# ==============================================================================

print("="*80)
print("CORRELATION MATRIX VISUALIZATION")
print("="*80)

fig_corr = plot_correlation_matrix(
    corr_matrix=corr_matrix,
    p_values=p_values,
    figsize=(16, 14),
    save_path='correlation_matrix.png'
)

print("\n📊 Interpretation:")
print("   Blue = Positive correlation")
print("   Red = Negative correlation")
print("   Stars = Significance (* p<0.05, ** p<0.01, *** p<0.001)")

# ==============================================================================
# CELL 7: CEI & BEI vs Engineering Metrics
# ==============================================================================

print("="*80)
print("ANALYZING: CEI & BEI vs Engineering Metrics")
print("="*80)

target_metrics = ['CEI', 'BEI']
engineering_metrics = [
    '% On-Time to Forecast',
    'pct_preventable_revisions',
    'design_error_rate',
    'pct_planned'
]

print("\n📊 Correlation Summary:")
print("="*80)

for target in target_metrics:
    if target not in corr_matrix.index:
        continue
        
    print(f"\n🎯 {target} correlations:")
    print("-"*80)
    
    correlations = []
    for eng_metric in engineering_metrics:
        if eng_metric in corr_matrix.columns:
            r = corr_matrix.loc[target, eng_metric]
            p = p_values.loc[target, eng_metric]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            
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
                'Engineering Metric': eng_metric,
                'r': f"{r:.3f}",
                'p-value': f"{p:.4f}",
                'Sig': sig,
                'Interpretation': f"{strength} {direction}"
            })
    
    if correlations:
        corr_df = pd.DataFrame(correlations)
        display(corr_df)

# ==============================================================================
# CELL 8: CEI vs BEI Relationship
# ==============================================================================

print("="*80)
print("CEI vs BEI RELATIONSHIP")
print("="*80)

plot_data = complete_data[['CEI', 'BEI']].dropna()

if len(plot_data) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter with regression
    axes[0].scatter(plot_data['CEI'], plot_data['BEI'], alpha=0.5, s=50, color='steelblue')
    z = np.polyfit(plot_data['CEI'], plot_data['BEI'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(plot_data['CEI'].min(), plot_data['CEI'].max(), 100)
    axes[0].plot(x_line, p(x_line), "r-", linewidth=2, label='Regression')
    
    r = plot_data.corr().iloc[0, 1]
    axes[0].text(0.05, 0.95, f'r = {r:.3f}\nn = {len(plot_data)}',
                transform=axes[0].transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[0].set_xlabel('CEI', fontsize=12)
    axes[0].set_ylabel('BEI', fontsize=12)
    axes[0].set_title('Relationship: CEI vs BEI', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Density plot
    axes[1].hexbin(plot_data['CEI'], plot_data['BEI'], gridsize=20, cmap='Blues', alpha=0.8)
    axes[1].set_xlabel('CEI', fontsize=12)
    axes[1].set_ylabel('BEI', fontsize=12)
    axes[1].set_title('Density Plot', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cei_vs_bei.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Correlation: {r:.3f}")
    if abs(r) > 0.7:
        print("✅ STRONG RELATIONSHIP")
    elif abs(r) > 0.5:
        print("⚠️  MODERATE RELATIONSHIP")
    else:
        print("❌ WEAK RELATIONSHIP")

# ==============================================================================
# CELL 9: Key Findings
# ==============================================================================

print("="*80)
print("KEY FINDINGS")
print("="*80)

# Top 10 correlations
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

print("\n📊 Top 10 Strongest Correlations:")
for i, (col1, col2, r, p) in enumerate(top_corr[:10], 1):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"{i}. {col1} ↔ {col2}: r={r:.3f} {sig}")

# ==============================================================================
# CELL 10: Export Results
# ==============================================================================

print("\n" + "="*80)
print("EXPORTING RESULTS")
print("="*80)

# Save summary
with open('analysis_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("CEI/BEI CORRELATION ANALYSIS - SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Records: {len(complete_data)}\n")
    f.write(f"Programs: {complete_data['PMM_Program_Name'].nunique()}\n\n")
    f.write("TOP 10 CORRELATIONS:\n")
    for i, (col1, col2, r, p) in enumerate(top_corr[:10], 1):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        f.write(f"{i}. {col1} ↔ {col2}: r={r:.3f} {sig} (p={p:.4f})\n")

print("✅ Results exported!")
print("\n📁 Files generated:")
print("   • complete_metrics_cei_bei.csv")
print("   • correlation_matrix.csv")
print("   • p_values_matrix.csv")
print("   • correlation_matrix.png")
print("   • cei_vs_bei.png")
print("   • analysis_summary.txt")

print("\n🎉 Analysis complete!")
