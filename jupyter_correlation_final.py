# ===============================================================================
# JUPYTER NOTEBOOK: CEI/BEI Correlation Analysis (With Program Name Mapping)
# ===============================================================================

# ==============================================================================
# CELL 1: Setup and Imports
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Import modules
from correlation_analysis_csv_final import (
    load_separate_csvs,
    add_derived_metrics_csv,
    create_mapping_template,
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
print("CEI/BEI CORRELATION ANALYSIS")
print("With Program Name Mapping Support")
print("="*80)

# ==============================================================================
# CELL 2: Create Program Name Mapping (Run Once)
# ==============================================================================

# ────────────────────────────────────────────────────────────────────────────
# OPTION 1: Create a template to fill in
# ────────────────────────────────────────────────────────────────────────────

# Uncomment to create template:
# create_mapping_template("program_name_mapping.csv")
# print("✅ Template created! Fill in your actual program names.")

# ────────────────────────────────────────────────────────────────────────────
# OPTION 2: Check what program names you have in each dataset
# ────────────────────────────────────────────────────────────────────────────

# Uncomment to check programs:
# cei_bei = pd.read_csv("cei_bei_data.csv")
# ontime = pd.read_csv("ontime_forecast.csv")
# 
# print("CEI/BEI Programs:")
# for prog in sorted(cei_bei['Program Name'].unique()):
#     print(f"  - {prog}")
# 
# print("\nMetrics Programs:")
# for prog in sorted(ontime['PRGRM_NAME'].unique()):
#     print(f"  - {prog}")

print("💡 Uncomment the code above to:")
print("   - Create a mapping template")
print("   - Check program names in your data")

# ==============================================================================
# CELL 3: Load Data with Program Name Mapping
# ==============================================================================

print("="*80)
print("LOADING DATA")
print("="*80)

# ────────────────────────────────────────────────────────────────────────────
# STEP 1: Define your program name mapping
# ────────────────────────────────────────────────────────────────────────────

# METHOD A: Use a CSV file (Recommended)
program_mapping = "program_name_mapping.csv"

# METHOD B: Use a Python dictionary
# program_mapping = {
#     "Program Alpha": "ALPHA_PROGRAM",
#     "Program Beta": "BETA_PROGRAM",
#     "Program Gamma": "GAMMA_PROGRAM",
#     # Add all your programs here
# }

# METHOD C: No mapping needed (if names already match)
# program_mapping = None

# ────────────────────────────────────────────────────────────────────────────
# STEP 2: Load all CSV files
# ────────────────────────────────────────────────────────────────────────────

complete_data = load_separate_csvs(
    cei_bei_csv="cei_bei_data.csv",              # ← UPDATE THIS PATH
    ontime_csv="ontime_forecast.csv",            # ← UPDATE THIS PATH
    preventable_csv="preventable_revisions.csv", # ← UPDATE THIS PATH
    design_error_csv="design_error_count.csv",   # ← UPDATE THIS PATH
    planned_ct_csv="planned_ct_releases.csv",    # ← UPDATE THIS PATH
    ct_releases_csv="ct_releases.csv",           # ← UPDATE THIS PATH
    program_mapping=program_mapping,             # ← Your mapping!
    verbose=True
)

# ────────────────────────────────────────────────────────────────────────────
# STEP 3: Add derived metrics
# ────────────────────────────────────────────────────────────────────────────

complete_data = add_derived_metrics_csv(complete_data, verbose=True)

print("\n✅ Data loaded successfully!")
print(f"   Total records: {len(complete_data)}")
print(f"   Programs: {complete_data['PRGRM_NAME'].nunique()}")
print(f"   Date range: {complete_data['FM_REPORTING_MONTH'].min()} to {complete_data['FM_REPORTING_MONTH'].max()}")

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

# Check for required columns
required_metrics = [
    'CEI', 'BEI',
    '% On-Time to Forecast',
    '% Preventable Revisions',
    'design_error_count',
    'planned_ct_releases',
    'ct_releases',
    'pct_planned',
    'design_error_rate'
]

# Add CEI/BEI components if available
if 'cei_hit' in complete_data.columns:
    required_metrics.extend(['cei_hit', 'cei_miss', 'cei_total', 'cei_success_rate'])
if 'bei_numerator' in complete_data.columns:
    required_metrics.extend(['bei_numerator', 'bei_denominator'])

print(f"\n✅ Data loaded: {len(complete_data)} records")
print(f"   Programs: {complete_data['PRGRM_NAME'].nunique()}")

print("\n📊 Data Completeness:")
for metric in required_metrics:
    if metric in complete_data.columns:
        non_null = complete_data[metric].notna().sum()
        pct = (non_null / len(complete_data)) * 100
        status = "✅" if pct > 80 else "⚠️" if pct > 50 else "❌"
        print(f"   {status} {metric}: {non_null}/{len(complete_data)} ({pct:.1f}%)")

# Preview data
print("\n📋 Data Preview:")
display_cols = ['PRGRM_NAME', 'FM_REPORTING_MONTH', 'CEI', 'BEI', 
                '% On-Time to Forecast', 'design_error_count', 'ct_releases']
display(complete_data[display_cols].head(10))

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

# Save correlation results
corr_matrix.to_csv('correlation_matrix_cei_bei.csv')
p_values.to_csv('p_values_matrix_cei_bei.csv')
print("\n💾 Saved:")
print("   - correlation_matrix_cei_bei.csv")
print("   - p_values_matrix_cei_bei.csv")

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
    save_path='correlation_matrix_cei_bei.png'
)

print("\n📊 Interpretation Guide:")
print("   - Blue = Positive correlation (move together)")
print("   - Red = Negative correlation (move opposite)")
print("   - Stars = Significance (* p<0.05, ** p<0.01, *** p<0.001)")

# ==============================================================================
# CELL 7: CEI & BEI Focus Analysis
# ==============================================================================

print("="*80)
print("ANALYZING: CEI & BEI vs Engineering Metrics")
print("="*80)

existing_metrics = ['CEI', 'BEI']
new_metrics = [
    '% On-Time to Forecast',
    '% Preventable Revisions',
    'design_error_rate',
    'pct_planned'
]

# Add component metrics if available
if 'cei_success_rate' in complete_data.columns:
    existing_metrics.append('cei_success_rate')

print("\n📊 Correlation Summary:")
print("="*80)

for existing in existing_metrics:
    if existing not in corr_matrix.index:
        continue
        
    print(f"\n🎯 {existing} correlations:")
    print("-"*80)
    
    correlations = []
    for new in new_metrics:
        if new in corr_matrix.columns:
            r = corr_matrix.loc[existing, new]
            p = p_values.loc[existing, new]
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
                'Engineering Metric': new,
                'r': f"{r:.3f}",
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
    
    print(f"\n📊 Correlation between CEI and BEI: {r:.3f}")
    print(f"   Sample size: {len(plot_data)}")
    
    if abs(r) > 0.7:
        print(f"\n✅ STRONG RELATIONSHIP!")
        if r > 0:
            print("   → Higher CEI → Higher BEI")
        else:
            print("   → Higher CEI → Lower BEI")
    elif abs(r) > 0.5:
        print(f"\n⚠️  MODERATE RELATIONSHIP")
    else:
        print(f"\n❌ WEAK RELATIONSHIP")
        print("   → CEI and BEI are mostly independent")

# ==============================================================================
# CELL 9: Key Findings Summary
# ==============================================================================

print("="*80)
print("KEY FINDINGS")
print("="*80)

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

print("\n📊 Top 10 Strongest Correlations:")
print("="*80)
for i, (col1, col2, r, p) in enumerate(top_corr[:10], 1):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"\n{i}. {col1} ↔ {col2}")
    print(f"   r = {r:.3f} {sig} (p = {p:.4f})")

# ==============================================================================
# CELL 10: Export Results
# ==============================================================================

print("\n" + "="*80)
print("EXPORTING RESULTS")
print("="*80)

# Save summary
with open('analysis_summary_cei_bei.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("CEI/BEI CORRELATION ANALYSIS - SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Records: {len(complete_data)}\n")
    f.write(f"Programs: {complete_data['PRGRM_NAME'].nunique()}\n")
    f.write(f"Date Range: {complete_data['FM_REPORTING_MONTH'].min()} to {complete_data['FM_REPORTING_MONTH'].max()}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("TOP CORRELATIONS\n")
    f.write("="*80 + "\n\n")
    
    for i, (col1, col2, r, p) in enumerate(top_corr[:10], 1):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        f.write(f"{i}. {col1} ↔ {col2}\n")
        f.write(f"   r = {r:.3f} {sig} (p = {p:.4f})\n\n")

print("✅ Results exported!")
print("\n📁 Generated files:")
print("   • complete_metrics_cei_bei.csv")
print("   • correlation_matrix_cei_bei.csv")
print("   • p_values_matrix_cei_bei.csv")
print("   • correlation_matrix_cei_bei.png")
print("   • cei_vs_bei.png")
print("   • analysis_summary_cei_bei.txt")

print("\n🎉 Analysis complete!")

# ==============================================================================
# CELL 11: Recommendations
# ==============================================================================

print("="*80)
print("ACTIONABLE RECOMMENDATIONS")
print("="*80)

print("""
Based on your correlation analysis:

1️⃣  FOCUS ON HIGH-IMPACT METRICS
   → Look for strong negative correlations with CEI/BEI
   → Example: If design_error_rate ↔ CEI = -0.75, reducing errors will improve CEI
   → Prioritize improvements that affect both CEI and BEI

2️⃣  UNDERSTAND CEI vs BEI RELATIONSHIP
   → If highly correlated (r > 0.7): They measure similar aspects
   → If weakly correlated (r < 0.3): They capture different dimensions
   → Use this to understand what each metric really measures

3️⃣  IDENTIFY LEADING INDICATORS
   → Which engineering metrics predict future CEI/BEI?
   → Strong correlations suggest predictive relationships
   → Use these for early warning systems

4️⃣  QUALITY vs PLANNING vs TIMELINESS
   → Compare correlations of:
     • Quality metrics (% Preventable, Design Errors) with CEI/BEI
     • Planning metrics (% Planned) with CEI/BEI
     • Timeliness metrics (% On-Time) with CEI/BEI
   → Focus on the dimension with strongest impact

5️⃣  PROGRAM-LEVEL ANALYSIS
   → Re-run analysis for individual programs
   → Some programs may have different patterns
   → Tailor interventions to specific programs

6️⃣  MONITOR OVER TIME
   → Re-run this analysis monthly/quarterly
   → Track how correlations change
   → Alert if relationships break down

📊 Use the visualizations to communicate findings to stakeholders!
""")

print("="*80)
