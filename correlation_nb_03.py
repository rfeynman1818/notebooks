

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

# Import refactored CSV analysis functions
from correlation_analysis_csv_refactored import (
    load_cei_data,
    load_bei_data,
    load_merged_metrics_csv,
    load_separate_csvs,
    add_derived_metrics_csv,
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
print("CEI/BEI CORRELATION ANALYSIS")
print("="*80)
print("\nThis analysis examines correlations between:")
print("  • CEI (Customer Experience Index) - with Hit/Miss components")
print("  • BEI (Business Excellence Index) - with Numerator/Denominator")
print("  • 5 Engineering Reports (On-Time, Preventable, Design Errors, CT Releases)")
print("\nNote: SPI is NOT included in this analysis")
print("="*80)

# ==============================================================================
# CELL 2: Prepare CSV Files (Optional - Run Once)
# ==============================================================================

# Uncomment to create template CSV files
# create_csv_template(output_dir="csv_templates")
# print("✅ Template CSV files created!")
# print("   Check csv_templates/ folder for examples")

print("💡 This cell creates template CSV files showing the expected format")
print("   Uncomment the code above if you need examples")

# ==============================================================================
# CELL 3: Load Data from CSV Files
# ==============================================================================

print("="*80)
print("LOADING DATA FROM CSV FILES")
print("="*80)

# ────────────────────────────────────────────────────────────────────────────
# METHOD 1: Load separate CSV files (most common)
# ────────────────────────────────────────────────────────────────────────────

# Update these paths to your actual CSV file locations
complete_data = load_separate_csvs(
    cei_csv="path/to/cei_data.csv",              # ← UPDATE THIS PATH
    bei_csv="path/to/bei_data.csv",              # ← UPDATE THIS PATH
    ontime_csv="path/to/ontime_forecast.csv",    # ← UPDATE THIS PATH
    preventable_csv="path/to/preventable_revisions.csv",  # ← UPDATE THIS PATH
    design_error_csv="path/to/design_error_count.csv",    # ← UPDATE THIS PATH
    planned_ct_csv="path/to/planned_ct_releases.csv",     # ← UPDATE THIS PATH
    ct_releases_csv="path/to/ct_releases.csv",            # ← UPDATE THIS PATH
    verbose=True
)

# ────────────────────────────────────────────────────────────────────────────
# METHOD 2: Load single merged CSV (if you already merged everything)
# ────────────────────────────────────────────────────────────────────────────

# Uncomment if you have ONE merged CSV:
# complete_data = load_merged_metrics_csv(
#     csv_path="path/to/merged_metrics.csv",
#     verbose=True
# )

# ────────────────────────────────────────────────────────────────────────────
# Add derived metrics
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
    'unplanned_ct_releases',
    'design_error_rate'
]

# Add CEI/BEI components if available
if 'CEI_Hit' in complete_data.columns:
    required_metrics.extend(['CEI_Hit', 'CEI_Miss', 'CEI_Total', 'CEI_Hit_Rate'])
if 'BEI_Numerator' in complete_data.columns:
    required_metrics.extend(['BEI_Numerator', 'BEI_Denominator'])

print(f"\n✅ Data loaded: {len(complete_data)} records")
print(f"   Programs: {complete_data['PRGRM_NAME'].nunique()}")
print(f"   Date range: {complete_data['FM_REPORTING_MONTH'].min()} to {complete_data['FM_REPORTING_MONTH'].max()}")

print("\n📊 Data Completeness:")
for metric in required_metrics:
    if metric in complete_data.columns:
        non_null = complete_data[metric].notna().sum()
        pct = (non_null / len(complete_data)) * 100
        status = "✅" if pct > 80 else "⚠️" if pct > 50 else "❌"
        print(f"   {status} {metric}: {non_null}/{len(complete_data)} ({pct:.1f}%)")

# Preview data
print("\n📋 Data Preview:")
display(complete_data[['PRGRM_NAME', 'FM_REPORTING_MONTH', 'CEI', 'BEI', 
                       '% On-Time to Forecast', 'design_error_count', 'ct_releases']].head(10))

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
    save_path='scatterplot_matrix_cei_bei.png'
)

print("\n📈 What to look for:")
print("   - Linear patterns → Valid Pearson correlation")
print("   - Curved patterns → Non-linear relationship")
print("   - Clusters → Different groups/regimes")
print("   - Outliers → Unusual data points to investigate")

# ==============================================================================
# CELL 8: Summary Statistics
# ==============================================================================

print_summary_statistics(
    df=complete_data,
    corr_matrix=corr_matrix
)

# ==============================================================================
# CELL 9: CEI & BEI Focus Analysis
# ==============================================================================

print("="*80)
print("ANALYZING: CEI & BEI vs Engineering Metrics")
print("="*80)

# Focus on correlations between CEI/BEI and new metrics
existing_metrics = ['CEI', 'BEI']
new_metrics = [
    '% On-Time to Forecast',
    '% Preventable Revisions',
    'design_error_rate',
    'pct_planned'
]

# Add CEI/BEI component metrics if available
if 'CEI_Hit_Rate' in complete_data.columns:
    existing_metrics.append('CEI_Hit_Rate')
if 'BEI_Numerator' in complete_data.columns:
    existing_metrics.extend(['BEI_Numerator', 'BEI_Denominator'])

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
    
    if correlations:
        corr_df = pd.DataFrame(correlations)
        display(corr_df)

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("\n✅ Look for:")
print("   - Strong correlations (|r| > 0.7): Key predictive relationships")
print("   - Significant relationships (p < 0.05): Not due to chance")
print("   - Negative correlations: Inverse relationships")

# ==============================================================================
# CELL 10: CEI Deep Dive (If Components Available)
# ==============================================================================

if 'CEI_Hit' in complete_data.columns and 'CEI_Miss' in complete_data.columns:
    print("="*80)
    print("CEI COMPONENT ANALYSIS")
    print("="*80)
    
    # Analyze CEI components
    cei_metrics = ['CEI', 'CEI_Hit', 'CEI_Miss', 'CEI_Total', 'CEI_Hit_Rate']
    
    print("\n📊 CEI Metrics Descriptive Statistics:")
    display(complete_data[cei_metrics].describe())
    
    print("\n📈 Correlations between CEI components:")
    cei_corr = complete_data[cei_metrics].corr()
    display(cei_corr)
    
    # Visualize CEI Hit vs Miss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter: CEI Hit Rate vs Overall CEI
    if 'CEI_Hit_Rate' in complete_data.columns:
        plot_data = complete_data[['CEI_Hit_Rate', 'CEI']].dropna()
        if len(plot_data) > 0:
            axes[0].scatter(plot_data['CEI_Hit_Rate'], plot_data['CEI'], alpha=0.5, s=50)
            z = np.polyfit(plot_data['CEI_Hit_Rate'], plot_data['CEI'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(plot_data['CEI_Hit_Rate'].min(), plot_data['CEI_Hit_Rate'].max(), 100)
            axes[0].plot(x_line, p(x_line), "r-", linewidth=2)
            
            r = plot_data.corr().iloc[0, 1]
            axes[0].text(0.05, 0.95, f'r = {r:.3f}', transform=axes[0].transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            axes[0].set_xlabel('CEI Hit Rate (%)', fontsize=12)
            axes[0].set_ylabel('CEI', fontsize=12)
            axes[0].set_title('CEI Hit Rate vs CEI Value', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
    
    # Distribution of CEI
    axes[1].hist(complete_data['CEI'].dropna(), bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(complete_data['CEI'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].axvline(complete_data['CEI'].median(), color='green', linestyle='--', linewidth=2, label='Median')
    axes[1].set_xlabel('CEI', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('CEI Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cei_component_analysis.png', dpi=300)
    plt.show()
    
    print("💾 Saved: cei_component_analysis.png")

# ==============================================================================
# CELL 11: BEI Deep Dive (If Components Available)
# ==============================================================================

if 'BEI_Numerator' in complete_data.columns and 'BEI_Denominator' in complete_data.columns:
    print("="*80)
    print("BEI COMPONENT ANALYSIS")
    print("="*80)
    
    # Analyze BEI components
    bei_metrics = ['BEI', 'BEI_Numerator', 'BEI_Denominator']
    
    print("\n📊 BEI Metrics Descriptive Statistics:")
    display(complete_data[bei_metrics].describe())
    
    print("\n📈 Correlations between BEI components:")
    bei_corr = complete_data[bei_metrics].corr()
    display(bei_corr)
    
    # Visualize BEI components
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter: BEI Numerator vs BEI
    plot_data = complete_data[['BEI_Numerator', 'BEI']].dropna()
    if len(plot_data) > 0:
        axes[0].scatter(plot_data['BEI_Numerator'], plot_data['BEI'], alpha=0.5, s=50)
        z = np.polyfit(plot_data['BEI_Numerator'], plot_data['BEI'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(plot_data['BEI_Numerator'].min(), plot_data['BEI_Numerator'].max(), 100)
        axes[0].plot(x_line, p(x_line), "r-", linewidth=2)
        
        r = plot_data.corr().iloc[0, 1]
        axes[0].text(0.05, 0.95, f'r = {r:.3f}', transform=axes[0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[0].set_xlabel('BEI Numerator', fontsize=12)
        axes[0].set_ylabel('BEI', fontsize=12)
        axes[0].set_title('BEI Numerator vs BEI Value', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
    
    # Distribution of BEI
    axes[1].hist(complete_data['BEI'].dropna(), bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(complete_data['BEI'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].axvline(complete_data['BEI'].median(), color='green', linestyle='--', linewidth=2, label='Median')
    axes[1].set_xlabel('BEI', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('BEI Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bei_component_analysis.png', dpi=300)
    plt.show()
    
    print("💾 Saved: bei_component_analysis.png")

# ==============================================================================
# CELL 12: CEI vs BEI Relationship
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
            print("   → These metrics move together closely")
        else:
            print("   → Higher CEI → Lower BEI")
            print("   → Inverse relationship")
    elif abs(r) > 0.5:
        print(f"\n⚠️  MODERATE RELATIONSHIP")
        print("   → Some predictive value, but other factors also matter")
    else:
        print(f"\n❌ WEAK RELATIONSHIP")
        print("   → These metrics are mostly independent")
    
    print("💾 Saved: cei_vs_bei.png")
else:
    print("⚠️  Not enough data for this analysis")

# ==============================================================================
# CELL 13: Export All Results
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
    'data_source': 'CSV Files',
    'metrics_analyzed': 'CEI, BEI, On-Time, Preventable Revisions, Design Errors, CT Releases'
}

# Save summary
with open('analysis_summary_cei_bei.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("CEI/BEI CORRELATION ANALYSIS - SUMMARY\n")
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
print("   • complete_metrics_cei_bei.csv")
print("   • correlation_matrix_cei_bei.csv")
print("   • p_values_matrix_cei_bei.csv")
print("   • correlation_matrix_cei_bei.png")
print("   • scatterplot_matrix_cei_bei.png")
print("   • cei_component_analysis.png (if CEI components available)")
print("   • bei_component_analysis.png (if BEI components available)")
print("   • cei_vs_bei.png")
print("   • analysis_summary_cei_bei.txt")

print("\n🎉 Analysis complete!")

# ==============================================================================
# CELL 14: Key Findings and Recommendations
# ==============================================================================

print("="*80)
print("KEY FINDINGS AND RECOMMENDATIONS")
print("="*80)

print("""
Based on your CEI/BEI correlation analysis:

1️⃣  CEI & BEI RELATIONSHIP
   → Review the CEI vs BEI correlation
   → If strongly correlated: Consider if they measure similar aspects
   → If weakly correlated: They capture different dimensions (good!)

2️⃣  IDENTIFY LEADING INDICATORS
   → Which engineering metrics predict CEI/BEI?
   → Strong correlations suggest early warning signals
   → Use these for proactive management

3️⃣  LEVERAGE CEI COMPONENTS
   → If CEI Hit/Miss data available: Analyze what drives hits vs misses
   → Target specific improvement areas based on Miss patterns

4️⃣  LEVERAGE BEI COMPONENTS
   → If Numerator/Denominator available: Focus on which drives BEI
   → Improve numerator OR reduce denominator (whichever has more impact)

5️⃣  QUALITY METRICS IMPACT
   → Check correlation between % Preventable Revisions and CEI/BEI
   → Check correlation between Design Error Rate and CEI/BEI
   → Focus quality efforts on metrics with strongest negative impact

6️⃣  PLANNING EFFECTIVENESS
   → Check correlation between % Planned and CEI/BEI
   → If strong positive: Better planning → Better CEI/BEI
   → Invest in planning processes

7️⃣  MONITOR OVER TIME
   → Re-run this analysis monthly/quarterly
   → Track how correlations change
   → Alert if relationships break down

📊 Share your findings with stakeholders!
   Use the visualizations to communicate insights clearly.
""")

print("="*80)
print("Note: SPI was not included in this analysis")
print("If you need SPI analysis, use the original notebook version")
print("="*80)
