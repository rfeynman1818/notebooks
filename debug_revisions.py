# ===============================================================================
# DIAGNOSTIC SCRIPT: Check Preventable Revisions Data
# ===============================================================================

from catalog_db import SqlServerConfig, connect
import pandas as pd

# Configure your SQL Server connection
solve_cfg = SqlServerConfig(
    server="your_server_name",
    user="your_username",
    password="your_password",
    database="your_database_name"
)

# -------------------------------------------------------------------------------
# STEP 1: Check what values exist in RSN_FOR_CHG_CODE_MJR
# -------------------------------------------------------------------------------

print("="*80)
print("STEP 1: Checking unique values in RSN_FOR_CHG_CODE_MJR column")
print("="*80)

sql = """
SELECT DISTINCT RSN_FOR_CHG_CODE_MJR, COUNT(*) as count
FROM dbo.FISCAL_MONTH_CT_REPORTING
GROUP BY RSN_FOR_CHG_CODE_MJR
ORDER BY count DESC
"""

with connect(solve_cfg, as_dict=True, verbose=True) as cur:
    cur.execute(sql)
    rows = cur.fetchall() or []

df_values = pd.DataFrame(rows)
print("\nUnique values found in RSN_FOR_CHG_CODE_MJR:")
print(df_values)

# Check for the exact values we're looking for
preventable_found = df_values[df_values['RSN_FOR_CHG_CODE_MJR'] == 'Preventable']
non_preventable_found = df_values[df_values['RSN_FOR_CHG_CODE_MJR'] == 'Non-Preventable']

print("\n" + "="*80)
if not preventable_found.empty:
    print(f"✅ Found 'Preventable' with {preventable_found['count'].values[0]} records")
else:
    print("❌ Did NOT find exact match for 'Preventable'")
    # Check for case-insensitive matches
    preventable_ci = df_values[df_values['RSN_FOR_CHG_CODE_MJR'].str.lower() == 'preventable']
    if not preventable_ci.empty:
        print(f"   But found case variation: '{preventable_ci['RSN_FOR_CHG_CODE_MJR'].values[0]}'")

if not non_preventable_found.empty:
    print(f"✅ Found 'Non-Preventable' with {non_preventable_found['count'].values[0]} records")
else:
    print("❌ Did NOT find exact match for 'Non-Preventable'")
    # Check for case-insensitive matches
    non_preventable_ci = df_values[df_values['RSN_FOR_CHG_CODE_MJR'].str.lower() == 'non-preventable']
    if not non_preventable_ci.empty:
        print(f"   But found case variation: '{non_preventable_ci['RSN_FOR_CHG_CODE_MJR'].values[0]}'")

# -------------------------------------------------------------------------------
# STEP 2: Sample some records to see the actual data
# -------------------------------------------------------------------------------

print("\n" + "="*80)
print("STEP 2: Sample records")
print("="*80)

sql_sample = """
SELECT TOP 10
    CHG_ACTVTY_NUM,
    RSN_FOR_CHG_CODE_MJR,
    PRGRM_NAME,
    PMM_Program_ID,
    Fiscal_Release_Month
FROM dbo.FISCAL_MONTH_CT_REPORTING
WHERE RSN_FOR_CHG_CODE_MJR IS NOT NULL
ORDER BY CHG_ACTVTY_CRT_DATE DESC
"""

with connect(solve_cfg, as_dict=True, verbose=False) as cur:
    cur.execute(sql_sample)
    rows = cur.fetchall() or []

df_sample = pd.DataFrame(rows)
print("\nSample records:")
print(df_sample)

# -------------------------------------------------------------------------------
# STEP 3: Check for specific variations
# -------------------------------------------------------------------------------

print("\n" + "="*80)
print("STEP 3: Checking for common variations")
print("="*80)

variations_to_check = [
    "Preventable",
    "preventable",
    "PREVENTABLE",
    "Non-Preventable",
    "non-preventable", 
    "NON-PREVENTABLE",
    "Non Preventable",  # space instead of dash
    "NonPreventable",   # no separator
]

for variation in variations_to_check:
    matches = df_values[df_values['RSN_FOR_CHG_CODE_MJR'] == variation]
    if not matches.empty:
        print(f"  Found: '{variation}' ({matches['count'].values[0]} records)")

# -------------------------------------------------------------------------------
# STEP 4: Check if there's leading/trailing whitespace
# -------------------------------------------------------------------------------

print("\n" + "="*80)
print("STEP 4: Checking for whitespace issues")
print("="*80)

# Check for any values that contain 'preventable' but don't match exactly
contains_preventable = df_values[
    df_values['RSN_FOR_CHG_CODE_MJR'].str.lower().str.contains('preventable', na=False)
]

print("\nAll values containing 'preventable' (case-insensitive):")
for _, row in contains_preventable.iterrows():
    value = row['RSN_FOR_CHG_CODE_MJR']
    count = row['count']
    print(f"  '{value}' (length={len(value)}, count={count})")
    # Show if there's whitespace
    if value != value.strip():
        print(f"    ⚠️  Has whitespace! Stripped: '{value.strip()}'")

# -------------------------------------------------------------------------------
# STEP 5: Test the current implementation
# -------------------------------------------------------------------------------

print("\n" + "="*80)
print("STEP 5: Testing current implementation logic")
print("="*80)

sql_test = """
SELECT
    CHG_ACTVTY_NUM,
    RSN_FOR_CHG_CODE_MJR,
    PRGRM_NAME,
    PMM_Program_ID,
    Fiscal_Release_Month
FROM dbo.FISCAL_MONTH_CT_REPORTING
WHERE RSN_FOR_CHG_CODE_MJR IN ('Preventable', 'Non-Preventable')
"""

with connect(solve_cfg, as_dict=True, verbose=False) as cur:
    cur.execute(sql_test)
    rows = cur.fetchall() or []

df_test = pd.DataFrame(rows)
print(f"\nRecords matching 'Preventable' or 'Non-Preventable': {len(df_test)}")

if len(df_test) > 0:
    print("\nBreakdown by value:")
    print(df_test['RSN_FOR_CHG_CODE_MJR'].value_counts())
else:
    print("❌ No records found! This means the exact values don't match.")
    print("   Try using the actual values shown in STEP 1 above.")

# -------------------------------------------------------------------------------
# RECOMMENDATIONS
# -------------------------------------------------------------------------------

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if len(df_test) == 0:
    print("\n❌ No records found with exact 'Preventable' or 'Non-Preventable' values.")
    print("\nLook at STEP 1 output above and use the EXACT values you see there.")
    print("Update the report to use those exact values (case-sensitive).")
    
    # Try to suggest the correct values
    likely_preventable = df_values[
        df_values['RSN_FOR_CHG_CODE_MJR'].str.lower().str.contains('preventable', na=False) &
        ~df_values['RSN_FOR_CHG_CODE_MJR'].str.lower().str.contains('non', na=False)
    ]
    
    likely_non_preventable = df_values[
        df_values['RSN_FOR_CHG_CODE_MJR'].str.lower().str.contains('non.*preventable', na=False, regex=True)
    ]
    
    if not likely_preventable.empty:
        print(f"\nLikely 'Preventable' value: '{likely_preventable.iloc[0]['RSN_FOR_CHG_CODE_MJR']}'")
    
    if not likely_non_preventable.empty:
        print(f"Likely 'Non-Preventable' value: '{likely_non_preventable.iloc[0]['RSN_FOR_CHG_CODE_MJR']}'")
else:
    print(f"\n✅ Found {len(df_test)} matching records - implementation should work!")
    print("If you're still getting unexpected results, run the actual report and share the output.")
