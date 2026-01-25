from catalog_db import SqlServerConfig, connect
import pandas as pd

solve_cfg = SqlServerConfig(
    server="your_server",
    user="your_user",
    password="your_password",
    database="your_database"
)

print("="*80)
print("CHECKING RSN_FOR_CHG_CODE_MJR VALUES")
print("="*80)

# Get ALL unique values (not just preventable)
sql = """
SELECT 
    RSN_FOR_CHG_CODE_MJR, 
    COUNT(*) as count
FROM dbo.FISCAL_MONTH_CT_REPORTING
WHERE RSN_FOR_CHG_CODE_MJR IS NOT NULL
GROUP BY RSN_FOR_CHG_CODE_MJR
ORDER BY count DESC
"""

with connect(solve_cfg, as_dict=True, verbose=False) as cur:
    cur.execute(sql)
    rows = cur.fetchall() or []

if rows:
    df = pd.DataFrame(rows)
    print(f"\n✅ Found {len(df)} unique values:")
    display(df)
    
    # Look for values containing 'prevent'
    if 'RSN_FOR_CHG_CODE_MJR' in df.columns:
        matches = df[df['RSN_FOR_CHG_CODE_MJR'].str.lower().str.contains('prevent', na=False)]
        if not matches.empty:
            print(f"\n🎯 Values containing 'prevent':")
            display(matches)
        else:
            print("\n⚠️  No values contain 'prevent'")
            print("Here are ALL the values:")
            for val in df['RSN_FOR_CHG_CODE_MJR'].tolist()[:20]:  # Show first 20
                print(f"  - {val}")
else:
    print("⚠️  No data found - table might be empty or column name is different")
