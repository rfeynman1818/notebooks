from catalog_db import SqlServerConfig, connect
import pandas as pd

solve_cfg = SqlServerConfig(
    server="your_server",
    user="your_user",
    password="your_password",
    database="your_database"
)

# Quick check - what values exist?
sql = """
SELECT DISTINCT RSN_FOR_CHG_CODE_MJR, COUNT(*) as count
FROM dbo.FISCAL_MONTH_CT_REPORTING
WHERE RSN_FOR_CHG_CODE_MJR LIKE '%preventable%'
GROUP BY RSN_FOR_CHG_CODE_MJR
ORDER BY count DESC
"""

with connect(solve_cfg, as_dict=True, verbose=False) as cur:
    cur.execute(sql)
    rows = cur.fetchall() or []

df = pd.DataFrame(rows)
print("Values containing 'preventable':")
display(df)

# Show if exact matches exist
print("\n" + "="*50)
if 'Preventable' in df['RSN_FOR_CHG_CODE_MJR'].values:
    print("✅ 'Preventable' exists")
else:
    print("❌ 'Preventable' NOT found")
    
if 'Non-Preventable' in df['RSN_FOR_CHG_CODE_MJR'].values:
    print("✅ 'Non-Preventable' exists")
else:
    print("❌ 'Non-Preventable' NOT found")
