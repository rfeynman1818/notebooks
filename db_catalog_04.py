DEFAULT_VIEWS_SQL = """
SELECT
    s.name AS schema_name,
    v.name AS view_name,
    v.create_date
FROM sys.views v
JOIN sys.schemas s ON v.schema_id = s.schema_id
ORDER BY s.name, v.name;
"""

def list_views_df(
    cfg: SqlServerConfig,
    *,
    sql: str = DEFAULT_VIEWS_SQL,
    column_order: Optional[Sequence[str]] = (
        "schema_name",
        "view_name",
        "create_date",
    ),
    verbose: bool = True,
) -> pd.DataFrame:
    with connect(cfg, as_dict=True, verbose=verbose) as cur:
        cur.execute(sql)
        rows = cur.fetchall() or []

    if not rows:
        if verbose:
            print("\n⚠️  No views were returned – check permissions on this database.")
        return pd.DataFrame(columns=column_order or [])

    df = pd.DataFrame(rows)

    if column_order is not None:
        cols = [c for c in column_order if c in df.columns]
        rest = [c for c in df.columns if c not in cols]
        df = df[cols + rest]

    return df
