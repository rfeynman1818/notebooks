def load_table_df(
    cfg: SqlServerConfig,
    *,
    schema: str,
    table: str,
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    if columns:
        col_sql = ", ".join(f"[{c}]" for c in columns)
    else:
        col_sql = "*"

    sql = f"SELECT {col_sql} FROM [{schema}].[{table}]"

    if where:
        sql += f" WHERE {where}"

    if limit is not None:
        sql = f"SELECT TOP ({limit}) {col_sql} FROM [{schema}].[{table}]"
        if where:
            sql += f" WHERE {where}"

    with connect(cfg, as_dict=True, verbose=verbose) as cur:
        cur.execute(sql)
        rows = cur.fetchall() or []

    if not rows:
        if verbose:
            print(f"\n⚠️  No rows returned from {schema}.{table}.")
        return pd.DataFrame(columns=columns or [])

    return pd.DataFrame(rows)

def load_table_chunks(
    cfg: SqlServerConfig,
    *,
    schema: str,
    table: str,
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    order_by: Optional[Sequence[str]] = None,
    chunksize: int = 100_000,
    verbose: bool = True,
):
    if chunksize <= 0:
        raise ValueError("chunksize must be > 0")

    col_sql = ", ".join(f"[{c}]" for c in columns) if columns else "*"
    base = f"SELECT {col_sql} FROM [{schema}].[{table}]"
    if where:
        base += f" WHERE {where}"

    if order_by and len(order_by) > 0:
        order_sql = ", ".join(f"[{c}]" for c in order_by)
    else:
        order_sql = "(SELECT NULL)"

    offset = 0
    first = True
    while True:
        sql = f"{base} ORDER BY {order_sql} OFFSET {offset} ROWS FETCH NEXT {chunksize} ROWS ONLY"
        with connect(cfg, as_dict=True, verbose=(verbose and first)) as cur:
            cur.execute(sql)
            rows = cur.fetchall() or []
        first = False

        if not rows:
            if offset == 0 and verbose:
                print(f"\n⚠️  No rows returned from {schema}.{table}.")
            break

        yield pd.DataFrame(rows)
        offset += chunksize


def load_table_df_chunked(
    cfg: SqlServerConfig,
    *,
    schema: str,
    table: str,
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    order_by: Optional[Sequence[str]] = None,
    chunksize: int = 100_000,
    max_chunks: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    dfs = []
    n = 0
    for chunk_df in load_table_chunks(
        cfg,
        schema=schema,
        table=table,
        columns=columns,
        where=where,
        order_by=order_by,
        chunksize=chunksize,
        verbose=verbose,
    ):
        dfs.append(chunk_df)
        n += 1
        if max_chunks is not None and n >= max_chunks:
            break

    if not dfs:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(dfs, ignore_index=True)
