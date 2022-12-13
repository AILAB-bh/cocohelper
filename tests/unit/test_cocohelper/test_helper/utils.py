from pandas import DataFrame


def df_equals(*dfs: DataFrame) -> bool:
    equality = None
    prev = None
    for df in dfs:
        if prev is None:
            prev = df
            equality = df == df
            continue
        else:
            equality = equality * (prev == df)
    return equality.all().all()
