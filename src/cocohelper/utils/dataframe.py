from pandas.core.util.hashing import hash_pandas_object
from typing import List, Optional, Tuple, Dict
from pandas import DataFrame
import warnings
from cocohelper.utils.colmapper import ColMap


def serialize_row(row):
    for idx in row.index:
        row[idx] = '{}'.format(row[idx])
    return row


def records_to_df(
        records: List[dict],
        id_col_mapper: Optional[ColMap] = None
) -> DataFrame:
    df = DataFrame.from_records(records)
    if id_col_mapper is not None and len(df) != 0:
        # df = df.rename(columns=id_col_mapper.to_new).set_index(id_col_mapper.new, drop=False)
        df = df.rename(columns=id_col_mapper.to_new).set_index(id_col_mapper.new)
    return df


def df_to_records(
        data_frame: DataFrame,
        id_col_mapper: Optional[ColMap] = None
) -> List[dict]:
    data_frame = data_frame.reset_index(drop=data_frame.index.name is None)
    if id_col_mapper is not None:
        data_frame = data_frame.rename(columns=id_col_mapper.to_orig)
    return data_frame.to_dict(orient='records')


def drop_duplicate_rows(
        df: DataFrame,
        ignore_columns: Optional[List[str]] = None
) -> Tuple[DataFrame, dict]:
    """Drop duplicates rows of a DataFrame and return a map of merged elements.

    Duplicate are defined as rows with the same values except the index. Some
    columns can be ignored at the end of identifying duplicates.

    Args:
        df: input DataFrame.
        ignore_columns: the columns to ignore for duplicates identification.

    Returns:
        - The DataFrame without duplicates.
        - A dict that maps indices of the dropped (merged) elements to the
          indices of the corresponding kept elements.
    """
    check_dup_cols = list(df.columns)

    if ignore_columns is not None:
        if not isinstance(ignore_columns, list):
            ignore_columns = list(ignore_columns)
        check_dup_cols = list(set(check_dup_cols) - set(ignore_columns))

    hashable_check_dup_cols = []
    for col in check_dup_cols:
        try:
            hash_pandas_object(df[col])
            hashable_check_dup_cols.append(col)
        except TypeError:
            warnings.warn(f"`drop_duplicate_rows` is trying to use column {col} to check duplications, "
                          f"but is not hashable and it will be skipped for this check.")

    if len(hashable_check_dup_cols) < 1:
        raise ValueError("There are no columns that can be used to check for duplicates.")

    index_name = df.index.name if df.index.name is not None else 'index'
    mapping = (df[hashable_check_dup_cols].reset_index()
               .groupby(hashable_check_dup_cols)[index_name]
               .agg(['first', tuple])
               .set_index('first')['tuple']
               .to_dict())

    # Reverse the mapping:
    mapping = {v: k for k, values in mapping.items() for v in values}

    return df.drop_duplicates(subset=check_dup_cols), mapping


def fix_fk_after_drop_duplicate(
        connected_df: DataFrame,
        fk_column: str,
        merge_index_mapping: Dict
) -> DataFrame:
    """Fix the foreign key of a dataframe connected to a dataframe with dropped
    duplicates.

    The foreign keys of connected_df that where pointing to indices that have
    been merged together should now point to the only instance of the duplicates
    that has been kept by the drop duplicate method.

    Args:
        connected_df: dataframe connected to a dataframe for which duplicates
            have been removed.
        fk_column: the column of connected_df that contains the foreign key that
            should be fixed.
        merge_index_mapping: a dict that maps the dropped keys to the key of the
            not-dropped duplicate row, e.g. if we merged rows with index (0, 1, 2)
            keeping only 0, and we merged rows with index (3, 4, 5) keeping only
            3, this map should be: {1: 0, 2: 0, 4: 3, 5: 3}.

    Returns:
        A copy of connected_df with fixed foreign key (values of fk_columns).
    """
    connected_df[fk_column] = connected_df[fk_column].map(merge_index_mapping).fillna(connected_df[fk_column])
    return connected_df
