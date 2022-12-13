from typing import Optional, Any, Tuple, Set
from pandas import DataFrame
from pandas.core.dtypes.inference import is_list_like
import pandas as pd


def filter_multi_rows_having_any(
        values: Optional[Any],
        column: str,
        df: pd.DataFrame
) -> DataFrame:
    if values is None:
        return df
    ids: Set[int] = set()
    for val in values:
        ids = ids.union(set(df[df[column] == val].index))
    return df.loc[ids]


def filter_rows_having(
        values: Optional[Any],
        column: str,
        df: pd.DataFrame
) -> DataFrame:
    if values is None:
        return df
    values = [values] if not is_list_like(values) else values
    return df[df[column].isin(values)]


def filter_multi_rows_having_all(
        values: Optional[Any],
        column: str,
        df: pd.DataFrame
) -> DataFrame:
    if values is None:
        return df
    values = [values] if not is_list_like(values) else values

    ids = set(df.index)
    for val in values:
        ids = ids.intersection(set(df[df[column] == val].index))
    return df.loc[ids]


def filter_rows_in_range(
        rng: Optional[Tuple[int, int]],
        column: str,
        df: pd.DataFrame,
        inclusive: str = "both"
) -> DataFrame:
    if inclusive is True or inclusive is False:
        inclusive = "both" if inclusive else "neither"

    if rng is None:
        return df
    else:
        if inclusive == "both":
            left, right = rng[0] <= df[column], df[column] <= rng[1]
        elif inclusive == "left":
            left, right = rng[0] <= df[column], df[column] < rng[1]
        elif inclusive == "right":
            left, right = rng[0] < df[column], df[column] <= rng[1]
        elif inclusive == "none":
            left, right = rng[0] < df[column], df[column] < rng[1]
        else:
            raise ValueError("Parameter `inclusive` has to be either string of 'both', 'left', 'right', or 'neither'.")
        return df[left * right]


def filter_rows_out_range(
        rng: Optional[Tuple[int, int]],
        column: str,
        df: pd.DataFrame,
        inclusive: str = "none"
) -> DataFrame:
    if inclusive is True or inclusive is False:
        inclusive = "both" if inclusive else "neither"

    if rng is None:
        return df
    else:
        if inclusive == "both":
            left, right = df[column] <= rng[0], rng[1] <= df[column]
        elif inclusive == "left":
            left, right = df[column] <= rng[0], rng[1] < df[column]
        elif inclusive == "right":
            left, right = df[column] < rng[0], rng[1] <= df[column]
        elif inclusive == "none" or inclusive == "neither":
            left, right = df[column] < rng[0], rng[1] < df[column]
        else:
            raise ValueError("Parameter `inclusive` has to be either string of 'both', 'left', 'right', or 'neither'.")
        return df[left + right]
