"""Filtering strategies, as classes.
"""
from __future__ import annotations  # for sure needed for python <= 3.7, don't know about python 3.8+
from typing import TYPE_CHECKING, Tuple
from pandas import DataFrame
import abc
from abc import ABC

from cocohelper.filters.strategies.functional import filter_multi_rows_having_any, filter_multi_rows_having_all, \
    filter_rows_in_range, \
    filter_rows_out_range, filter_rows_having


if TYPE_CHECKING:
    from cocohelper.filters.filter import ValueFilter, RangeFilter


# VALUE FILTER STRATEGY
class ValueFilterStrategy(ABC):
    """Strategy to filter rows in dataframe depending on a set of values."""

    @abc.abstractmethod
    def apply(
            self,
            values, column: str,
            df: DataFrame
    ):
        pass

    def __call__(
            self,
            fltr: ValueFilter
    ) -> ValueFilter:
        fltr.set_strategy(self)
        return fltr


class HavingValueFilterStrategy(ValueFilterStrategy):
    """ValueStrategy for filtering, keeping all the rows with one of the imposed
    values on the selected column."""

    def apply(
            self,
            values,
            column: str,
            df: DataFrame
    ) -> DataFrame:
        return filter_rows_having(values, column, df)


class AnyValueFilterStrategy(ValueFilterStrategy):
    """Filters multi-rows having at least one of the requested values.

    This strategy can be applied to dataframes that can contain multiple rows
    sharing the same index.

    In that case we consider each index associated with a list of values on each
    column; and if none of the values associated with a certain index are one
    of the values imposed by the filter all the rows with that index are
    removed, otherwise all the rows with that index are kept.

    To visualize easier, can consider each unique index as a multi-row
    containing a list of values on each column.
    In this case, a multi-row is excluded if the list of values in the selected
    column does not contain any the values imposed by the filter.


    For example, given the dataframe df:
    ```
    ----------------------
            A       B
    index
      0     0.0     0.0
      0     0.0     2.0
      1     1.0     0.0
      1     1.0     1.0
      2     2.0     0.0
      2     2.0     1.0
      2     2.0     2.0
    ---------------------
    ```

    Is considered a multi-row dataframe like this:
    ```
    -----------------------------------------
            A               B
    index
      0     [0.0, 0.0]      [0.0, 2.0]
      1     [1.0, 1.0]      [0.0, 1.0]
      2     [2.0, 2.0, 2.0] [0.0, 1.0, 2.0]
    -----------------------------------------
    ```

    If we apply `AnyValueFilterStrategy().apply([2.0, 3.0], 'B', df)`, we will
    remove the index 1 because column B does not contain any of the imposed
    values (2.0 and 3.0).
    At the same time we will keep indices 0 and 2, because they have at least
    one of the values imposed by the filter (`2.0` was sufficient in this case).

    Resulting multi-row dataframe will be:
    ```
    -----------------------------------------
            A               B
    index
      0     [0.0, 0.0]      [0.0, 2.0]
      2     [2.0, 2.0, 2.0] [0.0, 1.0, 2.0]
    -----------------------------------------
    ```

    Going back to a standard dataframe, this is the result:
    ```
    ----------------------
            A       B
    index
      0     0.0     0.0
      0     0.0     2.0
      2     2.0     0.0
      2     2.0     1.0
      2     2.0     2.0
    ----------------------
    """

    def apply(
            self,
            values,
            column: str,
            df: DataFrame
    ) -> DataFrame:
        return filter_multi_rows_having_any(values, column, df)


class AllValueFilterStrategy(ValueFilterStrategy):
    """Filters multi-rows having at least all the requested values.

    This strategy can be applied to dataframes that can contain multiple rows
    sharing the same index.

    In that case we consider each index associated with a list of values on each
    column; and if at least one of the values imposed by the filter are not
    associated with a certain index idx, all the rows with that index are
    removed from the dataframe.

    To visualize easier, can consider each unique index as a multi-row
    containing a list of values on each column.
    In this case, a multi-row is excluded if the list of values in the selected
    column does not contain all the values imposed by the filter.

    For example, the dataframe:
    ```
    ----------------------
            A       B
    index
      0     0.0     0.0
      0     0.0     2.0
      1     1.0     0.0
      1     1.0     1.0
      2     2.0     0.0
      2     2.0     1.0
      2     2.0     2.0
    ---------------------
    ```

    Is considered a multi-row dataframe like this:
    ```
    -----------------------------------------
            A               B
    index
      0     [0.0, 0.0]      [0.0, 2.0]
      1     [1.0, 1.0]      [0.0, 1.0]
      2     [2.0, 2.0, 2.0] [0.0, 1.0, 2.0]
    -----------------------------------------
    ```

    If we apply `AllValueFilterStrategy().apply([0.0, 1.0], 'B', df)`, we will
    remove the index 0 because column B does not contain one of the imposed
    values: 1.0.
    At the same time we will keep indices 1 and 2, because they have both
    values `0.0` and `1.0` in the column B, so we will have:
    ```
    -----------------------------------------
            A               B
    index
      1     [1.0, 1.0]      [0.0, 1.0]
      2     [2.0, 2.0, 2.0] [0.0, 1.0, 2.0]
    -----------------------------------------
    ```

    Going back to a standard dataframe, this is the result:
    ```
    ----------------------
            A       B
    index
      1     1.0     0.0
      1     1.0     1.0
      2     2.0     0.0
      2     2.0     1.0
      2     2.0     2.0
    ----------------------
    ```
    """

    def apply(
            self,
            values,
            column: str,
            df: DataFrame
    ) -> DataFrame:
        return filter_multi_rows_having_all(values=values, column=column, df=df)


# RANGE FILTER STRATEGY
class RangeFilterStrategy(ABC):
    """Strategy to filter rows in dataframe depending on a range of values."""

    @abc.abstractmethod
    def apply(
            self,
            values,
            column: str,
            df: DataFrame
    ) -> DataFrame:
        pass

    def __call__(
            self,
            fltr: RangeFilter
    ) -> RangeFilter:
        fltr.set_strategy(self)
        return fltr


class InRangeFilterStrategy(RangeFilterStrategy):
    """Strategy to filter rows in dataframe having values in a certain range."""

    def apply(
            self,
            rng: Tuple[int, int],
            column: str,
            df: DataFrame
    ) -> DataFrame:
        return filter_rows_in_range(rng=rng, column=column, df=df)


class NotInRangeFilterStrategy(RangeFilterStrategy):
    """Strategy to filter rows in dataframe having values out of a certain range."""

    def apply(
            self,
            rng: Tuple[int, int],
            column: str,
            df: DataFrame
    ) -> DataFrame:
        return filter_rows_out_range(rng=rng, column=column, df=df)


HAVING_VALUE = HavingValueFilterStrategy()
ANY_VALUE = AnyValueFilterStrategy()
ALL_VALUES = AllValueFilterStrategy()
IN_RANGE = InRangeFilterStrategy()
NOT_IN_RANGE = NotInRangeFilterStrategy()

################################################################################
# NB: we could use a different approach for multi-row dataframes:
# ```
# columns = ['index', "A", "B"]
# data = [[0, 0, 0],
#         [0, 0, 2],
#         [1, 1, 0],
#         [1, 1, 1],
#         [2, 2, 0],
#         [2, 2, 1],
#         [2, 2, 2]]
#
# df = DataFrame(data=data, columns=columns).set_index('index')
#
# # We create a multi-row dataframe aggregating by index using a set as each column value:
# mdf = df.groupby('index').aggregate(lambda x: set(v for v in x))
#
# # We apply a lambda on the multi-row dataframe to select the rows having at least a value `2` in column B:
# selector = (mdf['B']).apply(lambda x: len(x.intersection({2})) > 0)
#
# # We filter the multi-row dataframe:
# mdf_filtered = mdf[selector]
# ```
#
# In the future we should evaluate this strategy for cocojoins instead of using simple dataframes with not-unique
# indices:
# - Images joined with annotations: each row (image) I'll have a set of values for each annotation column.
# - Annotations joined with images: each row is an annotation, we don't have multi-rows.
# ...
#
