"""
Generic Filter interface and filtering operations.
"""
from pandas import DataFrame
from typing import Tuple
from numbers import Number
import abc
from abc import ABC
import warnings
from cocohelper.filters import strategies


class Filter(ABC):
    """Generic interface of a filter."""

    @abc.abstractmethod
    def apply(
            self,
            df: DataFrame
    ) -> DataFrame:
        """
        A method for applying a filter to the given DataFrame.

        Args:
            df: DataFrame to filter.

        Returns:
            The filtered DataFrame.
        """
        pass


class ComposeFilter(Filter, ABC):

    def __init__(
            self,
            *filters: Filter
    ):
        """
        Generic interface for filter composition.

        Args:
            *filters: The filters to combine.
        """
        self._filters = filters


class AndFilter(ComposeFilter):
    """Composite filters with an 'and' behaviour."""

    def apply(
            self,
            df: DataFrame
    ) -> DataFrame:
        """
        Applies the filter to the given DataFrame.

        Args:
            df: DataFrame to filter.

        Returns:
            The filtered DataFrame.
        """
        if len(self._filters) > 0:
            filtered_indices = [set(fltr.apply(df).index) for fltr in self._filters]
            return df.loc[list(set.intersection(*filtered_indices))].sort_index()
        return df


class OrFilter(ComposeFilter):
    """Composite filters with an 'or' behaviour."""

    def apply(
            self,
            df: DataFrame
    ) -> DataFrame:
        """
        Applies the filter to the given DataFrame.

        Args:
            df: DataFrame to filter.

        Returns:
            The filtered DataFrame.
        """
        if len(self._filters) > 0:
            filtered_indices = [set(fltr.apply(df).index) for fltr in self._filters]
            return df.loc[list(set.union(*filtered_indices))].sort_index()
        return df


class NotFilter(Filter):
    """Negate a filter."""

    def __init__(
            self,
            fltr: Filter
    ):
        """
        Negate a given filter to obtain the opposite behavior.

        Args:
            fltr: Filter to be negated
        """
        self._filter = fltr

    def apply(
            self,
            df: DataFrame
    ) -> DataFrame:
        """
        Applies the filter to the given DataFrame.

        Args:
            df: DataFrame to filter.

        Returns:
            The filtered DataFrame.
        """
        ids = self._filter.apply(df).index
        return df[~df.index.isin(ids)]


class ColumnFilter(Filter, ABC):

    def __init__(
            self,
            values,
            column: str
    ):
        """
        Generic interface of a filter applying constraints over column values.

        Args:
            values: values used to create the constraint and filter the rows.
            column: column where the constraint are applied to filter the rows.
        """
        self._values = values
        self._column = column

    def apply(
            self,
            df: DataFrame
    ) -> DataFrame:
        """
        Apply the filter to the DataFrame

        Args:
            df: DataFrame to filter

        Returns:
            The filtered DataFrame.
        """
        if self._values is None:
            pass  # df = df.reset_index()
        elif self._column in df.columns:
            df = self._apply(self._values, self._column, df)
        elif self._column == df.index.name:
            df = self._apply(self._values, self._column, df.reset_index()).set_index(self._column)
        else:
            # todo: log/waring/exception?
            warnings.warn("Filtering a dataframe using a column that does not exist: returning original data.")
        return df

    @abc.abstractmethod
    def _apply(
            self,
            values,
            column: str,
            df: DataFrame
    ) -> DataFrame:
        """
        Applies the filter to the given DataFrame rows.

        Args:
            values: values to filter in the DataFrame.
            column: column to filter in the DataFrame.
            df: DataFrame to filter.

        Returns:
            The filtered DataFrame.
        """
        pass


class ValueFilter(ColumnFilter):

    def __init__(
            self,
            values,
            column: str,
            strategy: strategies.ValueFilterStrategy = strategies.HAVING_VALUE
    ):
        """
        Filter the rows having certain values in the chosen column.

        Different strategies can be applied to the filter:

        - strategies.HAVING_VALUE: this simple strategy keeps the rows whose
            value of the selected column is one of the values imposed by the
            filter.

        - strategies.ANY_VALUE: this strategy can be applied to a dataframe with
            multiple rows with the same index. In that case we consider the
            dataframe as a multi-row dataframe (grouped by index). This strategy
            keeps a multi-row only if at least one of the values in the selected
            column is in the list of values imposed by the filter

        - strategies.ALL_VALUES: this strategy can be applied to a dataframe with
            multiple rows with the same index. In that case we consider the
            dataframe as a multi-row dataframe (grouped by index). A multi-row
            is discarded if at least one of the values imposed by the filters for
            the requested column is not in the list of values of the same column
            in that multi-row.

        Args:
            values: the values that filtered rows must have.
            column: the column on which values are checked.
            strategy: the ValueStrategy used by the filter
        """
        super().__init__(values, column)
        self._strategy: strategies.ValueFilterStrategy = strategy

    def set_strategy(
            self,
            strategy: strategies.ValueFilterStrategy
    ):
        """
        Set the filter strategy.

        Args:
            strategy: used for filtering the dataframe when using apply().

        Returns:
            None.
        """
        self._strategy = strategy

    def _apply(
            self,
            values,
            column: str,
            df: DataFrame
    ) -> DataFrame:
        """
        Applies the filter to the given DataFrame rows.

        Args:
            values: values to filter in the DataFrame.
            column: column to filter in the DataFrame.
            df: DataFrame to filter.

        Returns:
            The filtered DataFrame.
        """
        return self._strategy.apply(values, column, df)


class RangeFilter(ColumnFilter):

    def __init__(
            self,
            rng: Tuple[float, float],
            column: str,
            strategy: strategies.RangeFilterStrategy = strategies.IN_RANGE
    ):
        """
        Filter rows containing column values in a certain range.

        Args:
            rng: A tuple that contains the lower and upper bound.
            column: The column to filter on.
            strategy: The RangeStrategy used by the filter.
        """
        self._strategy = strategy
        super().__init__(rng, column)

    def set_strategy(
            self,
            strategy: strategies.RangeFilterStrategy
    ):
        """
        Set the filter strategy.

        Args:
            strategy: used for filtering the dataframe when using apply().

        Returns:
            None.
        """
        self._strategy = strategy

    def _apply(
            self,
            values,
            column: str,
            df: DataFrame
    ) -> DataFrame:
        """
        Applies the filter to the given DataFrame rows.

        Args:
            values: values to filter in the DataFrame.
            column: column to filter in the DataFrame.
            df: DataFrame to filter.

        Returns:
            The filtered DataFrame.
        """
        return self._strategy.apply(values, column, df)
