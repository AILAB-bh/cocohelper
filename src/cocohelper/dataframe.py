"""Extend pandas Dataframe to allow easier manipulation of COCO Datasets.
"""
from typing import Tuple, Hashable, Optional, Union
from pandas._typing import IndexLabel
from pandas import DataFrame
import warnings
from cocohelper.utils.colmapper import ColMap


class COCODataFrame(DataFrame):

    def __init__(
            self,
            dataframe: DataFrame,
            name: str,
            col_mappers: Tuple[ColMap, ...] = tuple()
    ):
        """Extends pandas Dataframe to easy column remapping and join with COCO
        Datasets.

        Args:
            dataframe: the original data.
            name: the name of the dataframe.
            col_mappers: a list of Column Mapper objects that can be used to
                remap other columns of the original dataframe.
        """
        # convert dtypes to the best possible, and convert numpy types (e.g. int64) to nullable pandas types (Int64)
        # TODO: do we need to convert_dtypes ? This create problem when we convert back to numpy
        #  if we don't explicitly specify the numpy dtype.
        # dataframe = dataframe.convert_dtypes()

        super(COCODataFrame, self).__init__(self.map_dataframe_cols(dataframe, col_mappers))

        if isinstance(dataframe, COCODataFrame):
            warnings.warn("COCODataFrame created by a COCODataFrame without using a copy constructor.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.__name = name
            self.__col_mappers = col_mappers

        if 'id' in self.columns:
            self.set_index('id', inplace=True)
            self.index.name = f"{self.name}_id"
        # self.index.name = self.name + '_id'

    def copy(
            self,
            deep: bool = True
    ) -> "COCODataFrame":
        """Returns a copy of the COCODataFrame. The copy is deep by default.

        Args:
            deep: if True, the copy is deep.

        Returns:
            A copy of the COCODataFrame.
        """
        data = super().copy(deep)
        return COCODataFrame(data, self.name, self.__col_mappers)

    # def __deepcopy__(self, memodict={}):
    #     return COCODataFrame(super().__deepcopy__(memodict), self.name, self.__col_mappers)

    def auto_reset_index(
            self,
            level: Optional[IndexLabel] = None,
            # todo: `Hashable | Sequence[Hashable] | None = None` gives error on python 3.8
            inplace: bool = False,
            col_level: Hashable = 0,
            col_fill: Hashable = "",
    ) -> DataFrame:
        """
        Similar to reset_index but, index will drop if index name is None.

        Args:
            level: only remove the given levels from the index. Removes all
                levels by default.
            inplace: if to modify the DataFrame rather than creating a new one.
            col_level: if the columns have multiple levels, determines which
                level the labels are inserted into. By default, it is inserted
                into the first level.
            col_fill: if the columns have multiple levels, determines how the
                other levels are named. If None then the index name is repeated.

        Returns:
            DataFrame or None. Changed row labels or None if ``inplace=True``.
        """
        drop = True if self.index.name is None else False
        return self.reset_index(level, drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill)

    def set_index(
            self,
            keys,
            drop: bool = True,
            append: bool = False,
            inplace: bool = False,
            verify_integrity: bool = False
    ) -> Union[DataFrame, None]:
        """
        Set the DataFrame index using existing columns.

        Set the DataFrame index (row labels) using one or more existing
        columns or arrays (of the correct length). The index can replace the
        existing index or expand on it.

        Args:
            keys: label or array-like or list of labels/arrays This parameter can be either a single column key, a
                single array of the same length as the calling DataFrame, or a list containing an arbitrary combination
                of column keys and arrays. Here, "array" encompasses :class:`Series`, :class:`Index`, ``np.ndarray``,
                 and instances of :class:`~collections.abc.Iterator`.
            drop: bool, default True. Delete columns to be used as the new index.
            append: bool, default False. Whether to append columns to existing index.
            inplace: bool, default False. Whether to modify the DataFrame rather than creating a new one.
            verify_integrity: bool, default False. Check the new index for duplicates. Otherwise, defer the check until
                necessary. Setting to False will improve the performance of this method.

        Returns:
            DataFrame or None. Changed row labels or None if ``inplace=True``.
        """
        # Before setting index, reset the index keeping it if the index has a name:
        self.auto_reset_index(inplace=True)
        return super(COCODataFrame, self).set_index(keys=keys,
                                                    drop=drop,
                                                    append=append,
                                                    inplace=inplace,
                                                    verify_integrity=verify_integrity)

    @property
    def data(self) -> DataFrame:
        """
        Returns the DataFrame with columns mapped to the new names.
        """
        return DataFrame(self._data)

    @property
    def orig_data(self) -> DataFrame:
        """
        Returns the original DataFrame remapping columns to the original names.
        """
        return self.unmap_dataframe_cols(self._data, self.__col_mappers)

    @property
    def name(self) -> str:
        """
        Get the name assigned to this dataframe.
        """
        return self.__name

    def cocojoin(
            self,
            cdf: "COCODataFrame",
            how: str = 'left'
    ) -> "COCODataFrame":
        """
        Automatically join the dataframe with an input dataframe exploiting the index column names as foreign keys.

        If the `cdf` has a column with the same name of the self dataframe index column, that column will be used as a
        foreign key and the join is performed. If self dataframe has a column with the same name of the `cdf` index
        column, that column will be used as a foreign key and the join is performed.

        Args:
            cdf: the COCODataFrame on which execute the join.
            how: the type of join

        Returns:
            a new COCODataFrame
        """
        if cdf is None:
            warnings.warn("Trying to join with `None` value, skipping and returning self.")
            return self

        if len(cdf) == 0:
            warnings.warn("Trying to join with an empty COCODataFrame, skipping and returning self.")

        joined_col_mappers = tuple(set(self.__col_mappers + cdf.__col_mappers))

        if f"{cdf.name}_id" in self.columns:
            left, right = self, cdf
        elif f"{self.name}_id" in cdf.columns:
            left, right = cdf, self
            how = COCODataFrame._invert_join_how(how)
        else:
            raise KeyError("We can't join the two COCODataFrame")

        out_df = left.reset_index(drop=left.index.name is None).join(right, how=how, on=f"{right.name}_id")
        # out_df = out_df.convert_dtypes()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return COCODataFrame(out_df, f"{left.name}_{right.name}", col_mappers=joined_col_mappers)

    @staticmethod
    def _invert_join_how(how: str):
        """
        Inverts the join option.
        Args:
            how: input `how` parameter for the join

        Returns:
            Inverted (opposite) `how` parameter for the join
        """
        if how == 'left':
            return 'right'
        if how == 'right':
            return 'left'
        return how

    @staticmethod
    def map_dataframe_cols(
            dataframe: DataFrame,
            col_mappers: Tuple[ColMap, ...] = tuple()
    ):
        """
        Applies a mapping to the DataFrame columns according to the provided column mappers.

        Args:
            dataframe: DataFrame to modify
            col_mappers: column mappers to be applied

        Returns:
            DataFrame with mapped columns according to the provided column mappers.
        """
        for col_mapper in col_mappers:
            dataframe = COCODataFrame.try_remap_col(dataframe, col_mapper)
        return dataframe

    @staticmethod
    def unmap_dataframe_cols(
            dataframe: DataFrame,
            col_mappers: Tuple[ColMap, ...] = tuple()
    ):
        """
        Unmaps a column in a DataFrame.

        Args:
            dataframe: DataFrame to be unmapped
            col_mappers: column mapper

        Returns:
            The DataFrame with unmapped columns.
        """
        for col_mapper in col_mappers:
            dataframe = dataframe.rename(columns=col_mapper.to_orig)
        return dataframe

    @staticmethod
    def try_remap_col(
            dataframe: DataFrame,
            col_mapper: ColMap,
            to_orig: bool = False
    ) -> DataFrame:
        """
        Attempts to remap a column in a DataFrame.

        Args:
            dataframe: DataFrame whose column you want to remap
            col_mapper: column mapper
            to_orig: if True, the color mapper will be mapped to the origin mapper instead of new

        Returns:
            The DataFrame with remapped columns.
        """
        col_map = col_mapper.to_orig if to_orig else col_mapper.to_new
        from_col = list(col_map.items())[0][0]
        to_col = list(col_map.items())[0][1]
        try:
            dataframe = dataframe.rename(columns=col_map)
            dataframe.set_index(to_col, drop=False)
        except KeyError:
            warnings.warn(f"Can't find column name `{from_col}`: remap to `{to_col}` is skipped.")
        return dataframe
