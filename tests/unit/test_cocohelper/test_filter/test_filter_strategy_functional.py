import pytest
from pandas import DataFrame

from cocohelper.filters.strategies.functional import filter_multi_rows_having_any
from cocohelper.filters.strategies.functional import filter_multi_rows_having_all
from cocohelper.filters.strategies.functional import filter_rows_in_range
from cocohelper.filters.strategies.functional import filter_rows_out_range
from cocohelper.filters.strategies.functional import filter_rows_having


class TestFilterStrategyUtils:

    @pytest.fixture
    def df(self) -> DataFrame:
        columns = ['index', "A", "B"]
        data = [[0, 0, 0],
                [0, 0, 2],
                [1, 1, 0],
                [1, 1, 1],
                [2, 2, 0],
                [2, 2, 1],
                [2, 2, 2],
                [3, 3, 3]]

        yield DataFrame(data=data, columns=columns).set_index('index')


    def test_filter_rows_having(self, df):
        filtered_df = filter_rows_having([1, 2], 'B', df)
        assert (filtered_df == df.iloc[[1, 3, 5, 6]]).all().all()

    def test_filter_multi_rows_having_any(self, df):
        filtered_df = filter_multi_rows_having_any([1, 2], 'B', df)
        assert (filtered_df == df.loc[[0, 1, 2]]).all().all()

    def test_filter_multi_rows_having_all(self, df):
        filtered_df = filter_multi_rows_having_all([1, 2], 'B', df)
        assert (filtered_df == df.loc[[2]]).all().all()

    def test_filter_rows_in_range(self, df):
        filtered_df = filter_rows_in_range((0, 2), 'B', df, inclusive="both")
        assert (filtered_df == df.iloc[[0, 1, 2, 3, 4, 5, 6]]).all().all()

    def test_filter_rows_out_range(self, df):
        filtered_df = filter_rows_out_range((1, 2), 'B', df, inclusive="none")
        assert (filtered_df == df.iloc[[0, 2, 4, 7]]).all().all()
