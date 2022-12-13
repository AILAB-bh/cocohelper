import pytest

from cocohelper import COCOHelper
from cocohelper.filters import RangeFilter, NOT
from cocohelper.filters.cocofilters import anns_filter
from cocohelper.filters.strategies import NOT_IN_RANGE, IN_RANGE
from .utils import df_equals


class TestFilteredImgsGetter:

    @pytest.fixture
    def coco(self) -> COCOHelper:
        return COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')

    def test_get_filtered_anns_with_ids(self, coco):
        a = coco.filtered_anns(ann_ids=[0, 1])
        b = coco.filtered_anns(anns_filter(ids=[0, 1]))
        c = anns_filter(ids=[0, 1]).apply(coco.anns)
        assert df_equals(a, b, c)

    def test_get_filtered_anns_with_ids_2(self, coco):
        a = coco.filtered_anns(ann_ids=[0, 10])
        b = coco.filtered_anns(anns_filter(ids=[0, 10]))
        c = anns_filter(ids=[0, 10]).apply(coco.anns)
        assert df_equals(a, b, c)

    def test_get_filtered_anns_with_failing_ids(self, coco):
        a = coco.filtered_anns(ann_ids=[0, 10000])
        b = coco.filtered_anns(anns_filter(ids=[0, 10000]))
        c = anns_filter(ids=[0, 10000]).apply(coco.anns)
        assert df_equals(a, b, c)

    def test_get_filtered_anns_with_area_in_range(self, coco):
        a = coco.filtered_anns(area_rng=[0, 10000])
        b = coco.filtered_anns(anns_filter(area_rng=[0, 10000]))
        c = anns_filter(area_rng=[0, 10000]).apply(coco.anns)
        d = RangeFilter(rng=(0, 10000), column='area').apply(coco.anns)
        assert df_equals(a, b, c, d)

    def test_get_filtered_anns_with_area_not_in_range(self, coco):
        a = coco.filtered_anns(area_rng=[0, 10000], invert=True)
        b = coco.filtered_anns(anns_filter(area_rng=[0, 10000], rng_strategy=NOT_IN_RANGE))
        b2 = coco.filtered_anns(NOT(anns_filter(area_rng=[0, 10000], rng_strategy=IN_RANGE)))
        c = anns_filter(area_rng=[0, 10000], rng_strategy=NOT_IN_RANGE).apply(coco.anns)
        c2 = NOT(anns_filter(area_rng=[0, 10000])).apply(coco.anns)
        d = RangeFilter(rng=(0, 10000), column='area', strategy=NOT_IN_RANGE).apply(coco.anns)
        d2 = NOT(RangeFilter(rng=(0, 10000), column='area')).apply(coco.anns)
        assert df_equals(a, b, b2, c, c2, d, d2)
