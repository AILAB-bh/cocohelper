import pytest
from cocohelper import COCOHelper
from cocohelper.filters.cocofilters import cats_filter
from .utils import df_equals


class TestFilteredCatsGetter:

    @pytest.fixture
    def coco(self) -> COCOHelper:
        return COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')

    def test_get_filtered_cats_with_name(self, coco):
        """Obtain category having name `balloon`, using different approaches, and compare results."""
        a = coco.filtered_cats(cat_nms='balloon')
        b = coco.filtered_cats(cats_filter(nms='balloon'))
        c = cats_filter(nms='balloon').apply(coco.cats)
        assert df_equals(a, b, c)

    def test_get_filtered_cats_with_names(self, coco):
        """Obtain categories `balloon` or `super_balloon`, using different approaches, and compare results."""
        a = coco.filtered_cats(cat_nms=['balloon', 'super_balloon'])
        b = coco.filtered_cats(cats_filter(nms=['balloon', 'super_balloon']))
        c = cats_filter(nms=['balloon', 'super_balloon']).apply(coco.cats)
        assert df_equals(a, b, c)

    def test_get_filtered_cats_with_ids(self, coco):
        """Obtain all categories having id 0 or 1, using different approaches, and compare results"""
        a = coco.filtered_cats(cat_ids=[0, 1])
        b = coco.filtered_cats(cats_filter(ids=[0, 1]))
        c = cats_filter(ids=[0, 1]).apply(coco.cats)
        assert df_equals(a, b, c)


    def test_get_filtered_cats_supercat(self, coco):
        """Obtain information about categories having super-category name `class`"""
        a = coco.filtered_cats(supercat_nms='class')
        b = coco.filtered_cats(cats_filter(super_nms='class'))
        c = cats_filter(super_nms='class').apply(coco.cats)
        assert df_equals(a, b, c)
