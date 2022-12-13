import pytest
from pandas import concat

from cocohelper import COCOHelper
from cocohelper.filters import OR, AND
from cocohelper.filters.cocofilters import cats_filter, imgs_filter
from cocohelper.filters.strategies import ANY_VALUE, ALL_VALUES
from .utils import df_equals


class TestFilteredImgsGetter:

    @pytest.fixture
    def coco(self) -> COCOHelper:
        return COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')

    def test_get_filtered_imgs_with_ids(self, coco):
        a = coco.filtered_imgs(img_ids=[0, 1, 2, 5, 8])
        b = coco.filtered_imgs(imgs_filter(ids=[0, 1, 2, 5, 8]))
        c = imgs_filter(ids=[0, 1, 2, 5, 8]).apply(coco.imgs)
        assert df_equals(a, b, c)


    def test_get_filtered_imgs_with_names(self, coco):
        a = coco.filtered_imgs(img_nms=["24631331976_defa3bb61f_k.jpg", "3825919971_93fb1ec581_b.jpg"])
        b = coco.filtered_imgs(imgs_filter(nms=["24631331976_defa3bb61f_k.jpg", "3825919971_93fb1ec581_b.jpg"]))
        c = imgs_filter(nms=["24631331976_defa3bb61f_k.jpg", "3825919971_93fb1ec581_b.jpg"]).apply(coco.imgs)
        assert df_equals(a, b, c)

    def test_get_filtered_imgs_with_category_id(self, coco):
        a = coco.filtered_imgs(cat_ids=[0])
        b = coco.filtered_imgs(cats_filter(ids=[0]))
        c = coco.joins.extract_imgs(cats_filter(ids=[0]).apply(coco.joins.imgs_cats_anns))
        assert df_equals(a, b, c)

    def test_get_filtered_imgs_with_category_name(self, coco):
        a = coco.filtered_imgs(cat_nms=['balloon'])
        b = coco.filtered_imgs(cats_filter(nms=['balloon']))
        c = coco.joins.extract_imgs(cats_filter(nms=['balloon']).apply(coco.joins.imgs_cats_anns))
        assert df_equals(a, b, c)

    def test_get_filtered_imgs_with_category_supercat(self, coco):
        a = coco.filtered_imgs(supercat_nms='class')
        b = coco.filtered_imgs(cats_filter(super_nms='class'))
        c = coco.joins.extract_imgs(cats_filter(super_nms='class').apply(coco.joins.imgs_cats_anns))
        assert df_equals(a, b, c)

    def test_get_filtered_imgs_with_category_ids(self, coco):
        a = coco.filtered_imgs(cat_ids=[0, 1])
        a2 = concat([coco.filtered_imgs(cat_ids=[0]), coco.filtered_imgs(cat_ids=[1])]).drop_duplicates().sort_index()
        b = coco.filtered_imgs(OR(cats_filter(ids=[0]), cats_filter(ids=[1])))
        b2 = coco.filtered_imgs(cats_filter(ids=[0, 1], strategy=ANY_VALUE))
        b3 = coco.filtered_imgs(cats_filter(ids=[0, 1]))
        c = coco.joins.extract_imgs(OR(cats_filter(ids=[0]), cats_filter(ids=[1])).apply(coco.joins.imgs_cats_anns))
        assert df_equals(a, a2, b, b2, b3, c)

    def test_get_filtered_imgs_having_multiple_category_ids(self, coco):
        # a = coco.filtered_imgs(cat_ids=[0, 1], strategy=ANY_VALUE)  # cannot be performed directly with filtered_xxxx methods
        b = coco.filtered_imgs(AND(cats_filter(ids=[0]), cats_filter(ids=[1])))
        b2 = coco.filtered_imgs(cats_filter(ids=[0, 1], strategy=ALL_VALUES))
        c = coco.joins.extract_imgs(AND(cats_filter(ids=[0]), cats_filter(ids=[1])).apply(coco.joins.imgs_cats_anns))
        assert df_equals(b, b2, c)
