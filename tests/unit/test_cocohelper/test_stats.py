import pytest
from numpy import isclose

from cocohelper import COCOHelper
from cocohelper.stats import COCOStats

# TODO: improve test suite, use AAA approach (Arrange, Act, Assert), use pytest test Classes and fixtures.


@pytest.fixture
def ch():
    return COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')


@pytest.fixture
def stats(ch):
    return COCOStats(ch)


def test_annotation_size_stats(stats):
    opt_h, opt_w = stats.get_optimal_image_size()
    assert opt_h == 1024 and opt_w == 1040


def test_annotation_size_stats_cardinality(stats):
    img_ss = stats.get_annotation_size_stats()
    assert len(img_ss) == 11


def test_cat_ids_ratios(stats):
    # Arrange:
    # expected_ids_ratios = {
    #     0: 0.8333333333333334,
    #     1: 0.10416666666666667,
    #     -1: 0.041666666666666664,  # -1 is for images having no annotation
    #     2: 0.020833333333333332
    # }
    expected_ids_ratios = {
        0: 0.5714285714285714,
        1: 0.35714285714285715,
        2: 0.07142857142857142
    }

    # Act:
    ids_ratios = stats.cat_ids_ratios

    # Assert:
    for key, ratio in ids_ratios.items():
        assert isclose(ratio, expected_ids_ratios[key], rtol=1e-6)


def test_cat_nms_ratios(stats):
    # Arrange:
    expected_nms_ratios = {
        'balloon': 0.5714285714285714,
        'super_balloon': 0.35714285714285715,
        'super_balloon_level2': 0.07142857142857142
    }

    # Act:
    nms_ratios = stats.cat_nms_ratios

    # Assert:
    for key, ratio in nms_ratios.items():
        assert isclose(ratio, expected_nms_ratios[key], rtol=1e-6)


def test_nb_imgs(ch):
    coco_stats = COCOStats(ch)
    assert coco_stats.nb_imgs == 14


def test_nb_cats(ch):
    coco_stats = COCOStats(ch)
    assert coco_stats.nb_cats == 3


def test_nb_anns(ch):
    coco_stats = COCOStats(ch)
    assert coco_stats.nb_anns == 46


def test_nb_imgs_wo_anns(ch):
    coco_stats = COCOStats(ch)
    assert coco_stats.nb_imgs_wo_anns == 2
