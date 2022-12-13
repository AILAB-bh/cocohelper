import pytest
from shutil import rmtree
from cocohelper import COCOHelper

# TODO: Create new tests (not dependant from pycocotools' COCO class)
from cocohelper.errors.validation_error import COCOValidationError
from cocohelper.validator import COCOValidator


# TODO: improve test suite, use AAA approach (Arrange, Act, Assert), use pytest test Classes and fixtures.

ch = COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')
coco = ch.to_coco()


def test_cocohelper_df_join():
    nb_anns = len(ch.anns)
    nb_imgs = len(ch.imgs)
    nb_cats = len(ch.cats)
    assert len(ch.joins.anns_cats) == nb_anns
    assert len(ch.joins.extract_anns(ch.joins.anns_cats)) == nb_anns

    # assert len(coco.joins.extract_cats(coco.joins.cats_anns)) == nb_cats
    # assert len(coco.joins.extract_imgs(coco.joins.imgs_anns)) == nb_imgs
    assert len(ch.joins.extract_anns(ch.joins.anns_imgs)) == nb_anns

    assert len(ch.joins.extract_anns(ch.joins.anns_cats_imgs)) == nb_anns
    assert len(ch.joins.extract_anns(ch.joins.anns_imgs_cats)) == nb_anns

    # assert len(coco.joins.extract_cats(coco.joins.cats_anns_imgs)) == nb_cats

    # assert len(coco.joins.extract_imgs(coco.joins.imgs_anns_cats)) == nb_imgs
    # assert len(coco.joins.extract_imgs(coco.joins.imgs_cats_anns)) == nb_imgs


def test_licenses_is_defined():
    ch_no_licenses = COCOHelper.load_json('tests/data/coco_dataset/annotations/coco_no_licenses.json')

    assert (ch_no_licenses.licenses is not None)
    assert (len(ch_no_licenses.licenses) == 0)
