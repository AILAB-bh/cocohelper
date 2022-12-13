import pytest
from shutil import rmtree
from cocohelper import COCOHelper

# TODO: Create new tests (not dependant from pycocotools' COCO class)
from cocohelper.errors.validation_error import COCOValidationError
from cocohelper.validator import COCOValidator


# TODO: improve test suite, use AAA approach (Arrange, Act, Assert), use pytest test Classes and fixtures.


ch = COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')
coco = ch.to_coco()


def test_filters():
    chf = ch.filter_imgs(img_ids=[1, 2, 3])
    assert len(chf.imgs) == 3
    assert len(chf.anns) == 22

    chf = ch.filter_imgs(img_ids=[1, 2, 3], invert=True)
    assert len(chf.imgs) == (len(ch.imgs) - 3)
    assert len(chf.anns) == 24

    chf = ch.filter_cats(cat_ids=[0, 1])
    assert len(chf.cats) == 2
    assert len(chf.anns) == 45

    chf = ch.filter_cats(cat_ids=[0, 1], invert=True)
    assert len(chf.cats) == (len(ch.cats) - 2)
    assert len(chf.anns) == 1

    chf = ch.filter_anns(ann_ids=[1, 2, 3])
    assert len(chf.anns) == 3

    chf = ch.filter_anns(ann_ids=[1, 2, 3], invert=True)
    assert len(chf.anns) == (len(ch.anns) - 3)
