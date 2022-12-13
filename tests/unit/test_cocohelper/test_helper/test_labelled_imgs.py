import pytest
from shutil import rmtree
from cocohelper import COCOHelper

# TODO: Create new tests (not dependant from pycocotools' COCO class)
from cocohelper.errors.validation_error import COCOValidationError
from cocohelper.validator import COCOValidator


# TODO: improve test suite, use AAA approach (Arrange, Act, Assert), use pytest test Classes and fixtures.


ch = COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')
coco = ch.to_coco()


def test_get_labelled_imgs():
    imgs = ch.unlabelled_imgs
    assert len(imgs) == 2

    imgs = ch.labelled_imgs
    assert len(imgs) == 12

    chf = ch.drop_labelled()
    assert len(chf.imgs) == 2

    chf = ch.drop_unlabelled()
    assert len(chf.imgs) == 12
