import pytest
from shutil import rmtree
from cocohelper import COCOHelper

# TODO: Create new tests (not dependant from pycocotools' COCO class)
from cocohelper.errors.validation_error import COCOValidationError
from cocohelper.validator import COCOValidator


# TODO: improve test suite, use AAA approach (Arrange, Act, Assert), use pytest test Classes and fixtures.

ch = COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')
coco = ch.to_coco()


def test_validation():
    has_error = False
    try:
        COCOHelper.load_json('tests/data/coco_dataset/annotations/coco_invalid.json',
                             validate=True)
    except COCOValidationError:
        has_error = True

    assert has_error
