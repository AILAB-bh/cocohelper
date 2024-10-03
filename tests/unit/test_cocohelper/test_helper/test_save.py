import pytest
from shutil import rmtree
from cocohelper import COCOHelper

# TODO: Create new tests (not dependant from pycocotools' COCO class)
from cocohelper.errors.validation_error import COCOValidationError
from cocohelper.validator import COCOValidator


# TODO: improve test suite, use AAA approach (Arrange, Act, Assert), use pytest test Classes and fixtures.

@pytest.fixture
def ch():
    return COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')


@pytest.fixture
def clean_saved_coco():
    yield
    try:
        rmtree('tests/data/coco_dataset_saved')
    finally:
        pass


@pytest.mark.usefixtures('clean_saved_coco')
def test_save(ch):
    ch_filtered = ch.filter_imgs(img_ids=[1, 2, 3])
    ch_filtered.save('tests/data/coco_dataset_saved')

    ch_saved = COCOHelper.load_json('tests/data/coco_dataset_saved/annotations/coco.json')
    assert COCOValidator(ch_saved.to_json_dataset(), 'tests/data/coco_dataset_saved/annotations').validate_dataset()
