import pytest
from shutil import rmtree
from cocohelper import COCOHelper

# TODO: Create new tests (not dependant from pycocotools' COCO class)
from cocohelper.errors.validation_error import COCOValidationError
from cocohelper.validator import COCOValidator


# TODO: improve test suite, use AAA approach (Arrange, Act, Assert), use pytest test Classes and fixtures.


ch = COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')
coco = ch.to_coco()


@pytest.fixture
def clean_saved_coco():
    yield
    try:
        rmtree('data/coco_dataset_saved')
    finally:
        pass


@pytest.mark.usefixtures('clean_saved_coco')
def test_save():
    ch_filtered = ch.filter_imgs(img_ids=[1, 2, 3])
    ch_filtered.save('data/coco_dataset_saved')

    ch_saved = ch.load_json('data/coco_dataset_saved/annotations/coco.json')
    assert COCOValidator(ch_saved).validate_dataset()
