import pytest
from cocohelper import COCOHelper
from cocohelper.errors.validation_error import COCOValidationError


@pytest.fixture
def ch():
    return COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')


@pytest.fixture
def ch_invalid():
    return COCOHelper.load_json('tests/data/coco_dataset/annotations/coco_invalid.json')


def test_init_validation():
    has_error = False
    try:
        COCOHelper.load_json('tests/data/coco_dataset/annotations/coco_invalid.json',
                             validate=True)
    except COCOValidationError:
        has_error = True

    assert has_error


def test_copy_validation(ch_invalid):
    has_error = False
    try:
        ch_invalid.copy(validate=True)

    except COCOValidationError:
        has_error = True

    assert has_error


def test_validations(ch, ch_invalid):
    # good dataset
    is_dir_valid = ch.validator.validate_dir(json_fname='coco.json')
    # invalid dataset
    is_data_valid, error_dict = ch_invalid.validator.validate_dataset()

    assert is_dir_valid
    assert not is_data_valid
    assert len(error_dict) == 11
    assert sum(error_dict.values()) == 10
