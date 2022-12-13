import numpy as np
from cocohelper import COCOHelper
from cocohelper.errors.not_found_error import COCOImageNotFoundError, COCOAnnotationNotFoundError


# TODO: improve test suite, use AAA approach (Arrange, Act, Assert), use pytest test Classes and fixtures.

ch = COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')


def test_load_ann_by_id():
    img, ann = ch.get_ann_sample(ann_id=0)

    assert type(img) == np.ndarray
    assert type(ann) == dict
    assert ann['category_id'] == 1


def test_load_ann_by_index():
    img, ann = ch.get_ann_sample(idx=0)

    assert type(img) == np.ndarray
    assert type(ann) == dict
    assert ann['category_id'] == 1


def test_load_ann_with_transform():
    class MockTransform:

        def apply(self, img, anns):
            ann = anns[0]
            return 1, [
                {**ann, 'category_id': 2}
            ]

    img, ann = ch.get_ann_sample(idx=0, transform=MockTransform())

    assert type(img) == int
    assert type(ann) == dict
    assert ann['category_id'] == 2


def test_load_img_by_id():
    img, ann = ch.get_img_sample(img_id=0)

    assert type(img) == dict
    assert type(img['image']) == np.ndarray
    assert type(ann) == list
    assert len(ann) == 1


def test_load_img_by_index():
    img, ann = ch.get_img_sample(idx=0)

    assert type(img) == dict
    assert type(img['image']) == np.ndarray
    assert type(ann) == list
    assert len(ann) == 1


def test_invalid_load_img():
    has_error = False
    try:
        ch.get_img_sample()
    except AssertionError:
        has_error = True

    assert has_error


def test_load_img_assertion():
    has_error = False
    try:
        ch.get_img_sample(img_id=0, idx=0)
    except AssertionError:
        has_error = True

    assert has_error


def test_invalid_load_ann():
    has_error = False
    try:
        ch.get_ann_sample()
    except AssertionError:
        has_error = True

    assert has_error


def test_load_ann_assertion():
    has_error = False
    try:
        ch.get_ann_sample(ann_id=0, idx=0)
    except AssertionError:
        has_error = True

    assert has_error


def test_load_img_with_transform():
    class MockTransform:

        def apply(self, img, anns):
            return 1, []

    img, ann = ch.get_img_sample(idx=0, transform=MockTransform())

    assert img['image'] == 1
    assert len(ann) == 0


def test_load_img_sample_not_found():
    img_id = 100
    error = None
    try:
        ch.get_img_sample(img_id=img_id)
    except COCOImageNotFoundError as e:
        error = e

    assert error is not None
    assert error.image_id == img_id


def test_load_ann_sample_not_found():
    ann_id = 1000
    error = None
    try:
        ch.get_ann_sample(ann_id=ann_id)
    except COCOAnnotationNotFoundError as e:
        error = e

    assert error is not None
    assert error.ann_id == ann_id


def test_load_img_not_found():
    img_id = 100
    error = None
    try:
        ch.get_img(img_id=img_id)
    except COCOImageNotFoundError as e:
        error = e

    assert error is not None
    assert error.image_id == img_id
