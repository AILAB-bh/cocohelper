from cocohelper import COCOHelper
from cocohelper.transforms import Resize, Crop, Compose


# TODO: improve test suite, use pytest test Classes and fixtures.

ch = COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')


def test_resize():
    resize = Resize([100, 100])
    img, ann = ch.get_img_sample(0, transform=resize)
    assert img['image'].shape == (100, 100, 3)


def test_crop():
    img, ann = ch.get_img_sample(0, transform=Crop((150, 100, 50, 20)))
    assert img['image'].shape == (20, 50, 3)


def test_compose():
    compose = Compose([Resize([100, 100]), Crop((0, 0, 50, 50))])
    img, ann = ch.get_img_sample(0, transform=compose)
    assert img['image'].shape == (50, 50, 3)
