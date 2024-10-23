from pathlib import Path
import cv2
from sklearn.metrics import jaccard_score

from cocohelper import COCOHelper
from cocohelper.adapters import BinaryMaskDatasetAdapter

from cocohelper.importer import Importer
from cocohelper.utils.segmentation import coco_to_binary_masks
from shutil import rmtree

import pytest

# TODO: improve test suite, use AAA approach (Arrange, Act, Assert), use pytest test Classes and fixtures.


@pytest.fixture
def ch():
    return COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')


@pytest.fixture
def clean_saved_coco():
    yield
    try:
        rmtree('tests/data/test_binary_reader')
    finally:
        pass


def test_binary_reader(ch, clean_saved_coco):
    images_dir = Path('tests/data/coco_dataset/images')
    mask_dir = Path('tests/data/test_binary_reader/masks')
    out_dir = Path('tests/data/test_binary_reader/out')
    out_mask_dir = Path('tests/data/test_binary_reader/out_masks')

    # convert coco_dataset to masks
    coco_to_binary_masks(ch, mask_dir)

    images = list(images_dir.glob('*.jpg'))
    masks = list(mask_dir.glob('*.png'))
    # remove images that are not present in masks
    mask_basenames = {mask_path.stem for mask_path in masks}
    images = [image_path for image_path in images if image_path.stem in mask_basenames]
    images.sort()
    masks.sort()
    data_paths = {}
    for (image_path, mask_path) in zip(images, masks):
        data_paths[str(image_path)] = [str(mask_path)]

    categories = {1: {"supercategory": "class", "id": 0, "name": "balloon"},
                  2: {"supercategory": "class", "id": 1, "name": "super_balloon"},
                  3: {"supercategory": "class", "id": 2, "name": "super_balloon_level2"}}

    adapter = BinaryMaskDatasetAdapter(data_paths=data_paths,
                                       image_loader=lambda pth: cv2.imread(pth),
                                       mask_loader=lambda pth: cv2.imread(pth)[..., 0],
                                       categories=categories)

    adapter = Importer(adapter=adapter)

    # convert masks to coco_from_masks
    ch_from_masks = adapter.create(out_coco_dir=out_dir)

    # convert coco_from_masks to masks
    coco_to_binary_masks(ch_from_masks, out_mask_dir)

    # assert masks difference
    out_masks = list(out_mask_dir.glob('*.png'))
    out_masks.sort()

    for (mask, out_mask) in zip(masks, out_masks):
        mask_array = cv2.imread(str(mask))[..., 0]
        out_mask_array = cv2.imread(str(out_mask))[..., 0]

        jac = jaccard_score(out_mask_array.ravel(), mask_array.ravel(), average='micro')
        assert jac > 0.95
