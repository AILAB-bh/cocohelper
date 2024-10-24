import pytest
import numpy as np
import os
from cocohelper import COCOHelper
from cocohelper.utils.segmentation import (
    mask_to_compressed_rle,
    mask_to_polygon,
    mask_to_rle,
    rle_to_mask,
    compressed_rle_to_mask,
    polygon_to_mask,
    encode_mask,
    decode_mask,
    get_segmentation_mode,
    convert_to_mask,
    convert_to_mode,
    compute_polygon_area,
    coco_to_binary_masks
)


@pytest.fixture
def mask():
    # Create a binary mask
    return np.array([
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1],
    ])


@pytest.fixture
def modes():
    return ['RLE', 'cRLE', 'polygon']


@pytest.fixture
def ch():
    # Load the COCO dataset
    return COCOHelper.load_json('tests/data/coco_dataset/annotations/coco.json')


@pytest.fixture
def folder():
    # create a folder to store the binary masks
    folder = 'tests/data/test_utils'
    # if the folder already exists, empty it
    if os.path.exists(folder):
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
    return folder


def test_mask_to_compressed_rle(mask):
    height, width = mask.shape

    # Convert the mask to compressed RLE format
    compressed_rle = mask_to_compressed_rle(mask)

    # Convert the compressed RLE back to a binary mask
    decoded_mask = compressed_rle_to_mask(compressed_rle, height, width)

    # Check if the decoded mask is the same as the original mask
    assert np.array_equal(mask, decoded_mask)


def test_mask_to_polygon(mask):
    # Convert the mask to polygon format
    polygon = mask_to_polygon(mask)

    # Convert the polygon back to a binary mask
    decoded_mask = polygon_to_mask(polygon, mask.shape[1], mask.shape[0])

    # Check if the decoded mask is the same as the original mask
    assert np.array_equal(mask, decoded_mask)


def test_mask_to_rle(mask):
    # Convert the mask to RLE format
    rle = mask_to_rle(mask)

    # Convert the RLE back to a binary mask
    decoded_mask = rle_to_mask(rle)

    # Check if the decoded mask is the same as the original mask
    assert np.array_equal(mask, decoded_mask)


def test_get_segmentation_mode_exception():
    try:
        # Determine the format of the encoded mask
        get_segmentation_mode(1)
    except ValueError as e:
        msg = e.args[0]

    # Check if the Exception is raised
    assert msg is not None
    assert msg == "Invalid argument type for argument `segmentation`. " \
                  "Input `segmentation` should have a list, dictionary, or string type."


def test_get_segmentation_mode(mask, modes):
    for mode in modes:
        # Encode the mask
        encoded = encode_mask(mask, mode)

        # Determine the format of the encoded mask
        determined_mode = get_segmentation_mode(encoded)

        # Check if the determined format is the same as the format used for encoding
        assert mode == determined_mode


def test_encode_mask(mask, modes):
    height, width = mask.shape

    for mode in modes:
        # Encode the mask
        encoded = encode_mask(mask, mode)

        # Decode the mask
        decoded = decode_mask(encoded, mode, height=height, width=width)

        # Check if the decoded mask is the same as the original mask
        assert np.array_equal(mask, decoded)


def test_decode_mask(mask, modes):
    height, width = mask.shape

    for mode in modes:
        # Encode the mask
        encoded = encode_mask(mask, mode)

        # Decode the mask
        decoded = decode_mask(encoded, mode, height=height, width=width)

        # Check if the decoded mask is the same as the original mask
        assert np.array_equal(mask, decoded)


def test_convert_to_mask(mask, modes):
    height, width = mask.shape
    for mode in modes:
        # Encode the mask
        encoded = encode_mask(mask, mode)

        # Convert the encoded mask back to a binary mask
        converted_mask = convert_to_mask(encoded, height, width)

        # Check if the converted mask is the same as the original mask
        assert np.array_equal(mask, converted_mask)


def test_convert_to_mode(mask, modes):
    height, width = mask.shape
    for mode in modes:
        # Encode the mask
        encoded = encode_mask(mask, mode)

        for target_mode in modes:
            # Convert the encoded mask to the target mode
            converted = convert_to_mode(encoded, target_mode, height, width)

            # Decode the converted mask
            decoded = decode_mask(converted, target_mode, height=height, width=width)

            # Check if the decoded mask is the same as the original mask
            assert np.array_equal(mask, decoded)


def test_compute_polygon_area():
    # Create a polygon as a list of vertices, e.g [x1, y1, x2, y2, ..., xn, yn]
    polygon = [0, 0, 1, 0, 1, 1, 0, 1]

    # Compute the area of the polygon
    area = compute_polygon_area(polygon)

    # Check if the computed area is as expected
    assert area == 1.0


def test_coco_to_binary_masks(ch, folder):
    # Convert the COCO-style segmentation to a binary mask
    coco_to_binary_masks(ch, dest_dir=folder)

    # Check if now the folder contains the binary masks
    assert len(os.listdir(folder)) > 0
