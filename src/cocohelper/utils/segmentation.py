"""
Utilities* for converting segmentation annotations between different formats.
"""
from typing import List, Tuple, Dict, Union, Iterable, Optional, Callable, Any

import numpy.typing as npt
from pycocotools import mask as coco_mask
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from pathlib import Path
import numpy as np
import base64
import zlib
import cv2
from cocohelper import COCOHelper


# TODO: improve software architecture for this file
#  (use classes and polymorphism instead of common-interface functions)


def mask_to_compressed_rle(
        mask: np.ndarray,
        **kwargs
) -> str:
    """
    Converts a binary mask to compressed RLE format.

    Args:
        mask: a binary mask to encode.
        **kwargs: extra parameters are ignored.
    Returns:
        A string encoding the mask as compressed RLE.
    """

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape((mask.shape[0], mask.shape[1], 1))
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode()


def mask_to_polygon(
        mask: np.ndarray,
        compression_factor: Union[float, Tuple[float, float]] = 1.0,
        **kwargs
) -> List:
    """
    Converts segmentation mask to a list of polygons.

    Args:
        mask: numpy array containing multiple segmentation masks. Each mask must
            be associated with a different number, where 0 is for background,
            and other numbers related to different objects. For example, objects
            of a class A may be associated with a value of 1, class B to values
            10, class C to 255, and so on.
        compression_factor: you can set this parameter > 1 to obtain
            compressed polygons. The compression algorithm first downsample the 
            input mask of a compression_factor along its dimensions, then 
            computes the polygon, finally rescales the polygon to the original 
            dimension. The resulting polygon has a reduced number of vertices.
            Polygons are obtained with cv2.CHAIN_APPROX_SIMPLE approximation.

        **kwargs: extra parameters are ignored.

    Returns:
        List of polygons segmentation masks.
    """
    polygons = []
    classes = list(np.unique(mask))
    if 0 in classes:
        classes.remove(0)

    if isinstance(compression_factor, float) or isinstance(compression_factor, int):
        hv_compression_factor = (compression_factor, compression_factor)
    else:
        hv_compression_factor = (compression_factor[0], compression_factor[1])

    if np.any(np.array(compression_factor) > 1.0):
        # resize mask to smaller dimension:
        new_size = [int(np.round(dim / factor)) for dim, factor in zip(mask.shape, hv_compression_factor)]
        # use cv2.INTER_NEAREST interpolation because we need to maintain the same numbers as in the initial mask
        # (i.e. classes must not change):
        mask = cv2.resize(mask, (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)
        # compute multiplicative factors for rescaling the polygon to the new size:
        multiplicative_factor = [1 / (int(np.round(dim / factor)) / dim)
                                 for dim, factor in zip(mask.shape, hv_compression_factor)]
    else:
        multiplicative_factor = [1, 1]

    # iterate over the different segmentations (classes) inside the mask:
    for cls in classes:
        # pick the class-related segmentation and convert it to binary:
        binary = np.zeros_like(mask)
        binary[mask == cls] = 1

        # compute mask contours and then convert them to polygons:
        binary = binary.astype(np.uint8)
        contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            sqz_cnt = np.squeeze(cnt, axis=1)

            if len(sqz_cnt) >= 3:  # a polygon must contain at least 3 points
                polygon = Polygon(sqz_cnt).exterior.coords
                # rescale polygons to the right size, if multiplicative_factors > 1
                x_coords = [c[0] * multiplicative_factor[1] for c in polygon]
                y_coords = [c[1] * multiplicative_factor[0] for c in polygon]
                sgm = []
                for x, y in zip(x_coords, y_coords):
                    sgm.extend([float(x), float(y)])
                polygons.append(sgm)

    return polygons


def mask_to_rle(
        binary_mask: np.ndarray,
        **kwargs
) -> Dict[str, list]:
    """
    Converts a binary mask to RLE encoding.

    Args:
        binary_mask: a binary mask as a numpy array.
        **kwargs: extra parameters are ignored.

    Returns:
        The RLE encoding of the binary mask.
    """
    counts = []
    last_elem = 0
    running_length = 0
    for i, elem in enumerate(binary_mask.ravel(order='F')):
        if not elem == last_elem:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1
    counts.append(running_length)
    return {'counts': counts, 'size': list(binary_mask.shape)}


def rle_to_mask(
        rle_code,
        dtype: npt.DTypeLike = bool,
        **kwargs,
):
    """
    Converts an RLE to a binary mask with the given output dtype.

    Args:
        rle_code:
        dtype: an output dtype for the converted mask.
        **kwargs: extra parameters are ignored.

    Returns:
        The RLE mask as a numpy array.
    """
    compressed_rle = coco_mask.frPyObjects(rle_code, rle_code.get('size')[0], rle_code.get('size')[1])
    binary_mask = coco_mask.decode(compressed_rle)
    binary_mask = binary_mask.astype(dtype)
    return binary_mask


def compressed_rle_to_mask(
        rle_code: str,
        height: int,
        width: int,
        dtype: npt.DTypeLike = bool,
        **kwargs
) -> np.ndarray:
    """
    Converts a compressed RLE to a binary mask with the given output dtype.

    Args:
        rle_code: RLE encoding of the semantic mask.
        width: width of the output image array.
        height: height of the output image array.
        dtype: an output dtype for the converted mask.
        **kwargs: extra parameters are ignored.

    Returns:
        The compressed RLE mask as a numpy array.
    """
    decoded_string = base64.b64decode(rle_code)
    uncompressed_string = zlib.decompress(decoded_string, wbits=zlib.MAX_WBITS)
    detection = {
        'size': [height, width],
        'counts': uncompressed_string
    }
    mask = coco_mask.decode([detection])
    binary_mask = mask.astype(dtype)
    return binary_mask[:, :, 0]


def polygon_to_mask(
        polygon_code: List[List],
        width: int,
        height: int,
        value: float = 0.0,
        dtype: npt.DTypeLike = np.float32,
        **kwargs,
) -> np.ndarray:
    """
    Creates and returns a mask from polygon in COCO format.

    Args:
        polygon_code: polygon coordinates.
        width: width of the output image array.
        height: height of the output image array.
        value: a value used for substituting zero-valued pixels in the output
            numpy array.
        dtype: an output dtype for the converted mask.
        **kwargs: extra parameters are ignored.


    Returns:
        The polygon mask as a numpy array.
    """
    # create a binary image
    img = Image.new(mode='L', size=(width, height), color=0)  # mode L = 8-bit pixels, black and white
    draw = ImageDraw.Draw(img)

    # draw polygons
    for polygon in polygon_code:
        draw.polygon(polygon, outline=1, fill=1)

    # replace 0 with 'value'
    mask = np.array(img).astype(dtype)
    mask[np.where(mask == 0)] = value

    return mask


# TODO: fix this pseudo-polymorphism
MASK_ENCODERS: Dict[str, Callable] = {
    'RLE': mask_to_rle,
    'cRLE': mask_to_compressed_rle,
    'polygon': mask_to_polygon
}

# TODO: fix this pseudo-polymorphism
MASK_DECODERS: Dict[str, Callable] = {
    'RLE': rle_to_mask,
    'cRLE': compressed_rle_to_mask,
    'polygon': polygon_to_mask
}


def encode_mask(
        mask: np.ndarray,
        mode: str,
        compression_factor: Union[float, Tuple[float, float]] = 1.0,
) -> List:
    """
    Encodes a binary mask numpy array into another format.

    Available formats are 'polygon', 'RLE', or compressed RLE 'cRLE' vector.

    Args:
        mask: numpy array containing the object mask
        mode: encoding mode. Can be 'polygon', 'RLE', or compressed RLE 'cRLE'
            vector.
        compression_factor: a compression factor. You can set this
            parameter > 1 to obtain compressed polygons. The compression
            algorithm first down-samples the input mask of a compression_factor
            along its dimensions, then computes the mask encoding.

    Returns:
        The encoded mask according to the given `mode`.
    """
    assert mode in MASK_ENCODERS.keys()
    return MASK_ENCODERS[mode](mask, compression_factor)


def decode_mask(
        segmentation,
        height: int,
        width: int,
        mode: str
) -> np.ndarray:
    """
    Decodes a binary mask from another format.

    Available conversion formats are 'polygon', 'RLE', or compressed RLE 'cRLE'
    vector. The mask, in one of these formats can be converted to a binary
    numpy array.

    Args:
        segmentation: a segmentation in COCO format.
        height: height of the output image array.
        width: width of the output image array.
        mode: decoding mode. Can be 'polygon', 'RLE', or compressed RLE 'cRLE'
            vector.

    Returns:
        The decoded mask according to the given `mode`.
    """
    assert mode in MASK_DECODERS.keys()
    return MASK_DECODERS[mode](segmentation, height=height, width=width)


def get_segmentation_mode(
        segmentation: Union[List[List], Dict, str]
):
    """
    Automatically detect the segmentation format: RLE, cRLE or polygon.

    The detection is based on the standard of coco format, where polygons
    are represented as lists, RLE as an object with size and counts properties,
    and cRLE as a string.

    Args:
        segmentation: the segmentation to inspect.

    Returns:
        The segmentation mode as a string.
    """
    if type(segmentation) == list:
        return 'polygon'
    elif type(segmentation) == dict:
        return 'RLE'
    return 'cRLE'


def convert_to_mask(
        segmentation: Union[List[List], Dict, str],
        height: int,
        width: int
) -> np.ndarray:
    """
    Convert segmentation from different formats to polygon format.

    Supported formats are 'RLE', 'cRLE' and 'polygon'.

    Args:
        segmentation: the segmentation to convert.
        height: height of the image.
        width: width of the image.

    Returns:
        The segmentation converted to mask.
    """
    mode = get_segmentation_mode(segmentation)
    return decode_mask(segmentation, height, width, mode=mode)


def convert_to_mode(
        segmentation: Union[List[List], Dict, str],
        mode: str,
        height: int, width: int,
        compression_factor: Union[float, Tuple[float, float]] = 1.0
):
    """
    Convert segmentation to the specified format.

    Uses `pycocotools` to handle the RLE format.

    Args:
        segmentation: the segmentation to convert.
        mode: the format to convert to.
        height: height of the image.
        width: width of the image.
        compression_factor: setting this > 1 compress the segmentation if
            `mode=polygon`.

    Returns:
        The converted segmentations.
    """
    curr_mode = get_segmentation_mode(segmentation)

    if curr_mode == mode:
        return segmentation

    mask = decode_mask(segmentation, height, width, mode=curr_mode)
    return encode_mask(mask, mode=mode, compression_factor=compression_factor)


def compute_polygon_area(
        polygon
) -> float:
    """
    Computes a segmentation area from its polygon coordinates.

    Args:
        polygon: polygon coordinates for the segmentation.

    Returns:
        The area of the segmentation.
    """
    # check we have the same number of x and y coordinates:
    assert len(polygon) % 2 == 0

    if len(polygon) >= 6:
        # i.e. we have at least 3 points:
        x_coords = polygon[0::2]
        y_coords = polygon[1::2]
        poly = Polygon(zip(x_coords, y_coords))
        return poly.area
    else:
        return 0


def coco_to_binary_masks(
        ch: COCOHelper,
        dest_dir: Union[str, Path],
        scaling: Optional[float] = 1.0
):
    """
    Converts annotations from COCO to binary masks.

    Args:
        ch: a COCOHelper containing the source COCO dataset.
        dest_dir: a destination directory for the output (converted) files.
        scaling: an optional scaling parameter to rescale annotation values and
            improve their visibility by the human eye when opened with GUI tools.

    Returns:
        None. Outputs the converted dataset in the given destination directory.
    """
    dest_annotation_dir = Path(dest_dir)
    dest_annotation_dir.mkdir(parents=True, exist_ok=True)

    image_ids = ch.imgs.index.tolist()
    for img_id in image_ids:
        image_dict = ch.filtered_imgs(img_ids=img_id).iloc[0].to_dict()
        image_fname_prefix = Path(image_dict["file_name"]).with_suffix('').name
        image_size = (image_dict["height"], image_dict["width"])

        annotations = ch.filtered_anns(img_ids=img_id).to_dict(orient='records')
        label_mask = np.zeros(image_size)
        for ann in annotations:
            mask = convert_to_mask(ann["segmentation"],
                                   height=image_dict['height'],
                                   width=image_dict['width'])
            label_mask[mask == 1] = (ann["category_id"] + 1) * scaling

        mask_filename = dest_annotation_dir / f'{image_fname_prefix}_annotation.png'
        cv2.imwrite(filename=str(mask_filename), img=label_mask)
