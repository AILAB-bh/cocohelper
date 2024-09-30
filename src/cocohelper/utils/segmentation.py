"""
Utilities* for converting segmentation annotations between different formats.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Union, Optional
import numpy.typing as npt
from pycocotools import mask as coco_mask
from pathlib import Path
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
import numpy as np
import base64
import zlib
import cv2
from cocohelper import COCOHelper


class MaskConverter(ABC):
    @abstractmethod
    def encode(self, mask: np.ndarray, **kwargs) -> Any:
        pass

    @abstractmethod
    def decode(self, code: Any, **kwargs) -> np.ndarray:
        pass


class RLEMaskConverter(MaskConverter):
    def encode(self, mask: np.ndarray, **kwargs) -> Any:
        """
        Converts a binary mask to RLE encoding.
        Args:
            mask: a binary mask as a numpy array.
            **kwargs: extra parameters are ignored.

        Returns:
            The RLE encoding of the binary mask.
        """
        return mask_to_rle(mask, **kwargs)

    def decode(self, code: Any, **kwargs) -> np.ndarray:
        """
        Converts an RLE to a binary mask with the given output dtype.
        Args:
            code: RLE encoding of the semantic mask.
            **kwargs: extra parameters such as:
                - `dtype` to indicate the type of the output mask.

        Returns:
            The RLE mask as a numpy array.
        """
        return rle_to_mask(code, **kwargs)


class CompressedRLEMaskConverter(MaskConverter):
    def encode(self, mask: np.ndarray, **kwargs) -> Any:
        """
        Converts a binary mask to compressed RLE format.
        Args:
            mask: a binary mask to encode.
            **kwargs: extra parameters are ignored.

        Returns:
            A string encoding the mask as compressed RLE.
        """
        return mask_to_compressed_rle(mask, **kwargs)

    def decode(self, code: Any, **kwargs) -> np.ndarray:
        """
        Converts a compressed RLE to a binary mask with the given output dtype.
        Args:
            code: RLE encoding of the semantic mask.
            **kwargs: extra parameters such as:
                - `height` and `width` to indicate the shape of the output mask.
                - `dtype` to indicate the type of the output mask.

        Returns:
            The compressed RLE mask as a numpy array.
        """
        return compressed_rle_to_mask(code, **kwargs)


class PolygonMaskConverter(MaskConverter):
    def encode(self, mask: np.ndarray, **kwargs) -> Any:
        """
        Converts segmentation mask to a list of polygons.
        Args:
            mask: numpy array containing multiple segmentation masks. Each mask must
            **kwargs: extra parameters are ignored.

        Returns:
            List of polygons segmentation masks.
        """
        return mask_to_polygon(mask, **kwargs)

    def decode(self, code: Any, **kwargs) -> np.ndarray:
        """
        Creates and returns a mask from polygon in COCO format.
        Args:
            code: polygon coordinates.
            **kwargs: extra parameters such as:
                - `width` and `height` to indicate the shape of the output mask.
                - `value` to indicate the value of the zero-valued pixels in the output mask.
                - `dtype` to indicate the type of the output mask.

        Returns:
            The polygon mask as a numpy array.
        """
        return polygon_to_mask(code, **kwargs)


# Mapping of segmentation modes to their respective converters
MASK_CONVERTERS = {
    'RLE': RLEMaskConverter(),
    'cRLE': CompressedRLEMaskConverter(),
    'polygon': PolygonMaskConverter()
}


def encode_mask(mask: np.ndarray, mode: str, **kwargs) -> Any:
    """
    Encodes a mask using the specified
    Args:
        mask: a binary mask as a numpy array.
        mode: the mode to use for encoding the mask.
        **kwargs: extra parameters are ignored.

    Returns:
        The encoded mask.
    """
    return MASK_CONVERTERS[mode].encode(mask, **kwargs)


def decode_mask(code: Any, mode: str, **kwargs) -> np.ndarray:
    """
    Decodes a mask using the specified mode.
    Args:
        code: the encoded mask.
        mode: the mode to use for decoding the mask.
        **kwargs: extra parameters such as:
            - `height` and `width` to indicate the shape of the output mask (if mode in [cRLE, polygon]).
            - `value` to indicate the value of the zero-valued pixels in the output mask (if mode=polygon).
            - `dtype` to indicate the type of the output mask (if mode in [RLE, cRLE, polygon]).

    Returns:
        The decoded mask as a numpy array.
    """
    return MASK_CONVERTERS[mode].decode(code, **kwargs)


def convert_to_mask(segmentation: Any, height: int, width: int, **kwargs) -> np.ndarray:
    """
    Converts a segmentation to a binary mask.
    Args:
        segmentation: the segmentation to convert.
        height: the height of the output mask.
        width: the width of the output mask.
        **kwargs: extra parameters

    Returns:
        The binary mask as a numpy array.
    """
    mode = get_segmentation_mode(segmentation)
    return decode_mask(segmentation, mode, height=height, width=width, **kwargs)


def convert_to_mode(segmentation: Any, mode: str, height: int, width: int, **kwargs) -> Any:
    """
    Converts a segmentation to the specified mode.
    Args:
        segmentation: the segmentation to convert.
        mode: the mode to convert the segmentation to.
        height: the height of the output mask.
        width: the width of the output mask.
        **kwargs: extra parameters

    Returns:
        The segmentation in the specified mode.
    """
    curr_mode = get_segmentation_mode(segmentation)
    if curr_mode == mode:
        return segmentation
    mask = decode_mask(segmentation, curr_mode, height=height, width=width)
    return encode_mask(mask, mode, **kwargs)


def get_segmentation_mode(
        segmentation: Union[List[List], Dict, str]
) -> str:
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


def compressed_rle_to_mask(
        rle_code: str,
        height: int = 512,
        width: int = 512,
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


def mask_to_polygon(
        mask: np.ndarray,
        simplify_tolerance: float = 1.0,
        **kwargs
) -> List:
    """
    Converts segmentation mask to a list of polygons.

    To reduce the number of vertices in the obtained polygon, try to increase
    the value of `polygon_simplify_tolerance`.

    Args:
        mask: numpy array containing multiple segmentation masks. Each mask must
            be associated with a different number, where 0 is for background,
            and other numbers related to different objects. For example, objects
            of a class A may be associated with a value of 1, class B to values
            10, class C to 255, and so on.
        simplify_tolerance: a tolerance value used to remove redundant
            vertexes for the polygons extracted from the mask.
        **kwargs: extra parameters are ignored.

    Returns:
        List of polygons segmentation masks.
    """
    polygons = []
    classes = list(np.unique(mask))
    if 0 in classes:
        classes.remove(0)

    # iterate over the different segmentations (classes) inside the mask:
    for cls in classes:
        # pick the class-related segmentation and convert it to binary:
        binary = np.zeros_like(mask)
        binary[mask == cls] = 1

        # compute mask contours and then convert them to polygons:
        binary = binary.astype(np.uint8)
        contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        # simplify contours
        contours = [cnt.reshape((-1, 2)) for cnt in contours]

        for cnt in contours:
            sqz_cnt = cnt
            if cnt.shape[1] == 1:
                sqz_cnt = np.squeeze(sqz_cnt, axis=1)

            if len(sqz_cnt) >= 3:  # a polygon must contain at least 3 points
                polygon = Polygon(sqz_cnt)
                polygon = polygon.simplify(tolerance=simplify_tolerance, preserve_topology=True)
                polygon = polygon.exterior.coords

                # get coordinates
                x_coords = [c[0] for c in polygon]
                y_coords = [c[1] for c in polygon]
                sgm = []
                for x, y in zip(x_coords, y_coords):
                    sgm.extend([float(x), float(y)])
                polygons.append(sgm)

    return polygons


def polygon_to_mask(
        polygon_code: List[List],
        width: int = 512,
        height: int = 512,
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
) -> None:
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
