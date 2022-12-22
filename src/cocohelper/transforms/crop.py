"""
Several Crop transformations for the COCO images and annotations.
"""
from typing import List, Tuple
from enum import Enum
import numpy as np
import logging
import random
from cocohelper.transforms import Transform
from cocohelper.utils import types


class SizeMode(Enum):
    pixel = 0
    percentage = 1


def _check_bbox(
        x: int = 0,
        y: int = 0,
        w: int = 0,
        h: int = 0,
        mode=SizeMode.pixel
):
    assert x >= 0
    assert y >= 0
    assert w > 0
    assert h > 0
    if mode == SizeMode.percentage:
        max_x = x + w
        max_h = y + h
        assert max_x <= 1
        assert max_h <= 1


def _norm_bbox(
        xywh: types.BBox,
        img_shape: Tuple[int, ...],
        mode=SizeMode.pixel
) -> types.BBox:
    x, y, w, h = xywh
    if mode == SizeMode.percentage:
        x = int(x * img_shape[1])
        y = int(y * img_shape[0])
        w = int(w * img_shape[1])
        h = int(h * img_shape[0])
    return x, y, w, h


def crop_img(
        img: np.ndarray,
        anns: List[dict],
        xywh: types.BBox
) -> Tuple[np.ndarray, List[dict]]:
    x, y, w, h = xywh

    if x < 0:
        logging.warning('CROP: clipping x to 0')
        x = 0

    if y < 0:
        logging.warning('CROP: clipping y to 0')
        y = 0

    if (x + w) > img.shape[1]:
        logging.warning('CROP: clipping width to max')
        w = img.shape[1] - x

    if (y + h) > img.shape[0]:
        logging.warning('CROP: clipping height to max')
        h = img.shape[0] - h

    img = img[y:y + h, x:x + w, :]
    r_anns = []

    for ann in anns:
        r_ann = ann.copy()
        r_ann['bbox'] = _crop_bbox(r_ann['bbox'], (x, y, w, h))
        r_ann['area'] = Transform.compute_bbox_area(r_ann['bbox'])

        if r_ann['area'] <= 0:
            continue

        # TODO: check segmentation format, convert to polygon if needed and convert back to the original.
        r_ann['segmentation'] = _crop_segmentation(r_ann['segmentation'], (x, y, w, h))
        r_anns.append(r_ann)

    return img, r_anns


def _crop_bbox(
        bbox: List[int],
        xywh: types.BBox
) -> List[int]:
    crop_x, crop_y, crop_w, crop_h = xywh
    x, y, w, h = bbox

    f_x = x - crop_x
    f_y = y - crop_y

    if f_x < 0:
        f_w = w + f_x
        f_x = 0
    else:
        f_w = w

    if f_y < 0:
        f_h = h + f_y
        f_y = 0
    else:
        f_h = h

    if (f_x + f_w) > crop_w:
        f_w = crop_w - f_x

    if (f_y + f_h) > crop_h:
        f_h = crop_h - f_y

    return [f_x, f_y, f_w, f_h]


def _crop_segmentation(
        segmentations: list,
        xywh: types.BBox
) -> list:
    crop_x, crop_y, crop_w, crop_h = xywh
    rs_segmentations = []
    for segm in segmentations:
        x_coords = segm[0::2]
        y_coords = segm[1::2]

        rs_segm = []
        for x, y in zip(x_coords, y_coords):
            f_x = max(x - crop_x, 0)
            f_y = max(y - crop_y, 0)

            f_x = min(f_x, crop_w)
            f_y = min(f_y, crop_h)

            rs_segm.append(f_x)
            rs_segm.append(f_y)

        rs_segmentations.append(rs_segm)
    return rs_segmentations


class Crop(Transform):

    def __init__(
            self,
            xywh: types.BBox,
            mode: SizeMode = SizeMode.pixel
    ):
        """
        Perform a crop on image and annotations.

        Args:
            xywh: The bbox of the area to crop.
            mode: How to handle bbox values, pixels or percentages.
        """
        if len(xywh) != 4:
            raise ValueError("The input xywh must be a sequence of 4 values.")
        x, y, w, h = xywh
        _check_bbox(x, y, w, h, mode)
        self.crop = xywh
        self.mode = mode

    def apply(
            self,
            img: np.ndarray,
            anns: List[dict]
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Apply the transformation to the image array and its annotations.

        Args:
            img: image array.
            anns: annotations for this image.

        Returns:
            Transformed image array and annotations.
        """
        _crop: types.BBox = _norm_bbox(self.crop, img.shape, self.mode)
        return crop_img(img, anns, _crop)


class RandomCrop(Transform):

    def __init__(
            self,
            w: int,
            h: int,
            mode: SizeMode = SizeMode.pixel
    ):
        """
        Perform a random crop on image and annotations.

        You must provide width and height, while the crop position is randomized.

        Args:
            w: The fixed width of the area to crop.
            h: The fixed height of the area to crop.
            mode: How to handle bbox values, pixels or percentages.
        """
        _check_bbox(w=w, h=h, mode=mode)
        self.w = w
        self.h = h
        self.mode = mode

    def apply(
            self,
            img: np.ndarray,
            anns: List[dict]
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Apply the transformation to the image array and its annotations.

        Args:
            img: image array.
            anns: annotations for this image.

        Returns:
            Transformed image array and annotations.
        """
        _, _, w, h = _norm_bbox((0, 0, self.w, self.h), img.shape, self.mode)
        max_x = img.shape[1] - w
        max_y = img.shape[0] - h
        x = int(random.uniform(0, max_x))
        y = int(random.uniform(0, max_y))
        return crop_img(img, anns, (x, y, w, h))


class CenterCrop(Transform):

    def __init__(
            self,
            w: int,
            h: int,
            mode: SizeMode = SizeMode.pixel
    ):
        """
        Perform a crop around the center of the image.

        Args:
            w: The width of the area to crop.
            h: The height of the area to crop.
            mode: How to handle bbox values, pixels or percentages.
        """
        _check_bbox(w=w, h=h, mode=mode)
        self.w = w
        self.h = h
        self.mode = mode

    def apply(
            self,
            img: np.ndarray,
            anns: List[dict]
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Apply the transformation to the image array and its annotations.

        Args:
            img: image array.
            anns: annotations for this image.

        Returns:
            Transformed image array and annotations.
        """
        _, _, w, h = _norm_bbox((0, 0, self.w, self.h), img.shape, self.mode)
        x = img.shape[1] // 2 - (w // 2)
        y = img.shape[0] // 2 - (h // 2)
        return crop_img(img, anns, (x, y, w, h))
