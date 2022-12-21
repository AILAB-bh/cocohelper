"""
Flipping transformations for the COCO images and annotations.
"""
from typing import List, Tuple
import numpy as np
import random
from cocohelper.transforms import Transform
from cocohelper.utils import types


class RandomFlip(Transform):

    def __init__(
            self,
            horizontal_prob: float = 0.5,
            vertical_prob: float = 0.0
    ):
        """
        Flip images and annotations in vertical and horizontal axes.

        Args:
            horizontal_prob: probability of the horizontal flip occurring [0-1].
            vertical_prob: probability of the vertical flip occurring [0-1].
        """
        self.horizontal_prob = horizontal_prob
        self.vertical_prob = vertical_prob

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
        flip_h = random.random() < self.horizontal_prob
        flip_v = random.random() < self.vertical_prob

        if flip_v or flip_h:
            img = self._flip_img(img, flip_h, flip_v)
            f_anns = []
            for ann in anns:
                f_ann = ann.copy()
                f_ann['bbox'] = self._flip_bbox(f_ann['bbox'], img.shape, flip_h, flip_v)
                f_ann['segmentation'] = self._flip_segmentations(f_ann['segmentation'], img.shape, flip_h, flip_v)
                f_anns.append(f_ann)
            anns = f_anns

        return img.copy(), anns

    @staticmethod
    def _flip_img(
            img: np.ndarray,
            horizontal: bool,
            vertical: bool
    ) -> np.ndarray:
        if horizontal:
            img = np.fliplr(img)
        if vertical:
            img = np.flipud(img)

        return img

    @staticmethod
    def _flip_bbox(
            bbox: types.BBox,
            shape: types.Shape,
            horizontal: bool,
            vertical: bool
    ) -> types.BBox:
        h, _, _ = shape
        x, y, w, h = bbox
        if vertical:
            y_max = y + h
            y = h - y_max
        if horizontal:
            x_max = x + w
            x = w - x_max
        return x, y, w, h

    @staticmethod
    def _flip_segmentations(
            segmentation: types.Segment,
            shape: types.Shape,
            horizontal: bool,
            vertical: bool
    ) -> types.Segment:
        h, w, _ = shape

        # TODO: check segmentation format, convert to polygon if needed and convert back to the original.
        f_segmentations = []
        for segm in segmentation:
            x_coords = segm[0::2]
            y_coords = segm[1::2]

            f_segm = []
            for x, y in zip(x_coords, y_coords):
                if horizontal:
                    x = w - x
                if vertical:
                    y = h - y
                f_segm.append(x)
                f_segm.append(y)

            f_segmentations.append(f_segm)
        return f_segmentations
