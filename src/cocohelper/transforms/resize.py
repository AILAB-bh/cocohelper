"""
Resizing transformations for the COCO images and annotations.
"""
from typing import List, Tuple
import numpy as np
import cv2
from cocohelper.utils.segmentation import get_segmentation_mode, convert_to_mode
from cocohelper.transforms import Transform


# TODO: clean code in this file (there is a small margin of improvement)

class Resize(Transform):

    def __init__(
            self,
            size: List[int]
    ):
        if len(size) != 2:
            raise ValueError("Size must be a tuple of two elements.")
        self.size = size

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
        rs_img = self._resize_image_array(img, self.size)

        rs_annotations = []
        rs_ratios = [float(self.size[0]) / img.shape[0],
                     float(self.size[1]) / img.shape[1]]

        for a in anns:
            rs_bbox = self._resize_bbox(a["bbox"], ratios=rs_ratios)
            rs_area = Transform.compute_bbox_area(rs_bbox)

            segmentations = a["segmentation"]

            # Convert annotations to polygon
            # Resize the polygons
            # Convert them back to original format
            mode = get_segmentation_mode(segmentations)
            polygons = convert_to_mode(segmentations, 'polygon', img.shape[0], img.shape[1])
            rs_polygons = self._resize_segmentations(polygons, ratios=rs_ratios)
            rs_segmentations = convert_to_mode(rs_polygons, mode, img.shape[0], img.shape[1])

            rs_a = a.copy()
            rs_a["bbox"] = rs_bbox
            rs_a["area"] = rs_area
            rs_a["segmentation"] = rs_segmentations
            rs_annotations.append(rs_a)

        return rs_img, rs_annotations

    @staticmethod
    def _resize_image_array(
            image: np.ndarray,
            output_size: List[int]
    ) -> np.ndarray:
        """Resize image array to the given output size."""
        if image.shape[0] * image.shape[1] > output_size[0] * output_size[1]:
            interp = cv2.INTER_AREA  # area is the suggested interpolation when shrinking the image
            return cv2.resize(image, (output_size[1], output_size[0]), interpolation=interp)
        elif image.shape[0] * image.shape[1] < output_size[0] * output_size[1]:
            interp = cv2.INTER_CUBIC  # liner and cubic are the suggested interpolations when enlarging the image
            return cv2.resize(image, (output_size[1], output_size[0]), interpolation=interp)
        else:
            return image

    @staticmethod
    def _resize_bbox(
            bbox: List[int],
            ratios: List[float]
    ) -> List[int]:
        """Resize bounding box with the given ratios."""
        x, y, w, h = bbox
        return [int(round(x * ratios[1])),
                int(round(y * ratios[0])),
                int(round(w * ratios[1])),
                int(round(h * ratios[0]))]

    @staticmethod
    def _resize_segmentations(
            segmentations: List[List],
            ratios: List[float]
    ) -> List[List]:
        """Resize the segmentation with the given ratios."""
        rs_segmentations = []
        for segm in segmentations:
            x_coords = segm[0::2]
            y_coords = segm[1::2]
            rs_segm = []
            for x, y in zip(x_coords, y_coords):
                rs_segm.append(x * ratios[1])
                rs_segm.append(y * ratios[0])
            rs_segmentations.append(rs_segm)
        return rs_segmentations
