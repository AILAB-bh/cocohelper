"""
An adapter for converting datasets with binary masks to COCO format.
"""
from typing import List, Dict, Callable, Tuple, Optional, Union
from scipy import ndimage
import numpy as np
from cocohelper.utils.segmentation import encode_mask, compute_polygon_area
from cocohelper.adapters.dataset_adapter import DatasetAdapter


class BinaryMaskDatasetAdapter(DatasetAdapter):

    def __init__(
            self,
            data_paths: Dict[str, List[str]],
            image_loader: Callable[[str], np.ndarray],
            mask_loader: Callable[[str], np.ndarray],
            categories: Dict[int, dict],
            mode: str = 'polygon',
            compression_factor: float = 1.0
    ):
        """
        A DatasetAdapter to convert dataset with binary masks to COCO format.

        Args:
            data_paths: A dictionary that maps image filenames to its masks
                filenames.
            image_loader: A function to load an image.
            mask_loader: A function to load a mask.
            categories: A dictionary that maps mask value to a category.
            mode: How to encode the mask, defaults to polygon.
            compression_factor: Compression factor the encoded mask.
        """
        self._ann_id = 0
        self.data_paths = list(data_paths.items())
        self.image_loader = image_loader
        self.mask_loader = mask_loader
        self.categories = categories
        self.mode = mode
        self.compression_factor = compression_factor

    def get_categories(self) -> List[dict]:
        """
        Get the list of categories.

        Returns:
            A list of categories
        """
        return list(self.categories.values())

    def get_sample(
            self,
            idx: int
    ) -> Optional[Tuple[dict, List[dict]]]:
        """
        Get the COCO representation for a specific sample and its index.

        Args:
            idx: sample index.

        Returns:
            The values of `image`, and `image_annotations` in COCO format.
        """
        if idx >= len(self.data_paths):
            return None

        (image_path, annotation_list) = self.data_paths[idx]
        img = self.read_image(idx)
        height, width = img.shape[:2]
        image = {
            "id": idx,
            "file_name": image_path,
            "height": height,
            "width": width
        }

        image_annotations = []

        for annotation_path in annotation_list:
            annotations = self.mask_loader(annotation_path)

            segmentations, bounding_boxes, categories = \
                self.get_individual_instances(annotations, self.mode, self.compression_factor)

            for segm, bbox, cat_id in zip(segmentations, bounding_boxes, categories):
                self._ann_id += 1
                area = compute_polygon_area(segm)
                coco_annotation = {
                    "image_id": idx,
                    "id": self._ann_id,
                    "category_id": int(cat_id),
                    "bbox": bbox,
                    "area": area,
                    "segmentation": [segm],
                    "iscrowd": 0  # TODO crowd not supported
                }
                image_annotations.append(coco_annotation)
        return image, image_annotations

    def read_image(
            self,
            idx: int
    ) -> np.ndarray:
        """
        Reads an image in from its positional id in the data paths.

        Args:
            idx: image index.

        Returns:
            An image array corresponding to the given index in the data paths.
        """
        (image_path, annotation_list) = self.data_paths[idx]
        return self.image_loader(image_path)

    @staticmethod
    def extract_bbox_from_binary_mask(
            binary_mask: np.ndarray
    ) -> List:
        """
        Extracts bounding box from segmentation mask.

        NB: we do not support rotated bounding boxes.

        Args:
            binary_mask: binary semantic mask of an object.

        Returns:
            A bounding box surrounding the input semantic mask.
        """
        x_coords, y_coords = np.where(binary_mask)
        x = min(x_coords)
        y = min(y_coords)
        height = max(x_coords) - x
        width = max(y_coords) - y
        return [int(y), int(x), int(width), int(height)]

    def get_individual_instances(
            self,
            mask: np.ndarray,
            mode: str,
            compression_factor: Union[float, Tuple[float, float]],
            **kwargs
    ) -> Tuple[List, List, List]:
        """
        Separates disjoint objects inside the same array.

        Objects are separated based on two rules:
          1. objects have different labels in the input mask (e.g. one is
            associated with 1, the other with 2);
          2. objects that have the same label are disjoint in space, with
            structuring element as in `scipy.ndimage.label`.

        Args:
            mask: numpy array containing the semantic masks.
            mode: encoding mode for the semantic mask. Can be 'RLE', 'cRLE', or
                'polygon'.
            compression_factor: compression factor of the semantic map before
                conversion to COCO format. Use a factor > 1 to compress the
                segmentation mask s.t. its encoding does not occupy too much
                memory. The compression consists in a down-sampling of the mask
                array to a lower resolution before the subsequent conversion to
                COCO format.
            **kwargs: optional keyword parameters for the encoding.

        Returns:
            Segmentations, bounding boxes, and categories contained in the input
            array.
        """
        segmentations, bounding_boxes, categories = [], [], []

        uv = np.unique(mask)

        for class_val in uv:
            if class_val == 0:
                continue

            labeled_array, num_feature = ndimage.label((mask == class_val).astype(int), **kwargs)

            for label_id in range(1, num_feature + 1):
                instance = np.zeros_like(labeled_array)
                instance[labeled_array == label_id] = 1

                # encode segmentation in COCO format from numpy array
                segm = encode_mask(instance, mode=mode, compression_factor=compression_factor)
                segmentations.extend(segm)  # segm is already a list of polygons for the same class --> extend

                # compute bounding box from mask
                bbox = self.extract_bbox_from_binary_mask(instance)
                bounding_boxes.append(bbox)

                categories.append(self.categories[class_val]['id'])

        return segmentations, bounding_boxes, categories
