import logging
from functools import cached_property
from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
from scipy import stats
from cocohelper import COCOHelper


class COCOStats:

    def __init__(self, coco_helper: COCOHelper):
        """This class contains methods to calculate stats on a dataset.

        Args:
            coco_helper: Coco dataset to calculate stats on.
        """
        self._coco_helper: COCOHelper = coco_helper

    @property
    def coco_helper(self) -> COCOHelper:
        return self._coco_helper

    def get_image_size_stats(
            self,
            eps: float = 1e-16
    ) -> Dict:
        """Obtain statistics about dataset images sizes for each individual axis.

        Args:
            eps: epsilon used to compute average height/width ratio.

        Returns:
            Size information about dataset images in the form of a dictionary.
        """
        # TODO: compute each of these information once offering a property-getter and using cached properties.
        #       NB: our target is to make COCOHelper an immutable class, so computed statistics can't be invalidated.
        #       NB: this means that we must avoid inplace operations and close the class to modifications
        #           (we should never return a reference to something contained in the class, but a referenco to a copy
        #           of that)
        imgs = self._coco_helper.imgs
        size_array = np.stack([imgs.height, imgs.width], axis=1)  # .astype(int)

        metrics = {
            "min": list(np.min(size_array, axis=0)),
            "max": list(np.max(size_array, axis=0)),
            "mean": list(np.mean(size_array, axis=0)),
            "median": list(np.median(size_array, axis=0)),
            "std": list(np.std(size_array, axis=0)),
            "mode": list(stats.mode(size_array, axis=0, keepdims=False)[0]),
            "iqr": list(stats.iqr(size_array, axis=0)),
            "skewness": list(stats.skew(size_array, axis=0)),
            "kurtosis": list(stats.kurtosis(size_array, axis=0)),
            # "mean height/width ratio": np.mean([x / (y + eps) for (x, y) in size_list])
            "avg_size_ratio": np.mean([x / (y + eps) for (x, y) in size_array])
        }

        return metrics

    def get_annotation_size_stats(
            self,
            mode: str = "bbox"
    ) -> Dict:
        """Obtain dataset labels size statistics in the form of a dictionary.

        The dictionary pairs each dataset size in the dataset to the list of the
        smallest bounding box size inside the image. This can be useful to define
        the optimal rescaling of the image and avoid loosing small boxes when
        resizing data to a smaller dimension.

        Args:
            mode: annotation type to be used to extract size statistics. Default
                to bbox.

        Returns:
            A dictionary with the statistics of the label size in the dataset.
        """
        # TODO: restructure the code, move to COCOHelper class using a getter on ann_id ?
        if mode not in ["bbox"]:
            raise ValueError("Modes different from 'mode' are not currently supported.")

        def _get_size_from_bbox(annotation):
            _, _, width, height = annotation[mode]
            return width, height

        annotation_size_by_image_size: Dict[Tuple[int, int], List[Tuple[int, int]]] = dict()
        for idx, image in self._coco_helper.imgs.iterrows():
            annotations = self._coco_helper.filtered_anns(img_ids=idx)

            # if there is at least one annotation: add it to the stats
            if len(annotations) > 0:

                widths, heights = zip(*[_get_size_from_bbox(a) for a in annotations.iloc])
                min_width = min(widths)
                min_height = min(heights)

                # set image size and width ad dictionary key:
                key = (image["height"], image["width"])
                if key not in annotation_size_by_image_size.keys():
                    annotation_size_by_image_size[key] = [(min_height, min_width)]
                else:
                    annotation_size_by_image_size[key].append((min_height, min_width))

        return annotation_size_by_image_size

    def get_optimal_image_size(
            self,
            mode: str = "median",
            n_pixels: int = 4
    ) -> Tuple:
        """Estimate optimal image size based on the dataset's image and
        annotation statistics.

        This function:
          1. Computes the label stats of the dataset paired to each image size.
          2. Computes the minimum image size that guarantees images resampled to
            that resolution do not lose labels.
          2. Computes the statistic defined by the argument mode (mode can be in
            ["mean", "median", "mode"]).
          4. returns the maximum between the minimum image size that guarantees
            to maintain an annotation size of at least n_pixels on both the width
            and the height image axis AND the values computed in (3).

        Args:
            mode: statistics to be used to define the optimal image size.
            n_pixels: minimum number of annotation pixels remaining over the
                width and height axis after resizing an image to the returned the
                optimal image size

        Returns:
            A tuple containing the optimal height and width for resizing all the
            images of the dataset to the same dimension while preserving label
            information.
        """
        if n_pixels < 1:
            raise ValueError("The minimum number of pixels must be >= 1.")

        logging.info(f"Computing the recommended image size values. "
                     f"The recommended size ensures that, after resizing, the annotations will have thickness of at "
                     f"least n_pixels: {n_pixels}. Statistics on the image size are computed in mode: {mode}."
                     f"\nProcessing...")

        # get info on the smaller annotation dimension for each image size:
        annotation_size_by_image_size = self.get_annotation_size_stats(mode="bbox")
        smaller_annotation_size_by_image_size = dict()
        for key in annotation_size_by_image_size.keys():
            smaller_annotation_size_by_image_size[key] = list(np.min(annotation_size_by_image_size[key], axis=0))

        # compute maximum compression ratio for each image size (given the smallest annotation within sizes)
        min_compressed_dims = dict()
        for (image_h, image_w), (annotation_h, annotation_w) in smaller_annotation_size_by_image_size.items():
            min_compressed_dims[(image_h, image_w)] = (n_pixels * float(image_h) / annotation_h,
                                                       n_pixels * float(image_w) / annotation_w)

        # get lower bound as the maximum value among the min compressed dims
        min_h, min_w = np.max(list(min_compressed_dims.values()), axis=0)

        # compute image size as the upper value between (min_h, min_w) and the median dimension in the dataset
        stat_h, stat_w = self.get_image_size_stats()[mode]

        optimal_h, optimal_w = int(max(stat_h, min_h)), int(max(stat_w, min_w))
        logging.info(f"Done. Optimal values are: ({optimal_h}, {optimal_w}).")
        return optimal_h, optimal_w

    @property
    def nb_imgs(self) -> int:
        """Number of images in the dataset"""
        return len(self._coco_helper.imgs)

    @property
    def nb_cats(self) -> int:
        """Number of categories in the dataset"""
        return len(self._coco_helper.cats)

    @property
    def nb_anns(self) -> int:
        """Number of annotations in the dataset"""
        return len(self._coco_helper.anns)

    def __get_annotations_ratios(
            self,
            col: str,
            na_value: Any = pd.NA
    ) -> dict:
        """Get the ratios of annotations for each value in column.

        The function picks the imgs/anns/cats join and uses value_counts to
        return a dict with the ratio for each value. Values are normalized in
        [0, 1] by the argument `normalize=True`. Missing values in the column
        will be replaced by `na_value`.

        Returns:
            Dict associating each value in col with the fraction of annotations
            (in [0, 1]).
        """
        _df = self._coco_helper.joins.imgs_anns_cats
        _df = _df[col].fillna(na_value)
        _df = _df.value_counts(normalize=True)
        return _df.to_dict()

    @cached_property
    def cat_nms_ratios(self) -> Dict:
        """For each category, compute how many images in the dataset contains at
        least an annotation for that category.

        Returns:
            A dictionary associating to each category name the fraction of images
            in the dataset containing at least an annotation of that category
            (in [0, 1]).
        """
        return self.__get_annotations_ratios('category_name', na_value='<NA>')

    @cached_property
    def cat_ids_ratios(self) -> Dict:
        """For each category, compute how many images in the dataset contains at
        least an annotation for that category.

        Returns:
            A dictionary associating to each category id the fraction of images
            in the dataset containing at least an annotation of that category
            (in [0, 1]).
        """
        return self.__get_annotations_ratios('category_id', na_value=-1)
