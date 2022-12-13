from typing import List
import random
from cocohelper.splitters.splitter import Splitter
from cocohelper import COCOHelper


class ProportionalDataSplitter(Splitter):

    def __init__(
            self,
            *proportions: float
    ):
        """Split a COCO dataset into N datasets.

        Each split will contain a part of the original dataset samples
        proportionally to the arguments.

        Args:
            *proportions: Describe the split proportions for each split.
        """
        if len(proportions) <= 1:
            raise ValueError("A ProportionalDataSplitter requires a number of proportion values > 1.")

        prop_tot = sum(proportions)
        self.proportions = [float(v) / prop_tot for v in proportions]

    def apply(
            self,
            coco: COCOHelper
    ) -> List[COCOHelper]:
        """Applies the splitter to the given COCOHelper.

        Args:
            coco: a COCOHelper containing the source dataset to be split.

        Returns:
            A list containing the splits of the source COCO dataset.
        """
        ids = self._get_ids(coco)

        splits = []

        for _ids in ids:
            splits.append(coco.filter_imgs(img_ids=_ids))

        return splits

    def _get_ids(
            self,
            ch: COCOHelper
    ) -> List:
        """For each split, GET a list of image ids for a target COCOHelper
        dataset, respecting the splitting proportions.

        Args:
            ch: target COCOHelper dataset.

        Returns:
            A list of indices lists (a list of indices per each split).
        """
        imgs_anns = ch.joins.imgs_anns.fillna(-1)
        ids_by_label = {k: list(set(v)) for k, v in imgs_anns.reset_index().groupby('category_id')['image_id']}

        list_of_image_ids = []
        for images in ids_by_label.values():
            for img in images:
                list_of_image_ids.append(img)
        list_of_image_ids = list(set(list_of_image_ids))
        random.shuffle(list_of_image_ids)

        n_images = len(list_of_image_ids)
        n_samples = [int(round(v * n_images)) for v in self.proportions]
        ids_subset_images: List[List[int]] = [list() for _ in self.proportions]
        id_list = list_of_image_ids
        for k, n in enumerate(n_samples):
            left, right = id_list[:n], id_list[n:]
            ids_subset_images[k] = left
            id_list = right

        return ids_subset_images
