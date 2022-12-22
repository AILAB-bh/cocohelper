"""
Split the COCO dataset according to a K-fold rule.
"""
from typing import List
import random
from cocohelper.splitters.proportional import ProportionalDataSplitter
from cocohelper.splitters.stratified import StratifiedDataSplitter
from cocohelper.splitters.splitter import Splitter
from cocohelper import COCOHelper


class KFoldSplitter(Splitter):

    def __init__(
            self,
            n_fold: int,
            stratified: bool = False
    ):
        """
        Split a COCO dataset into n datasets.

        Args:
            n_fold: Defines the number of folds to be used for k-fold
                cross-validation.
            stratified: If True the dataset is stratified.
        """
        if n_fold <= 1:
            raise ValueError("The number of folds must be greater than 1.")

        self.n_fold = n_fold
        self.stratified = stratified
        self._splitter: Splitter

        proportions = tuple([1] * n_fold)
        if stratified:
            self._splitter = StratifiedDataSplitter(*proportions)
        else:
            self._splitter = ProportionalDataSplitter(*proportions)

    def iter(
            self,
            coco: COCOHelper
    ):
        """
        Used to iterate over the dataset splits.

        Args:
            coco: a COCOHelper to be iterated over.

        Returns:
            A COCOHelper with filtered image ids.
        """
        splits = self.apply(coco)
        random.shuffle(splits)

        for split in splits:
            split_imgs = split.imgs.index
            train_imgs = coco.imgs[~coco.imgs.index.isin(split_imgs)]
            train_imgs_ids = list(train_imgs.index)
            yield coco.filter_imgs(img_ids=train_imgs_ids), split

    def apply(
            self,
            coco: COCOHelper
    ) -> List[COCOHelper]:
        """
        Applies the splitter to the given COCOHelper.

        Args:
            coco: a COCOHelper containing the source dataset to be split.

        Returns:
            A list containing the splits of the source COCO dataset.
        """
        return self._splitter.apply(coco)
