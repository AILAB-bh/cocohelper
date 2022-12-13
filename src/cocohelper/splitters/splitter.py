from abc import ABC, abstractmethod
from typing import List
from cocohelper import COCOHelper


class Splitter(ABC):

    @abstractmethod
    def apply(
            self,
            coco: COCOHelper
    ) -> List[COCOHelper]:
        """Split a COCODataset with the current splitting strategy.

        Args:
            coco: The dataset on which we want to apply the split.

        Returns:
            A list of COCOHelper datasets, subsets of the original set of data.
       """
