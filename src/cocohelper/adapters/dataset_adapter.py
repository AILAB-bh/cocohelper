"""
The abstraction of an Adapter for converting datasets an arbitrary format to COCO format.
"""
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np


class DatasetAdapter(ABC):

    def __iter__(self):
        """
        DatasetAdapter interface that converts a dataset format to COCOHelper
        format.

        Returns:
            self.
        """
        self.idx = 0
        return self

    def __next__(self):
        """
        Get the next sample in the sequence.

        Returns:
            self.

        Raises:
            StopIteration when the sequence is exhausted.
        """
        sample = self.get_sample(self.idx)

        if sample is None:
            raise StopIteration

        self.idx += 1
        return sample

    @abstractmethod
    def get_categories(self) -> List[dict]:
        """
        Get the categories.

        Returns:
            A list of categories as dictionaries.
        """
        pass

    @abstractmethod
    def get_sample(
            self,
            idx: int
    ) -> Optional[Tuple[dict, List[dict]]]:
        """
        A method for loading samples. To be implemented.

        Args:
            idx: index of the element to be fetched.

        Returns:
            A tuple of a dictionary or list of dictionaries.
        """
        pass

    @abstractmethod
    def read_image(
            self,
            idx: int
    ) -> np.ndarray:
        """
        A method for reading images. To be implemented.

        Args:
            idx: index of the element to be fetched.

        Returns:
            A numpy array with the image.
        """
        pass
