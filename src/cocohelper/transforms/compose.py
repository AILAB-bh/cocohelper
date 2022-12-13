from typing import List, Tuple
import numpy as np
from cocohelper.transforms import Transform


class Compose(Transform):

    def __init__(
            self,
            transforms: List[Transform]
    ):
        """Combine different Transform into one.

        Args:
            transforms: The list of Transform to combine.
        """
        self._transforms = transforms.copy()

    def append(
            self,
            transform: Transform
    ):
        """Append transformations to be applied to the COCO data.

        Args:
            transform: a Transformation to be applied.

        Returns:
            self.
        """
        self._transforms.append(transform)
        return self

    def insert(
            self,
            transform: Transform,
            index: int = 0):
        self._transforms.insert(index, transform)
        return self

    def reverse(self):
        self._transforms.reverse()
        return self

    def clear(self):
        self._transforms.clear()
        return self

    def remove(
            self,
            transform: Transform
    ):
        self._transforms.remove(transform)
        return self

    def pop(self,
            index: int = -1
            ):
        return self._transforms.pop(index)

    def __getitem__(
            self,
            key
    ):
        return self._transforms[key]

    def __repr__(self):
        return self._transforms.__repr__()

    def apply(
            self,
            img: np.ndarray,
            anns: List[dict]
    ) -> Tuple[np.ndarray, List[dict]]:
        """Apply the transformation to the image array and its annotations.

        Args:
            img: image array.
            anns: annotations for this image.

        Returns:
            Transformed image array and annotations.
        """
        for t in self._transforms:
            img, anns = t.apply(img, anns)

        return img, anns
