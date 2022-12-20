"""Custom types.
"""
from typing import Union, Tuple, List, Sequence, TypeVar


# custom types aggregating subtypes:
Number = Union[int, float]
BBox = Tuple[int, int, int, int]
Segment = List[List[int]]
Shape = Tuple[int, ...]


T = TypeVar("T")
IDXSelector = Union[Sequence[T], T]
