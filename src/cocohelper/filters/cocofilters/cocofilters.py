"""Special and complex filters used to obtain specific data from COCO `images`, `annotations` and `categories` tables.
"""
from typing import Type, List, Optional, Tuple
from cocohelper.filters.strategies import HAVING_VALUE, ValueFilterStrategy, RangeFilterStrategy, IN_RANGE
from cocohelper.filters.filter import ValueFilter, RangeFilter, ComposeFilter, Filter, ColumnFilter
from cocohelper.filters import AndFilter
from cocohelper.utils.types._types import IDXSelector


def anns_filter(
        ids: Optional[IDXSelector[int]] = None,
        area_rng: Optional[Tuple[float, float]] = None,
        is_crowd: Optional[bool] = None,
        composition: Type[ComposeFilter] = AndFilter,
        strategy: ValueFilterStrategy = HAVING_VALUE,
        rng_strategy: RangeFilterStrategy = IN_RANGE
) -> Filter:
    """
    A Filter for the annotations with the required characteristics.

    Args:
        ids: a filter for the annotation id.
        area_rng: a filter for the area of the annotation.
        is_crowd: a boolean for the `is_crowd` argument in the COCO annotation
        composition: a composition type for the filter (defaults to an AND
            between all the given conditions).
        strategy: a ValueFilterStrategy for the filter.
        rng_strategy: strategy for the range.

    Returns:
        A Filter for the annotations with the required characteristics.
    """
    f: List[ColumnFilter] = []
    if ids is not None:
        f.append(ValueFilter(ids, 'annotation_id', strategy))
    if area_rng is not None:
        f.append(RangeFilter(area_rng, 'area', rng_strategy))
    if is_crowd is not None:
        f.append(ValueFilter(is_crowd, 'iscrowd', strategy))
    if len(f) != 1:
        return composition(*f)
    return f[0]


def cats_filter(
        ids: Optional[IDXSelector[int]] = None,
        nms: Optional[IDXSelector[str]] = None,
        super_nms: Optional[IDXSelector[str]] = None,
        composition: Type[ComposeFilter] = AndFilter,
        strategy: ValueFilterStrategy = HAVING_VALUE
) -> Filter:
    """A Filter for the categories with the required characteristics.

    Args:
        ids: a filter for the category id.
        nms: a filter for the names of the categories.
        super_nms: a filter for the names of the super-categories.
        composition: a composition type for the filter (defaults to an AND
            between all the given conditions).
        strategy: a ValueFilterStrategy for the filter.

    Returns:
        A Filter for the categories with the required characteristics.
    """
    f = []
    if ids is not None:
        f.append(ValueFilter(ids, 'category_id', strategy))
    if nms is not None:
        f.append(ValueFilter(nms, 'name', strategy))
    if super_nms is not None:
        f.append(ValueFilter(super_nms, 'supercategory', strategy))
    if len(f) != 1:
        return composition(*f)
    return f[0]


def imgs_filter(
        ids: Optional[IDXSelector[int]] = None,
        nms: Optional[IDXSelector[str]] = None,
        composition: Type[ComposeFilter] = AndFilter,
        strategy: ValueFilterStrategy = HAVING_VALUE
) -> Filter:
    """A Filter for the images with the required characteristics.

    Args:
        ids: a filter for the image id.
        nms: a filter for the file names of the image.
        composition: a composition type for the filter (defaults to an AND
            between all the given conditions).
        strategy: a ValueFilterStrategy for the filter.

    Returns:
        A Filter for the images with the required characteristics.
    """
    f = []
    if ids is not None:
        f.append(ValueFilter(ids, 'image_id', strategy))
    if nms is not None:
        f.append(ValueFilter(nms, 'file_name', strategy))
    if len(f) != 1:
        return composition(*f)
    return f[0]
