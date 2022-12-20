"""Other utilities.
"""
from typing import Any


def isArrayLike(obj):
    """
    Check if an object has attribute __iter__ and __len__.
    """
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def not_empty_intersect(
        a: set,
        b: set
) -> set:
    """
    Return intersection of a and b if both a and b are not empty sets.
    If one of those is an empty set, it returns the non-empty set. If both are empty set, returns an empty set

    Args:
        a: A python set.
        b: Another python set.

    Returns:
        Intersection of a and b if both a and b are not empty, otherwise the non-empty set.
    """
    a = set(a) if a is not None else set()
    b = set(b) if b is not None else set()
    if len(a) > 0:
        if len(b) > 0:
            return a.intersection(b)
        return a
    return b


def fix_not_tuple_object(obj_val: Any):
    """
    If object is not an array-like object, it will return a tuple containing that object.

    Args:
        obj_val: an array-like object or an object/value.

    Returns:
        a tuple containing the object/value or a conversion of the array-like object to a tuple.
    """
    if isArrayLike(obj_val):
        return tuple(obj_val)
    return tuple([obj_val])


def fix_not_tuple(*args):
    """
    Convert each argument to a tuple.

    Args:
        *args: an array-like objects or value/objects.

    Returns:
        for each argument, returns that argument converted to a tuple.
    """
    ret_values = []
    for value in args:
        ret_values.append(fix_not_tuple_object(value))
    return tuple(ret_values)

#
# def is_empty_array(iterable):
#     """ Return True if iterable is empty. """
#     return len(iterable) == 0
#
#
# def are_empty_arrays(*args):
#     """ """
#     ret = True
#     for arg in args:
#         ret = ret and is_empty_array(arg)
#     return ret
#
#

#
# def run_once(f):
#     """ Decorator to run a function only once. """
#     def wrapper(*args, **kwargs):
#         if not wrapper.has_run:
#             wrapper.has_run = True
#             return f(*args, **kwargs)
#         # else: the function will return None
#     wrapper.has_run = False
#     return wrapper


# def reindex_category_ids_for_file(file_path: Union[str, Path]):
#     """ Reindex category ids as range(num_categories) in the specified COCO annotation file
#
#
#     Args:
#         file_path:  the path to the COCO annotation file
#
#     Returns: COCO annotations with re-indexed categories
#
#     """
#     with open(file_path) as file:
#         data = json.load(file)
#
#     # 1. build the mapping between old and new category ids
#     n_categories = len(data["categories"])
#     new_categories_ids = list(range(n_categories))
#     categories_ids = new_categories_ids.copy()
#     for i, cat in enumerate(data["categories"]):
#         categories_ids[i] = cat["id"]
#     map_to_new_id = dict(zip(categories_ids, new_categories_ids))
#
#     # 2. substitute the old category ids with the new ones
#     for ann in data["annotations"]:
#         ann["category_id"] = map_to_new_id[ann["category_id"]]
#     for cat in data["categories"]:
#         cat["id"] = map_to_new_id[cat["id"]]
#
#     return data
#
