"""Filters for the COCO dataset.
"""
from .filter import Filter
from .filter import ComposeFilter, AndFilter, OrFilter, NotFilter
from .filter import ColumnFilter, ValueFilter, RangeFilter


AND = AndFilter
OR = OrFilter
NOT = NotFilter
