"""Base filters that can be used, composed or extended to extract specific information from dataframes.
"""
from .filter import Filter
from .filter import ComposeFilter, AndFilter, OrFilter, NotFilter
from .filter import ColumnFilter, ValueFilter, RangeFilter


AND = AndFilter
OR = OrFilter
NOT = NotFilter
