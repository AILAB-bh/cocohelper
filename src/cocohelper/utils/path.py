"""Utilities* for path manipulation.
"""
from pathlib import Path
from typing import Union


def subtract(
        path_a: Union[str, Path],
        path_b: Union[str, Path]
) -> Union[Path, None]:
    """
    TODO documentation
    """
    path_a = str(Path(path_a))
    path_b = str(Path(path_b))
    first_common_idx = path_a.find(path_b)
    if first_common_idx >= 0:
        return Path(path_a[:first_common_idx])
    else:
        return None
