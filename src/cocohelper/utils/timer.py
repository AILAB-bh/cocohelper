"""A timer.
"""
# -*- encoding: utf-8 -*-
# ! python3
import logging
import time
from contextlib import ContextDecorator
from typing import Callable, Any


__all__ = ["Timer"]


class Timer(ContextDecorator):
    """Timer class that can be used both as a context manager and function/method decorator.

    Usage:

    >>> import math
    >>> import time
    >>>
    >>> with Timer():
    ...    for i in range(42):
    ...        print("{}! = {:.5}...".format(i**2, str(math.factorial(i**2))))
    >>>
    >>> @Timer(end_msg="Second")
    ... def some_func():
    ...     time.sleep(1)
    >>>
    >>> with Timer(start_msg="Starting operation.", end_msg="Operation executed.") as t:
    ...    for i in range(42):
    ...        print("{}! = {:.5}...".format(i**2, str(math.factorial(i**2))))
    >>>
    >>> print(t.elapsed)
    """

    def __init__(
            self,
            start_msg: str = "",
            end_msg: str = "done.",
            log_fn: Callable[[str], Any] = logging.info
    ):
        """Instantiate new Timer.
        
        Args:
            start_msg: message to print before starting the operation.
            end_msg: message (prefix) to print at the end of operation.
            log_fn: logging function.
        """

        self._start_msg = start_msg
        self._end_msg = end_msg
        self._elapsed = 0
        self._log_fn = log_fn

    def __float__(self) -> float:
        return float(self.elapsed)

    def __str__(self) -> str:
        """An “informal” or nicely printable string representation of an object."""
        return "Elapsed {}".format(self.__repr__())

    def __repr__(self) -> str:
        """The “official” string representation of an object."""
        return str(float(self))

    def __enter__(self) -> 'Timer':
        if self._log_fn is not None and self._start_msg is not None:
            self._log_fn(f"[{self._start_msg}]")

        self.start = time.perf_counter()

        return self

    def __exit__(self, *args):
        self._elapsed = time.perf_counter() - self.start

        if self._log_fn is not None:
            if self._end_msg:
                title = f"[{self._end_msg}] "
            else:
                title = ""

            if self._log_fn is not None:
                self._log_fn(f'{title}Total time {self._elapsed:.5f} seconds.')

    @property
    def elapsed(self) -> float:
        return self._elapsed
