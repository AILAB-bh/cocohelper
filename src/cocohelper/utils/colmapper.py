"""
Column Mapper.
"""
import dataclasses
from typing import List, Dict


@dataclasses.dataclass(frozen=True)
class ColMap:
    orig: str
    new: str

    @property
    def to_new(self) -> Dict[str, str]:
        return {self.orig: self.new}

    @property
    def to_orig(self):
        return {self.new: self.orig}

    @classmethod
    def to_origs(
            cls,
            renamers: List["ColMap"]
    ) -> Dict[str, str]:
        return {rn.new: rn.orig for rn in renamers}

    def __eq__(self, other):
        return isinstance(other, ColMap) and other.orig == self.orig and other.new == self.new


@dataclasses.dataclass(frozen=True)
class ColsMapper:

    @property
    def all(self) -> List[ColMap]:
        return [renamer for renamer in self.__dict__.values()]

    def __post_init__(self):
        unique_new_names = []
        for rn in self.all:
            assert isinstance(rn, ColMap)
            unique_new_names.append(rn.new)
        assert len(unique_new_names) == len(set(unique_new_names))

    def __getitem__(
            self,
            item: str
    ) -> ColMap:
        return self.__dict__[item]

    @property
    def to_origs(self) -> dict:
        return ColMap.to_origs(self.all)

