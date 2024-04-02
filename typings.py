from typing import Any, Union, Iterator, Type
from typing_extensions import Protocol

from numpy.typing import ArrayLike

class DiabaticModelT(Protocol):
    ndim_: int
    nstates_: int
    mass: ArrayLike

    def __init__(self, representation: str, *args: Any, **kwargs: Any):
        pass

    def V(self, x: ArrayLike) -> ArrayLike:
        pass

    def dV(self, x: ArrayLike) -> ArrayLike:
        pass