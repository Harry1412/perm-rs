from . import perm_rs
from .__settings import settings
from .perm import permanent, permanent_multi, permanent_single
from .perm_rs import performance_profiler

__doc__ = perm_rs.__doc__

__all__ = [
    "performance_profiler",
    "permanent",
    "permanent_multi",
    "permanent_single",
    "settings",
]
