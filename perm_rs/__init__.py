from .__settings import settings
from .perm import permanent, permanent_multi, permanent_single

__doc__ = perm_rs.__doc__  # noqa: F821

__all__ = ["permanent", "permanent_multi", "permanent_single", "settings"]
