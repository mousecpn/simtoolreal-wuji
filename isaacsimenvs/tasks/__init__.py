"""Task registry for isaacsimenvs.

Each task subpackage registers itself with gymnasium on import (side effect
in its ``__init__.py``). Importing ``isaacsimenvs`` (or any child) is enough
to expose all task ids to ``gym.make`` / ``gym.spec``.
"""

from . import simtoolreal  # side effect: gym.register("Isaacsimenvs-SimToolReal-Direct-v0", ...)
from . import tianji_wuji  # independent single-arm Wuji task

from . import tianji_wuji_dual  # independent dual-arm Wuji task

__all__ = ["simtoolreal", "tianji_wuji", "tianji_wuji_dual"]
