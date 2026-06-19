"""
humanization_utils.py — Re-export shim.

Moved to core.humanization_utils. This file kept for backward compatibility.
"""

import warnings
from core.humanization_utils import (  # noqa: F401
    human_delay,
    jitter_coords,
    jitter_value,
    should_missclick,
    bezier_curve_points,
    HumanPauseSimulator,
    APMController,
)

warnings.warn(
    "Import from core.humanization_utils instead of pylaai_real.humanization_utils",
    DeprecationWarning,
    stacklevel=2,
)
