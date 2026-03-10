from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EvalResult:
    values: np.ndarray
    contributions: dict[str, np.ndarray] = field(default_factory=dict)
    states: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
