# =========================
# File: models.py (part 1/3)
# =========================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SupportKind(str, Enum):
    E = "E"
    F = "F"


@dataclass(slots=True)
class Mechanism:
    """
    Four-bar geometry.

    A ----- D : ground link
    A ----- B : crank
    B ----- C : coupler
    C ----- D : rocker
    """

    ground: float
    crank: float
    coupler: float
    rocker: float


@dataclass(slots=True)
class KinematicResult:
    """
    Complete kinematic solution over one revolution.
    """

    theta: np.ndarray

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

    valid: bool

@dataclass(slots=True)
class Support:
    """
    Local definition of a candidate point.

    E : local frame attached to rocker CD.
    F : local frame attached to coupler BC.

    local_x : coordinate along the link.
    local_y : signed normal distance.
    """

    kind: SupportKind

    local_x: float
    local_y: float


@dataclass(slots=True)
class CandidateCurve:
    """
    Motion of one candidate support over one crank revolution.
    """

    theta: np.ndarray

    x: np.ndarray
    y: np.ndarray

    displacement: np.ndarray
    velocity: np.ndarray


@dataclass(slots=True)
class CandidateResult:
    """
    Result returned by objective.py
    """

    accepted: bool

    score: float

    reject_reason: Optional[str]

    score_components: dict[str, float] = field(default_factory=dict)

# =========================
# File: models.py (part 3/3)
# =========================

@dataclass(slots=True)
class Solution:
    """
    One retained solution.
    """

    score: float

    mechanism: Mechanism

    support: Support

    curve: CandidateCurve

    score_components: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class SearchStatistics:
    """
    Optional execution statistics.
    """

    mechanisms_tested: int = 0

    supports_generated: int = 0

    filter1_rejected: int = 0

    filter2_rejected: int = 0

    accepted: int = 0


__all__ = [
    "SupportKind",
    "Mechanism",
    "KinematicResult",
    "Support",
    "CandidateCurve",
    "CandidateResult",
    "Solution",
    "SearchStatistics",
]
