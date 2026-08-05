# =========================
# File: support_search.py (part 1/4)
# =========================

import math
from models import CandidateCurve, Support, SupportKind
from kinematics import point_on_rocker, point_on_coupler


def generate_candidates(mechanism):
    """
    Generate every candidate support for one mechanism.

    Returns
    -------
    list[Support]
    """

    candidates = []

    candidates.extend(_generate_E_candidates(mechanism))
    candidates.extend(_generate_F_candidates(mechanism))

    return candidates


def _generate_E_candidates(mechanism):

    radius = mechanism.rocker / 2.0

    candidates = []

    for angle_deg in range(0, 180, 5):

        angle = math.radians(angle_deg)

        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        candidates.append(
            Support(
                kind=SupportKind.E,
                local_x=x,
                local_y=y,
            )
        )

    return candidates

# =========================
# File: support_search.py (part 2/4)
# =========================

def _generate_F_candidates(mechanism):

    radius = 2.0 * mechanism.coupler
    step = 0.10 * mechanism.coupler

    candidates = []

    x = -radius

    while x <= radius:

        y = -radius

        while y <= radius:

            if x * x + y * y <= radius * radius:

                candidates.append(
                    Support(
                        kind=SupportKind.F,
                        local_x=x,
                        local_y=y,
                    )
                )

            y += step

        x += step

    return candidates

# =========================
# File: support_search.py (part 3/4)
# =========================

def generate_candidate_curves(kinematics, supports):
    """
    Build the motion curve of every candidate support.

    Parameters
    ----------
    kinematics : KinematicResult
    supports : iterable[Support]

    Returns
    -------
    list[CandidateCurve]
    """

    curves = []

    for support in supports:
        curves.append(_build_candidate_curve(kinematics, support))

    return curves


def _build_candidate_curve(kinematics, support):

    if support.kind is SupportKind.E:
        return _build_E_curve(kinematics, support)

    return _build_F_curve(kinematics, support)

# =========================
# File: support_search.py (part 4/4)
# =========================

def _build_E_curve(kinematics, support):
    """
    Point attached to rocker CD.
    """
    raise NotImplementedError(
        "_build_E_curve() will be implemented after kinematics.py"
    )


def _build_F_curve(kinematics, support):
    """
    Point attached to coupler BC.
    """
    raise NotImplementedError(
        "_build_F_curve() will be implemented after kinematics.py"
    )
