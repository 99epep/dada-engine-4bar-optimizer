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

    radius = mechanism.rocker

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

    # The disk is centered on the midpoint of BC.
    # local_x is measured from B along BC, so the center is BC/2.
    center_x = 0.5 * mechanism.coupler

    x = -radius

    while x <= radius:

        y = -radius

        while y <= radius:

            if x * x + y * y <= radius * radius:

                candidates.append(
                    Support(
                        kind=SupportKind.F,
                        local_x=center_x + x,
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
    points = point_on_rocker(kinematics, support)

    x = points[:, 0]
    y = points[:, 1]

    displacement = y - y[0]
    velocity = __import__("numpy").gradient(
        displacement,
        kinematics.theta,
    )

    return CandidateCurve(
        theta=kinematics.theta,
        x=x,
        y=y,
        displacement=displacement,
        velocity=velocity,
    )


def _build_F_curve(kinematics, support):
    """
    Point attached to coupler BC.
    """
    points = point_on_coupler(kinematics, support)

    x = points[:, 0]
    y = points[:, 1]

    displacement = y - y[0]
    velocity = __import__("numpy").gradient(
        displacement,
        kinematics.theta,
    )

    return CandidateCurve(
        theta=kinematics.theta,
        x=x,
        y=y,
        displacement=displacement,
        velocity=velocity,
    )
