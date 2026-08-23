# =========================
# File: support_search.py (part 1/4)
# =========================

import numpy as np
from models import Support, SupportKind
from kinematics import build_candidate_curve


def generate_candidates(mechanism, config, family=None):
    """
    Generate every candidate support for one mechanism.

    Returns
    -------
    list[Support]
    """

    candidates = []

    family = None if family is None else family.upper()
    if family in (None, "E"):
        candidates.extend(_generate_E_candidates(mechanism, config))
    if family in (None, "F"):
        candidates.extend(_generate_F_candidates(mechanism, config))

    return candidates


def _generate_E_candidates(mechanism, config):
    # The E score depends only on the rocker angle. Its physical CDE angle is
    # calculated after refinement, so one canonical support is sufficient.
    return [Support(SupportKind.E, mechanism.rocker, 0.0)]

# =========================
# File: support_search.py (part 2/4)
# =========================

def _generate_F_candidates(mechanism, config):

    radius = config.support_F_radius_factor * mechanism.coupler
    step = config.support_F_grid_step_factor * mechanism.coupler

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
    return build_candidate_curve(kinematics, support)

# =========================
# File: support_search.py (part 4/4)
# =========================

def _build_E_curve(kinematics, support):
    """
    Point attached to rocker CD.
    """
    return build_candidate_curve(kinematics, support)


def _build_F_curve(kinematics, support):
    """
    Point attached to coupler BC.
    """
    return build_candidate_curve(kinematics, support)
