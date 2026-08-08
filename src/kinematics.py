# =========================
# File: kinematics.py (part 1/8)
# =========================

import math
import numpy as np

from models import CandidateCurve, KinematicResult, SupportKind

def solve(mechanism, config):
    """
    Solve one complete revolution of the crank.

    Returns
    -------
    KinematicResult
    """

    theta = np.arange(
        config.angle_start_deg,
        config.angle_end_deg + config.angle_step_deg,
        config.angle_step_deg,
    )

    n = len(theta)

    A = np.zeros((n, 2))
    B = np.zeros((n, 2))
    C = np.zeros((n, 2))
    D = np.zeros((n, 2))

    A[:, :] = (0.0, 0.0)
    D[:, :] = (mechanism.ground, 0.0)

    for i, angle_deg in enumerate(theta):

        ok, b, c = _solve_position(
            mechanism,
            math.radians(angle_deg),
        )

        if not ok:
            return KinematicResult(
                theta=theta,
                A=A,
                B=B,
                C=C,
                D=D,
                valid=False,
            )

        B[i] = b
        C[i] = c

    return KinematicResult(
        theta=theta,
        A=A,
        B=B,
        C=C,
        D=D,
        valid=True,
    )

# =========================
# File: kinematics.py (part 2/8)
# =========================

def _solve_position(mechanism, theta):
    """
    Solve one crank position by circle-circle intersection.

    Returns
    -------
    (success, B, C)
    """

    Ax, Ay = 0.0, 0.0
    Dx, Dy = mechanism.ground, 0.0

    Bx = mechanism.crank * math.cos(theta)
    By = mechanism.crank * math.sin(theta)

    B = np.array([Bx, By])

    C = _circle_intersection(
        B,
        mechanism.coupler,
        np.array([Dx, Dy]),
        mechanism.rocker,
    )

    if C is None:
        return False, None, None

    return True, B, C

# =========================
# File: kinematics.py (part 3/8)
# =========================

def _circle_intersection(center1, radius1, center2, radius2):
    """
    Upper circle-circle intersection.

    Returns
    -------
    np.ndarray | None
    """

    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]

    d = math.hypot(dx, dy)

    if d > radius1 + radius2:
        return None

    if d < abs(radius1 - radius2):
        return None

    if d == 0.0:
        return None

    a = (radius1**2 - radius2**2 + d**2) / (2.0 * d)

    h2 = radius1**2 - a**2

    if h2 < 0.0:
        return None

    h = math.sqrt(max(0.0, h2))

    xm = center1[0] + a * dx / d
    ym = center1[1] + a * dy / d

    rx = -dy * h / d
    ry = dx * h / d

    c1 = np.array([xm + rx, ym + ry])
    c2 = np.array([xm - rx, ym - ry])

    return c1 if c1[1] >= c2[1] else c2

# =========================
# File: kinematics.py (part 4/8)
# =========================

def crank_angles(result):
    """
    Convenience accessor.
    """
    return result.theta


def rocker_vectors(result):
    """
    Returns DC vectors.
    """
    return result.C - result.D


def coupler_vectors(result):
    """
    Returns BC vectors.
    """
    return result.C - result.B


def rocker_length(result):
    """
    Mean rocker length over the cycle.
    """
    return np.linalg.norm(rocker_vectors(result), axis=1).mean()


def coupler_length(result):
    """
    Mean coupler length over the cycle.
    """
    return np.linalg.norm(coupler_vectors(result), axis=1).mean()

# =========================
# File: kinematics.py (part 5/8)
# =========================

def unit_vectors(vectors):
    """
    Normalize an array of 2D vectors.
    """

    norms = np.linalg.norm(vectors, axis=1)

    norms = np.where(norms == 0.0, 1.0, norms)

    return vectors / norms[:, None]


def normal_vectors(vectors):
    """
    Left-handed unit normal vectors.
    """

    u = unit_vectors(vectors)

    return np.column_stack((-u[:, 1], u[:, 0]))


def rocker_frame(result):
    """
    Local frame attached to rocker CD.

    ex : along D -> C
    ey : left normal
    """

    ex = unit_vectors(result.C - result.D)
    ey = normal_vectors(result.C - result.D)

    return ex, ey


def coupler_frame(result):
    """
    Local frame attached to coupler BC.

    ex : along B -> C
    ey : left normal
    """

    ex = unit_vectors(result.C - result.B)
    ey = normal_vectors(result.C - result.B)

    return ex, ey

# =========================
# File: kinematics.py (part 6/8)
# =========================

def point_on_rocker(result, support):
    """
    World coordinates of a support attached to rocker CD.
    """

    ex, ey = rocker_frame(result)

    return (
        result.D
        + support.local_x * ex
        + support.local_y * ey
    )


def point_on_coupler(result, support):
    """
    World coordinates of a support attached to coupler BC.
    """

    ex, ey = coupler_frame(result)

    return (
        result.B
        + support.local_x * ex
        + support.local_y * ey
    )

# =========================
# File: kinematics.py (part 7/8)
# =========================

def build_candidate_curve(result, support):
    """
    Build the complete motion of one support over one crank revolution.
    """

    if support.kind is SupportKind.E:
        points = point_on_rocker(result, support)
    else:
        points = point_on_coupler(result, support)

    x = points[:, 0]
    y = points[:, 1]

    # The pistons are parallel to the fixed AD axis.
    # Their displacement is therefore the perpendicular coordinate,
    # not the Euclidean distance travelled by the support point.
    displacement = y - y[0]

    velocity = np.gradient(
        displacement,
        result.theta,
    )

    return CandidateCurve(
        theta=result.theta,
        x=x,
        y=y,
        displacement=displacement,
        velocity=velocity,
    )

# =========================
# File: kinematics.py (part 8/8)
# =========================

__all__ = [
    "solve",
    "build_candidate_curve",
    "point_on_rocker",
    "point_on_coupler",
    "rocker_frame",
    "coupler_frame",
    "rocker_vectors",
    "coupler_vectors",
    "rocker_length",
    "coupler_length",
    "unit_vectors",
    "normal_vectors",
]
