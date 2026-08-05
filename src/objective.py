# =========================
# File: objective.py (part 1/3)
# =========================

import numpy as np


def evaluate_candidate(curve, config):
    """
    Evaluate one candidate support.

    Pipeline
    --------
        1. Compute metrics
        2. Apply Filter 1
        3. Apply Filter 2
        4. Compute score

    Returns
    -------
    CandidateResult
    """

    metrics = _compute_metrics(curve)

    ok, reason = _filter_1(metrics, config)
    if not ok:
        return CandidateResult(
            accepted=False,
            score=0.0,
            reject_reason=reason,
        )

    ok, reason = _filter_2(metrics, config)
    if not ok:
        return CandidateResult(
            accepted=False,
            score=0.0,
            reject_reason=reason,
        )

    return CandidateResult(
        accepted=True,
        score=_score(metrics, config),
        reject_reason=None,
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _compute_metrics(curve):

    displacement = curve.displacement
    velocity = curve.velocity
    theta = curve.theta

    stroke = displacement.max() - displacement.min()

    if stroke <= 0.0:
        stroke = 1e-12

    plateau = np.abs(velocity) <= 0.05 * np.max(np.abs(velocity))

    plateau_indices = np.where(plateau)[0]

    if len(plateau_indices):

        start = plateau_indices[0]
        end = plateau_indices[-1]

        plateau_amplitude = (
            displacement[plateau_indices].max()
            - displacement[plateau_indices].min()
        )

        plateau_center = 0.5 * (theta[start] + theta[end])

        plateau_width = theta[end] - theta[start]

    else:

        start = None
        end = None

        plateau_amplitude = stroke
        plateau_center = -999.0
        plateau_width = 0.0

    sign = np.sign(velocity)

    sign_change = np.where(sign[:-1] * sign[1:] < 0)[0] + 1

    return {
        "theta": theta,
        "velocity": velocity,
        "stroke": stroke,
        "plateau_mask": plateau,
        "plateau_start": start,
        "plateau_end": end,
        "plateau_amplitude": plateau_amplitude,
        "plateau_center": plateau_center,
        "plateau_width": plateau_width,
        "sign_changes": sign_change,
    }


# ---------------------------------------------------------------------
# Filter 1
# ---------------------------------------------------------------------

def _filter_1(metrics, config):

    if (
        metrics["plateau_amplitude"]
        > config.plateau_max_amplitude_ratio * metrics["stroke"]
    ):
        return False, "plateau amplitude"

    center = metrics["plateau_center"]

    ok90 = abs(center - config.plateau_center_1_deg) <= (
        config.plateau_center_tolerance_deg
    )

    ok270 = abs(center - config.plateau_center_2_deg) <= (
        config.plateau_center_tolerance_deg
    )

    if not (ok90 or ok270):
        return False, "plateau center"

    width = metrics["plateau_width"]

    if width < config.plateau_min_width_deg:
        return False, "plateau too short"

    if width > config.plateau_max_width_deg:
        return False, "plateau too long"

    start = metrics["plateau_start"]
    end = metrics["plateau_end"]

    if start is None or end is None:
        return False, "missing plateau"

    velocity = metrics["velocity"]

    before = velocity[max(0, start - 1)]
    after = velocity[min(len(velocity) - 1, end + 1)]

    if before * after >= 0.0:
        return False, "velocity sign"

    return True, None

# =========================
# File: objective.py (part 2/3)
# =========================

# ---------------------------------------------------------------------
# Filter 2
# ---------------------------------------------------------------------

def _filter_2(metrics, config):

    sign_changes = metrics["sign_changes"]

    start = metrics["plateau_start"]
    end = metrics["plateau_end"]

    if start is None or end is None:
        return False, "missing plateau"

    outside = []

    for index in sign_changes:

        if index < start or index > end:
            outside.append(index)

    if len(outside) != 1:
        return False, "number of sign changes"

    inversion = outside[0]

    theta = metrics["theta"]

    left_distance = abs(inversion - start)
    right_distance = abs(inversion - end)

    if left_distance < right_distance:
        nearest = theta[start]
    else:
        nearest = theta[end]

    bisector = 0.5 * (theta[inversion] + nearest)
    bisector %= 360.0

    for target in config.bisector_targets_deg:

        delta = abs((bisector - target + 180.0) % 360.0 - 180.0)

        if delta <= config.bisector_tolerance_deg:
            return True, None

    return False, "bisector"


# ---------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------

def _normalized_error(value, tolerance):

    if tolerance <= 0.0:
        return 0.0

    score = 1.0 - value / tolerance

    return float(np.clip(score, 0.0, 1.0))


def _plateau_quality(metrics):

    ratio = (
        metrics["plateau_amplitude"]
        / metrics["stroke"]
    )

    return _normalized_error(ratio, 0.10)


def _rapid_center_quality(metrics):

    center = metrics["plateau_center"]

    e90 = abs(center - 90.0)
    e270 = abs(center - 270.0)

    error = min(e90, e270)

    return _normalized_error(error, 20.0)


def _rapid_plateau_quality(metrics):

    width = metrics["plateau_width"]

    if width < 45.0:
        return width / 45.0

    if width > 110.0:
        return max(0.0, 1.0 - (width - 110.0) / 45.0)

    optimum = 77.5

    error = abs(width - optimum)

    return _normalized_error(error, 32.5)


def _slow_plateau_quality(metrics):

    velocity = metrics["velocity"]

    plateau = metrics["plateau_mask"]

    moving = velocity[~plateau]

    if len(moving) == 0:
        return 0.0

    reference = np.mean(np.abs(moving))

    spread = np.std(np.abs(moving))

    if reference == 0.0:
        return 0.0

    return _normalized_error(
        spread / reference,
        1.0,
    )

# =========================
# File: objective.py (part 3/3)
# =========================

# ---------------------------------------------------------------------
# Global score
# ---------------------------------------------------------------------

def _score(metrics, config):

    fast_score = _rapid_plateau_quality(metrics)

    slow_score = _slow_plateau_quality(metrics)

    center_score = _rapid_center_quality(metrics)

    plateau_score = _plateau_quality(metrics)

    return (
        config.weight_fast_plateau * fast_score
        + config.weight_slow_plateau * slow_score
        + config.weight_fast_center * center_score
        + config.weight_static_plateau * plateau_score
    )


__all__ = [
    "evaluate_candidate",
]