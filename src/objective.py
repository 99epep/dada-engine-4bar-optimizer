# =========================
# File: objective.py (part 1/3)
# =========================

import numpy as np

from models import CandidateResult

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

    score = _score(metrics, config)

    return CandidateResult(
        accepted=True,
        score=score,
        reject_reason=None,
        score_components=metrics.get("score_components", {}),
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

    vmax = np.max(np.abs(velocity))

    if vmax <= 0.0:
        plateau = np.zeros_like(velocity, dtype=bool)
    else:
        plateau = np.abs(velocity) <= 0.05 * vmax

    # Find contiguous low-speed regions.
    regions = []

    indices = np.flatnonzero(plateau)

    if len(indices):
        start = indices[0]
        previous = indices[0]

        for index in indices[1:]:
            if index != previous + 1:
                regions.append((start, previous))
                start = index
            previous = index

        regions.append((start, previous))

    # Keep only plateaus that are long enough to be relevant.
    candidates = []

    for start, end in regions:
        width = theta[end] - theta[start]

        if width < 45.0:
            continue

        center = 0.5 * (theta[start] + theta[end])

        # Normalize angular distance around 360°.
        d90 = abs((center - 90.0 + 180.0) % 360.0 - 180.0)
        d270 = abs((center - 270.0 + 180.0) % 360.0 - 180.0)

        if min(d90, d270) <= 20.0:
            candidates.append((start, end))

    if candidates:
        # Prefer the plateau closest to 90°/270°.
        start, end = min(
            candidates,
            key=lambda pair: min(
                abs((0.5 * (theta[pair[0]] + theta[pair[1]]) - 90.0 + 180.0) % 360.0 - 180.0),
                abs((0.5 * (theta[pair[0]] + theta[pair[1]]) - 270.0 + 180.0) % 360.0 - 180.0),
            ),
        )

        plateau_amplitude = (
            displacement[start:end + 1].max()
            - displacement[start:end + 1].min()
        )

        plateau_center = 0.5 * (theta[start] + theta[end])
        plateau_width = theta[end] - theta[start]

        # Make the plateau mask represent the selected plateau only.
        selected_plateau = np.zeros_like(plateau, dtype=bool)
        selected_plateau[start:end + 1] = True
        plateau = selected_plateau

    else:
        start = None
        end = None
        plateau_amplitude = stroke
        plateau_center = -999.0
        plateau_width = 0.0

    sign = np.sign(velocity)

    non_zero = sign != 0.0
    non_zero_indices = np.flatnonzero(non_zero)

    if len(non_zero_indices) >= 2:
        previous = sign[non_zero_indices[:-1]]
        current = sign[non_zero_indices[1:]]

        sign_change = non_zero_indices[1:][previous * current < 0.0]
    else:
        sign_change = np.array([], dtype=int)

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

def _angular_distance(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _normalized_error(value, tolerance):
    if tolerance <= 0.0:
        return 0.0
    return float(np.clip(1.0 - value / tolerance, 0.0, 1.0))


def _plateau_quality(metrics):
    ratio = metrics["plateau_amplitude"] / metrics["stroke"]
    return _normalized_error(ratio, 0.10)


def _rapid_center_quality(metrics):
    """
    The rapid phase is the quarter-turn immediately following
    the immobile quarter-turn.

    For a plateau centered at 90°, the rapid phase is centered at 180°.
    For a plateau centered at 270°, the rapid phase is centered at 0°.
    """
    center = metrics["plateau_center"]

    if center < 0.0:
        return 0.0

    if _angular_distance(center, 90.0) <= 20.0:
        target = 180.0
    elif _angular_distance(center, 270.0) <= 20.0:
        target = 0.0
    else:
        return 0.0

    return _normalized_error(
        _angular_distance(center + 90.0, target),
        20.0,
    )


def _rapid_plateau_quality(metrics):
    """
    Quality of the rapid transition.

    We want a sharp transition into the rapid phase and therefore
    favour a rapid-phase width close to one quarter-turn.
    """
    theta = metrics["theta"]
    velocity = metrics["velocity"]
    start = metrics["plateau_start"]
    end = metrics["plateau_end"]

    if start is None or end is None:
        return 0.0

    # Determine the direction immediately after the immobile plateau.
    after_index = min(end + 1, len(velocity) - 1)
    direction = np.sign(velocity[after_index])

    if direction == 0.0:
        return 0.0

    # Find the end of the same-sign rapid phase.
    i = after_index

    while i + 1 < len(velocity):
        if np.sign(velocity[i + 1]) != direction:
            break
        i += 1

    width = theta[i] - theta[after_index]

    # Ideal rapid phase = 90°.
    return _normalized_error(abs(width - 90.0), 45.0)


def _slow_plateau_quality(metrics):
    """
    Compare the rapid quarter-turn with the following slow half-turn.

    The analysis is performed in an angular coordinate system centred
    on the detected immobile plateau, so 90° and 270° cases are treated
    identically.

    Ideal velocity magnitudes:
        rapid = 1
        slow  = 0.5
    """
    theta = metrics["theta"]
    velocity = metrics["velocity"]
    start = metrics["plateau_start"]
    end = metrics["plateau_end"]

    if start is None or end is None:
        return 0.0

    center = metrics["plateau_center"]

    # Signed angular distance from the plateau centre.
    relative = (theta - center + 180.0) % 360.0 - 180.0

    # The rapid phase occupies the first quarter-turn after
    # the immobile quarter-turn.
    rapid_mask = (
        (relative >= 45.0)
        & (relative < 135.0)
    )

    # The slow phase occupies the following half-turn.
    slow_mask = (
        (relative >= 135.0)
        & (relative < 315.0)
    )

    rapid = np.abs(velocity[rapid_mask])
    slow = np.abs(velocity[slow_mask])

    if len(rapid) == 0 or len(slow) == 0:
        return 0.0

    rapid_mean = np.mean(rapid)
    slow_mean = np.mean(slow)

    if rapid_mean <= 0.0:
        return 0.0

    ratio = slow_mean / rapid_mean

    return _normalized_error(abs(ratio - 0.5), 0.5)


def _score(metrics, config):
    fast_score = _rapid_plateau_quality(metrics)
    slow_score = _slow_plateau_quality(metrics)
    center_score = _rapid_center_quality(metrics)
    static_score = _plateau_quality(metrics)

    score = (
        config.weight_fast_plateau * fast_score
        + config.weight_slow_plateau * slow_score
        + config.weight_fast_center * center_score
        + config.weight_static_plateau * static_score
    )

    metrics["score_components"] = {
        "fast": fast_score,
        "slow": slow_score,
        "center": center_score,
        "static": static_score,
    }

    return score


__all__ = [
    "evaluate_candidate",
]