# =========================
# File: objective.py
# =========================

import numpy as np

from models import CandidateResult


def evaluate_candidate(curve, config):
    """
    Evaluate one candidate using the current kinematic criteria.

    The evaluation is performed exclusively on the X projection
    represented by curve.displacement and curve.velocity.
    """

    metrics = _compute_metrics(curve, config)

    plateau_candidates = _find_plateau_candidates(metrics, config)

    if not plateau_candidates:
        return CandidateResult(
            accepted=False,
            score=0.0,
            reject_reason="plateau",
        )

    best_metrics = None
    best_acceleration = None

    for plateau in plateau_candidates:

        candidate = dict(metrics)
        candidate.update(plateau)

        ok, reason = _filter_2(candidate, config)

        if not ok:
            continue

        mean_abs_acceleration = _mean_abs_acceleration(
            candidate,
            config,
        )

        if (
            best_acceleration is None
            or mean_abs_acceleration < best_acceleration
        ):
            best_acceleration = mean_abs_acceleration
            best_metrics = candidate

    if best_metrics is None:
        return CandidateResult(
            accepted=False,
            score=0.0,
            reject_reason="bisector",
        )

    # The optimizer keeps the largest score.
    # This is only a monotonic transformation of the real criterion:
    # smaller mean absolute acceleration = better solution.
    score = 1.0 / (
        best_acceleration + config.epsilon
    )

    best_metrics["score_components"] = {
        "mean_abs_acceleration": best_acceleration,
        "plateau_start_angle": best_metrics["plateau_start_angle"],
        "plateau_end_angle": best_metrics["plateau_end_angle"],
        "a3_angle": best_metrics["a3_angle"],
        "ai_angle": best_metrics["ai_angle"],
        "bisector_angle": best_metrics["bisector_angle"],
    }

    return CandidateResult(
        accepted=True,
        score=score,
        reject_reason=None,
        score_components=best_metrics["score_components"],
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _compute_metrics(curve, config):
    theta = curve.theta
    displacement = curve.displacement
    velocity = curve.velocity

    stroke = float(
        np.max(displacement) - np.min(displacement)
    )

    if stroke <= 0.0:
        stroke = config.epsilon

    acceleration = np.gradient(
        velocity,
        theta,
    )

    return {
        "theta": theta,
        "displacement": displacement,
        "velocity": velocity,
        "acceleration": acceleration,
        "stroke": stroke,
    }


# ---------------------------------------------------------------------
# Filter 1
# ---------------------------------------------------------------------

def _find_plateau_candidates(metrics, config):
    """
    Find maximal circular portions whose total X amplitude is <= 10%
    of the stroke.

    The search is deliberately independent of the target angle:

        1. find maximal low-amplitude circular intervals;
        2. measure their true A1/A2 boundaries;
        3. keep only intervals with the required width;
        4. keep only intervals centered near 90° or 270°.

    Internal oscillations / velocity sign changes are allowed.
    """

    theta = metrics["theta"]
    displacement = metrics["displacement"]
    stroke = metrics["stroke"]

    n = len(theta)

    if n < 3:
        return []

    step = float(theta[1] - theta[0])

    amplitude_limit = 0.05 * stroke

    # Les plateaux recherchés doivent être situés sur un extremum
    # de la course de X : maximum global ou minimum global.
    x_max = np.max(displacement)
    x_min = np.min(displacement)

    extremum_tolerance = 1e-10

    extremum_indices = np.flatnonzero(
        (np.abs(displacement - x_max) <= extremum_tolerance)
        | (np.abs(displacement - x_min) <= extremum_tolerance)
    )

    extremum_indices = set(
        int(i) for i in extremum_indices
    )

    candidates = []

    # --------------------------------------------------------------
    # Work on two revolutions so that a plateau crossing 0° is
    # represented as one continuous interval.
    # --------------------------------------------------------------

    x = np.concatenate((displacement, displacement))

    # We only need starts in the first revolution.
    for start in range(n):

        current_min = x[start]
        current_max = x[start]

        end = start

        # ----------------------------------------------------------
        # Extend the interval as far as possible while its TOTAL
        # amplitude remains <= 10% of the stroke.
        # ----------------------------------------------------------

        while end + 1 < start + n:

            value = x[end + 1]

            new_min = min(current_min, value)
            new_max = max(current_max, value)

            if new_max - new_min > amplitude_limit:
                break

            current_min = new_min
            current_max = new_max
            end += 1

        width = (end - start) * step

        if (
            width < 60.0
            or width >= config.plateau_max_width_deg
        ):
            continue

        # ----------------------------------------------------------
        # The interval must be maximal on BOTH sides.
        #
        # Otherwise we would again detect an arbitrary sub-section
        # of a larger plateau.
        # ----------------------------------------------------------

        if start > 0:
            previous_value = x[start - 1]

            if (
                max(current_max, previous_value)
                - min(current_min, previous_value)
                <= amplitude_limit
            ):
                continue

        if end + 1 < start + n:
            next_value = x[end + 1]

            if (
                max(current_max, next_value)
                - min(current_min, next_value)
                <= amplitude_limit
            ):
                continue

        # ----------------------------------------------------------
        # Le plateau doit réellement contenir un extremum de la
        # course de X (maximum ou minimum global).
        #
        # Cela interdit de sélectionner une portion stable située
        # sur une pente simplement parce que son amplitude locale
        # est faible.
        # ----------------------------------------------------------

        contains_extremum = any(
            (i % n) in extremum_indices
            for i in range(start, end + 1)
        )

        if not contains_extremum:
            continue

        a1 = float(theta[start])

        a2_unwrapped = (
            a1 + width
        )

        a2 = a2_unwrapped % 360.0

        centre = (
            a1 + width / 2.0
        ) % 360.0

        # ----------------------------------------------------------
        # Only now do we impose the required location of the
        # immobile zone.
        # ----------------------------------------------------------

        distance_90 = _angular_distance(
            centre,
            90.0,
        )

        distance_270 = _angular_distance(
            centre,
            270.0,
        )

        if distance_90 <= 10.0:
            target = 90.0

        elif distance_270 <= 10.0:
            target = 270.0

        else:
            continue

        candidates.append({
            "plateau_start": start,
            "plateau_end": end % n,
            "plateau_start_angle": a1 % 360.0,
            "plateau_end_angle": a2,
            "plateau_center": centre,
            "plateau_width": width,
            "plateau_amplitude": current_max - current_min,
            "plateau_center_target": target,
        })

    # --------------------------------------------------------------
    # Remove duplicate representations of the same circular
    # interval.
    # --------------------------------------------------------------

    unique = {}

    for candidate in candidates:

        key = (
            round(
                candidate["plateau_start_angle"],
                8,
            ),
            round(
                candidate["plateau_end_angle"],
                8,
            ),
        )

        unique[key] = candidate

    candidates = list(unique.values())

    candidates.sort(
        key=lambda p: (
            -p["plateau_width"],
            _angular_distance(
                p["plateau_center"],
                p["plateau_center_target"],
            ),
            p["plateau_amplitude"],
        )
    )

    return candidates


# ---------------------------------------------------------------------
# Filter 2
# ---------------------------------------------------------------------

def _filter_2(metrics, config):
    """
    Require exactly one velocity inversion in the half-cycle opposite
    the selected immobile portion.

    A3 is the sampled angle where |velocity| is minimum in that half.
    Ai is the closest of A1/A2 to A3 using circular angular distance.

    The circular bisector between Ai and A3 must lie within ±10°
    of either 0° or 180°.
    """

    theta = metrics["theta"]
    velocity = metrics["velocity"]

    start_angle = metrics["plateau_start_angle"]
    end_angle = metrics["plateau_end_angle"]
    plateau_center = metrics["plateau_center"]

    # Determine which half-cycle is opposite the immobile zone.
    if abs(
        _angular_distance(
            plateau_center,
            config.plateau_center_1_deg,
        )
    ) < abs(
        _angular_distance(
            plateau_center,
            config.plateau_center_2_deg,
        )
    ):
        # Immobile zone around 90° -> inspect 180°..360°.
        half_mask = (
            (theta >= 180.0)
            & (theta <= 360.0)
        )
    else:
        # Immobile zone around 270° -> inspect 0°..180°.
        half_mask = (
            (theta >= 0.0)
            & (theta <= 180.0)
        )

    indices = np.flatnonzero(half_mask)

    if len(indices) < 3:
        return False, "half-cycle"

    half_velocity = velocity[indices]

    # Remove zero-valued samples before counting sign changes.
    signs = np.sign(half_velocity)
    non_zero = signs != 0.0

    sign_indices = indices[non_zero]
    sign_values = signs[non_zero]

    if len(sign_values) < 2:
        return False, "number of sign changes"

    change_positions = np.flatnonzero(
        sign_values[1:] * sign_values[:-1] < 0.0
    )

    if len(change_positions) != 1:
        return False, "number of sign changes"

    # A3 = angle of minimum absolute velocity in the selected half.
    local_min = int(
        np.argmin(np.abs(half_velocity))
    )

    a3 = float(theta[indices[local_min]])

    a1 = float(start_angle)
    a2 = float(end_angle)

    # Closest endpoint to A3 using circular distance.
    if _angular_distance(a1, a3) <= _angular_distance(a2, a3):
        ai = a1
    else:
        ai = a2

    # Midpoint of the shortest circular arc Ai -> A3.
    delta = (
        (a3 - ai + 180.0) % 360.0
        - 180.0
    )

    bisector = (
        ai + 0.5 * delta
    ) % 360.0

    valid_bisector = False

    for target in config.bisector_targets_deg:

        if (
            _angular_distance(
                bisector,
                target,
            )
            <= config.bisector_tolerance_deg
        ):
            valid_bisector = True
            break

    if not valid_bisector:
        return False, "bisector"

    metrics["a3_angle"] = a3
    metrics["ai_angle"] = ai
    metrics["bisector_angle"] = bisector

    return True, None


# ---------------------------------------------------------------------
# Ranking criterion
# ---------------------------------------------------------------------

def _mean_abs_acceleration(metrics, config):
    """
    Mean absolute X acceleration over the complete plateau A1 -> A2.

    For the two other phases, 5° are removed at each end.
    Acceleration is normalized by the total stroke.
    """

    theta = metrics["theta"]
    acceleration = metrics["acceleration"]

    a1 = metrics["plateau_start_angle"]
    a2 = metrics["plateau_end_angle"]
    a3 = metrics["a3_angle"]

    n = len(theta)

    # --------------------------------------------------------------
    # Plateau A1 -> A2
    #
    # Entire plateau is included. No exclusion around A1 or A2.
    # --------------------------------------------------------------

    plateau_width = (
        (a2 - a1) % 360.0
    )

    plateau_relative = (
        (theta - a1) % 360.0
    )

    in_plateau = (
        plateau_relative <= plateau_width
    )

    # --------------------------------------------------------------
    # Remaining two phases
    #
    # Remove 5° at each end of each phase.
    #
    # The remaining region is therefore everything outside:
    #   - plateau A1 -> A2
    #   - 5° around A1
    #   - 5° around A2
    #   - 5° around the two phase boundaries
    #
    # A3 is retained unless it falls inside one of these excluded
    # 5° zones.
    # --------------------------------------------------------------

    keep = np.ones(n, dtype=bool)

    # Exclude the plateau from this second measurement.
    keep &= ~in_plateau

    # Exclude 5° around A1 and A2.
    for angle in (a1, a2):

        distance = np.abs(
            (theta - angle + 180.0) % 360.0 - 180.0
        )

        keep &= distance > 5.0

    # For the remaining phases, exclude 5° around A3.
    distance_a3 = np.abs(
        (theta - a3 + 180.0) % 360.0 - 180.0
    )

    keep &= distance_a3 > 5.0

    # --------------------------------------------------------------
    # Acceleration used for the score.
    #
    # Plateau and remaining phases are both evaluated, with the
    # exclusions above.
    # --------------------------------------------------------------

    values = np.abs(acceleration)

    selected = (
        in_plateau | keep
    )

    values = values[selected]

    if len(values) == 0:
        return float("inf")

    mean_abs_acceleration = float(
        np.mean(values)
    )

    stroke = metrics["stroke"]

    if stroke <= config.epsilon:
        return float("inf")

    return mean_abs_acceleration / stroke


# ---------------------------------------------------------------------
# Angular helpers
# ---------------------------------------------------------------------

def _angular_distance(a, b):
    return abs(
        (a - b + 180.0) % 360.0
        - 180.0
    )


__all__ = [
    "evaluate_candidate",
]
