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
    Find immobile portions surrounding a genuine extremum of X.

    The angular domain is circular. A plateau may therefore cross
    0°/360°.

    The procedure is deliberately independent of the desired
    plateau location:

      1. find local maxima/minima of X cyclically;
      2. around each extremum, find the largest contiguous zone
         remaining within 10% of the total stroke;
      3. measure its real A1/A2 boundaries;
      4. only then test width and centering around 90° or 270°.
    """

    theta = metrics["theta"]
    displacement = metrics["displacement"]
    velocity = metrics["velocity"]
    stroke = metrics["stroke"]

    n = len(theta)

    if n < 3:
        return []

    step = float(theta[1] - theta[0])

    amplitude_limit = (
        config.plateau_max_amplitude_ratio * stroke
    )

    # --------------------------------------------------------------
    # Cyclic extrema.
    #
    # np.roll() makes the neighbours of 0° be 355° and 5°.
    # --------------------------------------------------------------

    previous = np.roll(displacement, 1)
    following = np.roll(displacement, -1)

    maxima = (
        (displacement >= previous)
        & (displacement >= following)
        & (
            (displacement > previous)
            | (displacement > following)
        )
    )

    minima = (
        (displacement <= previous)
        & (displacement <= following)
        & (
            (displacement < previous)
            | (displacement < following)
        )
    )

    extrema = np.flatnonzero(
        maxima | minima
    )

    candidates = []

    for extremum in extrema:

        extremum_value = displacement[extremum]

        # ----------------------------------------------------------
        # Points within 10% of the extremum.
        #
        # Since the extremum is a max or min, this directly defines
        # a zone whose total X amplitude is <= 10% of the stroke.
        # ----------------------------------------------------------

        stable = (
            np.abs(
                displacement - extremum_value
            )
            <= amplitude_limit
        )

        if not stable[extremum]:
            continue

        # ----------------------------------------------------------
        # Expand cyclically from the extremum.
        # ----------------------------------------------------------

        start = extremum
        end = extremum

        for _ in range(n - 1):
            previous_index = (
                start - 1
            ) % n

            if not stable[previous_index]:
                break

            start -= 1

        for _ in range(n - 1):
            next_index = (
                end + 1
            ) % n

            if not stable[next_index]:
                break

            end += 1

        count = end - start + 1

        if count >= n:
            continue

        width = (
            (count - 1) * step
        )

        if (
            width <= config.plateau_min_width_deg
            or width >= config.plateau_max_width_deg
        ):
            continue

        # ----------------------------------------------------------
        # Convert the unwrapped indices back to angles.
        #
        # This is what makes e.g. 340° -> 120° a single plateau.
        # ----------------------------------------------------------

        a1 = (
            theta[start % n]
            + 360.0 * np.floor(start / n)
        )

        a2 = (
            theta[end % n]
            + 360.0 * np.floor(end / n)
        )

        centre = (
            0.5 * (a1 + a2)
        ) % 360.0

        # ----------------------------------------------------------
        # The plateau must be centred on an extremum of the
        # projection: 90° or 270° ±10°.
        # ----------------------------------------------------------

        distance_90 = _angular_distance(
            centre,
            config.plateau_center_1_deg,
        )

        distance_270 = _angular_distance(
            centre,
            config.plateau_center_2_deg,
        )

        if (
            distance_90
            <= (
                config.plateau_center_max_deg
                - config.plateau_center_1_deg
            )
        ):
            target = 90.0

        elif (
            distance_270
            <= (
                config.plateau_center_max_2_deg
                - config.plateau_center_2_deg
            )
        ):
            target = 270.0

        else:
            continue

        # ----------------------------------------------------------
        # Exact amplitude of the resulting circular zone.
        # This remains the authoritative 10% criterion.
        # ----------------------------------------------------------

        values = np.array([
            displacement[i % n]
            for i in range(start, end + 1)
        ])

        amplitude = (
            np.max(values)
            - np.min(values)
        )

        if amplitude > amplitude_limit:
            continue

        # ----------------------------------------------------------
        # Un plateau doit correspondre à un seul extremum.
        #
        # Un rebond à l'intérieur de la zone crée plusieurs
        # inversions de vitesse et invalide donc le plateau.
        #
        # Le test est cyclique et ignore les éventuels zéros
        # numériques de la vitesse.
        # ----------------------------------------------------------

        candidate_indices = np.array([
            i % n
            for i in range(start, end + 1)
        ])

        candidate_velocity = velocity[candidate_indices]

        signs = np.sign(candidate_velocity)

        non_zero = signs != 0.0
        signs = signs[non_zero]

        if len(signs) < 2:
            continue

        sign_changes = np.count_nonzero(
            signs[1:] != signs[:-1]
        )

        if sign_changes != 1:
            continue

        candidates.append({
            "plateau_start": start % n,
            "plateau_end": end % n,
            "plateau_start_angle": a1 % 360.0,
            "plateau_end_angle": a2 % 360.0,
            "plateau_center": centre,
            "plateau_width": width,
            "plateau_amplitude": amplitude,
            "plateau_center_target": target,
        })

    # --------------------------------------------------------------
    # Several sampled points may belong to the same flat extremum.
    # Keep one representation of each plateau.
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

    candidates = list(
        unique.values()
    )

    candidates.sort(
        key=lambda p: (
            -p["plateau_width"],
            _angular_distance(
                p["plateau_center"],
                p["plateau_center_target"],
            ),
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
    Mean absolute X acceleration, excluding ±10° around A1, A2 and A3.
    """

    theta = metrics["theta"]
    acceleration = metrics["acceleration"]

    a1 = metrics["plateau_start_angle"]
    a2 = metrics["plateau_end_angle"]
    a3 = metrics["a3_angle"]

    exclusion = config.acceleration_exclusion_deg

    keep = np.ones(
        len(theta),
        dtype=bool,
    )

    for angle in (a1, a2, a3):

        keep &= (
            np.abs(
                (
                    theta - angle + 180.0
                ) % 360.0 - 180.0
            )
            > exclusion
        )

    values = np.abs(
        acceleration[keep]
    )

    if len(values) == 0:
        return float("inf")

    return float(
        np.mean(values)
    )


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
