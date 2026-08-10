# =========================
# File: objective.py
# =========================

import numpy as np

from models import CandidateResult


def evaluate_candidate(curve, config):
    """
    Evaluate one candidate.

    The filters determine the valid geometry.
    The score measures the RMS deviation of the real displacement
    curve from the ideal piecewise-linear displacement law defined
    by A3 and the plateau geometry.
    """

    metrics = _compute_metrics(curve, config)

    plateau_candidates = _find_plateau_candidates(
        metrics,
        config,
    )

    if not plateau_candidates:
        return CandidateResult(
            accepted=False,
            score=0.0,
            reject_reason="plateau",
        )

    best_score = None
    best_metrics = None

    for plateau in plateau_candidates:

        candidate = dict(metrics)
        candidate.update(plateau)

        ok, reason = _filter_2(
            candidate,
            config,
        )

        if not ok:
            continue

        deviation = _curve_deviation(
            candidate,
            config,
        )

        if (
            best_score is None
            or deviation < best_score
        ):
            best_score = deviation
            best_metrics = candidate

    if best_metrics is None:
        return CandidateResult(
            accepted=False,
            score=0.0,
            reject_reason="bisector",
        )

    # Smaller deviation = better solution.
    score = 1.0 / (
        best_score + config.epsilon
    )

    best_metrics["score_components"] = {
        "curve_deviation": best_score,
        "plateau_start_angle":
            best_metrics["plateau_start_angle"],
        "plateau_end_angle":
            best_metrics["plateau_end_angle"],
        "a3_angle":
            best_metrics["a3_angle"],
        "ai_angle":
            best_metrics["ai_angle"],
        "bisector_angle":
            best_metrics["bisector_angle"],
    }

    return CandidateResult(
        accepted=True,
        score=score,
        reject_reason=None,
        score_components=best_metrics[
            "score_components"
        ],
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _compute_metrics(curve, config):
    theta = curve.theta
    displacement = curve.displacement
    velocity = curve.velocity

    stroke = float(
        np.max(displacement)
        - np.min(displacement)
    )

    if stroke <= 0.0:
        stroke = config.epsilon

    return {
        "theta": theta,
        "displacement": displacement,
        "velocity": velocity,
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

    amplitude_limit = (
        config.plateau_max_amplitude_ratio * stroke
    )

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
            width < config.plateau_min_width_deg
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

def _curve_deviation(metrics, config):
    """
    RMS deviation between the real displacement curve and the
    ideal piecewise-linear displacement law.

    The ideal law is defined entirely by A3:

        A1 = 180° + A3
        A2 = 360° - A3

    The plateau A1 -> A2 is constant.

    The two remaining phases each span the complete stroke:
        A2 -> A3
        A3 -> A1 + 360°

    The transitions are deliberately included in the error.
    """

    theta = metrics["theta"]
    displacement = metrics["displacement"]
    stroke = metrics["stroke"]

    a3 = float(metrics["a3_angle"])

    # --------------------------------------------------------------
    # Ideal plateau boundaries.
    # --------------------------------------------------------------

    a1 = (
        180.0 + a3
    ) % 360.0

    a2 = (
        360.0 - a3
    ) % 360.0

    # --------------------------------------------------------------
    # Determine whether the real plateau is at the minimum or
    # maximum of the displacement.
    # --------------------------------------------------------------

    plateau_start = float(
        metrics["plateau_start_angle"]
    )
    plateau_end = float(
        metrics["plateau_end_angle"]
    )

    plateau_width = (
        (plateau_end - plateau_start)
        % 360.0
    )

    relative = (
        (theta - plateau_start)
        % 360.0
    )

    plateau_mask = (
        relative <= plateau_width
    )

    plateau_values = displacement[
        plateau_mask
    ]

    if len(plateau_values) == 0:
        return float("inf")

    plateau_level = float(
        np.mean(plateau_values)
    )

    x_min = float(
        np.min(displacement)
    )
    x_max = float(
        np.max(displacement)
    )

    # Plateau at maximum -> ideal value 1.
    # Plateau at minimum -> ideal value 0.
    if abs(plateau_level - x_max) <= abs(
        plateau_level - x_min
    ):
        plateau_is_max = True
    else:
        plateau_is_max = False

    # --------------------------------------------------------------
    # Build an unwrapped angular coordinate starting at A2.
    #
    # This makes the three ideal phases continuous:
    #
    #   A2 -> A3 -> A1 + 360°
    #
    # while the plateau occupies the remaining interval.
    # --------------------------------------------------------------

    u = (
        theta - a2
    ) % 360.0

    # Angular lengths of the two active phases.
    phase_1_width = (
        (a3 - a2)
        % 360.0
    )

    phase_2_width = (
        (a1 - a3)
        % 360.0
    )

    if (
        phase_1_width <= config.epsilon
        or phase_2_width <= config.epsilon
    ):
        return float("inf")

    # --------------------------------------------------------------
    # Ideal normalized displacement.
    # --------------------------------------------------------------

    ideal = np.empty_like(
        displacement,
        dtype=float,
    )

    if plateau_is_max:

        # A2 -> A3 : max -> min
        mask_1 = (
            u <= phase_1_width
        )

        ideal[mask_1] = (
            1.0
            - u[mask_1] / phase_1_width
        )

        # A3 -> A1 : min -> max
        mask_2 = (
            (u > phase_1_width)
            & (
                u
                <= phase_1_width
                + phase_2_width
            )
        )

        local = (
            u[mask_2]
            - phase_1_width
        )

        ideal[mask_2] = (
            local / phase_2_width
        )

        # Remaining interval = plateau.
        ideal[
            ~(
                mask_1
                | mask_2
            )
        ] = 1.0

    else:

        # A2 -> A3 : min -> max
        mask_1 = (
            u <= phase_1_width
        )

        ideal[mask_1] = (
            u[mask_1] / phase_1_width
        )

        # A3 -> A1 : max -> min
        mask_2 = (
            (u > phase_1_width)
            & (
                u
                <= phase_1_width
                + phase_2_width
            )
        )

        local = (
            u[mask_2]
            - phase_1_width
        )

        ideal[mask_2] = (
            1.0
            - local / phase_2_width
        )

        # Remaining interval = plateau.
        ideal[
            ~(
                mask_1
                | mask_2
            )
        ] = 0.0

    # --------------------------------------------------------------
    # Normalize the real displacement by its complete stroke.
    # --------------------------------------------------------------

    real = (
        displacement - x_min
    ) / stroke

    error = (
        real - ideal
    )

    rms = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    return rms


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
