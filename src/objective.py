# =========================
# File: objective.py
# =========================

import numpy as np

from models import CandidateResult


def evaluate_candidate(curve, config):
    """
    Évalue un candidat.

    Les filtres géométriques restent inchangés.

    Le score final est construit en deux blocs :

        Q_shape = qualité de la forme
        Q_accel = qualité des trois transitions

        score_base = 0.5 * Q_shape + 0.5 * Q_accel

    puis :

        score = score_base * symmetry_factor

    Plus le score est élevé, meilleure est la solution.
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

    best_score = -np.inf
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

        score_data = _score_candidate(
            candidate,
            config,
        )

        score = score_data["score"]

        if (
            best_metrics is None
            or score > best_score
        ):
            best_score = score

            best_metrics = candidate
            best_metrics.update(score_data)

    if best_metrics is None:
        return CandidateResult(
            accepted=False,
            score=0.0,
            reject_reason="bisector",
        )

    components = {
        "shape_quality":
            best_metrics["shape_quality"],

        "plateau_error":
            best_metrics["plateau_error"],

        "rapid_error":
            best_metrics["rapid_error"],

        "slow_error":
            best_metrics["slow_error"],

        "acceleration_quality":
            best_metrics["acceleration_quality"],

        "acceleration_a1":
            best_metrics["acceleration_a1"],

        "acceleration_a3":
            best_metrics["acceleration_a3"],

        "acceleration_a2":
            best_metrics["acceleration_a2"],

        "acceleration_q_a1":
            best_metrics["acceleration_q_a1"],

        "acceleration_q_a3":
            best_metrics["acceleration_q_a3"],

        "acceleration_q_a2":
            best_metrics["acceleration_q_a2"],

        "symmetry_factor":
            best_metrics["symmetry_factor"],

        "a1_angle":
            best_metrics["a1_angle"],

        "a3_angle":
            best_metrics["a3_angle"],

        "a2_angle":
            best_metrics["a2_angle"],

        "ai_angle":
            best_metrics["ai_angle"],

        "bisector_angle":
            best_metrics["bisector_angle"],
    }

    return CandidateResult(
        accepted=True,
        score=float(best_score),
        reject_reason=None,
        score_components=components,
    )



# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _compute_metrics(curve, config):
    theta = np.asarray(curve.theta, dtype=float)
    displacement = np.asarray(
        curve.displacement,
        dtype=float,
    )

    if len(theta) < 5:
        return {
            "theta": theta,
            "displacement": displacement,
            "velocity": np.zeros_like(displacement),
            "acceleration": np.zeros_like(displacement),
            "stroke": config.epsilon,
        }

    stroke = float(
        np.max(displacement)
        - np.min(displacement)
    )

    if stroke <= config.epsilon:
        stroke = config.epsilon

    # Recalcul propre des dérivées.
    #
    # On ne dépend pas de curve.velocity pour éviter qu'une
    # approximation ancienne du modèle contamine le nouveau score.

    velocity = np.gradient(
        displacement,
        theta,
        edge_order=2,
    )

    acceleration = np.gradient(
        velocity,
        theta,
        edge_order=2,
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

def _score_candidate(metrics, config):
    """
    Score final.

    1. Qualité de forme :
         1/3 plateau
         1/3 phase rapide
         1/3 phase lente

    2. Accélérations :
         A1, A3 et A2 sont normalisées INDIVIDUELLEMENT
         avant leur moyenne.

    3. Les deux blocs ont exactement le même poids.

    4. Une petite pénalisation de dissymétrie est appliquée
       autour de 0° / 180°.
    """

    theta = metrics["theta"]
    displacement = metrics["displacement"]
    acceleration = metrics["acceleration"]
    stroke = metrics["stroke"]

    a3 = float(metrics["a3_angle"])

    # --------------------------------------------------------------
    # A1 / A2 idéaux déduits de A3.
    # --------------------------------------------------------------

    a1 = (
        180.0 + a3
    ) % 360.0

    a2 = (
        360.0 - a3
    ) % 360.0

    metrics["a1_angle"] = a1
    metrics["a2_angle"] = a2

    # --------------------------------------------------------------
    # Coordonnée angulaire déroulée à partir de A2.
    # --------------------------------------------------------------

    u = (
        theta - a2
    ) % 360.0

    width_1 = (
        (a3 - a2)
        % 360.0
    )

    width_2 = (
        (a1 - a3)
        % 360.0
    )

    if (
        width_1 <= config.epsilon
        or width_2 <= config.epsilon
    ):
        return {
            "score": 0.0,
            "shape_quality": 0.0,
            "plateau_error": 1.0,
            "rapid_error": 1.0,
            "slow_error": 1.0,
            "acceleration_quality": 0.0,
            "acceleration_a1": 0.0,
            "acceleration_a3": 0.0,
            "acceleration_a2": 0.0,
            "acceleration_q_a1": 0.0,
            "acceleration_q_a3": 0.0,
            "acceleration_q_a2": 0.0,
            "symmetry_factor": 0.0,
        }

    # --------------------------------------------------------------
    # Détermination du niveau du plateau.
    # --------------------------------------------------------------

    plateau_start = float(
        metrics["plateau_start_angle"]
    )

    plateau_width = float(
        metrics["plateau_width"]
    )

    plateau_relative = (
        (theta - plateau_start)
        % 360.0
    )

    plateau_mask = (
        plateau_relative <= plateau_width
    )

    if not np.any(plateau_mask):
        return {
            "score": 0.0,
            "shape_quality": 0.0,
            "plateau_error": 1.0,
            "rapid_error": 1.0,
            "slow_error": 1.0,
            "acceleration_quality": 0.0,
            "acceleration_a1": 0.0,
            "acceleration_a3": 0.0,
            "acceleration_a2": 0.0,
            "acceleration_q_a1": 0.0,
            "acceleration_q_a3": 0.0,
            "acceleration_q_a2": 0.0,
            "symmetry_factor": 0.0,
        }

    plateau_values = displacement[plateau_mask]

    plateau_level = float(
        np.mean(plateau_values)
    )

    x_min = float(np.min(displacement))
    x_max = float(np.max(displacement))

    plateau_is_max = (
        abs(plateau_level - x_max)
        <=
        abs(plateau_level - x_min)
    )

    # --------------------------------------------------------------
    # Déplacement normalisé.
    # --------------------------------------------------------------

    if plateau_is_max:
        x_norm = (
            displacement - x_min
        ) / stroke
    else:
        x_norm = (
            x_max - displacement
        ) / stroke

    x_norm = np.clip(
        x_norm,
        -1.0,
        2.0,
    )

    # --------------------------------------------------------------
    # Loi idéale.
    #
    # A2 -> A3 : 1 -> 0
    # A3 -> A1 : 0 -> 1
    # plateau   : 1
    #
    # Le plateau est traité avec sa vraie position mesurée,
    # tandis que les deux phases actives utilisent A3.
    # --------------------------------------------------------------

    ideal = np.empty_like(
        x_norm,
    )

    mask_rapid = (
        u <= width_1
    )

    mask_slow = (
        (u > width_1)
        &
        (
            u <= width_1 + width_2
        )
    )

    mask_plateau = ~(
        mask_rapid | mask_slow
    )

    ideal[mask_rapid] = (
        1.0
        - u[mask_rapid] / width_1
    )

    local_slow = (
        u[mask_slow]
        - width_1
    )

    ideal[mask_slow] = (
        local_slow / width_2
    )

    ideal[mask_plateau] = 1.0

    # --------------------------------------------------------------
    # RMS séparés.
    #
    # Important :
    # aucune phase ne peut être "cachée" par une autre.
    # --------------------------------------------------------------

    plateau_error = _rms(
        x_norm[mask_plateau] - ideal[mask_plateau]
    )

    rapid_error = _rms(
        x_norm[mask_rapid] - ideal[mask_rapid]
    )

    slow_error = _rms(
        x_norm[mask_slow] - ideal[mask_slow]
    )

    # Normalisation conservatrice des erreurs.
    shape_error = (
        plateau_error
        + rapid_error
        + slow_error
    ) / 3.0

    shape_quality = np.clip(
        1.0 - shape_error,
        0.0,
        1.0,
    )

    # --------------------------------------------------------------
    # Accélérations aux trois changements de phase.
    #
    # On prend la valeur maximale locale de |a| dans une petite
    # fenêtre autour de chaque transition. Cela évite que le
    # résultat dépende d'un unique point d'échantillonnage.
    # --------------------------------------------------------------

    accel_a1 = _local_peak_acceleration(
        theta,
        acceleration,
        a1,
    )

    accel_a3 = _local_peak_acceleration(
        theta,
        acceleration,
        a3,
    )

    accel_a2 = _local_peak_acceleration(
        theta,
        acceleration,
        a2,
    )

    # --------------------------------------------------------------
    # Accélération idéale discrétisée.
    #
    # L'idéal mathématique possède des transitions instantanées,
    # donc une accélération infinie.
    #
    # Pour rendre le problème numérique et indépendant de la
    # géométrie, on utilise la même loi idéale échantillonnée sur
    # les mêmes angles que le candidat.
    # --------------------------------------------------------------

    ideal_acceleration = _ideal_acceleration(
        theta,
        a1,
        a3,
        a2,
        plateau_is_max,
    )

    ideal_a1 = _local_peak_acceleration(
        theta,
        ideal_acceleration,
        a1,
    )

    ideal_a3 = _local_peak_acceleration(
        theta,
        ideal_acceleration,
        a3,
    )

    ideal_a2 = _local_peak_acceleration(
        theta,
        ideal_acceleration,
        a2,
    )

    # --------------------------------------------------------------
    # NORMALISATION INDIVIDUELLE.
    #
    # C'est volontairement :
    #
    #     q1 = a1 / ideal_a1
    #     q3 = a3 / ideal_a3
    #     q2 = a2 / ideal_a2
    #
    # puis seulement ensuite :
    #
    #     Q_A = (q1 + q3 + q2) / 3
    #
    # On n'additionne donc jamais les trois accélérations brutes.
    # --------------------------------------------------------------

    q_a1 = _acceleration_quality(
        accel_a1,
        ideal_a1,
    )

    q_a3 = _acceleration_quality(
        accel_a3,
        ideal_a3,
    )

    q_a2 = _acceleration_quality(
        accel_a2,
        ideal_a2,
    )

    acceleration_quality = (
        q_a1
        + q_a3
        + q_a2
    ) / 3.0

    # --------------------------------------------------------------
    # Symétrie.
    #
    # La meilleure géométrie doit présenter une transition rapide
    # centrée autour de 0° ou 180°.
    #
    # On utilise le bisecteur déjà calculé par le filtre 2.
    # --------------------------------------------------------------

    bisector = float(
        metrics["bisector_angle"]
    )

    d0 = _angular_distance(
        bisector,
        0.0,
    )

    d180 = _angular_distance(
        bisector,
        180.0,
    )

    symmetry_error = min(
        d0,
        d180,
    )

    symmetry_factor = np.clip(
        1.0
        - symmetry_error / 15.0,
        0.0,
        1.0,
    )

    # --------------------------------------------------------------
    # Score final : 50 % forme + 50 % accélération.
    # --------------------------------------------------------------

    score_base = (
        0.5 * shape_quality
        +
        0.5 * acceleration_quality
    )

    score = (
        score_base
        * symmetry_factor
    )

    return {
        "score": float(score),

        "shape_quality":
            float(shape_quality),

        "plateau_error":
            float(plateau_error),

        "rapid_error":
            float(rapid_error),

        "slow_error":
            float(slow_error),

        "acceleration_quality":
            float(acceleration_quality),

        "acceleration_a1":
            float(accel_a1),

        "acceleration_a3":
            float(accel_a3),

        "acceleration_a2":
            float(accel_a2),

        "acceleration_q_a1":
            float(q_a1),

        "acceleration_q_a3":
            float(q_a3),

        "acceleration_q_a2":
            float(q_a2),

        "symmetry_factor":
            float(symmetry_factor),
    }

def _rms(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    if values.size == 0:
        return 1.0

    return float(
        np.sqrt(
            np.mean(
                values * values
            )
        )
    )


def _angular_distance(a, b):
    return abs(
        (
            a - b + 180.0
        ) % 360.0
        - 180.0
    )


def _local_peak_acceleration(
    theta,
    acceleration,
    angle,
):
    """
    Maximum |acceleration| dans une fenêtre locale.

    La fenêtre est choisie à partir du pas angulaire réel afin
    que le calcul fonctionne aussi bien en mode 5° qu'en haute
    définition.
    """

    theta = np.asarray(
        theta,
        dtype=float,
    )

    acceleration = np.asarray(
        acceleration,
        dtype=float,
    )

    if len(theta) == 0:
        return 0.0

    if len(theta) > 1:
        step = abs(
            float(theta[1] - theta[0])
        )
    else:
        step = 1.0

    half_window = max(
        step * 1.5,
        1.0,
    )

    distances = np.array([
        _angular_distance(
            float(t),
            angle,
        )
        for t in theta
    ])

    mask = (
        distances <= half_window
    )

    if not np.any(mask):
        index = int(
            np.argmin(distances)
        )

        return float(
            abs(acceleration[index])
        )

    return float(
        np.max(
            np.abs(
                acceleration[mask]
            )
        )
    )


def _ideal_acceleration(
    theta,
    a1,
    a3,
    a2,
    plateau_is_max,
):
    """
    Accélération numérique de la loi idéale.

    On construit d'abord la position idéale sur la grille réelle,
    puis on applique exactement les mêmes dérivées numériques
    que pour le candidat.

    Cela évite toute dépendance à une unité arbitraire.
    """

    theta = np.asarray(
        theta,
        dtype=float,
    )

    u = (
        theta - a2
    ) % 360.0

    width_1 = (
        (a3 - a2)
        % 360.0
    )

    width_2 = (
        (a1 - a3)
        % 360.0
    )

    ideal = np.empty_like(
        theta,
        dtype=float,
    )

    mask_1 = (
        u <= width_1
    )

    mask_2 = (
        (u > width_1)
        &
        (
            u <= width_1 + width_2
        )
    )

    mask_3 = ~(
        mask_1 | mask_2
    )

    ideal[mask_1] = (
        1.0
        - u[mask_1] / width_1
    )

    ideal[mask_2] = (
        u[mask_2] - width_1
    ) / width_2

    ideal[mask_3] = 1.0

    if len(theta) < 5:
        return np.zeros_like(
            ideal
        )

    velocity = np.gradient(
        ideal,
        theta,
        edge_order=2,
    )

    return np.gradient(
        velocity,
        theta,
        edge_order=2,
    )


def _acceleration_quality(
    actual,
    ideal,
):
    """
    Compare une accélération à sa propre référence.

    Chaque transition est traitée séparément.

    Une accélération réelle au moins égale à la référence
    discrète reçoit la note maximale 1.
    """

    actual = abs(
        float(actual)
    )

    ideal = abs(
        float(ideal)
    )

    if ideal <= 1e-15:
        return 1.0

    return float(
        np.clip(
            actual / ideal,
            0.0,
            1.0,
        )
    )

def _curve_deviation(metrics, config):
    """
    Compatibilité avec l'ancienne API.

    Le nouveau moteur de classement n'utilise plus cette fonction.
    """
    data = _score_candidate(
        metrics,
        config,
    )

    return float(
        1.0 - data["shape_quality"]
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
