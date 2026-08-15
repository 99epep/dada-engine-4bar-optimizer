"""Simple level-2 geometry refinement using the unchanged level-1 evaluator."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from kinematics import build_candidate_curve, solve
from models import CandidateCurve, Mechanism, Support, SupportKind
from objective import evaluate_candidate


@dataclass(slots=True)
class RefinedSolution:
    family: str
    source_index: int
    initial_score: float
    final_score: float
    initial_geometry: dict
    mechanism: Mechanism
    support: Support
    cde_angle_deg: float | None
    curve: CandidateCurve
    score_components: dict
    physical_metrics: dict
    scoring_angle_step_deg: float
    final_angle_step_deg: float

    @property
    def absolute_improvement(self):
        return self.final_score - self.initial_score

    @property
    def relative_improvement(self):
        return self.absolute_improvement / max(
            abs(self.initial_score), np.finfo(float).eps
        )

    def to_metadata(self, final_index):
        return {
            "family": self.family,
            "final_index": final_index,
            "final_rank": final_index + 1,
            "source_index": self.source_index,
            "source_rank": self.source_index + 1,
            "initial_score": self.initial_score,
            "final_score": self.final_score,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement": self.relative_improvement,
            "initial_geometry": self.initial_geometry,
            "final_geometry": {
                "ground": self.mechanism.ground,
                "crank": self.mechanism.crank,
                "coupler": self.mechanism.coupler,
                "rocker": self.mechanism.rocker,
            },
            # Alias retained for the visualizer and straightforward consumers.
            "mechanism": {
                "ground": self.mechanism.ground,
                "crank": self.mechanism.crank,
                "coupler": self.mechanism.coupler,
                "rocker": self.mechanism.rocker,
            },
            "support": {
                "kind": self.family,
                "local_x": self.support.local_x,
                "local_y": self.support.local_y,
                "cde_angle_deg": self.cde_angle_deg,
            },
            # Exact dictionary returned by level 1; nothing is recomputed here.
            "score_components": self.score_components,
            "score_definition": {
                "shape_weight": 0.35,
                "acceleration_weight": 0.65,
                "slow_adjacent_transition_weight": 2.0,
                "other_transition_weights": 1.0,
            },
            "physical_metrics": self.physical_metrics,
            "scoring_angle_step_deg": self.scoring_angle_step_deg,
            "final_angle_step_deg": self.final_angle_step_deg,
        }


def grid_step(values):
    values = np.unique(np.asarray(values, dtype=float))
    differences = np.diff(values)
    differences = differences[differences > 0.0]
    if differences.size == 0:
        raise ValueError("A level-1 grid must contain at least two values")
    return float(np.min(differences))


def support_from_parameters(family, mechanism, parameters):
    if family == "E":
        angle = math.radians(parameters["cde_angle_deg"])
        return Support(
            SupportKind.E,
            mechanism.rocker * math.cos(angle),
            mechanism.rocker * math.sin(angle),
        )
    return Support(SupportKind.F, parameters["local_x"], parameters["local_y"])


def evaluate_geometry(family, ground, parameters, config):
    """Run precisely the existing level-1 kinematics/filter/score chain."""
    if min(parameters[name] for name in ("crank", "coupler", "rocker")) <= 0.0:
        return None
    mechanism = Mechanism(
        ground, parameters["crank"], parameters["coupler"], parameters["rocker"]
    )
    kinematic = solve(mechanism, config)
    if not kinematic.valid:
        return None
    support = support_from_parameters(family, mechanism, parameters)
    curve = build_candidate_curve(kinematic, support)
    result = evaluate_candidate(curve, config)
    if not result.accepted or not np.isfinite(result.score):
        return None
    return mechanism, support, curve, result


def source_parameters(entry, family):
    mechanism = entry["mechanism"]
    support = entry["support"]
    parameters = {
        "crank": float(mechanism["crank"]),
        "coupler": float(mechanism["coupler"]),
        "rocker": float(mechanism["rocker"]),
    }
    if family == "E":
        parameters["cde_angle_deg"] = math.degrees(
            math.atan2(support["local_y"], support["local_x"])
        ) % 360.0
    else:
        parameters["local_x"] = float(support["local_x"])
        parameters["local_y"] = float(support["local_y"])
    return parameters


def initial_steps(config, family, initial_coupler):
    steps = {
        "crank": grid_step(config.crank_lengths),
        "coupler": grid_step(config.coupler_lengths),
        "rocker": grid_step(config.rocker_lengths),
    }
    if family == "E":
        steps["cde_angle_deg"] = float(config.support_E_step_deg)
    else:
        f_step = float(config.support_F_grid_step_factor * initial_coupler)
        steps.update(local_x=f_step, local_y=f_step)
    return steps


def refine_candidate(entry, family, source_index, config, rng):
    family = family.upper()
    parameters = source_parameters(entry, family)
    ground = float(entry["mechanism"]["ground"])
    current = evaluate_geometry(family, ground, parameters, config)
    if current is None:
        raise ValueError(f"Source {family} #{source_index} is rejected on reevaluation")
    recorded_score = float(entry["score"])
    if not math.isclose(current[3].score, recorded_score, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(
            f"Source score mismatch for {family} #{source_index}: "
            f"stored={recorded_score}, reevaluated={current[3].score}"
        )
    current_score = float(current[3].score)
    steps = initial_steps(config, family, parameters["coupler"])
    tolerances = {
        name: (0.1 if name == "cde_angle_deg" else 0.02) for name in steps
    }

    while True:
        round_improved = True
        while round_improved:
            round_improved = False
            names = list(parameters)
            rng.shuffle(names)
            for name in names:
                best = None
                best_parameters = None
                for direction in (1.0, -1.0):
                    trial_parameters = parameters.copy()
                    trial_parameters[name] += direction * steps[name]
                    if name == "cde_angle_deg":
                        trial_parameters[name] %= 360.0
                    trial = evaluate_geometry(family, ground, trial_parameters, config)
                    if trial is None or trial[3].score <= current_score:
                        continue
                    if best is None or trial[3].score > best[3].score:
                        best, best_parameters = trial, trial_parameters
                if best is not None:
                    current = best
                    parameters = best_parameters
                    current_score = float(best[3].score)
                    round_improved = True

        if all(steps[name] <= tolerances[name] for name in steps):
            break
        steps = {name: step / 5.0 for name, step in steps.items()}

    mechanism, support, _, result = current
    fine_config = replace(config, angle_step_deg=config.refinement_angle_step_deg)
    fine_kinematic = solve(mechanism, fine_config)
    if not fine_kinematic.valid:
        raise AssertionError("Final accepted geometry is kinematically invalid")
    fine_curve = build_candidate_curve(fine_kinematic, support)
    physical = compute_physical_metrics(fine_curve, result.score_components, config)

    if current_score + config.epsilon < recorded_score:
        raise AssertionError("Refinement score regressed")
    if family == "E" and not math.isclose(
        math.hypot(support.local_x, support.local_y), mechanism.rocker, abs_tol=1e-9
    ):
        raise AssertionError("E violates DE == rocker")

    return RefinedSolution(
        family=family,
        source_index=source_index,
        initial_score=recorded_score,
        final_score=current_score,
        initial_geometry={
            "mechanism": dict(entry["mechanism"]),
            "support": dict(entry["support"]),
        },
        mechanism=mechanism,
        support=support,
        cde_angle_deg=parameters.get("cde_angle_deg"),
        curve=fine_curve,
        score_components=dict(result.score_components),
        physical_metrics=physical,
        scoring_angle_step_deg=config.angle_step_deg,
        final_angle_step_deg=config.refinement_angle_step_deg,
    )


def circular_interpolate(theta, values, angles):
    theta = np.mod(np.asarray(theta, dtype=float), 360.0)
    values = np.asarray(values, dtype=float)
    order = np.argsort(theta)
    theta, values = theta[order], values[order]
    theta, indices = np.unique(theta, return_index=True)
    values = values[indices]
    theta_ext = np.r_[theta[-1] - 360.0, theta, theta[0] + 360.0]
    values_ext = np.r_[values[-1], values, values[0]]
    interpolated = np.interp(np.mod(angles, 360.0), theta_ext, values_ext)
    return float(interpolated) if np.ndim(angles) == 0 else interpolated


def detect_real_plateau(curve, score_components, config):
    theta = np.asarray(curve.theta[:-1], dtype=float)
    x = np.asarray(curve.x[:-1], dtype=float)
    step = float(theta[1] - theta[0])
    a1 = float(score_components["a1_angle"])
    a2 = float(score_components["a2_angle"])
    forward_width = (a2 - a1) % 360.0
    if forward_width <= 180.0:
        expected_start, expected_width = a1, forward_width
    else:
        expected_start, expected_width = a2, 360.0 - forward_width
    expected = (
        (theta - expected_start) % 360.0
    ) <= expected_width + config.epsilon
    indices = np.flatnonzero(expected)
    if indices.size == 0:
        raise ValueError("No samples in the expected plateau neighbourhood")

    expected_level = float(np.mean(x[indices]))
    empty_is_maximum = abs(expected_level - np.max(x)) <= abs(expected_level - np.min(x))
    extremum_index = int(indices[
        np.argmax(x[indices]) if empty_is_maximum else np.argmin(x[indices])
    ])
    extremum = float(x[extremum_index])
    stroke = float(np.ptp(x))
    threshold = config.real_plateau_max_amplitude_ratio * stroke

    def find_boundary(direction):
        current = extremum_index
        for _ in range(len(x)):
            following = (current + direction) % len(x)
            delta_current = abs(float(x[current]) - extremum)
            delta_following = abs(float(x[following]) - extremum)
            if delta_following > threshold:
                fraction = (threshold - delta_current) / (delta_following - delta_current)
                return (float(theta[current]) + direction * fraction * step) % 360.0
            current = following
        raise ValueError("The real plateau covers the complete revolution")

    start = find_boundary(-1)
    end = find_boundary(1)
    return start, end, (end - start) % 360.0, empty_is_maximum


def symmetric_values(theta, x):
    return circular_interpolate(theta, x, -np.asarray(theta, dtype=float))


def compute_physical_metrics(curve, score_components, config):
    start1, end1, width, empty_is_maximum = detect_real_plateau(
        curve, score_components, config
    )
    start2 = (-end1) % 360.0
    end2 = (-start1) % 360.0
    exchange_1_to_2 = (start2 - end1) % 360.0
    exchange_2_to_1 = (start1 - end2) % 360.0
    cycle_sum = 2.0 * width + exchange_1_to_2 + exchange_2_to_1
    if not math.isclose(cycle_sum, 360.0, abs_tol=1e-8):
        raise AssertionError(f"Plateau/exchange partition is not 360°: {cycle_sum}")

    x = np.asarray(curve.x, dtype=float)
    stroke = float(np.ptp(x))
    empty_position = float(np.max(x) if empty_is_maximum else np.min(x))

    def volume_normalized(piston, angle):
        """Swept volume: zero at empty position, one at full position."""
        local_angle = angle if piston == 1 else (-angle) % 360.0
        position = circular_interpolate(curve.theta, x, local_angle)
        volume = (
            empty_position - position if empty_is_maximum
            else position - empty_position
        )
        return float(np.clip(volume / max(stroke, config.epsilon), 0.0, 1.0))

    # At end1 piston 1 leaves its empty plateau; piston 2 is the full piston
    # already closing. At end2 the roles are exactly reversed.
    precompression_1_to_2 = 1.0 - volume_normalized(2, end1)
    precompression_2_to_1 = 1.0 - volume_normalized(1, end2)
    return {
        "useful_stroke": stroke,
        "real_plateau_start_deg": start1,
        "real_plateau_end_deg": end1,
        "real_plateau_width_deg": width,
        "symmetric_plateau_start_deg": start2,
        "symmetric_plateau_end_deg": end2,
        "symmetric_plateau_width_deg": width,
        "exchange_1_to_2_deg": exchange_1_to_2,
        "exchange_2_to_1_deg": exchange_2_to_1,
        "precompression_1_to_2_ratio": precompression_1_to_2,
        "precompression_2_to_1_ratio": precompression_2_to_1,
        "empty_plateau_is_maximum": bool(empty_is_maximum),
    }


__all__ = [
    "RefinedSolution", "evaluate_geometry", "refine_candidate",
    "detect_real_plateau", "compute_physical_metrics", "symmetric_values",
    "support_from_parameters", "source_parameters",
]
