# =========================
# File: optimizer.py (part 1/5)
# =========================

import numpy as np

from kinematics import build_candidate_curve, solve
from deduplication import deduplicate_solutions
from mechanism_generator import generate_mechanisms
from models import SearchStatistics, Solution
from objective import evaluate_candidate
from support_search import generate_candidates


def optimize(config):
    """
    Global optimization loop.

    Keeps independent rankings for E and F so that the best
    candidate of each support family is always preserved.
    """

    stats = SearchStatistics()

    best_E = []
    best_F = []

    for mechanism in generate_mechanisms(config):

        # --------------------------------------------------
        # Grashof : rejet immédiat des mécanismes qui ne
        # peuvent pas être des crank-rockers.
        # --------------------------------------------------

        if not is_grashof_crank_rocker(mechanism):
            stats.grashof_rejected += 1
            continue

        stats.mechanisms_tested += 1

        kinematics = solve(mechanism, config)

        if not kinematics.valid:
            continue

        supports = generate_candidates(mechanism, config)

        stats.supports_generated += len(supports)

        for support in supports:

            curve = build_candidate_curve(
                kinematics,
                support,
            )

            if support.kind.value == "F" and not _F_motion_is_valid(curve, config):
                stats.filter1_rejected += 1
                continue

            result = evaluate_candidate(
                curve,
                config,
            )

            if not result.accepted:

                if result.reject_reason == "bisector":
                    stats.filter2_rejected += 1
                else:
                    stats.filter1_rejected += 1

                continue

            stats.accepted += 1

            solution = Solution(
                score=result.score,
                mechanism=mechanism,
                support=support,
                curve=curve,
                score_components=result.score_components,
            )

            if support.kind.value == "E":
                best_E.append(solution)
            else:
                best_F.append(solution)

    return (
        _finalize_solutions(best_E, config),
        _finalize_solutions(best_F, config),
    )


def is_grashof_crank_rocker(mechanism):
    """
    Return True when the four-bar mechanism satisfies Grashof's
    condition with the crank AB selected as the shortest link
    candidate.

    For a crank-rocker:

        shortest + longest <= the other two links

    The fixed link AD is included in the four-link comparison.
    """

    lengths = (
        mechanism.ground,
        mechanism.crank,
        mechanism.coupler,
        mechanism.rocker,
    )

    shortest = min(lengths)
    longest = max(lengths)

    remaining = sum(lengths) - shortest - longest

    return (
        shortest + longest
        <= remaining
    )


def best_solutions(solutions):
    """
    Returns solutions sorted by decreasing score.
    """

    return sorted(
        solutions,
        key=lambda solution: solution.score,
        reverse=True,
    )


def best_scores(solutions):
    """
    Convenience function for debugging.
    """

    return [
        solution.score
        for solution in best_solutions(solutions)
    ]


def _finalize_solutions(solutions, config):
    """
    Deduplicate geometrically first, then keep the N best.
    """

    if not solutions:
        return []
    family = solutions[0].support.kind.value
    solutions = deduplicate_solutions(
        solutions, family, config, level=1, score_attribute="score"
    )

    return solutions[:config.max_solutions]


# =========================
# File: optimizer.py (part 3/5)
# =========================

def optimize_with_statistics(config):
    """
    Same optimization loop as optimize(), with independent E/F
    rankings and execution statistics.
    """

    stats = SearchStatistics()

    best_E = []
    best_F = []

    for mechanism in generate_mechanisms(config):

        if not is_grashof_crank_rocker(mechanism):
            stats.grashof_rejected += 1
            continue

        stats.mechanisms_tested += 1

        kinematics = solve(mechanism, config)

        if not kinematics.valid:
            continue

        supports = generate_candidates(mechanism, config)

        stats.supports_generated += len(supports)

        for support in supports:

            curve = build_candidate_curve(
                kinematics,
                support,
            )

            if support.kind.value == "F" and not _F_motion_is_valid(curve, config):
                stats.filter1_rejected += 1
                continue

            result = evaluate_candidate(
                curve,
                config,
            )

            if not result.accepted:

                if result.reject_reason == "bisector":
                    stats.filter2_rejected += 1
                else:
                    stats.filter1_rejected += 1

                continue

            stats.accepted += 1

            solution = Solution(
                score=result.score,
                mechanism=mechanism,
                support=support,
                curve=curve,
                score_components=result.score_components,
            )

            if support.kind.value == "E":
                best_E.append(solution)
            else:
                best_F.append(solution)

    return (
        _finalize_solutions(best_E, config),
        _finalize_solutions(best_F, config),
        stats,
    )


def optimize_family_with_statistics(config, family):
    """Run level 1 for E or F only, without generating the other supports."""
    family = family.upper()
    if family not in ("E", "F"):
        raise ValueError("family must be E or F")

    stats = SearchStatistics()
    accepted_solutions = []
    for mechanism in generate_mechanisms(config):
        if not is_grashof_crank_rocker(mechanism):
            stats.grashof_rejected += 1
            continue
        stats.mechanisms_tested += 1
        kinematics = solve(mechanism, config)
        if not kinematics.valid:
            continue
        supports = generate_candidates(mechanism, config, family=family)
        stats.supports_generated += len(supports)
        for support in supports:
            curve = build_candidate_curve(kinematics, support)
            if family == "F" and not _F_motion_is_valid(curve, config):
                stats.filter1_rejected += 1
                continue
            result = evaluate_candidate(curve, config)
            if not result.accepted:
                if result.reject_reason == "bisector":
                    stats.filter2_rejected += 1
                else:
                    stats.filter1_rejected += 1
                continue
            stats.accepted += 1
            accepted_solutions.append(Solution(
                score=result.score,
                mechanism=mechanism,
                support=support,
                curve=curve,
                score_components=result.score_components,
            ))
    return _finalize_solutions(accepted_solutions, config), stats


def _F_motion_is_valid(curve, config):
    """F must move no farther vertically than along the piston axis X."""
    x_stroke = float(np.ptp(curve.x))
    y_stroke = float(np.ptp(curve.y))
    return y_stroke <= x_stroke + config.epsilon


def _filter_geometrically_close(solutions, config):
    """
    Remove geometrically redundant solutions.

    Solutions are already sorted from best to worst.
    Therefore the first solution kept for a given geometric
    neighbourhood is necessarily the best-scoring one.

    Two mechanisms are considered geometrically close when all
    four principal lengths differ by no more than the configured
    tolerance:

        ground
        crank
        coupler
        rocker
    """

    if not solutions:
        return []
    return deduplicate_solutions(
        solutions, solutions[0].support.kind.value, config,
        level=1, score_attribute="score",
    )


def print_statistics(stats):
    """
    Simple execution summary.
    """

    print()
    print("=" * 60)
    print("Optimization statistics")
    print("=" * 60)

    print(f"Mechanisms tested      : {stats.mechanisms_tested}")
    print(f"Supports generated     : {stats.supports_generated}")
    print(f"Rejected by filter #1  : {stats.filter1_rejected}")
    print(f"Rejected by filter #2  : {stats.filter2_rejected}")
    print(f"Accepted candidates    : {stats.accepted}")

    total = max(stats.supports_generated, 1)

    print()
    print(f"Acceptance rate        : {100.0 * stats.accepted / total:.2f}%")
    print(
        f"Filter #1 rejection    : "
        f"{100.0 * stats.filter1_rejected / total:.2f}%"
    )
    print(
        f"Filter #2 rejection    : "
        f"{100.0 * stats.filter2_rejected / total:.2f}%"
    )

# =========================
# File: optimizer.py (part 5/5)
# =========================

__all__ = [
    "optimize",
    "optimize_with_statistics",
    "best_scores",
    "best_solutions",
    "print_statistics",
    "optimize_family_with_statistics",
]
