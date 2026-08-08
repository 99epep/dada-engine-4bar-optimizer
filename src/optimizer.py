# =========================
# File: optimizer.py (part 1/5)
# =========================

from heapq import heappush, heapreplace

from kinematics import build_candidate_curve, solve
from mechanism_generator import generate_mechanisms
from models import SearchStatistics, Solution
from objective import evaluate_candidate
from support_search import generate_candidates


def optimize(config):
    """
    Global optimization loop.

    Returns
    -------
    list[Solution]
    """

    stats = SearchStatistics()

    best = []

    for mechanism in generate_mechanisms(config):

        stats.mechanisms_tested += 1

        kinematics = solve(mechanism, config)

        if not kinematics.valid:
            continue

        supports = generate_candidates(mechanism)

        stats.supports_generated += len(supports)

        for support in supports:

            curve = build_candidate_curve(
                kinematics,
                support,
            )

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

            _push_solution(
                best,
                solution,
                config.max_solutions,
            )

    best.sort(
        key=lambda s: s.score,
        reverse=True,
    )

    return best

# =========================
# File: optimizer.py (part 2/5)
# =========================

def _push_solution(heap, solution, maximum):
    """
    Keep only the N best solutions.
    """

    item = (solution.score, solution)

    if len(heap) < maximum:
        heappush(heap, item)
        return

    if solution.score <= heap[0][0]:
        return

    heapreplace(heap, item)


def best_scores(heap):
    """
    Convenience function for debugging.
    """

    return sorted(
        (score for score, _ in heap),
        reverse=True,
    )


def best_solutions(heap):
    """
    Returns solutions sorted by decreasing score.
    """

    return [
        solution
        for _, solution in sorted(
            heap,
            key=lambda item: item[0],
            reverse=True,
        )
    ]

# =========================
# File: optimizer.py (part 3/5)
# =========================

def optimize_with_statistics(config):
    """
    Same optimization loop as optimize(), but also returns execution
    statistics for diagnostics and tuning.
    """

    stats = SearchStatistics()

    best = []

    for mechanism in generate_mechanisms(config):

        stats.mechanisms_tested += 1

        kinematics = solve(mechanism, config)

        if not kinematics.valid:
            continue

        supports = generate_candidates(mechanism)

        stats.supports_generated += len(supports)

        for support in supports:

            curve = build_candidate_curve(
                kinematics,
                support,
            )

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

            _push_solution(
                best,
                Solution(
                    score=result.score,
                    mechanism=mechanism,
                    support=support,
                    curve=curve,
                    score_components=result.score_components,
                ),
                config.max_solutions,
            )

    return (
        best_solutions(best),
        stats,
    )

# =========================
# File: optimizer.py (part 4/5)
# =========================

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
]