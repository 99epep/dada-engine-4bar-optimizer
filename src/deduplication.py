"""Family-aware geometric diversity filters for both optimization levels."""

from __future__ import annotations


def _relative_difference(first, second, epsilon):
    scale = max(0.5 * (abs(first) + abs(second)), epsilon)
    return abs(first - second) / scale


def solutions_are_close(first, second, family, config, level):
    family = family.upper()
    if level == 1:
        length_tolerance = config.level1_length_proximity_ratio
        support_tolerance = config.level1_F_support_proximity_ratio
    elif level == 2:
        length_tolerance = config.level2_length_proximity_ratio
        support_tolerance = config.level2_F_support_proximity_ratio
    else:
        raise ValueError("level must be 1 or 2")

    for name in ("crank", "coupler", "rocker"):
        if _relative_difference(
            getattr(first.mechanism, name), getattr(second.mechanism, name),
            config.epsilon,
        ) > length_tolerance:
            return False

    if family == "E":
        return True

    first_bc = first.mechanism.coupler
    second_bc = second.mechanism.coupler
    first_coordinates = (
        first.support.local_x / first_bc,
        first.support.local_y / first_bc,
    )
    second_coordinates = (
        second.support.local_x / second_bc,
        second.support.local_y / second_bc,
    )
    return all(
        abs(value1 - value2) <= support_tolerance
        for value1, value2 in zip(first_coordinates, second_coordinates)
    )


def deduplicate_solutions(solutions, family, config, level, score_attribute):
    """Greedily keep the best-scoring representative of each neighbourhood."""
    ordered = sorted(
        solutions, key=lambda item: getattr(item, score_attribute), reverse=True
    )
    kept = []
    for solution in ordered:
        if not any(
            solutions_are_close(solution, existing, family, config, level)
            for existing in kept
        ):
            kept.append(solution)
    return kept


__all__ = ["deduplicate_solutions", "solutions_are_close"]
