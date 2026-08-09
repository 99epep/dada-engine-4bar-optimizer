# =========================
# File: mechanism_generator.py (part 1/3)
# =========================

import itertools

from models import Mechanism


def generate_mechanisms(config):
    """
    Exhaustive generation of the mechanism design space.
    """

    for crank, coupler, rocker in itertools.product(
        config.crank_lengths,
        config.coupler_lengths,
        config.rocker_lengths,
    ):
        ground = config.ground_length

        if not _is_valid_lengths(
            ground,
            crank,
            coupler,
            rocker,
        ):
            continue

        yield Mechanism(
            ground=ground,
            crank=crank,
            coupler=coupler,
            rocker=rocker,
        )


def _is_valid_lengths(
    ground,
    crank,
    coupler,
    rocker,
):
    """
    Basic geometric checks.
    """

    lengths = sorted(
        [ground, crank, coupler, rocker]
    )

    # Impossible quadrilateral
    if lengths[0] + lengths[1] >= lengths[2] + lengths[3]:
        return False

    # Grashof condition
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return False

    return True

# =========================
# File: mechanism_generator.py (part 2/3)
# =========================

def mechanism_count(config):
    """
    Number of mechanisms before geometric filtering.
    """

    return (
        len(config.crank_lengths)
        * len(config.coupler_lengths)
        * len(config.rocker_lengths)
    )


def valid_mechanism_count(config):
    """
    Number of mechanisms after geometric filtering.
    """

    return sum(
        1
        for _ in generate_mechanisms(config)
    )


def mechanisms_as_list(config):
    """
    Convenience helper.
    """

    return list(generate_mechanisms(config))


# =========================
# File: mechanism_generator.py (part 3/3)
# =========================

__all__ = [
    "generate_mechanisms",
    "mechanism_count",
    "valid_mechanism_count",
    "mechanisms_as_list",
]