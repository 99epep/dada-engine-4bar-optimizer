"""
src/objective.py

Score computation for a candidate trajectory.

The score is based on:

- one 90° stationary phase
- one 90° fast phase
- one 180° slow phase

The stationary phase may be centred on 90° or 270°.

Velocity sign is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------
# Result
# ---------------------------------------------------------


@dataclass(slots=True)
class ObjectiveResult:

    total_score: float

    stop_score: float

    fast_score: float

    slow_score: float


# ---------------------------------------------------------
# Objective
# ---------------------------------------------------------


class Objective:

    def evaluate(
        self,
        speed: np.ndarray,
    ) -> ObjectiveResult:

        speed = np.abs(speed).astype(float)

        vmax = np.max(speed)

        if vmax < 1e-12:

            return ObjectiveResult(0.0, 0.0, 0.0, 0.0)

        speed /= vmax

        score_a = self._evaluate_layout(
            speed,
            stop_start=1,
        )

        score_b = self._evaluate_layout(
            speed,
            stop_start=5,
        )

        if score_a.total_score >= score_b.total_score:
            return score_a

        return score_b

    # -----------------------------------------------------

    def _evaluate_layout(
        self,
        speed,
        stop_start,
    ):

        n = len(speed)

        eighth = n // 8

        stop0 = stop_start * eighth
        stop1 = stop0 + 2 * eighth

        stop = speed[stop0:stop1]

        remaining = np.concatenate(

            (
                speed[:stop0],
                speed[stop1:],
            )

        )

        first = remaining[:2 * eighth]

        second = remaining[2 * eighth:]

        first_mean = np.mean(first)
        second_mean = np.mean(second)

        #
        # identify fast / slow
        #

        if first_mean >= second_mean:

            fast = first
            slow = second

        else:

            fast = second
            slow = first

        stop_score = self._plateau(
            stop,
            0.0,
        )

        fast_score = self._plateau(
            fast,
            1.0,
        )

        slow_score = self._plateau(
            slow,
            0.5,
        )

        total = (
            stop_score
            + fast_score
            + slow_score
        ) / 3.0

        return ObjectiveResult(

            total_score=total,

            stop_score=stop_score,

            fast_score=fast_score,

            slow_score=slow_score,

        )

    # -----------------------------------------------------

    @staticmethod
    def _plateau(
        values,
        target,
    ):

        mean = np.mean(values)

        std = np.std(values)

        error = abs(mean - target)

        penalty = (
            error
            + 0.50 * std
        )

        score = 100.0 * np.exp(-penalty)

        return float(score)