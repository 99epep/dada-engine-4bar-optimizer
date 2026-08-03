"""
src/objective.py

Objective function for four-bar optimization.
"""

from dataclasses import dataclass

import numpy as np

from trajectory import Trajectory
from target import TargetTrajectory


@dataclass(slots=True)
class ObjectiveResult:

    total_score: float

    stop_score: float
    forward_score: float
    return_score: float


class Objective:

    def evaluate(
        self,
        trajectory: Trajectory,
        target: TargetTrajectory,
    ) -> ObjectiveResult:

        v = trajectory.speed
        vt = target.velocity

        n = len(v)

        q1 = n // 4
        q2 = n // 2

        stop = self._segment_score(
            v[:q1],
            vt[:q1],
        )

        forward = self._segment_score(
            v[q1:q2],
            vt[q1:q2],
        )

        back = self._segment_score(
            v[q2:],
            vt[q2:],
        )

        total = (
            stop +
            forward +
            back
        ) / 3.0

        return ObjectiveResult(
            total_score=total,
            stop_score=stop,
            forward_score=forward,
            return_score=back,
        )

    @staticmethod
    def _segment_score(
        speed,
        target,
    ):

        #
        # erreur moyenne
        #

        mean_error = np.mean(
            np.abs(speed - target)
        )

        #
        # stabilité
        #

        std = np.std(speed)

        #
        # score
        #

        score = 100.0

        score -= 50.0 * mean_error

        score -= 50.0 * std

        return max(score, 0.0)
