"""
src/objective.py
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

        #
        # Invalid geometry
        #

        if not np.all(trajectory.valid):

            return ObjectiveResult(
                total_score=0.0,
                stop_score=0.0,
                forward_score=0.0,
                return_score=0.0,
            )

        #
        # Normalize velocity
        #

        speed = trajectory.speed.copy()

        vmax = np.max(np.abs(speed))

        if vmax > 1e-12:
            speed /= vmax

        target_speed = target.velocity

        n = len(speed)

        q1 = n // 4
        q2 = n // 2

        stop = self._segment_score(
            speed[:q1],
            target_speed[:q1],
        )

        forward = self._segment_score(
            speed[q1:q2],
            target_speed[q1:q2],
        )

        back = self._segment_score(
            speed[q2:],
            target_speed[q2:],
        )

        total = (stop + forward + back) / 3.0

        return ObjectiveResult(
            total_score=total,
            stop_score=stop,
            forward_score=forward,
            return_score=back,
        )

    @staticmethod
    def _segment_score(
        speed: np.ndarray,
        target: np.ndarray,
    ) -> float:

        #
        # Mean error
        #

        mean_error = np.mean(
            np.abs(speed - target)
        )

        #
        # Plateau stability
        #

        stability = np.std(speed)

        #
        # Combined penalty
        #

        penalty = (
            mean_error
            + 0.5 * stability
        )

        score = 100.0 * np.exp(-penalty)

        return float(score)