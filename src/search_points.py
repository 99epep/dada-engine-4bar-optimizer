"""
src/search_points.py

Search the best attachment points on:

- the balancer (point E)
- the coupler (point F)

for one given four-bar geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from kinematics import FourBarKinematics
from objective import Objective


# ---------------------------------------------------------
# Types
# ---------------------------------------------------------


class Support(Enum):
    BALANCER = "balancer"
    COUPLER = "coupler"


@dataclass(slots=True)
class Candidate:

    support: Support

    local: np.ndarray

    score: float

    trajectory: np.ndarray

    speed: np.ndarray


# ---------------------------------------------------------
# Search
# ---------------------------------------------------------


class PointSearcher:

    def __init__(self,
                 kinematics: FourBarKinematics,
                 objective: Objective):

        self.kin = kinematics
        self.objective = objective

    # -----------------------------------------------------

    def search(self,
               keep=20):

        candidates = []

        candidates.extend(
            self._search_balancer()
        )

        candidates.extend(
            self._search_coupler()
        )

        candidates.sort(
            key=lambda c: c.score,
            reverse=True,
        )

        return candidates[:keep]

    # -----------------------------------------------------

    def _search_balancer(self):

        traj = self.kin.solve_cycle()

        results = []

        r = self.kin.g.rocker

        #
        # 36 directions
        #

        for angle in np.linspace(
                -90.0,
                90.0,
                36,
                endpoint=False):

            a = np.deg2rad(angle)

            local = np.array([
                r*np.cos(a),
                r*np.sin(a)
            ])

            stroke = self.kin.stroke_from_balancer(
                traj,
                local,
            )

            speed = np.gradient(stroke)

            score = self.objective.evaluate(
                speed
            )

            results.append(

                Candidate(

                    support=Support.BALANCER,

                    local=local,

                    score=score.total_score,

                    trajectory=stroke,

                    speed=speed,

                )

            )

        return results

    # -----------------------------------------------------

    def _search_coupler(self):

        traj = self.kin.solve_cycle()

        results = []

        L = self.kin.g.coupler

        radius = 2.0 * L

        step = 0.25 * L

        xs = np.arange(
            -radius,
            radius + step,
            step,
        )

        ys = xs.copy()

        centre = np.array([
            0.5 * L,
            0.0,
        ])

        for x in xs:

            for y in ys:

                p = np.array([x, y])

                if np.linalg.norm(
                        p-centre) > radius:

                    continue

                stroke = self.kin.stroke_from_coupler(
                    traj,
                    p,
                )

                speed = np.gradient(stroke)

                score = self.objective.evaluate(
                    speed
                )

                results.append(

                    Candidate(

                        support=Support.COUPLER,

                        local=p,

                        score=score.total_score,

                        trajectory=stroke,

                        speed=speed,

                    )

                )

        return results