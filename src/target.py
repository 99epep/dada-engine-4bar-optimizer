"""
src/target.py

Target motion definition.

The target is specified by a velocity profile over one revolution:

0 - 25 %   : velocity = 0
25 - 50 %  : velocity = +1
50 - 100 % : velocity = -0.5

The corresponding stroke is obtained by numerical integration.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TargetTrajectory:

    theta: np.ndarray

    velocity: np.ndarray

    stroke: np.ndarray


class TargetMotion:

    def __init__(self, samples: int = 180):

        self.samples = samples

    def generate(self) -> TargetTrajectory:

        theta = np.linspace(
            0.0,
            2.0 * np.pi,
            self.samples,
            endpoint=False,
        )

        velocity = np.zeros(self.samples)

        q1 = self.samples // 4
        q2 = self.samples // 2

        #
        # Velocity law
        #

        velocity[:q1] = 0.0

        velocity[q1:q2] = 1.0

        velocity[q2:] = -0.5

        #
        # Numerical integration
        #

        dtheta = 2.0 * np.pi / self.samples

        stroke = np.zeros(self.samples)

        for i in range(1, self.samples):

            stroke[i] = (
                stroke[i - 1]
                + velocity[i - 1] * dtheta
            )

        #
        # Normalize between 0 and 1
        #

        stroke -= stroke.min()

        amplitude = stroke.max()

        if amplitude > 0.0:
            stroke /= amplitude

        return TargetTrajectory(
            theta=theta,
            velocity=velocity,
            stroke=stroke,
        )