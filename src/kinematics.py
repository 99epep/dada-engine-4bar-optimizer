"""
kinematics.py

Kinematic solver for a planar four-bar linkage.

The mechanism is solved by computing the intersection between:
- the circle centered on B (radius BC)
- the circle centered on D (radius CD)
"""

from dataclasses import dataclass
import numpy as np

from geometry import FourBarGeometry


@dataclass(slots=True)
class KinematicState:
    theta: float

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    E: np.ndarray

    theta_coupler: float
    theta_rocker: float


class FourBarKinematics:

    def __init__(self, geometry: FourBarGeometry):
        self.g = geometry
        self.g.validate()

    def solve(self, theta: float) -> KinematicState:

        A = self.g.A
        D = self.g.D

        B = np.array([
            self.g.crank * np.cos(theta),
            self.g.crank * np.sin(theta)
        ])

        C = self._solve_C(B, D)

        E = self.g.point_E(D, C)

        theta_coupler = np.arctan2(
            C[1] - B[1],
            C[0] - B[0]
        )

        theta_rocker = np.arctan2(
            C[1] - D[1],
            C[0] - D[0]
        )

        return KinematicState(
            theta=theta,
            A=A,
            B=B,
            C=C,
            D=D,
            E=E,
            theta_coupler=theta_coupler,
            theta_rocker=theta_rocker,
        )

    def _solve_C(self, B, D):

        r0 = self.g.coupler
        r1 = self.g.rocker

        d = np.linalg.norm(D - B)

        if d > r0 + r1:
            raise ValueError("Mechanism cannot close.")

        if d < abs(r0 - r1):
            raise ValueError("One circle lies inside the other.")

        if d == 0:
            raise ValueError("Coincident circle centers.")

        a = (r0**2 - r1**2 + d**2) / (2*d)

        h = np.sqrt(max(r0**2 - a**2, 0.0))

        P2 = B + a*(D-B)/d

        x3 = P2[0] + h*(D[1]-B[1])/d
        y3 = P2[1] - h*(D[0]-B[0])/d

        x4 = P2[0] - h*(D[1]-B[1])/d
        y4 = P2[1] + h*(D[0]-B[0])/d

        C1 = np.array([x3, y3])
        C2 = np.array([x4, y4])

        #
        # Pour la V0.1 on choisit simplement
        # le point ayant la plus grande ordonnée.
        #
        # Plus tard on suivra automatiquement
        # la branche continue.
        #
        if C1[1] >= C2[1]:
            return C1

        return C2