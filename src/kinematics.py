"""
src/kinematics.py

Four-bar linkage kinematics solver.

Part 1/2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from geometry import FourBarGeometry


# ============================================================
# Data structures
# ============================================================


@dataclass(slots=True)
class KinematicState:
    """Instantaneous mechanism state."""

    theta: float

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    E: np.ndarray

    theta_coupler: float
    theta_rocker: float

    valid: bool = True


@dataclass(slots=True)
class Trajectory:
    """One complete revolution."""

    theta: np.ndarray

    B: np.ndarray
    C: np.ndarray
    E: np.ndarray

    stroke: np.ndarray
    speed: np.ndarray
    acceleration: np.ndarray

    valid: np.ndarray


# ============================================================
# Solver
# ============================================================


class FourBarKinematics:

    def __init__(self, geometry: FourBarGeometry):

        geometry.validate()

        self.g = geometry

    # --------------------------------------------------------

    def solve(self, theta: float) -> KinematicState:

        A = self.g.A
        D = self.g.D

        B = np.array(
            [
                self.g.crank * np.cos(theta),
                self.g.crank * np.sin(theta),
            ],
            dtype=float,
        )

        C = self._solve_c(B, D)

        E = self.g.point_E(D, C)

        theta_coupler = np.arctan2(
            C[1] - B[1],
            C[0] - B[0],
        )

        theta_rocker = np.arctan2(
            C[1] - D[1],
            C[0] - D[0],
        )

        self._check_lengths(A, B, C, D)

        return KinematicState(
            theta=theta,
            A=A,
            B=B,
            C=C,
            D=D,
            E=E,
            theta_coupler=theta_coupler,
            theta_rocker=theta_rocker,
            valid=True,
        )

    # --------------------------------------------------------

    def _solve_c(
        self,
        B: np.ndarray,
        D: np.ndarray,
        previous: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        solutions = self._circle_intersections(
            B,
            self.g.coupler,
            D,
            self.g.rocker,
        )

        if len(solutions) == 0:
            raise ValueError("Mechanism cannot close.")

        if len(solutions) == 1:
            return solutions[0]

        if previous is None:

            if solutions[0][1] >= solutions[1][1]:
                return solutions[0]

            return solutions[1]

        d0 = np.linalg.norm(previous - solutions[0])
        d1 = np.linalg.norm(previous - solutions[1])

        if d0 <= d1:
            return solutions[0]

        return solutions[1]

    # --------------------------------------------------------

    @staticmethod
    def _circle_intersections(
        c0: np.ndarray,
        r0: float,
        c1: np.ndarray,
        r1: float,
    ):

        d = np.linalg.norm(c1 - c0)

        if d > r0 + r1:
            return []

        if d < abs(r0 - r1):
            return []

        if np.isclose(d, 0.0):
            return []

        a = (r0**2 - r1**2 + d**2) / (2 * d)

        h2 = r0**2 - a**2

        if h2 < 0:
            h2 = 0.0

        h = np.sqrt(h2)

        p2 = c0 + a * (c1 - c0) / d

        rx = -(c1[1] - c0[1]) * h / d
        ry = (c1[0] - c0[0]) * h / d

        p3 = np.array(
            [
                p2[0] + rx,
                p2[1] + ry,
            ]
        )

        p4 = np.array(
            [
                p2[0] - rx,
                p2[1] - ry,
            ]
        )

        if np.allclose(p3, p4):
            return [p3]

        return [p3, p4]

    # --------------------------------------------------------

    def _check_lengths(
        self,
        A,
        B,
        C,
        D,
    ):

        tol = 1e-9

        assert np.isclose(
            np.linalg.norm(B - A),
            self.g.crank,
            atol=tol,
        )

        assert np.isclose(
            np.linalg.norm(C - B),
            self.g.coupler,
            atol=tol,
        )

        assert np.isclose(
            np.linalg.norm(C - D),
            self.g.rocker,
            atol=tol,
        )
    # --------------------------------------------------------
    # Complete revolution
    # --------------------------------------------------------

    def solve_cycle(
        self,
        samples: int = 180,
    ) -> Trajectory:

        theta = np.linspace(
            0.0,
            2.0 * np.pi,
            samples,
            endpoint=False,
        )

        B = np.zeros((samples, 2), dtype=float)
        C = np.zeros((samples, 2), dtype=float)
        E = np.zeros((samples, 2), dtype=float)

        valid = np.ones(samples, dtype=bool)

        D = self.g.D
        previous_C = None

        for i, t in enumerate(theta):

            B[i] = np.array(
                [
                    self.g.crank * np.cos(t),
                    self.g.crank * np.sin(t),
                ]
            )

            try:

                C[i] = self._solve_c(
                    B[i],
                    D,
                    previous=previous_C,
                )

                previous_C = C[i]

                E[i] = self.g.point_E(
                    D,
                    C[i],
                )

            except ValueError:

                valid[i] = False

                if i > 0:
                    C[i] = C[i - 1]
                    E[i] = E[i - 1]

        #
        # Useful piston stroke
        # (parallel to crank angle = 0)
        #

        stroke = E[:, 0].copy()

        #
        # Numerical derivatives
        #

        dtheta = 2.0 * np.pi / samples

        speed = np.gradient(
            stroke,
            dtheta,
            edge_order=2,
        )

        acceleration = np.gradient(
            speed,
            dtheta,
            edge_order=2,
        )

        return Trajectory(
            theta=theta,
            B=B,
            C=C,
            E=E,
            stroke=stroke,
            speed=speed,
            acceleration=acceleration,
            valid=valid,
        )
