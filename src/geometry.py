"""
geometry.py

Definition of the geometry of a planar four-bar linkage.

Reference frame:
    - A = (0, 0)
    - D = (ground, 0)

The point E is attached to the output rocker (DC).
Its position is expressed in the local frame of DC:
    - e_longitudinal : distance along D -> C
    - e_normal       : perpendicular distance to DC
"""

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class FourBarGeometry:
    """
    Geometric definition of a four-bar linkage.

    Parameters
    ----------
    ground : float
        Length AD (fixed frame).

    crank : float
        Length AB (input crank).

    coupler : float
        Length BC.

    rocker : float
        Length CD (output rocker).

    e_longitudinal : float
        Position of E along the local axis D -> C.

    e_normal : float
        Signed distance of E perpendicular to DC.
    """

    ground: float
    crank: float
    coupler: float
    rocker: float

    e_longitudinal: float
    e_normal: float

    def validate(self) -> None:
        """Raises ValueError if the geometry is invalid."""

        lengths = (
            self.ground,
            self.crank,
            self.coupler,
            self.rocker,
        )

        if any(L <= 0.0 for L in lengths):
            raise ValueError("All bar lengths must be strictly positive.")

    @property
    def A(self) -> np.ndarray:
        return np.array([0.0, 0.0])

    @property
    def D(self) -> np.ndarray:
        return np.array([self.ground, 0.0])

    def point_E(self, D: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Computes the absolute coordinates of E.

        Parameters
        ----------
        D : ndarray
            Coordinates of pivot D.

        C : ndarray
            Coordinates of pivot C.

        Returns
        -------
        ndarray
            Coordinates of point E.
        """

        dc = C - D
        length = np.linalg.norm(dc)

        if length == 0:
            raise ValueError("Points D and C are coincident.")

        u = dc / length

        # Left-hand normal
        n = np.array([-u[1], u[0]])

        return (
            D
            + self.e_longitudinal * u
            + self.e_normal * n
        )