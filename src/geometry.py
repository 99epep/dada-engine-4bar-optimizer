from dataclasses import dataclass


@dataclass
class FourBarGeometry:
    """
    Geometry of a planar four-bar linkage.

    A ----- B
    |       |
    D ----- C

    Ground link : AD
    Input crank : AB
    Coupler     : BC
    Output link : CD

    Point E is defined in the local frame of BC.
    """

    ground: float      # AD
    crank: float       # AB
    coupler: float     # BC
    rocker: float      # CD

    # Coordinates of E in the local frame of BC
    e_longitudinal: float
    e_normal: float