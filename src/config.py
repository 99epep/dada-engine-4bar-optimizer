# =========================
# File: config.py
# =========================

from dataclasses import dataclass, field
import numpy as np


@dataclass(slots=True)
class SolverConfig:

    # --------------------------------------------------
    # Cinématique
    # --------------------------------------------------

    angle_start_deg: float = 0.0
    angle_end_deg: float = 360.0
    angle_step_deg: float = 5.0

    # --------------------------------------------------
    # Espace de recherche
    # --------------------------------------------------

    # Bâti AD fixé.
    ground_length: float = 100.0

    # Manivelle AB.
    crank_lengths: np.ndarray = field(
        default_factory=lambda: np.arange(45.0, 96.0, 10.0)
    )

    # Bielle BC et culbuteur CD.
    coupler_lengths: np.ndarray = field(
        default_factory=lambda: np.arange(100.0, 221.0, 40.0)
    )

    rocker_lengths: np.ndarray = field(
        default_factory=lambda: np.arange(100.0, 221.0, 40.0)
    )

    # --------------------------------------------------
    # Solutions
    # --------------------------------------------------

    max_solutions: int = 20

    # --------------------------------------------------
    # Filtre 1 : zone immobile
    # --------------------------------------------------

    plateau_max_amplitude_ratio: float = 0.10

    plateau_center_1_deg: float = 90.0
    plateau_center_2_deg: float = 270.0

    plateau_center_min_deg: float = 80.0
    plateau_center_max_deg: float = 100.0

    plateau_center_min_2_deg: float = 260.0
    plateau_center_max_2_deg: float = 280.0

    plateau_min_width_deg: float = 40.0
    plateau_max_width_deg: float = 120.0

    # --------------------------------------------------
    # Filtre 2 : phase rapide
    # --------------------------------------------------

    bisector_targets_deg: tuple = (
        0.0,
        180.0,
    )

    bisector_tolerance_deg: float = 10.0

    # --------------------------------------------------
    # Classement
    # --------------------------------------------------

    acceleration_exclusion_deg: float = 10.0

    # --------------------------------------------------
    # Recherche des supports
    # --------------------------------------------------

    support_E_step_deg: float = 10.0
    support_F_radius_factor: float = 3.0
    support_F_grid_step_factor: float = 0.20

    # --------------------------------------------------
    # Déduplication géométrique des résultats
    # --------------------------------------------------

    geometry_proximity_mm: float = 10.0

    # --------------------------------------------------
    # Divers
    # --------------------------------------------------

    epsilon: float = 1e-12
    random_seed: int | None = None
    verbose: bool = True


__all__ = [
    "SolverConfig",
]
