# =========================
# File: config.py
# TEST V0.1 — quelques secondes
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
    # Petit espace de recherche de test
    # --------------------------------------------------

    ground_lengths: np.ndarray = field(
        default_factory=lambda: np.arange(160.0, 201.0, 20.0)
    )

    crank_lengths: np.ndarray = field(
        default_factory=lambda: np.arange(80.0, 121.0, 20.0)
    )

    coupler_lengths: np.ndarray = field(
        default_factory=lambda: np.arange(160.0, 201.0, 20.0)
    )

    rocker_lengths: np.ndarray = field(
        default_factory=lambda: np.arange(160.0, 201.0, 20.0)
    )

    # --------------------------------------------------
    # Solutions
    # --------------------------------------------------

    max_solutions: int = 20

    # --------------------------------------------------
    # Filtre 1
    # --------------------------------------------------

    plateau_max_amplitude_ratio: float = 0.10

    plateau_center_1_deg: float = 90.0
    plateau_center_2_deg: float = 270.0
    plateau_center_tolerance_deg: float = 20.0

    plateau_min_width_deg: float = 45.0
    plateau_max_width_deg: float = 110.0

    # --------------------------------------------------
    # Filtre 2
    # --------------------------------------------------

    bisector_targets_deg: tuple = (
        0.0,
        180.0,
    )

    bisector_tolerance_deg: float = 15.0

    # --------------------------------------------------
    # Filtre 3
    # --------------------------------------------------

    weight_fast_plateau: float = 0.25
    weight_slow_plateau: float = 0.25
    weight_fast_center: float = 0.35
    weight_static_plateau: float = 0.15

    # --------------------------------------------------
    # Recherche des supports
    # --------------------------------------------------

    support_E_step_deg: float = 5.0
    support_F_radius_factor: float = 2.0
    support_F_grid_step_factor: float = 0.10

    # --------------------------------------------------
    # Divers
    # --------------------------------------------------

    epsilon: float = 1e-12
    random_seed: int | None = None
    verbose: bool = True


__all__ = [
    "SolverConfig",
]
