# =========================
# File: config.py
# =========================

from dataclasses import dataclass, field
import numpy as np


@dataclass(slots=True)
class SolverConfig:

    # --------------------------------------------------
    # Niveau de définition
    # --------------------------------------------------

    # True  : haute définition
    # False : mode rapide (~1 min)
    high_definition: bool = True

    # --------------------------------------------------
    # Cinématique
    # --------------------------------------------------

    angle_start_deg: float = 0.0
    angle_end_deg: float = 360.0
    angle_step_deg: float = 2.0

    # Niveau 2 : résolution réservée aux courbes et métriques finales.
    refinement_angle_step_deg: float = 0.1

    # --------------------------------------------------
    # Espace de recherche
    # --------------------------------------------------

    # Bâti AD fixé.
    ground_length: float = 100.0

    # Manivelle AB.
    crank_lengths: np.ndarray = field(
        default_factory=lambda: np.arange(25.0, 76.0, 4.0)
    )

    # Bielle BC et culbuteur CD.
    coupler_lengths: np.ndarray = field(
        default_factory=lambda: np.arange(100.0, 221.0, 3.0)
    )

    rocker_lengths: np.ndarray = field(
        default_factory=lambda: np.arange(100.0, 221.0, 3.0)
    )

    def __post_init__(self):
        if self.high_definition:
            return

        # --------------------------------------------------
        # Ancienne configuration rapide (~1 min)
        # --------------------------------------------------

        self.angle_step_deg = 5.0

        self.crank_lengths = np.arange(
            25.0, 76.0, 10.0
        )

        self.coupler_lengths = np.arange(
            100.0, 221.0, 20.0
        )

        self.rocker_lengths = np.array([
            100.0, 110.0, 120.0, 130.0, 140.0,
            160.0, 180.0, 200.0, 220.0,
        ])

    # --------------------------------------------------
    # Solutions
    # --------------------------------------------------

    max_solutions: int = 100

    # --------------------------------------------------
    # Filtre 1 : zone immobile
    # --------------------------------------------------

    plateau_max_amplitude_ratio: float = 0.03

    # Plateau physique de la machine complète, indépendant du filtre 1.
    real_plateau_max_amplitude_ratio: float = 0.01

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

    # A3 reste voisin de +/-40 deg afin de préserver une vraie phase rapide
    # et une durée significative du plateau de la loi idéale.
    a3_targets_deg: tuple = (
        40.0,
        320.0,
    )

    a3_tolerance_deg: float = 25.0

    # Une dissymétrie équivalente à ce décalage angulaire annule le score.
    symmetry_zero_score_shift_deg: float = 12.0

    # --------------------------------------------------
    # Classement
    # --------------------------------------------------

    acceleration_exclusion_deg: float = 5.0

    # --------------------------------------------------
    # Recherche des supports
    # --------------------------------------------------

    support_F_radius_factor: float = 1.5
    support_F_grid_step_factor: float = 0.20

    # --------------------------------------------------
    # Déduplication géométrique des résultats
    # --------------------------------------------------

    # Niveau 1 : voisinage assez large pour préserver des familles diverses.
    level1_length_proximity_ratio: float = 0.08
    level1_F_support_proximity_ratio: float = 0.25

    # Niveau 2 : ne fusionner que les solutions réellement convergentes.
    level2_length_proximity_ratio: float = 0.02
    level2_F_support_proximity_ratio: float = 0.05

    # --------------------------------------------------
    # Divers
    # --------------------------------------------------

    epsilon: float = 1e-12
    random_seed: int | None = None
    verbose: bool = True


__all__ = [
    "SolverConfig",
]
