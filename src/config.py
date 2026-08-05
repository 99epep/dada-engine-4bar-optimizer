# =========================
# File: config.py (part 1/2)
# =========================

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class SolverConfig:

    # --------------------------------------------------
    # Crank sampling
    # --------------------------------------------------

    angle_start_deg: float = 0.0
    angle_end_deg: float = 360.0
    angle_step_deg: float = 1.0

    # --------------------------------------------------
    # Mechanism search space
    # --------------------------------------------------

    ground_lengths: np.ndarray = np.arange(20.0, 101.0, 5.0)
    crank_lengths: np.ndarray = np.arange(10.0, 61.0, 5.0)
    coupler_lengths: np.ndarray = np.arange(20.0, 121.0, 5.0)
    rocker_lengths: np.ndarray = np.arange(20.0, 121.0, 5.0)

    # --------------------------------------------------
    # Number of retained solutions
    # --------------------------------------------------

    max_solutions: int = 500

    # --------------------------------------------------
    # Filter 1
    # --------------------------------------------------

    plateau_max_amplitude_ratio: float = 0.10

    plateau_center_1_deg: float = 90.0
    plateau_center_2_deg: float = 270.0
    plateau_center_tolerance_deg: float = 20.0

    plateau_min_width_deg: float = 45.0
    plateau_max_width_deg: float = 110.0

    # --------------------------------------------------
    # Filter 2
    # --------------------------------------------------

    bisector_targets_deg = (
        0.0,
        180.0,
    )

    bisector_tolerance_deg: float = 15.0

    # --------------------------------------------------
    # Filter 3 weights
    # --------------------------------------------------

    weight_fast_plateau: float = 0.25
    weight_slow_plateau: float = 0.25
    weight_fast_center: float = 0.35
    weight_static_plateau: float = 0.15

# =========================
# File: config.py (part 2/2)
# =========================

    # --------------------------------------------------
    # Numerical tolerances
    # --------------------------------------------------

    epsilon: float = 1e-12

    # --------------------------------------------------
    # Support search
    # --------------------------------------------------

    support_E_step_deg: float = 5.0
    support_F_radius_factor: float = 2.0
    support_F_grid_step_factor: float = 0.10

    # --------------------------------------------------
    # Miscellaneous
    # --------------------------------------------------

    random_seed: int | None = None

    verbose: bool = True


__all__ = [
    "SolverConfig",
]