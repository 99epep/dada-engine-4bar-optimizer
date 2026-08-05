# =========================
# File: config.py
# =========================

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverConfig:
    # ---------- Search ----------
    top_n: int = 100

    # ---------- Angular sampling ----------
    angle_start_deg: float = 0.0
    angle_end_deg: float = 360.0
    angle_step_deg: float = 1.0

    # ---------- E search ----------
    e_half_circle_step_deg: float = 5.0

    # ---------- F search ----------
    f_radius_factor: float = 2.0
    f_grid_step_factor: float = 0.10

    # ---------- Filter 1 ----------
    plateau_max_amplitude_ratio: float = 0.10

    plateau_center_1_deg: float = 90.0
    plateau_center_2_deg: float = 270.0
    plateau_center_tolerance_deg: float = 20.0

    plateau_min_width_deg: float = 45.0
    plateau_max_width_deg: float = 110.0

    # ---------- Filter 2 ----------
    bisector_targets_deg = (0.0, 180.0)
    bisector_tolerance_deg: float = 15.0

    # ---------- Scoring ----------
    weight_fast_plateau: float = 0.25
    weight_slow_plateau: float = 0.25
    weight_fast_center: float = 0.35
    weight_static_plateau: float = 0.15