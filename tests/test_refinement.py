import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from config import SolverConfig
from deduplication import solutions_are_close
from kinematics import build_candidate_curve, solve
from models import Mechanism, Support, SupportKind
from objective import _symmetry_quality
from optimizer import _F_motion_is_valid
from refinement import (
    circular_interpolate,
    compute_physical_metrics,
    evaluate_geometry,
    final_E_support,
    geometry_in_level1_domain,
    refine_candidate,
    symmetric_values,
)
from result_io import load_refined_results, save_refined_results
from support_search import generate_candidates


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SolverConfig()
        cls.e_parameters = {
            "crank": 54.6, "coupler": 100.0, "rocker": 109.3,
        }
        cls.f_parameters = {
            "crank": 73.0, "coupler": 166.0, "rocker": 175.0,
            "local_x": 116.2, "local_y": -66.4,
        }
        cls.e_evaluated = evaluate_geometry("E", 100.0, cls.e_parameters, cls.config)
        cls.f_evaluated = evaluate_geometry("F", 100.0, cls.f_parameters, cls.config)
        if cls.e_evaluated is None or cls.f_evaluated is None:
            raise AssertionError("Reference geometries must pass the current filters")

    @staticmethod
    def entry_from_evaluation(family, evaluated):
        mechanism, support, _, result = evaluated
        return {
            "score": result.score,
            "mechanism": {
                "ground": mechanism.ground, "crank": mechanism.crank,
                "coupler": mechanism.coupler, "rocker": mechanism.rocker,
            },
            "support": {
                "kind": family, "local_x": support.local_x,
                "local_y": support.local_y,
            },
            "score_components": dict(result.score_components),
        }

    def test_E_uses_rocker_angular_displacement(self):
        mechanism, support, curve, _ = self.e_evaluated
        kinematic = solve(mechanism, self.config)
        angles = np.unwrap(np.arctan2(
            kinematic.C[:, 1] - kinematic.D[:, 1],
            kinematic.C[:, 0] - kinematic.D[:, 0],
        ))
        expected = np.degrees(angles - angles[0])
        np.testing.assert_allclose(curve.displacement, expected, atol=1e-12)
        self.assertAlmostEqual(support.local_x, mechanism.rocker)
        self.assertAlmostEqual(support.local_y, 0.0)

    def test_E_generates_only_one_canonical_support(self):
        mechanism = Mechanism(100.0, 54.6, 100.0, 109.3)
        supports = generate_candidates(mechanism, self.config, family="E")
        self.assertEqual(len(supports), 1)
        self.assertEqual(supports[0].kind, SupportKind.E)

    def test_final_E_is_vertical_at_middle_of_rocker_course(self):
        mechanism = self.e_evaluated[0]
        fine = solve(mechanism, replace(
            self.config, angle_step_deg=self.config.refinement_angle_step_deg
        ))
        support, cde = final_E_support(mechanism, fine)
        rocker = np.unwrap(np.arctan2(
            fine.C[:, 1] - fine.D[:, 1], fine.C[:, 0] - fine.D[:, 0]
        ))
        middle = 0.5 * (float(np.min(rocker)) + float(np.max(rocker)))
        world_de = middle + math.radians(cde)
        self.assertAlmostEqual(math.cos(world_de), 0.0, places=12)
        self.assertGreater(math.sin(world_de), 0.0)
        self.assertAlmostEqual(math.hypot(support.local_x, support.local_y), mechanism.rocker)

    def test_F_disk_radius_and_vertical_motion_constraint(self):
        mechanism = Mechanism(100.0, 53.0, 100.0, 109.0)
        supports = generate_candidates(mechanism, self.config, family="F")
        radius = 1.5 * mechanism.coupler
        self.assertTrue(supports)
        for support in supports:
            distance = math.hypot(
                support.local_x - 0.5 * mechanism.coupler, support.local_y
            )
            self.assertLessEqual(distance, radius + 1e-9)
        self.assertTrue(_F_motion_is_valid(self.f_evaluated[2], self.config))
        invalid_curve = SimpleNamespace(x=np.array([0.0, 1.0]), y=np.array([0.0, 2.0]))
        self.assertFalse(_F_motion_is_valid(invalid_curve, self.config))

    def test_refinement_domain(self):
        valid_e = {"crank": 25.0, "coupler": 100.0, "rocker": 100.0}
        self.assertTrue(geometry_in_level1_domain("E", valid_e, self.config))
        self.assertFalse(geometry_in_level1_domain(
            "E", dict(valid_e, crank=24.99), self.config
        ))
        coupler = 100.0
        valid_f = dict(
            valid_e, local_x=0.5 * coupler + 1.5 * coupler, local_y=0.0
        )
        self.assertTrue(geometry_in_level1_domain("F", valid_f, self.config))
        self.assertFalse(geometry_in_level1_domain(
            "F", dict(valid_f, local_x=valid_f["local_x"] + 0.01), self.config
        ))

    def test_family_aware_deduplication(self):
        def item(crank, coupler, rocker, x=0.0, y=0.0):
            return SimpleNamespace(
                mechanism=Mechanism(100.0, crank, coupler, rocker),
                support=Support(SupportKind.F, x, y),
            )
        first = item(50.0, 100.0, 100.0, 50.0, 0.0)
        nearby = item(52.0, 104.0, 104.0, 52.0, 0.0)
        other_support = item(52.0, 104.0, 104.0, 104.0, 0.0)
        self.assertTrue(solutions_are_close(first, nearby, "E", self.config, 1))
        self.assertTrue(solutions_are_close(first, nearby, "F", self.config, 1))
        self.assertFalse(solutions_are_close(first, other_support, "F", self.config, 1))
        self.assertFalse(solutions_are_close(first, nearby, "E", self.config, 2))

    def test_A1_A2_score_and_A3_window(self):
        for evaluated in (self.e_evaluated, self.f_evaluated):
            result = evaluated[3]
            components = result.score_components
            a3 = components["a3_angle"]
            self.assertAlmostEqual(components["a1_angle"], (180.0 + a3) % 360.0)
            self.assertAlmostEqual(components["a2_angle"], (360.0 - a3) % 360.0)
            distance = min(
                abs((a3 - target + 180.0) % 360.0 - 180.0)
                for target in self.config.a3_targets_deg
            )
            self.assertLessEqual(distance, self.config.a3_tolerance_deg)
            expected = (
                0.35 * components["shape_quality"]
                + 0.65 * components["acceleration_quality"]
            ) * components["symmetry_quality"]
            self.assertAlmostEqual(result.score, expected)

    def test_refinement_non_regression_E_geometry_and_roundtrip(self):
        entry = self.entry_from_evaluation("E", self.e_evaluated)
        solution = refine_candidate(
            entry, "E", 0, self.config, np.random.default_rng(17)
        )
        self.assertGreaterEqual(solution.final_score, solution.initial_score)
        self.assertAlmostEqual(
            math.hypot(solution.support.local_x, solution.support.local_y),
            solution.mechanism.rocker,
        )
        metrics = solution.physical_metrics
        self.assertAlmostEqual(
            2 * metrics["real_plateau_width_deg"]
            + metrics["exchange_1_to_2_deg"]
            + metrics["exchange_2_to_1_deg"], 360.0, places=9,
        )
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "refined.npz"
            save_refined_results([solution], [], filename)
            loaded = load_refined_results(filename)
            try:
                self.assertEqual(loaded["metadata"]["E"][0]["source_index"], 0)
                self.assertEqual(len(loaded["data"]["E_0_theta"]), 3601)
            finally:
                loaded["data"].close()

    def test_exact_symmetric_curve(self):
        curve = self.e_evaluated[2]
        x2 = symmetric_values(curve.theta, curve.x)
        indices = np.mod(-np.arange(len(curve.theta) - 1), len(curve.theta) - 1)
        np.testing.assert_allclose(x2[:-1], curve.x[:-1][indices], atol=1e-12)

    def test_precompression_definition(self):
        mechanism, _, _, result = self.e_evaluated
        fine_config = replace(
            self.config, angle_step_deg=self.config.refinement_angle_step_deg
        )
        fine_kinematic = solve(mechanism, fine_config)
        support, _ = final_E_support(mechanism, fine_kinematic)
        curve = build_candidate_curve(fine_kinematic, support)
        metrics = compute_physical_metrics(curve, result.score_components, self.config)
        x1 = np.asarray(curve.x, dtype=float)
        x2 = symmetric_values(curve.theta, x1)
        stroke = float(np.ptp(x1))
        empty_is_maximum = metrics["empty_plateau_is_maximum"]
        empty_position = float(np.max(x1) if empty_is_maximum else np.min(x1))

        def normalized_volume(values, angle):
            position = circular_interpolate(curve.theta, values, angle)
            volume = empty_position - position if empty_is_maximum else position - empty_position
            return volume / stroke

        self.assertAlmostEqual(
            metrics["precompression_1_to_2_ratio"],
            1.0 - normalized_volume(x2, metrics["real_plateau_end_deg"]),
        )
        self.assertAlmostEqual(
            metrics["precompression_2_to_1_ratio"],
            1.0 - normalized_volume(x1, metrics["symmetric_plateau_end_deg"]),
        )

    def test_symmetry_quality_shift_calibration(self):
        theta = np.arange(0.0, 360.1, 0.1)
        def quality(shift):
            curve = np.sin(np.radians(theta - shift))
            return _symmetry_quality(theta, curve, 60.0, 12.0, 1e-12)
        self.assertAlmostEqual(quality(0.0)["symmetry_quality"], 1.0, places=12)
        self.assertAlmostEqual(quality(6.0)["symmetry_quality"], 0.5, delta=0.03)
        self.assertEqual(quality(12.0)["symmetry_quality"], 0.0)


if __name__ == "__main__":
    unittest.main()
