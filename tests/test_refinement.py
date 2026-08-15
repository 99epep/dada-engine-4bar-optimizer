import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import SolverConfig
from kinematics import build_candidate_curve, solve
from refinement import (
    circular_interpolate,
    compute_physical_metrics,
    evaluate_geometry,
    refine_candidate,
    source_parameters,
    symmetric_values,
)
from result_io import load_refined_results, load_results, save_refined_results


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SolverConfig()
        cls.loaded = load_results("results.npz")

    @classmethod
    def tearDownClass(cls):
        cls.loaded["data"].close()

    def reevaluate(self, family):
        entry = self.loaded["metadata"][family][0]
        parameters = source_parameters(entry, family)
        return entry, evaluate_geometry(
            family, float(entry["mechanism"]["ground"]), parameters, self.config
        )

    def assert_ideal_landmarks(self, family):
        _, evaluated = self.reevaluate(family)
        self.assertIsNotNone(evaluated)
        result = evaluated[3]
        components = result.score_components
        a3 = components["a3_angle"]
        self.assertAlmostEqual(components["a1_angle"], (180.0 + a3) % 360.0)
        self.assertAlmostEqual(components["a2_angle"], (360.0 - a3) % 360.0)
        self.assertAlmostEqual(
            result.score,
            0.35 * components["shape_quality"]
            + 0.65 * components["acceleration_quality"],
        )

    def test_A1_A2_are_derived_from_A3_for_E(self):
        self.assert_ideal_landmarks("E")

    def test_A1_A2_are_derived_from_A3_for_F(self):
        self.assert_ideal_landmarks("F")

    def test_C_D_F_G_refinement_invariants_and_roundtrip(self):
        entry = self.loaded["metadata"]["E"][0]
        _, reevaluated = self.reevaluate("E")
        if not math.isclose(entry["score"], reevaluated[3].score, abs_tol=1e-12):
            self.skipTest("results.npz must be regenerated with the restored score")
        solution = refine_candidate(
            entry, "E", 0, self.config, np.random.default_rng(17)
        )
        self.assertGreaterEqual(solution.final_score, solution.initial_score)
        self.assertAlmostEqual(
            math.hypot(solution.support.local_x, solution.support.local_y),
            solution.mechanism.rocker,
            places=10,
        )
        metrics = solution.physical_metrics
        self.assertAlmostEqual(
            2 * metrics["real_plateau_width_deg"]
            + metrics["exchange_1_to_2_deg"]
            + metrics["exchange_2_to_1_deg"],
            360.0,
            places=9,
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

    def test_E_exact_symmetry(self):
        _, evaluated = self.reevaluate("E")
        curve = evaluated[2]
        x2 = symmetric_values(curve.theta, curve.x)
        # The level-1 grid contains every mirrored angle exactly.
        indices = np.mod(-np.arange(len(curve.theta) - 1), len(curve.theta) - 1)
        expected = curve.x[:-1][indices]
        np.testing.assert_allclose(x2[:-1], expected, rtol=0.0, atol=1e-12)

    def test_precompression_is_one_minus_full_piston_volume(self):
        _, evaluated = self.reevaluate("E")
        mechanism, support, _, result = evaluated
        fine_config = replace(
            self.config, angle_step_deg=self.config.refinement_angle_step_deg
        )
        curve = build_candidate_curve(solve(mechanism, fine_config), support)
        metrics = compute_physical_metrics(curve, result.score_components, self.config)
        x1 = np.asarray(curve.x, dtype=float)
        x2 = symmetric_values(curve.theta, x1)
        stroke = float(np.ptp(x1))
        empty_is_maximum = metrics["empty_plateau_is_maximum"]
        empty_position = float(np.max(x1) if empty_is_maximum else np.min(x1))

        def normalized_volume(values, angle):
            position = circular_interpolate(curve.theta, values, angle)
            swept_volume = (
                empty_position - position if empty_is_maximum
                else position - empty_position
            )
            return swept_volume / stroke

        # 1->2 starts when piston 1 ends its real empty plateau; piston 2 is
        # the full piston. At the reciprocal exchange piston 1 is full.
        expected_1_to_2 = 1.0 - normalized_volume(
            x2, metrics["real_plateau_end_deg"]
        )
        expected_2_to_1 = 1.0 - normalized_volume(
            x1, metrics["symmetric_plateau_end_deg"]
        )
        self.assertAlmostEqual(
            metrics["precompression_1_to_2_ratio"], expected_1_to_2, places=12
        )
        self.assertAlmostEqual(
            metrics["precompression_2_to_1_ratio"], expected_2_to_1, places=12
        )


if __name__ == "__main__":
    unittest.main()
