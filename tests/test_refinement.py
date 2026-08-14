import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import SolverConfig
from refinement import (
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

    def test_A_source_E_score(self):
        entry, evaluated = self.reevaluate("E")
        self.assertIsNotNone(evaluated)
        self.assertAlmostEqual(evaluated[3].score, 0.364385853, places=9)
        self.assertAlmostEqual(evaluated[3].score, entry["score"], places=14)

    def test_B_source_F_score(self):
        entry, evaluated = self.reevaluate("F")
        self.assertIsNotNone(evaluated)
        self.assertAlmostEqual(evaluated[3].score, 0.362103121, places=9)
        self.assertAlmostEqual(evaluated[3].score, entry["score"], places=14)

    def test_C_D_F_G_refinement_invariants_and_roundtrip(self):
        entry = self.loaded["metadata"]["E"][0]
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

    def test_precompression_has_physical_direction(self):
        entry, evaluated = self.reevaluate("E")
        metrics = compute_physical_metrics(
            evaluated[2], evaluated[3].score_components, self.config
        )
        for name in (
            "precompression_1_to_2_ratio", "precompression_2_to_1_ratio"
        ):
            self.assertGreaterEqual(metrics[name], 0.0)
            self.assertLessEqual(metrics[name], 1.0)


if __name__ == "__main__":
    unittest.main()
