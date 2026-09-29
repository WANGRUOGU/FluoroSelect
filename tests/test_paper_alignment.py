import itertools
import unittest
from unittest.mock import patch

import numpy as np
import pulp

from metrics import compute_classification_accuracy, compute_true_class_rmse, summarize_performance
from sim_core import simulate_balanced_pixels, simulate_rods_and_unmix, spectral_angle_classify_and_estimate
from utils import _make_cbc_exact, _pick_integral_from_solution, _solve_model, similarity_matrix, solve_lexicographic_k


class PaperAlignmentTests(unittest.TestCase):
    def test_zero_photons_are_unclassified(self):
        true = np.array([[[1., 0.]], [[0., 1.]]])
        estimated, labels = spectral_angle_classify_and_estimate(np.zeros((2, 1, 2)), np.eye(2))
        np.testing.assert_array_equal(labels, -1)
        np.testing.assert_array_equal(estimated, 0)
        self.assertEqual(compute_classification_accuracy(true, labels), [0., 0.])
        self.assertEqual(compute_true_class_rmse(true, estimated), [1., 1.])

    def test_true_class_error_excludes_background_and_other_classes(self):
        true = np.array([[[.8, 0.], [0., .6], [0., 0.]]])
        estimated = np.array([[[0., .8], [0., .5], [50., 50.]]])
        r = summarize_performance(true, estimated, np.array([[1, 1, -1]]))
        np.testing.assert_allclose(r['rmse'], [.8, .1])
        self.assertAlmostEqual(r['true_class_rmse'], np.sqrt((.64 + .01) / 2))
        self.assertAlmostEqual(r['worst_class_rmse'], .8)
        self.assertEqual(r['macro_accuracy'], .5)
        self.assertEqual(r['worst_accuracy'], 0.)

    def test_balanced_counts_and_abundance_range(self):
        for k in (2, 4, 12):
            true, estimated, labels = simulate_balanced_pixels(np.eye(k), rng=np.random.default_rng(42))
            self.assertEqual(true.shape, (100*k, 1, k))
            np.testing.assert_array_equal((true > 0).sum(axis=(0, 1)), 100)
            self.assertTrue(np.all((true[true > 0] >= .5) & (true[true > 0] <= 1)))
            self.assertEqual(estimated.shape, true.shape)

    def test_balanced_global_scale_invariance(self):
        spectra = np.array([[1., .1], [.1, .7]])
        a = simulate_balanced_pixels(spectra, rng=np.random.default_rng(19))
        b = simulate_balanced_pixels(8*spectra, rng=np.random.default_rng(19))
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y)

    def test_rod_abundance_scale_invariance(self):
        a = simulate_rods_and_unmix(np.eye(2), rods_per=1, rng=np.random.default_rng(23))
        b = simulate_rods_and_unmix(8*np.eye(2), rods_per=1, rng=np.random.default_rng(23))
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y)

    def test_legacy_simulation_uses_canonical_code(self):
        import simulation
        self.assertIs(simulation.spectral_angle_classify_and_estimate, spectral_angle_classify_and_estimate)

    def test_zero_spectrum_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            simulate_balanced_pixels(np.zeros((3, 2)))

    def test_no_rounding_incomplete_or_fractional_panels(self):
        x = [pulp.LpVariable('a'), pulp.LpVariable('b')]
        x[0].varValue, x[1].varValue = .5, .5
        with self.assertRaisesRegex(ValueError, 'integral'):
            _pick_integral_from_solution(x, required_count=1)
        x[0].varValue, x[1].varValue = 0, 0
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            _pick_integral_from_solution(x, required_count=1)

    def test_feasible_incumbent_not_reported_as_optimal(self):
        model = pulp.LpProblem('status')
        model.sol_status = pulp.LpSolutionIntegerFeasible
        with patch.object(model, 'solve', return_value=pulp.LpStatusOptimal):
            with self.assertRaisesRegex(ValueError, 'Optimization failed'):
                _solve_model(model, [], required_count=0)

    def test_unsupported_pulp_has_actionable_message(self):
        with patch.object(pulp, 'PULP_CBC_CMD', None):
            with self.assertRaisesRegex(ValueError, 'requirements.txt'):
                _make_cbc_exact()

    def test_pool_matches_exhaustive_minimax_with_fixed_and_allowed(self):
        for seed in range(4):
            E = np.random.default_rng(seed).random((7, 6))
            C = similarity_matrix(E)
            panels = [(0, *p) for p in itertools.combinations([1, 2, 3, 4], 2)]
            expected = min(max(C[i, j] for i, j in itertools.combinations(p, 2)) for p in panels)
            selected, value = solve_lexicographic_k(E, [], list('abcdef'), required_count=3,
                                                   fixed_indices=[0], allowed_indices=[1, 2, 3, 4])
            self.assertIn(0, selected)
            self.assertNotIn(5, selected)
            self.assertAlmostEqual(value, expected, places=7)
            self.assertLessEqual(max(C[i, j] for i, j in itertools.combinations(selected, 2)), expected + 1e-8)

    def test_probe_uniqueness_matches_exhaustive(self):
        E = np.array([[1., .2, 1., .1], [.1, 1., .1, .7], [.2, .1, .2, 1.]])
        labels = ['p1 – A', 'p1 – B', 'p2 – A', 'p2 – C']
        C = similarity_matrix(E)
        expected = min(C[i, j] for i, j in [(0, 3), (1, 2), (1, 3)])
        selected, value = solve_lexicographic_k(E, [[0, 1], [2, 3]], labels)
        self.assertEqual(len({labels[j].split(' – ')[1] for j in selected}), 2)
        self.assertAlmostEqual(value, expected, places=7)

    def test_tie_breaking_and_preferences_preserve_minimax(self):
        E = np.eye(4)
        selected, value = solve_lexicographic_k(E, [], list('abcd'), required_count=2,
                                               candidate_penalties=[100, 100, 0, 0], soft_penalty_weight=100)
        self.assertEqual(set(selected), {2, 3})
        self.assertEqual(value, 0.)
        E = np.array([[1., 0., .8], [0., 1., .6]])
        selected, value = solve_lexicographic_k(E, [], list('abc'), required_count=2,
                                               candidate_penalties=[1000, 1000, 0], soft_penalty_weight=1000)
        self.assertEqual(set(selected), {0, 1})


if __name__ == '__main__':
    unittest.main()
