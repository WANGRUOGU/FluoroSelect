import unittest

import numpy as np

from utils import solve_lexicographic_k


class BrightnessConstraintTests(unittest.TestCase):
    def setUp(self):
        # Distinct nonzero spectra keep the tests focused on feasibility rather
        # than on ties caused by identical candidate columns.
        self.spectra = np.array(
            [
                [1.0, 0.2, 0.0, 0.1],
                [0.0, 1.0, 0.2, 0.1],
                [0.1, 0.0, 1.0, 0.2],
            ]
        )

    def test_pool_selection_respects_brightness_ratio(self):
        brightness = np.array([1.0, 3.0, 10.0, 12.0])

        selected, _ = solve_lexicographic_k(
            self.spectra,
            [],
            ["A", "B", "C", "D"],
            required_count=2,
            brightness_values=brightness,
            max_brightness_ratio=4.0,
        )

        chosen = brightness[selected]
        self.assertLessEqual(float(chosen.max() / chosen.min()), 4.0)

    def test_probe_assignment_respects_brightness_ratio(self):
        brightness = np.array([1.0, 8.0, 2.0, 12.0])

        selected, _ = solve_lexicographic_k(
            self.spectra,
            [[0, 1], [2, 3]],
            ["probe1 - A", "probe1 - B", "probe2 - C", "probe2 - D"],
            brightness_values=brightness,
            max_brightness_ratio=4.0,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(sum(index in {0, 1} for index in selected), 1)
        self.assertEqual(sum(index in {2, 3} for index in selected), 1)
        chosen = brightness[selected]
        self.assertLessEqual(float(chosen.max() / chosen.min()), 4.0)

    def test_incompatible_fixed_selections_are_infeasible(self):
        with self.assertRaisesRegex(ValueError, "Optimization failed: Infeasible"):
            solve_lexicographic_k(
                self.spectra,
                [],
                ["A", "B", "C", "D"],
                required_count=2,
                fixed_indices=[0, 2],
                brightness_values=np.array([1.0, 2.0, 10.0, 12.0]),
                max_brightness_ratio=4.0,
            )

    def test_nonpositive_brightness_candidates_are_excluded(self):
        selected, _ = solve_lexicographic_k(
            self.spectra,
            [],
            ["A", "B", "C", "D"],
            required_count=2,
            brightness_values=np.array([0.0, 2.0, 4.0, 5.0]),
            max_brightness_ratio=4.0,
        )

        self.assertNotIn(0, selected)


if __name__ == "__main__":
    unittest.main()
